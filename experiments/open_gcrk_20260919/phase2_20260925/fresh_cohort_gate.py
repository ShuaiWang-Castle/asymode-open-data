#!/usr/bin/env python3
"""Validate a prospective weather-only confirmatory event cohort.

This tool validates identities and declared access boundaries. It never reads
outage targets, panels, checkpoints or model results, and it cannot prove that
an undeclared source was not viewed.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEFAULT_PROTOCOL = HERE / "FRESH_COHORT_PROTOCOL.json"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_json_sha256(value: dict) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def parse_utc(value: str) -> datetime:
    text = value.replace("Z", "+00:00")
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        raise ValueError(f"timestamp lacks UTC offset: {value}")
    return dt.astimezone(timezone.utc)


def _events_from_snapshot(path: Path, event_field: str) -> set[str]:
    if event_field == "panels":
        return set(json.loads(path.read_text())["panels"])
    if event_field == "events.event":
        return {row["event"] for row in json.loads(path.read_text())["events"]}
    if event_field == "day":
        with path.open(newline="") as handle:
            return {row["day"] for row in csv.DictReader(handle)}
    raise ValueError(f"unsupported event_field: {event_field}")


def verify_tracked_sources(protocol: dict, root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    outcomes: set[str] = set()
    weather_screened: set[str] = set()
    for item in protocol["tracked_source_snapshots"]:
        path = root / item["path"]
        if not path.is_file():
            errors.append(f"missing tracked source: {item['path']}")
            continue
        actual = sha256_file(path)
        if actual != item["sha256"]:
            errors.append(
                f"tracked source hash mismatch: {item['path']} expected "
                f"{item['sha256']} got {actual}"
            )
            continue
        try:
            events = _events_from_snapshot(path, item["event_field"])
        except Exception as exc:  # report source-schema drift as a gate failure
            errors.append(f"cannot parse tracked source {item['path']}: {exc}")
            continue
        if item["role"] == "outcome_inspected":
            outcomes.update(events)
        elif item["role"] == "weather_metadata_screened":
            weather_screened.update(events)

    expected_outcomes = set(protocol["outcome_inspected_event_ids"])
    if outcomes != expected_outcomes:
        errors.append("outcome-inspected event union differs from frozen protocol")
    normalized = "\n".join(sorted(outcomes)) + "\n"
    digest = hashlib.sha256(normalized.encode()).hexdigest()
    if digest != protocol["outcome_inspected_event_ids_sha256"]:
        errors.append("outcome-inspected normalized-list hash mismatch")
    if weather_screened != set(protocol["weather_metadata_screened_event_ids"]):
        errors.append("weather-screened event set differs from frozen protocol")
    return errors


def _historical_windows(protocol: dict) -> list[tuple[datetime, datetime, str]]:
    cfg = protocol["historical_window_convention"]
    hours = int(cfg["window_hours"])
    offset = int(cfg["window_start_offset_hours"])
    out = []
    for event_id in protocol["outcome_inspected_event_ids"]:
        anchor = datetime.fromisoformat(event_id).replace(tzinfo=timezone.utc)
        start = anchor + timedelta(hours=offset)
        out.append((start, start + timedelta(hours=hours), event_id))
    return out


def validate_candidate(candidate: dict, protocol: dict) -> list[str]:
    errors: list[str] = []
    req = protocol["confirmatory_requirements"]

    if candidate.get("schema_version") != 1:
        errors.append("candidate schema_version must be 1")
    if candidate.get("purpose") != "confirmatory_process_graph":
        errors.append("purpose must be confirmatory_process_graph")
    try:
        parse_utc(candidate.get("lock_utc", ""))
    except Exception as exc:
        errors.append(f"invalid lock_utc: {exc}")

    if candidate.get("outcome_access_before_lock") != []:
        errors.append("outcome_access_before_lock must be an explicit empty list")
    if candidate.get("seeds") != req["required_seeds"]:
        errors.append(f"seeds must equal {req['required_seeds']}")

    for key in ("panel_id", "split_id", "weather_information_set_id"):
        value = candidate.get(key)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{key} must be a nonempty new identifier")

    rule_hash = candidate.get("selection_rule_sha256", "")
    if not SHA256_RE.fullmatch(rule_hash):
        errors.append("selection_rule_sha256 must be 64 lowercase hex characters")

    fields = candidate.get("selection_fields_used")
    allowed_fields = set(protocol["allowed_selection_fields"])
    forbidden_tokens = tuple(protocol["forbidden_selection_field_tokens"])
    if not isinstance(fields, list) or not fields:
        errors.append("selection_fields_used must be a nonempty list")
    else:
        bad = sorted(set(fields) - allowed_fields)
        if bad:
            errors.append(f"selection fields outside allow-list: {bad}")
        token_bad = sorted(
            field for field in fields
            if any(token in field.lower() for token in forbidden_tokens)
        )
        if token_bad:
            errors.append(f"selection fields contain forbidden tokens: {token_bad}")

    sources = candidate.get("source_files")
    allowed_roles = set(protocol["allowed_selection_source_roles"])
    forbidden_paths = tuple(protocol["forbidden_source_path_tokens"])
    if not isinstance(sources, list) or not sources:
        errors.append("source_files must be a nonempty list")
    else:
        for index, source in enumerate(sources):
            prefix = f"source_files[{index}]"
            if source.get("role") not in allowed_roles:
                errors.append(f"{prefix}.role is not allowed")
            if not SHA256_RE.fullmatch(source.get("sha256", "")):
                errors.append(f"{prefix}.sha256 is invalid")
            url = source.get("public_url", "")
            if not isinstance(url, str) or not url.startswith("https://"):
                errors.append(f"{prefix}.public_url must be https")
            if not isinstance(source.get("path"), str) or not source["path"]:
                errors.append(f"{prefix}.path is required")
            else:
                normalized_path = source["path"].lower().replace("\\", "/")
                hits = [token for token in forbidden_paths if token in normalized_path]
                if hits:
                    errors.append(f"{prefix}.path contains forbidden tokens: {hits}")

    events = candidate.get("events")
    if not isinstance(events, list):
        errors.append("events must be a list")
        events = []
    if not req["minimum_events"] <= len(events) <= req["maximum_events"]:
        errors.append(
            f"event count must be {req['minimum_events']}..{req['maximum_events']}"
        )

    inspected = set(protocol["outcome_inspected_event_ids"])
    screened = set(protocol["weather_metadata_screened_event_ids"])
    historical_windows = _historical_windows(protocol)
    candidate_windows: list[tuple[datetime, datetime, str]] = []
    seen: set[str] = set()
    for index, event in enumerate(events):
        event_id = event.get("event_id", "")
        if not DATE_RE.fullmatch(event_id):
            errors.append(f"events[{index}].event_id must be YYYY-MM-DD")
            continue
        if event_id in seen:
            errors.append(f"duplicate event_id: {event_id}")
        seen.add(event_id)
        if event_id in inspected:
            errors.append(f"outcome-inspected event reused: {event_id}")
        if req["selection_fresh"] and event_id in screened:
            errors.append(f"previously weather-screened event reused: {event_id}")
        try:
            start = parse_utc(event["window_start_utc"])
            end = parse_utc(event["window_end_utc"])
        except Exception as exc:
            errors.append(f"events[{index}] invalid window: {exc}")
            continue
        if end - start != timedelta(hours=215):
            errors.append(f"event {event_id} window must span 216 hourly timestamps")
        anchor = datetime.fromisoformat(event_id).replace(tzinfo=timezone.utc)
        if anchor - start != timedelta(hours=72):
            errors.append(f"event {event_id} window must start 72 hours before anchor")
        stop_exclusive = end + timedelta(hours=1)
        for old_start, old_stop, old_id in historical_windows:
            if start < old_stop and old_start < stop_exclusive:
                errors.append(
                    f"event {event_id} window overlaps historical event {old_id}"
                )
        candidate_windows.append((start, stop_exclusive, event_id))

    for i, (start, stop, event_id) in enumerate(candidate_windows):
        for other_start, other_stop, other_id in candidate_windows[i + 1:]:
            if start < other_stop and other_start < stop:
                errors.append(f"candidate windows overlap: {event_id} and {other_id}")
    return sorted(set(errors))


def verify_candidate_sources(candidate: dict, root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    for index, source in enumerate(candidate.get("source_files", [])):
        path = Path(source.get("path", ""))
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            errors.append(f"candidate source missing: {source.get('path', '')}")
            continue
        actual = sha256_file(path)
        if actual != source.get("sha256"):
            errors.append(
                f"candidate source hash mismatch: {source.get('path', '')} "
                f"expected {source.get('sha256', '')} got {actual}"
            )
    return errors


def _valid_example() -> dict:
    h = "a" * 64
    return {
        "schema_version": 1,
        "purpose": "confirmatory_process_graph",
        "lock_utc": "2026-09-25T12:00:00Z",
        "selection_rule_sha256": h,
        "selection_fields_used": [
            "event_type", "begin_utc", "end_utc", "wind_county_count",
            "wind_state_count", "era5_fg10_peak"
        ],
        "source_files": [
            {
                "path": "data/raw/storm_events/new.csv.gz",
                "sha256": h,
                "public_url": "https://www.ncei.noaa.gov/example.csv.gz",
                "role": "noaa_storm_events"
            }
        ],
        "outcome_access_before_lock": [],
        "panel_id": "panel216_confirmatory_v1",
        "split_id": "county_event_confirmatory_v1",
        "weather_information_set_id": "era5_reanalysis_confirmatory_v1",
        "seeds": [0, 1, 2, 3, 4],
        "events": [
            {
                "event_id": "2025-02-15",
                "window_start_utc": "2025-02-12T00:00:00Z",
                "window_end_utc": "2025-02-20T23:00:00Z"
            },
            {
                "event_id": "2025-05-15",
                "window_start_utc": "2025-05-12T00:00:00Z",
                "window_end_utc": "2025-05-20T23:00:00Z"
            },
            {
                "event_id": "2025-09-15",
                "window_start_utc": "2025-09-12T00:00:00Z",
                "window_end_utc": "2025-09-20T23:00:00Z"
            }
        ]
    }


def self_test(protocol: dict) -> dict:
    accepted = _valid_example()
    assert validate_candidate(accepted, protocol) == []
    rejected = {}

    cases = {}
    case = copy.deepcopy(accepted)
    case["events"][0]["event_id"] = "2024-05-08"
    case["events"][0]["window_start_utc"] = "2024-05-05T00:00:00Z"
    case["events"][0]["window_end_utc"] = "2024-05-13T23:00:00Z"
    cases["outcome_reuse"] = case

    case = copy.deepcopy(accepted)
    case["events"][0]["event_id"] = "2024-11-20"
    case["events"][0]["window_start_utc"] = "2024-11-17T00:00:00Z"
    case["events"][0]["window_end_utc"] = "2024-11-25T23:00:00Z"
    cases["weather_screen_reuse"] = case

    case = copy.deepcopy(accepted)
    case["selection_fields_used"].append("customers_out_peak")
    cases["forbidden_field"] = case

    case = copy.deepcopy(accepted)
    case["source_files"][0]["role"] = "eaglei_outages"
    cases["forbidden_source_role"] = case

    case = copy.deepcopy(accepted)
    case["source_files"][0]["path"] = "results/weather.csv"
    cases["forbidden_source_path"] = case

    case = copy.deepcopy(accepted)
    case["outcome_access_before_lock"] = ["panel216_targets.npz"]
    cases["outcome_access"] = case

    case = copy.deepcopy(accepted)
    case["events"][1]["window_start_utc"] = "2025-02-18T00:00:00Z"
    case["events"][1]["window_end_utc"] = "2025-02-26T23:00:00Z"
    case["events"][1]["event_id"] = "2025-02-21"
    cases["candidate_overlap"] = case

    case = copy.deepcopy(accepted)
    case["events"][0]["window_start_utc"] = "2025-02-13T00:00:00Z"
    cases["wrong_origin"] = case

    case = copy.deepcopy(accepted)
    case["events"] = case["events"][:2]
    cases["too_few_events"] = case

    for name, candidate in cases.items():
        errors = validate_candidate(candidate, protocol)
        assert errors, f"invalid case unexpectedly accepted: {name}"
        rejected[name] = errors
    return {"accepted": 1, "rejected": rejected}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--verify-tracked-sources", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text())
    result: dict = {"protocol": str(args.protocol), "errors": []}
    if args.verify_tracked_sources or args.self_test:
        result["tracked_source_errors"] = verify_tracked_sources(protocol)
        result["errors"].extend(result["tracked_source_errors"])
    if args.self_test:
        result["self_test"] = self_test(protocol)
    if args.candidate:
        candidate = json.loads(args.candidate.read_text())
        result["candidate"] = str(args.candidate)
        result["candidate_sha256"] = sha256_file(args.candidate)
        result["candidate_canonical_sha256"] = canonical_json_sha256(candidate)
        result["candidate_errors"] = validate_candidate(candidate, protocol)
        result["errors"].extend(result["candidate_errors"])
        result["candidate_source_errors"] = verify_candidate_sources(candidate)
        result["errors"].extend(result["candidate_source_errors"])
    if not args.self_test and not args.candidate and not args.verify_tracked_sources:
        parser.error("choose --self-test, --candidate, or --verify-tracked-sources")
    result["ok"] = not result["errors"]
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
