#!/usr/bin/env python3
"""Select a weather-only confirmatory cohort from a sanitized NOAA catalog.

The parquet reader requests an explicit allow-list of meteorological and
identity columns. It never reads outage targets, NOAA impacts, panels,
checkpoints, predictions or model metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEFAULT_RULE = HERE / "WEATHER_COHORT_SELECTION_RULE.json"
DEFAULT_PROTOCOL = HERE / "FRESH_COHORT_PROTOCOL.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_utc(value: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC").tz_localize(None)


def window(anchor: pd.Timestamp, rule: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    anchor = pd.Timestamp(anchor).normalize()
    prefix = pd.Timedelta(hours=int(rule["window"]["prefix_hours"]))
    hours = pd.Timedelta(hours=int(rule["window"]["hours"]))
    start = anchor - prefix
    return start, start + hours


def overlaps(a: tuple[pd.Timestamp, pd.Timestamp],
             b: tuple[pd.Timestamp, pd.Timestamp]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def excluded_windows(protocol: dict, rule: dict) -> list[tuple[pd.Timestamp, pd.Timestamp, str, str]]:
    out = []
    groups = (
        ("outcome_inspected", protocol["outcome_inspected_event_ids"]),
        ("weather_metadata_screened", protocol["weather_metadata_screened_event_ids"]),
    )
    for role, event_ids in groups:
        for event_id in event_ids:
            start, stop = window(pd.Timestamp(event_id), rule)
            out.append((start, stop, role, event_id))
    return out


def release_contains(start: pd.Timestamp, stop: pd.Timestamp, rule: dict) -> bool:
    anchor = start + pd.Timedelta(hours=int(rule["window"]["prefix_hours"]))
    year = str(anchor.year)
    if year not in rule["public_release_ranges"]:
        return False
    release_end = parse_utc(rule["public_release_ranges"][year])
    return stop - pd.Timedelta(hours=1) <= release_end


def family_metrics(events: pd.DataFrame, anchor: pd.Timestamp,
                   family: str, types: set[str], rule: dict) -> dict | None:
    start, stop = window(anchor, rule)
    local = events[(events["t_begin_utc"] >= start)
                   & (events["t_begin_utc"] < stop)
                   & events["EVENT_TYPE"].isin(types)].copy()
    if local.empty:
        return None
    local["hour"] = ((local["t_begin_utc"] - start).dt.total_seconds() // 3600).astype(int)
    prefix_h = int(rule["window"]["prefix_hours"])
    prefix = local[local["hour"] < prefix_h]
    forecast = local[local["hour"] >= prefix_h]
    if forecast.empty:
        return None
    counties = int(forecast["fips"].nunique())
    states = int(forecast["STATE"].nunique())
    prefix_counties = int(prefix["fips"].nunique())
    ratio = prefix_counties / max(counties, 1)
    span = float(np.subtract(*np.percentile(forecast["hour"], [95, 5])))
    daily = []
    for index in range(6):
        lower = prefix_h + index * 24
        upper = lower + 24
        daily.append(int(forecast[(forecast["hour"] >= lower)
                                  & (forecast["hour"] < upper)]["fips"].nunique()))
    return {
        "event_id": str(anchor.date()),
        "hazard_family": family,
        "window_start_utc": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "window_end_utc": (stop - pd.Timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "forecast_family_counties": counties,
        "forecast_family_states": states,
        "prefix_family_counties": prefix_counties,
        "prefix_to_forecast_family_county_ratio": round(float(ratio), 6),
        "forecast_family_span_5_95_hours": round(span, 6),
        "forecast_family_counties_by_day": daily,
    }


def qualifies(row: dict, rule: dict) -> bool:
    hard = rule["hard_filters"]
    return (
        row["forecast_family_counties"] >= hard["minimum_forecast_family_counties"]
        and row["forecast_family_states"] >= hard["minimum_forecast_family_states"]
        and row["prefix_to_forecast_family_county_ratio"]
        <= hard["maximum_prefix_to_forecast_family_county_ratio"]
        and row["forecast_family_span_5_95_hours"]
        >= hard["minimum_forecast_family_span_5_95_hours"]
    )


def screen(events: pd.DataFrame, anchors: pd.DataFrame,
           protocol: dict, rule: dict) -> tuple[list[dict], dict]:
    exclusions = excluded_windows(protocol, rule)
    rows: list[dict] = []
    excluded_anchor_count = 0
    release_excluded_count = 0
    for anchor_value in sorted(pd.to_datetime(anchors["day"]).unique()):
        anchor = pd.Timestamp(anchor_value).normalize()
        current = window(anchor, rule)
        if not release_contains(*current, rule):
            release_excluded_count += 1
            continue
        if any(overlaps(current, (start, stop)) for start, stop, _, _ in exclusions):
            excluded_anchor_count += 1
            continue
        for family, event_types in rule["hazard_families"].items():
            row = family_metrics(events, anchor, family, set(event_types), rule)
            if row is not None:
                row["passes_hard_filters"] = qualifies(row, rule)
                rows.append(row)

    qualified = [row for row in rows if row["passes_hard_filters"]]
    selected: list[dict] = []
    selected_windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    unfilled_slots = []
    for slot_index, family in enumerate(rule["selection_slots"]):
        ranked = sorted(
            (row for row in qualified if row["hazard_family"] == family),
            key=lambda row: (
                -row["forecast_family_counties"],
                -row["forecast_family_states"],
                row["event_id"],
            ),
        )
        pick = None
        for row in ranked:
            candidate_window = window(pd.Timestamp(row["event_id"]), rule)
            if not any(overlaps(candidate_window, prior) for prior in selected_windows):
                pick = dict(row)
                pick["selection_slot"] = slot_index
                break
        if pick is None:
            unfilled_slots.append({"selection_slot": slot_index, "hazard_family": family})
            continue
        selected.append(pick)
        selected_windows.append(window(pd.Timestamp(pick["event_id"]), rule))

    counts = {family: sum(row["hazard_family"] == family for row in qualified)
              for family in rule["hazard_families"]}
    summary = {
        "anchor_rows": int(len(anchors)),
        "unique_anchor_days": int(pd.to_datetime(anchors["day"]).nunique()),
        "excluded_anchor_days_by_prior_window": excluded_anchor_count,
        "excluded_anchor_days_by_release_range": release_excluded_count,
        "qualified_family_rows": len(qualified),
        "qualified_anchor_days": len({row["event_id"] for row in qualified}),
        "qualified_rows_by_family": counts,
        "selected_events": selected,
        "unfilled_slots": unfilled_slots,
        "negative_result": (
            "No tropical anchor met all fixed filters."
            if counts.get("tropical", 0) == 0 else None
        ),
    }
    return selected, summary


def load_verified(path: Path, expected_sha256: str, columns: list[str]) -> pd.DataFrame:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"source hash mismatch for {path}: expected {expected_sha256}, got {actual}")
    return pd.read_parquet(path, columns=columns)


def build_candidate(selected: list[dict], rule: dict, rule_path: Path,
                    protocol: dict, lock_utc: str) -> dict:
    lock = datetime.fromisoformat(lock_utc.replace("Z", "+00:00"))
    if lock.tzinfo is None:
        raise ValueError("lock_utc must include a UTC offset")
    base_url = (
        "https://github.com/ShuaiWang-Castle/asymode-open-data/blob/"
        "8dd47c5ccd829611f27b69a3d64c274a0a24c400/"
    )
    branch_url = (
        "https://github.com/ShuaiWang-Castle/asymode-open-data/blob/"
        "research/open-gcrk-data-mechanism-20260925/"
    )
    source_files = []
    for key in ("anchor_source", "event_source"):
        item = rule[key]
        source_files.append({
            "path": item["path"], "sha256": item["sha256"],
            "public_url": base_url + item["path"], "role": "noaa_storm_events",
        })
    source_files.append({
        "path": str(rule_path.relative_to(ROOT)),
        "sha256": sha256_file(rule_path),
        "public_url": branch_url + str(rule_path.relative_to(ROOT)),
        "role": "weather_selection_rule",
    })
    events = []
    for row in sorted(selected, key=lambda item: item["event_id"]):
        events.append({key: row[key] for key in (
            "event_id", "window_start_utc", "window_end_utc", "hazard_family",
            "selection_slot", "forecast_family_counties", "forecast_family_states",
            "prefix_to_forecast_family_county_ratio",
            "forecast_family_span_5_95_hours", "forecast_family_counties_by_day",
        )})
    return {
        "schema_version": 1,
        "purpose": "confirmatory_process_graph",
        "lock_utc": lock.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "selection_rule_sha256": sha256_file(rule_path),
        "selection_fields_used": [
            "event_anchor_utc", "event_type", "begin_utc", "end_utc",
            "state_fips", "county_fips", "hazard_family", "family_county_count",
            "family_state_count", "family_span_hours", "prefix_family_count",
            "prefix_to_forecast_family_ratio",
        ],
        "source_files": source_files,
        "outcome_access_before_lock": [],
        "panel_id": "panel216_noaa_fresh_v1",
        "split_id": "county_event_noaa_fresh_v1",
        "weather_information_set_id": "era5_reanalysis_noaa_fresh_v1",
        "seeds": protocol["confirmatory_requirements"]["required_seeds"],
        "events": events,
    }


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rule", type=Path, default=DEFAULT_RULE)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--lock-utc", required=True)
    parser.add_argument("--candidate-out", type=Path, required=True)
    parser.add_argument("--screen-out", type=Path, required=True)
    args = parser.parse_args()
    rule = json.loads(args.rule.read_text())
    protocol = json.loads(args.protocol.read_text())
    anchor_spec, event_spec = rule["anchor_source"], rule["event_source"]
    anchors = load_verified(ROOT / anchor_spec["path"], anchor_spec["sha256"],
                            anchor_spec["columns_read"])
    events = load_verified(ROOT / event_spec["path"], event_spec["sha256"],
                           event_spec["columns_read"])
    events["t_begin_utc"] = pd.to_datetime(events["t_begin_utc"])
    events["t_end_utc"] = pd.to_datetime(events["t_end_utc"])
    selected, summary = screen(events, anchors, protocol, rule)
    candidate = build_candidate(selected, rule, args.rule, protocol, args.lock_utc)
    write_json(args.screen_out, summary)
    write_json(args.candidate_out, candidate)
    print(json.dumps({
        "candidate_out": str(args.candidate_out),
        "candidate_sha256": sha256_file(args.candidate_out),
        "screen_out": str(args.screen_out),
        "screen_sha256": sha256_file(args.screen_out),
        "selected": [row["event_id"] for row in selected],
        "qualified_rows_by_family": summary["qualified_rows_by_family"],
        "negative_result": summary["negative_result"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
