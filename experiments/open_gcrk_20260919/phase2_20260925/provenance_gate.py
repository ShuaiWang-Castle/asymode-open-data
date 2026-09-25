#!/usr/bin/env python3
"""Check that one public-data experiment cohort is complete and unmixed.

This gate separates two questions:

1. Can an existing E3R2 artifact be identified byte-for-byte?
2. Is every upstream weather file needed to rebuild it independently logged?

The second is deliberately stricter. A verified derived panel is sufficient to
replay an archived fit, but not to claim that the panel can be reconstructed
from public sources in the current checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
ROOT = EXPERIMENT.parents[1]
PROV = EXPERIMENT / "data_provenance"


def _json(path: Path):
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _events() -> list[str]:
    return [row["event"] for row in _json(EXPERIMENT / "selected_events_e3.json")["events"]]


def _event_from_panel(path: str, prefix: str) -> str | None:
    name = Path(path).name
    if name.startswith(prefix) and name.endswith(".npz"):
        return name[len(prefix):-4]
    return None


def _source_index() -> tuple[dict[str, dict], dict[str, dict]]:
    sources = _json(PROV / "sources.json")
    declared = {row["file"]: row for group in ("raw", "derived") for row in sources[group]}
    logged = {}
    for line in (PROV / "era5_fetch_log.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            logged[row["file"]] = row
    return declared, logged


def _weather_rows(index: dict[str, dict], event: str, directory: str) -> list[dict]:
    prefix = f"data/raw/{directory}/era5_{event}"
    return sorted((row for path, row in index.items() if path.startswith(prefix)),
                  key=lambda row: row["file"])


def _manifest_map(path: Path, prefix: str) -> tuple[dict[str, dict], list[str]]:
    rows = _json(path)
    out, repeats = {}, []
    for row in rows:
        event = _event_from_panel(row["file"], prefix)
        if event is None:
            continue
        if event in out:
            repeats.append(event)
        out[event] = row
    return out, repeats


def _file_check(row: dict, root: Path) -> dict:
    path = root / row["file"]
    if not path.is_file():
        return {"file": row["file"], "status": "missing"}
    size_ok = row.get("bytes") is None or path.stat().st_size == row["bytes"]
    actual = _sha256(path)
    hash_ok = actual == row["sha256"]
    return {"file": row["file"], "status": "ok" if size_ok and hash_ok else "mismatch",
            "bytes_ok": size_ok, "sha256_ok": hash_ok, "actual_sha256": actual}


def inspect(root: Path, check_files: bool = False) -> dict:
    expected = _events()
    expected_set = set(expected)
    p1, repeat1 = _manifest_map(PROV / "panel216_checksums.json", "panel216_")
    p2, repeat2 = _manifest_map(PROV / "panel216r2_checksums.json", "panel216r2_")
    feature = _json(PROV / "features_e3r2_checksum.json")
    declared, logged = _source_index()

    per_event = []
    ambiguous_r2_names = []
    for event in expected:
        main = _weather_rows(declared, event, "era5")
        # Download log is authoritative for the separately fetched maximum-gust field.
        gust = _weather_rows(logged, event, "era5_fg10")
        r2_names = p2.get(event, {}).get("era5_files", [])
        duplicate_names = sorted({name for name in r2_names if r2_names.count(name) > 1})
        if duplicate_names:
            ambiguous_r2_names.append({"event": event, "duplicate_basenames": duplicate_names})
        per_event.append({
            "event": event,
            "panel216_manifest": event in p1,
            "panel216r2_manifest": event in p2,
            "main_era5_source_hashes": [r["file"] for r in main],
            "fg10_source_hashes": [r["file"] for r in gust],
            "rebuild_weather_provenance_complete": bool(main and gust),
        })

    main_missing = [r["event"] for r in per_event if not r["main_era5_source_hashes"]]
    gust_missing = [r["event"] for r in per_event if not r["fg10_source_hashes"]]
    manifest_ok = (set(p1) == expected_set and set(p2) == expected_set and not repeat1 and not repeat2
                   and feature.get("events") == expected and feature.get("units") == 6122
                   and feature.get("counties") == 2409 and feature.get("d_u") == 42
                   and feature.get("G") == 40)
    report = {
        "cohort": "E3R2-12-event-216-hour",
        "expected_events": expected,
        "manifest_identity_complete": manifest_ok,
        "rebuild_provenance_complete": not main_missing and not gust_missing,
        "missing_main_era5_source_hash_events": main_missing,
        "missing_fg10_source_hash_events": gust_missing,
        "panel216_missing_events": sorted(expected_set - set(p1)),
        "panel216_extra_events": sorted(set(p1) - expected_set),
        "panel216r2_missing_events": sorted(expected_set - set(p2)),
        "panel216r2_extra_events": sorted(set(p2) - expected_set),
        "duplicate_manifest_events": {"panel216": repeat1, "panel216r2": repeat2},
        "ambiguous_r2_weather_basenames": ambiguous_r2_names,
        "feature_manifest": feature,
        "per_event": per_event,
        "interpretation": (
            "A complete derived-artifact manifest permits identity checking after restoration; "
            "it does not establish independent source reconstruction when upstream hashes are absent."
        ),
    }
    if check_files:
        rows = [p1[e] for e in expected if e in p1] + [p2[e] for e in expected if e in p2] + [feature]
        report["local_file_checks"] = [_file_check(row, root) for row in rows]
        report["local_files_all_verified"] = all(r["status"] == "ok" for r in report["local_file_checks"])
    return report


def _self_test() -> None:
    report = inspect(ROOT, check_files=False)
    assert report["manifest_identity_complete"], report
    assert len(report["expected_events"]) == 12
    assert not report["missing_fg10_source_hash_events"], report
    assert report["missing_main_era5_source_hash_events"] == [
        "2019-02-24", "2019-11-27", "2021-08-11", "2022-04-13",
        "2022-06-17", "2024-05-08", "2024-06-26",
    ], report
    assert not report["rebuild_provenance_complete"]
    assert report["ambiguous_r2_weather_basenames"], report
    print("self-test passed: E3R2 manifests are complete; seven main-ERA5 source records are absent")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT,
                        help="repository/data root used for optional file verification")
    parser.add_argument("--check-files", action="store_true",
                        help="stream and verify all 25 derived artifacts against size and SHA-256")
    parser.add_argument("--out", type=Path, help="write JSON report instead of stdout")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return
    result = inspect(args.root, check_files=args.check_files)
    encoded = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded)
        print(f"wrote {args.out}")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
