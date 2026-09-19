"""Fetch ERA5 single-level fields for selected event windows that are not on disk.

Same variables, area and request layout as scripts/fetch_era5.py (one CDS request
per calendar month, because CDS returns the cross product of year x month x day).
Only the nine UTC days of each 216-hour window are requested. Writes
data/raw/era5/era5_<event>.nc (one month) or era5_<event>.m<i>.nc (several), and
appends source, request, date and SHA-256 to data_provenance/era5_fetch_log.jsonl.

Options for PREREG Amendment 2: --variables fetches another variable list into --out-dir
(e.g. the hourly maximum gust into data/raw/era5_fg10/), and --from-candidates takes the
windows from event_selection.csv (the pre-registered candidate pool) instead of
selected_events.json.

Requires ~/.cdsapirc (never read or printed here beyond what cdsapi does itself).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RAW = ROOT / "data" / "raw" / "era5"
sys.path.insert(0, str(ROOT / "scripts"))
from fetch_era5 import AREA, VARIABLES  # noqa: E402

LOG = HERE / "data_provenance" / "era5_fetch_log.jsonl"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", nargs="+", required=True)
    ap.add_argument("--variables", nargs="+", default=None, help="CDS variable names (default: the panel list)")
    ap.add_argument("--out-dir", default=None, help="directory under the repository root (default data/raw/era5)")
    ap.add_argument("--from-candidates", action="store_true", help="windows from event_selection.csv")
    a = ap.parse_args()
    import cdsapi
    c = cdsapi.Client()
    variables = a.variables or VARIABLES
    raw = ROOT / a.out_dir if a.out_dir else RAW
    if a.from_candidates:
        cand = pd.read_csv(HERE / "event_selection.csv")
        sel = {r.day: dict(window_start_utc=r.window_start_utc, window_end_utc=r.window_end_utc) for r in cand.itertuples()}
    else:
        sel = {e["event"]: e for e in json.loads((HERE / "selected_events.json").read_text())["events"]}
    raw.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    for ev in a.events:
        s = pd.Timestamp(sel[ev]["window_start_utc"])
        e = pd.Timestamp(sel[ev]["window_end_utc"])
        days = pd.date_range(s.normalize(), e.normalize(), freq="D")
        by_month: dict[tuple[str, str], list[str]] = {}
        for d in days:
            by_month.setdefault((f"{d.year:04d}", f"{d.month:02d}"), []).append(f"{d.day:02d}")
        parts = sorted(by_month.items())
        for i, ((yy, mm), dd) in enumerate(parts):
            dst = raw / (f"era5_{ev}.nc" if len(parts) == 1 else f"era5_{ev}.m{i}.nc")
            if dst.exists():
                print("exists", dst.name, flush=True)
                continue
            req = {"product_type": ["reanalysis"], "variable": variables, "year": [yy], "month": [mm],
                   "day": dd, "time": [f"{h:02d}:00" for h in range(24)], "area": AREA,
                   "data_format": "netcdf", "download_format": "unarchived"}
            tmp = dst.with_suffix(".part")
            print(f"{ev}: requesting {yy}-{mm} days {dd[0]}..{dd[-1]}", flush=True)
            c.retrieve("reanalysis-era5-single-levels", req, str(tmp))
            tmp.rename(dst)
            rec = dict(file=str(dst.relative_to(ROOT)), source="Copernicus Climate Data Store, "
                       "dataset reanalysis-era5-single-levels (https://cds.climate.copernicus.eu)",
                       request=req, downloaded_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                       bytes=dst.stat().st_size, sha256=sha256(dst))
            with open(LOG, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print("saved", dst.name, rec["bytes"], rec["sha256"][:12], flush=True)


if __name__ == "__main__":
    main()
