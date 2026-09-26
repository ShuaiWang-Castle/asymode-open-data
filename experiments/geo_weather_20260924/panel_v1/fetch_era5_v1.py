"""ERA5 windows of DATASET_DESIGN v1 systems (section 8) from ARCO-ERA5 with the fetch and file layout of
fetch_arco_windows.py (the CONUS box 50..24 N, -125..-66 E, which the panel and feature builders index), one file per
set and system:
  data/raw/era5/era5_<system>.nc         u10 v10 i10fg t2m d2m cape swvl1 tcc sp tp sf
  data/raw/era5_fg10/era5_<system>.nc    fg10
  data/raw/era5_extra2/era5_<system>.nc  ptype sd lai_hv cp swvl2
216 hours from the window start. Each file is logged to data_provenance/downloads.jsonl. The three files are deleted
once the system's features are built (section 8; a CONUS window is about 365 MB): `--delete`.
usage: python fetch_era5_v1.py --systems S00001 S00002 ...   |   --delete S00001 ..."""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import fetch_arco_windows as A  # noqa: E402
from frame_common import HORIZON_H, OUT, PREFIX_H, ROOT  # noqa: E402

SETS = ((A.MAIN, "era5"), (A.GUST, "era5_fg10"), (A.EXTRA, "era5_extra2"))


def paths(system: str) -> list[Path]:
    return [ROOT / "data" / "raw" / sub / f"era5_{system}.nc" for _, sub in SETS]


def fetch(system: str, t0: pd.Timestamp, workers: int = 24) -> None:
    times = pd.date_range(t0, periods=PREFIX_H + HORIZON_H, freq="h")
    for (varmap, sub), dst in zip(SETS, paths(system)):
        if dst.exists():
            continue
        A.write(dst, A.window(varmap, times, workers), times, "ARCO-ERA5 analysis-ready 0.25 deg, hourly")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="*", default=[])
    ap.add_argument("--delete", nargs="*", default=[])
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    sy = pd.read_parquet(OUT / "systems.parquet").set_index("system")
    for s in a.systems:
        t = dt.datetime.now()
        fetch(s, pd.Timestamp(sy.at[s, "window_start"]), a.workers)
        print("saved", s, f"{(dt.datetime.now() - t).total_seconds():.0f} s", flush=True)
    for s in a.delete:
        for p in paths(s):
            if p.exists():
                p.unlink()
        print("deleted raw windows of", s, flush=True)


if __name__ == "__main__":
    main()
