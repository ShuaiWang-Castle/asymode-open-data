"""ERA5 windows of DATASET_DESIGN v1 systems (section 8) from ARCO-ERA5, the store and decoding of fetch_arco_windows.py,
cropped to the system's domain box plus 1 degree and stored as float32 NetCDF:
  data/raw/era5_v1/<system>.nc   main (u10 v10 i10fg t2m d2m cape swvl1 tcc sp tp sf), gust (fg10) and extra
                                 (ptype sd lai_hv cp swvl2) sets, 216 hours from the window start
Every file is logged to data_provenance/downloads.jsonl. Raw windows are deleted once their features are built
(section 8), so a later rebuild fetches them again.
usage: python fetch_era5_v1.py --systems S00001 S00002 ... [--workers 24]"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import fetch_arco_windows as A  # noqa: E402
from frame_common import EXP, OUT, PREFIX_H, HORIZON_H, ROOT, counties  # noqa: E402

DST = ROOT / "data" / "raw" / "era5_v1"
VARS = {**A.MAIN, **A.GUST, **A.EXTRA}
T = PREFIX_H + HORIZON_H


def box(system: str, sc: pd.DataFrame, cty: pd.DataFrame) -> tuple[int, int, int, int]:
    """Row/column range on the 0.25-degree global grid (row 0 = 90 N, column 0 = 0 E) of the domain box + 1 degree."""
    f = set(sc[sc.system == system].fips)
    c = cty[cty.fips.isin(f)]
    la0, la1 = c.lat.min() - 1.0 - 0.5, c.lat.max() + 1.0 + 0.5        # centroids + 1 deg + half a county
    lo0, lo1 = c.lon.min() - 1.0 - 0.5, c.lon.max() + 1.0 + 0.5
    r0, r1 = int(np.floor((90.0 - la1) / 0.25)), int(np.ceil((90.0 - la0) / 0.25)) + 1
    c0, c1 = int(np.floor((lo0 % 360) / 0.25)), int(np.ceil((lo1 % 360) / 0.25)) + 1
    return max(r0, A.R0), min(r1, A.R1), max(c0, A.C0), min(c1, A.C1)


def chunk(var, t, rb):
    import numcodecs
    k = int((t - A.EPOCH) / pd.Timedelta(hours=1))
    r = A._session.get(f"{A.BASE}/{var}/{k}.0.0", timeout=120)
    r.raise_for_status()
    a = np.frombuffer(numcodecs.Blosc().decode(r.content), dtype="<f4").reshape(721, 1440)
    r0, r1, c0, c1 = rb
    return a[r0:r1, c0:c1].copy()


def fetch(system: str, t0: pd.Timestamp, rb, workers: int) -> Path:
    import xarray as xr
    dst = DST / f"{system}.nc"
    if dst.exists():
        return dst
    times = pd.date_range(t0, periods=T, freq="h")
    jobs = [(v, t) for v in VARS for t in times]
    with ThreadPoolExecutor(workers) as ex:
        arrs = list(ex.map(lambda j: chunk(j[0], j[1], rb), jobs))
    data, i = {}, 0
    for v in VARS:
        data[VARS[v]] = np.stack(arrs[i:i + T]); i += T
    r0, r1, c0, c1 = rb
    lat = 90.0 - 0.25 * np.arange(r0, r1); lon = 0.25 * np.arange(c0, c1) - 360.0
    ds = xr.Dataset({k: (("valid_time", "latitude", "longitude"), v.astype(np.float32)) for k, v in data.items()},
                    coords=dict(valid_time=times.values, latitude=lat, longitude=lon),
                    attrs=dict(source="ARCO-ERA5 analysis-ready 0.25 deg, hourly", system=system))
    DST.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part")
    enc = {k: dict(zlib=True, complevel=4) for k in data}
    ds.to_netcdf(tmp, engine="h5netcdf", encoding=enc)
    tmp.rename(dst)
    rec = dict(file=str(dst.relative_to(ROOT)), source=f"ARCO-ERA5 {A.BASE}", variables=sorted(data), box_rows_cols=list(rb),
               hours=[str(times[0]), str(times[-1])], downloaded_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
               bytes=dst.stat().st_size, sha256=hashlib.sha256(dst.read_bytes()).hexdigest())
    with open(EXP / "data_provenance" / "downloads.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    return dst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    sy = pd.read_parquet(OUT / "systems.parquet").set_index("system")
    sc = pd.read_parquet(OUT / "system_counties.parquet")
    cty = counties()
    for s in a.systems:
        t = dt.datetime.now()
        rb = box(s, sc, cty)
        p = fetch(s, pd.Timestamp(sy.at[s, "window_start"]), rb, a.workers)
        print("saved", s, rb, p.stat().st_size, f"{(dt.datetime.now() - t).total_seconds():.0f} s", flush=True)


if __name__ == "__main__":
    main()
