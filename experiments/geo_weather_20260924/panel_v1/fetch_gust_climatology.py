"""Local gust climatology for the hazard dictionary v2 (idea I12; physical review of the panel design, section 5.1:
damage scales with the exceedance of the local 98th percentile, Klawa & Ulbrich 2003). The hourly maximum 10 m gust
(ERA5 fg10, ARCO-ERA5) at one random hour of every day of 2008-2017 (seed 20260926; 3,653 hours, before the frame), on
the CONUS box of fetch_arco_windows.py; per cell the 90th, 95th, 98th and 99th percentiles and the mean.
Output: data/interim/panel_v1/era5_fg10_clim_2008_2017.npz; provenance line in data_provenance/downloads.jsonl."""
from __future__ import annotations

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
from frame_common import EXP, OUT, ROOT  # noqa: E402


def main() -> None:
    import numcodecs
    numcodecs.blosc.set_nthreads(1)          # one decompression thread per request (the machine is shared)
    rng = np.random.default_rng(20260926)
    days = pd.date_range("2008-01-01", "2017-12-31", freq="D")
    hours = [d + pd.Timedelta(hours=int(h)) for d, h in zip(days, rng.integers(0, 24, len(days)))]
    var = next(k for k, v in A.GUST.items() if v == "fg10")
    with ThreadPoolExecutor(4) as ex:
        arr = np.stack(list(ex.map(lambda t: A.chunk(var, t), hours))).astype(np.float32)
    q = np.percentile(arr, [90, 95, 98, 99], axis=0).astype(np.float32)
    f = OUT / "era5_fg10_clim_2008_2017.npz"
    lat = 50.0 - 0.25 * np.arange(A.R1 - A.R0); lon = -125.0 + 0.25 * np.arange(A.C1 - A.C0)
    np.savez_compressed(f, p90=q[0], p95=q[1], p98=q[2], p99=q[3], mean=arr.mean(0).astype(np.float32),
                        latitude=lat, longitude=lon, n_hours=len(hours))
    rec = dict(file=str(f.relative_to(ROOT)), source=f"ARCO-ERA5 {A.BASE} variable {var}, one random hour per day 2008-2017",
               hours=len(hours), seed=20260926, downloaded_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
               bytes=f.stat().st_size, sha256=hashlib.sha256(f.read_bytes()).hexdigest())
    with open(EXP / "data_provenance" / "downloads.jsonl", "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(rec, "p98 median over cells", float(np.median(q[2])), flush=True)


if __name__ == "__main__":
    main()
