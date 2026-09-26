"""3 km nodes with geography (population share, population-weighted elevation, canopy, poorly drained share; the rule of
build_nodes_hrrr.py) for every sampled county of a tranche: the nodes of nodes_hrrr_e3.parquet where they exist, built
by build_nodes_hrrr.one for the rest (county polygons of build_geography_v1.county_frame), and the latitude and
longitude of each node's HRRR cell centre (inverse Lambert conformal of the HRRR grid).
Output: data/interim/panel_v1/nodes_geo_<tranche>.parquet (fips, hr, hc, w_pop, z, canopy, wet, lat, lon)."""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import build_nodes_hrrr as NH  # noqa: E402
import build_subgrid_nodes as SN  # noqa: E402
from frame_common import INTERIM, OUT  # noqa: E402


def cell_latlon(hr, hc):
    import pyproj
    p = pyproj.Proj(proj="lcc", lat_1=38.5, lat_2=38.5, lat_0=38.5, lon_0=-97.5, R=6371229.0)
    x0, y0 = p(-122.719528, 21.138123)
    lon, lat = p(x0 + np.asarray(hc) * 3000.0, y0 + np.asarray(hr) * 3000.0, inverse=True)
    return lat, lon


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tranche", default="D")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    from build_geography_v1 import county_frame
    cs = pd.read_parquet(OUT / f"county_sample_{a.tranche}.parquet")
    need = sorted(set(cs[cs.sampled].fips))
    old = pd.read_parquet(INTERIM / "geo_weather" / "nodes_hrrr_e3.parquet")
    have = old[old.fips.isin(need)]
    todo = [f for f in need if f not in set(have.fips)]
    C = county_frame()
    parts, missing = [have], []
    with ProcessPoolExecutor(a.workers, initializer=SN._init) as ex:
        for n, (f, r) in enumerate(ex.map(NH.one, [(f, C.geometry[f]) for f in todo], chunksize=4), 1):
            (missing.append(f) if r is None else parts.append(r))
            if n % 50 == 0:
                print(f"{n}/{len(todo)}", flush=True)
    t = pd.concat(parts, ignore_index=True)[["fips", "hr", "hc", "w_pop", "z", "canopy", "wet"]]
    t["lat"], t["lon"] = cell_latlon(t.hr.to_numpy(), t.hc.to_numpy())
    t.to_parquet(OUT / f"nodes_geo_{a.tranche}.parquet", index=False)
    print("counties", t.fips.nunique(), "of", len(need), "nodes", len(t), "built", len(todo) - len(missing), "missing", missing[:10], flush=True)


if __name__ == "__main__":
    main()
