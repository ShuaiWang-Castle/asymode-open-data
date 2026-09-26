"""Complete the county rasters of the geography pipeline for every CONUS county of the 2020 vintage (DATASET_DESIGN v1,
sections 4.6 and 8): USGS 3DEP elevation, USFS tree canopy cover and NLCD land cover (build_geography.fetch, 150 m,
EPSG:5070) and gNATSGO map units (build_geography_ext.fetch_mukey, 80 m), for the counties that do not have them yet;
then the map-unit attribute table for all map units seen. Grids follow the 2023 cartographic boundaries, as the
existing rasters do; Connecticut's eight 2020 counties use the 2020 boundaries. Downloads are logged by the two
modules (open_gcrk_20260919/data_provenance/geography_log.jsonl, geography_ext_log.jsonl)."""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import numpy as np
import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "open_gcrk_20260919"))
import build_geography as BG  # noqa: E402
import build_geography_ext as GE  # noqa: E402
from frame_common import counties  # noqa: E402


def geoms():
    g23 = BG.counties()
    g20 = gpd.read_file("zip://" + str(ROOT / "data" / "raw" / "census" / "cb_2020_us_county_500k.zip"))
    g20["fips"] = g20.STATEFP + g20.COUNTYFP
    g20 = g20[["fips", "STUSPS", "geometry"]].to_crs(5070).set_index("fips")
    return {f: (g23.geometry[f] if f in g23.index else g20.geometry[f]) for f in counties().fips}


def main() -> None:
    G = geoms()
    fips = sorted(G)
    todo = {k: [f for f in fips if not (BG.RAW / k / f"{f}.tif").exists()] for k in ("dem", "tcc", "nlcd")}
    todo["gnatsgo80"] = [f for f in fips if not (BG.RAW / "gnatsgo80" / f"{f}.npz").exists()]
    print({k: len(v) for k, v in todo.items()}, flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(3) as ex:
        futs = {ex.submit(BG.fetch, k, f, G[f], requests.Session()): (k, f) for k in ("dem", "tcc", "nlcd") for f in todo[k]}
        for n, fu in enumerate(as_completed(futs), 1):
            try:
                fu.result()
            except Exception as e:
                print("FAILED", futs[fu], e, flush=True)
            if n % 100 == 0:
                print(f"rasters {n}/{len(futs)} {time.time() - t0:.0f}s", flush=True)
    session = requests.Session()
    items = GE.gnatsgo_items(session)
    signer = GE.Signer(session)
    with ThreadPoolExecutor(3) as ex:
        futs = {ex.submit(GE.fetch_mukey, f, G[f], items, signer): f for f in todo["gnatsgo80"]}
        for n, fu in enumerate(as_completed(futs), 1):
            try:
                fu.result()
            except Exception as e:
                print("FAILED gnatsgo", futs[fu], e, flush=True)
            if n % 50 == 0:
                print(f"gnatsgo {n}/{len(futs)} {time.time() - t0:.0f}s", flush=True)
    keys = np.unique(np.concatenate([np.unique(np.load(p)["mukey"]) for p in sorted((BG.RAW / "gnatsgo80").glob("*.npz"))]))
    GE.mapunit_table(session, keys)
    print("done", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
