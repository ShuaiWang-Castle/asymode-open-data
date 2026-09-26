"""County geography descriptors of the model (the 40 geo features of build_v3p.features) for every CONUS county of the
2020 vintage, with the unchanged functions of open_gcrk_20260919/build_geography.py (terrain, canopy, land cover,
SSURGO soils, 50 km smoothing) and build_geography_ext.py (gNATSGO soil shares, five-point elevation, FIA forest
land). The counties are the 2023 cartographic boundaries, as for the existing tables, plus Connecticut's eight 2020
counties (2020 boundaries). Outputs: data/interim/panel_v1/geography_v1.parquet, geography_ext_v1.parquet."""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "open_gcrk_20260919"))
import build_geography as BG  # noqa: E402
import build_geography_ext as GE  # noqa: E402
from frame_common import OUT, RAW, counties  # noqa: E402


def county_frame():
    import geopandas as gpd
    g23 = BG.counties()
    g20 = gpd.read_file("zip://" + str(RAW / "census" / "cb_2020_us_county_500k.zip"))
    g20["fips"] = g20.STATEFP + g20.COUNTYFP
    g20 = g20[["fips", "STUSPS", "geometry"]].to_crs(5070).set_index("fips")
    fips = list(counties().fips)
    extra = [f for f in fips if f not in g23.index]
    return pd.concat([g23[g23.index.isin(fips)], g20.loc[extra]])


def gazetteer():
    gz = pd.read_csv(RAW / "census" / "2020_Gaz_counties_national.txt", sep="\t", dtype={"GEOID": str}, encoding="latin-1")
    gz.columns = [c.strip() for c in gz.columns]
    return gz.assign(fips=gz.GEOID.str.zfill(5)).set_index("fips")


def base_table(C, gz) -> pd.DataFrame:
    need = sorted(C.index)
    lat, lon = np.radians(gz.loc[need, "INTPTLAT"].to_numpy(float)), np.radians(gz.loc[need, "INTPTLONG"].to_numpy(float))
    ids = np.array(need)
    within = {}
    for i, f in enumerate(ids):
        d = 6371.0 * 2 * np.arcsin(np.sqrt(np.sin((lat - lat[i]) / 2) ** 2 + np.cos(lat[i]) * np.cos(lat) * np.sin((lon - lon[i]) / 2) ** 2))
        within[f] = ids[d <= BG.SMOOTH_KM].tolist()
    t0 = time.time()
    with ThreadPoolExecutor(4) as ex:
        futs = {ex.submit(BG.raster_stats, f, C.geometry[f]): f for f in need}
        rows = []
        for n, fu in enumerate(as_completed(futs), 1):
            try:
                rows.append(fu.result())
            except Exception as e:
                print("raster_stats failed", futs[fu], e, flush=True)
            if n % 300 == 0:
                print(f"raster stats {n}/{len(need)} {time.time() - t0:.0f}s", flush=True)
    stats = pd.DataFrame(rows).set_index("fips")
    areas = sorted({C.STUSPS[f] + f[2:] for f in need})
    soil = BG.soils(areas)
    soil["fips"] = [next((f for f in need if C.STUSPS[f] + f[2:] == s), None) for s in soil.areasymbol]
    soil = soil.dropna(subset=["fips"]).set_index("fips")
    tot = soil.acres.where(soil.acres > 0)
    stats["poorly_drained_share"] = soil.poor / tot
    stats["hydric_share"] = soil.hydric / tot
    stats["shallow_share"] = soil.shallow / tot
    stats["high_water_table_share"] = soil.hiwt / tot
    stats["root_limiting_share"] = soil.rootlimit / tot
    stats["windthrow_susceptibility"] = stats.forest_frac * stats.root_limiting_share
    stats["log_land_area_km2"] = np.log(gz.loc[stats.index, "ALAND"].astype(float) / 1e6)
    area = gz.loc[stats.index, "ALAND"].astype(float)
    for c in BG.SMOOTH:
        v = stats[c]
        stats[f"s50_{c}"] = [np.nansum(v.reindex(within[f]) * area.reindex(within[f]))
                             / area.reindex(within[f])[v.reindex(within[f]).notna()].sum() for f in stats.index]
    stats.attrs = {}
    return stats


def ext_table(C, gz, fips) -> pd.DataFrame:
    session = requests.Session()
    items = GE.gnatsgo_items(session)
    signer = GE.Signer(session)
    t0, rows = time.time(), []
    with ThreadPoolExecutor(4) as ex:
        futs = {ex.submit(GE.county_samples, f, C.geometry[f], items, signer, requests.Session()): f for f in fips}
        for n, fu in enumerate(as_completed(futs), 1):
            rows.append(fu.result())
            if n % 300 == 0:
                print(f"soil samples {n}/{len(fips)} {time.time() - t0:.0f}s", flush=True)
    allk = np.unique(np.concatenate([r["_mukey"] for r in rows]))
    mu = GE.mapunit_table(session, allk)
    soil = pd.DataFrame([GE.soil_descriptors(r, mu) for r in rows]).set_index("fips")
    e5 = GE.elevation5(session, gz, fips, C)
    fia = GE.fia_forest(session, gz, fips)
    return soil.join(e5).join(fia).loc[fips]


def main() -> None:
    C = county_frame()
    BG.counties = lambda: C                      # the functions below read the county polygons through this
    gz = gazetteer()
    b = base_table(C, gz)
    b.index.name = "fips"
    b.to_parquet(OUT / "geography_v1.parquet")
    print("base", b.shape, "missing", b.isna().sum()[lambda s: s > 0].to_dict(), flush=True)
    x = ext_table(C, gz, list(b.index))
    x.index.name = "fips"
    x.to_parquet(OUT / "geography_ext_v1.parquet")
    print("ext", x.shape, "missing", x.isna().sum()[lambda s: s > 0].to_dict(), flush=True)
    old = pd.read_parquet(ROOT / "data" / "interim" / "open_gcrk" / "geography_e3.parquet")
    common = sorted(set(old.index) & set(b.index))
    cols = [c for c in old.columns if c in b.columns and not c.startswith("s50_")]
    diff = (b.loc[common, cols] - old.loc[common, cols]).abs().max()
    print("max |v1 - e3| on common counties (unsmoothed columns):", diff[diff > 1e-9].round(6).to_dict(), flush=True)


if __name__ == "__main__":
    main()
