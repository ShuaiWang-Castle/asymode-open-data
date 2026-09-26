"""Five geography axes per CONUS county for the coverage audit of DATASET_DESIGN v1 (section 4.6), the same for every
regime: relief (SD of 3DEP elevation over the county's land pixels, 150 m), tree canopy cover (mean over land pixels),
share of poorly drained soils (gNATSGO map units drained somewhat poorly, poorly or very poorly, as the node pipeline
defines them; 80 m), customer density (log EAGLE-I 2024 modelled customers per km2 of land, Gazetteer 2020) and
distance to the coast (km from the Gazetteer internal point to the Natural Earth 1:10m coastline).
Output: data/interim/panel_v1/county_axes.parquet (fips, relief_sd, canopy, poorly_drained, log_cust_density,
coast_km) with national terciles over the gated counties computed later by the audit."""
from __future__ import annotations

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
sys.path.insert(0, str(ROOT / "experiments" / "open_gcrk_20260919"))
sys.path.insert(0, str(HERE))
import build_geography as BG  # noqa: E402
from frame_common import OUT, RAW, counties  # noqa: E402

WET = None


def _wet():
    global WET
    if WET is None:
        WET = pd.read_parquet(BG.RAW / "gnatsgo_mapunits.parquet")["wet"]
    return WET


def one(args):
    fips, geom = args
    from rasterio.features import geometry_mask
    from rasterio.transform import Affine
    x0, y0, W, H = BG.grid(geom)
    out = dict(fips=fips, relief_sd=np.nan, canopy=np.nan, poorly_drained=np.nan)
    try:
        dem = BG.read(BG.RAW / "dem" / f"{fips}.tif", H, W, -9999)
        tcc = BG.read(BG.RAW / "tcc" / f"{fips}.tif", H, W, 255)
        lc = BG.read(BG.RAW / "nlcd" / f"{fips}.tif", H, W, 0)
    except Exception:
        return out
    inside = ~geometry_mask([geom], out_shape=(H, W), transform=Affine(BG.RES, 0, x0, 0, -BG.RES, y0 + H * BG.RES))
    land = inside & np.isfinite(dem) & (lc != 11)
    if land.sum() >= 20:
        out["relief_sd"] = float(np.nanstd(dem[land]))
        c = tcc[land]
        out["canopy"] = float(np.nanmean(np.where(c <= 100, c, np.nan)))
    g80 = BG.RAW / "gnatsgo80" / f"{fips}.npz"
    if g80.exists():
        z = np.load(g80)
        mk, (i0, j0, Wm, Hm) = z["mukey"], z["lattice"]
        gx0, gy1 = -2356155.0 + i0 * 80.0, 2399905.0 - j0 * 80.0
        ins = ~geometry_mask([geom], out_shape=(Hm, Wm), transform=Affine(80.0, 0, gx0, 0, -80.0, gy1))
        w = _wet().reindex(mk[ins].ravel()).to_numpy()
        if np.isfinite(w).sum() >= 20:
            out["poorly_drained"] = float(np.nanmean(w))
    return out


def main() -> None:
    import geopandas as gpd
    from fetch_geography_conus import geoms
    G = geoms()
    with ProcessPoolExecutor(4) as ex:
        rows = list(ex.map(one, list(G.items()), chunksize=20))
    ax = pd.DataFrame(rows).set_index("fips")
    c = counties().set_index("fips")
    g = pd.read_csv(RAW / "census" / "2020_Gaz_counties_national.txt", sep="\t", dtype=str)
    g.columns = [x.strip() for x in g.columns]
    land_km2 = g.set_index("GEOID").ALAND.astype(float) / 1e6
    ax["log_cust_density"] = np.log(c.customers.clip(lower=1) / land_km2.reindex(c.index).clip(lower=1))
    coast = gpd.read_file("zip://" + str(RAW / "geography" / "coast" / "ne_10m_coastline.zip")).to_crs(5070)
    coast = coast[coast.intersects(gpd.GeoSeries.from_xy([-130], [20], crs=4326).to_crs(5070).buffer(6e6).iloc[0])]
    pts = gpd.GeoSeries.from_xy(c.lon, c.lat, crs=4326).to_crs(5070)
    line = coast.geometry.union_all()
    ax["coast_km"] = pts.distance(line).to_numpy() / 1000.0
    ax = ax.reindex(c.index)
    ax.index.name = "fips"
    ax.reset_index().to_parquet(OUT / "county_axes.parquet", index=False)
    print(ax.describe().T.round(3).to_string())
    print("missing per axis:", ax.isna().sum().to_dict())


if __name__ == "__main__":
    main()
