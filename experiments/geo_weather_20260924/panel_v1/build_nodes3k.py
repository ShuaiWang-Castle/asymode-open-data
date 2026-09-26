"""Population nodes on the HRRR 3 km grid for every CONUS county (DATASET_DESIGN v1, sections 3.3 and 5): a node is
(county, HRRR cell), weighted by the WorldPop 2020 people of the 1 km cells whose centres fall in both. Its point is
the population-weighted mean of those centres. Used for the exposure share a of a county to a warning polygon or zone.
The node set of build_nodes_hrrr.py (150 m land pixels, geography attributes) covers 2,409 counties; this one covers
all CONUS counties of the 2020 vintage (the units of EAGLE-I) and carries population only.
Output: data/interim/panel_v1/nodes3k.parquet (fips, hr, hc, lon, lat, pop) and county_pop.parquet."""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
from build_nodes_hrrr import hrrr_ij  # noqa: E402

OUT = ROOT / "data" / "interim" / "panel_v1"
POP = ROOT / "data" / "raw" / "population" / "usa_ppp_2020_1km_Aggregated.tif"
CB = "zip://" + str(ROOT / "data" / "raw" / "census" / "cb_2020_us_county_500k.zip")
NOT_CONUS = {"02", "15", "60", "66", "69", "72", "78"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cty = gpd.read_file(CB)[["GEOID", "geometry"]]
    cty = cty[~cty.GEOID.str[:2].isin(NOT_CONUS)].to_crs(4326)
    with rasterio.open(POP) as r:
        win = rasterio.windows.from_bounds(-125.5, 24.0, -66.5, 49.8, r.transform)
        a = r.read(1, window=win)
        tr = r.window_transform(win)
    rows, cols = np.nonzero(a > 0)
    pop = a[rows, cols].astype("float64")
    lon, lat = rasterio.transform.xy(tr, rows, cols)
    pts = gpd.GeoDataFrame({"pop": pop}, geometry=gpd.points_from_xy(lon, lat), crs=4326)
    print("populated 1 km cells", len(pts), "people", round(pop.sum() / 1e6, 1), "M", flush=True)
    j = gpd.sjoin(pts, cty, predicate="within", how="inner")
    print("in CONUS counties", len(j), "people", round(j["pop"].sum() / 1e6, 1), "M", flush=True)
    hr, hc = hrrr_ij(j.geometry.y.to_numpy(), j.geometry.x.to_numpy())
    d = pd.DataFrame({"fips": j.GEOID.to_numpy(), "hr": hr, "hc": hc, "pop": j["pop"].to_numpy(),
                      "wx": j.geometry.x.to_numpy() * j["pop"].to_numpy(), "wy": j.geometry.y.to_numpy() * j["pop"].to_numpy()})
    n = d.groupby(["fips", "hr", "hc"], as_index=False)[["pop", "wx", "wy"]].sum()
    n["lon"], n["lat"] = n.wx / n["pop"], n.wy / n["pop"]
    n = n[["fips", "hr", "hc", "lon", "lat", "pop"]]
    missing = sorted(set(cty.GEOID) - set(n.fips))
    # a county with no populated 1 km cell centre inside it (tiny or empty): one node at its representative point
    if missing:
        rp = cty.set_index("GEOID").loc[missing].geometry.representative_point()
        r_, c_ = hrrr_ij(rp.y.to_numpy(), rp.x.to_numpy())
        n = pd.concat([n, pd.DataFrame({"fips": missing, "hr": r_, "hc": c_, "lon": rp.x.to_numpy(), "lat": rp.y.to_numpy(),
                                        "pop": 1.0})], ignore_index=True)
    n.to_parquet(OUT / "nodes3k.parquet", index=False)
    cp = n.groupby("fips")["pop"].sum().rename("pop").reset_index()
    cp.to_parquet(OUT / "county_pop.parquet", index=False)
    print("counties", n.fips.nunique(), "nodes", len(n), "median per county", int(n.groupby("fips").size().median()),
          "counties given a single representative node", len(missing), flush=True)


if __name__ == "__main__":
    main()
