"""Precompute area-overlap weights from the ERA5 0.25-degree grid to counties.

Needs no reanalysis data and no credentials: the ERA5 grid is a fixed regular
lat/lon lattice, so the weights depend only on the requested bounding box and the
Census boundaries. Computing them now means county aggregation is a matrix
multiply the moment the first field lands.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from asymode.weather import county_weights   # noqa: E402

SHP = ROOT / "data/raw/census/cb_county/cb_2023_us_county_500k.shp"
# CONUS plus a margin, matching the ERA5 request in scripts/fetch_era5.py.
AREA = dict(north=50.0, west=-125.0, south=24.0, east=-66.0)
RES = 0.25

if __name__ == "__main__":
    lats = np.arange(AREA["north"], AREA["south"] - 1e-9, -RES)
    lons = np.arange(AREA["west"], AREA["east"] + 1e-9, RES)
    print(f"ERA5 subgrid: {len(lats)} lat x {len(lons)} lon = {len(lats)*len(lons):,} cells")

    panels = sorted((ROOT / "data/interim").glob("panel_*.npz"))
    fips = sorted({f for p in panels
                   for f in np.load(p, allow_pickle=True)["fips"].tolist()})
    print(f"counties across {len(panels)} archived panels: {len(fips)}")

    w = county_weights(lats, lons, SHP, fips=fips)
    out = ROOT / "data/interim/era5_county_weights.parquet"
    w.to_parquet(out, index=False)

    per = w.groupby("fips")["w"].agg(["size", "max"])
    print(f"\nweights: {len(w):,} (county, cell) pairs over {w.fips.nunique()} counties")
    print(f"  cells per county : median {per['size'].median():.0f}  "
          f"p90 {per['size'].quantile(0.9):.0f}  max {per['size'].max()}")
    print(f"  largest single-cell weight: median {per['max'].median():.2f}  "
          f"(1.00 means the county sits inside one cell)")
    n1 = int((per["size"] == 1).sum())
    print(f"  counties fully inside one cell: {n1} ({100*n1/len(per):.0f}%) "
          f"-- for these, area vs population weighting cannot differ")
    miss = sorted(set(fips) - set(w.fips))
    if miss:
        print(f"  no overlap for {len(miss)} counties (outside the box): {miss[:8]}")
    print(f"\nwritten: {out.relative_to(ROOT)}")
