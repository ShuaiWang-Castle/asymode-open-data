"""ERA5 grid to county aggregation.

The reanalysis is a 0.25-degree grid; the target is a county. Something has to
turn one into the other, and the choice is a modelling decision that belongs in
the paper, not buried in a helper.

**Area weighting is what is implemented here.** For each county the overlapping
grid cells are found and weighted by the area of the intersection. It is exact
given the boundaries, needs no data beyond the Census cartographic boundaries,
and is reproducible from primary sources.

**Population weighting is not implemented, deliberately.** Outages are counted per
customer, so weighting the drivers by where the customers are is arguably the
better choice -- but doing it honestly needs a *gridded* population product
(WorldPop, GPW, LandScan) to distribute population within a county. County-level
population totals cannot do it: they are constant inside the county and reduce
exactly to area weighting. Adding a gridded layer is a later robustness check,
not a one-line switch, and claiming population weighting without one would be
false.

Large western counties are where the two would differ most, and they are also
where a 0.25-degree cell covers a large share of the county. The paper should say
which counties are sensitive to this rather than assert the choice does not matter.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def cell_polygons(lats: np.ndarray, lons: np.ndarray):
    """Grid-cell boxes for a regular lat/lon grid, as a GeoDataFrame."""
    import geopandas as gpd
    from shapely.geometry import box

    dlat = float(abs(np.diff(lats)[0])); dlon = float(abs(np.diff(lons)[0]))
    recs, geoms = [], []
    for i, la in enumerate(lats):
        for j, lo in enumerate(lons):
            recs.append((i, j))
            geoms.append(box(lo - dlon / 2, la - dlat / 2, lo + dlon / 2, la + dlat / 2))
    g = gpd.GeoDataFrame(pd.DataFrame(recs, columns=["i", "j"]), geometry=geoms,
                         crs="EPSG:4326")
    return g


def county_weights(lats: np.ndarray, lons: np.ndarray, shp: Path,
                   fips: list[str] | None = None) -> pd.DataFrame:
    """Area-overlap weights: one row per (fips, i, j) with weights summing to 1.

    Computed in an equal-area projection so the areas mean what they say.
    """
    import geopandas as gpd

    cty = gpd.read_file(shp)
    cty["fips"] = cty["STATEFP"].astype(str).str.zfill(2) + cty["COUNTYFP"].astype(str).str.zfill(3)
    if fips is not None:
        cty = cty[cty["fips"].isin(set(fips))]
    cells = cell_polygons(lats, lons)

    EA = "EPSG:5070"                      # NAD83 / Conus Albers, equal area
    cty_ea = cty[["fips", "geometry"]].to_crs(EA)
    cells_ea = cells.to_crs(EA)

    inter = gpd.overlay(cty_ea, cells_ea, how="intersection", keep_geom_type=True)
    inter["area"] = inter.geometry.area
    inter = inter[inter["area"] > 0]
    inter["w"] = inter["area"] / inter.groupby("fips")["area"].transform("sum")
    return inter[["fips", "i", "j", "w"]].reset_index(drop=True)


def apply_weights(field: np.ndarray, w: pd.DataFrame, fips: list[str]) -> np.ndarray:
    """(T, nlat, nlon) grid -> (C, T) county series, in the order of `fips`."""
    idx = {f: k for k, f in enumerate(fips)}
    out = np.zeros((len(fips), field.shape[0]), dtype=np.float32)
    sub = w[w["fips"].isin(idx)]
    for f, g in sub.groupby("fips"):
        vals = field[:, g["i"].to_numpy(), g["j"].to_numpy()]      # (T, n_cells)
        out[idx[f]] = (vals * g["w"].to_numpy()[None, :]).sum(axis=1)
    return out


def derive_channels(ds) -> dict:
    """Turn raw ERA5 fields into the driver channels the rates read.

    Kept explicit rather than clever: every channel is a named function of named
    reanalysis variables, so the covariate list in the paper can be read off this
    function and nothing enters the model unnamed.
    """
    import numpy as np

    v = {k: ds[k].values for k in ds.data_vars}
    out = {}
    if "u10" in v and "v10" in v:
        out["wind_speed"] = np.hypot(v["u10"], v["v10"])
    for src, dst in (("i10fg", "gust"), ("fg10", "gust"), ("t2m", "t2m"),
                     ("tp", "precip"), ("sf", "snowfall"), ("cape", "cape"),
                     ("swvl1", "soil_moisture"), ("tcc", "cloud"), ("sp", "pressure")):
        if src in v and dst not in out:
            out[dst] = v[src]
    if "t2m" in v and "d2m" in v:
        # Magnus formula; relative humidity from temperature and dewpoint.
        a, b = 17.625, 243.04
        tc, dc = v["t2m"] - 273.15, v["d2m"] - 273.15
        out["rh"] = 100 * np.exp(a * dc / (b + dc) - a * tc / (b + tc))
    if "t2m" in out:
        out["t2m_c"] = out.pop("t2m") - 273.15
    # ERA5 reports accumulations in metres of water equivalent per hour. Millimetres
    # per hour is the unit anyone reading a rain rate expects, and keeping the raw
    # metres invites a silent factor of a thousand in a coefficient.
    for k in ("precip", "snowfall"):
        if k in out:
            out[k] = out[k] * 1000.0
    return out
