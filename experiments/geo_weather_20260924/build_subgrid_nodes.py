"""Sub-grid quadrature nodes (data v3, DESIGN section 2a): where a county's customers live, and under what
local conditions, relative to the ERA5 cell whose weather reaches them.

For every county of the twelve-event panel, every 150 m land pixel inside the county (the rasters of
open_gcrk_20260919/build_geography.py: 3DEP elevation, NLCD tree canopy, Annual NLCD land cover; gNATSGO
map units on the 80 m lattice of build_geography_ext.py) gets:
  cell      the ERA5 0.25-degree cell containing it
  dz        elevation minus that cell's ERA5 orography (surface geopotential / g), m
  tpi       elevation minus its mean over a 1.05 km window (terrain exposure), m
  slope, canopy, forest (NLCD 41-43, 90), developed (21-24), wet (map unit's dominant drainage somewhat
            poorly or wetter)
  weight    WorldPop 2020 people in its 1 km cell (plus 1% of the county mean, so empty land keeps a
            small weight)
Pixels are stratified into K = 16 nodes: population-weighted quartiles of dz x canopy above / below its
population-weighted median x wet / not wet. Per node: population share rho, population-weighted means of
the attributes, and the mixing weights pi[k, c] over the county's cells (population of node k in cell c),
so that a node's weather is sum_c pi[k, c] w_c(t) and its temperature can be moved by a lapse rate times dz.
Also per county: population-weighted and area-weighted cell weights (exposure-weighted weather).
Output: data/interim/geo_weather/nodes_e3.npz; provenance in data_provenance/nodes_e3.json.
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "open_gcrk_20260919"))
import build_geography as BG  # noqa: E402

OUT = ROOT / "data" / "interim" / "geo_weather"
POP = ROOT / "data" / "raw" / "population" / "usa_ppp_2020_1km_Aggregated.tif"
ORO = ROOT / "data" / "raw" / "era5_extra" / "era5_orography.nc"
FEAT = ROOT / "data" / "interim" / "open_gcrk" / "features_e3r2.npz"
K, C_MAX = 16, 24
ATTR = ["dz", "z", "tpi", "slope", "canopy", "forest", "developed", "wet", "area_share"]
LAT0, LON0, STEP = 50.0, -125.0, 0.25


def wquantile(v, w, q):
    o = np.argsort(v); c = np.cumsum(w[o]); c /= c[-1]
    return np.interp(q, c, v[o])


def orography():
    import xarray as xr
    ds = xr.open_dataset(ORO, engine="h5netcdf")
    z = ds["z"].squeeze().values / 9.80665
    lat = ds["latitude"].values; lon = ds["longitude"].values
    assert abs(lat[0] - LAT0) < 1e-6 and abs(lon[0] - LON0) < 1e-6 and abs(lat[1] - lat[0] + STEP) < 1e-6
    return z


def one(args):
    fips, geom, zc = args
    import pyproj, rasterio
    from rasterio.features import geometry_mask
    from rasterio.transform import Affine
    from scipy.ndimage import uniform_filter
    x0, y0, W, H = BG.grid(geom)
    dem = BG.read(BG.RAW / "dem" / f"{fips}.tif", H, W, -9999)
    tcc = BG.read(BG.RAW / "tcc" / f"{fips}.tif", H, W, 255)
    lc = BG.read(BG.RAW / "nlcd" / f"{fips}.tif", H, W, 0)
    inside = ~geometry_mask([geom], out_shape=(H, W), transform=Affine(BG.RES, 0, x0, 0, -BG.RES, y0 + H * BG.RES))
    land = inside & np.isfinite(dem) & (lc != 11)
    if land.sum() < 20:
        return fips, None
    zf = np.where(np.isfinite(dem), dem, np.nanmean(dem))
    tpi = zf - uniform_filter(zf, 7, mode="nearest")
    gy, gx = np.gradient(zf, BG.RES)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    rr, cc = np.nonzero(land)
    xs = x0 + (cc + 0.5) * BG.RES; ys = y0 + H * BG.RES - (rr + 0.5) * BG.RES
    lon, lat = pyproj.Transformer.from_crs(5070, 4326, always_xy=True).transform(xs, ys)
    ci = np.rint((LAT0 - lat) / STEP).astype(int); cj = np.rint((lon - LON0) / STEP).astype(int)
    ok = (ci >= 0) & (ci < zc.shape[0]) & (cj >= 0) & (cj < zc.shape[1])
    ci, cj = np.clip(ci, 0, zc.shape[0] - 1), np.clip(cj, 0, zc.shape[1] - 1)
    z = dem[rr, cc]; dz = z - zc[ci, cj]
    # wet soil from the gNATSGO lattice of build_geography_ext.py
    wet = np.zeros(len(rr))
    g80 = BG.RAW / "gnatsgo80" / f"{fips}.npz"
    if g80.exists():
        gz = np.load(g80); mk = gz["mukey"]; i0, j0, Wm, Hm = gz["lattice"]
        gx0 = -2356155.0 + i0 * 80.0; gy1 = 2399905.0 - j0 * 80.0
        col = np.floor((xs - gx0) / 80.0).astype(int); row = np.floor((gy1 - ys) / 80.0).astype(int)
        inb = (col >= 0) & (col < Wm) & (row >= 0) & (row < Hm)
        keys = np.full(len(rr), -1, np.int64); keys[inb] = mk[row[inb], col[inb]]
        wet = WET.reindex(keys).fillna(0.0).to_numpy()
    with rasterio.open(POP) as src:
        inv = ~src.transform
        pcf, prf = inv * (np.asarray(lon), np.asarray(lat))
        pr, pc = np.floor(prf).astype(np.int64), np.floor(pcf).astype(np.int64)
        r0, r1, c0, c1 = pr.min(), pr.max() + 1, pc.min(), pc.max() + 1
        win = src.read(1, window=((r0, r1), (c0, c1)))
    pop = win[pr - r0, pc - c0].astype(np.float64); pop = np.where(pop > 0, pop, 0.0)
    w = pop + 0.01 * max(pop.mean(), 1e-6)
    can = tcc[rr, cc]; can = np.where(np.isfinite(can) & (can <= 100), can, 0.0)
    lcl = lc[rr, cc]
    attr = dict(dz=dz, z=z, tpi=tpi[rr, cc], slope=slope[rr, cc], canopy=can, forest=np.isin(lcl, (41, 42, 43, 90)).astype(float),
                developed=np.isin(lcl, (21, 22, 23, 24)).astype(float), wet=wet)
    qz = wquantile(dz, w, [0.25, 0.5, 0.75]); qc = wquantile(can, w, 0.5)
    bz = np.digitize(dz, qz); bc = (can > qc).astype(int); bw = (wet >= 0.5).astype(int)
    node = bz * 4 + bc * 2 + bw
    cells, cinv = np.unique(np.stack([ci, cj], 1), axis=0, return_inverse=True)
    cinv = cinv.ravel()
    cw_pop = np.bincount(cinv, w, len(cells)); cw_area = np.bincount(cinv, None, len(cells)).astype(float)
    order = np.argsort(-cw_pop)[:C_MAX]                       # at most C_MAX cells, by population
    keep_cell = np.full(len(cells), -1); keep_cell[order] = np.arange(len(order))
    kc = keep_cell[cinv]; sel = kc >= 0
    A = np.zeros((K, len(ATTR))); rho = np.zeros(K); mix = np.zeros((K, C_MAX))
    for k in range(K):
        m = (node == k) & sel
        if not m.any():
            continue
        wk = w[m]; rho[k] = wk.sum()
        for a, name in enumerate(ATTR[:-1]):
            A[k, a] = np.average(attr[name][m], weights=wk)
        A[k, -1] = m.sum()
        mix[k] = np.bincount(kc[m], wk, C_MAX) / wk.sum()
    A[:, -1] /= max(A[:, -1].sum(), 1)
    rho /= rho.sum()
    cell_ij = np.full((C_MAX, 2), -1, int); cell_ij[:len(order)] = cells[order]
    pw = np.zeros(C_MAX); aw = np.zeros(C_MAX)
    pw[:len(order)] = cw_pop[order] / cw_pop[order].sum(); aw[:len(order)] = cw_area[order] / cw_area[order].sum()
    return fips, dict(attr=A, rho=rho, mix=mix, cells=cell_ij, pop_w=pw, area_w=aw, n_pixels=int(land.sum()),
                      pop_total=float(pop.sum()), dropped_pop_share=float(1 - w[sel].sum() / w.sum()))


def _init():
    global WET
    t = pd.read_parquet(BG.RAW / "gnatsgo_mapunits.parquet")
    WET = t["wet"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    zc = orography()
    fips = sorted(set(np.load(FEAT)["fips"].astype(str)))
    C = BG.counties()
    jobs = [(f, C.geometry[f], zc) for f in fips]
    res = {}
    with ProcessPoolExecutor(4, initializer=_init) as ex:
        for n, (f, r) in enumerate(ex.map(one, jobs, chunksize=8), 1):
            res[f] = r
            if n % 200 == 0:
                print(f"{n}/{len(fips)}", flush=True)
    good = [f for f in fips if res[f] is not None]
    arr = dict(fips=np.array(good), attr_names=np.array(ATTR),
               attr=np.stack([res[f]["attr"] for f in good]).astype(np.float32),
               rho=np.stack([res[f]["rho"] for f in good]).astype(np.float32),
               mix=np.stack([res[f]["mix"] for f in good]).astype(np.float32),
               cells=np.stack([res[f]["cells"] for f in good]).astype(np.int16),
               pop_w=np.stack([res[f]["pop_w"] for f in good]).astype(np.float32),
               area_w=np.stack([res[f]["area_w"] for f in good]).astype(np.float32),
               pop_total=np.array([res[f]["pop_total"] for f in good]),
               dropped_pop_share=np.array([res[f]["dropped_pop_share"] for f in good]))
    np.savez_compressed(OUT / "nodes_e3.npz", **arr)
    meta = dict(counties=len(good), missing=[f for f in fips if res[f] is None], K=K, C_MAX=C_MAX,
                max_dropped_pop_share=float(arr["dropped_pop_share"].max()),
                nodes_nonempty_median=float(np.median((arr["rho"] > 0).sum(1))))
    (HERE / "data_provenance" / "nodes_e3.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
