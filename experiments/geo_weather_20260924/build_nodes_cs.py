"""Cell x elevation-band nodes (DESIGN v1 (a); REVIEW_formal section 5, LITERATURE_framework Part C.1).

A node is (ERA5 0.25-degree cell c, band b of dz), dz = elevation - the cell's ERA5 model orography
(surface geopotential / g), with fixed band edges -300, -150, -50, +50, +150, +300 m (finer resolution in the
tails, the same edges for every county). Nodes live inside one cell, so a node's weather is its own cell's
weather, and only the bands change it (through the lapse rate). Per node, from the 150 m land pixels of the
county (the rasters of open_gcrk_20260919/build_geography.py; gNATSGO on the 80 m lattice):
  w_pop    WorldPop 2020 people (1 km value per pixel, plus 1% of the county mean)
  w_area   pixel count
  population-weighted means of dz, tpi, slope, canopy (%), forest (NLCD 41-43, 90), deciduous (41 + half of
  43), developed (21-24), wet (poorly drained or wetter)
Output: data/interim/geo_weather/nodes_cs.parquet (one row per county x cell x band);
provenance data_provenance/nodes_cs.json.
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
sys.path.insert(0, str(HERE))
import build_geography as BG  # noqa: E402
import build_subgrid_nodes as SN  # noqa: E402

OUT = ROOT / "data" / "interim" / "geo_weather"
EDGES = np.array([-300.0, -150.0, -50.0, 50.0, 150.0, 300.0])
COLS = ["dz", "tpi", "slope", "canopy", "forest", "deciduous", "developed", "wet"]


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
    ci = np.clip(np.rint((SN.LAT0 - lat) / SN.STEP).astype(int), 0, zc.shape[0] - 1)
    cj = np.clip(np.rint((lon - SN.LON0) / SN.STEP).astype(int), 0, zc.shape[1] - 1)
    z = dem[rr, cc]; dz = z - zc[ci, cj]
    wet = np.zeros(len(rr))
    g80 = BG.RAW / "gnatsgo80" / f"{fips}.npz"
    if g80.exists():
        gz = np.load(g80); mk = gz["mukey"]; i0, j0, Wm, Hm = gz["lattice"]
        gx0 = -2356155.0 + i0 * 80.0; gy1 = 2399905.0 - j0 * 80.0
        col = np.floor((xs - gx0) / 80.0).astype(int); row = np.floor((gy1 - ys) / 80.0).astype(int)
        inb = (col >= 0) & (col < Wm) & (row >= 0) & (row < Hm)
        keys = np.full(len(rr), -1, np.int64); keys[inb] = mk[row[inb], col[inb]]
        wet = SN.WET.reindex(keys).fillna(0.0).to_numpy()
    with rasterio.open(SN.POP) as src:
        inv = ~src.transform
        pcf, prf = inv * (np.asarray(lon), np.asarray(lat))
        pr, pc = np.floor(prf).astype(np.int64), np.floor(pcf).astype(np.int64)
        r0, r1, c0, c1 = pr.min(), pr.max() + 1, pc.min(), pc.max() + 1
        win = src.read(1, window=((r0, r1), (c0, c1)))
    pop = win[pr - r0, pc - c0].astype(np.float64); pop = np.where(pop > 0, pop, 0.0)
    w = pop + 0.01 * max(pop.mean(), 1e-6)
    can = tcc[rr, cc]; can = np.where(np.isfinite(can) & (can <= 100), can, 0.0)
    lcl = lc[rr, cc]
    att = dict(dz=dz, tpi=tpi[rr, cc], slope=slope[rr, cc], canopy=can,
               forest=np.isin(lcl, (41, 42, 43, 90)).astype(float),
               deciduous=(lcl == 41).astype(float) + 0.5 * (lcl == 43), developed=np.isin(lcl, (21, 22, 23, 24)).astype(float),
               wet=wet)
    band = np.digitize(dz, EDGES)
    d = pd.DataFrame(dict(i=ci, j=cj, band=band, w_pop=w, w_area=1.0, **{k: v * w for k, v in att.items()}))
    g = d.groupby(["i", "j", "band"], as_index=False).sum()
    for k in COLS:
        g[k] = g[k] / g["w_pop"]
    g["fips"] = fips
    return fips, g


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    zc = SN.orography()
    fips = sorted(set(np.load(SN.FEAT)["fips"].astype(str)))
    C = BG.counties()
    jobs = [(f, C.geometry[f], zc) for f in fips]
    parts, missing = [], []
    with ProcessPoolExecutor(2, initializer=SN._init) as ex:
        for n, (f, r) in enumerate(ex.map(one, jobs, chunksize=8), 1):
            (missing.append(f) if r is None else parts.append(r))
            if n % 300 == 0:
                print(f"{n}/{len(fips)}", flush=True)
    t = pd.concat(parts, ignore_index=True)
    t["w_pop"] = t["w_pop"] / t.groupby("fips")["w_pop"].transform("sum")
    t["w_area"] = t["w_area"] / t.groupby("fips")["w_area"].transform("sum")
    t = t[["fips", "i", "j", "band", "w_pop", "w_area"] + COLS]
    t.to_parquet(OUT / "nodes_cs.parquet", index=False)
    per = t.groupby("fips").size()
    sd = t.groupby("fips").apply(lambda g: float(np.sqrt(np.average((g.dz - np.average(g.dz, weights=g.w_pop)) ** 2,
                                                                      weights=g.w_pop))))
    meta = dict(counties=int(t.fips.nunique()), missing=missing, rows=int(len(t)), band_edges_m=EDGES.tolist(),
                nodes_per_county=dict(median=float(per.median()), p90=float(per.quantile(0.9)), max=int(per.max())),
                pop_weighted_sd_dz_m=dict(median=float(sd.median()), p90=float(sd.quantile(0.9)),
                                          p99=float(sd.quantile(0.99))),
                pop_share_outside_pm50m=float((t.w_pop * (t.band != 3)).sum() / t.fips.nunique()))
    (HERE / "data_provenance" / "nodes_cs.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
