"""Customer nodes on the HRRR 3 km grid: a node is (county, HRRR cell). From the 150 m land pixels of the county (the
rasters of build_nodes_cs.py) with WorldPop 2020 weights (plus 1% of the county mean): the node's population share
and its population-weighted elevation (3DEP), canopy and poorly drained share. HRRR grid: Lambert conformal, 1799 x
1059 cells of 3 km, parameters as in contrib/DATA_SOURCES_input.md section (c) (La1 21.138123, Lo1 237.280472, LoV
262.5, Latin 38.5, sphere R = 6,371,229 m); rows run south to north.
Output: data/interim/geo_weather/nodes_hrrr_<tag>.parquet (fips, hr, hc, w_pop, z, canopy, wet).
usage: python build_nodes_hrrr.py --feat data/interim/geo_weather/features_w1.npz --tag w1
"""
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
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "open_gcrk_20260919"))
sys.path.insert(0, str(HERE))
import build_geography as BG  # noqa: E402
import build_subgrid_nodes as SN  # noqa: E402

OUT = ROOT / "data" / "interim" / "geo_weather"
R_E, N_L = 6371229.0, np.sin(np.deg2rad(38.5))
F_L = np.cos(np.deg2rad(38.5)) * np.tan(np.pi / 4 + np.deg2rad(38.5) / 2) ** N_L / N_L


def _rho(lat):
    return R_E * F_L / np.tan(np.pi / 4 + np.deg2rad(lat) / 2) ** N_L


def _lcc(lat, lon):
    th = N_L * np.deg2rad(np.asarray(lon) + 97.5)
    return _rho(np.asarray(lat)) * np.sin(th), _rho(38.5) - _rho(np.asarray(lat)) * np.cos(th)


X0, Y0 = _lcc(21.138123, -122.719528)


def hrrr_ij(lat, lon):
    x, y = _lcc(lat, lon)
    return np.rint((y - Y0) / 3000).astype(int), np.rint((x - X0) / 3000).astype(int)


def one(args):
    fips, geom = args
    import pyproj, rasterio
    from rasterio.features import geometry_mask
    from rasterio.transform import Affine
    x0, y0, W, H = BG.grid(geom)
    dem = BG.read(BG.RAW / "dem" / f"{fips}.tif", H, W, -9999)
    tcc = BG.read(BG.RAW / "tcc" / f"{fips}.tif", H, W, 255)
    lc = BG.read(BG.RAW / "nlcd" / f"{fips}.tif", H, W, 0)
    inside = ~geometry_mask([geom], out_shape=(H, W), transform=Affine(BG.RES, 0, x0, 0, -BG.RES, y0 + H * BG.RES))
    land = inside & np.isfinite(dem) & (lc != 11)
    if land.sum() < 20:
        return fips, None
    rr, cc = np.nonzero(land)
    xs = x0 + (cc + 0.5) * BG.RES; ys = y0 + H * BG.RES - (rr + 0.5) * BG.RES
    lon, lat = pyproj.Transformer.from_crs(5070, 4326, always_xy=True).transform(xs, ys)
    hr, hc = hrrr_ij(lat, lon)
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
    d = pd.DataFrame(dict(hr=hr, hc=hc, w_pop=w, z=dem[rr, cc] * w, canopy=can * w, wet=wet * w))
    g = d.groupby(["hr", "hc"], as_index=False).sum()
    for k in ("z", "canopy", "wet"):
        g[k] = g[k] / g["w_pop"]
    g["w_pop"] = g["w_pop"] / g["w_pop"].sum()
    g["fips"] = fips
    return fips, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    fips = sorted(set(np.load(ROOT / a.feat)["fips"].astype(str)))
    C = BG.counties()
    parts, missing = [], []
    with ProcessPoolExecutor(3, initializer=SN._init) as ex:
        for n, (f, r) in enumerate(ex.map(one, [(f, C.geometry[f]) for f in fips], chunksize=8), 1):
            (missing.append(f) if r is None else parts.append(r))
            if n % 300 == 0:
                print(f"{n}/{len(fips)}", flush=True)
    t = pd.concat(parts, ignore_index=True)[["fips", "hr", "hc", "w_pop", "z", "canopy", "wet"]]
    assert t.hr.between(0, 1058).all() and t.hc.between(0, 1798).all()
    t.to_parquet(OUT / f"nodes_hrrr_{a.tag}.parquet", index=False)
    print("counties", t.fips.nunique(), "nodes", len(t), "median per county", int(t.groupby("fips").size().median()),
          "missing", missing[:5])


if __name__ == "__main__":
    main()
