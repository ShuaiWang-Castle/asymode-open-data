"""Hazard dictionary v2 (hazard_v2.py) for the sampled counties of each system of a tranche, from two weather sources:
  --source era5   the system's cropped ERA5 archive (data/raw/era5_v1/<system>.nc)
  --source hrrr   HRRR wrfsfcf01 records read by byte range (build_eih_hrrr.fetch), at the 3 km cells of the nodes
Nodes are the 3 km geography nodes of build_nodes_geo_v1.py (county x HRRR cell: population share, elevation, canopy,
poorly drained share); an ERA5 source maps each node to its 0.25-degree cell and moves the temperature to the node's
elevation. Output: data/interim/panel_v1/hazard_v2/<source>_<system>.npz (X [C, 216, K] float16, names, fips,
missing_hours). usage: python build_hazard_v2.py --source era5 --tranche D [--threads 3]"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import hazard_v2 as HZ  # noqa: E402
from frame_common import OUT, PREFIX_H, HORIZON_H, RAW  # noqa: E402

T = PREFIX_H + HORIZON_H
DST = OUT / "hazard_v2"
HRRR_KEYS = {"GUST:surface:1 hour fcst": "g", "TMP:2 m above ground:1 hour fcst": "t2m",
             "APCP:surface:0-1 hour acc fcst": "p", "FRZR:surface:0-1 hour acc fcst": "frz",
             "CRAIN:surface:1 hour fcst": "crain", "CICEP:surface:1 hour fcst": "cicep",
             "CSNOW:surface:1 hour fcst": "csnow", "VUCSH:0-6000 m above ground:1 hour fcst": "vush",
             "VVCSH:0-6000 m above ground:1 hour fcst": "vvsh", "MXUPHL:5000-2000 m above ground:0-1 hour max fcst": "uh",
             "REFC:entire atmosphere:1 hour fcst": "refc", "LTNG:entire atmosphere:1 hour fcst": "ltng",
             "TMP:850 mb:1 hour fcst": "t850", "MSTAV:0 m underground:1 hour fcst": "mstav"}


def g98_at(lat, lon):
    z = np.load(OUT / "era5_fg10_clim_2008_2017.npz")
    i = np.clip(np.rint((50.0 - lat) / 0.25).astype(int), 0, z["p98"].shape[0] - 1)
    j = np.clip(np.rint((lon + 125.0) / 0.25).astype(int), 0, z["p98"].shape[1] - 1)
    return z["p98"][i, j]


def era5_fields(system, lat_c, lon_c):
    """Hourly ERA5 fields at the given cell centres from the cropped archive."""
    import xarray as xr
    ds = xr.open_dataset(RAW / "era5_v1" / f"{system}.nc", engine="h5netcdf")
    la, lo = ds.latitude.values, ds.longitude.values
    ii = np.array([int(np.argmin(np.abs(la - x))) for x in lat_c]); jj = np.array([int(np.argmin(np.abs(lo - x))) for x in lon_c])
    get = lambda v: ds[v].values.astype(np.float32)[:, ii, jj]      # noqa: E731
    tp = np.maximum(get("tp"), 0) * 1000.0; sf = np.maximum(get("sf"), 0) * 1000.0
    pt = np.rint(get("ptype"))
    f = dict(g=get("fg10"), t2m=get("t2m") - 273.15,
             rain=np.where(np.isin(pt, (1, 7)), tp, 0.0), frz=np.where(np.isin(pt, (3, 12)), tp, 0.0),
             sleet=np.where(pt == 8, tp, 0.0), snow=sf, wet=np.clip(get("swvl1")[PREFIX_H] / 0.5, 0, 1), lai=get("lai_hv"))
    ds.close()
    return f


SLOW = {"vush", "vvsh", "t850"}          # slowly varying: read every 6 h and held (the data volume of a 3 km field)
ONCE = {"mstav"}                          # antecedent soil wetness: read at the origin hour only


def hrrr_fields(t0, rows, cols, threads):
    from build_eih_hrrr import fetch
    hours = pd.date_range(t0, periods=T, freq="h")

    def keys_at(i):
        return {k: v for k, v in HRRR_KEYS.items()
                if (v not in SLOW and v not in ONCE) or (v in SLOW and i % 6 == 0) or (v in ONCE and i == PREFIX_H)}

    with ThreadPoolExecutor(threads) as ex:
        res = list(ex.map(lambda i: fetch(hours[i], keys_at(i), rows, cols), range(T)))
    miss = [str(h) for h, r in zip(hours, res) if r is None]
    N = len(rows)
    z = {k: np.zeros((T, N), np.float32) for k in HRRR_KEYS.values()}
    for t, r in enumerate(res):
        if r is not None:
            for k, v in r.items():
                z[k][t] = v
    for k in SLOW:                          # hold the last 6-hourly value
        last = None
        for t in range(T):
            if t % 6 == 0 and res[t] is not None:
                last = z[k][t].copy()
            elif last is not None:
                z[k][t] = last
    p = np.maximum(z["p"], 0.0)
    f = dict(g=z["g"], t2m=z["t2m"] - 273.15, rain=p * (z["crain"] > 0.5), frz=np.maximum(z["frz"], 0.0),
             sleet=p * (z["cicep"] > 0.5), snow=p * (z["csnow"] > 0.5), wet=np.clip(z["mstav"][PREFIX_H] / 100.0, 0, 1),
             shear=np.hypot(z["vush"], z["vvsh"]), uh=np.maximum(z["uh"], 0.0), refc=np.maximum(z["refc"], 0.0),
             ltng=np.maximum(z["ltng"], 0.0), warm_nose=((z["t850"] > 273.15) & (z["t2m"] < 273.15)).astype(np.float32))
    return f, miss


LAPSE = 6.5e-3          # C per m


def era5_orography():
    import xarray as xr
    ds = xr.open_dataset(RAW / "era5_extra" / "era5_orography.nc")
    v = [x for x in ds.data_vars][0]
    z = ds[v].values.squeeze() / 9.80665
    la, lo = ds.latitude.values, ds.longitude.values
    return la, np.where(lo > 180, lo - 360, lo), z


def build(system, source, sy, cs, nodes, threads, oro):
    fips = cs[(cs.system == system) & cs.sampled].fips.tolist()
    nd = nodes[nodes.fips.isin(fips)].reset_index(drop=True)
    cidx = {f: i for i, f in enumerate(fips)}
    node_county = nd.fips.map(cidx).to_numpy()
    w = nd.w_pop.to_numpy(float); w = w / np.bincount(node_county, w, len(fips))[node_county]
    geo = dict(canopy=np.clip(nd.canopy.to_numpy(float) / 100.0, 0, 1), drain=np.clip(nd.wet.to_numpy(float), 0, 1))
    t0 = pd.Timestamp(sy.at[system, "window_start"])
    lat, lon = nd.lat.to_numpy(), nd.lon.to_numpy()
    if source == "era5":
        ci = np.rint((50.0 - lat) / 0.25).astype(int); cj = np.rint((lon + 125.0) / 0.25).astype(int)
        cells, cell_of_node = np.unique(np.stack([ci, cj], 1), axis=0, return_inverse=True)
        cell_of_node = cell_of_node.ravel()
        lat_c, lon_c = 50.0 - 0.25 * cells[:, 0], -125.0 + 0.25 * cells[:, 1]
        fc = era5_fields(system, lat_c, lon_c); miss = []
        f = {k: (v[:, cell_of_node] if v.ndim == 2 else v[cell_of_node]) for k, v in fc.items()}
        ola, olo, oz = oro
        zc = oz[np.array([int(np.argmin(np.abs(ola - x))) for x in lat_c]),
                np.array([int(np.argmin(np.abs(olo - x))) for x in lon_c])]
        f["t2m"] = f["t2m"] - LAPSE * (nd.z.to_numpy(float) - zc[cell_of_node])[None, :]
    else:
        cells, cell_of_node = np.unique(nd[["hr", "hc"]].to_numpy(), axis=0, return_inverse=True)
        cell_of_node = cell_of_node.ravel()
        fc, miss = hrrr_fields(t0, cells[:, 0], cells[:, 1], threads)
        f = {k: (v[:, cell_of_node] if v.ndim == 2 else v[cell_of_node]) for k, v in fc.items()}
    d = HZ.derive(f, g98_at(lat, lon))
    X, names = HZ.aggregate(d, node_county, w, len(fips), geo)
    DST.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DST / f"{source}_{system}.npz", X=X.astype(np.float16), names=np.array(names), fips=np.array(fips),
                        missing_hours=np.array(miss))
    return len(fips), len(nd), len(cells), len(miss)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["era5", "hrrr"], required=True)
    ap.add_argument("--tranche", default="D")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--systems", nargs="*", default=None)
    a = ap.parse_args()
    assert a.tranche != "C", "the sealed tranche is built only by the confirmatory script"
    sy = pd.read_parquet(OUT / "systems.parquet").set_index("system")
    dr = pd.read_parquet(OUT / "draws.parquet")
    cs = pd.read_parquet(OUT / f"county_sample_{a.tranche}.parquet")
    nodes = pd.read_parquet(OUT / f"nodes_geo_{a.tranche}.parquet")
    oro = era5_orography()
    systems = a.systems or sorted(dr[dr.tranche == a.tranche].system)
    for s in systems:
        if (DST / f"{a.source}_{s}.npz").exists():
            continue
        t = time.time()
        n = build(s, a.source, sy, cs, nodes, a.threads, oro)
        print(s, a.source, "counties %d nodes %d cells %d missing hours %d" % n, f"{time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    main()
