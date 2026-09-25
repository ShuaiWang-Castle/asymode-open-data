"""Variant `hrrr_coarse` of the exposure-integrated hazard features: HRRR weather averaged onto the ERA5 0.25-degree
cells, then the `pop` construction of build_eih.py (nodes = the county's ERA5 cells, population weights, cell-level
modulators, no downscaling). With it the contrast hrrr - pop splits into
  hrrr_coarse - pop    the weather source and definitions (HRRR forecast hour 1 vs ERA5 reanalysis) at one resolution
  hrrr - hrrr_coarse   the resolution (3 km cells and HRRR-terrain downscaling) from one source
(suggested by the formal contributor). HRRR cells are assigned to the ERA5 cell containing their centre (inverse
Lambert conformal projection of the HRRR grid); the coarse value is the plain mean over those HRRR cells.
Missing HRRR hours (neither mirror has the file) leave the instantaneous (tau = 0) features NaN at those hours; a
window missing more than 5% of its hours marks its units unit_ok = False (PREREG_W2 amendment 4).
Output: data/interim/geo_weather/eih_<tag>hrrr_coarse.npz (phi, names, fips, event, missing_hours, unit_ok).
usage: python build_eih_hrrr_coarse.py --feat data/interim/geo_weather/features_w1.npz \
       --events-file experiments/geo_weather_20260924/selected_events_winter_build.json --tag w1_
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import build_eih_hrrr as BH  # noqa: E402
import build_nodes_hrrr as NH  # noqa: E402
from build_eih import MOD, NAMES, ORIGIN, PSI, T, TAUS, filt, modulators, psi  # noqa: E402

OUT = ROOT / "data" / "interim" / "geo_weather"
NLAT, NLON = 105, 237


def hrrr_to_era5() -> np.ndarray:
    """ERA5 cell id (ci * NLON + cj) of every HRRR cell centre, -1 outside the panel box."""
    rows, cols = np.mgrid[0:1059, 0:1799]
    x = NH.X0 + cols * 3000.0; y = NH.Y0 + rows * 3000.0
    rho0 = NH._rho(38.5)
    rho = np.sign(NH.N_L) * np.sqrt(x ** 2 + (rho0 - y) ** 2)
    theta = np.arctan2(x, rho0 - y)
    lon = np.rad2deg(theta / NH.N_L) - 97.5
    lat = np.rad2deg(2 * np.arctan((NH.R_E * NH.F_L / rho) ** (1 / NH.N_L))) - 90.0
    ci = np.rint((50.0 - lat) / 0.25).astype(int); cj = np.rint((lon + 125.0) / 0.25).astype(int)
    ok = (ci >= 0) & (ci < NLAT) & (cj >= 0) & (cj < NLON)
    return np.where(ok, ci * NLON + cj, -1)


def fetch_full(valid: pd.Timestamp, keys: dict):
    """Full HRRR fields (1059 x 1799) for one valid hour (same records and mirrors as build_eih_hrrr.fetch)."""
    rows, cols = np.mgrid[0:1059, 0:1799]
    return BH.fetch(valid, keys, rows, cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", required=True)
    ap.add_argument("--events-file", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    F = np.load(ROOT / a.feat)
    fips_u, ev_u = F["fips"].astype(str), F["event"].astype(str)
    lab = hrrr_to_era5().ravel()
    nodes = pd.read_parquet(OUT / "nodes_cs.parquet")
    by = {f: d.reset_index(drop=True) for f, d in nodes.groupby("fips")}
    sel = {e["event"]: e for e in json.loads((ROOT / a.events_file).read_text())["events"]}
    phi = np.zeros((len(fips_u), T - ORIGIN, len(NAMES)), np.float16)
    unit_ok = np.ones(len(fips_u), bool)
    inst0 = [i for i, n in enumerate(NAMES) if n.endswith("@0")]
    missing = {}
    for ev in sorted(set(ev_u)):
        t0e = time.time()
        units = np.where(ev_u == ev)[0]
        month = int(ev[5:7]); leaf = float(5 <= month <= 10)
        # the `pop` nodes: the county's ERA5 cells, population weights, cell-level (population-weighted) attributes
        cells, per_unit = [], []
        for j, u in enumerate(units):
            nd = by[fips_u[u]]
            agg = nd.assign(wcan=nd.canopy * nd.w_pop, wwet=nd.wet * nd.w_pop).groupby(["i", "j"], as_index=False).agg(
                w=("w_pop", "sum"), wcan=("wcan", "sum"), wwet=("wwet", "sum"))
            agg["canopy"] = agg.wcan / agg.w; agg["wet"] = agg.wwet / agg.w; agg["w"] = agg.w / agg.w.sum()
            agg["unit"] = j
            per_unit.append(agg)
        nd = pd.concat(per_unit, ignore_index=True)
        cid = (nd.i * NLON + nd.j).to_numpy()
        need = np.unique(cid)
        pos = {c: k for k, c in enumerate(need)}
        sel_h = np.isin(lab, need)                                    # HRRR cells inside the needed ERA5 cells
        lab_s = np.array([pos[c] for c in lab[sel_h]])
        cnt = np.bincount(lab_s, minlength=len(need)).astype(float)
        assert (cnt > 0).all(), "an ERA5 cell without HRRR cells"
        idx = np.array([pos[c] for c in cid])
        from scipy.sparse import csr_matrix
        S = csr_matrix((nd.w.to_numpy(), (nd.unit.to_numpy(), np.arange(len(nd)))), shape=(len(units), len(nd)))
        mods = modulators(nd, leaf)
        times = pd.date_range(pd.Timestamp(sel[ev]["window_start_utc"]), periods=T, freq="h")

        def coarse(tt):
            g = fetch_full(tt, BH.KEYS)
            if g is None:
                return None
            return {k: (np.bincount(lab_s, v.ravel()[sel_h], minlength=len(need)) / cnt)[idx] for k, v in g.items()}
        with ThreadPoolExecutor(a.workers) as ex:
            got = list(ex.map(coarse, times))
        missing[ev] = [str(t) for t, g in zip(times, got) if g is None]
        inst = np.zeros((T, len(units), len(MOD), len(PSI)), np.float64)
        for s, g in enumerate(got):
            if g is None:
                continue
            tc, tdc = g["t"] - 273.15, g["td"] - 273.15
            a_, b_ = 17.625, 243.04
            rh = 100.0 * np.exp(a_ * tdc / (b_ + tdc) - a_ * tc / (b_ + tc))
            ps = psi(tc[None], tdc[None], rh[None], g["p"][None], g["g"][None], g["cape"][None], np.zeros(len(nd)), month)[0]
            inst[s] = (S @ (mods[:, :, None] * ps[:, None, :]).reshape(len(nd), -1)).reshape(len(units), len(MOD), len(PSI))
        x = inst.reshape(T, len(units), -1)
        miss_s = [s for s, g in enumerate(got) if g is None]
        for j, u in enumerate(units):
            phi[u] = np.concatenate([filt(x[:, j], tau)[ORIGIN:] for tau in TAUS], -1).astype(np.float16)
            for s in miss_s:                  # a missing HRRR hour: the instantaneous features are NaN, not zero
                if s >= ORIGIN:
                    phi[u, s - ORIGIN, inst0] = np.nan
        if len(miss_s) > 0.05 * T:            # PREREG_W2 amendment 4 coverage rule: > 5% missing hours drops the event
            unit_ok[units] = False
        print(ev, len(units), "units", len(nd), "cells; missing hours", len(missing[ev]), f"{time.time() - t0e:.0f} s", flush=True)
    np.savez(OUT / f"eih_{a.tag}hrrr_coarse.npz", phi=phi, names=np.array(NAMES), fips=fips_u, event=ev_u,
             missing_hours=np.array(json.dumps(missing)), unit_ok=unit_ok)
    print("saved", phi.shape, "units dropped by the coverage rule:", int((~unit_ok).sum()))
    print("HRRR_COARSE_BUILD_DONE", flush=True)


if __name__ == "__main__":
    main()
