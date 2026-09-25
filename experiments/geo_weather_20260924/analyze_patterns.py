"""Regularities of the public outage panel that bear on model design (notes/DATA_PATTERNS.md). No model is trained.

Inputs (read only; nothing is downloaded or modified):
  data/interim/open_gcrk/features_e3r2.npz    6,122 county-events: y [U,144] (forecast hours 72..215), m, y_full,
                                              obs_full [U,216], cust, fips, event, y0; host inputs xu [U,216,42]
                                              (area-weighted ERA5 channels + summaries; streamed, selected channels);
                                              xr hour 0: the base's six static county-context inputs
  runs/open_gcrk_20260919/e3r2/main/seed0/fold0{1..5}/W+Cin/outer.npz   out-of-fold base predictions P (idx, P)
  data/interim/geo_weather/eih_pop.npz, eih_area.npz   exposure-integrated hazards phi [U,144,160] (memory-mapped)
  data/interim/geo_weather/train_mask_e3.npz  EAGLE-I artefact flags (dip, spike, plateau, over) [U,216]
  data/interim/geo_weather/nodes_cs.parquet   population weights, canopy, poorly drained share per node
  data/interim/county_statics.parquet         2020 population (USDA ERS) for customers per person
Output: results/data_patterns/patterns.json (every number quoted in the note, by section q1..q6).

Definitions used throughout
  residual      r = y - P on observed forecast cells (m = 1); r > 0 is under-prediction.
  SSE           sum of r^2 over observed forecast cells (pooled RMSE = sqrt(SSE / cells)).
  peak          t* = first hour of the maximum of the observed curve over the whole 216-h window (prefix
                included), y* = that maximum; forecast-window peak y*_f = maximum over observed hours 72..215.
  phases        forecast hour t is 'rise' if t < t*, 'peak' if t = t*, 'decay' if t > t*; units with y* = 0 are
                'no outage'. Sensitivity: t* taken inside the forecast window.
  active        y* >= 0.01 (at least 1% of customers out at the peak).
  season        May-Oct events (leaf-on in build_eih.py: 2021-08-11, 2022-06-08, 2022-06-17, 2024-05-08,
                2024-06-26) versus Nov-Apr events (the other seven).
  rank partial  variables rank-transformed (pooled ranks / n), event fixed effects removed by demeaning, then
                residualised on the rank-transformed controls by least squares; partial correlation = correlation
                of the two residual vectors, partial R2 = its square (rank scale). 'raw dR2' = increase of R2 of the
                untransformed residual target when the ranked feature is added (squared-error scale).
  cluster CI    county-cluster bootstrap (B = 1000) of the residualised cross-products (residualisation not
                refitted inside the bootstrap); hour level: unit-cluster bootstrap of the same kind.

Run (one process, single thread, about 1.5-2 minutes, peak RSS about 0.7 GB):
    OMP_NUM_THREADS=1 .venv/bin/python experiments/geo_weather_20260924/analyze_patterns.py
    ... --sections q1 q3 --out /some/other.json     (subset, for development)
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import struct
import time
import zipfile

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "1")

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import sparse, stats  # noqa: E402
from scipy.signal import find_peaks  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FEAT = ROOT / "data" / "interim" / "open_gcrk" / "features_e3r2.npz"
GW = ROOT / "data" / "interim" / "geo_weather"
RUNS = ROOT / "runs" / "open_gcrk_20260919" / "e3r2" / "main" / "seed0"
STATICS = ROOT / "data" / "interim" / "county_statics.parquet"
OUT = HERE / "results" / "data_patterns" / "patterns.json"
ORIGIN, T, H = 72, 216, 144
SEED = 20260925
B_BOOT = 1000
ACTIVE = 0.01
WARM = ("2021-08-11", "2022-06-08", "2022-06-17", "2024-05-08", "2024-06-26")   # May-Oct = leaf-on
HOST = ["gust", "precip", "t2m_c", "snowfall", "soil_moisture", "wind_speed", "cape", "gust_cell_max",
        "gust_exceed15_share", "gust_excess_energy_sum72", "gust_excess_energy_sum6", "gust_max6", "gust_max24",
        "gust_max72"]
CTX = ["log_cust", "rucc", "log_pop_density", "coop_share", "log1p_n_utilities", "log1p_saidi"]   # the base's Cin
STATE = {"01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT", "10": "DE", "11": "DC",
         "12": "FL", "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY",
         "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS", "29": "MO", "30": "MT",
         "31": "NE", "32": "NV", "33": "NH", "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH",
         "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT",
         "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI", "56": "WY"}


# ----------------------------------------------------------------------------------------------------------- I/O
def stream_channels(path: Path, key: str, idx: list[int], chunk: int = 256) -> np.ndarray:
    """Selected last-axis channels of a (compressed) .npy inside an .npz, decompressed in unit chunks."""
    with zipfile.ZipFile(path) as z, z.open(key + ".npy") as f:
        v = np.lib.format.read_magic(f)
        shape, fortran, dtype = np.lib.format._read_array_header(f, v)
        assert not fortran
        per = int(np.prod(shape[1:])) * dtype.itemsize
        out = np.empty(shape[:-1] + (len(idx),), dtype)
        u = 0
        while u < shape[0]:
            n = min(chunk, shape[0] - u)
            out[u:u + n] = np.frombuffer(f.read(n * per), dtype).reshape((n,) + shape[1:])[..., idx]
            u += n
    return out


def npz_memmap(path: Path, key: str) -> np.memmap:
    """Memory map of an uncompressed (stored) .npy member of an .npz."""
    with zipfile.ZipFile(path) as z:
        info = z.getinfo(key + ".npy")
        assert info.compress_type == 0, f"{key} in {path.name} is compressed"
    with open(path, "rb") as fh:
        fh.seek(info.header_offset)
        lh = fh.read(30)
        assert lh[:4] == b"PK\x03\x04"
        n_name, n_extra = struct.unpack("<HH", lh[26:30])
        fh.seek(info.header_offset + 30 + n_name + n_extra)
        v = np.lib.format.read_magic(fh)
        shape, fortran, dtype = np.lib.format._read_array_header(fh, v)
        off = fh.tell()
    return np.memmap(path, dtype=dtype, mode="r", offset=off, shape=shape, order="F" if fortran else "C")


def load() -> dict:
    Z = np.load(FEAT)
    D = {k: Z[k] for k in ("y", "m", "y_full", "obs_full", "cust", "y0")}
    D["y"] = D["y"].astype(np.float64)
    D["m"] = D["m"] > 0
    D["event"], D["fips"] = Z["event"].astype(str), Z["fips"].astype(str)
    names = [str(s) for s in Z["damage_features"]]
    U = len(D["y"])
    X = stream_channels(FEAT, "xu", [names.index(c) for c in HOST])
    D["X"] = {c: X[..., j] for j, c in enumerate(HOST)}                                   # views [U, 216]
    rn = [str(s) for s in Z["recovery_features"]]
    D["ctx"] = stream_channels(FEAT, "xr", [rn.index(c) for c in CTX])[:, 0, :].astype(np.float64)   # static
    P = np.full((U, H), np.nan)
    for f in range(1, 6):
        z = np.load(RUNS / f"fold{f:02d}" / "W+Cin" / "outer.npz")
        assert np.isnan(P[z["idx"]]).all(), "unit in two outer folds"
        P[z["idx"]] = z["P"]
    assert np.isfinite(P).all(), "unit without an out-of-fold prediction"
    D["P"] = P
    D["r"] = np.where(D["m"], D["y"] - P, 0.0)
    D["state"] = np.array([STATE.get(f[:2], f[:2]) for f in D["fips"]])
    yv = np.where(D["obs_full"], D["y_full"], -np.inf)
    D["tstar"], D["ystar"] = yv.argmax(1), yv.max(1)
    ym = np.where(D["m"], D["y"], -np.inf)
    D["ystar_f"], D["tstar_f"] = ym.max(1), ym.argmax(1) + ORIGIN
    D["pstar_f"] = np.where(D["m"], P, -np.inf).max(1)
    D["season"] = np.where(np.isin(D["event"], WARM), "May-Oct", "Nov-Apr")
    D["events"] = sorted(set(D["event"]))
    return D


# ------------------------------------------------------------------------------------------------------ helpers
def g6(x):
    """JSON-friendly rounding (6 significant digits) of scalars, arrays, dicts and lists."""
    if isinstance(x, dict):
        return {str(k): g6(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [g6(v) for v in x]
    if isinstance(x, np.ndarray):
        return [g6(v) for v in x.tolist()]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return None if not np.isfinite(x) else float(f"{x:.6g}")
    return x


def rank01(a: np.ndarray) -> np.ndarray:
    """Average ranks / n along axis 0 (column-wise for 2-D)."""
    a = np.asarray(a, np.float64)
    if a.ndim == 1:
        return stats.rankdata(a) / len(a)
    return np.column_stack([stats.rankdata(a[:, j]) / len(a) for j in range(a.shape[1])])


def group_demean(A: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Subtract group means (g integer codes) from the columns of A."""
    A = np.asarray(A, np.float64)
    two = A.ndim == 1
    A2 = A[:, None] if two else A
    codes, g = np.unique(g, return_inverse=True)
    S = sparse.csr_matrix((np.ones(len(g)), (g, np.arange(len(g)))), shape=(len(codes), len(g)))
    mu = (S @ A2) / np.asarray(S.sum(1))
    out = A2 - mu[g]
    return out[:, 0] if two else out


def residualise(Y: np.ndarray, C: np.ndarray | None) -> np.ndarray:
    if C is None or C.shape[1] == 0:
        return Y
    beta, *_ = np.linalg.lstsq(C, Y, rcond=None)
    return Y - C @ beta


def partial(y: np.ndarray, Fm: np.ndarray, C: np.ndarray | None = None, groups: list | None = None,
            y_raw: np.ndarray | None = None) -> dict:
    """Partial correlations of y with each column of Fm given controls C and fixed effects (a list of group-code
    arrays, demeaned in turn). y, Fm, C arrive transformed (ranks). y_raw (optional, untransformed target) adds the
    squared-error-scale increase of R2 from each feature (Frisch-Waugh: (e_r . e_f)^2 / (|e_f|^2 TSS))."""
    Fm = Fm[:, None] if Fm.ndim == 1 else Fm
    k, nc = Fm.shape[1], (0 if C is None else C.shape[1])
    blocks = [y[:, None], Fm] + ([C] if nc else []) + ([y_raw[:, None]] if y_raw is not None else [])
    Z = np.column_stack(blocks).astype(np.float64)
    if groups:
        for _ in range(3 if len(groups) > 1 else 1):         # alternating projections for two-way effects
            for g in groups:
                Z = group_demean(Z, g)
    else:
        Z = Z - Z.mean(0)
    Zc = Z[:, 1 + k:1 + k + nc] if nc else None
    lhs = np.column_stack([Z[:, :1 + k]] + ([Z[:, -1:]] if y_raw is not None else []))
    R = residualise(lhs, Zc)
    ey, eF = R[:, 0], R[:, 1:1 + k]
    den = np.sqrt((eF ** 2).sum(0) * (ey ** 2).sum())
    rho = np.where(den > 0, (eF * ey[:, None]).sum(0) / np.where(den > 0, den, 1), np.nan)
    out = dict(rho=rho, ey=ey, eF=eF)
    if y_raw is not None:
        er, tss = R[:, -1], (Z[:, -1] ** 2).sum()
        ff = (eF ** 2).sum(0)
        out["raw_dR2"] = np.where(ff > 0, (eF * er[:, None]).sum(0) ** 2 / np.where(ff > 0, ff, 1) / tss, np.nan)
        out["raw_R2_controls"] = float(1 - (er ** 2).sum() / tss)
    return out


def cluster_ci(ey: np.ndarray, eF: np.ndarray, clusters: np.ndarray, B: int = B_BOOT, seed: int = SEED) -> np.ndarray:
    """95% cluster-bootstrap interval of corr(ey, eF_j) from per-cluster cross-products -> [k, 2]."""
    codes, c = np.unique(clusters, return_inverse=True)
    S = sparse.csr_matrix((np.ones(len(c)), (c, np.arange(len(c)))), shape=(len(codes), len(c)))
    syf, sff, syy = S @ (eF * ey[:, None]), S @ (eF ** 2), S @ (ey ** 2)
    rng = np.random.default_rng(seed)
    w = rng.multinomial(len(codes), np.full(len(codes), 1 / len(codes)), size=B).astype(np.float64)   # [B, C]
    rb = (w @ syf) / np.sqrt((w @ sff) * (w @ syy)[:, None])
    return np.quantile(rb, [0.025, 0.975], axis=0).T


def km_median(time_: np.ndarray, event: np.ndarray) -> float:
    """Kaplan-Meier median of right-censored durations (event = 1 observed, 0 censored)."""
    if len(time_) == 0:
        return float("nan")
    order = np.argsort(time_, kind="stable")
    t, e = time_[order], event[order]
    s, n = 1.0, len(t)
    for tt in np.unique(t):
        at = (t == tt)
        d = e[at].sum()
        if d:
            s *= 1 - d / n
            if s <= 0.5:
                return float(tt)
        n -= at.sum()
    return float("nan")      # median not reached


def share_top(v: np.ndarray, q: float) -> float:
    s = np.sort(v)[::-1]
    k = max(1, int(round(q * len(s))))
    return float(s[:k].sum() / s.sum())


def n_for(v: np.ndarray, frac: float) -> int:
    s = np.sort(v)[::-1]
    return int(np.searchsorted(np.cumsum(s) / s.sum(), frac) + 1)


# ------------------------------------------------------------------------------ Q1 where the squared error sits
def q1(D: dict) -> dict:
    y, m, P, r, ev, st, cust = D["y"], D["m"], D["P"], D["r"], D["event"], D["state"], D["cust"]
    se = r ** 2
    SSE, N = se.sum(), m.sum()
    y2 = np.where(m, y ** 2, 0.0)
    sse_u, y2_u, n_u = se.sum(1), y2.sum(1), m.sum(1)
    U = len(y)
    out = dict(units=U, counties=len(set(D["fips"])), events=len(D["events"]), states=len(set(st)),
               observed_forecast_cells=int(N), unobserved_share=float(1 - N / m.size),
               rmse=float(np.sqrt(SSE / N)), rmse_zero_forecast=float(np.sqrt(y2.sum() / N)),
               mse_skill_vs_zero=float(1 - SSE / y2.sum()),
               under_prediction_share_of_sse=float(se[r > 0].sum() / SSE))

    def block(mask_u: np.ndarray) -> dict:
        s = sse_u[mask_u].sum()
        return dict(units=int(mask_u.sum()), unit_share=float(mask_u.mean()), sse_share=float(s / SSE),
                    y2_share=float(y2_u[mask_u].sum() / y2.sum()),
                    rmse=float(np.sqrt(s / max(n_u[mask_u].sum(), 1))),
                    skill_vs_zero=float(1 - s / max(y2_u[mask_u].sum(), 1e-12)))

    out["by_event"] = {e: dict(block(ev == e), active_units=int(((ev == e) & (D["ystar"] >= ACTIVE)).sum()),
                               median_peak_active=float(np.median(D["ystar"][(ev == e) & (D["ystar"] >= ACTIVE)])))
                       for e in D["events"]}
    out["by_season"] = {s: block(D["season"] == s) for s in ("Nov-Apr", "May-Oct")}
    sts = pd.Series(sse_u).groupby(st).sum().sort_values(ascending=False)
    out["by_state_top10"] = {s: block(st == s) for s in sts.index[:10]}
    out["state_count"] = int(len(sts))
    out["states_for_50pct_sse"] = n_for(sts.to_numpy(), 0.5)
    es = pd.Series(sse_u).groupby([ev, st]).sum().sort_values(ascending=False)
    out["event_state_blocks"] = int(len(es))
    out["event_state_top10"] = {f"{a}|{b}": dict(block((ev == a) & (st == b))) for a, b in es.index[:10]}
    out["event_state_blocks_for_50pct_sse"] = n_for(es.to_numpy(), 0.5)
    out["top5_event_state_share"] = float(es.iloc[:5].sum() / SSE)
    # by lead (24-h blocks) and the single worst lead hours
    lead_sse = se.sum(0)
    lead_n = m.sum(0)
    out["by_lead_block"] = {f"{a + 1}-{a + 24}": dict(sse_share=float(lead_sse[a:a + 24].sum() / SSE),
                                                     rmse=float(np.sqrt(lead_sse[a:a + 24].sum() / lead_n[a:a + 24].sum())),
                                                     y2_share=float(y2.sum(0)[a:a + 24].sum() / y2.sum()))
                            for a in range(0, H, 24)}
    # by county size (customer quintiles over units)
    qs = np.quantile(cust, [0.2, 0.4, 0.6, 0.8])
    qi = np.digitize(cust, qs)
    out["by_cust_quintile"] = {f"Q{k + 1}": dict(block(qi == k), cust_min=float(cust[qi == k].min()),
                                                cust_max=float(cust[qi == k].max()),
                                                median_peak=float(np.median(D["ystar_f"][qi == k])))
                               for k in range(5)}
    # by forecast-window peak level
    pk = D["ystar_f"]
    bins = [(-1, 0), (0, 0.001), (0.001, 0.01), (0.01, 0.05), (0.05, 0.2), (0.2, 0.5), (0.5, 1.01)]
    out["by_peak_level"] = {("0" if lo < 0 else f"({lo},{hi if hi <= 1 else 1}]"): block((pk > lo) & (pk <= hi))
                            for lo, hi in bins}
    for name, sub in (("peak_gt_0p2", pk > 0.2), ("peak_lt_0p01", pk < 0.01)):
        out[name] = block(sub)
    # concentration over units, counties, and of the target itself
    sse_c = pd.Series(sse_u).groupby(D["fips"]).sum().to_numpy()
    out["concentration"] = dict(
        top1pct_units_sse_share=share_top(sse_u, 0.01), top5pct_units_sse_share=share_top(sse_u, 0.05),
        top10pct_units_sse_share=share_top(sse_u, 0.10), units_for_50pct_sse=n_for(sse_u, 0.5),
        units_for_80pct_sse=n_for(sse_u, 0.8), top1pct_units_y2_share=share_top(y2_u, 0.01),
        top5pct_units_y2_share=share_top(y2_u, 0.05), top1pct_counties_sse_share=share_top(sse_c, 0.01),
        top5pct_counties_sse_share=share_top(sse_c, 0.05),
        top1pct_units=int(round(0.01 * U)), top5pct_units=int(round(0.05 * U)),
        top1pct_units_median_peak=float(np.median(pk[np.argsort(sse_u)[::-1][:int(round(0.01 * U))]])),
        top1pct_units_median_cust=float(np.median(cust[np.argsort(sse_u)[::-1][:int(round(0.01 * U))]])),
        top1pct_units_events=pd.Series(ev[np.argsort(sse_u)[::-1][:int(round(0.01 * U))]]).value_counts().to_dict())
    # level vs shape: SSE_u = n_u rbar_u^2 + within-unit
    rbar = r.sum(1) / np.maximum(n_u, 1)
    out["level_share_of_sse"] = float((n_u * rbar ** 2).sum() / SSE)
    # phases relative to the observed peak of the whole window
    t = np.arange(ORIGIN, T)[None, :]

    def phases(tstar: np.ndarray, none: np.ndarray) -> dict:
        ph = {"rise": (t < tstar[:, None]), "peak": (t == tstar[:, None]), "decay": (t > tstar[:, None])}
        ph = {k: v & m & ~none[:, None] for k, v in ph.items()}
        ph["no outage"] = m & none[:, None]
        return {k: dict(cell_share=float(v.sum() / N), sse_share=float(se[v].sum() / SSE),
                        rmse=float(np.sqrt(se[v].sum() / max(v.sum(), 1))),
                        under_share_of_phase_sse=float(se[v & (r > 0)].sum() / max(se[v].sum(), 1e-12)),
                        mean_residual=float(r[v].mean()) if v.any() else None)
                for k, v in ph.items()}
    none = D["ystar"] <= 0
    out["phase_full_window_peak"] = phases(D["tstar"], none)
    out["phase_forecast_window_peak"] = phases(D["tstar_f"], D["ystar_f"] <= 0)
    out["units_peak_in_prefix_share_active"] = float((D["tstar"][D["ystar"] >= ACTIVE] < ORIGIN).mean())
    # hours relative to the observed peak
    rel = t - D["tstar"][:, None]
    edges = [(-999, -25), (-24, -13), (-12, -7), (-6, -1), (0, 0), (1, 6), (7, 24), (25, 72), (73, 999)]
    out["by_hours_from_peak"] = {f"[{a},{b}]": dict(sse_share=float(se[(rel >= a) & (rel <= b) & m & ~none[:, None]].sum() / SSE),
                                                    cell_share=float(((rel >= a) & (rel <= b) & m & ~none[:, None]).sum() / N),
                                                    under_share=float(se[(rel >= a) & (rel <= b) & m & ~none[:, None] & (r > 0)].sum()
                                                                      / max(se[(rel >= a) & (rel <= b) & m & ~none[:, None]].sum(), 1e-12)))
                                 for a, b in edges}
    near = (rel >= -24) & (rel <= 24) & m & ~none[:, None]
    out["within_24h_of_peak"] = dict(sse_share=float(se[near].sum() / SSE), cell_share=float(near.sum() / N))
    near6 = (rel >= -6) & (rel <= 6) & m & ~none[:, None]
    out["within_6h_of_peak"] = dict(sse_share=float(se[near6].sum() / SSE), cell_share=float(near6.sum() / N))
    # peak magnitude / timing of the base on active forecast windows
    act = D["ystar_f"] >= ACTIVE
    ratio = D["pstar_f"][act] / D["ystar_f"][act]
    tp = np.where(m, P, -np.inf).argmax(1) + ORIGIN
    dt = (tp - D["tstar_f"])[act]
    out["peak_errors_active_forecast_window"] = dict(
        units=int(act.sum()), median_pred_to_obs_peak=float(np.median(ratio)),
        share_pred_peak_below_half=float((ratio < 0.5).mean()), share_pred_peak_above_obs=float((ratio > 1).mean()),
        median_timing_error_h=float(np.median(dt)), median_abs_timing_error_h=float(np.median(np.abs(dt))),
        by_peak_level={f"({lo},{hi}]": dict(units=int(((D['ystar_f'][act] > lo) & (D['ystar_f'][act] <= hi)).sum()),
                                            median_ratio=float(np.median(ratio[(D['ystar_f'][act] > lo) & (D['ystar_f'][act] <= hi)])))
                       for lo, hi in [(0.01, 0.05), (0.05, 0.2), (0.2, 1.0)]})
    # zeros
    zero = m & (y == 0)
    out["zeros"] = dict(zero_cell_share=float(zero.sum() / N), sse_share_on_zero_cells=float(se[zero].sum() / SSE),
                        false_activity_share=float((P[zero] > 0.001).mean()),
                        units_zero_forecast_window=int((D["ystar_f"] <= 0).sum()))
    return out



# ------------------------------------------------------------------------ Q2 which hazard features the residual wants
PSI = ["gust_x10", "gust_x15", "gust_x20", "rain", "p_tw-3.0", "p_tw-1.5", "p_tw+0.0", "p_tw+1.5", "p_tw+3.0",
       "convective"]
MODS = ["one", "canopy", "canopy_leafon", "wet"]
TAUS_Q2 = (0, 12)
UNIT_CTRL = ["gust max (72-215)", "gust mean (72-215)", "gust_cell_max max", "gust_exceed15_share mean",
             "gust_excess_energy_sum72 max", "gust max prefix (0-71)"]
HOUR_CTRL = ["gust", "gust_max6", "gust_max24", "gust_cell_max", "gust_exceed15_share", "gust_excess_energy_sum6",
             "gust_max72"]


def eih_names() -> list[str]:
    return [str(s) for s in np.load(GW / "eih_pop.npz")["names"]]


def eih_unit_summaries(variant: str, idx: list[int], m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Window mean over observed forecast hours and window max of the selected EIH features -> [U, k] each."""
    phi = npz_memmap(GW / f"eih_{variant}.npz", "phi")
    U = phi.shape[0]
    mean, mx = np.empty((U, len(idx))), np.empty((U, len(idx)))
    for u0 in range(0, U, 400):
        a = np.asarray(phi[u0:u0 + 400], np.float32)[:, :, idx]
        w = m[u0:u0 + 400, :, None]
        mean[u0:u0 + 400] = (a * w).sum(1) / np.maximum(w.sum(1), 1)
        mx[u0:u0 + 400] = np.where(w, a, -np.inf).max(1)
    del phi
    return mean, mx


def unit_controls(D: dict) -> np.ndarray:
    X, f = D["X"], slice(ORIGIN, T)
    return np.column_stack([X["gust"][:, f].max(1), X["gust"][:, f].mean(1), X["gust_cell_max"][:, f].max(1),
                            X["gust_exceed15_share"][:, f].mean(1), X["gust_excess_energy_sum72"][:, f].max(1),
                            X["gust"][:, :ORIGIN].max(1)]).astype(np.float64)


def county_modulators(D: dict) -> dict:
    nd = pd.read_parquet(GW / "nodes_cs.parquet", columns=["fips", "w_pop", "w_area", "canopy", "wet"])
    nd = nd.assign(cp=nd.canopy * nd.w_pop, wp=nd.wet * nd.w_pop, ca=nd.canopy * nd.w_area, wa=nd.wet * nd.w_area)
    g = nd.groupby("fips")[["w_pop", "w_area", "cp", "wp", "ca", "wa"]].sum()
    c = dict(canopy_pop=g.cp / g.w_pop / 100.0, wet_pop=g.wp / g.w_pop, canopy_area=g.ca / g.w_area / 100.0,
             wet_area=g.wa / g.w_area)
    return {k: v.reindex(D["fips"]).to_numpy(np.float64) for k, v in c.items()}


def feature_counts(dct: dict) -> dict:
    """Counts of features by hazard family and by modulator, split by the sign of their partial."""
    c = {}
    for f, v in dct.items():
        psi, rest = f.split("*")
        fam = "rain_or_convective" if psi in ("rain", "convective") else ("gust" if psi.startswith("gust") else "wetbulb_band")
        for k in (f"{fam}|{'+' if v > 0 else '-'}", f"mod_{rest.split('@')[0]}|{'+' if v > 0 else '-'}"):
            c[k] = c.get(k, 0) + 1
    return c


def unit_block(target: np.ndarray, Fs: np.ndarray, C: np.ndarray, D: dict, sub: np.ndarray) -> dict:
    """All rank statistics of one target against the feature columns on the units `sub`."""
    ev, fips = D["event"][sub], D["fips"][sub]
    Rt, RF, RC = rank01(target[sub]), rank01(Fs[sub]), rank01(C[sub])
    sp = partial(Rt, RF)["rho"]
    within = partial(Rt, RF, None, [ev])["rho"]
    nofe = partial(Rt, RF, RC, None)["rho"]
    main = partial(Rt, RF, RC, [ev], y_raw=target[sub])
    return dict(n=int(sub.sum()), spearman=sp, within_event=within, partial_gust=nofe, partial_gust_event=main["rho"],
                raw_dR2=main["raw_dR2"], raw_R2_controls=main["raw_R2_controls"], _ey=main["ey"], _eF=main["eF"],
                _fips=fips, _ev=ev, _Rt=Rt, _RF=RF, _RC=RC)


def q2(D: dict) -> dict:
    names = eih_names()
    sel = [j for j, n in enumerate(names) if int(n.split("@")[1]) in TAUS_Q2]
    fn = [names[j] for j in sel]
    m, U = D["m"], len(D["y"])
    mean_pop, max_pop = eih_unit_summaries("pop", sel, m)
    C = unit_controls(D)
    n_u = m.sum(1)
    rbar = D["r"].sum(1) / np.maximum(n_u, 1)
    dpk = D["ystar_f"] - D["pstar_f"]
    pbar = np.where(m, D["P"], 0).sum(1) / np.maximum(n_u, 1)
    C_plus = np.column_stack([C, pbar, np.log(D["cust"])])
    allu = np.ones(U, bool)
    act = (D["ystar_f"] >= ACTIVE) | (D["pstar_f"] >= ACTIVE)
    targets = {"window_mean_residual": (rbar, mean_pop), "peak_residual": (dpk, max_pop)}
    out = dict(features=fn, controls=UNIT_CTRL, active_units=int(act.sum()),
               target_summary=dict(window_mean_residual=dict(median=float(np.median(rbar)), mean=float(rbar.mean()),
                                                             share_positive=float((rbar > 0).mean())),
                                   peak_residual=dict(median=float(np.median(dpk)), mean=float(dpk.mean()),
                                                      share_positive=float((dpk > 0).mean()))))
    area_cache = None
    for tname, (tv, Fs) in targets.items():
        blk = unit_block(tv, Fs, C, D, allu)
        blk_act = unit_block(tv, Fs, C, D, act)
        blk_plus = partial(blk["_Rt"], blk["_RF"], rank01(C_plus), [D["event"]])["rho"]
        table = {fn[j]: dict(spearman=blk["spearman"][j], within_event=blk["within_event"][j],
                             partial_gust=blk["partial_gust"][j], partial_gust_event=blk["partial_gust_event"][j],
                             partial_R2=blk["partial_gust_event"][j] ** 2, raw_dR2=blk["raw_dR2"][j],
                             partial_gust_event_plusP_cust=blk_plus[j], partial_active=blk_act["partial_gust_event"][j])
                 for j in range(len(fn))}
        order_sp = np.argsort(-np.abs(np.nan_to_num(blk["spearman"])))
        order_pr = np.argsort(-np.abs(np.nan_to_num(blk["partial_gust_event"])))
        top = order_pr[:10]
        ci = cluster_ci(blk["_ey"], blk["_eF"][:, top], blk["_fips"])
        # per-event sign agreement of the partial (controls, within one event)
        same = np.zeros(len(top), int)
        for e in D["events"]:
            i = D["event"] == e
            rho_e = partial(rank01(tv[i]), rank01(Fs[i][:, top]), rank01(C[i]))["rho"]
            same += (np.sign(rho_e) == np.sign(blk["partial_gust_event"][top])).astype(int)
        if area_cache is None:
            area_cache = eih_unit_summaries("area", sel, m)
        Fa = area_cache[0] if tname == "window_mean_residual" else area_cache[1]
        blk_area = partial(blk["_Rt"], rank01(Fa[:, top]), blk["_RC"], [D["event"]])["rho"]
        # pop minus area, given the area feature itself (is the exposure weighting what carries it?)
        pa = [partial(blk["_Rt"], rank01(Fs[:, j]), np.column_stack([blk["_RC"], rank01(Fa[:, j])]), [D["event"]])["rho"][0]
              for j in top]
        out[tname] = dict(
            n=blk["n"], raw_R2_controls=blk["raw_R2_controls"],
            top10_by_spearman=[fn[j] for j in order_sp[:10]],
            top10_by_partial=[dict(feature=fn[j], spearman=blk["spearman"][j], partial_gust_event=blk["partial_gust_event"][j],
                                   partial_R2=blk["partial_gust_event"][j] ** 2, ci95=ci[k], raw_dR2=blk["raw_dR2"][j],
                                   events_same_sign=int(same[k]), partial_plusP_cust=blk_plus[j],
                                   partial_active=blk_act["partial_gust_event"][j], partial_area_variant=blk_area[k],
                                   partial_pop_given_area=pa[k])
                              for k, j in enumerate(top)],
            max_abs_partial=float(np.nanmax(np.abs(blk["partial_gust_event"]))),
            max_raw_dR2=float(np.nanmax(blk["raw_dR2"])),
            n_features_abs_partial_ge_0p05=int((np.abs(blk["partial_gust_event"]) >= 0.05).sum()),
            features_abs_partial_ge_0p05={fn[j]: blk["partial_gust_event"][j] for j in order_pr
                                          if abs(blk["partial_gust_event"][j]) >= 0.05},
            features_abs_partial_ge_0p05_counts=feature_counts({fn[j]: blk["partial_gust_event"][j] for j in order_pr
                                                                if abs(blk["partial_gust_event"][j]) >= 0.05}),
            all_features=table)
        # modulators: feature(psi, mod, tau) beyond feature(psi, one, tau) and the host gust, within event
        mods = {}
        Rt, RC = blk["_Rt"], blk["_RC"]
        for tau in TAUS_Q2:
            for b in PSI:
                j1 = fn.index(f"{b}*one@{tau}")
                for a_ in ("canopy", "canopy_leafon", "wet"):
                    ja = fn.index(f"{b}*{a_}@{tau}")
                    rec = {}
                    for sname, subm in (("all", allu), ("Nov-Apr", D["season"] == "Nov-Apr"), ("May-Oct", D["season"] == "May-Oct")):
                        if a_ == "canopy_leafon" and sname != "May-Oct":
                            continue          # zero in Nov-Apr events: identified only from May-Oct events
                        Rt_s = rank01(tv[subm]); RFa = rank01(Fs[subm][:, ja]); RF1 = rank01(Fs[subm][:, j1])
                        pr = partial(Rt_s, RFa, np.column_stack([rank01(C[subm]), RF1]), [D["event"][subm]])
                        rec[sname] = dict(rho=pr["rho"][0], ci95=cluster_ci(pr["ey"], pr["eF"], D["fips"][subm])[0])
                    mods[f"{b}*{a_}@{tau}"] = rec
        out[tname]["modulators_given_unmodulated"] = mods
    # physical check on the observed peak (not the residual): canopy / wet share at equal host wind
    cm = county_modulators(D)
    phys = {}
    for tname, tv in (("observed_peak", D["ystar_f"]), ("window_mean_residual", rbar), ("peak_residual", dpk)):
        rec = {}
        for mname in ("canopy_pop", "wet_pop", "canopy_area"):
            rr = {}
            for sname, subm in (("all", allu), ("Nov-Apr", D["season"] == "Nov-Apr"), ("May-Oct", D["season"] == "May-Oct")):
                pr = partial(rank01(tv[subm]), rank01(cm[mname][subm]), rank01(C[subm]), [D["event"][subm]])
                rr[sname] = dict(rho=pr["rho"][0], ci95=cluster_ci(pr["ey"], pr["eF"], D["fips"][subm])[0],
                                 n=int(subm.sum()))
            rec[mname] = rr
        phys[tname] = rec
    # binned: within event x host-gust tercile, observed peak in the top vs bottom canopy tercile
    ratios = []
    gmax = C[:, 0]
    for e in D["events"]:
        i = np.where(D["event"] == e)[0]
        gt = np.digitize(gmax[i], np.quantile(gmax[i], [1 / 3, 2 / 3]))
        for k in range(3):
            j = i[gt == k]
            ct = np.digitize(cm["canopy_pop"][j], np.quantile(cm["canopy_pop"][j], [1 / 3, 2 / 3]))
            lo, hi = np.median(D["ystar_f"][j[ct == 0]]), np.median(D["ystar_f"][j[ct == 2]])
            ratios.append(hi / lo if lo > 0 else np.nan)
    ratios = np.array(ratios)
    phys["binned_canopy_top_vs_bottom_tercile"] = dict(
        cells=int(np.isfinite(ratios).sum()), median_ratio=float(np.nanmedian(ratios)),
        share_ratio_gt1=float(np.mean(ratios[np.isfinite(ratios)] > 1)))
    phys["corr_canopy_pop_vs_area_units"] = float(stats.spearmanr(cm["canopy_pop"], cm["canopy_area"])[0])
    # is canopy a proxy for the county context the base already reads (customers, rurality, density, co-op share,
    # utilities, reliability)? add those six as controls
    Cctx = np.column_stack([C, D["ctx"]])
    phys["canopy_pop_given_gust_and_context"] = {}
    for tname, tv in (("observed_peak", D["ystar_f"]), ("peak_residual", dpk), ("window_mean_residual", rbar)):
        pr = partial(rank01(tv), rank01(cm["canopy_pop"]), rank01(Cctx), [D["event"]])
        phys["canopy_pop_given_gust_and_context"][tname] = dict(rho=pr["rho"][0],
                                                                ci95=cluster_ci(pr["ey"], pr["eF"], D["fips"])[0])
    # gating: canopy effect within event-specific terciles of the host gust maximum
    gt = np.zeros(U, int)
    for e in D["events"]:
        i = D["event"] == e
        gt[i] = np.digitize(gmax[i], np.quantile(gmax[i], [1 / 3, 2 / 3]))
    phys["canopy_pop_by_gust_tercile"] = {}
    for k in range(3):
        sub = gt == k
        rec = dict(median_gust_max=float(np.median(gmax[sub])))
        for tname, tv in (("observed_peak", D["ystar_f"]), ("peak_residual", dpk)):
            pr = partial(rank01(tv[sub]), rank01(cm["canopy_pop"][sub]), rank01(C[sub]), [D["event"][sub]])
            rec[tname] = dict(rho=pr["rho"][0], ci95=cluster_ci(pr["ey"], pr["eF"], D["fips"][sub])[0])
        phys["canopy_pop_by_gust_tercile"][f"T{k + 1}"] = rec
    out["canopy_wet_physical_check"] = phys
    out["hour_level"] = q2_hour(D, fn, sel)
    return out


def q2_hour(D: dict, fn: list[str], sel: list[int], n_units: int = 1500, chunk: int = 20) -> dict:
    """County-hour level on a seeded sample of units: pooled Spearman, and the within-unit partial (ranks; unit and
    lead fixed effects; host hourly gust controls). Features processed in chunks (exact, column-wise)."""
    rng = np.random.default_rng(SEED)
    U = len(D["y"])
    us = np.sort(rng.choice(U, n_units, replace=False))
    mm = D["m"][us]
    uu, ll = np.nonzero(mm)                           # cells of the sample
    r = D["r"][us][uu, ll]
    unit_code, lead_code = uu, ll
    ctrl = np.column_stack([D["X"][c][us][:, ORIGIN:][uu, ll] for c in HOUR_CTRL]).astype(np.float64)
    RC, Rr = rank01(ctrl), rank01(r)
    phi = npz_memmap(GW / "eih_pop.npz", "phi")
    Fall = np.asarray(phi[us], np.float32)[:, :, sel][uu, ll]         # [cells, k] float32
    del phi
    sp, pr, dr2 = np.empty(len(sel)), np.empty(len(sel)), np.empty(len(sel))
    ey_keep, eF_keep = None, np.empty((len(uu), len(sel)), np.float32)
    for c0 in range(0, len(sel), chunk):
        RF = rank01(Fall[:, c0:c0 + chunk])
        sp[c0:c0 + chunk] = partial(Rr, RF)["rho"]
        res = partial(Rr, RF, RC, [unit_code, lead_code], y_raw=r)
        pr[c0:c0 + chunk], dr2[c0:c0 + chunk] = res["rho"], res["raw_dR2"]
        ey_keep = res["ey"]
        eF_keep[:, c0:c0 + chunk] = res["eF"]
    top = np.argsort(-np.abs(np.nan_to_num(pr)))[:10]
    ci = cluster_ci(ey_keep, eF_keep[:, top].astype(np.float64), us[uu])
    return dict(units=n_units, cells=int(len(uu)), controls=HOUR_CTRL,
                top10_by_spearman=[dict(feature=fn[j], spearman=sp[j]) for j in np.argsort(-np.abs(sp))[:10]],
                top10_by_partial=[dict(feature=fn[j], spearman=sp[j], partial_within_unit=pr[j], partial_R2=pr[j] ** 2,
                                       ci95=ci[k], raw_dR2=dr2[j]) for k, j in enumerate(top)],
                max_abs_partial=float(np.nanmax(np.abs(pr))), max_raw_dR2=float(np.nanmax(dr2)),
                all_features={fn[j]: dict(spearman=sp[j], partial_within_unit=pr[j], raw_dR2=dr2[j]) for j in range(len(fn))})


# ---------------------------------------------------------------------------------------------------- Q3 timing
def lag_stats(l: np.ndarray) -> dict:
    l = np.asarray(l, np.float64)
    if len(l) == 0:
        return dict(n=0)
    return dict(n=int(len(l)), median=float(np.median(l)), q25=float(np.quantile(l, 0.25)),
                q75=float(np.quantile(l, 0.75)), share_abs_le6=float((np.abs(l) <= 6).mean()),
                share_lt_minus6=float((l < -6).mean()), share_gt24=float((l > 24).mean()))


def smooth3(y: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """One curve: unobserved hours forward-filled, then a centred 3-h running median (removes one-hour dips/spikes)."""
    v = pd.Series(np.where(obs, y, np.nan)).ffill().bfill().fillna(0.0)
    return v.rolling(3, center=True, min_periods=1).median().to_numpy()


def waves(s: np.ndarray, frac: float) -> np.ndarray:
    top = s.max()
    if top <= 0:
        return np.array([], int)
    pk, _ = find_peaks(np.r_[0.0, s, 0.0], height=max(0.005, frac * top), prominence=frac * top, distance=12)
    return pk - 1


def q3(D: dict) -> dict:
    X, ev, season, U = D["X"], D["event"], D["season"], len(D["y"])
    g, pr, sm = X["gust"], X["precip"], X["soil_moisture"]
    yf, of, ts, ys = D["y_full"], D["obs_full"], D["tstar"], D["ystar"]
    act = ys >= ACTIVE
    tg = g.argmax(1)
    lag = ts - tg
    seasons = ("Nov-Apr", "May-Oct")

    def by(l, sub):
        return dict(all=lag_stats(l[sub]), **{s: lag_stats(l[sub & (season == s)]) for s in seasons},
                    by_event={e: lag_stats(l[sub & (ev == e)]) for e in D["events"]})
    out = dict(active_units=int(act.sum()))
    out["outage_peak_minus_gust_peak_full_window"] = by(lag, act)
    out["gust_peak_in_prefix_share_active"] = float((tg[act] < ORIGIN).mean())
    out["median_gust_max_active"] = {s: float(np.median(g[act & (season == s)].max(1))) for s in seasons}
    out["median_gust_max_active_by_event"] = {e: float(np.median(g[act & (ev == e)].max(1))) for e in D["events"]}
    # which forcing's peak locates the outage peak (peak and forcing both inside the forecast window)
    fw = act & (ts >= ORIGIN)
    names = eih_names()
    cols = ["gust_x10*one@0", "convective*one@0", "rain*one@0", "gust_x10*one@12", "convective*one@12"]
    jj = [names.index(c) for c in cols]
    phi = npz_memmap(GW / "eih_pop.npz", "phi")
    E = np.empty((U, H, len(cols)), np.float32)
    for u0 in range(0, U, 500):
        E[u0:u0 + 500] = np.asarray(phi[u0:u0 + 500], np.float32)[:, :, jj]
    del phi
    forcings = {"host gust (area mean)": g[:, ORIGIN:], "host gust_cell_max": X["gust_cell_max"][:, ORIGIN:],
                "host precip": pr[:, ORIGIN:], "host cape": X["cape"][:, ORIGIN:],
                **{f"EIH pop {c}": E[:, :, k] for k, c in enumerate(cols)}}
    fc = {}
    for name, arr in forcings.items():
        ok = fw & (arr.max(1) > 0)
        l = ts - (arr.argmax(1) + ORIGIN)
        fc[name] = dict(defined_share=float(ok.sum() / fw.sum()), all=lag_stats(l[ok]),
                        **{s: lag_stats(l[ok & (season == s)]) for s in seasons},
                        by_event_share_abs_le6={e: float((np.abs(l[ok & (ev == e)]) <= 6).mean()) for e in D["events"]})
    out["forcing_peak_alignment_forecast_window"] = dict(units=int(fw.sum()), forcings=fc)
    del E
    # local lag (largest gust of the 24 h up to the outage peak) and rise time (last hour at <= 10% of the peak)
    loc, rise = np.full(U, np.nan), np.full(U, np.nan)
    for u in np.where(fw)[0]:
        t = ts[u]
        loc[u] = 24 - g[u, t - 24:t + 1].argmax()
        v = np.where(of[u, :t], yf[u, :t], np.nan)
        below = np.where(v <= 0.1 * ys[u])[0]
        if len(below):
            rise[u] = t - below[-1]
    rec_l, rec_r = {}, {}
    for sname, sub in (("all", fw), ("Nov-Apr", fw & (season == "Nov-Apr")), ("May-Oct", fw & (season == "May-Oct"))):
        rec_l[sname] = dict(n=int(sub.sum()), median=float(np.median(loc[sub])), share_le3=float((loc[sub] <= 3).mean()),
                            share_ge12=float((loc[sub] >= 12).mean()))
        rr = rise[sub]
        rec_r[sname] = dict(n=int(sub.sum()), censored_share=float(np.isnan(rr).mean()),
                            median=float(np.nanmedian(rr)), q25=float(np.nanquantile(rr, 0.25)),
                            q75=float(np.nanquantile(rr, 0.75)))
    out["local_gust_max_to_outage_peak_h"] = rec_l
    out["rise_time_10pct_to_peak_h"] = rec_r
    # secondary waves on the smoothed observed curve
    res = {}
    for frac in (0.25, 0.5):
        nw = np.zeros(U, int)
        ratio, sep, after, t_sec = [], [], [], []
        ev_sec = []
        for u in np.where(act)[0]:
            s = smooth3(yf[u], of[u])
            pk = waves(s, frac)
            nw[u] = len(pk)
            if len(pk) >= 2:
                main = pk[np.argmax(s[pk])]
                gm = g[u, max(main - 12, 0):main + 1].max()
                for p in pk:
                    if p == main:
                        continue
                    ratio.append(g[u, max(p - 12, 0):p + 1].max() / gm)
                    sep.append(abs(p - main)); after.append(p > main); t_sec.append(p); ev_sec.append(ev[u])
        ratio, sep, after = np.array(ratio), np.array(sep), np.array(after)
        res[f"threshold_{frac}"] = dict(
            share_multi_wave=float((nw[act] >= 2).mean()), units_multi_wave=int((nw[act] >= 2).sum()),
            by_season={s: float((nw[act & (season == s)] >= 2).mean()) for s in seasons},
            by_event={e: float((nw[act & (ev == e)] >= 2).mean()) for e in D["events"]},
            secondary_peaks=int(len(ratio)), share_after_main=float(after.mean()), median_separation_h=float(np.median(sep)),
            gust_ratio_median=float(np.median(ratio)), share_gust_ratio_ge_0p8=float((ratio >= 0.8).mean()),
            share_gust_ratio_lt_0p6=float((ratio < 0.6).mean()),
            median_secondary_hour_by_event={e: float(np.median(np.array(t_sec)[np.array(ev_sec) == e]))
                                            for e in D["events"] if (np.array(ev_sec) == e).any()})
    out["secondary_wave"] = res
    # antecedent precipitation (48 h before the forecast-window gust peak) and outage at equal wind
    tgf = g[:, ORIGIN:].argmax(1) + ORIGIN
    P48 = np.take_along_axis(pr, tgf[:, None] - np.arange(1, 49)[None, :], 1).sum(1)
    storm = np.array([pr[u, tgf[u] - 6:min(tgf[u] + 7, T)].sum() for u in range(U)])
    SM = sm[np.arange(U), tgf]
    gmaxf, gcell = g[:, ORIGIN:].max(1), X["gust_cell_max"][:, ORIGIN:].max(1)
    Cq = np.column_stack([gmaxf, gcell])
    dpk = D["ystar_f"] - D["pstar_f"]
    ant = dict(P48_median_mm=float(np.median(P48)), P48_vs_soil_moisture_within_event=float(
        partial(rank01(P48), rank01(SM), None, [ev])["rho"][0]),
        P48_vs_storm_precip_within_event=float(partial(rank01(P48), rank01(storm), None, [ev])["rho"][0]))
    for tname, tv in (("observed_peak", D["ystar_f"]), ("peak_residual", dpk)):
        rec = {}
        for sname, sub in (("all", np.ones(U, bool)), ("gust_max_ge15", gmaxf >= 15), ("Nov-Apr", season == "Nov-Apr"),
                           ("May-Oct", season == "May-Oct")):
            Rt, RC = rank01(tv[sub]), rank01(Cq[sub])
            p1 = partial(Rt, rank01(np.column_stack([P48[sub], SM[sub]])), RC, [ev[sub]])
            ci = cluster_ci(p1["ey"], p1["eF"], D["fips"][sub])
            p2 = partial(Rt, rank01(P48[sub]), np.column_stack([RC, rank01(np.column_stack([SM[sub], storm[sub]]))]),
                         [ev[sub]])
            rec[sname] = dict(n=int(sub.sum()), P48=dict(rho=p1["rho"][0], ci95=ci[0]),
                              soil_moisture=dict(rho=p1["rho"][1], ci95=ci[1]),
                              P48_given_soil_and_storm_precip=dict(rho=p2["rho"][0],
                                                                   ci95=cluster_ci(p2["ey"], p2["eF"], D["fips"][sub])[0]))
        ant[tname] = rec
    ratios = []
    for e in D["events"]:
        i = np.where(ev == e)[0]
        gt = np.digitize(gmaxf[i], np.quantile(gmaxf[i], [1 / 3, 2 / 3]))
        for k in range(3):
            j = i[gt == k]
            pt = np.digitize(P48[j], np.quantile(P48[j], [1 / 3, 2 / 3]))
            lo, hi = np.median(D["ystar_f"][j[pt == 0]]), np.median(D["ystar_f"][j[pt == 2]])
            ratios.append(hi / lo if lo > 0 else np.nan)
    ratios = np.array(ratios)
    ant["binned_P48_top_vs_bottom_tercile"] = dict(cells=int(np.isfinite(ratios).sum()),
                                                   median_ratio=float(np.nanmedian(ratios)),
                                                   share_ratio_gt1=float(np.mean(ratios[np.isfinite(ratios)] > 1)))
    out["antecedent_precipitation"] = ant
    return out


# ---------------------------------------------------------------------------------------------- Q4 artefact flags
LEVELS = [(-1, 0), (0, 0.001), (0.001, 0.01), (0.01, 0.05), (0.05, 0.2), (0.2, 1.01)]


def lvl(lo, hi):
    return "0" if lo < 0 else f"({lo},{min(hi, 1)}]"


def q4(D: dict) -> dict:
    Z = np.load(GW / "train_mask_e3.npz")
    m, r, y, ev, st = D["m"], D["r"], D["y"], D["event"], D["state"]
    se, y2 = r ** 2, np.where(m, y ** 2, 0.0)
    SSE, N = se.sum(), m.sum()
    fl = {k: Z[f"flag_{k}"][:, ORIGIN:] & m for k in ("dip", "spike", "plateau", "over")}
    anyf = np.zeros_like(m)
    for v in fl.values():
        anyf |= v
    fl_all = dict(fl, any=anyf)
    out = dict(cells={k: int(v.sum()) for k, v in fl_all.items()}, any_share_of_observed=float(anyf.sum() / N),
               units_with_any=int(anyf.any(1).sum()),
               sse_share={k: float(se[v].sum() / SSE) for k, v in fl_all.items()},
               y2_share={k: float(y2[v].sum() / y2.sum()) for k, v in fl_all.items()},
               mean_residual={k: float(r[v].mean()) for k, v in fl_all.items()},
               under_share_of_flag_sse={k: float(se[v & (r > 0)].sum() / max(se[v].sum(), 1e-12)) for k, v in fl_all.items()},
               median_y={k: float(np.median(y[v])) for k, v in fl_all.items()},
               sse_per_cell_vs_average={k: float((se[v].sum() / v.sum()) / (SSE / N)) for k, v in fl_all.items()})
    upl = fl["plateau"].any(1)
    out["plateau_units"] = dict(units=int(upl.sum()), sse_share_whole_units=float(se[upl].sum() / SSE),
                                y2_share_whole_units=float(y2[upl].sum() / y2.sum()))
    out["by_event"] = {e: dict(flag_share=float(anyf[ev == e].sum() / m[ev == e].sum()),
                               share_of_all_flagged=float(anyf[ev == e].sum() / anyf.sum()),
                               **{k: int(fl[k][ev == e].sum()) for k in fl}) for e in D["events"]}
    fs = pd.Series(anyf.sum(1)).groupby(st).sum().sort_values(ascending=False)
    out["by_state_top6"] = {s: dict(flagged_cells=int(fs[s]), share_of_all_flagged=float(fs[s] / anyf.sum()),
                                    rate=float(fs[s] / m[st == s].sum()),
                                    plateau=int(fl["plateau"][st == s].sum()), spike=int(fl["spike"][st == s].sum()))
                            for s in fs.index[:6]}
    out["states_for_50pct_flagged"] = n_for(fs.to_numpy(), 0.5)
    es = pd.Series(anyf.sum(1)).groupby([ev, st]).sum().sort_values(ascending=False)
    out["top3_event_state_blocks"] = {f"{a}|{b}": dict(flagged_cells=int(es[(a, b)]),
                                                       share_of_all_flagged=float(es[(a, b)] / anyf.sum()))
                                      for a, b in es.index[:3]}
    out["by_hour_level"] = {}
    for lo, hi in LEVELS:
        b = m & (y > lo) & (y <= hi)
        out["by_hour_level"][lvl(lo, hi)] = dict(cells=int(b.sum()), any_rate=float(anyf[b].sum() / b.sum()),
                                                 share_of_flagged=float(anyf[b].sum() / anyf.sum()),
                                                 **{f"{k}_rate": float(fl[k][b].sum() / b.sum()) for k in fl})
    # plateau runs over the full window
    pf = Z["flag_plateau"] & D["obs_full"]
    runs = []
    for u in np.where(pf.any(1))[0]:
        idx = np.where(pf[u])[0]
        br = np.where(np.diff(idx) > 1)[0]
        for a_, b_ in zip(np.r_[idx[0], idx[br + 1]], np.r_[idx[br], idx[-1]]):
            runs.append((u, a_, b_, float(D["y_full"][u, a_]), float(D["ystar"][u])))
    ru = np.array(runs, dtype=float)
    ln = ru[:, 2] - ru[:, 1] + 1
    out["plateau_run_list"] = [dict(fips=D["fips"][int(q[0])], state=D["state"][int(q[0])], event=D["event"][int(q[0])],
                                    start_h=int(q[1]), end_h=int(q[2]), value=float(q[3]), unit_peak=float(q[4]),
                                    cust=float(D["cust"][int(q[0])]),
                                    unit_sse_share=float(se[int(q[0])].sum() / SSE)) for q in ru]
    ep = [q for q in out["plateau_run_list"] if q["event"] == "2024-05-08" and 84 <= q["start_h"] <= 88 and q["end_h"] == T - 1]
    out["stale_episode_2024-05-08"] = dict(runs=len(ep), states=sorted(set(q["state"] for q in ep)),
                                           start_h_range=[min(q["start_h"] for q in ep), max(q["start_h"] for q in ep)])
    out["plateau_runs"] = dict(runs=int(len(ru)), units=int(len(set(ru[:, 0].astype(int)))), median_length_h=float(np.median(ln)),
                               share_start_at_hour0=float((ru[:, 1] == 0).mean()),
                               share_end_at_hour215=float((ru[:, 2] == T - 1).mean()),
                               median_value=float(np.median(ru[:, 3])), share_value_ge_0p01=float((ru[:, 3] >= 0.01).mean()),
                               median_value_over_unit_peak=float(np.median(ru[:, 3] / ru[:, 4])))
    ov = np.where(fl["over"].any(1))[0]
    out["over_units_whole_sse_share"] = float(se[ov].sum() / SSE)
    out["over_units"] = [dict(fips=D["fips"][u], event=D["event"][u], cust=float(D["cust"][u]),
                              hours=int(fl["over"][u].sum())) for u in ov]
    return out


# -------------------------------------------------------------------------------------------------- Q5 recovery
def q5(D: dict) -> dict:
    yf, of, ts, ys, cust = D["y_full"], D["obs_full"], D["tstar"], D["ystar"], D["cust"]
    X, ev, st, season, U = D["X"], D["event"], D["state"], D["season"], len(D["y"])
    ar = np.arange(U)
    elig = (ys >= ACTIVE) & (ys * cust >= 100) & (ts <= T - 1 - 24)
    S = (6, 12, 24, 48, 72, 96)
    R = {}
    for s in S:
        tt = np.minimum(ts + s, T - 1)
        ok = elig & (ts + s <= T - 1) & of[ar, tt]
        R[s] = np.where(ok, yf[ar, tt] / np.where(ys > 0, ys, 1), np.nan)
    h = {}
    for q in (0.5, 0.1):
        dur, cen = np.full(U, np.nan), np.zeros(U, bool)
        for u in np.where(elig)[0]:
            seg = np.where(of[u, ts[u] + 1:], yf[u, ts[u] + 1:], np.inf)
            hit = np.where(seg <= q * ys[u])[0]
            if len(hit):
                dur[u] = hit[0] + 1
            else:
                dur[u], cen[u] = T - 1 - ts[u], True
        h[q] = (dur, cen)
    d50, c50 = h[0.5]

    d10, c10 = h[0.1]

    def summ(sub):
        return dict(n=int(sub.sum()), km_median_half_life_h=km_median(d50[sub], (~c50[sub]).astype(int)),
                    censored_share=float(c50[sub].mean()),
                    km_median_time_to_10pct_h=km_median(d10[sub], (~c10[sub]).astype(int)),
                    censored_share_10pct=float(c10[sub].mean()),
                    median_R24=float(np.nanmedian(R[24][sub])), median_R48=float(np.nanmedian(R[48][sub])))
    PEAKBINS = {"[0.01,0.05)": (0.01, 0.05), "[0.05,0.2)": (0.05, 0.2), "[0.2,1]": (0.2, 1.01)}
    out = dict(eligible=int(elig.sum()), definition="y* >= 0.01, >= 100 customers out at the peak, >= 24 h of window after the peak",
               all=summ(elig), by_season={s: summ(elig & (season == s)) for s in ("Nov-Apr", "May-Oct")},
               by_peak_size={k: summ(elig & (ys >= lo) & (ys < hi)) for k, (lo, hi) in PEAKBINS.items()},
               by_peak_size_and_season={f"{k}|{s}": summ(elig & (ys >= lo) & (ys < hi) & (season == s))
                                        for k, (lo, hi) in PEAKBINS.items() for s in ("Nov-Apr", "May-Oct")},
               by_event={e: summ(elig & (ev == e)) for e in D["events"]},
               by_event_peak_ge_0p05={e: summ(elig & (ev == e) & (ys >= 0.05)) for e in D["events"]})
    sc = pd.Series(elig).groupby(st).sum()
    out["by_state_n_ge_60"] = {s: summ(elig & (st == s)) for s in sc[sc >= 60].sort_values(ascending=False).index}
    # relations of the share remaining at 24 h with peak size and post-peak weather (within event)
    g, t2, sn, pr = X["gust"], X["t2m_c"], X["snowfall"], X["precip"]
    W = {}
    for name, arr, win, fn in (("gust_mean_24h", g, 24, np.mean), ("gust_max_24h", g, 24, np.max),
                               ("t2m_mean_48h", t2, 48, np.mean), ("t2m_min_48h", t2, 48, np.min),
                               ("hours_below_0C_48h", t2, 48, lambda a: np.sum(a < 0)),
                               ("snowfall_sum_48h", sn, 48, np.sum), ("precip_sum_24h", pr, 24, np.sum)):
        v = np.full(U, np.nan)
        for u in np.where(elig)[0]:
            v[u] = fn(arr[u, ts[u] + 1:min(ts[u] + 1 + win, T)])
        W[name] = v
    ok = elig & np.isfinite(R[24])
    Rt = rank01(R[24][ok])
    base_c = rank01(np.column_stack([np.log(ys[ok]), np.log(cust[ok])]))
    rel = dict(n=int(ok.sum()))
    lys = np.log(np.maximum(ys, 1e-9))
    for name, v in (("log_peak", lys), ("log_cust", np.log(cust)), ("log_peak_customers", lys + np.log(cust))):
        p = partial(Rt, rank01(v[ok]), None, [ev[ok]])
        rel[name] = dict(within_event=p["rho"][0], ci95=cluster_ci(p["ey"], p["eF"], D["fips"][ok])[0])
    for name, v in W.items():
        p0 = partial(Rt, rank01(v[ok]), None, [ev[ok]])
        p1 = partial(Rt, rank01(v[ok]), base_c, [ev[ok]])
        rel[name] = dict(within_event=p0["rho"][0], partial_given_peak_cust=p1["rho"][0],
                         ci95=cluster_ci(p1["ey"], p1["eF"], D["fips"][ok])[0])
    # cold / snow effects only in the season where they exist
    okc = ok & (season == "Nov-Apr")
    for name in ("t2m_mean_48h", "hours_below_0C_48h", "snowfall_sum_48h", "gust_mean_24h"):
        p1 = partial(rank01(R[24][okc]), rank01(W[name][okc]),
                     rank01(np.column_stack([np.log(ys[okc]), np.log(cust[okc])])), [ev[okc]])
        rel[f"{name}_Nov-Apr_only"] = dict(partial_given_peak_cust=p1["rho"][0],
                                           ci95=cluster_ci(p1["ey"], p1["eF"], D["fips"][okc])[0], n=int(okc.sum()))
    out["R24_relations"] = rel
    # decay shape on a fixed cohort (>= 96 h of window after the peak)
    Z = np.load(GW / "train_mask_e3.npz")
    pfl = Z["flag_plateau"]
    cohort = elig & (ts <= T - 1 - 96)
    noplat = ~np.array([pfl[u, ts[u]:].any() for u in range(U)])
    shape = {}
    SS = (3, 6, 12, 24, 48, 72, 96)

    def implied(c, s):
        return float(s * np.log(2) / -np.log(c)) if 0 < c < 1 else None
    cohorts = [("cohort", cohort), ("cohort_without_plateau_flags", cohort & noplat)]
    cohorts += [(f"cohort_peak_{k}", cohort & (ys >= lo) & (ys < hi)) for k, (lo, hi) in PEAKBINS.items()]
    for cname, cmask in cohorts:
        curve, wcurve = np.full(97, np.nan), np.full(97, np.nan)
        for s in range(97):
            tt = np.minimum(ts + s, T - 1)
            okc = cmask & of[ar, tt]
            v = np.where(okc, yf[ar, tt] / np.where(ys > 0, ys, 1), np.nan)
            curve[s] = np.nanmedian(v[cmask])
            wcurve[s] = yf[ar, tt][okc].sum() / ys[okc].sum()          # peak-weighted: sum y(t*+s) / sum y*
        imp = {s: implied(curve[s], s) for s in SS}
        wimp = {s: implied(wcurve[s], s) for s in SS}
        fits_b_better, npts = 0, 0
        for u in np.where(cmask)[0]:
            s_ = np.arange(1, 97)
            v = yf[u, ts[u] + 1:ts[u] + 97]
            o = of[u, ts[u] + 1:ts[u] + 97] & (v > 0)
            if o.sum() < 24:
                continue
            ly = np.log(v[o])
            A = np.polyfit(s_[o], ly, 1, full=True)[1]
            Bf = np.polyfit(np.log1p(s_[o]), ly, 1, full=True)[1]
            npts += 1
            fits_b_better += int((Bf[0] if len(Bf) else 0) < (A[0] if len(A) else 0))
        shape[cname] = dict(units=int(cmask.sum()), median_remaining={s: float(curve[s]) for s in SS},
                            implied_half_life_h_median_curve=imp,
                            peak_weighted_remaining={s: float(wcurve[s]) for s in SS},
                            implied_half_life_h_peak_weighted=wimp,
                            peak_weighted_exponential_from_24h_predicts_96h=float(wcurve[24] ** 4),
                            peak_weighted_observed_96h=float(wcurve[96]),
                            units_fitted=npts, share_powerlaw_fits_better=float(fits_b_better / max(npts, 1)))
    out["decay_shape"] = shape
    return out


# ------------------------------------------------------------------------------------------- Q6 other surprises
def q6(D: dict) -> dict:
    m, r, y, ev, U = D["m"], D["r"], D["y"], D["event"], len(D["y"])
    se, y2 = r ** 2, np.where(m, y ** 2, 0.0)
    SSE = se.sum()
    out = {}
    cty = pd.DataFrame(dict(fips=D["fips"], cust=D["cust"])).groupby("fips").cust.agg(["min", "max"])
    out["cust_constant_across_events"] = bool((cty["min"] == cty["max"]).all())
    stt = pd.read_parquet(STATICS, columns=["fips", "log_pop"]).set_index("fips")
    pop = np.exp(stt.log_pop.reindex(cty.index))
    cpp = (cty["max"] / pop)
    out["customers_per_person_counties"] = dict(
        counties=int(cpp.notna().sum()), quantiles={q: float(cpp.quantile(q)) for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)},
        n_lt_0p25=int((cpp < 0.25).sum()), n_lt_0p3=int((cpp < 0.3).sum()), n_gt_0p8=int((cpp > 0.8).sum()),
        n_gt_1=int((cpp > 1.0).sum()))
    cu = cpp.reindex(D["fips"]).to_numpy()
    bands = {"<0.3": cu < 0.3, "0.3-0.8": (cu >= 0.3) & (cu <= 0.8), ">0.8": cu > 0.8}
    out["by_customers_per_person"] = {k: dict(units=int(b.sum()), unit_share=float(b.mean()), sse_share=float(se[b].sum() / SSE),
                                              y2_share=float(y2[b].sum() / y2.sum()),
                                              median_peak=float(np.median(D["ystar_f"][b])),
                                              units_reaching_1=int((D["y_full"][b] >= 0.999).any(1).sum()))
                                      for k, b in bands.items()}
    hit1 = np.where((np.where(D["obs_full"], D["y_full"], 0) >= 0.999).any(1))[0]
    out["units_reaching_fraction_1"] = [dict(fips=D["fips"][u], state=D["state"][u], event=D["event"][u],
                                             cust=float(D["cust"][u]), customers_per_person=float(cu[u]),
                                             hours=int((np.where(D["obs_full"][u], D["y_full"][u], 0) >= 0.999).sum()))
                                        for u in hit1]
    # stale curves: the same non-zero value over every observed hour of the window
    yv = np.where(D["obs_full"], D["y_full"], np.nan)
    const = (np.nanmax(yv, 1) == np.nanmin(yv, 1)) & (np.nanmax(yv, 1) > 0)
    out["units_constant_nonzero_whole_window"] = int(const.sum())
    pre = D["y0"] > 0.01
    out["prefix_active_units"] = dict(units=int(pre.sum()), sse_share=float(se[pre].sum() / SSE))
    # events dominated by a few county-events
    dom = {}
    for e in D["events"]:
        i = ev == e
        s_u, y_u = se[i].sum(1), y2[i].sum(1)
        top1 = np.argsort(s_u)[::-1][:1]
        dom[e] = dict(top1_sse_share=share_top(s_u, 1 / i.sum()), top5_sse_share=float(np.sort(s_u)[::-1][:5].sum() / s_u.sum()),
                      top5_y2_share=float(np.sort(y_u)[::-1][:5].sum() / y_u.sum()),
                      top1_fips=D["fips"][i][top1[0]], top1_state=D["state"][i][top1[0]],
                      sse_share_window_hours_ge_168=float(se[i][:, 168 - ORIGIN:].sum() / s_u.sum()),
                      active_units_peaking_at_or_after_168=float((D["tstar"][i & (D["ystar"] >= ACTIVE)] >= 168).mean()),
                      unobserved_forecast_share=float(1 - m[i].mean()))
    out["by_event"] = dom
    Z = np.load(GW / "train_mask_e3.npz")
    flagged = np.zeros_like(m)
    for k in ("dip", "spike", "plateau", "over"):
        flagged |= Z[f"flag_{k}"][:, ORIGIN:] & m
    sse_u = se.sum(1)
    out["top10_units"] = [dict(fips=D["fips"][u], state=D["state"][u], event=D["event"][u], cust=float(D["cust"][u]),
                               customers_per_person=float(cu[u]), sse_share=float(sse_u[u] / SSE),
                               obs_peak_f=float(D["ystar_f"][u]), pred_peak_f=float(D["pstar_f"][u]),
                               peak_hour=int(D["tstar"][u]), flagged_hours=int(flagged[u].sum()),
                               window_mean_obs=float(y[u][m[u]].mean()), window_mean_pred=float(D["P"][u][m[u]].mean()))
                          for u in np.argsort(sse_u)[::-1][:10]]
    # counties that recur among the top 1% of county-events
    top = np.argsort(sse_u)[::-1][:int(round(0.01 * U))]
    vc = pd.Series(D["fips"][top]).value_counts()
    out["top1pct_repeat_counties"] = {f: int(c) for f, c in vc[vc >= 2].items()}
    return out


# --------------------------------------------------------------------------------------------------------- main
SECTIONS = {"q1": "q1", "q2": "q2", "q3": "q3", "q4": "q4", "q5": "q5", "q6": "q6"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", nargs="+", default=list(SECTIONS))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    t0 = time.time()
    D = load()
    res = dict(meta=dict(inputs=dict(features=str(FEAT.relative_to(ROOT)), base="runs/open_gcrk_20260919/e3r2/main/"
                                     "seed0/fold0{1..5}/W+Cin/outer.npz (out-of-fold, county-grouped five folds, seed 0)",
                                     eih=["data/interim/geo_weather/eih_pop.npz", "data/interim/geo_weather/eih_area.npz"],
                                     flags="data/interim/geo_weather/train_mask_e3.npz",
                                     nodes="data/interim/geo_weather/nodes_cs.parquet",
                                     statics="data/interim/county_statics.parquet"),
                         seed=SEED, bootstrap=B_BOOT, active_threshold=ACTIVE, warm_events=list(WARM)))
    for s in a.sections:
        ts = time.time()
        res[s] = globals()[SECTIONS[s]](D)
        print(f"{s} done in {time.time() - ts:.1f}s, max RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.0f} MB",
              flush=True)
    res["meta"]["runtime_s"] = time.time() - t0
    res["meta"]["max_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(g6(res), indent=1) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
