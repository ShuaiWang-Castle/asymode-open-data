"""Inference-only diagnostics (PREREG sections 9-11). No training happens here.

  concentration   per-unit gains D_u = SSE_u(W) - SSE_u(GCRK) per seed; top-k shares, event
                  shares, pooled differences without the top 1% of units / without each
                  event (H1); seed-pair Spearman correlations and sign agreement (H3)
  swap            geography exchange test (H2): every OUTER unit through its fold's GCRK
                  model with its own geography and with 200 donor counties' geographies
  figure3         data for Figure 3: the representative county's kernel contribution by
                  deposit hour (integrated gradients along kernel exit closed -> open,
                  exact for the piecewise-linear readout up to quadrature), the drivers
                  of its push-window push (integrated gradients over weather inputs and
                  geography), and its forecast curves
Outputs: results/*.csv and results/figure_data/*.npz.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

for _n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_n] = "1"

import numpy as np
import pandas as pd
import torch
from scipy import stats as sps

import common as C

sys.path.insert(0, str(C.ROOT / "src"))
from asymode.asym_host import AsymODE  # noqa: E402
from asymode.gcrk_train import make_batch, _std  # noqa: E402

torch.set_num_threads(1)
FIGDATA = C.RESULTS / "figure_data"
PUSH = (168, 215)                     # PREREG 11: wave B of the representative event
REP_EVENT = "2021-12-11"


# ----------------------------------------------------------------------------- models
def load_model(design, seed, fold, arm, F):
    snap = torch.load(C.cell(design, seed, fold, arm) / "final.pt", map_location="cpu", weights_only=False)
    m = AsymODE(F["xu"].shape[-1], F["xr"].shape[-1], F["xo"].shape[-1])
    if arm == "GCRK":
        m.attach_gcrk(torch.zeros(F["geo"].shape[-1]))
    m.load_state_dict(snap["model_state"], strict=True)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, snap["stats"]


def fold_of(design: str) -> dict[int, str]:
    sp = C.load_splits()[design]
    return {u: f for f, spec in sp.items() for u in spec["outer"]}


# ---------------------------------------------------------------------- concentration
def concentration(design="main"):
    F = C.load_features(); n = len(F["y"])
    D, rows = {}, []
    for s in C.SEEDS:
        W, G = C.collect(design, "W", s, n), C.collect(design, "GCRK", s, n)
        if W is None or G is None:
            continue
        D[s] = C.unit_sse(W, F) - C.unit_sse(G, F)
    per_unit = pd.DataFrame(dict(event=F["event"], fips=F["fips"], **{f"gain_seed{s}": d for s, d in D.items()}))
    per_unit.to_csv(C.RESULTS / f"concentration_units_{design}.csv", index=False)
    m = F["m"].astype(bool); ncell = m.sum()
    for s, d in D.items():
        W = C.collect(design, "W", s, n); G = C.collect(design, "GCRK", s, n)
        sw, sg = C.unit_sse(W, F), C.unit_sse(G, F)
        rm = lambda keep: np.sqrt(sg[keep].sum() / m[keep].sum()) - np.sqrt(sw[keep].sum() / m[keep].sum())
        order = np.argsort(-d)
        top1 = order[:max(1, int(np.ceil(0.01 * n)))]
        keep = np.ones(n, bool); keep[top1] = False
        row = dict(seed=s, total_gain=d.sum(), pooled_rmse_diff=rm(np.ones(n, bool)),
                   share_top10_units=d[order[:10]].sum() / d.sum() if d.sum() != 0 else np.nan,
                   share_top1pct_units=d[top1].sum() / d.sum() if d.sum() != 0 else np.nan,
                   pooled_rmse_diff_without_top1pct=rm(keep), n_units_gain_positive=int((d > 0).sum()),
                   n_units_gain_negative=int((d < 0).sum()))
        for ev in sorted(set(F["event"].tolist())):
            e = F["event"] == ev
            row[f"share_{ev}"] = d[e].sum() / d.sum() if d.sum() != 0 else np.nan
            row[f"rmse_diff_without_{ev}"] = rm(~e)
        rows.append(row)
    summ = pd.DataFrame(rows)
    summ.to_csv(C.RESULTS / f"concentration_summary_{design}.csv", index=False)
    seeds = sorted(D)
    cor = [dict(a=a, b=b, spearman=sps.spearmanr(D[a], D[b]).statistic) for i, a in enumerate(seeds) for b in seeds[i + 1:]]
    cor = pd.DataFrame(cor)
    Dm = np.stack([D[s] for s in seeds])
    big = np.abs(Dm).mean(0) > np.median(np.abs(Dm).mean(0))
    agree = ((np.sign(Dm) > 0).sum(0) >= 4) | ((np.sign(Dm) < 0).sum(0) >= 4)
    h3 = dict(mean_pairwise_spearman=float(cor.spearman.mean()) if len(cor) else np.nan,
              share_sign_agree_4of5_among_large=float(agree[big].mean()), n_large=int(big.sum()))
    cor.to_csv(C.RESULTS / f"seed_pair_correlation_{design}.csv", index=False)
    (C.RESULTS / f"h3_{design}.json").write_text(json.dumps(h3, indent=1) + "\n")
    print(summ.to_string()); print(h3)


# ------------------------------------------------------------------------------- swap
def swap(design="main", n_donors=200, chunk=8, seeds=None):
    F = C.load_features(); n = len(F["y"])
    fips = F["fips"]
    uniq = sorted(set(fips.tolist()))
    first = {f: int(np.where(fips == f)[0][0]) for f in uniq}
    geo_raw = {f: F["geo"][first[f]] for f in uniq}
    sp = C.load_splits()[design]
    for s in (C.SEEDS if seeds is None else seeds):
        out_rows = []
        if not all((C.cell(design, s, f, "GCRK") / "final.pt").exists() for f in sp):
            print("seed", s, "incomplete, skipped"); continue
        for f, spec in sp.items():
            m, st = load_model(design, s, f, "GCRK", F)
            outer = np.array(spec["outer"])
            for k in range(0, len(outer), chunk):
                us = outer[k:k + chunk]
                bb, donors_all = [], []
                for u in us:
                    rng = np.random.default_rng(1000003 * s + int(u))
                    pool = [g for g in uniq if g != fips[u]]
                    donors = [fips[u]] + list(rng.choice(pool, n_donors, replace=False))
                    donors_all.append(donors)
                b = make_batch(F, np.repeat(us, n_donors + 1), st)
                g = np.stack([geo_raw[d] for ds in donors_all for d in ds])
                b["geo"] = torch.from_numpy(_std(g, st["geo"], 0))
                with torch.no_grad():
                    P = m(b)["P"].numpy()
                y = b["y"].numpy(); mk = b["m"].numpy()
                r = np.sqrt(((P - y) ** 2 * mk).sum(1) / mk.sum(1)).reshape(len(us), n_donors + 1)
                for j, u in enumerate(us):
                    own, don = r[j, 0], r[j, 1:]
                    out_rows.append(dict(seed=s, fold=int(f), unit=int(u), event=F["event"][u], fips=fips[u],
                                         rmse_own=own, rmse_donor_median=float(np.median(don)),
                                         percentile=float((don < own).mean()), donor_sd=float(don.std()),
                                         rmse_donor_min=float(don.min()), rmse_donor_max=float(don.max())))
            print("swap", s, f, flush=True)
        pd.DataFrame(out_rows).to_csv(C.RESULTS / f"swap_units_{design}_seed{s}.csv", index=False)
    parts = [pd.read_csv(C.RESULTS / f"swap_units_{design}_seed{s}.csv") for s in C.SEEDS
             if (C.RESULTS / f"swap_units_{design}_seed{s}.csv").exists()]
    d = pd.concat(parts, ignore_index=True)
    summ = []
    for s, g in d.groupby("seed"):
        act = g[g.donor_sd > 1e-5]
        w = sps.wilcoxon(act.percentile - 0.5, alternative="less") if len(act) > 10 else None
        summ.append(dict(seed=s, n_units=len(g), n_active=len(act), median_percentile=float(act.percentile.median()),
                         share_own_better_than_median_donor=float((act.rmse_own < act.rmse_donor_median).mean()),
                         wilcoxon_p_less=float(w.pvalue) if w is not None else np.nan,
                         mean_rmse_own_minus_median_donor=float((act.rmse_own - act.rmse_donor_median).mean())))
    pd.DataFrame(summ).to_csv(C.RESULTS / f"swap_summary_{design}.csv", index=False)
    print(pd.DataFrame(summ).to_string())


# ---------------------------------------------------------------------------- figure 3
def representative_county(F) -> int:
    """PREREG 11: two-wave county of the representative event, >= 10,000 customers,
    largest observed rise of p inside the push window."""
    sel = json.loads((C.HERE / "selected_events.json").read_text())["events"]
    e = [x for x in sel if x["event"] == REP_EVENT][0]
    se = pd.read_parquet(C.ROOT / "data/interim/storm_events_county.parquet")
    W = {"Thunderstorm Wind", "High Wind", "Strong Wind", "Tornado"}
    s = pd.Timestamp(e["window_start_utc"])
    w = se[se.EVENT_TYPE.isin(W) & (se.t_begin_utc >= s + pd.Timedelta(hours=72)) & (se.t_begin_utc < s + pd.Timedelta(hours=216))].copy()
    w["d"] = ((w.t_begin_utc - s).dt.total_seconds() // 3600 - 72) // 24
    two = set(w[w.d == 0].fips) & set(w[w.d.isin([4, 5])].fips)
    cand = [u for u in np.where(F["event"] == REP_EVENT)[0] if F["fips"][u] in two and F["cust"][u] >= 10000]
    yf = np.where(F["obs_full"], F["y_full"], np.nan)
    rise = {u: np.nanmax(yf[u, PUSH[0]:PUSH[1] + 1]) - yf[u, PUSH[0] - 1] for u in cand}
    ranked = sorted(cand, key=lambda u: -rise[u])
    (C.RESULTS / "figure_choice.json").write_text(json.dumps(dict(
        event=REP_EVENT, push_window=PUSH, candidates=[dict(unit=int(u), fips=F["fips"][u], rise=float(rise[u]))
                                                         for u in ranked]), indent=1) + "\n")
    return int(ranked[0]), [int(u) for u in ranked]


def kernel_contribution(m, b, T=216):
    """C[t-72, v]: contribution of the deposit at hour v to the raw damage logit at t."""
    k = m.kernel
    with torch.no_grad():
        out = m(b, diagnostics=True)
        off = m(b, exit_open=False)
    d = out["kernel_deposit"][0].double().numpy()
    lam = out["kernel_damping"][0].double().numpy(); a = out["kernel_interaction"][0].double().numpy()
    nu = float(lam.min()); om = out["kernel_coordinate_gain"][0].double().numpy()
    s_vec = k.ramp() * float(torch.tanh(k.alpha)) * float(k.scale) * om
    W2 = k.weight.double().numpy(); w3 = m.damage[4].weight.double().numpy()[0]
    a2 = out["a2"][0].double().numpy(); eff = out["kernel_effect"][0].double().numpy()
    dW = eff @ W2.T; a2c = a2 - dW
    alphas = (np.arange(64) + 0.5) / 64
    D = len(lam)
    X = np.zeros((D, T)); Cm = np.zeros((T - 72, T)); state = np.zeros(D); states = np.zeros((T, D))
    for t in range(T):
        q = nu * d[t]
        M = np.diag(1 + lam) - (np.outer(a, q) - np.outer(q, a))
        R = np.linalg.inv(M)
        X[:, :t] = R @ X[:, :t]; X[:, t] = R @ q
        state = R @ (state + q); states[t] = state
        if t >= 72:
            act = (a2c[t][None, :] + alphas[:, None] * dW[t][None, :] > 0).mean(0)
            gbar = s_vec * (W2.T @ (w3 * act))
            Cm[t - 72, :t + 1] = gbar @ X[:, :t + 1]
    raw_open = out["raw_logit"][0].double().numpy(); raw_closed = off["raw_logit"][0].double().numpy()
    checks = dict(state_err=float(np.abs(states - out["kernel_state"][0].double().numpy()).max()),
                  ig_completeness_err=float(np.abs(Cm.sum(1) - (raw_open - raw_closed)).max()))
    return Cm, np.linalg.norm(d, axis=1), raw_open, raw_closed, out, off, checks


def drivers(m, b, F, st, window=PUSH, steps=48):
    """Integrated gradients of the window push (sum of raw logit open - closed) over the
    damage inputs (all hours) and geography, from weather at the fitting mean (no
    departures, zero push) and the neutral geographic code."""
    k = m.kernel
    act = dict(xu=b["xu"].clone(), geo=b["geo"].clone())
    g0 = (3 * torch.atanh(k.geo_center.clamp(-0.999, 0.999)))[None]
    base = dict(xu=torch.zeros_like(act["xu"]), geo=g0)
    lo, hi = window[0] - 72, window[1] - 72
    alphas = ((torch.arange(steps, dtype=torch.float32) + 0.5) / steps)

    def push(xu, geo):
        B = xu.shape[0]
        bb = {kk: (v.expand(B, *v.shape[1:]) if torch.is_tensor(v) and v.ndim >= 1 and v.shape[0] == 1 else v)
              for kk, v in b.items() if kk != "idx"}
        bb.update(xu=xu, geo=geo)
        ro = m(bb)["raw_logit"]; rc = m(bb, exit_open=False)["raw_logit"]
        return (ro - rc)[:, lo:hi + 1].sum(1)

    path = {kk: (base[kk] + alphas.view(-1, *([1] * (act[kk].ndim - 1))) * (act[kk] - base[kk])).requires_grad_(True)
            for kk in act}
    tgt = push(path["xu"], path["geo"])
    tgt.sum().backward()
    ig = {kk: ((act[kk] - base[kk]) * path[kk].grad.mean(0, keepdim=True))[0].detach().numpy() for kk in act}
    with torch.no_grad():
        full = float(push(act["xu"], act["geo"])[0]); zero = float(push(base["xu"], base["geo"])[0])
    w_attr = ig["xu"].sum(0)
    return w_attr, ig["geo"], full, zero


GROUPS = {
    "Wind and gusts": ["gust", "wind_speed", "u10", "v10", "gust_excess_energy", "gust_excess_energy_sum6", "path_gust_max",
                       "path_gust_excess_energy", "hours_since_path_gust_max", "gust_max6", "gust_max12", "gust_max24",
                       "wind_speed_mean12"],
    "Wind-driven rain": ["wet_wind", "wet_wind_sum12", "path_wet_wind"],
    "Snow, ice, freezing": ["snowfall", "snow_ice_load", "near_freeze", "cold_precip", "path_snow_ice_load",
                            "subzero_hours24", "zero_crossings24"],
    "Precipitation": ["precip", "precip_sum6", "precip_sum12", "precip_sum24"],
    "Temperature, humidity": ["t2m_c", "rh"],
    "Soil moisture": ["soil_moisture", "soil_moisture_mean24"],
    "Cloud, convection": ["cloud", "cape"],
    "Pressure": ["pressure"],
    "Wind direction, shifts": ["dir_change3", "dir_change6", "dir_sin", "dir_cos"],
    "Time of day": ["clock_sin", "clock_cos"],
    "Terrain relief, slope": ["elev_mean", "relief_p95_p5", "slope_mean_deg", "steep_frac", "ruggedness_tri",
                              "s50_relief_p95_p5"],
    "Slope aspect": [f"aspect_{s}" for s in ("N", "NE", "E", "SE", "S", "SW", "W", "NW")],
    "Forest and canopy": ["canopy_mean", "canopy_dense_frac", "canopy_in_developed", "forest_frac", "forest_on_steep_frac",
                          "s50_canopy_mean"],
    "Windthrow susceptibility": ["windthrow_susceptibility", "s50_windthrow_susceptibility", "root_limiting_share"],
    "Wet, poorly drained soils": ["poorly_drained_share", "hydric_share", "high_water_table_share", "wetland_frac",
                                  "s50_poorly_drained_share"],
    "Shallow soils": ["shallow_share"],
    "Developed land": ["developed_frac"],
    "County area": ["log_land_area_km2"],
}
FEATURE_GROUP = {f: g for g, fs in GROUPS.items() for f in fs}


def figure3(design="main", seed=0, unit=None, tag=""):
    """tag="" -> the PREREG county; otherwise a unit chosen elsewhere (files get the tag)."""
    F = C.load_features(); n = len(F["y"])
    u, ranked = representative_county(F) if unit is None else (int(unit), None)
    fo = fold_of(design)[u]
    mg, stg = load_model(design, seed, fo, "GCRK", F)
    mw, stw = load_model(design, seed, fo, "W", F)
    bg = make_batch(F, np.array([u]), stg); bw = make_batch(F, np.array([u]), stw)
    Cm, dep_norm, raw_open, raw_closed, out, off, checks = kernel_contribution(mg, bg)
    with torch.no_grad():
        pw = mw(bw)["P"][0].numpy()
    w_attr, g_attr, full, zero = drivers(mg, bg, F, stg)
    names = list(F["damage_features"]) + list(F["geo_features"])
    kind = ["weather"] * len(F["damage_features"]) + ["geography"] * len(F["geo_features"])
    vals = np.concatenate([w_attr, g_attr])
    drv = pd.DataFrame(dict(feature=names, kind=kind, attribution=vals, group=[FEATURE_GROUP.get(x, "Other") for x in names]))
    drv["window_push"] = full; drv["baseline_push"] = zero
    FIGDATA.mkdir(parents=True, exist_ok=True)
    drv.to_csv(FIGDATA / f"fig3{tag}_drivers.csv", index=False)
    np.savez_compressed(FIGDATA / f"fig3{tag}_kernel.npz", unit=u, fips=F["fips"][u], event=F["event"][u], fold=int(fo), seed=seed,
                        C=Cm.astype(np.float32), deposit_norm=dep_norm.astype(np.float32), raw_open=raw_open, raw_closed=raw_closed,
                        P_open=out["P"][0].numpy(), P_closed=off["P"][0].numpy(), P_W=pw,
                        y_full=F["y_full"][u], obs_full=F["obs_full"][u], push_window=np.array(PUSH), **checks)
    print("figure 3 county", F["fips"][u], F["event"][u], "fold", fo, checks, "push", full, "baseline", zero,
          "attr sum", vals.sum())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["concentration", "swap", "figure3"])
    ap.add_argument("--design", default="main")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--unit", type=int, default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    {"concentration": lambda: concentration(a.design),
     "swap": lambda: swap(a.design, seeds=None if a.seeds is None else [int(x) for x in a.seeds.split(",")]),
     "figure3": lambda: figure3(a.design, a.seed, a.unit, a.tag)}[a.what]()
