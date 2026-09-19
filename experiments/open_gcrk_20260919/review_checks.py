"""Checks raised by two external reviews of the results (post hoc; no retraining).

  bootstrap   cluster bootstrap of the relative RMSE change GCRK vs W (and closed vs W) from the
              seed-mean squared error of every unit, resampling counties, event x state blocks or
              events; also the effective number of events (inverse Simpson of each event's share of
              the all-zero squared error)
  scaling     each model's forecast multiplied by the oracle scalar k = sum(P y) / sum(P^2) over the
              observed held-out cells (pooled, and per event): does the ordering survive a rescaling?
  thresholds  false-activity share at 0.001, 0.005, 0.01 and 0.02
  alpha       the kernel opening alpha (tanh(alpha) = beta) against the selected step t*, by seed
  memory      trained damping lambda_j per county and coordinate: memory length 1/log(1 + lambda_j)
              and the steady-state gain nu / lambda_j that the shared input scale nu = min_j lambda_j
              leaves each coordinate
  residual    E0: can the 31 (or 40) geographic descriptors predict W's held-out errors? Ridge with
              county- or event-grouped cross-validation; null distribution from permuting the
              descriptors within event x state blocks
  gate        E1 (main design): share of unit-hours with the kernel gate above 0.5, prefix vs forecast
              leads, and the forecast-window deposit norm when one input group is held at its own
              prefix mean for all hours (path summaries, slow thermal/snow/soil, wind, precipitation,
              other)

Writes results/review_*.csv / .json.
"""
from __future__ import annotations

import json
import os
import sys

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
import pandas as pd
import torch

import common as C
import build_features as BF
from diagnostics_frozen import load_cell

sys.path.insert(0, str(C.ROOT / "src"))
from asymode.gcrk_train import make_batch  # noqa: E402

torch.set_num_threads(1)
R = C.RESULTS
RNG = np.random.default_rng(20260919)
B_BOOT, B_PERM = 5000, 200


def unit_sse_all(F, design):
    n = len(F["y"])
    out = {}
    for arm, key, name in (("W", "P", "W"), ("GCRK", "P", "GCRK"), ("GCRK", "P_closed", "closed")):
        out[name] = np.mean([C.unit_sse(C.collect(design, arm, s, n, key), F) for s in C.SEEDS], 0)
    return out


def bootstrap(F, design):
    sse = unit_sse_all(F, design)
    ev = F["event"].astype(str); st = np.array([f[:2] for f in F["fips"].astype(str)])
    labels = {"county": F["fips"].astype(str), "event x state": np.char.add(np.char.add(ev, "|"), st), "event": ev}
    rows = []
    for a, b in (("GCRK", "W"), ("closed", "W"), ("GCRK", "closed")):
        point = float(np.sqrt(sse[a].sum() / sse[b].sum()) - 1)
        for lab, g in labels.items():
            u, inv = np.unique(g, return_inverse=True)
            sa, sb = np.bincount(inv, sse[a]), np.bincount(inv, sse[b])
            w = RNG.multinomial(len(u), np.full(len(u), 1 / len(u)), size=B_BOOT)
            r = np.sqrt((w @ sa) / (w @ sb)) - 1
            rows.append(dict(design=design, contrast=f"{a} vs {b}", cluster=lab, n_clusters=len(u),
                             rel_rmse_change=point, ci_lo=float(np.quantile(r, .025)), ci_hi=float(np.quantile(r, .975)),
                             p_worse=float((r > 0).mean())))
    zero = np.where(F["m"].astype(bool), F["y"].astype(float) ** 2, 0).sum(1)
    share = pd.Series(zero).groupby(ev).sum() / zero.sum()
    eff = dict(design=design, zero_mass_share=share.round(4).to_dict(), effective_events=float(1 / (share ** 2).sum()))
    return pd.DataFrame(rows), eff


def scaling(F, design):
    n = len(F["y"]); m = F["m"].astype(bool); y = F["y"].astype(float); ev = F["event"].astype(str)
    rows = []
    for s in C.SEEDS:
        for arm, key, name in (("W", "P", "W"), ("GCRK", "P", "GCRK"), ("GCRK", "P_closed", "closed")):
            P = C.collect(design, arm, s, n, key)
            k = float((P[m] * y[m]).sum() / (P[m] ** 2).sum())
            rows.append(dict(design=design, seed=s, model=name, scope="pooled", k=k,
                             rmse=float(np.sqrt(((P - y)[m] ** 2).mean())), rmse_scaled=float(np.sqrt(((k * P - y)[m] ** 2).mean())),
                             mean_pred=float(P[m].mean()), mean_obs=float(y[m].mean())))
            se_s = 0.0
            for e in np.unique(ev):
                mm = m & (ev == e)[:, None]
                ke = float((P[mm] * y[mm]).sum() / max((P[mm] ** 2).sum(), 1e-12))
                se_s += float(((ke * P - y)[mm] ** 2).sum())
            rows.append(dict(design=design, seed=s, model=name, scope="per event", k=np.nan,
                             rmse=float(np.sqrt(((P - y)[m] ** 2).mean())), rmse_scaled=float(np.sqrt(se_s / m.sum())),
                             mean_pred=float(P[m].mean()), mean_obs=float(y[m].mean())))
    return pd.DataFrame(rows)


def thresholds(F, design):
    n = len(F["y"]); m = F["m"].astype(bool); y = F["y"].astype(float)
    z = m & (y == 0)
    rows = []
    for s in C.SEEDS:
        for arm, key, name in (("W", "P", "W"), ("GCRK", "P", "GCRK"), ("GCRK", "P_closed", "closed")):
            P = C.collect(design, arm, s, n, key)
            for thr in (0.001, 0.005, 0.01, 0.02):
                rows.append(dict(design=design, seed=s, model=name, threshold=thr, false_activity=float((P[z] > thr).mean())))
    return pd.DataFrame(rows)


def alpha_table(design):
    rows = []
    sp = C.load_splits()[design]
    for s in C.SEEDS:
        for f in sp:
            d = json.loads((C.cell(design, s, f, "GCRK") / "DONE.json").read_text())
            b = d["kernel"]["beta"]
            rows.append(dict(design=design, seed=s, fold=f, t_star=d["best_step"], beta=b,
                             alpha=float(np.arctanh(np.clip(b, -0.999999, 0.999999)))))
    t = pd.DataFrame(rows)
    t["abs_alpha_per_step"] = t.alpha.abs() / t.t_star
    return t


def memory(F, design="main"):
    sp = C.load_splits()[design]
    rows = []
    for s in C.SEEDS:
        for f in sp:
            idx = np.array(sp[f]["outer"])
            model, stats = load_cell(design, s, f, "GCRK", F)
            b = make_batch(F, idx, stats)
            with torch.no_grad():
                lam, _, gain = model.kernel.condition(b["geo"])
            lam = lam.double().numpy()
            tau = 1.0 / np.log1p(lam)                              # memory length, hours
            rel = lam.min(1, keepdims=True) / lam                  # steady-state gain nu / lambda_j
            rows.append(dict(design=design, seed=s, fold=f,
                             tau_min_median=float(np.median(tau.min(1))), tau_max_median=float(np.median(tau.max(1))),
                             tau_ratio_median=float(np.median(tau.max(1) / tau.min(1))),
                             share_coords_gain_below_0p1=float((rel < 0.1).mean()),
                             share_coords_gain_below_0p5=float((rel < 0.5).mean()),
                             median_gain=float(np.median(rel)),
                             tau_spread_across_units=float(np.median(tau.std(0) / tau.mean(0)))))
    return pd.DataFrame(rows)


def residual_test(F, design, geo_sets):
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    n = len(F["y"]); m = F["m"].astype(bool); y = F["y"].astype(float)
    P = np.mean([C.collect(design, "W", s, n) for s in C.SEEDS], 0)
    tgt = {"mean residual (y - P)": np.array([(y[i] - P[i])[m[i]].mean() for i in range(n)]),
           "log peak ratio": np.log((np.where(m, y, -1).max(1) + 0.01) / (np.where(m, P, -1).max(1) + 0.01))}
    ev = F["event"].astype(str); st = np.array([f[:2] for f in F["fips"].astype(str)])
    block = np.char.add(np.char.add(ev, "|"), st)
    groups = F["fips"].astype(str) if design == "main" else ev
    k = 5
    cv = list(GroupKFold(k).split(np.zeros(n), groups=groups))

    def oof_r2(X, t):
        pred = np.empty(n)
        for tr, te in cv:
            mdl = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13)))
            mdl.fit(X[tr], t[tr]); pred[te] = mdl.predict(X[te])
        return float(1 - ((t - pred) ** 2).sum() / ((t - t.mean()) ** 2).sum())

    blocks = [np.where(block == b)[0] for b in np.unique(block)]
    rows = []
    for gname, G in geo_sets.items():
        X = np.where(np.isnan(G), np.nanmedian(G, 0), G)
        for tname, t in tgt.items():
            obs = oof_r2(X, t)
            null = []
            for _ in range(B_PERM):
                perm = np.arange(n)
                for bi in blocks:
                    perm[bi] = RNG.permutation(bi)
                null.append(oof_r2(X[perm], t))
            null = np.array(null)
            rows.append(dict(design=design, cv=("county-grouped" if design == "main" else "event-grouped"),
                             geography=gname, target=tname, oof_r2=obs, null_q95=float(np.quantile(null, .95)),
                             null_median=float(np.median(null)), p_value=float((1 + (null >= obs).sum()) / (1 + B_PERM))))
    return pd.DataFrame(rows)


GROUPS = {
    "path summaries": BF.PATH,
    "slow thermal / snow / soil": ["t2m_c", "rh", "snowfall", "soil_moisture", "near_freeze", "snow_ice_load", "cold_precip",
                                   "subzero_hours24", "zero_crossings24", "soil_moisture_mean24"],
    "wind": ["gust", "u10", "v10", "wind_speed", "gust_excess_energy", "gust_excess_energy_sum6", "gust_max6", "gust_max12",
             "gust_max24", "wind_speed_mean12", "dir_change3", "dir_change6", "dir_sin", "dir_cos"],
    "precipitation": ["precip", "precip_sum6", "precip_sum12", "precip_sum24", "wet_wind", "wet_wind_sum12"],
    "other": ["cape", "cloud", "pressure", "clock_sin", "clock_cos"],
}


def gate_usage(F, design="main"):
    names = list(BF.DAMAGE_FEATURES)
    assert sorted(sum(GROUPS.values(), [])) == sorted(names), "feature groups must partition the damage inputs"
    sp = C.load_splits()[design]
    lead = np.arange(216) - 71
    segs = {"prefix 1-71": (lead >= -70) & (lead <= 0), "lead 1-6": (lead >= 1) & (lead <= 6),
            "lead 7-24": (lead >= 7) & (lead <= 24), "lead 25-48": (lead >= 25) & (lead <= 48),
            "lead 49-144": (lead >= 49) & (lead <= 144)}
    occ_rows, dep_rows = [], []
    for s in C.SEEDS:
        for f in sp:
            idx = np.array(sp[f]["outer"])
            model, stats = load_cell(design, s, f, "GCRK", F)
            b0 = make_batch(F, idx, stats)
            with torch.no_grad():
                _, kd = model.kernel(model.hidden(b0["xu"]), b0["geo"], diagnostics=True)
                gate = kd["gate"].numpy(); dn = torch.linalg.vector_norm(kd["deposit"], dim=-1).numpy()
                for lab, mk in segs.items():
                    occ_rows.append(dict(seed=s, fold=f, segment=lab, gate_above_half=float((gate[:, mk] > 0.5).mean()),
                                         mean_deposit_norm=float(dn[:, mk].mean())))
                base = dn[:, 72:].mean()
                for g, cols in GROUPS.items():
                    xu = F["xu"][idx].astype(np.float64).copy()
                    j = [names.index(c) for c in cols]
                    xu[:, :, j] = xu[:, :72, j].mean(1, keepdims=True)
                    Fo = {kk: (F[kk][idx] if kk != "xu" else xu.astype(np.float32))
                          for kk in ("xu", "xr", "xo", "geo", "y0", "y", "m")}
                    bo = make_batch(Fo, np.arange(len(idx)), stats)
                    _, ko = model.kernel(model.hidden(bo["xu"]), bo["geo"], diagnostics=True)
                    do = torch.linalg.vector_norm(ko["deposit"], dim=-1).numpy()[:, 72:].mean()
                    dep_rows.append(dict(seed=s, fold=f, group=g, n_features=len(cols), base_deposit=float(base),
                                         occluded_deposit=float(do), reduction_share=float(1 - do / base)))
    return pd.DataFrame(occ_rows), pd.DataFrame(dep_rows)


def main():
    F = C.load_features()
    G40 = np.load(C.ROOT / "data" / "interim" / "open_gcrk" / "features_geo40.npz")["geo"].astype(np.float64)
    out = {}
    for design in ("main", "loeo"):
        bt, eff = bootstrap(F, design)
        out[f"bootstrap_{design}"] = bt
        (R / f"review_effective_events_{design}.json").write_text(json.dumps(eff, indent=1) + "\n")
        out[f"scaling_{design}"] = scaling(F, design)
        out[f"thresholds_{design}"] = thresholds(F, design)
        out[f"alpha_{design}"] = alpha_table(design)
        out[f"residual_{design}"] = residual_test(F, design, {"31 descriptors": F["geo"].astype(np.float64), "40 descriptors": G40})
        print(f"{design}: bootstrap, scaling, thresholds, alpha, residual done", flush=True)
    out["memory_main"] = memory(F, "main")
    occ, dep = gate_usage(F, "main")
    out["gate_occupancy_main"], out["gate_occlusion_main"] = occ, dep
    for k, v in out.items():
        v.to_csv(R / f"review_{k}.csv", index=False)
    pd.set_option("display.width", 220)
    for design in ("main", "loeo"):
        print(f"\n== {design}: cluster bootstrap (seed-mean unit SSE)\n", out[f"bootstrap_{design}"].round(4).to_string(index=False))
        print(json.loads((R / f"review_effective_events_{design}.json").read_text()))
        sc = out[f"scaling_{design}"].groupby(["scope", "model"])[["k", "rmse", "rmse_scaled", "mean_pred", "mean_obs"]].mean()
        print(f"\n== {design}: oracle scalar rescaling (mean over seeds)\n", sc.round(5).to_string())
        th = out[f"thresholds_{design}"].pivot_table(index="threshold", columns="model", values="false_activity")
        print(f"\n== {design}: false activity by threshold\n", th.round(4).to_string())
        a = out[f"alpha_{design}"]
        print(f"\n== {design}: alpha: mean |alpha|/t* {a.abs_alpha_per_step.mean():.2e}; corr(|alpha|, t*) "
              f"{np.corrcoef(a.alpha.abs(), a.t_star)[0, 1]:+.2f}; sign by seed "
              f"{a.groupby('seed').alpha.apply(lambda x: ''.join('+' if v > 0 else '-' for v in x)).to_dict()}")
        print(f"\n== {design}: E0 residual information\n", out[f"residual_{design}"].round(4).to_string(index=False))
    print("\n== main: trained memory lengths (median over cells)\n", out["memory_main"].drop(columns=["seed", "fold", "design"]).median().round(4).to_string())
    print("\n== main: kernel gate occupancy\n", occ.groupby("segment")[["gate_above_half", "mean_deposit_norm"]].mean().round(4).to_string())
    print("\n== main: deposit reduction when one input group is held at its prefix mean\n",
          dep.groupby("group")[["reduction_share"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
