"""Diagnostics of PREREG Amendment 4 (no retraining of W or GCRK). Run with OPEN_GCRK_ROUND=e3r2.

  D1  calibration of the host as a conditional mean (units by predicted peak; county-hours by predicted p)
  D2  by event: GCRK - W, prefix-to-forecast wind-report ratio, kernel deposit norm and gate occupancy
  D3  information ceiling: gradient-boosted regressor on unit-level summaries, without / with geography /
      with geography permuted within event x state blocks, on the E3 county- and event-grouped outer folds

Writes results/<round>/review2_*.csv and prints a summary.
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
from diagnostics_frozen import load_cell

sys.path.insert(0, str(C.ROOT / "src"))
from asymode.gcrk_train import make_batch  # noqa: E402

torch.set_num_threads(1)
RNG = np.random.default_rng(20260920)
N_PERM, N_BOOT = 20, 2000
SEED = 0


def unit_targets(F):
    m = F["m"].astype(bool); y = F["y"].astype(np.float64)
    peak = np.where(m, y, -np.inf).max(1)
    mean = np.where(m, y, 0).sum(1) / np.maximum(m.sum(1), 1)
    return peak, mean


def implied(P, F):
    m = F["m"].astype(bool)
    return np.where(m, P, -np.inf).max(1), np.where(m, P, 0).sum(1) / np.maximum(m.sum(1), 1)


def d1(F):
    n = len(F["y"]); m = F["m"].astype(bool); y = F["y"].astype(np.float64)
    peak, _ = unit_targets(F)
    rows, cells = [], []
    for design in ("main", "event"):
        for arm in ("W", "GCRK"):
            P = C.collect(design, arm, SEED, n)
            if P is None:
                continue
            pp, _ = implied(P, F)
            q = pd.qcut(pp, 10, labels=False, duplicates="drop")
            for b in np.unique(q):
                k = q == b
                rows.append(dict(design=design, model=arm, bin=int(b), n=int(k.sum()), pred_peak=float(pp[k].mean()),
                                 obs_peak=float(peak[k].mean()), obs_below_1pct=float((peak[k] < 0.01).mean()),
                                 obs_above_10pct=float((peak[k] > 0.10).mean()), obs_peak_median=float(np.median(peak[k])),
                                 obs_peak_p90=float(np.quantile(peak[k], .9))))
            edges = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 1.01]
            for lo, hi in zip(edges[:-1], edges[1:]):
                k = m & (P >= lo) & (P < hi)
                if k.sum():
                    cells.append(dict(design=design, model=arm, pred_lo=lo, pred_hi=hi, n_cells=int(k.sum()),
                                      pred_mean=float(P[k].mean()), obs_mean=float(y[k].mean()),
                                      obs_zero_share=float((y[k] == 0).mean()), obs_above_pred_x3=float((y[k] > 3 * P[k]).mean())))
    return pd.DataFrame(rows), pd.DataFrame(cells)


def d2(F):
    n = len(F["y"]); ev = F["event"].astype(str)
    Pw, Pg = C.collect("main", "W", SEED, n), C.collect("main", "GCRK", SEED, n)
    cand = pd.read_csv(C.HERE / "event_selection.csv").set_index("day")
    dep = np.full(n, np.nan); occ = np.full(n, np.nan); dep_pre = np.full(n, np.nan)
    sp = C.load_splits()["main"]
    for f in sp:
        idx = np.array(sp[f]["outer"])
        model, stats = load_cell("main", SEED, f, "GCRK", F)
        b = make_batch(F, idx, stats)
        with torch.no_grad():
            _, kd = model.kernel(model.hidden(b["xu"]), b["geo"], diagnostics=True)
        dn = torch.linalg.vector_norm(kd["deposit"], dim=-1).numpy(); g = kd["gate"].numpy()
        dep[idx] = dn[:, 72:].mean(1); occ[idx] = (g[:, 72:] > 0.5).mean(1); dep_pre[idx] = dn[:, 1:72].mean(1)
    rows = []
    for e in sorted(set(ev)):
        u = np.where(ev == e)[0]
        w, g = C.metrics(Pw, F, u)["rmse"], C.metrics(Pg, F, u)["rmse"]
        rows.append(dict(event=e, units=len(u), W=w, GCRK=g, rel_change=g / w - 1,
                         prefix_to_fc_ratio=float(cand.loc[e, "prefix_to_fc_ratio"]), calm_prefix=bool(cand.loc[e, "F3_calm_prefix"]),
                         deposit_forecast=float(np.nanmean(dep[u])), deposit_prefix=float(np.nanmean(dep_pre[u])),
                         gate_open_share=float(np.nanmean(occ[u]))))
    return pd.DataFrame(rows)


def unit_features(F):
    xu = F["xu"].astype(np.float64); xr = F["xr"].astype(np.float64)
    fw = np.concatenate([xu[:, 72:].max(1), xu[:, 72:].mean(1), xu[:, :72].mean(1)], 1)          # 3 x 42
    nb = xr[:, 72:, 25:33].max(1)                                                               # 8 neighbour summaries
    ctx = xr[:, 0, 14:25]                                                                        # 6 context + 5 prefix outage
    return np.concatenate([fw, nb, ctx], 1)


def d3(F):
    from sklearn.ensemble import HistGradientBoostingRegressor
    n = len(F["y"]); ev = F["event"].astype(str); fips = F["fips"].astype(str)
    st = np.array([f[:2] for f in fips]); block = np.char.add(np.char.add(ev, "|"), st)
    blocks = [np.where(block == b)[0] for b in np.unique(block)]
    X0 = unit_features(F); G = F["geo"].astype(np.float64)
    peak, mean = unit_targets(F)
    targets = {"peak": peak, "mean": mean}
    u_c, inv_c = np.unique(fips, return_inverse=True)

    def fit_predict(X, t, folds, log):
        tt = np.log(t + 0.002) if log else t
        pred = np.full(n, np.nan)
        for spec in folds.values():
            dev, out = np.array(spec["dev"]), np.array(spec["outer"])
            mdl = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
                                                l2_regularization=1.0, early_stopping=False, random_state=0)
            mdl.fit(X[dev], tt[dev]); pred[out] = mdl.predict(X[out])
        return pred

    def score(pred, t, log):
        tt = np.log(t + 0.002) if log else t
        se = (pred - tt) ** 2
        return float(np.sqrt(se.mean())), float(1 - se.sum() / ((tt - tt.mean()) ** 2).sum()), se

    rows = []
    sp = C.load_splits()
    for design in ("main", "event"):
        folds = sp[design]
        ref = {}
        for arm in ("W", "GCRK"):
            P = C.collect(design, arm, SEED, n)
            if P is not None:
                ref[arm] = dict(zip(("peak", "mean"), implied(P, F)))
        for tname, t in targets.items():
            for log in (False, True):
                p0 = fit_predict(X0, t, folds, log)
                p1 = fit_predict(np.concatenate([X0, G], 1), t, folds, log)
                r0, q0, se0 = score(p0, t, log); r1, q1, se1 = score(p1, t, log)
                null = []
                for _ in range(N_PERM):
                    perm = np.arange(n)
                    for bi in blocks:
                        perm[bi] = RNG.permutation(bi)
                    null.append(score(fit_predict(np.concatenate([X0, G[perm]], 1), t, folds, log), t, log)[0])
                null = np.array(null)
                d = np.bincount(inv_c, se1 - se0); wts = RNG.multinomial(len(u_c), np.full(len(u_c), 1 / len(u_c)), size=N_BOOT)
                a0, a1 = np.bincount(inv_c, se0), np.bincount(inv_c, se1)
                boot = np.sqrt((wts @ a1) / (wts @ a0)) - 1
                row = dict(design=design, target=tname, scale="log" if log else "raw", rmse_no_geo=r0, r2_no_geo=q0,
                           rmse_geo=r1, r2_geo=q1, rel_change=r1 / r0 - 1, ci_lo=float(np.quantile(boot, .025)),
                           ci_hi=float(np.quantile(boot, .975)), null_rmse_median=float(np.median(null)),
                           null_rmse_p05=float(np.quantile(null, .05)), beats_null=bool(r1 < np.quantile(null, .05)))
                for arm, d_ in ref.items():
                    rr, qq, _ = score(np.log(d_[tname] + 0.002) if log else d_[tname], t, log)
                    row[f"rmse_{arm}"] = rr; row[f"r2_{arm}"] = qq
                rows.append(row)
                print(design, tname, "log" if log else "raw", "done", flush=True)
    return pd.DataFrame(rows)


def main():
    F = C.load_features()
    out = C.RESULTS; out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 230)
    units, cells = d1(F)
    units.to_csv(out / "review2_calibration_units.csv", index=False); cells.to_csv(out / "review2_calibration_cells.csv", index=False)
    print("== D1 units by predicted peak (main design, W)\n", units[(units.design == "main") & (units.model == "W")].drop(columns=["design", "model"]).round(4).to_string(index=False))
    print("\n== D1 county-hours by predicted p (main design, W)\n", cells[(cells.design == "main") & (cells.model == "W")].drop(columns=["design", "model"]).round(4).to_string(index=False))
    ev = d2(F)
    ev.to_csv(out / "review2_events.csv", index=False)
    print("\n== D2 by event (main design, seed 0)\n", ev.round(4).to_string(index=False))
    from scipy.stats import spearmanr
    for col in ("prefix_to_fc_ratio", "deposit_forecast", "gate_open_share"):
        r = spearmanr(ev[col], ev.rel_change)
        print(f"spearman(rel_change, {col}) = {r.statistic:+.2f} (p = {r.pvalue:.2f}, n = {len(ev)})")
    t = d3(F)
    t.to_csv(out / "review2_information.csv", index=False)
    print("\n== D3 information ceiling\n", t.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
