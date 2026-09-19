"""Frozen-checkpoint diagnostics of the trained W and GCRK models (no retraining).

Every cell's REFIT checkpoint (`runs/.../final.pt`) is reloaded and replayed on its own OUTER
units; the replay must reproduce the exported forecasts (checked). From the replay:

  segments      squared error by observed regime: true zero (y_t = 0), rise (y_t - y_{t-1} > 0.001),
                fall (< -0.001), other active hours
  pathways      where outage predicted in true-zero hours comes from: the same model's rollout with
                the background rate removed, and with only the background rate (recovery unchanged)
  reach         one-step reachability on observed rises: the damage rate an observed step needs,
                u_req = (y_t - y_{t-1} + r_t y_{t-1}) / (1 - y_{t-1}) with the model's own r_t, against
                the most the model can supply at that hour, u_max = b_t + 0.5 pi_t (gate pi_t, background b_t)
  origin        GCRK only: every OUTER unit's weather held at its hour-71 value for all 216 hours;
                kernel effect (exit open - closed) on the damage logit and on p, with the path summaries
                as trained (accumulated from hour 72, zero before) and accumulated from hour 0 instead
  inner         best pooled INNER MSE of GCRK vs W in each cell (from selection_trace.csv)

Writes results/frozen_*_<design>.csv and prints a summary.
"""
from __future__ import annotations

import os
import sys

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
import pandas as pd
import torch

import common as C
import build_features as BF

sys.path.insert(0, str(C.ROOT / "src"))
from asymode.asym_host import AsymODE, U_CAP, BKG_CAP  # noqa: E402
from asymode.gcrk_train import make_batch  # noqa: E402

torch.set_num_threads(1)
EPS = 0.001


def load_cell(design, seed, fold, arm, F):
    snap = torch.load(C.cell(design, seed, fold, arm) / "final.pt", weights_only=False)
    m = AsymODE(F["xu"].shape[-1], F["xr"].shape[-1], F["xo"].shape[-1])
    if arm == "GCRK":
        m.attach_gcrk(torch.zeros(F["geo"].shape[-1]))
    m.load_state_dict(snap["model_state"])
    m.eval()
    return m, snap["stats"]


def stock(u, r, y0):
    p, out = y0.astype(np.float64).copy(), np.empty_like(u, dtype=np.float64)
    for s in range(u.shape[1]):
        p = np.clip(p + u[:, s] * (1 - p) - r[:, s] * p, 0, 1)
        out[:, s] = p
    return out


def synthetic_const(F, idx, origin):
    """Weather frozen at hour 71 for all hours; path summaries accumulated from `origin`."""
    X = F["xu"][idx][:, :, :len(BF.CH)].astype(np.float64)
    Xc = np.repeat(X[:, 71:72], BF.T, 1)
    keep = BF.ORIGIN
    BF.ORIGIN = origin
    try:
        wf = BF.weather_features(Xc)
    finally:
        BF.ORIGIN = keep
    xu = np.concatenate([Xc, np.stack([wf[k] for k in BF.HAZARD + BF.PATH + BF.PAST + BF.FREEZE + BF.DIRECTION], -1)], -1)
    xo = np.stack([wf[k] for k in BF.HAZARD], -1)
    return dict(xu=xu.astype(np.float32), xr=F["xr"][idx], xo=xo.astype(np.float32), geo=F["geo"][idx],
                y0=F["y0"][idx], y=F["y"][idx], m=F["m"][idx])


def run(design):
    F = C.load_features()
    sp = C.load_splits()[design]
    seg_rows, path_rows, reach_rows, orig_rows, inner_rows = [], [], [], [], []
    for seed in C.SEEDS:
        acc = {a: {k: [] for k in ("P", "u", "r", "gate", "bkg", "cond", "idx")} for a in ("W", "GCRK", "GCRK closed")}
        for fold in sp:
            idx = np.array(sp[fold]["outer"])
            best = {}
            for arm in ("W", "GCRK"):
                model, stats = load_cell(design, seed, fold, arm, F)
                b = make_batch(F, idx, stats)
                ref = np.load(C.cell(design, seed, fold, arm) / "outer.npz")
                order = np.argsort(ref["idx"]); pos = np.searchsorted(ref["idx"][order], idx)
                with torch.no_grad():
                    for name, kw in ((arm, {}),) + ((("GCRK closed", dict(exit_open=False)),) if arm == "GCRK" else ()):
                        o = model(b, **kw)
                        key = "P" if name != "GCRK closed" else "P_closed"
                        err = float(np.abs(o["P"].numpy() - ref[key][order][pos]).max())
                        assert err < 1e-5, (design, seed, fold, name, err)
                        for k, v in (("P", o["P"]), ("u", o["u"]), ("r", o["r"]), ("gate", o["gate"]),
                                     ("bkg", o["background"]), ("cond", o["conditional"])):
                            acc[name][k].append(v.numpy().astype(np.float64))
                        acc[name]["idx"].append(idx)
                    if arm == "GCRK":
                        for origin, label in ((BF.ORIGIN, "path from hour 72 (as trained)"), (0, "path from hour 0")):
                            s = make_batch(synthetic_const(F, idx, origin), np.arange(len(idx)), stats)
                            on, off = model(s), model(s, exit_open=False)
                            dl = (on["raw_logit"] - off["raw_logit"]).numpy()
                            dp = (on["P"] - off["P"]).numpy()
                            orig_rows.append(dict(seed=seed, fold=fold, variant=label, n_units=len(idx),
                                                  mean_dlogit=float(dl.mean()), mean_abs_dlogit=float(np.abs(dl).mean()),
                                                  mean_dlogit_h72_95=float(dl[:, :24].mean()),
                                                  mean_dlogit_h168_215=float(dl[:, 96:].mean()),
                                                  mean_dp=float(dp.mean()), max_dp=float(dp.max()),
                                                  share_units_dp_gt_1pp=float((dp.max(1) > 0.01).mean()),
                                                  mean_p_open=float(on["P"].numpy().mean()),
                                                  mean_p_closed=float(off["P"].numpy().mean())))
                        a = pd.read_csv(C.cell(design, seed, fold, "W") / "selection_trace.csv")
                        g = pd.read_csv(C.cell(design, seed, fold, "GCRK") / "selection_trace.csv")
                        inner_rows.append(dict(seed=seed, fold=fold, W_best=float(a.pooled_inner_mse.min()),
                                               GCRK_best=float(g.pooled_inner_mse.min()),
                                               W_step=int(a.best_step.iloc[-1]), GCRK_step=int(g.best_step.iloc[-1])))
        y, m = F["y"].astype(np.float64), F["m"].astype(bool)
        y0 = F["y0"].astype(np.float64)
        for name, d in acc.items():
            idx = np.concatenate(d["idx"])
            A = {k: np.concatenate(v) for k, v in d.items() if k != "idx"}
            Y, M, Y0 = y[idx], m[idx], y0[idx]
            prev = np.concatenate([Y0[:, None], Y[:, :-1]], 1)
            prev_ok = np.concatenate([np.ones((len(idx), 1), bool), M[:, :-1]], 1)
            dy = Y - prev
            zero = M & (Y == 0)
            rise = M & prev_ok & (dy > EPS)
            fall = M & prev_ok & (dy < -EPS)
            other = M & ~zero & ~rise & ~fall
            se = (A["P"] - Y) ** 2
            tot = se[M].sum()
            for lab, mk in (("true zero", zero), ("rise", rise), ("fall", fall), ("other active", other)):
                seg_rows.append(dict(seed=seed, model=name, segment=lab, n_cells=int(mk.sum()),
                                     sse=float(se[mk].sum()), sse_share=float(se[mk].sum() / tot),
                                     mean_pred=float(A["P"][mk].mean()), mean_obs=float(Y[mk].mean())))
            capu = U_CAP + BKG_CAP
            p_nobkg = stock(np.clip(A["gate"] * A["cond"], 0, capu), A["r"], Y0)
            p_bkg = stock(np.clip(A["bkg"], 0, capu), A["r"], Y0)
            fa = lambda P: float((P[zero] > EPS).mean())
            path_rows.append(dict(seed=seed, model=name, false_activity=fa(A["P"]),
                                  false_activity_no_background=fa(p_nobkg), false_activity_background_only=fa(p_bkg),
                                  zero_mass=float(A["P"][zero].sum()), zero_mass_no_background=float(p_nobkg[zero].sum()),
                                  zero_mass_background_only=float(p_bkg[zero].sum()),
                                  mean_bkg_zero=float(A["bkg"][zero].mean()),
                                  mean_gated_zero=float((A["gate"] * A["cond"])[zero].mean()),
                                  mean_gate_zero=float(A["gate"][zero].mean()),
                                  mean_r_zero_active=float(A["r"][zero & (A["P"] > EPS)].mean()),
                                  mean_r_all=float(A["r"][M].mean())))
            ok = rise & (prev < 1)
            ureq = (Y - prev + A["r"] * prev) / np.clip(1 - prev, 1e-9, None)
            umax = np.minimum(A["bkg"] + 0.5 * A["gate"], capu)
            umax1 = np.minimum(A["bkg"] + 0.5, capu)
            unr = ok & (ureq > umax + 1e-9)
            big = ok & (dy > 0.01)
            reach_rows.append(dict(seed=seed, model=name, n_rise=int(ok.sum()), n_big_rise=int(big.sum()),
                                   unreachable_share=float(unr[ok].mean()),
                                   unreachable_share_big=float(unr[big].mean()) if big.any() else np.nan,
                                   unreachable_rise_mass_share=float(dy[unr].sum() / dy[ok].sum()),
                                   gate_limited_share=float((unr & (ureq <= umax1 + 1e-9))[ok].mean()),
                                   mean_gate_rise=float(A["gate"][ok].mean()),
                                   mean_gate_big_rise=float(A["gate"][big].mean()) if big.any() else np.nan,
                                   mean_sigma_big_rise=float((A["cond"][big] / U_CAP).mean()) if big.any() else np.nan,
                                   mean_u_over_ureq_big=float(np.median(A["u"][big] / np.clip(ureq[big], 1e-9, None)))
                                   if big.any() else np.nan))
        print(f"{design} seed {seed} done", flush=True)
    R = C.RESULTS
    out = {}
    for name, rows in (("segments", seg_rows), ("pathways", path_rows), ("reach", reach_rows),
                       ("origin", orig_rows), ("inner", inner_rows)):
        df = pd.DataFrame(rows)
        df.to_csv(R / f"frozen_{name}_{design}.csv", index=False)
        out[name] = df
    return out


def summary(design, out):
    pd.set_option("display.width", 200)
    s = out["segments"].groupby(["model", "segment"])[["sse", "sse_share", "mean_pred", "mean_obs"]].mean()
    print(f"\n== {design}: squared error by segment (mean over seeds)\n", s.round(5).to_string())
    print(f"\n== {design}: false-activity pathways (mean over seeds)\n",
          out["pathways"].groupby("model").mean(numeric_only=True).drop(columns="seed").round(4).T.to_string())
    print(f"\n== {design}: one-step reachability on observed rises (mean over seeds)\n",
          out["reach"].groupby("model").mean(numeric_only=True).drop(columns="seed").round(4).T.to_string())
    o = out["origin"]
    w = o.groupby(["variant"]).apply(lambda g: pd.Series({c: np.average(g[c], weights=g.n_units)
                                                           for c in o.columns if c.startswith(("mean", "share"))}))
    print(f"\n== {design}: kernel effect under constant weather (open - closed; unit-weighted over cells)\n",
          w.round(5).T.to_string())
    i = out["inner"]
    print(f"\n== {design}: best pooled INNER MSE, GCRK lower than W in {int((i.GCRK_best < i.W_best).sum())}/{len(i)} cells")


if __name__ == "__main__":
    for design in (sys.argv[1:] or ["main", "loeo"]):
        summary(design, run(design))
