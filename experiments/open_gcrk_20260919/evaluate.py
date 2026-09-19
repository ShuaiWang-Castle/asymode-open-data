"""Result tables for the open-data W vs GCRK campaign (PREREG sections 8-9).

Writes to results/:
  table_main_<design>.csv        rows all-zero / persistence / TimesFM / TimesFM (history) / W / GCRK;
                                 W and GCRK as mean and sd over seeds
  seeds_<design>.csv             every seed's W, GCRK and GCRK with the kernel exit closed, all metrics
  paired_<design>.csv            per metric: seed-wise GCRK - W, relative %, sign count, rule verdict
  decomposition_<design>.csv     GCRK - W split into (closed - W) + (open - closed), per seed and mean
  events_<design>.csv            full-rollout RMSE and MAE by event for every row
  pathology_<design>.csv         per cell: selected step, stop step, kernel opening, flags
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import common as C

METRICS = ["rmse", "rmse 1-6 h", "rmse 7-24 h", "rmse 25-48 h", "rmse 49-144 h", "mae",
           "peak_mag_abs_mean", "peak_time_abs_mean", "false_activity_share", "under_half_share"]


def verdict(d: np.ndarray) -> str:
    neg, pos = int((d < 0).sum()), int((d > 0).sum())
    if d.mean() < 0 and neg >= 4:
        return "GCRK better (5/5)" if neg == 5 else "GCRK better (4/5)"
    if d.mean() > 0 and pos >= 4:
        return "GCRK worse (5/5)" if pos == 5 else "GCRK worse (4/5)"
    return "no consistent difference"


def pathology(design: str) -> pd.DataFrame:
    rows = []
    sp = C.load_splits()[design]
    for s in C.SEEDS:
        for f in sp:
            for arm in ("W", "GCRK"):
                p = C.cell(design, s, f, arm) / "DONE.json"
                if not p.exists():
                    continue
                d = json.loads(p.read_text())
                k = d.get("kernel") or {}
                rows.append(dict(seed=s, fold=int(f), arm=arm, best_step=d["best_step"], stop_step=d["stop_step"],
                                 best_inner_mse=d["best_inner_mse"], beta=k.get("beta"), scale=k.get("scale"),
                                 theta=k.get("theta"), seconds=round(d["seconds"]),
                                 flag_step_edge=d["best_step"] in (0, 1600),
                                 flag_kernel_closed=(arm == "GCRK" and abs(k.get("beta", 1.0)) < 1e-3)))
    return pd.DataFrame(rows)


def main(design: str = "main"):
    F = C.load_features()
    n = len(F["y"])
    C.RESULTS.mkdir(exist_ok=True)
    rows, seed_rows = [], []
    base = {"all-zero": np.zeros((n, 144)), "persistence": C.persistence(F),
            "TimesFM": C.timesfm(F, "WEATHER"), "TimesFM (history only)": C.timesfm(F, "HISTORY")}
    for name, P in base.items():
        if P is not None:
            rows.append(dict(model=name, **C.metrics(P, F)))
    preds = {}
    for arm, key in (("W", "P"), ("GCRK", "P"), ("GCRK exit closed", "P_closed")):
        for s in C.SEEDS:
            P = C.collect(design, "GCRK" if arm.startswith("GCRK") else "W", s, n, key)
            if P is None:
                continue
            preds[(arm, s)] = P
            seed_rows.append(dict(model=arm, seed=s, **C.metrics(P, F)))
    sd = pd.DataFrame(seed_rows, columns=["model", "seed"] + (list(seed_rows[0].keys())[2:] if seed_rows else []))
    sd.to_csv(C.RESULTS / f"seeds_{design}.csv", index=False)
    for arm in ("W", "GCRK", "GCRK exit closed"):
        g = sd[sd.model == arm]
        if len(g):
            rows.append(dict(model=arm, n_seeds=len(g), **{k: g[k].mean() for k in g.columns if k not in ("model", "seed")},
                             **{f"{k} sd": g[k].std(ddof=1) for k in METRICS}))
    pd.DataFrame(rows).to_csv(C.RESULTS / f"table_main_{design}.csv", index=False)
    seeds = sorted(s for (a, s) in preds if a == "GCRK" and ("W", s) in preds)
    if not seeds:
        print("no complete paired seeds yet"); return
    w = sd[sd.model == "W"].set_index("seed").loc[seeds]
    g = sd[sd.model == "GCRK"].set_index("seed").loc[seeds]
    c = sd[sd.model == "GCRK exit closed"].set_index("seed").loc[seeds]
    pr = []
    for k in METRICS:
        d = (g[k] - w[k]).to_numpy()
        pr.append(dict(metric=k, W_mean=w[k].mean(), GCRK_mean=g[k].mean(), diff_mean=d.mean(), diff_sd=d.std(ddof=1) if len(d) > 1 else np.nan,
                       rel_pct_mean=float((100 * (g[k] / w[k] - 1)).mean()), n_seeds=len(d),
                       n_gcrk_lower=int((d < 0).sum()), per_seed_diff=";".join(f"{x:+.3e}" for x in d),
                       verdict=verdict(d) if len(d) == 5 else "incomplete"))
    pd.DataFrame(pr).to_csv(C.RESULTS / f"paired_{design}.csv", index=False)
    dec = []
    for s in seeds:
        dec.append(dict(seed=s, W=w.loc[s, "rmse"], closed=c.loc[s, "rmse"], open=g.loc[s, "rmse"],
                        joint_training=c.loc[s, "rmse"] - w.loc[s, "rmse"], kernel_output=g.loc[s, "rmse"] - c.loc[s, "rmse"],
                        total=g.loc[s, "rmse"] - w.loc[s, "rmse"]))
    dec = pd.DataFrame(dec)
    dec.loc[len(dec)] = dict(seed="mean", **{k: dec[k].mean() for k in dec.columns if k != "seed"})
    dec.to_csv(C.RESULTS / f"decomposition_{design}.csv", index=False)
    ev_rows = []
    for ev in sorted(set(F["event"].tolist())):
        u = np.where(F["event"] == ev)[0]
        for name, P in base.items():
            if P is not None:
                ev_rows.append(dict(event=ev, model=name, n_units=len(u), **{k: v for k, v in C.metrics(P, F, u).items()
                                                                              if k in ("rmse", "mae")}))
        for arm in ("W", "GCRK"):
            vals = [C.metrics(preds[(arm, s)], F, u) for s in seeds]
            ev_rows.append(dict(event=ev, model=arm, n_units=len(u), rmse=np.mean([v["rmse"] for v in vals]),
                                rmse_sd=np.std([v["rmse"] for v in vals], ddof=1), mae=np.mean([v["mae"] for v in vals]),
                                gcrk_lower_seeds=None))
        d = np.array([C.metrics(preds[("GCRK", s)], F, u)["rmse"] - C.metrics(preds[("W", s)], F, u)["rmse"] for s in seeds])
        ev_rows.append(dict(event=ev, model="GCRK - W", n_units=len(u), rmse=d.mean(), rmse_sd=d.std(ddof=1),
                            gcrk_lower_seeds=int((d < 0).sum())))
    pd.DataFrame(ev_rows).to_csv(C.RESULTS / f"events_{design}.csv", index=False)
    pathology(design).to_csv(C.RESULTS / f"pathology_{design}.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(pd.DataFrame(rows)[["model", "rmse", "rmse 1-6 h", "rmse 7-24 h", "rmse 25-48 h", "rmse 49-144 h", "mae"]].to_string())
        print(pd.DataFrame(pr)[["metric", "W_mean", "GCRK_mean", "rel_pct_mean", "n_gcrk_lower", "verdict"]].to_string())
        print(dec.to_string())


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "main")
