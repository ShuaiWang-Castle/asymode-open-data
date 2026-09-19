"""Twelve-event round (PREREG Amendment 3): primary contrast and diagnostics over the seeds finished.

Run with OPEN_GCRK_ROUND set to the round (e3r1 or e3r2). For each design (main, event):
  per seed     W, GCRK, GCRK exit closed: pooled RMSE, lead segments, MAE, event-equal-weighted RMSE
               (mean over events of each event's RMSE), false activity at 0.001 / 0.005, oracle scale
  bootstrap    relative RMSE change GCRK vs W and exit closed vs W from the seed-mean unit squared
               error, resampling counties, event x state blocks and events (5000 draws)
  reading      Amendment 3: "GCRK better/worse" only if the county interval excludes zero and at
               least four of five seeds agree; otherwise "not distinguishable" ("incomplete" before
               five seeds)
Writes results/<round>/e3_{seeds,bootstrap,verdict}_<design>.csv/json.
"""
from __future__ import annotations

import json
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
import pandas as pd

import common as C

RNG = np.random.default_rng(20260919)


def event_equal_rmse(P, F):
    ev = F["event"].astype(str)
    return float(np.mean([C.metrics(P, F, np.where(ev == e)[0])["rmse"] for e in np.unique(ev)]))


def main():
    F = C.load_features()
    n = len(F["y"]); m = F["m"].astype(bool); y = F["y"].astype(float); z = m & (y == 0)
    ev = F["event"].astype(str); st = np.array([f[:2] for f in F["fips"].astype(str)])
    out = C.RESULTS
    out.mkdir(parents=True, exist_ok=True)
    zero = np.zeros((n, 144))
    base = dict(zero=C.metrics(zero, F)["rmse"], zero_event_equal=event_equal_rmse(zero, F),
                persistence=C.metrics(C.persistence(F), F)["rmse"])
    for design in ("main", "event"):
        rows, sse = [], {"W": [], "GCRK": [], "closed": []}
        seeds = []
        for s in C.SEEDS:
            Pw, Pg = C.collect(design, "W", s, n), C.collect(design, "GCRK", s, n)
            if Pw is None or Pg is None:
                continue
            Pc = C.collect(design, "GCRK", s, n, "P_closed")
            seeds.append(s)
            for name, P in (("W", Pw), ("GCRK", Pg), ("closed", Pc)):
                met = C.metrics(P, F)
                rows.append(dict(design=design, seed=s, model=name, rmse=met["rmse"], mae=met["mae"],
                                 **{k: met[k] for k in met if k.startswith("rmse ")}, rmse_event_equal=event_equal_rmse(P, F),
                                 false_activity=float((P[z] > 0.001).mean()), false_activity_005=float((P[z] > 0.005).mean()),
                                 oracle_scale=float((P[m] * y[m]).sum() / (P[m] ** 2).sum()),
                                 peak_mag=met["peak_mag_abs_mean"], peak_time=met["peak_time_abs_mean"]))
                sse[name].append(C.unit_sse(P, F))
        if not seeds:
            print(design, "no finished seed"); continue
        sd = pd.DataFrame(rows)
        sd.to_csv(out / f"e3_seeds_{design}.csv", index=False)
        mean_sse = {k: np.mean(v, 0) for k, v in sse.items()}
        labels = {"county": F["fips"].astype(str), "event x state": np.char.add(np.char.add(ev, "|"), st), "event": ev}
        bt = []
        for a, b in (("GCRK", "W"), ("closed", "W")):
            point = float(np.sqrt(mean_sse[a].sum() / mean_sse[b].sum()) - 1)
            for lab, g in labels.items():
                u, inv = np.unique(g, return_inverse=True)
                sa, sb = np.bincount(inv, mean_sse[a]), np.bincount(inv, mean_sse[b])
                w = RNG.multinomial(len(u), np.full(len(u), 1 / len(u)), size=5000)
                r = np.sqrt((w @ sa) / (w @ sb)) - 1
                bt.append(dict(design=design, contrast=f"{a} vs {b}", cluster=lab, n_clusters=len(u), seeds=len(seeds),
                               rel_rmse_change=point, ci_lo=float(np.quantile(r, .025)), ci_hi=float(np.quantile(r, .975)),
                               p_worse=float((r > 0).mean())))
        bt = pd.DataFrame(bt)
        bt.to_csv(out / f"e3_bootstrap_{design}.csv", index=False)
        w_ = sd[sd.model == "W"].set_index("seed").rmse; g_ = sd[sd.model == "GCRK"].set_index("seed").rmse
        d = (g_ - w_).to_numpy()
        ci = bt[(bt.contrast == "GCRK vs W") & (bt.cluster == "county")].iloc[0]
        if len(seeds) < 5:
            reading = "incomplete"
        elif ci.ci_hi < 0 and (d < 0).sum() >= 4:
            reading = "GCRK better"
        elif ci.ci_lo > 0 and (d > 0).sum() >= 4:
            reading = "GCRK worse"
        else:
            reading = "not distinguishable"
        verdict = dict(design=design, seeds=seeds, gcrk_minus_w_per_seed=d.tolist(), n_gcrk_lower=int((d < 0).sum()),
                       county_interval=[float(ci.ci_lo), float(ci.ci_hi)], reading=reading, baselines=base)
        (out / f"e3_verdict_{design}.json").write_text(json.dumps(verdict, indent=1) + "\n")
        pd.set_option("display.width", 220)
        cols = ["rmse", "rmse 1-6 h", "rmse 7-24 h", "rmse 25-48 h", "rmse 49-144 h", "mae", "rmse_event_equal",
                "false_activity", "false_activity_005", "oracle_scale", "peak_mag", "peak_time"]
        print(f"\n== {C.ROUND} {design}: seeds {seeds}; all-zero {base['zero']:.5f} (event-equal {base['zero_event_equal']:.5f}),"
              f" persistence {base['persistence']:.5f}\n", sd.groupby("model")[cols].mean().round(5).to_string())
        print(bt.round(4).to_string(index=False))
        print("reading:", reading, "| GCRK - W per seed:", np.round(d, 6).tolist())


if __name__ == "__main__":
    main()
