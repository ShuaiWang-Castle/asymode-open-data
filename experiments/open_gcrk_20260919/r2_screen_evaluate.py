"""R2 screen (PREREG Amendment 2): round 2 vs round 1 at seed 0, both designs, both arms.

Continuation rule (fixed in Amendment 2): R2 continues (and, by Amendment 3, sets the inputs of
the twelve-event round) if R2's W improves on round 1's W at seed 0 by more than round 1's
seed-to-seed standard deviation of W's RMSE in at least one design (main 0.00036, LOEO 0.00102).
Writes results/r2/screen.csv and results/r2/screen_decision.json.
"""
from __future__ import annotations

import json
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
import pandas as pd

import common as C

R2 = C.ROOT / "runs" / "open_gcrk_20260919" / "r2"
SD_R1 = {"main": 0.00036, "loeo": 0.00102}


def collect(root, design, arm, seed, n, key="P"):
    sp = C.load_splits()[design]                      # round-1 outer folds (round 2 keeps them)
    P = np.full((n, 144), np.nan)
    for f in sp:
        z = np.load(root / design / f"seed{seed}" / f"fold{int(f):02d}" / arm / "outer.npz")
        P[z["idx"]] = z[key]
    assert np.isfinite(P).all()
    return P


def main():
    F = C.load_features()
    n = len(F["y"]); m = F["m"].astype(bool); y = F["y"].astype(float); z = m & (y == 0)
    rows, t_star = [], []
    for design in ("main", "loeo"):
        for rnd, root in (("round 1", C.RUNS), ("round 2", R2)):
            for arm, key, name in (("W", "P", "W"), ("GCRK", "P", "GCRK"), ("GCRK", "P_closed", "GCRK exit closed")):
                P = collect(root, design, arm, 0, n, key)
                met = C.metrics(P, F)
                rows.append(dict(design=design, round=rnd, model=name, rmse=met["rmse"], mae=met["mae"],
                                 **{k: met[k] for k in met if k.startswith("rmse ")},
                                 peak_mag=met["peak_mag_abs_mean"], peak_time=met["peak_time_abs_mean"],
                                 false_activity=float((P[z] > 0.001).mean()), false_activity_005=float((P[z] > 0.005).mean()),
                                 oracle_scale=float((P[m] * y[m]).sum() / (P[m] ** 2).sum())))
            for arm in ("W", "GCRK"):
                for f in C.load_splits()[design]:
                    d = json.loads((root / design / "seed0" / f"fold{int(f):02d}" / arm / "DONE.json").read_text())
                    t_star.append(dict(design=design, round=rnd, arm=arm, fold=f, t_star=d["best_step"]))
    df = pd.DataFrame(rows)
    out = C.RESULTS / "r2"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "screen.csv", index=False)
    ts = pd.DataFrame(t_star)
    ts.to_csv(out / "screen_t_star.csv", index=False)
    dec = {}
    for design in ("main", "loeo"):
        w1 = float(df[(df.design == design) & (df["round"] == "round 1") & (df.model == "W")].rmse.iloc[0])
        w2 = float(df[(df.design == design) & (df["round"] == "round 2") & (df.model == "W")].rmse.iloc[0])
        dec[design] = dict(w_round1=w1, w_round2=w2, improvement=w1 - w2, threshold=SD_R1[design],
                           passes=bool(w1 - w2 > SD_R1[design]))
    dec["continue"] = any(v["passes"] for k, v in dec.items() if k in ("main", "loeo"))
    (out / "screen_decision.json").write_text(json.dumps(dec, indent=1) + "\n")
    pd.set_option("display.width", 220)
    cols = ["rmse", "rmse 1-6 h", "rmse 7-24 h", "rmse 25-48 h", "rmse 49-144 h", "mae", "peak_mag", "peak_time",
            "false_activity", "false_activity_005", "oracle_scale"]
    print(df.set_index(["design", "round", "model"])[cols].round(5).to_string())
    print("\nmedian t*:\n", ts.groupby(["design", "round", "arm"]).t_star.median().to_string())
    print("\ndecision:", json.dumps(dec, indent=1))


if __name__ == "__main__":
    main()
