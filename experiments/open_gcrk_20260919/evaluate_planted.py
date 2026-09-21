"""A6.3 evaluation: planted-effect recovery in the semi-synthetic worlds (PREREG Amendment 6).

For each world and arm (seed 0, the outer folds finished by every arm): RMSE against the noisy paths
(the protocol's metric) and against the truth's mean path; the truth and the neutral-geography truth
as references; and the share of the planted effect that the geography-conditioned kernel recovers
relative to the shared kernel, (mse_S - mse_G) / mse_neutral, on the error to the true mean path.
Writes results/planted_recovery.csv.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ("W", "GCRK-S", "GCRK", "GCRK-P", "W+G")


def main():
    rows = []
    for world in ("T0", "TB", "TA", "TBc"):
        os.environ["OPEN_GCRK_ROUND"] = f"syn{world}"
        for mod in ("common",):
            sys.modules.pop(mod, None)
        import common as C
        if not C.FEATURES.exists():
            continue
        z = np.load(C.FEATURES); y = z["y"].astype(float); m = z["m"].astype(bool); mu = z["mu"].astype(float); mun = z["mu_neutral"].astype(float)
        sp = json.loads(C.SPLITS_FILE.read_text())["main"]
        done = {a: [f for f in sp if (C.cell("main", 0, f, a) / "DONE.json").exists()] for a in ARMS}
        folds = sorted(set.intersection(*[set(v) for a, v in done.items() if v]) if any(done.values()) else set(), key=int)
        if not folds:
            continue
        idx = np.concatenate([np.array(sp[f]["outer"]) for f in folds])
        mm = m[idx]
        r = lambda a, b: float(np.sqrt(((a - b)[mm] ** 2).mean()))
        ref = dict(world=world, folds=",".join(folds), units=len(idx), zero=r(np.zeros_like(y[idx]), y[idx]), truth=r(mu[idx], y[idx]),
                   neutral_truth=r(mun[idx], y[idx]), neutral_to_truth=r(mun[idx], mu[idx]))
        res = {}
        for a in ARMS:
            if not done[a]:
                continue
            P = np.full_like(y, np.nan)
            for f in folds:
                d = np.load(C.cell("main", 0, f, a) / "outer.npz"); P[d["idx"]] = d["P"]
            res[a] = dict(noisy=r(P[idx], y[idx]), to_truth=r(P[idx], mu[idx]))
            t = [json.loads((C.cell("main", 0, f, a) / "DONE.json").read_text())["best_step"] for f in folds]
            res[a]["t_star"] = int(np.median(t))
        for a, v in res.items():
            rows.append(dict(**ref, arm=a, rmse_noisy=v["noisy"], rmse_to_truth=v["to_truth"], t_star=v["t_star"],
                             vs_W_noisy=v["noisy"] / res["W"]["noisy"] - 1 if "W" in res else np.nan))
        if "GCRK" in res and "GCRK-S" in res and ref["neutral_to_truth"] > 0:
            rec = (res["GCRK-S"]["to_truth"] ** 2 - res["GCRK"]["to_truth"] ** 2) / ref["neutral_to_truth"] ** 2
            print(f"{world}: share of the planted effect recovered by geography conditioning (GCRK vs GCRK-S): {rec:+.2f}")
        if "W+G" in res and "W" in res and ref["neutral_to_truth"] > 0:
            rec = (res["W"]["to_truth"] ** 2 - res["W+G"]["to_truth"] ** 2) / ref["neutral_to_truth"] ** 2
            print(f"{world}: share recovered by the level term (W+G vs W): {rec:+.2f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "results", "planted_recovery.csv"), index=False)
    pd.set_option("display.width", 220)
    print(df.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
