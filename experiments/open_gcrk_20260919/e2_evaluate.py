"""E2 evaluation (PREREG Amendment 2): W refit at GCRK's t* against W at its own t*, GCRK and GCRK
exit closed. Per design and seed: pooled RMSE, lead segments, MAE, false activity at 0.001 and
0.005, mean prediction, and the gated-damage / background pathways of the replayed models.
Writes results/e2_{design}.csv and prints the seed means.
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
from diagnostics_frozen import stock

sys.path.insert(0, str(C.ROOT / "src"))
from asymode.asym_host import AsymODE, U_CAP, BKG_CAP  # noqa: E402
from asymode.gcrk_train import make_batch  # noqa: E402

torch.set_num_threads(1)
E2 = C.RUNS / "e2"


def collect_e2(design, seed, n):
    sp = C.load_splits()[design]
    P = np.full((n, 144), np.nan)
    for f in sp:
        z = np.load(E2 / design / f"seed{seed}" / f"fold{int(f):02d}" / "W" / "outer.npz")
        P[z["idx"]] = z["P"]
    assert np.isfinite(P).all()
    return P


def pathways(design, seed, F, root):
    """False activity of the model's own rollout with the background removed / alone (W models)."""
    sp = C.load_splits()[design]
    y, m = F["y"].astype(float), F["m"].astype(bool)
    acc = {k: [] for k in ("gate", "cond", "bkg", "r", "idx")}
    for f in sp:
        idx = np.array(sp[f]["outer"])
        snap = torch.load(root / design / f"seed{seed}" / f"fold{int(f):02d}" / "W" / "final.pt", weights_only=False)
        mdl = AsymODE(F["xu"].shape[-1], F["xr"].shape[-1], F["xo"].shape[-1])
        mdl.load_state_dict(snap["model_state"]); mdl.eval()
        with torch.no_grad():
            o = mdl(make_batch(F, idx, snap["stats"]))
        for k, v in (("gate", o["gate"]), ("cond", o["conditional"]), ("bkg", o["background"]), ("r", o["r"])):
            acc[k].append(v.numpy().astype(float))
        acc["idx"].append(idx)
    idx = np.concatenate(acc.pop("idx")); A = {k: np.concatenate(v) for k, v in acc.items()}
    z = m[idx] & (y[idx] == 0)
    capu = U_CAP + BKG_CAP
    p_nob = stock(np.clip(A["gate"] * A["cond"], 0, capu), A["r"], F["y0"][idx].astype(float))
    p_bkg = stock(np.clip(A["bkg"], 0, capu), A["r"], F["y0"][idx].astype(float))
    return float((p_nob[z] > 0.001).mean()), float((p_bkg[z] > 0.001).mean()), float(A["r"][m[idx]].mean())


def main():
    F = C.load_features()
    n = len(F["y"]); m = F["m"].astype(bool); y = F["y"].astype(float)
    z = m & (y == 0)
    for design in ("main", "loeo"):
        rows = []
        for s in C.SEEDS:
            models = {"W own t*": C.collect(design, "W", s, n), "W at GCRK t*": collect_e2(design, s, n),
                      "GCRK": C.collect(design, "GCRK", s, n), "GCRK exit closed": C.collect(design, "GCRK", s, n, "P_closed")}
            for name, P in models.items():
                met = C.metrics(P, F)
                row = dict(design=design, seed=s, model=name, rmse=met["rmse"], mae=met["mae"],
                           **{k: met[k] for k in met if k.startswith("rmse ")},
                           false_activity=float((P[z] > 0.001).mean()), false_activity_005=float((P[z] > 0.005).mean()),
                           mean_pred=float(P[m].mean()))
                if name in ("W own t*", "W at GCRK t*"):
                    root = C.RUNS if name == "W own t*" else E2
                    row["fa_no_background"], row["fa_background_only"], row["mean_r"] = pathways(design, s, F, root)
                rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(C.RESULTS / f"e2_{design}.csv", index=False)
        pd.set_option("display.width", 220)
        cols = ["rmse", "rmse 1-6 h", "rmse 7-24 h", "rmse 25-48 h", "rmse 49-144 h", "mae", "false_activity",
                "false_activity_005", "mean_pred", "fa_no_background", "fa_background_only", "mean_r"]
        print(f"\n== E2 {design} (mean over seeds)\n", df.groupby("model")[cols].mean().round(5).to_string())
        w0, w1 = df[df.model == "W own t*"].set_index("seed"), df[df.model == "W at GCRK t*"].set_index("seed")
        cl = df[df.model == "GCRK exit closed"].set_index("seed")
        print("per seed RMSE: W own", w0.rmse.round(5).tolist(), "| W at GCRK t*", w1.rmse.round(5).tolist(),
              "| closed", cl.rmse.round(5).tolist())


if __name__ == "__main__":
    main()
