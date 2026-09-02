"""D-3 -- split each arm's squared error into a level term and a ranking term.

Written before any out-of-fold prediction exists, so the decomposition cannot be
chosen after seeing which one flatters an arm. It is the follow-up registered in
`docs/PREREGISTRATION_external_priors.md` (D-2) and the test that the H-E
interpretive note depends on: a long-horizon win is a *level* win unless this
says otherwise.

Decomposition, per (arm, seed, fold, horizon), over the scored cells of one
forecast origin at a time and then pooled:

    e_i   = pred_i - y_i                            county i, one origin
    e_bar = mean_i e_i                              the origin's level error
    MSE   = e_bar^2  +  mean_i (e_i - e_bar)^2
            \\_level_/    \\____ranking + spread____/

The first term is what a model that got the cross-county mean exactly right
would still be charged for getting it wrong; the second is everything that
depends on *which* counties are high and low. This is an exact identity, not a
model. "Ranking term" is a slight misnomer -- it also carries within-origin
spread error -- and the paper must call it "within-origin" rather than "ranking"
unless a rank-only variant (below) agrees.

A rank-only variant is reported alongside: Spearman rho between pred and y
across counties at each origin, averaged. Where the D-2 ceiling is low this
number is bounded above by the ceiling and should not be read on its own.

Input: `results/oof_<arm>.npz` with arrays `pred[seed,fold]` and `y`, `mask`,
`origin_id` aligned to the sample index, plus `horizons`. Produced by the
experiment lane's export; this script refuses to run on anything else.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]


def decompose(pred, y, mask, origin_id):
    """Return level_mse, within_mse, mean_rho over origins for one horizon."""
    lev, within, rhos, n_cells = [], [], [], 0
    for o in np.unique(origin_id):
        sel = (origin_id == o) & mask
        if sel.sum() < 20:
            continue
        e = pred[sel] - y[sel]
        eb = e.mean()
        lev.append(eb ** 2 * sel.sum()); within.append(((e - eb) ** 2).sum())
        n_cells += int(sel.sum())
        a, b = pred[sel], y[sel]
        rhos.append(0.0 if (a.std() < 1e-12 or b.std() < 1e-12)
                    else float(np.nan_to_num(spearmanr(a, b).correlation)))
    if n_cells == 0:
        return None
    return {"level_mse": float(sum(lev) / n_cells),
            "within_mse": float(sum(within) / n_cells),
            "mean_rho": float(np.mean(rhos)), "n_origins": len(rhos), "n_cells": n_cells}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof-dir", default=str(ROOT / "results"))
    ap.add_argument("--out", default="results/d3_level_rank_decomp.json")
    a = ap.parse_args()
    files = sorted(Path(a.oof_dir).glob("oof_*.npz"))
    if not files:
        sys.exit("no out-of-fold prediction files (results/oof_*.npz); nothing to decompose")
    out = {}
    print(f"{'arm':<20}{'h':>4}{'level MSE':>12}{'within MSE':>12}{'level share':>13}{'rho':>7}")
    for f in files:
        z = np.load(f, allow_pickle=True)
        arm = f.stem.replace("oof_", "")
        H = [int(h) for h in z["horizons"]]
        y, mask, oid = z["y"], z["mask"].astype(bool), z["origin_id"]
        out[arm] = {}
        for hi, h in enumerate(H):
            rows = []
            for key in [k for k in z.files if k.startswith("pred_")]:
                r = decompose(z[key][:, hi], y[:, hi], mask[:, hi], oid)
                if r:
                    rows.append(r)
            if not rows:
                continue
            agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
            agg["units"] = len(rows)
            agg["level_share"] = agg["level_mse"] / max(agg["level_mse"] + agg["within_mse"], 1e-18)
            out[arm][h] = agg
            print(f"{arm:<20}{h:>4}{agg['level_mse']:>12.3e}{agg['within_mse']:>12.3e}"
                  f"{agg['level_share']:>13.2f}{agg['mean_rho']:>7.2f}")
    dst = ROOT / a.out
    dst.write_text(json.dumps(out, indent=1))
    print(f"\nwritten: {a.out}")


if __name__ == "__main__":
    main()
