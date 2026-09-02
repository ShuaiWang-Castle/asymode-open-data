"""D-3 -- split each arm's squared error into a level term and a within-origin term.

Written and fixed BEFORE any out-of-fold prediction exists, so neither the
decomposition nor its weighting can be chosen after seeing which flatters an
arm. It is the follow-up registered for D-2 and the test the H-E interpretive
note depends on: a long-horizon win is a *level* win unless this says otherwise.

Decomposition, per (arm, seed, horizon), inside one forecast origin at a time:

    e_i   = pred_i - y_i                      county i, one origin, one horizon
    e_bar = mean_i e_i                        that origin's level error
    sum_i e_i^2 = n * e_bar^2 + sum_i (e_i - e_bar)^2
                  \\__level__/   \\_____within-origin_____/

An exact identity, not a model. The second term also carries within-origin
spread error, so the paper calls it "within-origin", not "ranking", unless the
rank-only statistic below agrees.

WEIGHTING, FIXED IN ADVANCE: **per scored cell.** Level and within-origin sums
are accumulated over all origins and divided by the total number of scored
cells. This is the same unit as the archived RMSE, so

    level_mse + within_mse  ==  archived MSE   (per arm, seed, horizon)

must hold to float precision, and the script checks it against the result JSON
when one is present. Per-origin equal weighting was considered and rejected
because it would let small storms speak as loudly as large ones and would not
reconcile with any number already in the ledger.

The rank-only statistic -- Spearman rho between pred and y across counties at
each origin -- is by construction a per-origin quantity and is averaged over
origins with equal weight. It is labelled as such and is bounded above by the
D-2 ceiling, so it is never read on its own.

INPUT LAYOUT (the experiment lane's out-of-fold export, `results/oof_<arm>.npz`):

    pred        [n_seeds, n_samples, n_horizons]  float32  each sample predicted by the fold that held it out
    fold_of     [n_seeds, n_samples]              int8     which fold held it out (audited below)
    y           [n_samples, n_horizons]           float32
    mask        [n_samples, n_horizons]           bool
    origin_id   [n_samples]                       int32    dense code of (panel, origin step)
    fips        [n_samples]                       str
    panel       [n_samples]                       str
    horizons    [n_horizons]                      int32
    seeds       [n_seeds]                         int32
    panel_digest, channel_digest, source          scalars, matching the result JSON

Audit performed before anything is computed: for every seed, every sample has a
fold in [0, k) and the fold assignment agrees with `make_folds(fips, k, seed)`.
A sample held out zero or two times is a broken export, and the script refuses.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from asymode.evalproto import make_folds   # noqa: E402


def audit(z, k: int) -> None:
    fips = z["fips"].astype(str); seeds = z["seeds"]; fold_of = z["fold_of"]
    uniq = sorted(set(fips))
    for si, seed in enumerate(seeds):
        expect = dict(zip(uniq, make_folds(uniq, k=k, seed=int(seed))))
        got = fold_of[si]
        if got.min() < 0 or got.max() >= k:
            sys.exit(f"seed {seed}: fold_of outside [0,{k})")
        mism = int(np.sum(got != np.array([expect[f] for f in fips])))
        if mism:
            sys.exit(f"seed {seed}: {mism} samples whose fold_of disagrees with make_folds")


def decompose_one(pred, y, mask, origin_id):
    """One (seed, horizon). Returns per-cell-weighted level/within MSE and per-origin mean rho."""
    lev_sum = within_sum = 0.0; n_cells = 0; rhos = []
    for o in np.unique(origin_id):
        sel = (origin_id == o) & mask
        n = int(sel.sum())
        if n < 20:
            continue
        e = pred[sel] - y[sel]; eb = e.mean()
        lev_sum += n * eb * eb; within_sum += float(((e - eb) ** 2).sum()); n_cells += n
        a, b = pred[sel], y[sel]
        rhos.append(0.0 if (a.std() < 1e-12 or b.std() < 1e-12)
                    else float(np.nan_to_num(spearmanr(a, b).correlation)))
    if n_cells == 0:
        return None
    return {"level_mse": lev_sum / n_cells, "within_mse": within_sum / n_cells,
            "total_mse": (lev_sum + within_sum) / n_cells,
            "mean_rho_per_origin": float(np.mean(rhos)), "n_origins": len(rhos), "n_cells": n_cells}


def archived_mse(result_json: Path, arm: str, seed: int, h: int):
    """Mean over folds of archived RMSE^2 for reconciliation; None if unavailable."""
    if not result_json.exists():
        return None
    rows = [r for r in json.loads(result_json.read_text())["rows"]
            if r["arm"] == arm and r["seed"] == seed and np.isfinite(r.get(f"rmse_h{h}", np.nan))]
    if not rows:
        return None
    n = np.array([r.get(f"n_h{h}", 1) for r in rows], float)
    m = np.array([r[f"rmse_h{h}"] ** 2 for r in rows])
    return float((m * n).sum() / n.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof-dir", default=str(ROOT / "results"))
    ap.add_argument("--result-json", default=str(ROOT / "results" / "exp05_real_dynamics.json"))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default="results/d3_level_rank_decomp.json")
    a = ap.parse_args()
    files = sorted(Path(a.oof_dir).glob("oof_*.npz"))
    if not files:
        sys.exit("no out-of-fold prediction files (results/oof_*.npz); nothing to decompose")

    out = {}
    print(f"{'arm':<20}{'h':>4}{'level':>11}{'within':>11}{'level share':>13}{'rho':>7}{'reconcile':>11}")
    for f in files:
        z = np.load(f, allow_pickle=True); arm = f.stem.replace("oof_", "")
        audit(z, a.k)
        H = [int(h) for h in z["horizons"]]; seeds = [int(s) for s in z["seeds"]]
        y, mask, oid = z["y"], z["mask"].astype(bool), z["origin_id"]
        out[arm] = {"panel_digest": str(z.get("panel_digest", "")), "channel_digest": str(z.get("channel_digest", ""))}
        for hi, h in enumerate(H):
            per_seed = []
            for si, seed in enumerate(seeds):
                r = decompose_one(z["pred"][si, :, hi], y[:, hi], mask[:, hi], oid)
                if r is None:
                    continue
                ref = archived_mse(Path(a.result_json), arm, seed, h)
                r["archived_mse"] = ref
                r["reconciles"] = (None if ref is None
                                   else bool(abs(r["total_mse"] - ref) <= 1e-6 + 1e-3 * ref))
                per_seed.append(r)
            if not per_seed:
                continue
            agg = {k: float(np.mean([r[k] for r in per_seed])) for k in
                   ("level_mse", "within_mse", "total_mse", "mean_rho_per_origin")}
            agg["level_share"] = agg["level_mse"] / max(agg["total_mse"], 1e-18)
            agg["seeds"] = len(per_seed)
            rec = [r["reconciles"] for r in per_seed if r["reconciles"] is not None]
            agg["reconciles_with_archive"] = (None if not rec else all(rec))
            out[arm][h] = agg
            print(f"{arm:<20}{h:>4}{agg['level_mse']:>11.3e}{agg['within_mse']:>11.3e}"
                  f"{agg['level_share']:>13.2f}{agg['mean_rho_per_origin']:>7.2f}"
                  f"{'n/a' if agg['reconciles_with_archive'] is None else ('ok' if agg['reconciles_with_archive'] else 'MISMATCH'):>11}")
    dst = ROOT / a.out; dst.write_text(json.dumps(out, indent=1))
    print(f"\nwritten: {a.out}")
    bad = [(arm, h) for arm, v in out.items() for h, g in v.items()
           if isinstance(g, dict) and g.get("reconciles_with_archive") is False]
    if bad:
        print("RECONCILIATION FAILED for:", bad)
        sys.exit(1)


if __name__ == "__main__":
    main()
