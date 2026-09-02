"""D-1 -- the oracle shrinkage bound: how much could ANY de-shrinkage recover?

Written and fixed before any out-of-fold prediction exists. Registered in
docs/PREREGISTRATION_external_priors.md as a generic diagnostic, with a rule
attached: no peak weighting, large-value loss, or output-side normalisation is
attempted in this project unless this script first shows headroom.

For each (arm, seed, horizon), fit

    pred' = a * pred^lambda            a > 0, lambda > 0, pred >= 0

by minimising per-cell MSE **on the evaluation cells themselves**. That is
deliberate cheating: the fit sees the answers, so whatever it gains is an upper
bound on what any honest post-hoc de-shrinkage could gain. Two numbers are
reported:

    lambda*      the oracle exponent. Below 1 means the oracle wants the
                 predictions MORE compressed, not less -- under imperfect
                 ranking the MSE-optimal prediction shrinks toward the mean,
                 and the residual uncertainty is exactly the shrinkage.
    headroom     1 - MSE(pred') / MSE(pred). The most any monotone power
                 rescaling could recover. If this is near zero, the family of
                 "make the peaks bigger" interventions is capped there.

The fit is a two-parameter grid over lambda with a closed-form least-squares a
for each lambda, so it is deterministic, needs no optimiser, and cannot get
stuck. Cells with pred == 0 contribute pred' == 0 for every lambda, which is the
correct behaviour: a power law cannot lift an exact zero, and neither can any
monotone rescaling.

WEIGHTING: per scored cell, as in D-3, so MSE(pred) reconciles with the archived
RMSE^2 and the two diagnostics share one unit.

DEPENDENCE BETWEEN FORECASTS, STATED SO NO ONE ATTACHES A TEST TO THE WRONG UNIT:
with a 48-hour horizon and a 12-hour origin stride, consecutive forecasts in one
storm share 36 of 48 target hours. The forecasts are therefore not independent
units -- the effective number of independent units is closer to the number of
panels than to the number of forecasts. This script reports point estimates
only. Any uncertainty statement must be block-resampled by PANEL; per-forecast
standard errors would be badly understated and are not computed here. (The
paired t statistics elsewhere in this project use (seed, fold) as the unit,
which is the protocol unit and unaffected.)

INPUT: the same `results/oof_<arm>.npz` layout as D-3 (audited the same way).
`pred` must be the array that was SCORED -- for an arm whose head is unbounded
and is clipped to [0, 1] at scoring time, export the clipped values. The
reconciliation check enforces this: an export that does not reproduce the
archived MSE fails.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
from d3_level_rank_decomp import audit, archived_mse   # noqa: E402  same audit, same reconciliation

LAMBDAS = np.round(np.concatenate([np.arange(0.30, 1.00, 0.05), np.arange(1.00, 2.01, 0.10)]), 3)


def oracle(pred, y):
    """Best (a, lambda) on these cells; returns lambda*, a*, mse_before, mse_after.

    `base` is the MSE of the prediction EXACTLY AS SCORED -- no clipping, no
    transform -- so it reconciles with the archive and the identity transform is
    a genuine member of the candidate family. The power law needs a non-negative
    input, so the clip lives inside the candidate only. (An earlier version
    clipped before computing `base`; on a target near zero that clip is itself a
    large MSE gain, and the "oracle" was mostly measuring it.)
    """
    raw = pred.astype(np.float64); t = y.astype(np.float64)
    base = float(np.mean((raw - t) ** 2))
    best = (1.0, 1.0, base)
    p = np.clip(raw, 0, None)
    for lam in LAMBDAS:
        q = p ** lam
        den = float(np.dot(q, q))
        if den <= 0:
            continue
        a = float(np.dot(q, t) / den)                 # closed-form LS scale
        if a <= 0:
            continue
        mse = float(np.mean((a * q - t) ** 2))
        if mse < best[2]:
            best = (float(lam), a, mse)
    lam, a, mse = best
    return lam, a, base, mse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof-dir", default=str(ROOT / "results"))
    ap.add_argument("--result-json", default=str(ROOT / "results" / "exp05_real_dynamics.json"))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default="results/d1_oracle_shrinkage.json")
    a = ap.parse_args()
    files = sorted(Path(a.oof_dir).glob("oof_*.npz"))
    if not files:
        sys.exit("no out-of-fold prediction files (results/oof_*.npz); nothing to bound")

    out = {}
    print(f"{'arm':<20}{'h':>4}{'lambda*':>9}{'a*':>8}{'headroom':>10}{'reconcile':>11}")
    for f in files:
        z = np.load(f, allow_pickle=True); arm = f.stem.replace("oof_", "")
        audit(z, a.k)
        H = [int(h) for h in z["horizons"]]; seeds = [int(s) for s in z["seeds"]]
        y, mask = z["y"], z["mask"].astype(bool)
        out[arm] = {}
        for hi, h in enumerate(H):
            rows = []
            for si, seed in enumerate(seeds):
                m = mask[:, hi]
                lam, aa, base, after = oracle(z["pred"][si, m, hi], y[m, hi])
                ref = archived_mse(Path(a.result_json), arm, seed, h)
                rows.append({"seed": seed, "lambda_star": lam, "a_star": aa, "mse_before": base,
                             "mse_after": after, "headroom": 1.0 - after / max(base, 1e-18),
                             "archived_mse": ref,
                             "reconciles": None if ref is None else bool(abs(base - ref) <= 1e-6 + 1e-3 * ref)})
            agg = {k: float(np.mean([r[k] for r in rows])) for k in
                   ("lambda_star", "a_star", "mse_before", "mse_after", "headroom")}
            agg["lambda_star_per_seed"] = [r["lambda_star"] for r in rows]
            rec = [r["reconciles"] for r in rows if r["reconciles"] is not None]
            agg["reconciles_with_archive"] = None if not rec else all(rec)
            out[arm][h] = agg
            print(f"{arm:<20}{h:>4}{agg['lambda_star']:>9.2f}{agg['a_star']:>8.3f}{agg['headroom']:>10.3f}"
                  f"{'n/a' if agg['reconciles_with_archive'] is None else ('ok' if agg['reconciles_with_archive'] else 'MISMATCH'):>11}")
    dst = ROOT / a.out; dst.write_text(json.dumps(out, indent=1))
    print(f"\nwritten: {a.out}")
    if any(g.get("reconciles_with_archive") is False for v in out.values() for g in v.values() if isinstance(g, dict)):
        print("RECONCILIATION FAILED"); sys.exit(1)


if __name__ == "__main__":
    main()
