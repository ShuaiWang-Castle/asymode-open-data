"""D-4 comparison -- is arm A's pseudo-trajectory less coherent than arm B's?

Written before the trees' out-of-fold export exists, so the test cannot be
shaped by it. This is the re-registered D-4 decision: the "more accurate per
point, but not a trajectory" framing is available only if the per-horizon
regressor's excess-sign-change rate S1 exceeds the two-rate rollout's by a
margin that survives county-block resampling.

Statistic: S1(A) - S1(B), where S1 is the share of samples whose four-horizon
pseudo-trajectory has a sign change in its first difference that the truth does
not have (definition in d4_trajectory_coherence.py, TOL 1e-4). Both arms are
evaluated on the same samples (same fips, panel, origin_step, and mask at all
four horizons), which is asserted, not assumed.

Uncertainty: samples from one county are not independent -- a county appears at
many origins and in several storms -- so resampling is done by COUNTY block
(all of a county's samples move together), B = 2000, per seed, then pooled. A
per-sample bootstrap would understate the interval and is not computed.

Decision, fixed in advance: the framing is AVAILABLE only if the 95% interval
of S1(A) - S1(B) lies entirely above zero for every seed. Otherwise it is
unavailable and the paper does not use it.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from d4_trajectory_coherence import _sign   # noqa: E402  same sign convention, same tolerance


def excess_flags(pred4, y4):
    """Per-sample boolean: pseudo-trajectory has a sign change the truth lacks."""
    sp, sy = _sign(np.diff(pred4, axis=1)), _sign(np.diff(y4, axis=1))
    return np.any(((sp[:, :-1] * sp[:, 1:]) < 0) & ~((sy[:, :-1] * sy[:, 1:]) < 0), axis=1)


def block_bootstrap(diff_by_county: dict, B: int, rng) -> np.ndarray:
    """Resample counties with replacement; return the bootstrap distribution of the mean difference."""
    keys = list(diff_by_county); sums = np.array([diff_by_county[k][0] for k in keys], float)
    counts = np.array([diff_by_county[k][1] for k in keys], float)
    out = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, len(keys), len(keys))
        out[b] = sums[idx].sum() / max(counts[idx].sum(), 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm_a"); ap.add_argument("arm_b")
    ap.add_argument("--oof-dir", default=str(ROOT / "results"))
    ap.add_argument("--B", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    za = np.load(Path(a.oof_dir) / f"oof_{a.arm_a}.npz", allow_pickle=True)
    zb = np.load(Path(a.oof_dir) / f"oof_{a.arm_b}.npz", allow_pickle=True)
    for k in ("fips", "panel", "horizons", "seeds"):
        if not np.array_equal(za[k], zb[k]):
            sys.exit(f"exports differ on {k}; not the same samples")
    if "origin_step" in za.files and "origin_step" in zb.files and not np.array_equal(za["origin_step"], zb["origin_step"]):
        sys.exit("exports differ on origin_step; not the same samples")
    if not np.array_equal(za["mask"], zb["mask"]):
        sys.exit("exports differ on mask; not the same scored cells")
    if [int(h) for h in za["horizons"]] != [1, 6, 24, 48]:
        sys.exit("expected horizons [1, 6, 24, 48]")
    y, full = za["y"], za["mask"].astype(bool).all(axis=1)
    fips = za["fips"].astype(str)[full]
    rng = np.random.default_rng(a.seed)
    res = {"arm_a": a.arm_a, "arm_b": a.arm_b, "B": a.B, "by_seed": []}
    print(f"{a.arm_a} vs {a.arm_b}: {int(full.sum()):,} samples scored at all four horizons, {len(set(fips)):,} counties")
    print(f"{'seed':>5}{'S1(A)':>9}{'S1(B)':>9}{'diff':>9}{'95% CI (county-block)':>26}{'verdict':>12}")
    all_above = True
    for si, seed in enumerate([int(s) for s in za["seeds"]]):
        fa = excess_flags(za["pred"][si][full], y[full]); fb = excess_flags(zb["pred"][si][full], y[full])
        d = fa.astype(float) - fb.astype(float)
        by_c = {}
        for c, v in zip(fips, d):
            acc = by_c.setdefault(c, [0.0, 0]); acc[0] += v; acc[1] += 1
        boot = block_bootstrap(by_c, a.B, rng)
        lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
        above = lo > 0; all_above &= above
        res["by_seed"].append({"seed": seed, "S1_a": float(fa.mean()), "S1_b": float(fb.mean()),
                               "diff": float(d.mean()), "ci_lo": lo, "ci_hi": hi, "above_zero": bool(above)})
        print(f"{seed:>5}{fa.mean():>9.4f}{fb.mean():>9.4f}{d.mean():>+9.4f}{f'[{lo:+.4f}, {hi:+.4f}]':>26}{'A less coherent' if above else 'not separable':>12}")
    res["framing_available"] = bool(all_above)
    print("\nDECISION (fixed in advance):", "framing AVAILABLE -- A is less coherent than B on every seed"
          if all_above else "framing UNAVAILABLE -- the interval touches zero on at least one seed")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
