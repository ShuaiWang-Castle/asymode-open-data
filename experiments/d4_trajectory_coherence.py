"""D-4 -- are an arm's per-horizon predictions a coherent trajectory?

Written and fixed before any out-of-fold prediction exists. It exists to test a
sentence the paper might otherwise assert: that a per-horizon regressor (one
model per horizon) is "more accurate per point but not a trajectory", while a
rollout is coherent by construction. Whether the regressor's four point
forecasts hang together is a property of its output and is measured here.

For each sample the four scored horizons (1, 6, 24, 48) give a pseudo-trajectory
p = (p_1, p_6, p_24, p_48); the truth gives y = (y_1, y_6, y_24, y_48).

  S1  excess sign changes -- share of samples where sign(diff(p)) changes at a
      step where sign(diff(y)) does NOT (a wiggle the truth does not have).
      Steps where |diff| < tol (1e-4 on the fraction scale) count as flat.
  S2  roughness -- mean absolute second difference of p across the four
      horizons, per arm, in the target's units. The rollout arms give the floor.

Both are computed over cells scored at all four horizons (mask true at every
horizon), per seed, then averaged. Weighting is per sample, fixed now.

Interpretation, fixed in advance: an arm whose S1 is materially above the
rollout arms' AND above the truth's own baseline (sign changes the truth has are
not counted, so the truth's S1 is 0 by construction; the comparison is arm vs
rollout) is "not a trajectory" in a measured sense; otherwise that framing is
unavailable and the paper does not use it.

INPUT: the agreed `results/oof_<arm>.npz` layout (audited as in D-3).
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
from d3_level_rank_decomp import audit   # noqa: E402

TOL = 1e-4


def _sign(d):
    s = np.sign(d); s[np.abs(d) < TOL] = 0
    return s


def coherence(pred4, y4):
    """pred4, y4: (n, 4) at horizons ordered 1,6,24,48. Returns S1, S2."""
    dp, dy = np.diff(pred4, axis=1), np.diff(y4, axis=1)          # (n, 3)
    sp, sy = _sign(dp), _sign(dy)
    # a sign change between consecutive steps in p that y does not have
    ch_p = (sp[:, :-1] * sp[:, 1:]) < 0
    ch_y = (sy[:, :-1] * sy[:, 1:]) < 0
    excess = np.any(ch_p & ~ch_y, axis=1)
    s1 = float(excess.mean())
    s2 = float(np.mean(np.abs(np.diff(pred4, n=2, axis=1))))
    return s1, s2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof-dir", default=str(ROOT / "results"))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default="results/d4_trajectory_coherence.json")
    a = ap.parse_args()
    files = sorted(Path(a.oof_dir).glob("oof_*.npz"))
    if not files:
        sys.exit("no out-of-fold prediction files (results/oof_*.npz); nothing to measure")
    out = {}
    print(f"{'arm':<22}{'S1 excess sign chg':>20}{'S2 roughness':>14}{'n':>8}")
    for f in files:
        z = np.load(f, allow_pickle=True); arm = f.stem.replace("oof_", "")
        audit(z, a.k)
        H = [int(h) for h in z["horizons"]]
        if H != [1, 6, 24, 48]:
            sys.exit(f"{arm}: horizons {H}, expected [1, 6, 24, 48]")
        y, mask = z["y"], z["mask"].astype(bool)
        full = mask.all(axis=1)
        per_seed = [coherence(z["pred"][si][full], y[full]) for si in range(len(z["seeds"]))]
        s1 = float(np.mean([r[0] for r in per_seed])); s2 = float(np.mean([r[1] for r in per_seed]))
        out[arm] = {"S1_excess_sign_change": s1, "S2_roughness": s2, "n_samples": int(full.sum()),
                    "seeds": len(per_seed)}
        print(f"{arm:<22}{s1:>20.4f}{s2:>14.5f}{int(full.sum()):>8}")
    # the truth's own roughness, for scale
    z = np.load(files[0], allow_pickle=True); y = z["y"]; full = z["mask"].astype(bool).all(axis=1)
    out["_truth"] = {"S2_roughness": float(np.mean(np.abs(np.diff(y[full], n=2, axis=1))))}
    print(f"{'(truth)':<22}{'—':>20}{out['_truth']['S2_roughness']:>14.5f}")
    (ROOT / a.out).write_text(json.dumps(out, indent=1)); print(f"\nwritten: {a.out}")


if __name__ == "__main__":
    main()
