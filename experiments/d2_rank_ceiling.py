"""D-2 -- how much of each horizon's ranking is intrinsically unpredictable?

Zero training. For every scored horizon h and every forecast origin o, rank the
counties by an ILLEGAL predictor -- the truth one step ahead, y[o+1] -- and score
that ranking against the truth at y[o+h]. No legal model can rank better than a
predictor that has already seen the future, so this is a ceiling. Where the
ceiling is low, the MSE-optimal answer is each county's level, not its ordering,
and a model with dynamics is pushed toward a constant.

Registered in docs/PREREGISTRATION_external_priors.md as a generic diagnostic.
Reads only the panels; touches nothing another run may be writing.
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from asymode.evalproto import to_hourly   # noqa: E402

INTERIM = ROOT / "data" / "interim"
H = (1, 6, 24, 48)
MIN_HIST, STRIDE = 24, 6


def main():
    man = json.load(open(INTERIM / "PANEL_MANIFEST.json"))
    days = man["panels"]
    out = {h: [] for h in H}
    for day in days:
        z = np.load(INTERIM / f"panel_{day}.npz", allow_pickle=True)
        yh, oh = to_hourly(z["y"], z["observed"])
        yh = np.nan_to_num(yh)
        T = yh.shape[1]
        for o in range(MIN_HIST, T - max(H) - 1, STRIDE):
            m1 = oh[:, o + 1]
            for h in H:
                m = m1 & oh[:, o + h]
                if m.sum() < 20:
                    continue
                a, b = yh[m, o + 1], yh[m, o + h]
                if a.std() < 1e-12 or b.std() < 1e-12:
                    rho = 0.0
                else:
                    rho = float(spearmanr(a, b).correlation)
                    if np.isnan(rho):
                        rho = 0.0
                out[h].append({"day": day, "origin": int(o), "n": int(m.sum()),
                               "ceiling_rho": rho,
                               "frac_zero_target": float((b == 0).mean())})
    res = {"manifest": man.get("generation"), "digest": man.get("digest"),
           "illegal_predictor": "truth at origin+1", "per_horizon": {}}
    print(f"manifest {man.get('generation')}  {len(days)} panels\n")
    print(f"{'h':>4}{'origins':>9}{'ceiling p50':>13}{'p25':>7}{'frac<0.3':>10}{'frac<0.5':>10}{'zero-target p50':>17}")
    for h in H:
        r = np.array([x["ceiling_rho"] for x in out[h]])
        fz = np.array([x["frac_zero_target"] for x in out[h]])
        res["per_horizon"][h] = {
            "n_origins": len(r), "ceiling_p50": float(np.median(r)),
            "ceiling_p25": float(np.percentile(r, 25)),
            "frac_below_0.3": float((r < 0.3).mean()), "frac_below_0.5": float((r < 0.5).mean()),
            "zero_target_p50": float(np.median(fz)), "rows": out[h]}
        print(f"{h:>4}{len(r):>9}{np.median(r):>13.3f}{np.percentile(r,25):>7.3f}"
              f"{(r<0.3).mean():>10.2f}{(r<0.5).mean():>10.2f}{np.median(fz):>17.2f}")
    dst = ROOT / "results" / "d2_rank_ceiling.json"
    dst.write_text(json.dumps(res, indent=1))
    print(f"\nwritten: {dst.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
