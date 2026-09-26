"""Evaluation on the designed panel (DATASET_DESIGN v1 sections 1 and 9.3; amendment 2 S7, M9, M10).

Per regime r (a county-event takes its system's regime), the design-weighted MSE over the held-out county-hours of the
five event folds, MSE_r(f) = sum_i w_i SSE_i(f) / sum_i w_i N_i, for an arm, the host and the all-zero forecast;
predictions are averaged over the labels given for one model (its seeds). Reported:
  per regime: MSE of arm, host, zero; skill 1 - MSE(arm)/MSE(zero); relative change MSE(arm)/MSE(host) - 1
  regime-balanced skill and gain over the headline regimes (all five unless --headline), the worst regime (all five),
  the frame-weighted pooled value (regimes pooled with the design weights), untrimmed weights beside the trimmed ones
  intervals: family-cluster bootstrap within each regime, 2,000 draws, seed 20260924 (80% and 95%)
usage: python evaluate_v1.py --arm base_s3,base_s4,base_s5 --host base_s0,base_s1,base_s2 --out results/v1/aa.json"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS = ROOT / "runs" / "geo_weather_20260924"
FEAT = ROOT / "data" / "interim" / "panel_v1" / "features_v1D.npz"
REGIMES = ["tropical", "winter", "synoptic_wind", "convective", "heavy_rain"]
B, SEED = 2000, 20260924


def predictions(labels: list[str], n: int, design: str = "event") -> np.ndarray:
    P = np.zeros((len(labels), n, 144)); seen = np.zeros((len(labels), n), bool)
    for li, lab in enumerate(labels):
        for k in range(1, 6):
            f = RUNS / lab / f"fold{k:02d}" / "outer.npz"
            z = np.load(f)
            P[li, z["idx"]] = z["P"]; seen[li, z["idx"]] = True
    assert seen.all(), "every unit must be held out once per label"
    return P.mean(0)


def per_unit(P, y, m):
    return (m * (P - y) ** 2).sum(1), m.sum(1)


def groups_of(reg, fam) -> dict:
    idx = np.arange(len(reg))
    return {r: [idx[(reg == r) & (fam == f)] for f in np.unique(fam[reg == r])] for r in REGIMES if (reg == r).any()}


def stats(sse: dict, N, w, reg, fam, headline, rng=None, groups=None):
    """sse: {name: per-unit SSE}. Returns per-regime MSEs and the summaries, optionally on a family bootstrap draw
    (families resampled with replacement within each regime)."""
    idx = np.arange(len(w))
    if rng is not None:
        take = []
        for r, gl in groups.items():
            take += [gl[j] for j in rng.integers(0, len(gl), len(gl))]
        idx = np.concatenate(take)
    out = {}
    for r in REGIMES:
        ii = idx[reg[idx] == r]
        if len(ii) == 0:
            continue
        den = (w[ii] * N[ii]).sum()
        out[r] = {k: float((w[ii] * v[ii]).sum() / den) for k, v in sse.items()}
    den = (w[idx] * N[idx]).sum()
    pooled = {k: float((w[idx] * v[idx]).sum() / den) for k, v in sse.items()}
    hl = [r for r in headline if r in out]
    s = dict(per_regime=out, pooled=pooled,
             balanced_skill=float(np.mean([1 - out[r]["arm"] / out[r]["zero"] for r in hl])),
             balanced_gain=float(np.mean([out[r]["arm"] / out[r]["host"] - 1 for r in hl])),
             worst_gain=float(max(out[r]["arm"] / out[r]["host"] - 1 for r in out)),
             pooled_gain=float(pooled["arm"] / pooled["host"] - 1),
             host_vs_zero={r: float(out[r]["host"] / out[r]["zero"] - 1) for r in out})
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--host", required=True)
    ap.add_argument("--headline", nargs="*", default=REGIMES)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    F = np.load(FEAT, allow_pickle=False)
    y, m = F["y"].astype(float), F["m"].astype(float)
    reg, fam = F["regime"].astype(str), F["family"].astype(str)
    n = len(y)
    Pa, Ph = predictions(a.arm.split(","), n), predictions(a.host.split(","), n)
    sa, N = per_unit(Pa, y, m); sh, _ = per_unit(Ph, y, m); sz, _ = per_unit(np.zeros_like(y), y, m)
    sse = dict(arm=sa, host=sh, zero=sz)
    res = {}
    for wname in ("w", "w_raw"):
        w = F[wname].astype(float)
        point = stats(sse, N, w, reg, fam, a.headline)
        rng = np.random.default_rng(SEED)
        gr = groups_of(reg, fam)
        draws = [stats(sse, N, w, reg, fam, a.headline, rng, gr) for _ in range(B)]

        def iv(get):
            v = np.array([get(d) for d in draws])
            return {"80": [float(np.quantile(v, .1)), float(np.quantile(v, .9))],
                    "95": [float(np.quantile(v, .025)), float(np.quantile(v, .975))]}
        point["intervals"] = dict(
            balanced_skill=iv(lambda d: d["balanced_skill"]), balanced_gain=iv(lambda d: d["balanced_gain"]),
            pooled_gain=iv(lambda d: d["pooled_gain"]),
            gain={r: iv(lambda d, r=r: d["per_regime"][r]["arm"] / d["per_regime"][r]["host"] - 1) for r in point["per_regime"]},
            host_vs_zero={r: iv(lambda d, r=r: d["host_vs_zero"][r]) for r in point["per_regime"]})
        res[wname] = point
    res["meta"] = dict(arm=a.arm, host=a.host, headline=a.headline, units=int(n), bootstrap=dict(draws=B, seed=SEED),
                       families_per_regime={r: int(len(np.unique(fam[reg == r]))) for r in REGIMES})
    out = HERE / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1) + "\n")
    p = res["w"]
    print("regime-balanced gain", round(100 * p["balanced_gain"], 2), "%", p["intervals"]["balanced_gain"]["95"],
          "| skill", round(100 * p["balanced_skill"], 2), "% | worst", round(100 * p["worst_gain"], 2), "%")
    for r, v in p["per_regime"].items():
        print(f"  {r:14s} arm {v['arm']:.3e} host {v['host']:.3e} zero {v['zero']:.3e} host/zero-1 "
              f"{100 * (v['host'] / v['zero'] - 1):+.1f}% gain {100 * (v['arm'] / v['host'] - 1):+.2f}%")


if __name__ == "__main__":
    main()
