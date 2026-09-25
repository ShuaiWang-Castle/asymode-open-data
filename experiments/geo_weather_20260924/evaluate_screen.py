"""Screen evaluation: pooled hourly RMSE of the held-out counties' rollouts, and each label against a base
label on the same units, with cluster-bootstrap intervals (counties; event x state).

  python evaluate_screen.py --base base_e3r2 --labels pop_v3p ... [--folds 1 2] [--out results/screen_<tag>.json]
Relative change = RMSE(label) / RMSE(base) - 1 (negative = better). 2000 bootstrap draws, seed 20260924.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS = ROOT / "runs" / "geo_weather_20260924"
FEAT = ROOT / "data" / "interim" / "open_gcrk" / "features_e3r2.npz"
B, SEED = 2000, 20260924


def unit_errors(label: str, folds, y, m):
    idx, s = [], []
    for k in folds:
        z = np.load(RUNS / label / f"fold{k:02d}" / "outer.npz")
        i = z["idx"]; idx.append(i)
        s.append(((z["P"] - y[i]) ** 2 * m[i]).sum(1))
    idx = np.concatenate(idx); o = np.argsort(idx)
    return idx[o], np.concatenate(s)[o]


def cluster_ratio(sa, sb, n, cl, rng):
    u, inv = np.unique(cl, return_inverse=True)
    A = np.bincount(inv, sa, len(u)); Bb = np.bincount(inv, sb, len(u)); N = np.bincount(inv, n, len(u))
    draws = rng.integers(0, len(u), (B, len(u)))
    ra = np.sqrt(A[draws].sum(1) / N[draws].sum(1)); rb = np.sqrt(Bb[draws].sum(1) / N[draws].sum(1))
    r = ra / rb - 1
    return [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))], float((r < 0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--folds", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--out", default=None)
    ap.add_argument("--feat", default=str(FEAT.relative_to(ROOT)), help="feature file of the panel (targets, masks)")
    a = ap.parse_args()
    z = np.load(ROOT / a.feat)
    y, m = z["y"].astype(np.float64), z["m"].astype(np.float64)
    fips, ev = z["fips"].astype(str), z["event"].astype(str)
    ib, sb = unit_errors(a.base, a.folds, y, m)
    n = m[ib].sum(1)
    rows = [dict(label=a.base, rmse=float(np.sqrt(sb.sum() / n.sum())), units=int(len(ib)))]
    for L in a.labels:
        il, sl = unit_errors(L, a.folds, y, m)
        assert np.array_equal(il, ib), f"{L}: units differ from the base"
        rmse = float(np.sqrt(sl.sum() / n.sum()))
        rel = rmse / rows[0]["rmse"] - 1
        rng = np.random.default_rng(SEED)
        ci_c, pc = cluster_ratio(sl, sb, n, fips[ib], rng)
        ci_e, pe = cluster_ratio(sl, sb, n, np.char.add(ev[ib], np.char.add("_", np.array([f[:2] for f in fips[ib]]))), rng)
        per_event = {}
        for e in sorted(set(ev[ib])):
            k = ev[ib] == e
            per_event[e] = round(float(np.sqrt(sl[k].sum() / n[k].sum()) / np.sqrt(sb[k].sum() / n[k].sum()) - 1), 4)
        rows.append(dict(label=L, rmse=rmse, rel_vs_base=rel, county_ci=ci_c, county_p_better=pc,
                         event_state_ci=ci_e, event_state_p_better=pe,
                         events_better=int(sum(v < 0 for v in per_event.values())), per_event=per_event))
    for r in rows:
        if "rel_vs_base" in r:
            print(f"{r['label']:<28} RMSE {r['rmse']:.6f}  {100 * r['rel_vs_base']:+.2f}%  county "
                  f"[{100 * r['county_ci'][0]:+.2f}, {100 * r['county_ci'][1]:+.2f}]  event x state "
                  f"[{100 * r['event_state_ci'][0]:+.2f}, {100 * r['event_state_ci'][1]:+.2f}]  "
                  f"events better {r['events_better']}/{len(r['per_event'])}")
        else:
            print(f"{r['label']:<28} RMSE {r['rmse']:.6f}  (base, {r['units']} units)")
    if a.out:
        p = HERE / a.out
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(dict(base=a.base, folds=a.folds, rows=rows), indent=1) + "\n")


if __name__ == "__main__":
    main()
