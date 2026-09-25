"""Seed-replicated screen: each arm against the base of the same seed, per seed and pooled.

  python evaluate_seeds.py --feat data/interim/geo_weather/features_w1.npz --arms Hq Hp \
      --labels "w1_{arm}" "w1s1_{arm}" "w1s2_{arm}" --folds 1 2 --out results/w1_seeds.json
For every seed label pattern the base is the pattern with arm = base. Reports, per seed, the relative change of
pooled RMSE and the county-cluster interval (evaluate_screen.cluster_ratio); across seeds, the mean change and the
change of the seed-averaged prediction (the mean of the seeds' rollouts, arm against base), which removes most of
the seed noise of a single run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import evaluate_screen as ES

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def preds(label: str, folds) -> tuple[np.ndarray, np.ndarray]:
    idx, P = [], []
    for k in folds:
        z = np.load(ES.RUNS / label / f"fold{k:02d}" / "outer.npz")
        idx.append(z["idx"]); P.append(z["P"])
    idx = np.concatenate(idx); o = np.argsort(idx)
    return idx[o], np.concatenate(P)[o]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True, help="patterns with {arm}")
    ap.add_argument("--folds", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    z = np.load(ROOT / a.feat)
    y, m, fips = np.nan_to_num(z["y"]).astype(np.float64), z["m"].astype(np.float64), z["fips"].astype(str)
    seeds = []
    for pat in a.labels:
        try:
            ib, pb = preds(pat.format(arm="base"), a.folds)
        except FileNotFoundError:
            continue
        row = dict(pattern=pat, base=pb, arms={})
        for arm in a.arms:
            try:
                ia, pa = preds(pat.format(arm=arm), a.folds)
            except FileNotFoundError:
                continue
            assert np.array_equal(ia, ib)
            row["arms"][arm] = pa
        row["idx"] = ib
        seeds.append(row)
    idx = seeds[0]["idx"]; yy, mm = y[idx], m[idx]; n = mm.sum(1)
    se = lambda P: ((P - yy) ** 2 * mm).sum(1)
    out = dict(folds=a.folds, seeds=[s["pattern"] for s in seeds], per_seed={}, pooled={})
    rng = np.random.default_rng(ES.SEED)
    for arm in a.arms:
        rows = []
        for s in seeds:
            if arm not in s["arms"]:
                continue
            sb, sa = se(s["base"]), se(s["arms"][arm])
            rel = float(np.sqrt(sa.sum() / n.sum()) / np.sqrt(sb.sum() / n.sum()) - 1)
            ci, _ = ES.cluster_ratio(sa, sb, n, fips[idx], rng)
            rows.append(dict(pattern=s["pattern"], rel=rel, county_ci=ci))
        out["per_seed"][arm] = rows
        have = [s for s in seeds if arm in s["arms"]]
        if have:
            Pb = np.mean([s["base"] for s in have], 0); Pa = np.mean([s["arms"][arm] for s in have], 0)
            ci, p = ES.cluster_ratio(se(Pa), se(Pb), n, fips[idx], rng)
            out["pooled"][arm] = dict(n_seeds=len(have), mean_rel=float(np.mean([r["rel"] for r in rows])),
                                      ensemble_rel=float(np.sqrt(se(Pa).sum() / n.sum()) / np.sqrt(se(Pb).sum() / n.sum()) - 1),
                                      ensemble_county_ci=ci)
    for arm in a.arms:
        for r in out["per_seed"][arm]:
            print(f"{arm:6s} {r['pattern']:14s} {100 * r['rel']:+.2f}%  county [{100 * r['county_ci'][0]:+.2f}, {100 * r['county_ci'][1]:+.2f}]")
        if arm in out["pooled"]:
            q = out["pooled"][arm]
            print(f"{arm:6s} mean over {q['n_seeds']} seeds {100 * q['mean_rel']:+.2f}%; seed-averaged prediction "
                  f"{100 * q['ensemble_rel']:+.2f}% [{100 * q['ensemble_county_ci'][0]:+.2f}, {100 * q['ensemble_county_ci'][1]:+.2f}]")
    if a.out:
        (HERE / a.out).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
