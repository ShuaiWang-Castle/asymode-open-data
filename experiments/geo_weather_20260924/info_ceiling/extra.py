"""Supplementary, descriptive analyses added after the registered scoring (they change no registered rule):

  1. event-level S discrimination: design-weighted and unweighted AUC for the key pairs, with family-cluster intervals
     (2,000 draws within regime, seed 20260924);
  2. the trade-off frontier between the host and a probe: P_a = (1 - a) host + a probe for a fixed grid of a, with the
     S improvement, the non-S error and the severe false peaks at each a (a grid, not a choice made on held-out data).
Writes results/v1/info_ceiling/info_ceiling_extra.json."""
from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import roc_auc_score

from probe import BOOT_SEED, OUT, REGIMES, RES, host_paths, load_static

B = 2000


def main() -> None:
    s = load_static()
    y, m, w = s["y"].astype(float), s["m"].astype(float), s["w"].astype(float)
    reg, fam, peak = s["regime"], s["family"], s["peak"].astype(float)
    n = len(y); S = peak >= .10; nonS = ~S
    keys = sorted({(r, f) for r, f in zip(reg, fam)}); gid = {kk: j for j, kk in enumerate(keys)}
    g = np.array([gid[(r, f)] for r, f in zip(reg, fam)])
    members = [np.where(g == j)[0] for j in range(len(keys))]
    rng = np.random.default_rng(BOOT_SEED)
    draws = []
    for _ in range(B):
        idx = []
        for r in REGIMES:
            js = [gid[kk] for kk in keys if kk[0] == r]
            for j in rng.choice(js, len(js)):
                idx.append(members[j])
        draws.append(np.concatenate(idx))
    ev = np.load(OUT / "event_oof.npz")
    host5, _ = host_paths(n)
    score = {"host5": host5.max(1)}
    for k in ("M1", "M1H", "M2", "M2p", "M4", "M4p", "M5"):
        score[k] = ev[f"{k}_prob"]

    def auc(sc, idx, weighted):
        yy = S[idx]
        if yy.all() or not yy.any():
            return np.nan
        return roc_auc_score(yy, sc[idx], sample_weight=w[idx] if weighted else None)

    out = {"auc": {}, "auc_diff": {}}
    for k, sc in score.items():
        out["auc"][k] = dict(weighted=float(auc(sc, np.arange(n), True)), unweighted=float(auc(sc, np.arange(n), False)))
    for a, b in (("M1H", "M1"), ("M4", "M4p"), ("M2", "M2p"), ("M1", "host5")):
        res = {}
        for wt in (True, False):
            pt = auc(score[a], np.arange(n), wt) - auc(score[b], np.arange(n), wt)
            d = np.array([auc(score[a], ix, wt) - auc(score[b], ix, wt) for ix in draws])
            d = d[np.isfinite(d)]
            res["weighted" if wt else "unweighted"] = dict(point=float(pt), ci95=[float(np.quantile(d, .025)), float(np.quantile(d, .975))])
        out["auc_diff"][f"{a} - {b}"] = res
    # frontier
    sse_cnt = m.sum(1)

    def rmse(P, sub):
        e = np.where(m > 0, P - y, 0.0)
        return float(np.sqrt((w * (m * e ** 2).sum(1))[sub].sum() / (w * sse_cnt)[sub].sum()))

    out["frontier"] = {}
    for probe in ("M1R", "M5"):
        Pp = np.load(OUT / f"traj_{probe}.npy").astype(float)
        rows = []
        for a in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
            P = (1 - a) * host5 + a * Pp
            pk = P.max(1)
            rows.append(dict(a=a, S_improvement=1 - rmse(P, S) / rmse(host5, S), all_improvement=1 - rmse(P, np.ones(n, bool)) / rmse(host5, np.ones(n, bool)),
                             nonS_change=rmse(P, nonS) / rmse(host5, nonS) - 1, false_peaks=int(((pk >= .10) & nonS).sum()),
                             false_peaks_5pct=int(((pk >= .05) & nonS).sum())))
        out["frontier"][probe] = rows
    (RES / "info_ceiling_extra.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["auc"], indent=0))
    for k, v in out["auc_diff"].items():
        print(k, {kk: f"{vv['point']:+.3f} [{vv['ci95'][0]:+.3f}, {vv['ci95'][1]:+.3f}]" for kk, vv in v.items()})
    for probe, rows in out["frontier"].items():
        print(probe)
        for r in rows:
            print(f"  a={r['a']:.2f}  S {100 * r['S_improvement']:+.2f}%  all {100 * r['all_improvement']:+.2f}%  nonS {100 * r['nonS_change']:+.1f}%  "
                  f"false peaks >=10%: {r['false_peaks']}  >=5%: {r['false_peaks_5pct']}")


if __name__ == "__main__":
    main()
