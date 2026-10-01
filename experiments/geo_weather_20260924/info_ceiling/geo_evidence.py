"""Geography evidence probes (notes/GEO_EVIDENCE_PROBES_SCOPE_20261001_ZH.md): does geography act on (A) the allocation of
damage among counties within one storm, or (B) the speed of restoration after the peak? Each test against the
county-permuted null. Same learner and folds as probe.py; no existing file is modified.

  python geo_evidence.py a1 | a2 | b
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from probe import BOOT_SEED, B, EVENT_PARAMS, EVENT_ROUNDS, EXP, OUT, REGIMES, RES, ROOT, folds, host_paths, load_static

sys.path.insert(0, str(EXP / "paper_v1"))
from twins_v1 import peak3  # noqa: E402

ADJ = ROOT / "data" / "raw" / "census" / "county_adjacency2023.txt"
FEAT = ROOT / "data" / "interim" / "panel_v1" / "features_v1D.npz"


def fold_of_unit(n):
    f = np.zeros(n, int)
    for k, dev, outer in folds():
        f[outer] = k
    return f


def cluster_draws(groups_regime: list[tuple], rng, n_draw=B):
    """Resample group labels within regime; returns a list of index arrays into the group list."""
    by_reg = {}
    for j, (r, _) in enumerate(groups_regime):
        by_reg.setdefault(r, []).append(j)
    return [np.concatenate([rng.choice(v, len(v)) for v in by_reg.values()]) for _ in range(n_draw)]


# ------------------------------------------------------------------------------------------ A1
def a1() -> None:
    from scipy.stats import spearmanr
    s = load_static(); n = len(s["peak"])
    obs = s["peak"].astype(float); sysv, reg = s["system"].astype(str), s["regime"].astype(str)
    host5, _ = host_paths(n)
    preds = {"host5": host5.max(1)}
    for k in ("M1", "M2", "M2p", "M4", "M4p", "M5"):
        preds[k] = np.load(OUT / f"traj_{k}.npy").max(1)
    systems = [S for S in np.unique(sysv) if (sysv == S).sum() >= 10]
    rho = {k: np.array([spearmanr(v[sysv == S], obs[sysv == S]).statistic for S in systems]) for k, v in preds.items()}
    sreg = [(reg[sysv == S][0], S) for S in systems]
    draws = cluster_draws(sreg, np.random.default_rng(BOOT_SEED))
    out = {"systems": len(systems), "mean_spearman": {k: float(np.nanmean(v)) for k, v in rho.items()}, "differences": {}}
    for a, b in (("M2", "M2p"), ("M2", "M1"), ("M4", "M4p"), ("M1", "host5"), ("M5", "host5")):
        d = rho[a] - rho[b]
        bs = [np.nanmean(d[ix]) for ix in draws]
        out["differences"][f"{a} - {b}"] = dict(point=float(np.nanmean(d)), ci95=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                                                systems_better=int((d > 0).sum()), systems_worse=int((d < 0).sum()))
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "geo_evidence_a1.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


# ------------------------------------------------------------------------------------------ A2
def twin_pairs(s):
    z = np.load(FEAT)
    fips, sysv = z["fips"].astype(str), z["system"].astype(str)
    cust, obsf = z["cust"].astype(float), z["obs_full"].astype(bool)
    names = list(z["weather_channels"].astype(str))
    xu = z["xu"][:, 72:216, :len(names)]
    gust, prec = xu[..., names.index("gust")], xu[..., names.index("precip")]
    adj = pd.read_csv(ADJ, sep="|", dtype=str)
    nb = {(a, b) for a, b in zip(adj["County GEOID"], adj["Neighbor GEOID"]) if a != b}
    y = np.where(obsf, z["y_full"], np.nan).astype(float)
    pairs = []
    for S in np.unique(sysv):        # the rule of paper_v1/twins_v1.py
        ii = [i for i in np.where(sysv == S)[0] if cust[i] >= 5000 and obsf[i, 72:216].mean() >= 0.9]
        for a in range(len(ii)):
            for b in range(a + 1, len(ii)):
                i, j = ii[a], ii[b]
                if (fips[i], fips[j]) not in nb:
                    continue
                if np.corrcoef(gust[i], gust[j])[0, 1] < .95 or np.corrcoef(prec[i], prec[j])[0, 1] < .90:
                    continue
                gm = (gust[i].max(), gust[j].max()); pt = (prec[i].sum(), prec[j].sum())
                if abs(gm[0] - gm[1]) > .1 * max(gm) or abs(pt[0] - pt[1]) > .25 * max(max(pt), 1e-9):
                    continue
                pairs.append((i, j) if fips[i] < fips[j] else (j, i))
    p3 = np.array([peak3(row) for row in y])
    return np.array(pairs), p3


def a2() -> None:
    import lightgbm as lgb
    s = load_static(); n = len(s["peak"])
    pairs, p3 = twin_pairs(s)
    fold = fold_of_unit(n)
    i, j = pairs[:, 0], pairs[:, 1]
    dy = np.log(p3[i] + .005) - np.log(p3[j] + .005)
    ok = np.isfinite(dy)
    i, j, dy = i[ok], j[ok], dy[ok]
    pf = fold[i]
    blocks = {"B": s["B"], "W": s["We"], "G": s["G"], "Gp": s["Gp"], "H": s["H"], "Hp": s["Hp"], "C": s["C"]}
    variants = {"A2W": ["B", "W"], "A2G": ["B", "W", "G"], "A2Gp": ["B", "W", "Gp"], "A2H": ["B", "W", "H"], "A2Hp": ["B", "W", "Hp"],
                "A2GH": ["B", "W", "G", "H", "C"]}
    preds = {}
    for name, bl in variants.items():
        X = np.concatenate([blocks[b][i] - blocks[b][j] for b in bl], 1)
        P = np.full(len(dy), np.nan)
        for k in range(1, 6):
            tr, te = pf != k, pf == k
            Xa = np.concatenate([X[tr], -X[tr]]); ya = np.concatenate([dy[tr], -dy[tr]])
            bst = lgb.train(EVENT_PARAMS, lgb.Dataset(Xa, ya), EVENT_ROUNDS)
            P[te] = 0.5 * (bst.predict(X[te]) - bst.predict(-X[te]))        # exact antisymmetry
        preds[name] = P
        print("a2", name, X.shape, flush=True)
    fam = s["family"].astype(str)[i]; reg = s["regime"].astype(str)[i]
    keys = sorted(set(zip(reg, fam))); gid = {kk: q for q, kk in enumerate(keys)}
    g = np.array([gid[kk] for kk in zip(reg, fam)])
    members = [np.where(g == q)[0] for q in range(len(keys))]
    draws = [np.concatenate([members[q] for q in ix]) for ix in cluster_draws(keys, np.random.default_rng(BOOT_SEED))]
    clear = np.abs(dy) > np.log(2)

    def r2(P, ix):
        return 1 - np.sum((dy[ix] - P[ix]) ** 2) / np.sum(dy[ix] ** 2)

    def sign_acc(P, ix):
        ix = ix[clear[ix]]
        return float(np.mean(np.sign(P[ix]) == np.sign(dy[ix])))

    out = {"pairs": int(len(dy)), "clear_pairs": int(clear.sum()), "systems": int(len(set(s["system"][i]))), "metrics": {}, "differences": {}}
    allix = np.arange(len(dy))
    for k, P in preds.items():
        out["metrics"][k] = dict(r2=float(r2(P, allix)), sign_accuracy_clear=sign_acc(P, allix))
    for a, b in (("A2G", "A2Gp"), ("A2G", "A2W"), ("A2H", "A2Hp"), ("A2H", "A2W"), ("A2GH", "A2W")):
        for met, fn in (("r2", r2), ("sign_accuracy_clear", sign_acc)):
            bs = [fn(preds[a], ix) - fn(preds[b], ix) for ix in draws]
            out["differences"][f"{a} - {b} {met}"] = dict(point=float(fn(preds[a], allix) - fn(preds[b], allix)),
                                                          ci95=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))])
    (RES / "geo_evidence_a2.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


# ------------------------------------------------------------------------------------------ B
def b() -> None:
    import lightgbm as lgb
    s = load_static(); n = len(s["peak"])
    y, m, w = s["y"].astype(float), s["m"].astype(float), s["w"].astype(float)
    Wh = np.load(OUT / "Wh.npy"); wc = list(s["Whcols"])
    yo = np.where(m > 0, y, -1.0)
    tpk = yo.argmax(1); pk = yo.max(1)
    sel = np.where((pk >= .05) & (tpk + 24 <= 143))[0]
    sel = sel[m[sel, tpk[sel] + 24] > 0]
    r24 = np.clip(y[sel, tpk[sel] + 24] / pk[sel], 0, 1.5)
    post = []
    for u, t in zip(sel, tpk[sel]):
        seg = Wh[u, t:t + 25]
        post.append([seg[:, wc.index("gust")].max(), seg[:, wc.index("wind_speed")].max(), seg[:, wc.index("gust_excess_energy")].max(),
                     seg[:, wc.index("precip")].sum(), seg[:, wc.index("snowfall")].sum(), seg[:, wc.index("t2m_c")].min()])
    post = np.array(post, np.float32)
    base = np.column_stack([np.log(pk[sel]), tpk[sel], post]).astype(np.float32)
    blocks = {"P": base, "B": s["B"][sel], "C": s["C"][sel], "G": s["G"][sel], "Gp": s["Gp"][sel], "H": s["H"][sel], "Hp": s["Hp"][sel]}
    variants = {"BW": ["P", "B"], "BG": ["P", "B", "G"], "BGp": ["P", "B", "Gp"], "BH": ["P", "B", "H"], "BHp": ["P", "B", "Hp"],
                "BGH": ["P", "B", "G", "H", "C"]}
    fold = fold_of_unit(n)[sel]; ws = w[sel]
    preds = {}
    for name, bl in variants.items():
        X = np.concatenate([blocks[q] for q in bl], 1)
        P = np.full(len(sel), np.nan)
        for k in range(1, 6):
            tr, te = fold != k, fold == k
            bst = lgb.train(EVENT_PARAMS, lgb.Dataset(X[tr], r24[tr], weight=ws[tr]), EVENT_ROUNDS)
            P[te] = bst.predict(X[te])
        preds[name] = P
        print("b", name, X.shape, flush=True)
    fam, reg = s["family"].astype(str)[sel], s["regime"].astype(str)[sel]
    keys = sorted(set(zip(reg, fam))); gid = {kk: q for q, kk in enumerate(keys)}
    g = np.array([gid[kk] for kk in zip(reg, fam)])
    members = [np.where(g == q)[0] for q in range(len(keys))]
    draws = [np.concatenate([members[q] for q in ix]) for ix in cluster_draws(keys, np.random.default_rng(BOOT_SEED))]

    def rmse(P, ix):
        return float(np.sqrt(np.average((P[ix] - r24[ix]) ** 2, weights=ws[ix])))

    allix = np.arange(len(sel))
    mean_r = float(np.average(r24, weights=ws))
    out = {"units": int(len(sel)), "mean_R24": mean_r, "metrics": {}, "differences": {}}
    for k, P in preds.items():
        out["metrics"][k] = dict(rmse=rmse(P, allix), r2=float(1 - rmse(P, allix) ** 2 / np.average((r24 - mean_r) ** 2, weights=ws)))
    for a, bb in (("BG", "BGp"), ("BG", "BW"), ("BH", "BHp"), ("BH", "BW"), ("BGH", "BW")):
        bs = [1 - rmse(preds[a], ix) / rmse(preds[bb], ix) for ix in draws]
        out["differences"][f"{a} vs {bb}"] = dict(point=float(1 - rmse(preds[a], allix) / rmse(preds[bb], allix)),
                                                  ci95=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))])
    (RES / "geo_evidence_b.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"a1": a1, "a2": a2, "b": b}[sys.argv[1]]()
