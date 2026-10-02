"""County vulnerability vector for the dose-fragility kernel and the event-level tests of the geography prior
(notes/DOSE_FRAGILITY_KERNEL_DESIGN_20261002_ZH.md, sections 2-3).

  python vuln_prior.py build   history (EAGLE-I 2014-2017, NaN-safe), geography prior muG (cross-fitted by state), the
                               prior from permuted geography muGp, the shrunk posterior; writes
                               data/interim/panel_v1/vuln_v1D.npz (z [N, 7], z_perm) and runs/.../vuln_prior.npz
  python vuln_prior.py test    twin pairs, severe-risk Brier and trajectories with muG against muGp
"""
from __future__ import annotations

import json
import sys

import numpy as np

from geo_evidence import cluster_draws, fold_of_unit, twin_pairs
from probe import (BOOT_SEED, B, EVENT_PARAMS, EVENT_ROUNDS, OUT, REGIMES, RES, ROOT, TRAJ_PARAMS, TRAJ_ROUNDS,
                   county_permutation, folds, load_static, traj_rows)
from probe2 import eaglei_daily

VULN = ROOT / "data" / "interim" / "panel_v1" / "vuln_v1D.npz"
KAPPA = 2.0
PRIOR_SEED = 20261002


def build() -> None:
    import lightgbm as lgb
    s = load_static()
    fips = s["fips"].astype(str)
    first = {f: i for i, f in reversed(list(enumerate(fips)))}
    cty = np.array(sorted(first)); idx = np.array([first[c] for c in cty])
    G = s["G"][idx]
    donor = county_permutation(fips)
    pos = {c: q for q, c in enumerate(cty)}
    Gp = np.stack([G[pos[donor[c]]] for c in cty])
    # NaN-safe history per county
    d = eaglei_daily(set(cty.tolist())).dropna(subset=["frac"])
    yrs = d.groupby("fips")["year"].nunique()
    n10 = d[d["frac"] >= .10].groupby("fips").size()
    n5 = d[d["frac"] >= .05].groupby("fips").size()
    top5 = d.sort_values("frac", ascending=False).groupby("fips")["frac"].apply(lambda x: float(x.head(5).mean()))
    years = yrs.reindex(cty).fillna(0).to_numpy(float)
    has = years >= 1
    hist = np.column_stack([np.log(n10.reindex(cty).fillna(0).to_numpy(float) / np.maximum(years, 1) + 0.1),
                            np.log(n5.reindex(cty).fillna(0).to_numpy(float) + 1.0),
                            np.log(top5.reindex(cty).fillna(0).to_numpy(float) + 1e-3)])
    hist[~has] = np.nan
    train = years >= 3
    state = np.array([c[:2] for c in cty])
    rng = np.random.default_rng(PRIOR_SEED)
    sts = np.unique(state[train]); grp = dict(zip(sts, rng.integers(0, 5, len(sts))))
    fold = np.array([grp.get(st, -1) for st in state])
    params = dict(EVENT_PARAMS, min_data_in_leaf=20)

    def prior(X):
        mu = np.full((len(cty), 3), np.nan)
        for t in range(3):
            for k in range(5):
                tr, te = train & (fold != k), train & (fold == k)
                mu[te, t] = lgb.train(params, lgb.Dataset(X[tr], hist[tr, t]), 300).predict(X[te])
            full = lgb.train(params, lgb.Dataset(X[train], hist[train, t]), 300)
            mu[~train, t] = full.predict(X[~train])            # counties never used to fit their own prior
        return mu

    muG, muGp = prior(G), prior(Gp)
    r2 = {f"t{t}": float(1 - np.sum((hist[train, t] - muG[train, t]) ** 2) / np.sum((hist[train, t] - hist[train, t].mean()) ** 2)) for t in range(3)}
    r2p = {f"t{t}": float(1 - np.sum((hist[train, t] - muGp[train, t]) ** 2) / np.sum((hist[train, t] - hist[train, t].mean()) ** 2)) for t in range(3)}
    hfill = np.where(np.isnan(hist), muG, hist)
    post = (years[:, None] * hfill + KAPPA * muG) / (years[:, None] + KAPPA)
    z = np.concatenate([muG, hfill, has[:, None].astype(float)], 1)
    ui = np.array([pos[f] for f in fips])
    up = np.array([pos[donor[f]] for f in fips])
    np.savez(VULN, fips=fips, system=s["system"], z=z[ui].astype(np.float32), z_perm=z[up].astype(np.float32),
             cols=np.array(["mu_n10", "mu_n5", "mu_top5", "hist_n10", "hist_n5", "hist_top5", "has_history"]))
    hp = np.where(np.isnan(hist), muGp, hist)
    np.savez(OUT / "vuln_prior.npz", muG=muG[ui].astype(np.float32), muGp=muGp[ui].astype(np.float32), post=post[ui].astype(np.float32),
             hist=hist[ui].astype(np.float32), years=years[ui].astype(np.float32),
             post_p=((years[:, None] * hp + KAPPA * muGp) / (years[:, None] + KAPPA))[ui].astype(np.float32))
    man = dict(counties=int(len(cty)), with_history=int(has.sum()), train_counties=int(train.sum()), prior_r2_out_of_state=r2,
               prior_r2_permuted=r2p, units_without_history=int((~has[ui]).sum()), units_thin_history=int((years[ui] <= 2).sum()))
    (OUT / "vuln_prior_manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    print(json.dumps(man, indent=1))


def test() -> None:
    import lightgbm as lgb
    from scipy.stats import spearmanr
    s = load_static(); v = dict(np.load(OUT / "vuln_prior.npz")); n = len(s["peak"])
    out = {}
    # --- twin pairs
    pairs, p3 = twin_pairs(s)
    fold = fold_of_unit(n)
    i, j = pairs[:, 0], pairs[:, 1]
    dy = np.log(p3[i] + .005) - np.log(p3[j] + .005); ok = np.isfinite(dy); i, j, dy = i[ok], j[ok], dy[ok]
    pf = fold[i]
    blocks = {"B": s["B"], "W": s["We"], "H": s["H"], "muG": v["muG"], "muGp": v["muGp"], "post": v["post"], "post_p": v["post_p"]}
    var = {"W": ["B", "W"], "W+muG": ["B", "W", "muG"], "W+muGp": ["B", "W", "muGp"], "W+H": ["B", "W", "H"], "W+H+muG": ["B", "W", "H", "muG"],
           "W+post": ["B", "W", "post"], "W+post_p": ["B", "W", "post_p"]}
    pr = {}
    for name, bl in var.items():
        X = np.concatenate([blocks[b][i] - blocks[b][j] for b in bl], 1); P = np.full(len(dy), np.nan)
        for k in range(1, 6):
            tr, te = pf != k, pf == k
            bst = lgb.train(EVENT_PARAMS, lgb.Dataset(np.concatenate([X[tr], -X[tr]]), np.concatenate([dy[tr], -dy[tr]])), EVENT_ROUNDS)
            P[te] = 0.5 * (bst.predict(X[te]) - bst.predict(-X[te]))
        pr[name] = P
    fam, reg = s["family"].astype(str)[i], s["regime"].astype(str)[i]
    keys = sorted(set(zip(reg, fam))); gid = {kk: q for q, kk in enumerate(keys)}
    g = np.array([gid[kk] for kk in zip(reg, fam)]); members = [np.where(g == q)[0] for q in range(len(keys))]
    draws = [np.concatenate([members[q] for q in ix]) for ix in cluster_draws(keys, np.random.default_rng(BOOT_SEED))]
    clear = np.abs(dy) > np.log(2)
    thin = (v["years"][i] <= 2) | (v["years"][j] <= 2)

    def acc(P, ix, sub=None):
        mk = clear[ix] if sub is None else (clear[ix] & sub[ix])
        ix = ix[mk]
        return float(np.mean(np.sign(P[ix]) == np.sign(dy[ix]))) if len(ix) else np.nan

    allix = np.arange(len(dy))
    tw = {"pairs": int(len(dy)), "clear": int(clear.sum()), "clear_thin_history": int((clear & thin).sum()),
          "accuracy": {k: acc(P, allix) for k, P in pr.items()}, "accuracy_thin": {k: acc(P, allix, thin) for k, P in pr.items()}, "diff": {}}
    for a, b in (("W+muG", "W+muGp"), ("W+muG", "W"), ("W+H+muG", "W+H"), ("W+post", "W+H"), ("W+post", "W+post_p")):
        for tag, sub in (("all", None), ("thin", thin)):
            bs = [acc(pr[a], ix, sub) - acc(pr[b], ix, sub) for ix in draws]
            tw["diff"][f"{a} - {b} [{tag}]"] = dict(point=acc(pr[a], allix, sub) - acc(pr[b], allix, sub),
                                                    ci95=[float(np.nanquantile(bs, .025)), float(np.nanquantile(bs, .975))])
    out["twins"] = tw
    # --- severe risk (event classifier)
    peak, w = s["peak"].astype(float), s["w"].astype(float); S = (peak >= .10).astype(float)
    regu, famu = s["regime"].astype(str), s["family"].astype(str)
    keys = sorted(set(zip(regu, famu))); gid = {kk: q for q, kk in enumerate(keys)}; gu = np.array([gid[kk] for kk in zip(regu, famu)])
    rng = np.random.default_rng(BOOT_SEED); counts = np.zeros((B, len(keys)))
    for r in REGIMES:
        js = np.array([gid[kk] for kk in keys if kk[0] == r]); counts[:, js] = rng.multinomial(len(js), np.full(len(js), 1 / len(js)), size=B)
    ev = {"M1": ["B", "W"], "M1+muG": ["B", "W", "muG"], "M1+muGp": ["B", "W", "muGp"], "M1H": ["B", "W", "H"], "M1H+muG": ["B", "W", "H", "muG"]}
    prob = {}
    for name, bl in ev.items():
        X = np.concatenate([blocks[b] for b in bl], 1); p = np.full(n, np.nan)
        for k, dev, outer in folds():
            okk = peak[dev] >= 0
            p[outer] = lgb.train(dict(EVENT_PARAMS, objective="binary"), lgb.Dataset(X[dev][okk], S[dev][okk], weight=w[dev][okk]), EVENT_ROUNDS).predict(X[outer])
        prob[name] = np.clip(p, 1e-4, 1 - 1e-4)

    def gs(val, wt):
        o = np.zeros(len(keys)); np.add.at(o, gu, val * wt); return o
    risk = {}
    for a, b in (("M1+muG", "M1+muGp"), ("M1+muG", "M1"), ("M1H+muG", "M1H")):
        for tag, wt in (("weighted", w), ("unweighted", np.ones(n))):
            la, lb = (prob[a] - S) ** 2, (prob[b] - S) ** 2
            d = 1 - (counts @ gs(la, wt)) / (counts @ gs(lb, wt))
            risk[f"{a} vs {b} brier {tag}"] = dict(point=float(1 - (la * wt).sum() / (lb * wt).sum()), ci95=[float(np.quantile(d, .025)), float(np.quantile(d, .975))])
    out["risk"] = risk
    # --- trajectories
    Wh, Rh = np.load(OUT / "Wh.npy"), np.load(OUT / "Rh.npy")
    y, m = s["y"], s["m"]
    s3 = dict(s); s3["muG"], s3["muGp"] = v["muG"], v["muGp"]
    traj = {}
    for name, bl in (("M1muG", ["B", "W", "muG"]), ("M1muGp", ["B", "W", "muGp"])):
        f = OUT / f"traj_{name}.npy"
        if not f.exists():
            P = np.full(y.shape, np.nan, np.float32)
            for k, dev, outer in folds():
                X, ui, ti = traj_rows(s3, Wh, Rh, bl, dev, True); okk = m[ui, ti] > 0
                bst = lgb.train(TRAJ_PARAMS, lgb.Dataset(X[okk], y[ui, ti][okk], weight=s["w"][ui][okk]), TRAJ_ROUNDS)
                Xo, uo, to = traj_rows(s3, Wh, Rh, bl, outer, False); P[uo, to] = np.clip(bst.predict(Xo), 0, 1)
            np.save(f, P)
        traj[name] = np.load(f).astype(float)
    yy, mm = y.astype(float), m.astype(float); Sb = peak >= .10
    sse = {k: (mm * np.where(mm > 0, P - yy, 0) ** 2).sum(1) for k, P in traj.items()}; cnt = mm.sum(1)

    def rel(sub):
        def gsum(val):
            o = np.zeros(len(keys)); np.add.at(o, gu[sub], (w * val)[sub]); return o
        qa, qb = counts @ gsum(sse["M1muG"]), counts @ gsum(sse["M1muGp"])
        d = 1 - np.sqrt(qa / qb)
        return dict(point=float(1 - np.sqrt((w * sse["M1muG"])[sub].sum() / (w * sse["M1muGp"])[sub].sum())), ci95=[float(np.quantile(d, .025)), float(np.quantile(d, .975))])
    sysv = s["system"].astype(str); systems = [q for q in np.unique(sysv) if (sysv == q).sum() >= 10]
    rho = {k: np.array([spearmanr(P.max(1)[sysv == q], peak[sysv == q]).statistic for q in systems]) for k, P in traj.items()}
    dr = rho["M1muG"] - rho["M1muGp"]
    sd = cluster_draws([(regu[sysv == q][0], q) for q in systems], np.random.default_rng(BOOT_SEED))
    bs = [np.nanmean(dr[ix]) for ix in sd]
    out["trajectory"] = dict(S=rel(Sb), all=rel(np.ones(n, bool)), spearman=dict(muG=float(np.nanmean(rho["M1muG"])), muGp=float(np.nanmean(rho["M1muGp"])),
                             diff=float(np.nanmean(dr)), ci95=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]))
    (RES / "geo_prior_tests.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"build": build, "test": test}[sys.argv[1]]()
