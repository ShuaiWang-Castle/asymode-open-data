"""Restoration-capacity probes (notes/RESTORATION_CAPACITY_PROBE_SCOPE_20261002_ZH.md). Public D only; no training of
the neural models; no existing file is modified.

  python restore_capacity.py size   observed restoration after the peak by outage size and regime; the host's and the
                                    dose-kernel arm's restoration rate at the observed peak (descriptive)
  python restore_capacity.py sim    post-hoc simulation on the host's held-out rates (seed 0): first-order restoration
                                    r p replaced by the capacity-limited flow r p / (1 + kappa r p) (descriptive)
  python restore_capacity.py geo    registered test R1: does geography predict the restoration of large outages?
"""
from __future__ import annotations

import json
import sys

import numpy as np

from geo_evidence import cluster_draws, fold_of_unit
from probe import BOOT_SEED, EVENT_PARAMS, EVENT_ROUNDS, FEAT, OUT, REGIMES, RES, ROOT, load_static
from score_dose import collect

BINS = [(.005, .02), (.02, .05), (.05, .10), (.10, .25), (.25, 1.01)]
KAPPAS = [0, 5, 10, 20, 40, 80, 160, 320]


def wq(v, ww, q=.5):
    o = np.argsort(v); c = np.cumsum(ww[o]) / ww.sum()
    return float(v[o][min(np.searchsorted(c, q), len(v) - 1)])


def base():
    s = load_static()
    y, m, w = s["y"].astype(float), s["m"].astype(float), s["w"].astype(float)
    reg, peak = s["regime"].astype(str), s["peak"].astype(float)
    yo = np.where(m > 0, y, -1.0); tpk = yo.argmax(1)
    ok = (tpk + 24 <= 143) & (m[np.arange(len(y)), np.minimum(tpk + 24, 143)] > 0)
    return s, y, m, w, reg, peak, tpk, ok


def size() -> None:
    s, y, m, w, reg, peak, tpk, ok = base(); n = len(y)
    host = collect("v1_host_s0", n); dkv = collect("v1_dkv_s0", n)
    out = {"bins": BINS, "note": "R24 = outage fraction 24 h after the observed peak over the peak (design-weighted median); "
           "implied hourly rate = 1 - R24^(1/24), floor of R24 1e-4; model rates are medians at the observed peak hour", "by_regime": {}}
    for r in REGIMES:
        rows = []
        for lo, hi in BINS:
            f = ok & (reg == r) & (peak >= lo) & (peak < hi)
            if f.sum() < 8:
                rows.append(dict(n=int(f.sum()))); continue
            r24 = y[f, tpk[f] + 24] / peak[f]; med = wq(r24, w[f])
            rows.append(dict(n=int(f.sum()), R24=med, R24_unweighted=float(np.median(r24)), share_R24_above_0p2=float(np.average(r24 > .2, weights=w[f])),
                             implied_rate=float(1 - max(med, 1e-4) ** (1 / 24)), host_rate=float(np.median(host[2][f, tpk[f]])),
                             dkv_rate=float(np.median(dkv[2][f, tpk[f]])) if len(dkv[3]) == 5 else None))
        out["by_regime"][r] = rows
        print(f"{r:14s} " + " | ".join(f"n {q['n']:3d} R24 {q['R24']:.2f} (>0.2: {q['share_R24_above_0p2']:.2f}) rate {q['implied_rate']:.3f} host {q['host_rate']:.3f}" if "R24" in q else f"n {q['n']:3d}" for q in rows))
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "restore_size.json").write_text(json.dumps(out, indent=1) + "\n")


def sim() -> None:
    s, y, m, w, reg, peak, tpk, ok = base(); n = len(y)
    y0 = np.load(FEAT, allow_pickle=False)["y0"].astype(float)
    P0, U, R, _, _ = collect("v1_host_s0", n)
    S = peak >= .10; A = np.ones(n, bool); cnt = m.sum(1); fold = fold_of_unit(n)

    def roll(kappa):
        p = y0.copy(); out = np.empty_like(U)
        for t in range(144):
            p = np.clip(p + U[:, t] * (1 - p) - R[:, t] * p / (1 + kappa * R[:, t] * p), 0, 1); out[:, t] = p
        return out

    def rmse(P, sub):
        return float(np.sqrt((w * (m * (P - y) ** 2).sum(1))[sub].sum() / (w * cnt)[sub].sum()))

    def row(P):
        pk = P.max(1)
        return dict(S=1 - rmse(P, S) / rmse(P0, S), all=1 - rmse(P, A) / rmse(P0, A), nonS=1 - rmse(P, ~S) / rmse(P0, ~S),
                    false_peaks=int(((pk >= .1) & ~S).sum()),
                    S_by_regime={r: 1 - rmse(P, S & (reg == r)) / rmse(P0, S & (reg == r)) for r in REGIMES},
                    all_by_regime={r: 1 - rmse(P, reg == r) / rmse(P0, reg == r) for r in REGIMES})

    sims = {k: roll(float(k)) for k in KAPPAS}
    out = {"reproduction_max_abs_diff": float(np.abs(sims[0] - P0).max()), "origin_stock_share_ge_1pct": dict(S=float((y0[S] >= .01).mean()), nonS=float((y0[~S] >= .01).mean())),
           "S_peak_hour_after_origin_median": {r: float(np.median(tpk[S & (reg == r)])) for r in REGIMES},
           "global": {str(k): row(sims[k]) for k in KAPPAS}}
    Pg, Pr, cg, cr = np.empty_like(P0), np.empty_like(P0), {}, {}
    for k in range(1, 6):
        cg[k] = min(KAPPAS, key=lambda q: rmse(sims[q], fold != k)); Pg[fold == k] = sims[cg[k]][fold == k]
        for r in REGIMES:
            cr[f"{k}:{r}"] = min(KAPPAS, key=lambda q: rmse(sims[q], (fold != k) & (reg == r)))
            Pr[(fold == k) & (reg == r)] = sims[cr[f"{k}:{r}"]][(fold == k) & (reg == r)]
    out["cross_fitted_global"] = dict(chosen=cg, **row(Pg)); out["cross_fitted_by_regime"] = dict(chosen=cr, **row(Pr))
    (RES / "restore_saturation_sim.json").write_text(json.dumps(out, indent=1) + "\n")
    pc = lambda d: f"S {100 * d['S']:+.2f}% all {100 * d['all']:+.2f}% nonS {100 * d['nonS']:+.2f}% FP {d['false_peaks']}"  # noqa: E731
    for k in KAPPAS:
        print(f"kappa {k:4d}: {pc(out['global'][str(k)])}")
    print("cross-fitted global:", cg, pc(out["cross_fitted_global"]))
    print("cross-fitted by regime:", pc(out["cross_fitted_by_regime"]), {r: f"{100 * v:+.1f}%" for r, v in out["cross_fitted_by_regime"]["S_by_regime"].items()},
          "all:", {r: f"{100 * v:+.1f}%" for r, v in out["cross_fitted_by_regime"]["all_by_regime"].items()})


def geo() -> None:
    import lightgbm as lgb
    from scipy.stats import spearmanr
    s, y, m, w, reg, peak, tpk, ok = base(); n = len(y)
    fam, sysv = s["family"].astype(str), s["system"].astype(str)
    Wh = np.load(OUT / "Wh.npy"); wc = list(s["Whcols"].astype(str))
    zv = np.load(ROOT / "data" / "interim" / "panel_v1" / "vuln_v1D.npz")
    fold = fold_of_unit(n)
    out = {}
    for pop, sel in (("A_tropical_winter", ok & (peak >= .10) & np.isin(reg, ["tropical", "winter"])), ("B_all_regimes", ok & (peak >= .10))):
        ix = np.where(sel)[0]
        seg_ok = np.array([m[u, t + 1:t + 25].sum() >= 12 for u, t in zip(ix, tpk[ix])]); ix = ix[seg_ok]
        a24 = np.clip(np.array([(m[u, t + 1:t + 25] * y[u, t + 1:t + 25]).sum() / m[u, t + 1:t + 25].sum() for u, t in zip(ix, tpk[ix])]) / peak[ix], 0, 1.5)
        r24 = np.clip(y[ix, tpk[ix] + 24] / peak[ix], 0, 1.5)
        post = []
        for u, t in zip(ix, tpk[ix]):
            seg = Wh[u, t:t + 25]
            post.append([seg[:, wc.index("gust")].max(), seg[:, wc.index("wind_speed")].max(), seg[:, wc.index("gust_excess_energy")].max(),
                         seg[:, wc.index("precip")].sum(), seg[:, wc.index("snowfall")].sum(), seg[:, wc.index("t2m_c")].min()])
        Pb = np.column_stack([np.log(peak[ix]), tpk[ix], np.array(post, np.float32)]).astype(np.float32)
        blocks = {"P": Pb, "B": s["B"][ix], "C": s["C"][ix], "G": s["G"][ix], "Gp": s["Gp"][ix], "H": s["H"][ix], "Hp": s["Hp"][ix],
                  "V": zv["z"][ix].astype(np.float32), "Vp": zv["z_perm"][ix].astype(np.float32)}
        variants = {"BW": ["P", "B"], "BG": ["P", "B", "G"], "BGp": ["P", "B", "Gp"], "BH": ["P", "B", "H"], "BHp": ["P", "B", "Hp"],
                    "BV": ["P", "B", "V"], "BVp": ["P", "B", "Vp"], "BC": ["P", "B", "C"]}
        f, ws = fold[ix], w[ix]
        keys = sorted(set(zip(reg[ix], fam[ix]))); gid = {kk: q for q, kk in enumerate(keys)}
        g = np.array([gid[kk] for kk in zip(reg[ix], fam[ix])]); members = [np.where(g == q)[0] for q in range(len(keys))]
        draws = [np.concatenate([members[q] for q in d]) for d in cluster_draws(keys, np.random.default_rng(BOOT_SEED))]
        res = dict(units=int(len(ix)), systems=int(len(np.unique(sysv[ix]))), families=len(keys), by_regime={r: int((reg[ix] == r).sum()) for r in REGIMES})
        for tname, tgt in (("A24", a24), ("R24", r24)):
            preds = {}
            for name, bl in variants.items():
                X = np.concatenate([blocks[q] for q in bl], 1); P = np.full(len(ix), np.nan)
                for k in range(1, 6):
                    tr, te = f != k, f == k
                    if te.sum() == 0:
                        continue
                    P[te] = lgb.train(EVENT_PARAMS, lgb.Dataset(X[tr], tgt[tr], weight=ws[tr]), EVENT_ROUNDS).predict(X[te])
                preds[name] = P

            def rmse(P, j):
                return float(np.sqrt(np.average((P[j] - tgt[j]) ** 2, weights=ws[j])))

            al = np.arange(len(ix)); mu = float(np.average(tgt, weights=ws)); var = float(np.average((tgt - mu) ** 2, weights=ws))
            d = dict(mean=mu, sd=float(np.sqrt(var)), r2={k: float(1 - rmse(P, al) ** 2 / var) for k, P in preds.items()}, differences={})
            for a, b in (("BG", "BGp"), ("BG", "BW"), ("BH", "BHp"), ("BH", "BW"), ("BV", "BVp"), ("BV", "BW"), ("BC", "BW")):
                bs = [1 - rmse(preds[a], j) / rmse(preds[b], j) for j in draws]
                d["differences"][f"{a} vs {b}"] = dict(point=float(1 - rmse(preds[a], al) / rmse(preds[b], al)), ci95=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))])
            res[tname] = d
            print(pop, tname, f"n {len(ix)} mean {mu:.3f} sd {np.sqrt(var):.3f} | R2", {k: round(v, 3) for k, v in d["r2"].items()})
            print("   ", {k: f"{100 * v['point']:+.2f}% [{100 * v['ci95'][0]:+.2f}, {100 * v['ci95'][1]:+.2f}]" for k, v in d["differences"].items()})
        # descriptive: within-system partial Spearman of A24 with six county descriptors, controlling for log peak
        gc, cc = list(s["Gcols"].astype(str)), list(s["Ccols"].astype(str))
        desc = {"canopy_mean": s["G"][ix, gc.index("geo_canopy_mean")], "forest_frac": s["G"][ix, gc.index("geo_forest_frac")],
                "developed_frac": s["G"][ix, gc.index("geo_developed_frac")], "relief": s["G"][ix, gc.index("geo_relief_p95_p5")],
                "log_pop_density": s["C"][ix, cc.index("log_pop_density")], "rucc": s["C"][ix, cc.index("rucc")]}
        tab = {}
        for name, v in desc.items():
            rho = []
            for q in np.unique(sysv[ix]):
                j = np.where(sysv[ix] == q)[0]
                if len(j) < 8 or np.std(v[j]) == 0 or np.std(a24[j]) == 0:
                    continue
                rk = lambda x: np.argsort(np.argsort(x)).astype(float)  # noqa: E731
                lp = rk(np.log(peak[ix][j])); ra = rk(a24[j]); rv = rk(v[j].astype(float))
                ea = ra - np.polyval(np.polyfit(lp, ra, 1), lp); ev = rv - np.polyval(np.polyfit(lp, rv, 1), lp)
                rho.append(float(np.corrcoef(ea, ev)[0, 1]))
            rho = np.array(rho)
            tab[name] = dict(systems=int(len(rho)), mean=float(np.nanmean(rho)), positive=int((rho > 0).sum()), negative=int((rho < 0).sum()))
        res["within_system_partial_spearman_A24"] = tab
        print("    within-system partial Spearman (A24):", {k: f"{v['mean']:+.3f} ({v['positive']}+/{v['negative']}-)" for k, v in tab.items()})
        out[pop] = res
    (RES / "restore_geo.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    {"size": size, "sim": sim, "geo": geo}[sys.argv[1]]()
