"""Scores of the dose-fragility kernel screen (notes/DOSE_FRAGILITY_KERNEL_DESIGN_20261002_ZH.md), seed 0, event folds.

Each arm is compared with the host of the same seed on the county-events of the folds that arm has finished (all five
when complete): full 144 h design-weighted RMSE on S, all and non-S, severe false peaks, S by regime, S peak ratio,
within-storm Spearman, sign accuracy on the weather-twin pairs, the learned kernel parameters, and the host's and the
arm's restoration and damage rates at the observed peak of tropical severe county-events. Intervals: family-cluster
bootstrap within regime, 2,000 draws, seed 20260924.
usage: python score_dose.py [labels ...]  -> results/v1/info_ceiling/dose_scores.json"""
from __future__ import annotations

import json
import sys

import numpy as np
from scipy.stats import spearmanr

from geo_evidence import cluster_draws, twin_pairs
from probe import BOOT_SEED, B, OUT, REGIMES, RES, RUNS, load_static

ARMS = {"DKV": "v1_dkv_s0", "DK": "v1_dk_s0", "DKVp": "v1_dkvp_s0", "DKVr": "v1_dkvr_s0", "hostF": "v1_hostf_s0"}
REFS = {"host0": "v1_host_s0", "GCRK": "v1_gcrk_s0"}


def collect(label, n):
    P = np.full((n, 144), np.nan); U = np.full((n, 144), np.nan); R = np.full((n, 144), np.nan); folds, meta = [], {}
    for k in range(1, 6):
        d = RUNS / label / f"fold{k:02d}"
        if (d / "DONE.json").exists():
            z = np.load(d / "outer.npz"); P[z["idx"]] = z["P"]; U[z["idx"]] = z["u"]; R[z["idx"]] = z["r"]; folds.append(k)
            meta[k] = {kk: vv for kk, vv in json.loads((d / "DONE.json").read_text()).items() if kk.startswith("dose_") or kk in ("seconds", "fit_loss")}
    return P, U, R, folds, meta


def main(names) -> None:
    s = load_static()
    y, m, w = s["y"].astype(float), s["m"].astype(float), s["w"].astype(float)
    reg, fam, peak, sysv = s["regime"].astype(str), s["family"].astype(str), s["peak"].astype(float), s["system"].astype(str)
    n = len(y); S = peak >= .10
    data = {k: collect(lab, n) for k, lab in {**REFS, **{a: ARMS[a] for a in names}}.items()}
    host5 = np.mean([collect(f"v1_host_s{q}", n)[0] for q in range(5)], 0)
    keys = sorted(set(zip(reg, fam))); gid = {kk: j for j, kk in enumerate(keys)}; g = np.array([gid[kk] for kk in zip(reg, fam)])
    rng = np.random.default_rng(BOOT_SEED); counts = np.zeros((B, len(keys)))
    for r in REGIMES:
        js = np.array([gid[kk] for kk in keys if kk[0] == r]); counts[:, js] = rng.multinomial(len(js), np.full(len(js), 1 / len(js)), size=B)
    cnt = m.sum(1)

    def sse(P):
        return (m * np.where(m > 0, np.nan_to_num(P) - y, 0) ** 2).sum(1)

    def rmse(P, sub):
        return float(np.sqrt((w * sse(P))[sub].sum() / (w * cnt)[sub].sum()))

    def rel(Pa, Pb, sub):
        def gs(v):
            o = np.zeros(len(keys)); np.add.at(o, g[sub], (w * v)[sub]); return o
        qa, qb = counts @ gs(sse(Pa)), counts @ gs(sse(Pb))
        ok = qb > 0
        d = 1 - np.sqrt(qa[ok] / qb[ok])
        return dict(point=1 - rmse(Pa, sub) / rmse(Pb, sub), ci95=[float(np.quantile(d, .025)), float(np.quantile(d, .975))])

    yo = np.where(m > 0, y, -1.0); tpk = yo.argmax(1)
    pairs, p3 = twin_pairs(s)
    dy = np.log(p3[pairs[:, 0]] + .005) - np.log(p3[pairs[:, 1]] + .005)
    clear = np.isfinite(dy) & (np.abs(dy) > np.log(2))
    out = {}
    host0 = data["host0"][0]
    for name in ["GCRK"] + list(names):
        P, U, R, folds, meta = data[name]
        if not folds:
            continue
        av = ~np.isnan(P).any(1)
        allu, Su, nSu = av, av & S, av & ~S
        pk = np.where(av, np.nan_to_num(P).max(1), np.nan); pk0 = host0.max(1)
        res = dict(folds=folds, units=int(av.sum()), S_units=int(Su.sum()),
                   S_rmse=rmse(P, Su), host_S_rmse=rmse(host0, Su),
                   S_vs_host0=rel(P, host0, Su), all_vs_host0=rel(P, host0, allu), nonS_vs_host0=rel(P, host0, nSu),
                   S_vs_host5=rel(P, host5, Su), all_vs_host5=rel(P, host5, allu),
                   false_peaks=int(((pk >= .10) & nSu).sum()), host_false_peaks=int(((pk0 >= .10) & nSu).sum()),
                   S_median_peak_ratio=float(np.median(pk[Su] / peak[Su])), host_S_median_peak_ratio=float(np.median(pk0[Su] / peak[Su])),
                   S_share_half_peak=float(np.mean(pk[Su] >= .5 * peak[Su])), host_S_share_half_peak=float(np.mean(pk0[Su] >= .5 * peak[Su])),
                   S_by_regime={r: (1 - rmse(P, Su & (reg == r)) / rmse(host0, Su & (reg == r))) for r in REGIMES if (Su & (reg == r)).sum() > 0},
                   all_by_regime={r: (1 - rmse(P, allu & (reg == r)) / rmse(host0, allu & (reg == r))) for r in REGIMES if (allu & (reg == r)).sum() > 0})
        systems = [q for q in np.unique(sysv[av]) if (av & (sysv == q)).sum() >= 10]
        rho = np.array([spearmanr(pk[av & (sysv == q)], peak[av & (sysv == q)]).statistic for q in systems])
        rho0 = np.array([spearmanr(pk0[av & (sysv == q)], peak[av & (sysv == q)]).statistic for q in systems])
        dr = rho - rho0
        bs = [np.nanmean(dr[ix]) for ix in cluster_draws([(reg[sysv == q][0], q) for q in systems], np.random.default_rng(BOOT_SEED), 1000)]
        res["within_storm_spearman"] = dict(arm=float(np.nanmean(rho)), host0=float(np.nanmean(rho0)), diff=float(np.nanmean(dr)),
                                           ci95=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))])
        pm = clear & av[pairs[:, 0]] & av[pairs[:, 1]]
        dp = np.log(pk[pairs[:, 0]] + .005) - np.log(pk[pairs[:, 1]] + .005); dp0 = np.log(pk0[pairs[:, 0]] + .005) - np.log(pk0[pairs[:, 1]] + .005)
        res["twin_sign_accuracy"] = dict(pairs=int(pm.sum()), arm=float(np.mean(np.sign(dp[pm]) == np.sign(dy[pm]))), host0=float(np.mean(np.sign(dp0[pm]) == np.sign(dy[pm]))))
        ii = np.where(Su & (reg == "tropical") & (tpk + 24 <= 143))[0]
        if len(ii):
            U0, R0 = data["host0"][1], data["host0"][2]
            res["tropical_S_rates_at_peak"] = dict(n=int(len(ii)), r_arm=float(np.median(R[ii, tpk[ii]])), r_host0=float(np.median(R0[ii, tpk[ii]])),
                                                   u_arm=float(np.median(U[ii, tpk[ii]])), u_host0=float(np.median(U0[ii, tpk[ii]])),
                                                   r_arm_mean_24h=float(np.median([np.mean(R[q, t:t + 25]) for q, t in zip(ii, tpk[ii])])),
                                                   observed_implied_r=float(np.median(1 - np.clip(y[ii, tpk[ii] + 24] / peak[ii], 1e-4, 1.5) ** (1 / 24))))
        res["kernel_parameters"] = meta
        out[name] = res
    for a, b in (("DKV", "DKVp"), ("DKV", "DK"), ("DKV", "DKVr"), ("DK", "GCRK"), ("DK", "hostF"), ("DKV", "hostF")):
        if a in out and b in out:
            Pa, Pb = data[a][0], data[b][0]
            av = ~np.isnan(Pa).any(1) & ~np.isnan(Pb).any(1)
            out[f"{a} vs {b}"] = dict(units=int(av.sum()), S=rel(Pa, Pb, av & S), all=rel(Pa, Pb, av), nonS=rel(Pa, Pb, av & ~S))
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "dose_scores.json").write_text(json.dumps(out, indent=1) + "\n")
    f = lambda x: f"{100 * x['point']:+.2f}% [{100 * x['ci95'][0]:+.2f}, {100 * x['ci95'][1]:+.2f}]"  # noqa: E731
    for k, v in out.items():
        if "S_vs_host0" in v:
            print(f"{k}: folds {v['folds']} units {v['units']} S {v['S_units']} | S vs host0 {f(v['S_vs_host0'])} | all {f(v['all_vs_host0'])} | nonS {f(v['nonS_vs_host0'])} | "
                  f"FP {v['false_peaks']} (host {v['host_false_peaks']}) | peak ratio {v['S_median_peak_ratio']:.3f} (host {v['host_S_median_peak_ratio']:.3f}) | half-peak {v['S_share_half_peak']:.3f} (host {v['host_S_share_half_peak']:.3f})")
            print("   S by regime:", {r: f"{100 * x:+.1f}%" for r, x in v["S_by_regime"].items()}, "| all by regime:", {r: f"{100 * x:+.1f}%" for r, x in v["all_by_regime"].items()})
            ws = v["within_storm_spearman"]; tw = v["twin_sign_accuracy"]
            print(f"   within-storm Spearman {ws['arm']:.3f} vs host {ws['host0']:.3f} (diff {ws['diff']:+.3f} [{ws['ci95'][0]:+.3f}, {ws['ci95'][1]:+.3f}]) | twins sign acc {tw['arm']:.3f} vs host {tw['host0']:.3f} ({tw['pairs']} pairs) | S vs host5 {f(v['S_vs_host5'])} all vs host5 {f(v['all_vs_host5'])}")
            if "tropical_S_rates_at_peak" in v:
                print("   tropical S at peak:", {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v["tropical_S_rates_at_peak"].items()})
        else:
            print(f"{k}: units {v['units']} | S {f(v['S'])} | all {f(v['all'])} | nonS {f(v['nonS'])}")


if __name__ == "__main__":
    main(sys.argv[1:] or [a for a in ARMS])
