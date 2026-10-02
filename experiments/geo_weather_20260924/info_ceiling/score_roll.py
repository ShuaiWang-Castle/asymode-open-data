"""Scores of the restoration-kernel screen on the panel with the later forecast origin
(notes/RESTORATION_KERNEL_DESIGN_20261002_ZH.md), seed 0, event folds.

Each arm is compared with the host retrained on the same shifted panel, on the county-events of the folds both have
finished: design-weighted RMSE over the 96 forecast hours for all county-events, for those already out at the origin
(stock >= 1%), for the severe county-events of the registered window, by regime, and over the first 48 h; the count of
false peaks; the kernel weights. Intervals: family-cluster bootstrap within regime, 2,000 draws, seed 20260924.
usage: python score_roll.py [shift] [labels ...]  -> results/v1/info_ceiling/roll_scores_<shift>.json"""
from __future__ import annotations

import json
import sys

import numpy as np

from probe import BOOT_SEED, B, EXP, REGIMES, RES, RUNS, load_static
from score_dose import collect

sys.path.insert(0, str(EXP))
ARMS = {"RK": "rk", "RKl": "rkl", "RKp": "rkp", "RKs": "rks", "RKu": "rku", "Bin": "bin", "hostF": "hostf", "BinF": "binf"}


def main(shift: int, names) -> None:
    import screen
    F = screen.shift_origin(screen.load("v1D"), shift)
    s = load_static()
    y, m, y0, w = F["y"].astype(float), F["m"].astype(float), F["y0"].astype(float), s["w"].astype(float)
    reg, fam, peak = s["regime"].astype(str), s["family"].astype(str), s["peak"].astype(float)
    n = len(y); cnt = m.sum(1); H = 144 - shift
    base = collect(f"v1r{shift}_host_s0", n)
    old = collect("v1_host_s0", n)                      # the registered host (origin 72): its rates rolled from the new origin
    Pold = np.zeros_like(y); p = y0.copy()
    for t in range(H):
        p = np.clip(p + old[1][:, t + shift] * (1 - p) - old[2][:, t + shift] * p, 0, 1); Pold[:, t] = p
    keys = sorted(set(zip(reg, fam))); gid = {kk: j for j, kk in enumerate(keys)}; g = np.array([gid[kk] for kk in zip(reg, fam)])
    rng = np.random.default_rng(BOOT_SEED); counts = np.zeros((B, len(keys)))
    for r in REGIMES:
        js = np.array([gid[kk] for kk in keys if kk[0] == r]); counts[:, js] = rng.multinomial(len(js), np.full(len(js), 1 / len(js)), size=B)

    def sse(P, hours=None):
        mm = m if hours is None else m * (np.arange(144) < hours)[None, :]
        return (mm * (np.nan_to_num(P) - y) ** 2).sum(1)

    def rel(a, b, sub):
        def gs(v):
            o = np.zeros(len(keys)); np.add.at(o, g[sub], (w * v)[sub]); return o
        qa, qb = counts @ gs(a), counts @ gs(b); okb = qb > 0
        d = 1 - np.sqrt(qa[okb] / qb[okb])
        return dict(point=float(1 - np.sqrt((w * a)[sub].sum() / (w * b)[sub].sum())), ci95=[float(np.quantile(d, .025)), float(np.quantile(d, .975))])

    act = y0 >= .01; S = peak >= .10
    obs_pk = np.where(m > 0, y, 0).max(1)
    persist = np.repeat(y0[:, None], 144, 1)
    out = {"shift": shift, "horizon_h": H}
    f = lambda x: f"{100 * x['point']:+.2f}% [{100 * x['ci95'][0]:+.2f}, {100 * x['ci95'][1]:+.2f}]"  # noqa: E731
    for name in names:
        arm = collect(f"v1r{shift}_{ARMS[name]}_s0", n)
        av = ~np.isnan(arm[0]).any(1) & ~np.isnan(base[0]).any(1) & (cnt > 0)
        if not av.any():
            continue
        folds = sorted(set(arm[3]) & set(base[3]))
        Pa, Pb = arm[0], base[0]
        res = dict(folds=folds, units=int(av.sum()), active_units=int((av & act).sum()), S_units=int((av & S).sum()),
                   rmse=dict(arm=float(np.sqrt((w * sse(Pa))[av].sum() / (w * cnt)[av].sum())), host=float(np.sqrt((w * sse(Pb))[av].sum() / (w * cnt)[av].sum()))),
                   all=rel(sse(Pa), sse(Pb), av), active=rel(sse(Pa), sse(Pb), av & act), not_active=rel(sse(Pa), sse(Pb), av & ~act), S=rel(sse(Pa), sse(Pb), av & S),
                   all_48h=rel(sse(Pa, 48), sse(Pb, 48), av), active_48h=rel(sse(Pa, 48), sse(Pb, 48), av & act),
                   all_by_regime={r: rel(sse(Pa), sse(Pb), av & (reg == r))["point"] for r in REGIMES if (av & (reg == r)).any()},
                   active_by_regime={r: rel(sse(Pa), sse(Pb), av & act & (reg == r))["point"] for r in REGIMES if (av & act & (reg == r)).sum() > 5},
                   by_fold={k: dict(all=rel(sse(Pa), sse(Pb), av & (np.isin(np.arange(n), np.where(~np.isnan(Pa).any(1))[0])) & (fold_of(n) == k))["point"],
                                    active=rel(sse(Pa), sse(Pb), av & act & (fold_of(n) == k))["point"]) for k in folds},
                   false_peaks=dict(arm=int(((np.nan_to_num(Pa)[:, :H].max(1) >= .10) & (obs_pk < .10) & av).sum()), host=int(((Pb[:, :H].max(1) >= .10) & (obs_pk < .10) & av).sum())),
                   host_vs_registered_host_rolled=dict(all=rel(sse(Pb), sse(Pold), av), active=rel(sse(Pb), sse(Pold), av & act)),
                   arm_vs_persistence=dict(all=rel(sse(Pa), sse(persist), av), active=rel(sse(Pa), sse(persist), av & act)),
                   host_vs_persistence=dict(all=rel(sse(Pb), sse(persist), av), active=rel(sse(Pb), sse(persist), av & act)),
                   kernel={k: {kk: vv for kk, vv in json.loads((RUNS / f"v1r{shift}_{ARMS[name]}_s0" / f"fold{k:02d}" / "DONE.json").read_text()).items() if kk.startswith("rest_") or kk in ("seconds", "fit_loss", "burden")} for k in folds},
                   host_fit_loss={k: json.loads((RUNS / f"v1r{shift}_host_s0" / f"fold{k:02d}" / "DONE.json").read_text())["fit_loss"] for k in folds})
        if name == "RK":                # how strongly the regional term slows the restoration of the county-events already out
            bz = np.load(screen.ROOT / "data" / "interim" / "panel_v1" / "burden_v1D.npz")
            bur = bz["rings"][:, int(bz["col_origin"]) + shift - 1].astype(float)
            from geo_evidence import fold_of_unit
            fo = fold_of_unit(n); fac = np.full(n, np.nan)
            for k in folds:
                fac[fo == k] = 1.0 + bur[fo == k] @ np.array(res["kernel"][k]["rest_kappa_b"])
            sub = av & act
            res["regional_slowdown_factor_active"] = dict(
                median=float(np.median(fac[sub])), q25=float(np.quantile(fac[sub], .25)), q75=float(np.quantile(fac[sub], .75)), q90=float(np.quantile(fac[sub], .9)),
                by_regime={r: float(np.median(fac[sub & (reg == r)])) for r in REGIMES if (sub & (reg == r)).sum() > 5},
                not_active_median=float(np.median(fac[av & ~act])))
        out[name] = res
        print(f"{name}: folds {folds} units {res['units']} active {res['active_units']} | all {f(res['all'])} | active {f(res['active'])} | not active {f(res['not_active'])} | S {f(res['S'])}")
        if "regional_slowdown_factor_active" in res:
            q = res["regional_slowdown_factor_active"]
            print(f"   regional slow-down factor 1 + sum kappa_k b_k, counties already out: median {q['median']:.2f} (quartiles {q['q25']:.2f}-{q['q75']:.2f}, 90th {q['q90']:.2f}); by regime", {r: round(v, 2) for r, v in q["by_regime"].items()}, f"| not yet out: median {q['not_active_median']:.2f}")
        print(f"   first 48 h: all {f(res['all_48h'])} active {f(res['active_48h'])} | false peaks {res['false_peaks']} | all by regime", {r: f"{100 * v:+.1f}%" for r, v in res["all_by_regime"].items()},
              "| active by regime", {r: f"{100 * v:+.1f}%" for r, v in res["active_by_regime"].items()})
        print("   by fold:", {k: f"all {100 * v['all']:+.1f}% active {100 * v['active']:+.1f}%" for k, v in res["by_fold"].items()})
        print(f"   retrained host vs registered host rolled: all {f(res['host_vs_registered_host_rolled']['all'])} active {f(res['host_vs_registered_host_rolled']['active'])} | vs persistence: host all {f(res['host_vs_persistence']['all'])} arm all {f(res['arm_vs_persistence']['all'])}")
        print("   kernel:", {k: (round(v["rest_kappa_l"], 1), [round(x, 1) for x in v["rest_kappa_b"]]) for k, v in res["kernel"].items()}, "| fit loss arm/host:", {k: f"{res['kernel'][k]['fit_loss']:.3e}/{res['host_fit_loss'][k]:.3e}" for k in folds})
    for a, b in (("RK", "RKp"), ("RK", "RKl"), ("RK", "RKs"), ("RKl", "RKp"), ("RK", "hostF"), ("RK", "BinF"), ("RK", "RKu"), ("RK", "Bin"), ("BinF", "hostF")):
        if a in out and b in out:
            Pa, Pb = collect(f"v1r{shift}_{ARMS[a]}_s0", n)[0], collect(f"v1r{shift}_{ARMS[b]}_s0", n)[0]
            av = ~np.isnan(Pa).any(1) & ~np.isnan(Pb).any(1) & (cnt > 0)
            out[f"{a} vs {b}"] = dict(units=int(av.sum()), all=rel(sse(Pa), sse(Pb), av), active=rel(sse(Pa), sse(Pb), av & act), S=rel(sse(Pa), sse(Pb), av & S))
            print(f"{a} vs {b}: units {int(av.sum())} | all {f(out[f'{a} vs {b}']['all'])} | active {f(out[f'{a} vs {b}']['active'])} | S {f(out[f'{a} vs {b}']['S'])}")
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"roll_scores_{shift}.json").write_text(json.dumps(out, indent=1) + "\n")


def fold_of(n):
    from geo_evidence import fold_of_unit
    return fold_of_unit(n)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 48, sys.argv[2:] or list(ARMS))
