"""AsymODE (W+Cin), AsymODE + GCRK (temporal kernel) and AsymODE + spatio-temporal GCRK on the development tranche,
the same initialisation (paired), five event folds, one open-loop path per county-event.

Reported, design-weighted and pooled over the held-out county-hours (as paper_v1/evaluate_paper.py):
  * MAE and RMSE at +1, +6, +24, +48 h for the three models and the all-zero forecast;
  * pairwise relative RMSE (kernel vs host, spatio-temporal vs temporal) with family-cluster intervals (2,000 draws
    within regime, seed 20260924), overall, by regime, by initial state, and without the three systems where the host
    errs most;
  * large outages (observed peak >= 10%): median forecast peak / observed peak;
  * the learned coupling of the spatio-temporal kernel per fold (kappa_s, kappa_a, gamma) and the opening tanh(alpha).
usage: python compare_kernels_v1.py [--seed 0] [--st v1_stgcrk_s{}] -> results/v1/kernels_s<seed>.json"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "paper_v1")); sys.path.insert(0, str(HERE))
from evaluate_paper import B, FEAT, HORIZONS, SEED, pd_sum, pooled, rollout, row_sums  # noqa: E402
from evaluate_v1 import REGIMES, RUNS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--st", default="v1_stgcrk_s{}")
    a = ap.parse_args()
    F = np.load(FEAT)
    y, m, w = F["y"].astype(float), F["m"].astype(float), F["w"].astype(float)
    reg, fam, sysv = F["regime"].astype(str), F["family"].astype(str), F["system"].astype(str)
    n = len(y)
    labels = {"AsymODE": f"v1_host_s{a.seed}", "GCRK": f"v1_gcrk_s{a.seed}", "ST-GCRK": a.st.format(a.seed)}
    paths = {"All zero": np.zeros((n, 144))}
    for nm, lab in labels.items():
        P = rollout(lab, n)
        if P is None:
            print(nm, lab, "incomplete"); continue
        paths[nm] = P
    S = {nm: row_sums(P, y, m) for nm, P in paths.items()}
    res = {"labels": labels, "metrics": {nm: pooled(s, w) for nm, s in S.items()},
           "by_regime": {nm: {r: pooled(s, w, np.where(reg == r)[0]) for r in REGIMES} for nm, s in S.items()}}
    keys = sorted({(r, f) for r, f in zip(reg, fam)}); gid = {k: j for j, k in enumerate(keys)}
    g = np.array([gid[(r, f)] for r, f in zip(reg, fam)])
    G = np.zeros((len(keys), n)); G[g, np.arange(n)] = w
    rng = np.random.default_rng(SEED); counts = np.zeros((B, len(keys)))
    for r in REGIMES:
        js = np.array([gid[k] for k in keys if k[0] == r])
        counts[:, js] = rng.multinomial(len(js), np.full(len(js), 1 / len(js)), size=B)

    def rel(sa, sb, idx=None):
        """point and 95% interval of RMSE(a)/RMSE(b) - 1 per horizon, optionally on a subset of county-events."""
        if idx is None:
            Ga = G
        else:
            mask = np.zeros(n); mask[idx] = 1.0; Ga = G * mask
        qa, qb, na = counts @ (Ga @ sa[1]), counts @ (Ga @ sb[1]), counts @ (Ga @ sa[2])
        d = np.sqrt(qa / na) / np.sqrt(qb / na) - 1
        pa, pb = pooled(sa, w, idx), pooled(sb, w, idx)
        return {f"+{h}": dict(point=pa[f"RMSE+{h}"] / pb[f"RMSE+{h}"] - 1,
                              ci95=[float(np.quantile(d[:, j], q)) for q in (0.025, 0.975)]) for j, h in enumerate(HORIZONS)}

    pairs = [("GCRK", "AsymODE"), ("ST-GCRK", "AsymODE"), ("ST-GCRK", "GCRK"), ("AsymODE", "All zero")]
    pairs = [(p, q) for p, q in pairs if p in S and q in S]
    res["relative_rmse"] = {f"{p} vs {q}": rel(S[p], S[q]) for p, q in pairs}
    res["relative_rmse_by_regime_h1"] = {f"{p} vs {q}": {r: pooled(S[p], w, np.where(reg == r)[0])["RMSE+1"] /
                                                          pooled(S[q], w, np.where(reg == r)[0])["RMSE+1"] - 1
                                                          for r in REGIMES} for p, q in pairs}
    y0 = F["y0"].astype(float)
    groups = {"served (<= 0.1% out at the origin)": np.where(y0 <= 1e-3)[0], "stock (> 0.1%)": np.where(y0 > 1e-3)[0]}
    res["relative_rmse_by_initial_state_h1"] = {f"{p} vs {q}": {gname: rel(S[p], S[q], idx)["+1"] for gname, idx in groups.items()}
                                                for p, q in pairs}
    sse = pd_sum(sysv, w[:, None] * S["AsymODE"][1][:, :1]) if "AsymODE" in S else {}
    worst = [s for s, _ in sorted(sse.items(), key=lambda kv: -kv[1])][:3]
    keep = np.where(~np.isin(sysv, worst))[0]
    res["without_three_worst_systems"] = {"regimes": sorted(set(reg[np.isin(sysv, worst)])),
                                          **{f"{p} vs {q}": rel(S[p], S[q], keep)["+1"] for p, q in pairs}}
    yo = np.where(F["obs_full"], F["y_full"], np.nan)[:, 72:216].astype(float)
    opk = np.nanmax(np.where(np.isnan(yo), -1.0, yo), 1); big = opk >= 0.10
    res["large_outages"] = dict(n=int(big.sum()), **{nm: dict(median_peak_ratio=float(np.median(P.max(1)[big] / opk[big])),
                                                              share_half_peak=float(np.mean(P.max(1)[big] >= 0.5 * opk[big])))
                                                     for nm, P in paths.items() if nm != "All zero"})
    coupling = {}
    for k in range(1, 6):
        f = RUNS / labels["ST-GCRK"] / f"fold{k:02d}" / "final.pt"
        if f.exists():
            st = torch.load(f, weights_only=False)["model_state"]
            coupling[f"fold{k}"] = {nm: float(st[f"damage.2.{nm}"]) for nm in ("kappa_s", "kappa_a", "geo_sim")}
            coupling[f"fold{k}"]["opening"] = float(torch.tanh(st["damage.2.alpha"]))
        g0 = RUNS / labels["GCRK"] / f"fold{k:02d}" / "final.pt"
        if g0.exists():
            coupling.setdefault(f"fold{k}", {})["gcrk_opening"] = float(torch.tanh(torch.load(g0, weights_only=False)["model_state"]["damage.2.alpha"]))
    res["kernel_parameters"] = coupling
    out = HERE / "results" / "v1" / f"kernels_s{a.seed}.json"
    out.write_text(json.dumps(res, indent=1) + "\n")
    cols = [f"MAE+{h}" for h in HORIZONS] + [f"RMSE+{h}" for h in HORIZONS]
    print("| model | " + " | ".join(cols) + " |"); print("|" + "---|" * (len(cols) + 1))
    for nm, v in res["metrics"].items():
        print(f"| {nm} | " + " | ".join(f"{100 * v[c]:.3f}" for c in cols) + " |")
    for pq, v in res["relative_rmse"].items():
        print(pq, {h: f"{100 * x['point']:+.2f}% [{100 * x['ci95'][0]:+.2f}, {100 * x['ci95'][1]:+.2f}]" for h, x in v.items()})
    for pq, v in res["relative_rmse_by_regime_h1"].items():
        print(pq, "by regime (+1):", {r: f"{100 * x:+.1f}%" for r, x in v.items()})
    print("by initial state:", json.dumps({pq: {gname: f"{100 * x['point']:+.2f}%" for gname, x in v.items()}
                                           for pq, v in res["relative_rmse_by_initial_state_h1"].items()}))
    print("without the three worst systems:", {k: (f"{100 * v['point']:+.2f}%" if isinstance(v, dict) else v)
                                                for k, v in res["without_three_worst_systems"].items()})
    print("large outages:", res["large_outages"])
    print("kernel parameters:", json.dumps(coupling))
    print("->", out)


if __name__ == "__main__":
    main()
