"""AsymODE (W+Cin), AsymODE + GCRK (temporal kernel) and AsymODE + spatio-temporal GCRK on the development tranche,
the same initialisation (paired), five event folds, one open-loop path per county-event.

Reported, design-weighted and pooled over the held-out county-hours (as paper_v1/evaluate_paper.py):
  * MAE and RMSE at +1, +6, +24, +48 h for the three models and the all-zero forecast;
  * pairwise relative RMSE (kernel vs host, spatio-temporal vs temporal) with family-cluster intervals (2,000 draws
    within regime, seed 20260924), overall, by regime, by initial state, and without the three systems where the host
    errs most;
  * large outages (observed peak >= 10%): median forecast peak / observed peak;
  * the learned coupling of the spatio-temporal kernel per fold (kappa_s, kappa_a, gamma) and the opening tanh(alpha).
usage: python compare_kernels_v1.py [--seed 0] [--st v1_stgcrk_s{}] -> results/v1/kernels_s<seed>.json
       python compare_kernels_v1.py --candidate v1_gcrk_georms_s{} --candidate-name GCRK-RMS
       --out can select an explicit result path; candidates otherwise use a separate output filename."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "paper_v1")); sys.path.insert(0, str(HERE))
from evaluate_paper import B, FEAT, HORIZONS, SEED, pd_sum, pooled, row_sums  # noqa: E402
from evaluate_v1 import REGIMES, RUNS  # noqa: E402


def checked_predictions(label: str, n: int, folds: list[int], seed: int,
                        expected: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Require the intended event-fold indices exactly once; inspect legacy metadata when available."""
    if not folds or len(set(folds)) != len(folds) or any(k not in range(1, 6) for k in folds):
        raise ValueError(f"folds must be distinct values in 1..5, got {folds}")
    P = np.zeros((n, 144)); seen = np.zeros(n, dtype=np.int64); metadata = {}
    for k in folds:
        folder = RUNS / label / f"fold{k:02d}"
        f = folder / "outer.npz"
        if not f.exists():
            raise FileNotFoundError(f"incomplete run {label}: missing {f}")
        with np.load(f, allow_pickle=False) as z:
            idx, pred = z["idx"], z["P"]
            if idx.ndim != 1 or not np.issubdtype(idx.dtype, np.integer):
                raise ValueError(f"{f}: idx must be a one-dimensional integer array")
            if np.any(idx < 0) or np.any(idx >= n) or len(np.unique(idx)) != len(idx):
                raise ValueError(f"{f}: out-of-range or duplicate held-out indices")
            if pred.shape != (len(idx), 144) or not np.isfinite(pred).all():
                raise ValueError(f"{f}: invalid prediction shape or non-finite values")
            if not np.array_equal(np.sort(idx), np.sort(expected[str(k)]["outer"])):
                raise ValueError(f"{f}: held-out indices do not match event fold {k}")
            P[idx] = pred; seen[idx] += 1
        done = folder / "DONE.json"
        if done.exists():
            meta = json.loads(done.read_text())
            for key, value in dict(label=label, data="v1D", design="event", fold=k, seed=seed).items():
                if key in meta and meta[key] != value:
                    raise ValueError(f"{done}: {key}={meta[key]!r}, expected {value!r}")
            # Historical runs need not have newer provenance / training-baseline fields.
            metadata[f"fold{k}"] = meta
    want = np.concatenate([np.asarray(expected[str(k)]["outer"], dtype=int) for k in folds])
    if len(np.unique(want)) != len(want) or np.any(seen[want] != 1):
        raise ValueError(f"{label}: each requested held-out unit must occur exactly once")
    if set(folds) == set(range(1, 6)) and not np.all(seen == 1):
        raise ValueError(f"{label}: incomplete or duplicate five-fold coverage")
    return P, np.sort(want), metadata


def kernel_parameters(label: str, k: int) -> dict:
    """Spatial parameters are present only for spatial kernels; legacy checkpoints remain supported."""
    f = RUNS / label / f"fold{k:02d}" / "final.pt"
    if not f.exists():
        return {}
    checkpoint = torch.load(f, weights_only=False)
    st = checkpoint["model_state"]
    out = {nm: float(st[f"damage.2.{nm}"]) for nm in ("kappa_s", "kappa_a", "geo_sim", "geo_rms_scale")
           if f"damage.2.{nm}" in st}
    if "damage.2.alpha" in st:
        out["opening"] = float(torch.tanh(st["damage.2.alpha"]))
    return out


def fold_report(seed: int, st: str, folds: list[int], candidate_name: str = "ST-GCRK") -> None:
    """Early read while folds are still running: the three models on the held-out county-events of the finished
    folds only (the same systems for every model), pooled RMSE and MAE, overall and by regime."""
    F = np.load(FEAT)
    y, m, w, reg = F["y"].astype(float), F["m"].astype(float), F["w"].astype(float), F["regime"].astype(str)
    labels = {"AsymODE": f"v1_host_s{seed}", "GCRK": f"v1_gcrk_s{seed}", candidate_name: st.format(seed)}
    expected = json.loads((HERE / "splits_v1D.json").read_text())["event"]
    P = {}
    for nm, lab in labels.items():
        P[nm], idx, _ = checked_predictions(lab, len(y), folds, seed, expected)
    S = {nm: row_sums(p, y, m) for nm, p in P.items()}
    S["All zero"] = row_sums(np.zeros_like(y), y, m)
    out = {"folds": folds, "n": int(len(idx)), "systems": int(len(set(F["system"][idx])))}
    for nm, s in S.items():
        out[nm] = pooled(s, w, idx)
    base = out["AsymODE"]
    print(f"folds {folds}: {out['n']} county-events, {out['systems']} systems")
    for nm in ("All zero", *labels):
        v = out[nm]
        print(f"  {nm:8s} RMSE+1 {100 * v['RMSE+1']:.3f}  +48 {100 * v['RMSE+48']:.3f}  MAE+1 {100 * v['MAE+1']:.3f}  "
              f"vs AsymODE {100 * (v['RMSE+1'] / base['RMSE+1'] - 1):+.2f}% (+1), {100 * (v['RMSE+48'] / base['RMSE+48'] - 1):+.2f}% (+48)")
    for r in REGIMES:
        ii = idx[reg[idx] == r]
        if len(ii) == 0:
            continue
        b = pooled(S["AsymODE"], w, ii)["RMSE+1"]
        print(f"  {r:13s} n={len(ii):5d}  GCRK {100 * (pooled(S['GCRK'], w, ii)['RMSE+1'] / b - 1):+.2f}%  "
              f"{candidate_name} {100 * (pooled(S[candidate_name], w, ii)['RMSE+1'] / b - 1):+.2f}%  (vs AsymODE, +1)")
    for k in folds:
        print(f"  fold {k} {candidate_name} parameters:", kernel_parameters(labels[candidate_name], k))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--st", default="v1_stgcrk_s{}")
    ap.add_argument("--candidate", help="candidate label or label template containing {} for the seed")
    ap.add_argument("--candidate-name", help="display name for --candidate (default: Candidate)")
    ap.add_argument("--out", help="output JSON path relative to this experiment directory, or an absolute path")
    ap.add_argument("--folds", nargs="*", type=int, default=None, help="early read on these finished folds only")
    a = ap.parse_args()
    if a.candidate_name and not a.candidate:
        ap.error("--candidate-name requires --candidate")
    candidate_name = (a.candidate_name or "Candidate") if a.candidate else "ST-GCRK"
    if candidate_name in {"AsymODE", "GCRK", "All zero"}:
        ap.error("--candidate-name must differ from AsymODE, GCRK, and All zero")
    candidate = a.candidate if a.candidate else a.st
    if a.folds:
        fold_report(a.seed, candidate, a.folds, candidate_name)
        return
    F = np.load(FEAT)
    y, m, w = F["y"].astype(float), F["m"].astype(float), F["w"].astype(float)
    reg, fam, sysv = F["regime"].astype(str), F["family"].astype(str), F["system"].astype(str)
    n = len(y)
    labels = {"AsymODE": f"v1_host_s{a.seed}", "GCRK": f"v1_gcrk_s{a.seed}", candidate_name: candidate.format(a.seed)}
    paths = {"All zero": np.zeros((n, 144))}
    expected = json.loads((HERE / "splits_v1D.json").read_text())["event"]
    metadata = {}
    for nm, lab in labels.items():
        P, _, metadata[nm] = checked_predictions(lab, n, list(range(1, 6)), a.seed, expected)
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

    pairs = [("GCRK", "AsymODE"), (candidate_name, "AsymODE"), (candidate_name, "GCRK"), ("AsymODE", "All zero")]
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
        params = kernel_parameters(labels[candidate_name], k)
        if params:
            coupling[f"fold{k}"] = params
        baseline = kernel_parameters(labels["GCRK"], k)
        if "opening" in baseline:
            coupling.setdefault(f"fold{k}", {})["gcrk_opening"] = baseline["opening"]
    res["kernel_parameters"] = coupling
    if a.candidate:
        res["run_metadata"] = metadata
        res["candidate"] = dict(name=candidate_name, label=labels[candidate_name], folds=list(range(1, 6)),
                                units=int(n), seed=a.seed)
    if a.out:
        out = HERE / a.out
    elif a.candidate:
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", labels[candidate_name]).strip("._") or "candidate"
        out = HERE / "results" / "v1" / f"kernels_candidate_{slug}.json"
    else:
        out = HERE / "results" / "v1" / f"kernels_s{a.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
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
