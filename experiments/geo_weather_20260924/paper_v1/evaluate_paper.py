"""The paper's accuracy tables on the development tranche of the designed panel (DATASET_DESIGN v1): five forecasters,
event-held-out (five folds, whole weather systems held out), one open-loop 144-hour path per county-event from the
last observed hour.

  All zero        p = 0
  TimesFM         zero-shot, the 14 host weather channels as past and future covariates (paper_v1/timesfm_v1.py)
  AsymODE         the host W+Cin, seeds 0-4 averaged in forecast space
  AsymODE + GCRK  the geography-conditioned response kernel, opening bounded, seeds 0-4 averaged

Horizon h: the error of the path at hours 72 + h .. 215 (the target hours that a forecast issued at origins 72.. with
lead h would reach), for h = 1, 6, 24, 48. Errors pooled over the observed held-out county-hours with the design
weights w (DATASET_DESIGN section 5.4); unweighted values beside them. Intervals: family-cluster bootstrap within
regime, 2,000 draws, seed 20260924. Per-seed rows compare each learned model with AsymODE of the same seed (paired
initialisation).
Output: results/v1/paper_tables.json and markdown tables on stdout. Missing seeds are reported and averaged over
what exists."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP))
from evaluate_v1 import REGIMES, RUNS  # noqa: E402

FEAT = ROOT / "data" / "interim" / "panel_v1" / "features_v1D.npz"
HORIZONS = (1, 6, 24, 48)
SEEDS = (0, 1, 2, 3, 4)
MODELS = [("All zero", None), ("TimesFM", "timesfm"), ("AsymODE", "v1_host_s{}"), ("AsymODE + GCRK", "v1_gcrk_s{}")]
B, SEED = 2000, 20260924


def pd_sum(keys: np.ndarray, v: np.ndarray) -> dict:
    out: dict = {}
    for k, x in zip(keys, v.sum(1)):
        out[k] = out.get(k, 0.0) + float(x)
    return out


def rollout(label: str, n: int):
    P = np.full((n, 144), np.nan)
    for k in range(1, 6):
        f = RUNS / label / f"fold{k:02d}" / "outer.npz"
        if not f.exists():
            return None
        z = np.load(f)
        P[z["idx"]] = z["P"]
    return None if np.isnan(P).any() else P


def model_paths(pattern, n, F):
    """(seed-averaged path, per-seed paths, seeds used)."""
    if pattern is None:
        return np.zeros((n, 144)), {}, ["-"]
    if pattern == "timesfm":
        f = RUNS / "timesfm_v1D" / "timesfm_WEATHER.npz"
        if not f.exists():
            return None, {}, []
        z = np.load(f)
        assert np.array_equal(z["system"], F["system"]) and np.array_equal(z["fips"], F["fips"])
        return z["P"].astype(float), {}, ["zero-shot"]
    per = {s: P for s in SEEDS if (P := rollout(pattern.format(s), n)) is not None}
    return (np.mean(list(per.values()), 0), per, sorted(per)) if per else (None, {}, [])


def row_sums(P, y, m):
    """Per county-event sums over hours 72 + h .. 215 for every horizon: |e|, e^2 and observed cells [n, H]."""
    e = np.where(m > 0, P - y, 0.0)
    rc = lambda a: np.flip(np.cumsum(np.flip(a, 1), 1), 1)[:, list(HORIZONS)]  # noqa: E731
    return rc(np.abs(e) * m), rc(e ** 2 * m), rc(m)


def pooled(S, w, idx=None):
    A, Q, N = S
    idx = slice(None) if idx is None else idx
    a, q, nn = (w[idx, None] * A[idx]).sum(0), (w[idx, None] * Q[idx]).sum(0), (w[idx, None] * N[idx]).sum(0)
    out = {f"MAE+{h}": float(a[j] / nn[j]) for j, h in enumerate(HORIZONS)}
    out.update({f"RMSE+{h}": float(np.sqrt(q[j] / nn[j])) for j, h in enumerate(HORIZONS)})
    return out


def main() -> None:
    F = np.load(FEAT)
    y, m = F["y"].astype(float), F["m"].astype(float)
    w, wu = F["w"].astype(float), np.ones(len(y))
    reg, fam = F["regime"].astype(str), F["family"].astype(str)
    n = len(y)
    # family groups within regime; family sums of the weighted row sums make the bootstrap a matrix product
    keys = sorted({(r, f) for r, f in zip(reg, fam)})
    gid = {k: j for j, k in enumerate(keys)}
    g = np.array([gid[(r, f)] for r, f in zip(reg, fam)])
    G = np.zeros((len(keys), n)); G[g, np.arange(n)] = w
    rng = np.random.default_rng(SEED)
    counts = np.zeros((B, len(keys)))
    for r in REGIMES:
        js = np.array([gid[k] for k in keys if k[0] == r])
        counts[:, js] = rng.multinomial(len(js), np.full(len(js), 1 / len(js)), size=B)

    res, sums, per_sums, paths = {}, {}, {}, {}
    for name, pat in MODELS:
        P, per, used = model_paths(pat, n, F)
        if P is None:
            res[name] = dict(missing=True); continue
        S = row_sums(P, y, m); sums[name] = S; paths[name] = P
        per_sums[name] = {s: row_sums(Ps, y, m) for s, Ps in per.items()}
        res[name] = dict(seeds=used, weighted=pooled(S, w), unweighted=pooled(S, wu),
                         by_regime={r: pooled(S, w, np.where(reg == r)[0]) for r in REGIMES},
                         per_seed={str(s): pooled(Ss, w) for s, Ss in per_sums[name].items()})

    def rel_draws(Sa, Sb):
        """Bootstrap draws of RMSE(a) / RMSE(b) - 1 [B, H] from family sums."""
        qa, qb = counts @ (G @ Sa[1]), counts @ (G @ Sb[1])
        na = counts @ (G @ Sa[2])
        return np.sqrt(qa / na) / np.sqrt(qb / na) - 1

    for ref in ("AsymODE", "All zero"):
        if ref not in sums:
            continue
        for nm, S in sums.items():
            if nm == ref:
                continue
            point = {f"RMSE+{h}": res[nm]["weighted"][f"RMSE+{h}"] / res[ref]["weighted"][f"RMSE+{h}"] - 1 for h in HORIZONS}
            d = rel_draws(S, sums[ref])
            ci = {f"RMSE+{h}": [float(np.quantile(d[:, j], q)) for q in (0.025, 0.975)] for j, h in enumerate(HORIZONS)}
            byr = {r: res[nm]["by_regime"][r]["RMSE+1"] / res[ref]["by_regime"][r]["RMSE+1"] - 1 for r in REGIMES}
            key = "rmse_vs_asymode" if ref == "AsymODE" else "rmse_vs_zero"
            res[nm][key] = dict(point=point, ci95=ci, by_regime_h1=byr)
    # paired seeds: each learned model against AsymODE of the same initialisation (full path, h = 1)
    for nm in ("AsymODE + GCRK",):
        if nm in per_sums and "AsymODE" in per_sums:
            res[nm]["per_seed_vs_asymode_h1"] = {
                str(s): pooled(per_sums[nm][s], w)["RMSE+1"] / pooled(per_sums["AsymODE"][s], w)["RMSE+1"] - 1
                for s in per_sums[nm] if s in per_sums["AsymODE"]}
    # robustness (weighted, full path h = 1 unless stated)
    rob = {}
    if "AsymODE" in sums:
        sysv = F["system"].astype(str)
        sse = pd_sum(sysv, w[:, None] * sums["AsymODE"][1][:, :1])
        worst = [s for s, _ in sorted(sse.items(), key=lambda kv: -kv[1])]
        rob["drop_worst_systems"] = {}
        for k in (3, 5):
            keep = ~np.isin(sysv, worst[:k])
            rob["drop_worst_systems"][str(k)] = {nm: pooled(S, w, np.where(keep)[0])["RMSE+1"] /
                                                 pooled(sums["AsymODE"], w, np.where(keep)[0])["RMSE+1"] - 1
                                                 for nm, S in sums.items() if nm != "AsymODE"}
            rob["drop_worst_systems"][str(k)]["regimes_dropped"] = sorted(set(reg[np.isin(sysv, worst[:k])]))
        y0 = F["y0"].astype(float)
        rob["by_initial_state"] = {g: {nm: pooled(S, w, np.where(sel)[0])["RMSE+1"] for nm, S in sums.items()}
                                   for g, sel in (("served", y0 <= 1e-3), ("stock", y0 > 1e-3))}
        rob["by_initial_state"]["n"] = dict(served=int((y0 <= 1e-3).sum()), stock=int((y0 > 1e-3).sum()))
        yo = np.where(F["obs_full"], F["y_full"], np.nan)[:, 72:216].astype(float)
        opk = np.nanmax(np.where(np.isnan(yo), -1.0, yo), 1)
        big = opk >= 0.10
        rob["large_events"] = dict(n=int(big.sum()), **{nm: dict(
            median_peak_ratio=float(np.median(model_peak[big] / opk[big])),
            share_half_peak=float(np.mean(model_peak[big] >= 0.5 * opk[big])))
            for nm, model_peak in ((nm, paths[nm].max(1)) for nm in paths if nm != "All zero")})
    res["robustness"] = rob
    (EXP / "results" / "v1" / "paper_tables.json").write_text(json.dumps(res, indent=1) + "\n")
    print("robustness:", json.dumps(rob, indent=1)[:3000])

    cols = [f"MAE+{h}" for h in HORIZONS] + [f"RMSE+{h}" for h in HORIZONS]
    print("| model | seeds | " + " | ".join(cols) + " |"); print("|" + "---|" * (len(cols) + 2))
    for name, _ in MODELS:
        r = res[name]
        if r.get("missing"):
            print(f"| {name} | missing |"); continue
        print(f"| {name} | {','.join(map(str, r['seeds']))} | " + " | ".join(f"{r['weighted'][c]:.5f}" for c in cols) + " |")
    for key in ("rmse_vs_asymode", "rmse_vs_zero"):
        for nm, r in res.items():
            if key in r:
                p, c = r[key]["point"], r[key]["ci95"]
                print(f"{nm} {key}:", {k: f"{100 * p[k]:+.1f}% [{100 * c[k][0]:+.1f}, {100 * c[k][1]:+.1f}]" for k in p},
                      "| by regime +1:", {k: f"{100 * v:+.1f}%" for k, v in r[key]["by_regime_h1"].items()})
    for nm, r in res.items():
        if "per_seed_vs_asymode_h1" in r:
            print(nm, "per seed vs AsymODE (+1):", {k: f"{100 * v:+.2f}%" for k, v in r["per_seed_vs_asymode_h1"].items()})
    write_latex(res)


KEY = {"All zero": "Zero", "TimesFM": "TimesFM", "AsymODE": "Host", "AsymODE + GCRK": "GCRK"}
HWORD = {1: "One", 6: "Six", 24: "TwentyFour", 48: "FortyEight"}
RWORD = {"tropical": "Tropical", "winter": "Winter", "synoptic_wind": "Synoptic", "convective": "Convective",
         "heavy_rain": "Rain"}


def write_latex(res: dict) -> None:
    """paper_v1/generated/table2.tex (Table 2, errors in percentage points of customers) and numbers.tex (one macro
    per number quoted in the text), so the manuscript never carries a hand-copied value. A model with fewer than five
    seeds, or a missing one, is marked provisional."""
    out = HERE / "generated"; out.mkdir(exist_ok=True)
    cols = [f"MAE+{h}" for h in HORIZONS] + [f"RMSE+{h}" for h in HORIZONS]
    have = [nm for nm, _ in MODELS if not res[nm].get("missing")]
    best = {c: min(res[nm]["weighted"][c] for nm in have) for c in cols}
    lines = []
    for nm, pat in MODELS:
        r = res[nm]
        label = nm.replace(" x ", r" $\times$ ")
        if r.get("missing"):
            lines.append(f"{label} & " + " & ".join(["--"] * len(cols)) + r" \\"); continue
        short = pat not in (None, "timesfm") and len(r["seeds"]) < len(SEEDS)
        cells = []
        for c in cols:
            v = f"{100 * r['weighted'][c]:.3f}"
            cells.append(rf"\textbf{{{v}}}" if r["weighted"][c] == best[c] else v)
        mark = rf"$^{{\dagger{len(r['seeds'])}}}$" if short else ""
        lines.append(f"{label}{mark} & " + " & ".join(cells) + r" \\")
    (out / "table2.tex").write_text("\n".join(lines) + "\n")
    mac = []
    fmt = lambda v: f"{v:.3f}"  # noqa: E731
    pct = lambda v: f"{100 * v:+.1f}".replace("-", "$-$").replace("+", "$+$")  # noqa: E731
    for nm, k in KEY.items():
        r = res[nm]
        if r.get("missing"):
            continue
        mac.append(rf"\newcommand{{\Seeds{k}}}{{{len(r['seeds']) if r['seeds'] != ['-'] else 0}}}")
        for h in HORIZONS:
            mac.append(rf"\newcommand{{\Rmse{k}{HWORD[h]}}}{{{fmt(100 * r['weighted'][f'RMSE+{h}'])}}}")
            mac.append(rf"\newcommand{{\Mae{k}{HWORD[h]}}}{{{fmt(100 * r['weighted'][f'MAE+{h}'])}}}")
        for ref, rk in (("rmse_vs_asymode", "Host"), ("rmse_vs_zero", "Zero")):
            if ref not in r:
                continue
            for h in HORIZONS:
                p, (lo, hi) = r[ref]["point"][f"RMSE+{h}"], r[ref]["ci95"][f"RMSE+{h}"]
                mac.append(rf"\newcommand{{\{k}Vs{rk}{HWORD[h]}}}{{{pct(p)}\%}}")
                mac.append(rf"\newcommand{{\{k}Vs{rk}{HWORD[h]}CI}}{{[{pct(lo)}, {pct(hi)}]}}")
            for reg, v in r[ref]["by_regime_h1"].items():
                mac.append(rf"\newcommand{{\{k}Vs{rk}{RWORD[reg]}}}{{{pct(v)}\%}}")
        if "per_seed_vs_asymode_h1" in r:
            vals = list(r["per_seed_vs_asymode_h1"].values())
            mac.append(rf"\newcommand{{\{k}SeedsBetter}}{{{sum(v < 0 for v in vals)}}}")
            mac.append(rf"\newcommand{{\{k}SeedRange}}{{{pct(min(vals))}\% to {pct(max(vals))}\%}}")
    rob = res.get("robustness", {})
    for k, word in (("3", "Three"), ("5", "Five")):
        for nm, kk in KEY.items():
            v = rob.get("drop_worst_systems", {}).get(k, {}).get(nm)
            if v is not None:
                mac.append(rf"\newcommand{{\{kk}VsHostDrop{word}}}{{{pct(v)}\%}}")
    if "by_initial_state" in rob:
        b = rob["by_initial_state"]
        mac.append(rf"\newcommand{{\NServed}}{{{b['n']['served']:,}}}".replace(",", "{,}"))
        mac.append(rf"\newcommand{{\NStock}}{{{b['n']['stock']:,}}}".replace(",", "{,}"))
        for g, word in (("served", "Served"), ("stock", "Stock")):
            h = b[g].get("AsymODE")
            for nm, kk in KEY.items():
                if nm != "AsymODE" and nm in b[g] and h:
                    mac.append(rf"\newcommand{{\{kk}VsHost{word}}}{{{pct(b[g][nm] / h - 1)}\%}}")
    if "large_events" in rob:
        L = rob["large_events"]
        mac.append(rf"\newcommand{{\NLarge}}{{{L['n']}}}")
        for nm, kk in KEY.items():
            if nm in L:
                mac.append(rf"\newcommand{{\{kk}PeakRatio}}{{{100 * L[nm]['median_peak_ratio']:.0f}\%}}")
                mac.append(rf"\newcommand{{\{kk}HalfPeak}}{{{100 * L[nm]['share_half_peak']:.0f}\%}}")
    (out / "numbers.tex").write_text("% generated by paper_v1/evaluate_paper.py; do not edit\n" + "\n".join(mac) + "\n")


if __name__ == "__main__":
    main()
