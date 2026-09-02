"""Regenerate every figure from archived results. Nothing is drawn by hand."""
import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)


def agg(rows, key, by):
    out = {}
    for v in sorted({r[by] for r in rows}):
        g = [r[key] for r in rows if r[by] == v]
        out[v] = (np.mean(g), np.std(g))
    return out


def fig_identifiability():
    rows = json.loads((ROOT / "results/exp01_identifiability.json").read_text())["rows"]
    sp = agg(rows, "state_spread", "pulse_scale")
    x = [sp[k][0] for k in sorted(sp)]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))

    for k, lab, c in (("nrmse_u", r"interruption  $u$", "#c1121f"),
                      ("nrmse_r", r"restoration  $r$", "#003049")):
        a = agg(rows, k, "pulse_scale")
        m = np.array([a[j][0] for j in sorted(a)]); s = np.array([a[j][1] for j in sorted(a)])
        ax[0].plot(x, m, "o-", color=c, label=lab)
        ax[0].fill_between(x, m - s, m + s, color=c, alpha=0.18)
    ax[0].set_yscale("log"); ax[0].set_xlabel("state spread"); ax[0].set_ylabel("rate recovery nRMSE")
    ax[0].set_title("recovery improves as the state moves", fontsize=10)
    ax[0].legend(frameon=False, fontsize=9)

    a = agg(rows, "traj_rmse", "pulse_scale")
    m = np.array([a[j][0] for j in sorted(a)]); s = np.array([a[j][1] for j in sorted(a)])
    ax[1].plot(x, m, "s-", color="#7f7f7f"); ax[1].fill_between(x, m - s, m + s, color="#7f7f7f", alpha=0.18)
    ax[1].set_yscale("log"); ax[1].set_xlabel("state spread"); ax[1].set_ylabel("trajectory RMSE")
    # Both panels share one decade range so the comparison is read off the page
    # rather than taken on trust: recovery error spans nearly two decades over
    # the same sweep in which trajectory error spans a factor of four.
    lo = min(ax[0].get_ylim()[0], m.min() * 0.5)
    hi = max(ax[0].get_ylim()[1], m.max() * 2)
    ax[0].set_ylim(lo, hi); ax[1].set_ylim(lo, hi)
    ax[1].set_title("trajectory fit barely moves\n(same axis as left)", fontsize=10)

    a = agg(rows, "err_corr", "pulse_scale")
    m = np.array([a[j][0] for j in sorted(a)]); s = np.array([a[j][1] for j in sorted(a)])
    ax[2].plot(x, m, "^-", color="#2a9d8f"); ax[2].fill_between(x, m - s, m + s, color="#2a9d8f", alpha=0.18)
    ax[2].axhline(0, color="k", lw=0.6)
    ax[2].set_xlabel("state spread"); ax[2].set_ylabel(r"corr($\hat u - u^*,\ \hat r - r^*$)")
    ax[2].set_title("the ridge: errors trade off", fontsize=10)
    for a_ in ax: a_.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "fig01_identifiability.png", dpi=200)
    print("figures/fig01_identifiability.png")


def fig_onset():
    rows = json.loads((ROOT / "results/exp02_onset.json").read_text())["rows"]
    arms = ["susceptible", "transmission_seed", "transmission"]
    names = ["susceptible\n$u(1-y)$", "seeded epidemic\n$u(y+\\epsilon)(1-y)$", "epidemic\n$u\\,y(1-y)$"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for j, kap in enumerate([1.0, 1.5]):
        w = 0.36
        for i, (key, col, lab) in enumerate((("rmse_onset", "#c1121f", "starts at $y=0$"),
                                             ("rmse_started", "#8ecae6", "already out"))):
            m = [np.mean([r[key] for r in rows if r["arm"] == a and r["kappa"] == kap]) for a in arms]
            s = [np.std([r[key] for r in rows if r["arm"] == a and r["kappa"] == kap]) for a in arms]
            ax[j].bar(np.arange(3) + (i - 0.5) * w, m, w, yerr=s, color=col, label=lab, capsize=3)
        ax[j].set_xticks(range(3)); ax[j].set_xticklabels(names, fontsize=8)
        ax[j].set_yscale("log")
        ax[j].set_title(f"generator $(1-y)^{{{kap}}}$" + ("  — neutral to all arms" if kap == 1.5 else ""),
                        fontsize=9)
        ax[j].spines[["top", "right"]].set_visible(False)
    ax[0].set_ylabel("trajectory RMSE"); ax[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig02_onset.png", dpi=200)
    print("figures/fig02_onset.png")




def fig_onset_real():
    """Where counties sit before the storm that interrupts them, on public data."""
    import numpy as np
    days = json.loads((ROOT / "results/panel_onset_audit.json").read_text())["days"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))

    lab = [d["event_day"] for d in days]
    z = np.array([d["frac_typ_zero"] for d in days]) * 100
    e4 = np.array([d["frac_lt_1e4"] for d in days]) * 100
    order = np.argsort(-z)
    yy = np.arange(len(days))
    ax[0].barh(yy, e4[order], color="#cfe3f5", label=r"typical $y<10^{-4}$")
    ax[0].barh(yy, z[order], color="#c1121f", label=r"typical $y=0$ exactly")
    ax[0].set_yticks(yy); ax[0].set_yticklabels([lab[i] for i in order], fontsize=8)
    ax[0].set_xlabel("% of interrupted counties"); ax[0].set_xlim(0, 100)
    ax[0].axvline(z.mean(), color="k", ls="--", lw=0.8)
    ax[0].text(z.mean() + 1.5, len(days) - 0.4, f"mean {z.mean():.0f}%", fontsize=8)
    ax[0].set_title("before the storm, the county is dark", fontsize=10)
    ax[0].legend(frameon=False, fontsize=8, loc="lower left",
                 bbox_to_anchor=(0.0, -0.42), ncol=2)

    # What the epidemic form's inflow is multiplied by, county-wise.
    all_pre = []
    for d in days:
        f = ROOT / f"data/interim/panel_{d['event_day']}.npz"
        if not f.exists():
            continue
        z_ = np.load(f, allow_pickle=True)
        y, obs, ts = z_["y"], z_["observed"], np.array(z_["ts"], dtype="datetime64[ns]")
        lead = ts < np.datetime64(d["event_day"])
        with np.errstate(all="ignore"):
            ever = np.nanmax(np.where(obs, y, np.nan), axis=1)
            pre = np.nanmedian(np.where(obs[:, lead], y[:, lead], np.nan), axis=1)
        m = np.nan_to_num(ever, nan=-1) >= 0.01
        all_pre.append(pre[m][np.isfinite(pre[m])])
    v = np.concatenate(all_pre)
    frac0 = (v <= 0).mean() * 100
    pos = v[v > 0]
    ax[1].hist(np.log10(pos), bins=40, color="#003049")
    ax[1].axvline(np.log10(1e-4), color="#c1121f", ls="--", lw=1)
    ax[1].set_xlabel(r"$\log_{10}$ typical pre-storm $y$   (counties with $y>0$)")
    ax[1].set_ylabel("counties")
    ax[1].set_title(f"the remainder is small, not large\n"
                    f"({frac0:.0f}% of all interrupted counties are at exactly 0, off-scale)",
                    fontsize=9)
    for a_ in ax: a_.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "fig03_onset_real.png", dpi=200)
    print("figures/fig03_onset_real.png")



def _paired(rows, a, b, h):
    A = {(r["seed"], r["fold"]): r for r in rows if r["arm"] == a}
    B = {(r["seed"], r["fold"]): r for r in rows if r["arm"] == b}
    ks = sorted(set(A) & set(B))
    d = np.array([A[k][f"rmse_h{h}"] - B[k][f"rmse_h{h}"] for k in ks])
    ref = np.mean([B[k][f"rmse_h{h}"] for k in ks])
    return 100 * d.mean() / ref, int((d < 0).sum()), len(ks)


def fig_he_by_family():
    """H-E: two-rate advantage over the parameter-matched single rate, by family."""
    d = json.loads((ROOT / "results/exp06_by_family.json").read_text())["rows"]
    probe = json.loads((ROOT / "results/exp06_convergence_probe.json").read_text())["rows"]
    fams = ["tropical", "wind", "convective", "winter"]; ratio = {"tropical": 4.0, "wind": 2.1, "convective": 1.7, "winter": 1.0}
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for j, h in enumerate((24, 48)):
        vals, wins = [], []
        for f in fams:
            fr = [r for r in d if r["family"] == f]
            rel, w, n = _paired(fr, "susceptible", "net_scaled", h); vals.append(rel); wins.append(f"{w}/{n}")
        cols = ["#c1121f" if v < 0 else "#7f7f7f" for v in vals]
        ax[j].bar(range(4), vals, color=cols, width=0.6)
        for i, (v, wt) in enumerate(zip(vals, wins)):
            ax[j].text(i, v + (0.25 if v >= 0 else -0.25), wt, ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
        for i, f in enumerate(("tropical", "winter")):
            fr = [r for r in probe if r["family"] == f]
            if fr:
                rel, w, n = _paired(fr, "susceptible", "net_scaled", h)
                ax[j].plot([fams.index(f)], [rel], marker="D", color="k", ms=6, ls="none",
                           label="converged probe (400 ep, seed 0)" if (j == 0 and i == 0) else None)
        ax[j].axhline(0, color="k", lw=0.6)
        ax[j].set_xticks(range(4)); ax[j].set_xticklabels([f"{f}\nfall/rise {ratio[f]:.1f}" for f in fams], fontsize=8)
        ax[j].set_title(f"h+{h}: two-rate vs parameter-matched single rate", fontsize=10)
        ax[j].spines[["top", "right"]].set_visible(False)
    ax[0].set_ylabel("paired Δ RMSE (%)   negative = two-rate better")
    ax[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("The two-rate advantage follows phase separation and reverses on winter", fontsize=11)
    fig.tight_layout(); fig.savefig(FIG / "fig04_he_by_family.png", dpi=200); print("figures/fig04_he_by_family.png")


def fig_per_horizon():
    """Per horizon: who beats whom on the primary (convective, g2) study."""
    e5 = json.loads((ROOT / "results/exp05_g2_sixarm.json").read_text())["rows"]
    e7 = json.loads((ROOT / "results/exp07_g2_oof.json").read_text())["rows"]
    e10 = json.loads((ROOT / "results/exp10_per_horizon.json").read_text())["rows"]
    # merge per-horizon rows of exp10 into unit rows
    u = {}
    for r in e10:
        k = (r["seed"], r["fold"]); u.setdefault(k, {"arm": "per_horizon", "seed": r["seed"], "fold": r["fold"]}).update({kk: v for kk, v in r.items() if kk.startswith("rmse_h")})
    e10u = list(u.values())
    ref = [r for r in e5 if r["arm"] == "susceptible"]
    series = [("gradient boosting (same inputs)", e7, "trees_matched", "#003049"),
              ("two-rate, one model per horizon", e10u, "per_horizon", "#2a9d8f"),
              ("single signed rate, param-matched", e5, "net_scaled", "#7f7f7f"),
              ("damped persistence", e5, "damped_persistence", "#cfcfcf")]
    H = [1, 6, 24, 48]; fig, ax = plt.subplots(figsize=(8.5, 3.8)); w = 0.2
    for i, (lab, rows, arm, col) in enumerate(series):
        vals = [_paired(rows + ref, arm, "susceptible", h)[0] for h in H]
        ax.bar(np.arange(4) + (i - 1.5) * w, vals, w, color=col, label=lab)
    ax.axhline(0, color="k", lw=0.6); ax.set_xticks(range(4)); ax.set_xticklabels([f"h+{h}" for h in H])
    ax.set_ylabel("paired Δ RMSE vs two-rate model (%)\nnegative = comparator better")
    ax.set_title("Convective study: the two-rate model wins at 1 h and loses to a per-horizon regressor beyond", fontsize=10)
    ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "fig05_per_horizon.png", dpi=200); print("figures/fig05_per_horizon.png")

if __name__ == "__main__":
    fig_identifiability(); fig_onset(); fig_onset_real(); fig_he_by_family(); fig_per_horizon()
