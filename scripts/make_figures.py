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
    ax[0].set_title("recovery improves as the state moves"); ax[0].legend(frameon=False, fontsize=9)

    a = agg(rows, "traj_rmse", "pulse_scale")
    m = np.array([a[j][0] for j in sorted(a)]); s = np.array([a[j][1] for j in sorted(a)])
    ax[1].plot(x, m, "s-", color="#7f7f7f"); ax[1].fill_between(x, m - s, m + s, color="#7f7f7f", alpha=0.18)
    ax[1].set_yscale("log"); ax[1].set_xlabel("state spread"); ax[1].set_ylabel("trajectory RMSE")
    ax[1].set_ylim(ax[0].get_ylim())
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


if __name__ == "__main__":
    fig_identifiability(); fig_onset()
