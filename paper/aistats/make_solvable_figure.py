#!/usr/bin/env python3
"""Reproduce the single solvable-case figure used in the paper.

Panels (a) and (c) re-express the supplied balanced two-state experiment in the
single index Gamma. Panel (b) adds the complementary coactivity slice. No neural
model is trained by this script.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "solvable_case_results.csv"
OUT = ROOT / "figures" / "flow_selection_solvable_case.pdf"

U = 0.20
R = 0.12
N = 40
SIGMA_EPS = 0.15
COST = SIGMA_EPS**2 / N
DELTA_REF = 0.15


def chi_square_one_cdf(x: np.ndarray) -> np.ndarray:
    """CDF of chi-square with one degree of freedom, without SciPy."""
    return np.array([math.erf(math.sqrt(max(float(z), 0.0) / 2.0)) for z in x])


def main() -> None:
    results = pd.read_csv(DATA).sort_values("gamma_theory")
    gamma_grid = np.linspace(0.0, max(13.0, float(results["gamma_theory"].max())), 600)

    ratio = np.linspace(0.0, 1.5, 500)
    variance = DELTA_REF**2
    second_moment = 0.25 + variance
    restoration = U * ratio
    coactivity = np.minimum(restoration**2 / second_moment, U**2 / second_moment)
    gamma_ratio = N * variance * coactivity / SIGMA_EPS**2

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.05))

    ax = axes[0]
    ax.plot(gamma_grid, gamma_grid - 1.0, label=r"theory: $\Gamma-1$")
    ax.errorbar(
        results["gamma_theory"],
        results["risk_difference_mc"] / COST,
        yerr=1.96 * results["risk_difference_mc_se"] / COST,
        fmt="o",
        markersize=3.0,
        linewidth=0.7,
        label="50,000-replication MC",
    )
    ax.axhline(0.0, linewidth=0.8)
    ax.axvline(1.0, linestyle=":", label=r"$\Gamma=1$")
    ax.set_xlabel(r"flow-selection index $\Gamma$")
    ax.set_ylabel(r"$(\mathbb{E}\,\mathrm{Err}_1-\mathbb{E}\,\mathrm{Err}_2)/(\sigma_\varepsilon^2/n)$")
    ax.set_title(r"(a) Risk difference collapses on $\Gamma$")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(alpha=0.2)

    ax = axes[1]
    ax.plot(ratio, gamma_ratio)
    ax.axhline(1.0, linestyle="--", label=r"$\Gamma=1$")
    ax.axvline(1.0, linestyle=":", label="one-flow branch switch")
    ax.set_xlabel(r"component ratio $R/U$")
    ax.set_ylabel(r"$\Gamma$ at fixed state dispersion")
    ax.set_title("(b) Coactivity determines the gain")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(alpha=0.2)

    ax = axes[2]
    ax.plot(gamma_grid, chi_square_one_cdf(gamma_grid), label=r"$F_{\chi^2_1}(\Gamma)$")
    ax.scatter(
        results["gamma_theory"],
        results["two_wins_probability_mc"],
        s=12,
        label="50,000-replication MC",
    )
    ax.axvline(1.0, linestyle=":")
    ax.set_xlabel(r"flow-selection index $\Gamma$")
    ax.set_ylabel(r"$\Pr(\mathrm{Err}_2<\mathrm{Err}_1)$")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("(c) Pairwise win probability")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(alpha=0.2)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.8, w_pad=1.0)
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
