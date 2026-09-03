#!/usr/bin/env python3
"""Reproduce the single solvable-case figure used in the paper.

The first two panels preserve the supplied balanced two-state case. The third
panel adds the coactivity slice requested by the independent review. No model
is trained by this script.
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
DELTA_C = SIGMA_EPS / (2.0 * math.sqrt(N * R**2 - SIGMA_EPS**2))


def main() -> None:
    results = pd.read_csv(DATA)
    delta = np.linspace(0.001, 0.49, 500)
    gap = 4.0 * R**2 * delta**2 / (1.0 + 4.0 * delta**2)
    one_risk = gap + COST
    two_risk = np.full_like(delta, 2.0 * COST)

    flow_ratio = np.linspace(0.0, 1.5, 500)
    normalized_coactivity = np.minimum(flow_ratio**2, 1.0)

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.05))

    ax = axes[0]
    ax.plot(delta, gap, label=r"$G(\delta)$")
    ax.axhline(COST, linestyle="--", label=r"$\sigma_\varepsilon^2/n$")
    ax.axvline(DELTA_C, linestyle=":", label=rf"$\delta_c={DELTA_C:.3f}$")
    ax.set_xlabel(r"state spread $\delta$")
    ax.set_ylabel("squared-risk scale")
    ax.set_title("(a) Representation gain vs. cost")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(alpha=0.2)

    ax = axes[1]
    ax.plot(delta, one_risk, label="one-flow theory")
    ax.plot(delta, two_risk, label="two-flow theory")
    ax.errorbar(
        results["delta"],
        results["one_risk_mc"],
        yerr=1.96 * results["one_risk_mc_se"],
        fmt="o",
        markersize=2.6,
        linewidth=0.7,
        label="one-flow MC",
    )
    ax.errorbar(
        results["delta"],
        results["two_risk_mc"],
        yerr=1.96 * results["two_risk_mc_se"],
        fmt="s",
        markersize=2.3,
        linewidth=0.7,
        label="two-flow MC",
    )
    ax.axvline(DELTA_C, linestyle=":")
    ax.set_xlabel(r"state spread $\delta$")
    ax.set_ylabel("expected fitted-mean MSE")
    ax.set_title("(b) Exact finite-sample crossover")
    ax.legend(frameon=False, fontsize=6.7)
    ax.grid(alpha=0.2)

    ax = axes[2]
    ax.plot(flow_ratio, normalized_coactivity)
    ax.axvline(1.0, linestyle=":", label="branch switch")
    ax.axhline(0.0, linewidth=0.8)
    ax.set_xlabel(r"flow ratio $R/U$")
    ax.set_ylabel(r"$\kappa/\kappa_{\max}$")
    ax.set_title("(c) Coactivity controls the gain")
    ax.set_ylim(-0.03, 1.08)
    ax.legend(frameon=False, fontsize=7)
    ax.grid(alpha=0.2)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.8, w_pad=1.1)
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
