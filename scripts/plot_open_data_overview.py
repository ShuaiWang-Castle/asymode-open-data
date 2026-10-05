"""Render four descriptive figures for the public 26-event outage panels.

The compact input in results/open_data_figure_data.json was extracted from the
archived D-7 audit on the aistats-current branch. These plots are descriptive;
they do not show model performance.

    python scripts/plot_open_data_overview.py

Requires matplotlib and numpy. Figure output is written to figures/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "results/open_data_figure_data.json").read_text())
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

FAMILIES = ["convective", "winter", "wind", "tropical", "flood"]
COLORS = {
    "convective": "#c43b2b",
    "winter": "#2377ad",
    "wind": "#2a9d8f",
    "tropical": "#d18a10",
    "flood": "#6c7280",
}
NAVY = "#012169"
GRAY = "#53616e"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.edgecolor": "#8795a3",
    "xtick.color": GRAY,
    "ytick.color": GRAY,
    "text.color": NAVY,
    "axes.labelcolor": NAVY,
    "savefig.facecolor": "white",
})


def base_axes(title: str, subtitle: str):
    fig, ax = plt.subplots(figsize=(8.6, 4.8), layout="constrained")
    fig.set_facecolor("white")
    ax.set_title(title, loc="left", pad=25, color=NAVY)
    ax.text(0, 1.025, subtitle, transform=ax.transAxes, color=GRAY, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dce3e9", linewidth=0.7)
    ax.set_axisbelow(True)
    return fig, ax


def save(fig, filename: str):
    fig.savefig(OUT / filename, dpi=160)
    plt.close(fig)


def zero_support():
    fig, ax = base_axes(
        "Most scored county-hours have little or no outage",
        "Observed cells only · 26 storm events · four forecast horizons",
    )
    horizons = [1, 6, 24, 48]
    series = [
        ("Exactly zero", "exact_zero", NAVY),
        ("At most 0.01%", "le_1e4", "#2377ad"),
        ("At most 0.1%", "le_1e3", "#81c8df"),
    ]
    x = np.arange(len(horizons))
    width = 0.23
    for i, (label, key, color) in enumerate(series):
        vals = [100 * DATA["zero_by_horizon"][str(h)][key] for h in horizons]
        ax.bar(x + (i - 1) * width, vals, width, label=label, color=color)
    ax.set_xticks(x, [f"+{h} h" for h in horizons])
    ax.set_ylabel("Share of scored cells (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", frameon=False, ncol=3, fontsize=9)
    save(fig, "open_data_zero_support.png")


def family_dots(metric: str, title: str, subtitle: str, ylabel: str,
                filename: str, median_label: str):
    fig, ax = base_axes(title, subtitle)
    panels = DATA["per_panel"]
    vals = []
    for i, family in enumerate(FAMILIES):
        rows = [p for p in panels if p["family"] == family]
        y = np.array([100 * p[metric] for p in rows])
        vals.extend(y)
        offsets = np.linspace(-0.18, 0.18, len(rows)) if len(rows) > 1 else np.array([0])
        ax.scatter(i + offsets, y, s=52, color=COLORS[family], alpha=0.9, zorder=3)
        ax.plot([i - 0.28, i + 0.28], [np.median(y)] * 2,
                color=NAVY, lw=2.2, zorder=4)
    median = np.median(vals)
    ax.axhline(median, color="#9aa5ae", lw=1, ls="--", zorder=1)
    ax.text(4.45, median + 1.4, f"{median_label}: {median:.0f}%",
            ha="right", fontsize=10, color=GRAY)
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(0, min(100, max(vals) * 1.20))
    ax.set_xticks(range(5), [f"{f}\n(n={sum(p['family'] == f for p in panels)})"
                             for f in FAMILIES])
    ax.set_ylabel(ylabel)
    save(fig, filename)


def daily_energy():
    fig, ax = base_axes(
        "Outage intensity persists after storm day",
        "Mean share of each event's total squared outage fraction · 26 storms",
    )
    panels = DATA["per_panel"]
    days = np.arange(-2, 5)
    for family in FAMILIES:
        rows = [p for p in panels if p["family"] == family]
        values = np.array([[100 * p["daily_share"].get(str(d), 0) for d in days]
                           for p in rows])
        ax.plot(days, values.mean(axis=0), marker="o", lw=2.2, ms=5,
                color=COLORS[family], label=f"{family} (n={len(rows)})")
    ax.axvline(0, color="#9aa5ae", lw=1, ls="--")
    ax.set_xticks(days, [str(d) for d in days])
    ax.set_xlabel("Day relative to catalogued storm day")
    ax.set_ylabel("Share of event y² (%)")
    ax.set_ylim(0, 70)
    ax.legend(frameon=False, ncol=2, fontsize=9, loc="upper left")
    save(fig, "open_data_daily_energy.png")


def main():
    zero_support()
    family_dots(
        "top3_share",
        "Outage intensity is concentrated in a few counties",
        "Each dot is one event; bars mark family medians · 26 storms",
        "Top 3 counties' share of y² (%)",
        "open_data_county_concentration.png",
        "Overall median",
    )
    daily_energy()
    family_dots(
        "multiwave_share",
        "Repeat waves occur across storm families",
        "Each dot is one event; bars mark family medians · 26 storms",
        "Interrupted counties with ≥2 waves (%)",
        "open_data_repeat_waves.png",
        "Event median",
    )


if __name__ == "__main__":
    main()