"""Publication style shared by the open-data figures (two-column letter paper).

Text width 7.0 in, column width 3.375 in; fonts are set at printed size (6-7.5 pt), so
include figures at width=\\textwidth (figure*) or \\columnwidth without rescaling.
One colour per entity in every figure; the host W is always dashed so that it never
relies on colour alone.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEXT_WIDTH, COLUMN_WIDTH = 7.0, 3.375
OUT = Path(__file__).resolve().parent / "figures"
INK = "#1a1a1a"          # observed
GCRK = "#2a78d6"         # AsymODE + GCRK
TIMESFM = "#eb6834"      # TimesFM zero-shot
HOST = "#1baf7a"         # weather host W, dashed
MUTED = "#8c8c8c"
GRID = "#e6e6e3"
WASH = "#f3f1ec"         # observed-prefix wash
TEXT2 = "#52514e"
HOST_DASH = (0, (4.0, 1.6))
OFF, OFF_DASH = "#8c8c8c", (0, (1.2, 1.2))          # GCRK with the kernel exit closed
RAISE, LOWER = "#c2412a", "#5b3f9e"
RAISE_LIGHT, LOWER_LIGHT = "#e6a898", "#b9acdd"
RAISE_FILL, LOWER_FILL = "#f2c4ba", "#d6cdee"


def apply_style(font_size: float = 7.0) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "stixsans", "font.size": font_size, "axes.titlesize": font_size + 0.5,
        "axes.labelsize": font_size, "xtick.labelsize": font_size - 0.5, "ytick.labelsize": font_size - 0.5,
        "legend.fontsize": font_size, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "axes.edgecolor": "#3a3a3a", "axes.labelcolor": INK,
        "xtick.color": "#3a3a3a", "ytick.color": "#3a3a3a", "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.2,
        "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round", "legend.frameon": False,
        "savefig.dpi": 600, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })


def save(fig, name: str) -> list[Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = OUT / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight", pad_inches=0.02, dpi=600 if ext == "png" else None,
                    metadata={"CreationDate": None} if ext == "pdf" else None)
        paths.append(p)
    plt.close(fig)
    return paths
