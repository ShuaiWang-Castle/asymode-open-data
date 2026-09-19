"""Markdown and LaTeX tables from results/*.csv (no computation beyond formatting).

Writes results/tables_<design>.md and results/table_main_<design>.tex.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

import common as C

COLS = ["rmse", "rmse 1-6 h", "rmse 7-24 h", "rmse 25-48 h", "rmse 49-144 h", "mae"]
HEAD = ["RMSE all", "1-6 h", "7-24 h", "25-48 h", "49-144 h", "MAE"]
PEAK = ["peak_mag_abs_mean", "peak_time_abs_mean", "false_activity_share", "under_half_share"]
ROWS = ["all-zero", "persistence", "TimesFM", "TimesFM (history only)", "W", "GCRK"]


def fmt(v, sd=None, d=5):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    s = f"{v:.{d}f}"
    return s if sd is None or np.isnan(sd) else f"{s} ± {sd:.{d}f}"


def main(design="main"):
    R = C.RESULTS
    t = pd.read_csv(R / f"table_main_{design}.csv").set_index("model")
    md = [f"## Main table ({design})", "",
          "Pooled over held-out observed county-hours of all folds; W and GCRK = mean ± sd over seeds 0-4.",
          "Source: `evaluate.py " + design + "` -> `results/table_main_" + design + ".csv`.", "",
          "| model | " + " | ".join(HEAD) + " |", "|---|" + "---:|" * len(HEAD)]
    for r in ROWS:
        if r in t.index:
            md.append(f"| {r} | " + " | ".join(fmt(t.loc[r, c], t.loc[r, c + ' sd'] if c + " sd" in t.columns else None)
                                              for c in COLS) + " |")
    md += ["", "| model | |peak magnitude error| | |peak time error| (h) | false activity share | under-half share |",
           "|---|---:|---:|---:|---:|"]
    for r in ROWS:
        if r in t.index:
            ptime = ("n/a (constant forecast)" if r in ("all-zero", "persistence") else
                     fmt(t.loc[r, PEAK[1]], t.loc[r, PEAK[1] + ' sd'] if PEAK[1] + ' sd' in t.columns else None, 1))
            md.append(f"| {r} | {fmt(t.loc[r, PEAK[0]], t.loc[r, PEAK[0] + ' sd'] if PEAK[0] + ' sd' in t.columns else None, 4)} | "
                      f"{ptime} | "
                      f"{fmt(t.loc[r, PEAK[2]], None, 3)} | {fmt(t.loc[r, PEAK[3]], None, 3)} |")
    p = pd.read_csv(R / f"paired_{design}.csv")
    md += ["", f"## Paired seed-wise differences, GCRK - W ({design})", "",
           "Source: `results/paired_" + design + ".csv` (per-seed values in `results/seeds_" + design + ".csv`).", "",
           "| metric | W | GCRK | mean diff | rel. change | seeds GCRK lower | per-seed diff | verdict (PREREG 9) |",
           "|---|---:|---:|---:|---:|---:|---|---|"]
    for r in p.itertuples():
        md.append(f"| {r.metric} | {r.W_mean:.5g} | {r.GCRK_mean:.5g} | {r.diff_mean:+.3e} | {r.rel_pct_mean:+.2f}% | "
                  f"{r.n_gcrk_lower}/{r.n_seeds} | {r.per_seed_diff} | {r.verdict} |")
    dec = pd.read_csv(R / f"decomposition_{design}.csv")
    md += ["", f"## Kernel exit decomposition of the full-rollout RMSE ({design})", "",
           "closed = the trained GCRK network with its response contribution removed. Source: `results/decomposition_"
           + design + ".csv`.", "", "| seed | W | GCRK exit closed | GCRK | joint training (closed - W) | kernel output (GCRK - closed) | total |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for r in dec.itertuples():
        md.append(f"| {r.seed} | {r.W:.6f} | {r.closed:.6f} | {r.open:.6f} | {r.joint_training:+.2e} | {r.kernel_output:+.2e} | {r.total:+.2e} |")
    e = pd.read_csv(R / f"events_{design}.csv")
    md += ["", f"## By event: full-rollout RMSE ({design})", "", "Source: `results/events_" + design + ".csv`.", "",
           "| event | units | all-zero | persistence | TimesFM | W | GCRK | GCRK - W | seeds GCRK lower |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for ev, g in e.groupby("event"):
        g = g.set_index("model")
        get = lambda m: g.loc[m, "rmse"] if m in g.index else np.nan
        gs = lambda m: g.loc[m, "rmse_sd"] if m in g.index else np.nan
        md.append(f"| {ev} | {int(g.n_units.iloc[0])} | {get('all-zero'):.5f} | {get('persistence'):.5f} | {get('TimesFM'):.5f} | "
                  f"{fmt(get('W'), gs('W'))} | {fmt(get('GCRK'), gs('GCRK'))} | {get('GCRK - W'):+.2e} | "
                  f"{int(g.loc['GCRK - W', 'gcrk_lower_seeds'])}/5 |")
    (R / f"tables_{design}.md").write_text("\n".join(md) + "\n")
    tex = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
           r"&\multicolumn{5}{c}{RMSE of the outage fraction}&\\", r"\cmidrule(lr){2-6}",
           r"Model&all&1--6 h&7--24 h&25--48 h&49--144 h&MAE\\", r"\midrule"]
    names = {"all-zero": "All zero", "persistence": "Persistence", "TimesFM": "TimesFM", "W": "Weather host W",
             "GCRK": "AsymODE + GCRK"}
    for r in ("all-zero", "persistence", "TimesFM", "W", "GCRK"):
        if r in t.index:
            vals = [f"{t.loc[r, c]:.5f}" + (f"\\,{{\\scriptsize$\\pm${t.loc[r, c + ' sd']:.5f}}}" if r in ("W", "GCRK") else "")
                    for c in COLS]
            tex.append(f"{names[r]}&" + "&".join(vals) + r"\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (R / f"table_main_{design}.tex").write_text("\n".join(tex) + "\n")
    print("\n".join(md[:14]))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "main")
