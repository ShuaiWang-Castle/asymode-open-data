"""Figures 1-4 of the paper on the development tranche (PDF + PNG in paper_v1/figures/).

Figure 1  observed county peak outage fractions (hours 72-215): tropical-cyclone systems, winter systems, all systems;
          a county in several systems shows its largest peak.
Figure 2  model schematic: damage and recovery networks, GCRK and the population balance.
Figure 3  geography in GCRK, from paper_v1/counterfactual_v1.py: the prediction with the fitting counties' mean
          descriptors, the change of the peak with each county's own descriptors (percentage points), and the
          prediction with them. Per county, the largest predicted peak over its events.
Figure 4  trajectories of eight held-out counties: the top weather-twin pair of the tropical and of the winter systems
          (paper_v1/twins_v1.py), then per regime (synoptic wind, convective, heavy rain, tropical, winter) the county-event
          with at least 10,000 customers and the largest observed 3-hour peak not yet shown, until eight.
Learned models average the seeds that have all five folds; TimesFM is zero-shot.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP.parent / "open_gcrk_20260919")); sys.path.insert(0, str(HERE))
import fig_style as S  # noqa: E402
from evaluate_paper import FEAT, MODELS, model_paths  # noqa: E402
from twins_v1 import peak3  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402  (fig_style selects the Agg backend)
from matplotlib.colors import LinearSegmentedColormap, LogNorm, SymLogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402

S.OUT = HERE / "figures"
CF = ROOT / "runs" / "geo_weather_20260924" / "paper_v1"
SHAPES = ROOT / "data" / "raw" / "census" / "cb_2020_us_county_500k.zip"
GAZ = ROOT / "data" / "raw" / "census" / "2020_Gaz_counties_national.txt"
LINES = {"TimesFM": (S.TIMESFM, "-", 1.0), "AsymODE": (S.HOST, S.HOST_DASH, 1.1), "AsymODE + GCRK": (S.GCRK, "-", 1.3)}
LABEL = {"TimesFM": "TimesFM (zero-shot)"}
REGIME_NAME = {"tropical": "tropical cyclone", "winter": "winter", "synoptic_wind": "synoptic wind",
               "convective": "convective", "heavy_rain": "heavy rain"}
DIV = LinearSegmentedColormap.from_list("change", [S.LOWER, "#efeeea", S.RAISE])
PEAK_NORM, PEAK_CMAP = LogNorm(0.1, 50), plt.get_cmap("YlOrRd")
PREFIX_END = 71


def features() -> dict:
    z = np.load(FEAT)
    return {k: z[k] for k in ("fips", "system", "regime", "origin", "cust", "y0", "obs_full", "y_full")}


def conus():
    import geopandas as gpd
    shp = gpd.read_file(f"zip://{SHAPES}")
    shp["fips"] = shp.STATEFP + shp.COUNTYFP
    c = shp[~shp.STATEFP.isin(["02", "15", "60", "66", "69", "72", "78"])].to_crs(5070)
    states = c.dissolve("STATEFP")
    states["geometry"] = states.geometry.simplify(800)
    c = c.copy(); c["geometry"] = c.geometry.simplify(800)
    return c, states


def county_names() -> dict[str, str]:
    gz = pd.read_csv(GAZ, sep="\t", dtype={"GEOID": str}, encoding="latin-1")
    gz.columns = [c.strip() for c in gz.columns]
    return {g.zfill(5): f"{n.replace(' County', '').replace(' Parish', '')}, {s}" for g, n, s in zip(gz.GEOID, gz.NAME, gz.USPS)}


def county_max(fips: np.ndarray, v: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(dict(fips=fips, v=v)).groupby("fips", as_index=False)["v"].max()


def draw_map(ax, counties, states, d: pd.DataFrame, cmap, norm, title: str, letter: str | None = None):
    states.plot(ax=ax, color="#eeede9", lw=0, zorder=0, rasterized=True)
    g = counties.merge(d, on="fips", how="inner")
    g.plot(ax=ax, column="v", cmap=cmap, norm=norm, lw=0, zorder=2, rasterized=True)
    states.boundary.plot(ax=ax, color="white", lw=0.3, zorder=3)
    ax.set_axis_off()
    ax.set_title(title, fontsize=6.8, loc="left", pad=2)
    if letter:
        ax.text(-0.01, 1.0, letter, transform=ax.transAxes, fontsize=8.2, fontweight="bold", ha="right", va="bottom")


def colorbar(fig, rect, cmap, norm, label, extend="both"):
    cax = fig.add_axes(rect)
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, extend=extend)
    cb.set_label(label, fontsize=6.2); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=5.8)
    if isinstance(norm, LogNorm):
        ticks = [t for t in (0.1, 0.3, 1, 3, 10, 30, 50) if norm.vmin <= t <= norm.vmax]
        cb.set_ticks(ticks); cb.set_ticklabels([f"{t:g}" for t in ticks]); cb.ax.minorticks_off()
    return cb


def observed(F) -> np.ndarray:
    return np.where(F["obs_full"], F["y_full"], np.nan).astype(float)


def fig1():
    S.apply_style(7.0)
    F = features(); y = observed(F)
    peak = 100 * np.array([peak3(r) for r in y])
    peak = np.nan_to_num(peak, nan=0.0)
    counties, states = conus()
    reg = F["regime"].astype(str)
    panels = [("a", "Tropical-cyclone systems", reg == "tropical"), ("b", "Winter systems", reg == "winter"),
              ("c", "All 81 systems", np.ones(len(reg), bool))]
    fig, axes = plt.subplots(1, 3, figsize=(S.TEXT_WIDTH, 1.95))
    fig.subplots_adjust(left=0.02, right=0.925, top=0.9, bottom=0.02, wspace=0.04)
    for ax, (letter, title, sel) in zip(axes, panels):
        d = county_max(F["fips"][sel].astype(str), np.clip(peak[sel], 0.1, 50))
        draw_map(ax, counties, states, d, PEAK_CMAP, PEAK_NORM, f"{title} ({len(d):,} counties)", letter)
    colorbar(fig, [0.935, 0.14, 0.011, 0.66], PEAK_CMAP, PEAK_NORM, "peak customers out (%)")
    S.save(fig, "fig1_impacts_v1")


def fig3():
    S.apply_style(7.0)
    F = features(); fips = F["fips"].astype(str)
    counties, states = conus()
    fig, axes = plt.subplots(1, 3, figsize=(S.TEXT_WIDTH, 1.95))
    fig.subplots_adjust(left=0.02, right=0.925, top=0.9, bottom=0.02, wspace=0.04)
    dnorm = SymLogNorm(linthresh=0.1, linscale=0.6, vmin=-10, vmax=10, base=10)
    z = np.load(CF / "counterfactual_gcrk.npz")
    a, b = county_max(fips, 100 * z["P_mean_geo"].max(1)), county_max(fips, 100 * z["P_own"].max(1))
    d = a.merge(b, on="fips", suffixes=("_ref", "_own")).assign(v=lambda t: t.v_own - t.v_ref)[["fips", "v"]]
    draw_map(axes[0], counties, states, a.assign(v=a.v.clip(0.1, 50)), PEAK_CMAP, PEAK_NORM, "Prediction with mean geography", "a")
    draw_map(axes[1], counties, states, d.assign(v=d.v.clip(-10, 10)), DIV, dnorm, "Effect of own geography in GCRK", "b")
    draw_map(axes[2], counties, states, b.assign(v=b.v.clip(0.1, 50)), PEAK_CMAP, PEAK_NORM, "Prediction with own geography", "c")
    cax = axes[1].inset_axes([0.04, 0.05, 0.36, 0.035])
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=dnorm, cmap=DIV), cax=cax, orientation="horizontal", extend="both")
    cb.set_ticks([-10, -1, 0, 1, 10]); cb.set_ticklabels(["−10", "−1", "0", "1", "10"]); cb.ax.minorticks_off()
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=5.2, length=1.5, pad=1)
    cb.set_label("peak change (points)", fontsize=5.4, labelpad=1); cb.ax.xaxis.set_label_position("top")
    colorbar(fig, [0.935, 0.14, 0.011, 0.66], PEAK_CMAP, PEAK_NORM, "peak customers out (%)")
    summary = dict(seeds=[int(s) for s in z["seeds"]], counties=int(len(d)), raised_by_more_than_0p1=float((d.v > 0.1).mean()),
                   lowered_by_more_than_0p1=float((d.v < -0.1).mean()), mean_abs_change_points=float(np.abs(d.v).mean()))
    (HERE / "figures" / "fig3_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    S.save(fig, "fig3_geography_v1")


def fig4_units(F) -> list[int]:
    twins = json.loads((EXP / "results" / "v1" / "weather_twin_pairs.json").read_text())["pairs"]
    fips, sysv, reg = F["fips"].astype(str), F["system"].astype(str), F["regime"].astype(str)
    unit = {(s, f): i for i, (s, f) in enumerate(zip(sysv, fips))}
    chosen = []
    for r in ("tropical", "winter"):
        p = next(t for t in twins if t["regime"] == r)
        chosen += [unit[(p["system"], p["fips_high"])], unit[(p["system"], p["fips_low"])]]
    y = observed(F)
    pk = np.array([peak3(row) for row in y])
    order = ("synoptic_wind", "convective", "heavy_rain", "tropical", "winter")
    k = 0
    while len(chosen) < 8:
        r = order[k % len(order)]; k += 1
        c = [i for i in np.where((reg == r) & (F["cust"] >= 10000))[0] if i not in chosen and np.isfinite(pk[i])]
        chosen.append(max(c, key=lambda i: pk[i]))
    return chosen


def fig4():
    S.apply_style(7.0)
    z = np.load(FEAT); F = {k: z[k] for k in z.files}
    n = len(F["fips"])
    paths = {}
    for name, pat in MODELS:
        if pat is None:
            continue
        P, _, used = model_paths(pat, n, F)
        if P is not None:
            paths[name] = P
    units = fig4_units(F)
    names = county_names()
    y = observed(F)
    fig, axes = plt.subplots(2, 4, figsize=(S.TEXT_WIDTH, 3.15), sharex=True)
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.11, top=0.82, wspace=0.28, hspace=0.5)
    h = np.arange(216)
    choice = []
    for k, (ax, u) in enumerate(zip(axes.ravel(), units)):
        obs = 100 * y[u]
        curves = {nm: 100 * np.r_[np.full(71, np.nan), F["y0"][u], P[u]] for nm, P in paths.items()}
        top = np.nanmax(np.r_[obs, *curves.values()]) if curves else np.nanmax(obs)
        ax.axvspan(0, PREFIX_END, color=S.WASH, lw=0, zorder=0); ax.axvline(PREFIX_END, color=S.MUTED, lw=0.5, zorder=1)
        ax.plot(h, obs, color=S.INK, lw=1.0, zorder=3)
        for j, (nm, c) in enumerate(curves.items()):
            col, ls, lw = LINES[nm]
            ax.plot(h, c, color=col, ls=ls, lw=lw, zorder=4 + j)
        ax.set_xlim(0, 215); ax.set_ylim(0, top * 1.12 if top > 0 else 1.0)
        ax.yaxis.set_major_locator(MaxNLocator(3, min_n_ticks=3)); ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xticks([0, 48, 96, 144, 192])
        fp = str(F["fips"][u]); reg = str(F["regime"][u]); day = str(F["origin"][u])[:10]
        ax.set_title(names.get(fp, fp), loc="left", fontsize=7.0, fontweight="bold", pad=9)
        ax.text(0.0, 1.015, f"{REGIME_NAME[reg]} · {day}", transform=ax.transAxes, fontsize=5.8, color=S.TEXT2, va="bottom")
        if k % 4 == 0:
            ax.set_ylabel("Customers out (%)", fontsize=6.8)
        if k >= 4:
            ax.set_xlabel("Hour", fontsize=6.8, labelpad=1.5)
        choice.append(dict(unit=int(u), fips=fp, system=str(F["system"][u]), regime=reg, origin=str(F["origin"][u])))
    handles = [Line2D([], [], color=S.INK, lw=1.2, label="Observed")]
    handles += [Line2D([], [], color=LINES[nm][0], ls=LINES[nm][1], lw=LINES[nm][2] + 0.2, label=LABEL.get(nm, nm)) for nm in paths]
    handles.append(Patch(facecolor=S.WASH, edgecolor="none", label="observed, hours 0–71"))
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0), fontsize=6.8, handlelength=2.0,
               columnspacing=1.4, frameon=False)
    (HERE / "figures" / "fig4_choice.json").write_text(json.dumps(choice, indent=1) + "\n")
    S.save(fig, "fig4_trajectories_v1")
    # the numbers the text quotes: observed 3-hour peaks and forecast peaks per panel (A-H, row by row)
    key = {"TimesFM": "TimesFM", "AsymODE": "Host", "AsymODE + GCRK": "GCRK"}
    mac = []
    for L, u in zip("ABCDEFGH", units):
        mac.append(rf"\newcommand{{\FigFourName{L}}}{{{names.get(str(F['fips'][u]), str(F['fips'][u]))}}}")
        mac.append(rf"\newcommand{{\FigFourObs{L}}}{{{100 * peak3(y[u]):.0f}\%}}")
        for nm, P in paths.items():
            mac.append(rf"\newcommand{{\FigFour{key[nm]}{L}}}{{{100 * P[u].max():.0f}\%}}")
    efg = max(100 * P[u].max() for P in paths.values() for u in units[4:7])
    mac.append(rf"\newcommand{{\FigFourMaxEFG}}{{{efg:.0f}\%}}")
    (HERE / "generated").mkdir(exist_ok=True)
    (HERE / "generated" / "numbers_fig4.tex").write_text("% generated by paper_v1/figures_v1.py; do not edit\n" + "\n".join(mac) + "\n")


def fig2():
    """Model schematic: damage and recovery networks, GCRK and the population balance."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    S.apply_style(7.0)
    fig = plt.figure(figsize=(S.TEXT_WIDTH, 2.45))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 35); ax.set_axis_off()
    LIGHT = {"in": "#f3f1ec", "dmg": "#e3f4ec", "gcrk": "#e1ecfa", "rec": "#fbe9dd", "bal": "#ffffff"}
    EDGE = {"in": "#8c8c8c", "dmg": S.HOST, "gcrk": S.GCRK, "rec": S.TIMESFM, "bal": S.INK}

    def box(x, y, w, h, kind, title, body="", fs=6.4):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8", fc=LIGHT[kind],
                                    ec=EDGE[kind], lw=0.8))
        ax.text(x + w / 2, y + h - 0.9, title, ha="center", va="top", fontsize=fs + 0.4, fontweight="bold", color=S.INK)
        if body:
            ax.text(x + w / 2, y + h - 3.1, body, ha="center", va="top", fontsize=fs - 0.6, color=S.TEXT2, linespacing=1.15)

    def arrow(p, q, color=S.INK):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=6, lw=0.8, color=color, shrinkA=1, shrinkB=1))

    box(2, 26, 24, 8, "in", "Weather", "ERA5 channels and\ncausal summaries")
    box(36, 26, 24, 8, "in", "Geography $g_i$", "40 county descriptors:\nterrain, canopy, soils, forest")
    box(72, 26, 26, 8, "in", "Recovery inputs", "weather, outages of hours 0-71,\ncounty context, neighbours")
    box(2, 13.5, 24, 8, "dmg", "Damage network", "two layers, width 32;\noccurrence gate, background")
    box(36, 13.5, 24, 8, "gcrk", "GCRK", "one bounded response state;\ndissipation, interaction and\nreadout gain set by $g_i$")
    box(72, 13.5, 26, 8, "rec", "Recovery network", "two layers, width 16,\nrecomputed every hour")
    box(12, 1.5, 76, 6.6, "bal", "Population balance",
        "$p_{i,t} = p_{i,t-1} + u_{i,t}\\,(1 - p_{i,t-1}) - r_{i,t}\\,p_{i,t-1}$:   damage $u$ acts on customers still served,"
        "\nrecovery $r$ on customers out; one open-loop path from the observed $p_{i,71}$", fs=6.6)
    arrow((14, 26), (14, 21.9)); arrow((48, 26), (48, 21.9), S.GCRK); arrow((85, 26), (85, 21.9), S.TIMESFM)
    arrow((26.4, 18.6), (35.6, 18.6), S.GCRK); arrow((35.6, 15.6), (26.4, 15.6), S.GCRK)
    ax.text(31, 19.2, "$h_{i,t}$", ha="center", va="bottom", fontsize=5.8, color=S.GCRK)
    ax.text(31, 14.9, "response", ha="center", va="top", fontsize=5.4, color=S.GCRK)
    arrow((14, 13.5), (26, 8.4), S.HOST); ax.text(21.2, 11.4, "$u_{i,t}$", fontsize=6.2, color=S.HOST)
    arrow((85, 13.5), (74, 8.4), S.TIMESFM); ax.text(81.8, 11.2, "$r_{i,t}$", fontsize=6.2, color=S.TIMESFM)
    S.save(fig, "fig2_model_v1")


if __name__ == "__main__":
    for w in sys.argv[1:] or ["fig1", "fig2", "fig3", "fig4"]:
        {"fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4}[w]()
