"""Figures 3 and 4 of the open-data replication (PDF + PNG in figures/).

Figure 3  (a) the kernel's contribution to the representative county's damage logit by
          deposit hour; (b) what drives its push-window push, raises and lowers separated,
          weather dark and geography light; (c) net push per forecast hour over the
          forecasts: observed, W, AsymODE + GCRK and the same GCRK with its kernel exit
          closed, the push's effect shaded; (d) pooled RMSE change of GCRK relative to W
          by lead time, mean over seeds with every seed shown.
Figure 4  eight county trajectories chosen by PREREG 11: observed, GCRK (seed 0, own
          outer fold) and TimesFM zero-shot.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator

import common as C
from fig_style import (GCRK, HOST, HOST_DASH, INK, LOWER, LOWER_FILL, LOWER_LIGHT, MUTED, OFF, OFF_DASH, RAISE,
                       RAISE_FILL, RAISE_LIGHT, TEXT2, TEXT_WIDTH, TIMESFM, WASH, apply_style, save)

FIGDATA = C.RESULTS / "figure_data"
DIV = LinearSegmentedColormap.from_list("contribution", [LOWER, "#efeeea", RAISE])
PREFIX_END = 71


def county_names() -> dict[str, str]:
    gz = pd.read_csv(C.ROOT / "data/raw/census/2023_Gaz_counties_national.txt", sep="\t", dtype={"GEOID": str},
                     encoding="latin-1")
    gz.columns = [c.strip() for c in gz.columns]
    return {g.zfill(5): f"{n.replace(' County', '').replace(' Parish', '')}, {s}" for g, n, s in zip(gz.GEOID, gz.NAME, gz.USPS)}


def title(ax, letter, text, fs, y=1.0):
    ax.text(0.0, y, text, transform=ax.transAxes, ha="left", va="bottom", fontsize=fs + 0.3, color=INK)
    ax.text(-0.03, y, letter, transform=ax.transAxes, ha="right", va="bottom", fontsize=fs + 1.5, fontweight="bold", color=INK)


def panel_a(ax, ax_in, cax, z, fs):
    Cm = z["C"].astype(float)
    t, v = np.meshgrid(np.arange(72, 216), np.arange(216), indexing="ij")
    Cm = np.where(v <= t, Cm, np.nan)
    lim = float(np.nanpercentile(np.abs(Cm), 99.5)) or 1e-6
    im = ax.imshow(Cm, origin="lower", aspect="auto", extent=[-0.5, 215.5, 71.5, 215.5], cmap=DIV,
                   norm=TwoSlopeNorm(0, -lim, lim), interpolation="nearest")
    ax.plot([72, 215], [72, 215], color=MUTED, lw=0.5)
    ax.axvline(PREFIX_END + 0.5, color=MUTED, lw=0.5, ls=(0, (1.5, 1.5)))
    pw = z["push_window"]
    ax.axhspan(pw[0] - 0.5, pw[1] + 0.5, xmin=0, xmax=0.012, color=RAISE, lw=0)
    ax.set_xlim(0, 215); ax.set_ylim(72, 215); ax.set_xticks([0, 48, 96, 144, 192]); ax.set_yticks([72, 120, 168, 215])
    ax.set_xlabel("Deposit hour $v$", labelpad=1.5); ax.set_ylabel("Forecast hour $t$")
    ax.text(PREFIX_END / 2, 210, "observed\nprefix", fontsize=fs - 1.1, color=TEXT2, ha="center", va="top", linespacing=1.0)
    dep = z["deposit_norm"]
    ax_in.axvspan(0, PREFIX_END, color=WASH, lw=0, zorder=0)
    ax_in.fill_between(np.arange(216), 0, dep, color=TEXT2, alpha=0.2, lw=0)
    ax_in.plot(np.arange(216), dep, color=TEXT2, lw=0.45)
    ax_in.set_xlim(0, 215); ax_in.set_ylim(0, max(1e-6, dep.max() * 1.1))
    for s in ax_in.spines.values():
        s.set_visible(False)
    ax_in.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax_in.set_ylabel(r"$\|\mathbf{d}_v\|$", rotation=0, ha="right", va="center", fontsize=fs - 0.4, color=TEXT2, labelpad=2)
    cb = plt.colorbar(im, cax=cax, extend="both")
    cb.set_ticks([-lim, 0, lim]); cb.set_ticklabels([f"{-lim:.2g}".replace("-", "−"), "0", f"+{lim:.2g}"])
    cb.outline.set_visible(False); cb.ax.tick_params(length=2, labelsize=fs - 0.8, pad=1.5)
    cb.ax.set_title("logit", fontsize=fs - 0.8, pad=3, color=TEXT2)


def panel_b(ax_up, ax_dn, drv, fs, n_up=6):
    g = drv.groupby(["group", "kind"]).attribution.sum().reset_index()
    push = float(drv.window_push.iloc[0])
    by_kind = drv.groupby("kind").attribution.sum()
    up = g[g.attribution > 0].sort_values("attribution", ascending=False)
    rest = up.iloc[n_up:]; up = up.iloc[:n_up]
    if len(rest):
        up = pd.concat([up, pd.DataFrame([dict(group=f"Other groups ({len(rest)})", kind="mixed", attribution=rest.attribution.sum())])])
    dn = g[g.attribution < 0].sort_values("attribution")
    rest = dn.iloc[5:]; dn = dn.iloc[:5]
    if len(rest):
        dn = pd.concat([dn, pd.DataFrame([dict(group=f"Other groups ({len(rest)})", kind="mixed", attribution=rest.attribution.sum())])])
    xmax = 1.25 * max(up.attribution.max() if len(up) else 0, -dn.attribution.min() if len(dn) else 0, 1e-6)
    for ax, rows, cw, cg, sign, label in ((ax_up, up, RAISE, RAISE_LIGHT, 1, "Raises damage"),
                                          (ax_dn, dn, LOWER, LOWER_LIGHT, -1, "Lowers damage")):
        y = np.arange(len(rows))[::-1]
        mag = sign * rows.attribution.to_numpy()
        col = [cw if k == "weather" else (cg if k == "geography" else "#d9d7d2") for k in rows.kind]
        ax.barh(y, mag, height=0.66, color=col, edgecolor="white", linewidth=0.6, zorder=2)
        for yi, m in zip(y, mag):
            ax.text(m + 0.03 * xmax, yi, f"{sign * m:+.2f}".replace("-", "−"), va="center", ha="left", fontsize=fs - 1.0)
        ax.set_yticks(y); ax.set_yticklabels(rows.group, fontsize=fs - 0.8)
        ax.tick_params(axis="y", length=0, pad=2); ax.spines["left"].set_visible(False)
        ax.set_xlim(0, xmax); ax.set_ylim(-0.6, max(len(rows), 1) - 0.4)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.set_title(label, loc="left", fontsize=fs - 0.4, color=cw, fontweight="bold", pad=1.5)
    ax_up.tick_params(labelbottom=False)
    ax_dn.set_xlabel("Contribution to the damage logit, push window", labelpad=1.5)
    ax_dn.legend(handles=[Patch(facecolor="#6f6e6a", label="weather"), Patch(facecolor="#c9c7c2", label="geography")],
                 loc="lower right", fontsize=fs - 1.1, handlelength=1.1, handleheight=0.8, borderaxespad=0.1)
    f = lambda x: f"{x:+.2f}".replace("-", "−")
    return (f"net push {f(push)} = weather {f(by_kind.get('weather', 0.0))}, "
            f"geography {f(by_kind.get('geography', 0.0))}")


def panel_c(ax_s, ax, z, fs):
    t = np.arange(72, 216)
    push = z["C"].astype(float).sum(1)
    ax_s.fill_between(t, 0, push, where=push >= 0, color=RAISE, lw=0, interpolate=True)
    ax_s.fill_between(t, 0, push, where=push < 0, color=LOWER, lw=0, interpolate=True)
    ax_s.axhline(0, color=TEXT2, lw=0.4)
    lim = max(1e-6, np.abs(push).max() * 1.1)
    ax_s.set_xlim(PREFIX_END, 215); ax_s.set_ylim(-lim, lim)
    ax_s.tick_params(labelbottom=False, labelsize=fs - 1.3); ax_s.spines["bottom"].set_visible(False)
    ax_s.tick_params(axis="x", length=0); ax_s.yaxis.set_major_locator(MaxNLocator(3, symmetric=True))
    ax_s.set_ylabel("net push\n(logit)", fontsize=fs - 1.0, labelpad=2)
    y = np.where(z["obs_full"], z["y_full"], np.nan)
    start = float(y[PREFIX_END])
    x = np.r_[PREFIX_END, t]
    ser = lambda k: 100 * np.r_[start, z[k]]
    off, on = ser("P_closed"), ser("P_open")
    pw = z["push_window"]
    ax.axvspan(pw[0], pw[1], color="#f7e6e1", lw=0, zorder=0)
    ax.fill_between(x, off, on, where=on >= off, color=RAISE_FILL, lw=0, interpolate=True, zorder=1)
    ax.fill_between(x, off, on, where=on < off, color=LOWER_FILL, lw=0, interpolate=True, zorder=1)
    ax.plot(np.arange(PREFIX_END - 11, 216), 100 * y[PREFIX_END - 11:], color=INK, lw=1.0, zorder=3)
    ax.plot(x, ser("P_W"), color=HOST, lw=1.1, ls=HOST_DASH, zorder=4)
    ax.plot(x, off, color=OFF, lw=1.1, ls=OFF_DASH, zorder=4)
    ax.plot(x, on, color=GCRK, lw=1.3, zorder=5)
    top = np.nanmax(np.r_[100 * y[PREFIX_END:], on, off, ser("P_W")])
    ax.set_xlim(PREFIX_END, 215); ax.set_ylim(0, top * 1.35 if top > 0 else 1)
    ax.set_xticks([72, 120, 168, 215]); ax.set_xlabel("Forecast hour $t$", labelpad=1.5)
    ax.set_ylabel("Customers out (%)")
    ax.legend(handles=[Line2D([], [], color=INK, lw=1.0, label="Observed"),
                       Line2D([], [], color=HOST, lw=1.2, ls=HOST_DASH, label="Weather host W"),
                       Line2D([], [], color=OFF, lw=1.2, ls=OFF_DASH, label="GCRK, kernel off"),
                       Line2D([], [], color=GCRK, lw=1.4, label="AsymODE + GCRK"),
                       Patch(facecolor=RAISE_FILL, label="effect of the push")],
              loc="upper left", ncol=2, fontsize=fs - 1.3, handlelength=1.6, labelspacing=0.25, columnspacing=0.8,
              borderaxespad=0.1)


def panel_d(ax, fs):
    sd = pd.read_csv(C.RESULTS / "seeds_main.csv")
    keys = ["rmse 1-6 h", "rmse 7-24 h", "rmse 25-48 h", "rmse 49-144 h", "rmse"]
    labels = ["1–6 h", "7–24 h", "25–48 h", "49–144 h", "all"]
    w = sd[sd.model == "W"].set_index("seed"); g = sd[sd.model == "GCRK"].set_index("seed")
    seeds = sorted(set(w.index) & set(g.index))
    rel = np.array([[100 * (g.loc[s, k] / w.loc[s, k] - 1) for k in keys] for s in seeds])
    x = np.arange(len(keys))
    m = rel.mean(0)
    ax.bar(x, m, 0.6, color=[GCRK if v < 0 else "#9fc0ea" for v in m], zorder=2)
    for j in range(len(keys)):
        ax.scatter(np.full(len(seeds), x[j]) + np.linspace(-0.18, 0.18, len(seeds)), rel[:, j], s=6, color=INK, zorder=3,
                   linewidths=0)
    ax.axhline(0, color=HOST, lw=1.1, ls=HOST_DASH, zorder=1)
    for xi, v, r in zip(x, m, rel.T):
        ax.text(xi, max(v, r.max()) + 0.08 * max(1e-6, np.abs(rel).max()), f"{v:+.1f}%\n{int((r < 0).sum())}/{len(r)}".replace("-", "−"),
                ha="center", va="bottom", fontsize=fs - 1.3, color=INK, linespacing=1.0)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("Lead time (held-out counties, all events)", labelpad=1.5)
    ax.set_ylabel("RMSE change vs W (%)")
    lim = 1.45 * max(1.0, np.abs(rel).max())
    ax.set_ylim(-lim, lim)
    ax.text(len(keys) - 0.4, lim * 0.97, "worse than W", ha="right", va="top", fontsize=fs - 1.1, color=TEXT2)
    ax.text(len(keys) - 0.4, -lim * 0.97, "better than W", ha="right", va="bottom", fontsize=fs - 1.1, color=TEXT2)


def fig3():
    fs = 7.0; apply_style(fs)
    z = np.load(FIGDATA / "fig3_kernel.npz")
    drv = pd.read_csv(FIGDATA / "fig3_drivers.csv")
    name = county_names().get(str(z["fips"]), str(z["fips"]))
    fig = plt.figure(figsize=(TEXT_WIDTH, 5.0))
    top = fig.add_gridspec(1, 3, left=0.07, right=0.99, top=0.9, bottom=0.58, wspace=0.0, width_ratios=[1.0, 0.62, 0.78])
    bot = fig.add_gridspec(1, 3, left=0.07, right=0.99, top=0.44, bottom=0.08, wspace=0.0, width_ratios=[1.0, 0.26, 0.9])
    sa = top[0, 0].subgridspec(2, 2, height_ratios=[0.24, 1.0], width_ratios=[1.0, 0.045], hspace=0.07, wspace=0.06)
    ax_in = fig.add_subplot(sa[0, 0]); ax_a = fig.add_subplot(sa[1, 0], sharex=ax_in)
    panel_a(ax_a, ax_in, fig.add_subplot(sa[1, 1]), z, fs); fig.add_subplot(sa[0, 1]).set_axis_off()
    title(ax_in, "a", f"Kernel at work: {name}", fs)
    sb = top[0, 2].subgridspec(2, 1, height_ratios=[7, 6], hspace=0.45)
    ax_up, ax_dn = fig.add_subplot(sb[0]), fig.add_subplot(sb[1])
    head = panel_b(ax_up, ax_dn, drv, fs)
    ax_up.text(-0.95, 1.33, "What drives the push-window push", transform=ax_up.transAxes, fontsize=fs + 0.3, color=INK)
    ax_up.text(-0.98, 1.33, "b", transform=ax_up.transAxes, ha="right", fontsize=fs + 1.5, fontweight="bold")
    ax_up.text(-0.95, 1.16, head, transform=ax_up.transAxes, fontsize=fs - 1.1, color=TEXT2)
    sc = bot[0, 0].subgridspec(2, 1, height_ratios=[0.27, 1.0], hspace=0.06)
    ax_s = fig.add_subplot(sc[0]); ax_c = fig.add_subplot(sc[1], sharex=ax_s)
    panel_c(ax_s, ax_c, z, fs)
    title(ax_s, "c", f"Forecast, {name} (event {z['event']})", fs, y=1.1)
    ax_d = fig.add_subplot(bot[0, 2]); panel_d(ax_d, fs)
    title(ax_d, "d", "Held-out error against the weather host W", fs, y=1.03)
    save(fig, "fig3_gcrk_interpretation_open")


def fig4_counties(F) -> list[int]:
    """PREREG 11: the Figure 3 county; per event the >= 10,000-customer county with the
    largest observed forecast-window peak; then further two-wave counties of the
    representative event by push-window rise, until eight."""
    choice = json.loads((C.RESULTS / "figure_choice.json").read_text())
    ranked = [c["unit"] for c in choice["candidates"]]
    chosen = [ranked[0]]
    yf = np.where(F["obs_full"], F["y_full"], np.nan)
    for ev in sorted(set(F["event"].tolist())):
        u = [i for i in np.where(F["event"] == ev)[0] if F["cust"][i] >= 10000 and i not in chosen]
        chosen.append(max(u, key=lambda i: np.nanmax(yf[i, 72:])))
    for u in ranked[1:]:
        if len(chosen) >= 8:
            break
        if u not in chosen:
            chosen.append(u)
    return chosen[:8]


def fig4(seed=0):
    apply_style(7.0)
    F = C.load_features(); n = len(F["y"])
    G = C.collect("main", "GCRK", seed, n); T = C.timesfm(F, "WEATHER")
    units = fig4_counties(F)
    (C.RESULTS / "figure4_choice.json").write_text(json.dumps([dict(unit=int(u), fips=str(F["fips"][u]), event=str(F["event"][u]))
                                                               for u in units], indent=1) + "\n")
    names = county_names()
    fig, axes = plt.subplots(2, 4, figsize=(TEXT_WIDTH, 3.1), sharex=True)
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.12, top=0.84, wspace=0.28, hspace=0.42)
    h = np.arange(216)
    for k, (ax, u) in enumerate(zip(axes.ravel(), units)):
        obs = 100 * np.where(F["obs_full"][u], F["y_full"][u], np.nan)
        g = 100 * np.r_[np.full(71, np.nan), F["y0"][u], G[u]]
        tf = 100 * np.r_[np.full(71, np.nan), F["y0"][u], T[u]] if T is not None else None
        top = np.nanmax(np.r_[obs, g] if tf is None else np.r_[obs, g, tf])
        ymax = top * 1.12 if top > 0 else 1.0
        ax.axvspan(0, PREFIX_END, color=WASH, lw=0, zorder=0); ax.axvline(PREFIX_END, color=MUTED, lw=0.5, zorder=1)
        ax.plot(h, obs, color=INK, lw=1.0, zorder=3)
        if tf is not None:
            ax.plot(h, tf, color=TIMESFM, lw=1.0, zorder=4)
        ax.plot(h, g, color=GCRK, lw=1.25, zorder=5)
        ax.set_xlim(0, 215); ax.set_ylim(0, ymax)
        ax.yaxis.set_major_locator(MaxNLocator(3, min_n_ticks=3)); ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xticks([0, 48, 96, 144, 192])
        ax.set_title(f"{names.get(str(F['fips'][u]), F['fips'][u])}", loc="left", fontsize=7.2, fontweight="bold", pad=9)
        ax.text(0.0, 1.015, f"event {F['event'][u]}", transform=ax.transAxes, fontsize=5.8, color=TEXT2, va="bottom")
        if k % 4 == 0:
            ax.set_ylabel("Customers out (%)", fontsize=6.8)
        if k >= 4:
            ax.set_xlabel("Hour", fontsize=6.8, labelpad=1.5)
    handles = [Line2D([], [], color=INK, lw=1.2, label="Observed"), Line2D([], [], color=GCRK, lw=1.4, label="AsymODE + GCRK"),
               Line2D([], [], color=TIMESFM, lw=1.2, label="TimesFM (zero-shot)"),
               Patch(facecolor=WASH, edgecolor="none", label="outages observed, hours 0–71")]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0), fontsize=7.0, handlelength=2.0,
               columnspacing=1.6, frameon=False)
    save(fig, "fig4_county_trajectories_open")


def fig1():
    """Representative event, observed only: county peak outage share in wave A (hours
    72-95), wave B (hours 168-215) and over the whole forecast window."""
    import geopandas as gpd
    from matplotlib.colors import LogNorm
    apply_style(7.0)
    F = C.load_features()
    choice = json.loads((C.RESULTS / "figure_choice.json").read_text())
    ev = choice["event"]
    u = np.where(F["event"] == ev)[0]
    y = 100 * np.where(F["obs_full"][u], F["y_full"][u], np.nan)
    peaks = {"Wave A (hours 72-95)": np.nanmax(y[:, 72:96], 1), "Wave B (hours 168-215)": np.nanmax(y[:, 168:216], 1),
             "Whole forecast window (hours 72-215)": np.nanmax(y[:, 72:216], 1)}
    shp = gpd.read_file(C.ROOT / "data/raw/census/cb_county/cb_2023_us_county_500k.shp")
    shp["fips"] = shp.STATEFP + shp.COUNTYFP
    shp = shp[shp.STATEFP.isin({f[:2] for f in F["fips"][u]})].to_crs(5070)
    states = shp.dissolve("STATEFP")
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.35))
    fig.subplots_adjust(left=0.01, right=0.93, top=0.9, bottom=0.02, wspace=0.03)
    norm = LogNorm(0.1, 50)
    cmap = plt.get_cmap("YlOrRd")
    for ax, (lab, v) in zip(axes, peaks.items()):
        d = shp.merge(pd.DataFrame(dict(fips=F["fips"][u], v=v)), on="fips", how="left")
        states.boundary.plot(ax=ax, color="#b0aea8", lw=0.3, zorder=1)
        d[d.v.isna()].plot(ax=ax, color="#f3f1ec", lw=0, zorder=0)
        dd = d[d.v.notna()].copy(); dd["v"] = dd.v.clip(0.1, 50)
        dd.plot(ax=ax, column="v", cmap=cmap, norm=norm, lw=0.05, edgecolor="white", zorder=2)
        ax.set_axis_off(); ax.set_title(lab, fontsize=7.2, loc="left", pad=2)
    cax = fig.add_axes([0.94, 0.15, 0.012, 0.65])
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, extend="both")
    cb.set_label("peak customers out (%)", fontsize=6.5); cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6)
    save(fig, "fig1_event_county_impacts_open")


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["fig1", "fig3", "fig4"]
    for w in which:
        {"fig1": fig1, "fig3": fig3, "fig4": fig4}[w]()
