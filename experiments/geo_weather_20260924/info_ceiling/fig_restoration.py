"""Figure for the restoration-capacity probes: the share of the peak outage that remains in the hours after the peak,
by the outage fraction of the other counties within 150 km at the peak hour (severe county-events, design-weighted
means), with the host's median restoration for reference. Writes figures/restoration_by_neighbour_burden.{png,pdf}."""
from __future__ import annotations

import numpy as np

from probe import EXP, FEAT, load_static
from restore_capacity import region_table
from score_dose import collect


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = load_static(); z = np.load(FEAT, allow_pickle=False)
    y, m, w, peak = s["y"].astype(float), s["m"].astype(float), s["w"].astype(float), s["peak"].astype(float)
    cust = z["cust"].astype(float); n = len(y); a = np.arange(n)
    yo = np.where(m > 0, y, -1.0); tpk = yo.argmax(1)
    r = region_table(); rf = (r["out_ex"] / np.maximum(r["base_in"] - cust, 1.0)[:, None])[a, tpk]
    H = 48
    sel = (peak >= .10) & (tpk + H <= 143)
    host_r = collect("v1_host_s0", n)[2]
    classes = [(0, .005, "neighbours < 0.5% out"), (.005, .02, "0.5-2%"), (.02, .05, "2-5%"), (.05, 1.01, ">= 5%")]
    colors = ["#2c7fb8", "#7fcdbb", "#fdae61", "#d7191c"]
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)
    for k, (lo, hi, lab) in enumerate(classes):
        for j, (plo, phi, title) in enumerate([(.10, .25, "own peak 10-25% of customers"), (.25, 1.01, "own peak >= 25% of customers")]):
            f = np.where(sel & (rf >= lo) & (rf < hi) & (peak >= plo) & (peak < phi))[0]
            if len(f) < 8:
                continue
            curve = np.array([np.average(np.where(m[f, tpk[f] + h] > 0, y[f, tpk[f] + h] / peak[f], np.nan)[m[f, tpk[f] + h] > 0], weights=w[f][m[f, tpk[f] + h] > 0]) for h in range(H + 1)])
            ax[j].plot(range(H + 1), curve, color=colors[k], lw=2, label=f"{lab} (n = {len(f)})")
            ax[j].set_title(title, fontsize=10)
    hr = float(np.median(host_r[sel, tpk[sel]]))
    for j in range(2):
        ax[j].plot(range(H + 1), (1 - hr) ** np.arange(H + 1), color="k", ls="--", lw=1.2, label=f"host's restoration rate ({hr:.2f} per hour)")
        ax[j].set_xlabel("hours after the county's outage peak"); ax[j].legend(fontsize=8, frameon=False); ax[j].set_ylim(0, 1.02); ax[j].set_xlim(0, H)
        ax[j].spines[["top", "right"]].set_visible(False)
    ax[0].set_ylabel("share of the peak outage still out")
    fig.suptitle("Restoration after the peak, by the outage state of the other counties within 150 km at the peak hour", fontsize=10.5)
    fig.tight_layout()
    out = EXP / "figures"; out.mkdir(exist_ok=True)
    fig.savefig(out / "restoration_by_neighbour_burden.png", dpi=170); fig.savefig(out / "restoration_by_neighbour_burden.pdf")
    print("written", out / "restoration_by_neighbour_burden.png")


if __name__ == "__main__":
    main()
