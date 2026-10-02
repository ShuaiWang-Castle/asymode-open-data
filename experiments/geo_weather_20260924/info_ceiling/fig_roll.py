"""Figure for the restoration-kernel screen: customer-weighted outage fraction of the county-events already out at the
later origin (stock >= 1%), by weather system, observed against the host retrained on the shifted panel and the
restoration kernel (held-out folds only). Writes figures/roll_trajectories_<shift>.{png,pdf}.
usage: python fig_roll.py [shift]"""
from __future__ import annotations

import sys

import numpy as np

from probe import EXP, FEAT, load_static
from score_dose import collect

sys.path.insert(0, str(EXP))


def main(shift: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import screen
    F = screen.shift_origin(screen.load("v1D"), shift)
    s = load_static()
    y, m, y0 = F["y"].astype(float), F["m"].astype(float), F["y0"].astype(float)
    cust = np.load(FEAT, allow_pickle=False)["cust"].astype(float)
    sysv, reg = s["system"].astype(str), s["regime"].astype(str)
    n = len(y); H = 144 - shift
    Ph, Pk = collect(f"v1r{shift}_host_s0", n)[0], collect(f"v1r{shift}_rk_s0", n)[0]
    av = ~np.isnan(Ph).any(1) & ~np.isnan(Pk).any(1) & (m[:, :H].min(1) > 0) & (y0 >= .01)
    top = sorted(np.unique(sysv[av]), key=lambda q: -(cust * y0)[av & (sysv == q)].sum())[:6]
    fig, ax = plt.subplots(2, 3, figsize=(11, 5.6), sharex=True)
    for a, q in zip(ax.ravel(), top):
        f = av & (sysv == q); c = cust[f] / cust[f].sum()
        t = np.arange(1, H + 1)
        a.plot(t, c @ y[f][:, :H], color="k", lw=2, label="observed")
        a.plot(t, c @ Ph[f][:, :H], color="#2c7fb8", lw=1.6, label="host (retrained)")
        a.plot(t, c @ Pk[f][:, :H], color="#d7191c", lw=1.6, label="restoration kernel")
        a.set_title(f"{q} ({reg[f][0].replace('_', ' ')}), {int(f.sum())} counties already out", fontsize=9)
        a.spines[["top", "right"]].set_visible(False)
    ax[0, 0].legend(fontsize=8, frameon=False)
    for a in ax[1]:
        a.set_xlabel(f"hours after the origin (registered origin + {shift} h)")
    for a in ax[:, 0]:
        a.set_ylabel("customers out (fraction)")
    fig.suptitle("Held-out weather systems: counties already out at the origin, customer-weighted", fontsize=10.5)
    fig.tight_layout()
    out = EXP / "figures"; out.mkdir(exist_ok=True)
    fig.savefig(out / f"roll_trajectories_{shift}.png", dpi=170); fig.savefig(out / f"roll_trajectories_{shift}.pdf")
    print("written", f"figures/roll_trajectories_{shift}.png", "systems", top)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 48)
