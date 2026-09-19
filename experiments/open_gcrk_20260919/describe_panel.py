"""Descriptive statistics of the public panel (observed data only; no model output).

Replacements for the paper's data-section counts, computed on the 2660 county-events:
  * per event: counties, states, customers, share with a forecast-window peak >= 1% and
    >= 5%, median and maximum peak;
  * renewed growth: a two-hour increase of p by >= 2 percentage points inside the
    forecast window, and how many of those units had already reached p >= 2% earlier
    in the window (prefix included);
  * onset from a fully served state: units with p_71 = 0 exactly, and how many reach
    p >= 2% in the forecast window;
  * for the representative event, the two-wave counties' observed peaks in waves A and B.
Writes results/panel_description.csv and results/panel_description.json.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import common as C


def main():
    F = C.load_features()
    y = np.where(F["obs_full"], F["y_full"], np.nan).astype(np.float64)
    fc = y[:, 72:]
    rows = []
    for ev in sorted(set(F["event"].tolist())):
        u = F["event"] == ev
        pk = np.nanmax(fc[u], 1)
        rows.append(dict(event=ev, county_events=int(u.sum()), states=len({f[:2] for f in F["fips"][u]}),
                         customers_millions=float(F["cust"][u].sum() / 1e6),
                         share_peak_ge_1pct=float((pk >= 0.01).mean()), share_peak_ge_5pct=float((pk >= 0.05).mean()),
                         median_peak_pct=float(100 * np.median(pk)), max_peak_pct=float(100 * pk.max()),
                         share_p71_zero=float((F["y0"][u] == 0).mean())))
    d = pd.DataFrame(rows)
    rise2 = np.nanmax(fc[:, 2:] - fc[:, :-2], 1) >= 0.02
    t_rise = np.array([int(np.argmax((fc[i, 2:] - fc[i, :-2]) >= 0.02)) + 2 if rise2[i] else -1 for i in range(len(fc))])
    earlier = np.array([np.nanmax(y[i, :72 + t_rise[i] - 2]) >= 0.02 if t_rise[i] >= 0 else False for i in range(len(fc))])
    zero = F["y0"] == 0
    reach = np.nanmax(fc, 1) >= 0.02
    summ = dict(county_events=int(len(fc)), counties=int(len(set(F["fips"].tolist()))),
                states=int(len({f[:2] for f in F["fips"]})),
                renewed_growth_2pp_2h=int(rise2.sum()), renewed_growth_after_earlier_2pct=int((rise2 & earlier).sum()),
                p71_zero=int(zero.sum()), p71_zero_reach_2pct=int((zero & reach).sum()),
                observed_cell_share=float(F["obs_full"].mean()))
    d.to_csv(C.RESULTS / "panel_description.csv", index=False)
    (C.RESULTS / "panel_description.json").write_text(json.dumps(summ, indent=1) + "\n")
    with pd.option_context("display.width", 200):
        print(d.to_string()); print(summ)


if __name__ == "__main__":
    C.RESULTS.mkdir(exist_ok=True)
    main()
