"""Weather twins in the development tranche (Section 2 of the paper): neighbouring counties of the same weather system
that receive nearly the same ERA5 weather over the forecast window yet reach very different outage peaks.

Rule, fixed before looking at the outcomes of any pair:
  * the two counties share a border or a water boundary (Census 2023 county adjacency) and belong to the same system;
  * both serve at least 5,000 customers and are observed at >= 90% of hours 72-215;
  * hourly ERA5 gust and precipitation correlate at >= 0.95 and >= 0.90 over hours 72-215, the maximum gusts differ by
    at most 10% and the precipitation totals by at most 25%;
  * the pairs are ranked by the difference of their 3-hour peaks (the largest 3-hour mean outage fraction over
    hours 72-215, so a single-hour spike does not count).
Writes results/v1/weather_twin_pairs.json (the number of qualifying pairs and the top 50, ranked, with named fields)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
FEAT = ROOT / "data" / "interim" / "panel_v1" / "features_v1D.npz"
ADJ = ROOT / "data" / "raw" / "census" / "county_adjacency2023.txt"
TOP = 50


def peak3(y: np.ndarray) -> float:
    v = pd.Series(y[72:216]).rolling(3, min_periods=2).mean().to_numpy()
    return float(np.nanmax(v)) if np.isfinite(v).any() else float("nan")


def main() -> None:
    z = np.load(FEAT)
    fips, sysv, reg = z["fips"].astype(str), z["system"].astype(str), z["regime"].astype(str)
    cust, obs = z["cust"].astype(float), z["obs_full"].astype(bool)
    y = np.where(obs, z["y_full"], np.nan).astype(float)
    names = list(z["weather_channels"].astype(str))
    xu = z["xu"][:, 72:216, :len(names)]
    gi, pi = names.index("gust"), names.index("precip")
    adj = pd.read_csv(ADJ, sep="|", dtype=str)
    nb = {(a, b) for a, b in zip(adj["County GEOID"], adj["Neighbor GEOID"]) if a != b}
    gust, prec = xu[..., gi], xu[..., pi]            # the leading x^U channels are ERA5 in physical units
    rows = []
    for S in np.unique(sysv):
        ii = [i for i in np.where(sysv == S)[0] if cust[i] >= 5000 and obs[i, 72:216].mean() >= 0.9]
        for a in range(len(ii)):
            for b in range(a + 1, len(ii)):
                i, j = ii[a], ii[b]
                if (fips[i], fips[j]) not in nb:
                    continue
                cg = np.corrcoef(gust[i], gust[j])[0, 1]; cp = np.corrcoef(prec[i], prec[j])[0, 1]
                gm = (gust[i].max(), gust[j].max()); pt = (prec[i].sum(), prec[j].sum())
                if not (cg >= 0.95 and cp >= 0.90 and abs(gm[0] - gm[1]) <= 0.1 * max(gm)
                        and abs(pt[0] - pt[1]) <= 0.25 * max(max(pt), 1e-9)):
                    continue
                p = (peak3(y[i]), peak3(y[j]))
                hi, lo = (i, j) if p[0] >= p[1] else (j, i)
                rows.append(dict(system=S, regime=reg[i], fips_high=fips[hi], fips_low=fips[lo],
                                 peak3_high=max(p), peak3_low=min(p), difference=abs(p[0] - p[1]),
                                 gust_corr=float(cg), precip_corr=float(cp),
                                 gust_max_high=float(gust[hi].max()), gust_max_low=float(gust[lo].max()),
                                 precip_total_high=float(prec[hi].sum()), precip_total_low=float(prec[lo].sum()),
                                 customers_high=float(cust[hi]), customers_low=float(cust[lo])))
    rows.sort(key=lambda r: -r["difference"])
    out = dict(rule=__doc__.split("Rule, fixed before looking at the outcomes of any pair:\n")[1].split("Writes")[0].strip(),
               units="peaks as fractions; gust m/s and precipitation mm over hours 72-215", n_pairs=len(rows),
               pairs=rows[:TOP])
    (EXP / "results" / "v1" / "weather_twin_pairs.json").write_text(json.dumps(out, indent=1) + "\n")
    for r in rows[:8]:
        print(r["regime"], r["system"], r["fips_high"], r["fips_low"], f"{r['peak3_high']:.3f} {r['peak3_low']:.3f}",
              f"g {r['gust_max_high']:.1f}/{r['gust_max_low']:.1f} p {r['precip_total_high']:.0f}/{r['precip_total_low']:.0f}",
              f"r {r['gust_corr']:.2f}/{r['precip_corr']:.2f}")
    print(len(rows), "pairs")


if __name__ == "__main__":
    main()
