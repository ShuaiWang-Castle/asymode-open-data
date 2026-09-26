"""Descriptive table of the hazard dictionary v2 by regime on the development tranche (weather only, no outcome):
per county-event, the forecast-window maximum of selected county series; the table gives the share of county-events
above a physical level and the median. usage: python describe_hazard_v2.py --source era5"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from frame_common import OUT, PREFIX_H

ROWS = [("g_exc_max", 0.0, "gust above local p98 (any node)"), ("frz_mean", 0.5, "freezing rain > 0.5 mm/h (county mean)"),
        ("ice_max", 6.0, "ice load > 6 mm (any node)"), ("wetsnow_max", 10.0, "wet-snow load > 10 mm (any node)"),
        ("snow_mean", 1.0, "snowfall > 1 mm/h"), ("rain_mean", 10.0, "rain > 10 mm/h"), ("ice_wind_max", 1.0, "ice x wind > 1"),
        ("snow_wind_max", 1.0, "wet snow x wind > 1"), ("uh_max", 75.0, "updraft helicity > 75 m2/s2 (HRRR)"),
        ("shear_mean", 20.0, "0-6 km shear > 20 m/s (HRRR)"), ("ltng_mean", 0.1, "lightning (HRRR)"),
        ("warm_nose_mean", 0.1, "warm layer aloft over a sub-zero surface (HRRR)"), ("canopy*g_exc", 0.0, "canopy x exceedance > 0")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="era5")
    a = ap.parse_args()
    dr = pd.read_parquet(OUT / "draws.parquet").set_index("system")
    rows = []
    for s in dr[dr.tranche == "D"].index:
        f = OUT / "hazard_v2" / f"{a.source}_{s}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        X = z["X"][:, PREFIX_H:, :].astype(np.float32).max(1)
        d = pd.DataFrame(X, columns=z["names"].astype(str)); d["regime"] = dr.at[s, "regime"]
        rows.append(d)
    d = pd.concat(rows, ignore_index=True)
    out = {}
    for c, thr, lab in ROWS:
        if c in d:
            out[lab] = d.groupby("regime")[c].apply(lambda x: f"{(x > thr).mean():.0%}")
    print(pd.DataFrame(out).T.to_string())


if __name__ == "__main__":
    main()
