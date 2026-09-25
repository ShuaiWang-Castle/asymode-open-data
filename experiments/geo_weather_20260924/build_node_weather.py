"""Node weather (data v3): the ERA5 cell weather that reaches each sub-grid node, for every county-event of the
twelve-event panel, in the unit order of features_e3r2.npz.

Channels per node and hour (from the ERA5 files of open_gcrk_20260919: main fields + hourly-maximum gust):
  t2m_c, d2m_c (dew point from t2m and rh, Magnus), rh, precip (mm/h), snowfall (mm w.e./h),
  gust (hourly max, m/s), wind_speed (10 m, m/s), soil_moisture (layer 1), cape
node weather = sum_c pi[k, c] w_c(t) with the population mixing weights of build_subgrid_nodes.py (no
downscaling here - the lapse-rate move by dz is a learnable part of the model). Also per unit: the
population-weighted county weather (exposure-weighted) for the same channels.
Output: data/interim/geo_weather/node_weather_e3.npz (float16), with node attributes and weights per unit.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "open_gcrk_20260919"))
import build_panel216_r2 as P2  # noqa: E402

OUT = ROOT / "data" / "interim" / "geo_weather"
CH = ["t2m_c", "d2m_c", "rh", "precip", "snowfall", "gust", "wind_speed", "soil_moisture", "cape"]


def main():
    F = np.load(ROOT / "data" / "interim" / "open_gcrk" / "features_e3r2.npz")
    fips_u, ev_u = F["fips"].astype(str), F["event"].astype(str)
    N = np.load(OUT / "nodes_e3.npz")
    pos = {f: i for i, f in enumerate(N["fips"].astype(str))}
    missing = sorted(set(fips_u) - set(pos))
    assert not missing, f"counties without nodes: {missing[:5]}"
    sel = {e["event"]: e for e in json.loads((ROOT / "experiments/open_gcrk_20260919/selected_events_e3.json").read_text())["events"]}
    U, K, T = len(fips_u), N["rho"].shape[1], 144 + 72
    node_w = np.zeros((U, K, T, len(CH)), np.float16)
    pop_w = np.zeros((U, T, len(CH)), np.float32)
    for ev in sorted(set(ev_u)):
        t0 = pd.Timestamp(sel[ev]["window_start_utc"])
        times, ch, names = P2.fields_r2(ev, t0)
        assert (pd.DatetimeIndex(times) == pd.date_range(t0, periods=T, freq="h")).all(), ev
        a, b = 17.625, 243.04                                                    # Magnus, as in asymode.weather
        g = np.log(np.clip(ch["rh"], 1e-3, None) / 100.0) + a * ch["t2m_c"] / (b + ch["t2m_c"])
        ch["d2m_c"] = b * g / (a - g)
        ch["soil_moisture"] = np.nan_to_num(ch["soil_moisture"], nan=0.0)
        arr = np.stack([np.asarray(ch[c], np.float32) for c in CH], -1)          # (T, nlat, nlon, C)
        units = np.where(ev_u == ev)[0]
        for u in units:
            n = pos[fips_u[u]]
            cells = N["cells"][n]; ok = cells[:, 0] >= 0
            cs = arr[:, cells[ok, 0], cells[ok, 1], :]                            # (T, c, C)
            mix = N["mix"][n][:, ok]                                              # (K, c)
            node_w[u] = np.einsum("kc,tcx->ktx", mix, cs).astype(np.float16)
            pop_w[u] = np.einsum("c,tcx->tx", N["pop_w"][n][ok], cs)
        print(ev, len(units), "units", flush=True)
    idx = np.array([pos[f] for f in fips_u])
    np.savez_compressed(OUT / "node_weather_e3.npz", node_weather=node_w, pop_weather=pop_w, channels=np.array(CH),
                        node_attr=N["attr"][idx], node_rho=N["rho"][idx], attr_names=N["attr_names"],
                        fips=fips_u, event=ev_u)
    print("saved", node_w.shape)


if __name__ == "__main__":
    main()
