"""Round-2 model inputs (PREREG Amendment 2, R2 (a)-(d)) from the panel216r2_<event>.npz files.

Differences from round 1 (`build_features.py`), for W and GCRK alike:
  gust          the hour's maximum gust (county area mean) wherever gust is used
  hazards       gust exceedance energy, wet wind, near-freeze, snow-ice load and cold precipitation
                are computed per ERA5 cell and then area-averaged (Hg); their 6 h / 12 h sums follow
  path          trailing 72-hour summaries, defined identically in the prefix and the forecast window:
                gust_max72, gust_excess_energy_sum72, wet_wind_sum72, snow_ice_load_sum72,
                hours_since_gust_max72 (hours since the latest maximum in the trailing 72 h)
  support       two more damage inputs: gust_cell_max, gust_exceed15_share
  geography     40 descriptors (Amendment 1)
Everything else (past windows, freeze counts, direction, statics, prefix-outage summaries,
neighbour weather) is computed as in round 1 from the round-2 channels.

Writes data/interim/open_gcrk/features_<tag>.npz and data_provenance/features_<tag>_checksum.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import pandas as pd

import build_features as BF

PATH72 = ["gust_max72", "gust_excess_energy_sum72", "wet_wind_sum72", "snow_ice_load_sum72", "hours_since_gust_max72"]
SUPPORT = ["gust_cell_max", "gust_exceed15_share"]
DAMAGE_R2 = BF.CH + BF.HAZARD + PATH72 + BF.PAST + BF.FREEZE + BF.DIRECTION + SUPPORT
WINDOW = 72


def since_trailing_max(g: np.ndarray, n: int = WINDOW) -> np.ndarray:
    out = np.zeros_like(g)
    for t in range(g.shape[1]):
        w = g[:, max(0, t - n + 1):t + 1]
        last = w.shape[1] - 1 - np.argmax(w[:, ::-1], axis=1)      # latest position of the maximum
        out[:, t] = w.shape[1] - 1 - last
    return out


def weather_features_r2(X: np.ndarray, Hg: np.ndarray, S: np.ndarray) -> dict:
    f = BF.weather_features(X)                       # past windows, freeze, direction from round-2 channels
    for k, name in enumerate(["gust_excess_energy", "wet_wind", "near_freeze", "snow_ice_load", "cold_precip"]):
        f[name] = Hg[..., k].astype(np.float64)
    f["gust_excess_energy_sum6"] = BF.rolling(f["gust_excess_energy"], 6, "sum")
    f["wet_wind_sum12"] = BF.rolling(f["wet_wind"], 12, "sum")
    gust = X[..., BF.CH.index("gust")].astype(np.float64)
    f["gust_max72"] = BF.rolling(gust, WINDOW, "max")
    f["gust_excess_energy_sum72"] = BF.rolling(f["gust_excess_energy"], WINDOW, "sum")
    f["wet_wind_sum72"] = BF.rolling(f["wet_wind"], WINDOW, "sum")
    f["snow_ice_load_sum72"] = BF.rolling(f["snow_ice_load"], WINDOW, "sum")
    f["hours_since_gust_max72"] = since_trailing_max(gust)
    f["gust_cell_max"] = S[..., 0].astype(np.float64)
    f["gust_exceed15_share"] = S[..., 1].astype(np.float64)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="r2")
    ap.add_argument("--events-file", default="selected_events.json")
    ap.add_argument("--geo-base", default="geography.parquet")
    ap.add_argument("--geo-ext", default="geography_ext.parquet")
    a = ap.parse_args()
    events = json.loads((BF.HERE / a.events_file).read_text())["events"]
    geo = pd.read_parquet(BF.OUT / a.geo_base)
    geo_cols = [c for c in geo.columns if c != "n_land_pixels"]
    geo = geo.join(pd.read_parquet(BF.OUT / a.geo_ext)[BF.GEO_EXT], how="left")
    geo_cols = geo_cols + BF.GEO_EXT
    parts = {k: [] for k in ("xu", "xr", "xo", "geo", "y0", "y", "m", "y_full", "obs_full", "cust", "fips", "event")}
    for e in events:
        z = np.load(BF.OUT / f"panel216r2_{e['event']}.npz")
        fips = [str(f) for f in z["fips"]]
        X, Xn = z["X"].astype(np.float64), z["X_nbr"].astype(np.float64)
        assert list(z["channels"]) == BF.CH
        wf = weather_features_r2(X, z["Hg"], z["S"])
        xu = np.concatenate([X, np.stack([wf[k] for k in DAMAGE_R2[len(BF.CH):]], -1)], -1)
        pos = {str(g): i for i, g in enumerate(z["nbr_fips"])}
        nb = np.stack([[pos[str(g)] for g in row] for row in z["nbr_of"]])
        nbr = []
        for c in ("gust", "wind_speed", "precip", "soil_moisture"):
            v = Xn[nb, :, BF.CH.index(c)]
            nbr += [v.mean(1), v.max(1)]
        st = BF.statics(fips)
        hist = BF.hist_features(z["y"], z["observed"])
        xr = np.concatenate([X, np.broadcast_to(st[:, None], (len(fips), BF.T, st.shape[1])),
                             np.broadcast_to(hist[:, None], (len(fips), BF.T, hist.shape[1])), np.stack(nbr, -1)], -1)
        xo = np.stack([wf[k] for k in BF.HAZARD], -1)
        y, obs = z["y"].astype(np.float64), z["observed"]
        parts["xu"].append(xu); parts["xr"].append(xr); parts["xo"].append(xo)
        parts["geo"].append(geo.reindex(fips)[geo_cols].to_numpy(float))
        parts["y0"].append(y[:, 71]); parts["y"].append(np.nan_to_num(y[:, BF.ORIGIN:]))
        parts["m"].append(obs[:, BF.ORIGIN:].astype(np.float64))
        parts["y_full"].append(y); parts["obs_full"].append(obs)
        parts["cust"].append(z["denominator"]); parts["fips"] += fips; parts["event"] += [e["event"]] * len(fips)
        print(e["event"], len(fips), "static NaN", int(np.isnan(st).sum()), "geo NaN", int(np.isnan(parts["geo"][-1]).sum()),
              flush=True)
    arr = {k: (np.concatenate(v).astype(np.float32) if k not in ("fips", "event", "obs_full") else
               (np.concatenate(v) if k == "obs_full" else np.array(v))) for k, v in parts.items()}
    assert np.isfinite(arr["xu"]).all() and np.isfinite(arr["xo"]).all() and np.isfinite(arr["y0"]).all()
    f = BF.OUT / f"features_{a.tag}.npz"
    np.savez_compressed(f, **arr, damage_features=np.array(DAMAGE_R2), recovery_features=np.array(BF.RECOVERY_FEATURES),
                        occurrence_features=np.array(BF.HAZARD), geo_features=np.array(geo_cols),
                        weather_channels=np.array(BF.CH))
    rec = dict(file=str(f.relative_to(BF.ROOT)), sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
               units=int(len(arr["fips"])), counties=int(len(set(arr["fips"]))), events=[e["event"] for e in events],
               d_u=int(arr["xu"].shape[-1]), d_r=int(arr["xr"].shape[-1]), d_occ=int(arr["xo"].shape[-1]),
               G=int(arr["geo"].shape[-1]))
    (BF.HERE / "data_provenance" / f"features_{a.tag}_checksum.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(rec)


if __name__ == "__main__":
    main()
