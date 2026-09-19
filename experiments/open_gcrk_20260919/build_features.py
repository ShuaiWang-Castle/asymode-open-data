"""Model inputs for every county-event unit, from the 216-hour panels (raw, unstandardised).

Damage inputs x^U (weather only; standardisation happens per fitting set at train time):
  current weather      the 14-channel ERA5 block (12 fields + sin/cos UTC hour)
  hazard composites    7 instantaneous weather composites (also the occurrence-gate input):
                         gust_excess_energy   max(gust - 15 m/s, 0)^2
                         wet_wind             max(gust - 12 m/s, 0) x precip x (1 + soil moisture)
                         snow_ice_load        (snowfall + precip x 1[t2m <= 0.5 C]) x near_freeze x (0.25 + rh/100)
                         near_freeze          exp(-(t2m / 3.5 C)^2)
                         cold_precip          precip x near_freeze
                         gust_excess_energy summed over the last 6 h, wet_wind over the last 12 h
  path summaries       cumulated from the forecast origin (hour 72; zero before it):
                         running max of gust, cumulative gust_excess_energy, wet_wind,
                         snow_ice_load, hours since the running gust maximum
  past windows         gust max over 6/12/24 h, precip sum over 6/12/24 h, wind-speed mean
                         over 12 h, soil-moisture mean over 24 h
  freeze cycle         sub-zero hours and 0 C crossings in the last 24 h
  direction            3 h and 6 h absolute change of wind direction; sin/cos of direction
Every summary at hour t uses hours <= t only.

Recovery inputs x^R:
  current weather      the same 14 channels
  county background    log customers (EAGLE-I 2024), rural-urban continuum code (USDA ERS
                       2023), log population density (Census 2020 / Gazetteer land area),
                       cooperative share of serving utilities and log(1 + number of
                       utilities) (EIA-861 2023), log(1 + SAIDI) (EIA-861 2023)
  prefix outages       p_71, max p over hours 0-71, mean p over hours 66-71, p_71 - p_65,
                       share of prefix hours with p > 0.005
  neighbour weather    mean and max of gust, wind speed, precip and soil moisture over the
                       county's five nearest counties (Census centroids)

Geography g: data/interim/open_gcrk/geography.parquet (build_geography.py); enters only
through the response kernel.

Target: hourly outage fraction p over hours 72..215 with its observation mask; the
initial condition is the observed p_71.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INTERIM = ROOT / "data" / "interim"
OUT = INTERIM / "open_gcrk"
T, ORIGIN = 216, 72
CH = ["cape", "cloud", "gust", "precip", "pressure", "rh", "snowfall", "soil_moisture", "t2m_c", "u10", "v10",
      "wind_speed", "clock_sin", "clock_cos"]
HAZARD = ["gust_excess_energy", "wet_wind", "snow_ice_load", "near_freeze", "cold_precip",
          "gust_excess_energy_sum6", "wet_wind_sum12"]
PATH = ["path_gust_max", "path_gust_excess_energy", "path_wet_wind", "path_snow_ice_load", "hours_since_path_gust_max"]
PAST = ["gust_max6", "gust_max12", "gust_max24", "precip_sum6", "precip_sum12", "precip_sum24",
        "wind_speed_mean12", "soil_moisture_mean24"]
FREEZE = ["subzero_hours24", "zero_crossings24"]
DIRECTION = ["dir_change3", "dir_change6", "dir_sin", "dir_cos"]
DAMAGE_FEATURES = CH + HAZARD + PATH + PAST + FREEZE + DIRECTION
STATIC = ["log_cust", "rucc", "log_pop_density", "coop_share", "log1p_n_utilities", "log1p_saidi"]
HIST = ["p71", "p_max_prefix", "p_mean_66_71", "p_trend_65_71", "prefix_active_share"]
NEIGHBOUR = [f"nbr_{c}_{s}" for c in ("gust", "wind_speed", "precip", "soil_moisture") for s in ("mean", "max")]
RECOVERY_FEATURES = CH + STATIC + HIST + NEIGHBOUR


def rolling(a: np.ndarray, n: int, how: str) -> np.ndarray:
    """Trailing window over axis 1 (hours <= t), min_periods = 1."""
    out = np.empty_like(a)
    for t in range(a.shape[1]):
        w = a[:, max(0, t - n + 1):t + 1]
        out[:, t] = {"max": w.max(1), "sum": w.sum(1), "mean": w.mean(1)}[how]
    return out


def weather_features(X: np.ndarray) -> dict[str, np.ndarray]:
    c = {n: X[..., k].astype(np.float64) for k, n in enumerate(CH)}
    gust, prec, snow = c["gust"], np.clip(c["precip"], 0, None), np.clip(c["snowfall"], 0, None)
    t2, rh, soil = c["t2m_c"], c["rh"], np.clip(c["soil_moisture"], 0, None)
    f = {}
    f["gust_excess_energy"] = np.clip(gust - 15.0, 0, None) ** 2
    f["wet_wind"] = np.clip(gust - 12.0, 0, None) * prec * (1.0 + soil)
    f["near_freeze"] = np.exp(-(t2 / 3.5) ** 2)
    f["snow_ice_load"] = (snow + prec * (t2 <= 0.5)) * f["near_freeze"] * (0.25 + rh / 100.0)
    f["cold_precip"] = prec * f["near_freeze"]
    f["gust_excess_energy_sum6"] = rolling(f["gust_excess_energy"], 6, "sum")
    f["wet_wind_sum12"] = rolling(f["wet_wind"], 12, "sum")
    fut = np.arange(T) >= ORIGIN
    def path_cum(v, op):
        out = np.zeros_like(v)
        out[:, fut] = op(v[:, fut], axis=1)
        return out
    f["path_gust_max"] = path_cum(gust, np.maximum.accumulate)
    f["path_gust_excess_energy"] = path_cum(f["gust_excess_energy"], np.cumsum)
    f["path_wet_wind"] = path_cum(f["wet_wind"], np.cumsum)
    f["path_snow_ice_load"] = path_cum(f["snow_ice_load"], np.cumsum)
    since = np.zeros_like(gust)
    g = gust[:, fut]
    arg = np.zeros_like(g, dtype=int)
    best = np.full(g.shape[0], -np.inf); bi = np.zeros(g.shape[0], dtype=int)
    for t in range(g.shape[1]):
        upd = g[:, t] >= best
        best = np.where(upd, g[:, t], best); bi = np.where(upd, t, bi); arg[:, t] = bi
    since[:, fut] = np.arange(g.shape[1])[None, :] - arg
    f["hours_since_path_gust_max"] = since
    for n in (6, 12, 24):
        f[f"gust_max{n}"] = rolling(gust, n, "max")
        f[f"precip_sum{n}"] = rolling(prec, n, "sum")
    f["wind_speed_mean12"] = rolling(c["wind_speed"], 12, "mean")
    f["soil_moisture_mean24"] = rolling(soil, 24, "mean")
    below = (t2 < 0).astype(float)
    sgn = np.sign(t2)
    cross = np.concatenate([np.zeros((t2.shape[0], 1)), (np.abs(np.diff(sgn, axis=1)) > 0).astype(float)], 1)
    f["subzero_hours24"] = rolling(below, 24, "sum")
    f["zero_crossings24"] = rolling(cross, 24, "sum")
    ang = np.degrees(np.arctan2(-c["u10"], -c["v10"])) % 360.0          # direction the wind blows from
    for h in (3, 6):
        prev = np.concatenate([ang[:, :1].repeat(h, 1), ang[:, :-h]], 1)
        d = np.abs(ang - prev) % 360.0
        f[f"dir_change{h}"] = np.minimum(d, 360.0 - d)
    f["dir_sin"], f["dir_cos"] = np.sin(np.radians(ang)), np.cos(np.radians(ang))
    return f


def hist_features(y: np.ndarray, obs: np.ndarray) -> np.ndarray:
    v = np.where(obs[:, :ORIGIN], y[:, :ORIGIN], np.nan)
    with np.errstate(all="ignore"):
        p71 = v[:, 71]
        pmax = np.nanmax(v, 1)
        pm6 = np.nanmean(v[:, 66:72], 1)
        p65 = pd.DataFrame(v).T.ffill().T.to_numpy()[:, 65]
        trend = p71 - np.nan_to_num(p65, nan=p71)
        active = np.nanmean((v > 0.005).astype(float) + 0 * v, 1)
    return np.stack([p71, pmax, pm6, trend, active], 1)


def statics(fips: list[str]) -> np.ndarray:
    st = pd.read_parquet(INTERIM / "county_statics.parquet").set_index("fips")
    s = pd.DataFrame(index=fips)
    s["log_cust"] = st.log_cust.reindex(fips)
    s["rucc"] = st.rucc.reindex(fips)
    s["log_pop_density"] = st.log_pop_density.reindex(fips)
    s["coop_share"] = st.coop_share.reindex(fips)
    s["log1p_n_utilities"] = np.log1p(st.n_utilities.reindex(fips))
    s["log1p_saidi"] = np.log1p(st.saidi.reindex(fips))
    return s[STATIC].to_numpy(float)


GEO_EXT = ["soil_wet_share", "soil_windthrow_hazard", "forest_wet_coloc", "wet_in_forest", "hazard_in_forest",
           "forest_near_developed", "elev_mean5", "relief5", "fia_forest_land_share"]


def main(geo_ext: bool = False):
    """geo_ext: append the nine descriptors of build_geography_ext.py (PREREG Amendment 1) and
    write features_geo40.npz; the pre-registered features.npz is untouched."""
    events = json.loads((HERE / "selected_events.json").read_text())["events"]
    geo = pd.read_parquet(OUT / "geography.parquet")
    geo_cols = [c for c in geo.columns if c != "n_land_pixels"]
    if geo_ext:
        geo = geo.join(pd.read_parquet(OUT / "geography_ext.parquet")[GEO_EXT], how="left")
        geo_cols = geo_cols + GEO_EXT
    name = "features_geo40" if geo_ext else "features"
    parts = {k: [] for k in ("xu", "xr", "xo", "geo", "y0", "y", "m", "y_full", "obs_full", "cust", "fips", "event")}
    for e in events:
        z = np.load(OUT / f"panel216_{e['event']}.npz")
        fips = [str(f) for f in z["fips"]]
        X, Xn = z["X"].astype(np.float64), z["X_nbr"].astype(np.float64)
        wf = weather_features(X)
        xu = np.concatenate([X, np.stack([wf[k] for k in HAZARD + PATH + PAST + FREEZE + DIRECTION], -1)], -1)
        pos = {str(f): i for i, f in enumerate(z["nbr_fips"])}
        nb = np.stack([[pos[str(g)] for g in row] for row in z["nbr_of"]])          # (C, 5)
        nbr = []
        for c in ("gust", "wind_speed", "precip", "soil_moisture"):
            k = CH.index(c)
            v = Xn[nb, :, k]                                                          # (C, 5, T)
            nbr += [v.mean(1), v.max(1)]
        st = statics(fips)
        hist = hist_features(z["y"], z["observed"])
        xr = np.concatenate([X, np.broadcast_to(st[:, None], (len(fips), T, st.shape[1])),
                             np.broadcast_to(hist[:, None], (len(fips), T, hist.shape[1])),
                             np.stack(nbr, -1)], -1)
        xo = np.stack([wf[k] for k in HAZARD], -1)
        y, obs = z["y"].astype(np.float64), z["observed"]
        parts["xu"].append(xu); parts["xr"].append(xr); parts["xo"].append(xo)
        parts["geo"].append(geo.reindex(fips)[geo_cols].to_numpy(float))
        parts["y0"].append(y[:, 71]); parts["y"].append(np.nan_to_num(y[:, ORIGIN:]))
        parts["m"].append(obs[:, ORIGIN:].astype(np.float64))
        parts["y_full"].append(y); parts["obs_full"].append(obs)
        parts["cust"].append(z["denominator"]); parts["fips"] += fips; parts["event"] += [e["event"]] * len(fips)
        print(e["event"], len(fips), "static NaN", int(np.isnan(st).sum()), "geo NaN",
              int(np.isnan(parts["geo"][-1]).sum()), flush=True)
    arr = {k: (np.concatenate(v).astype(np.float32) if k not in ("fips", "event", "obs_full") else
               (np.concatenate(v) if k == "obs_full" else np.array(v))) for k, v in parts.items()}
    assert np.isfinite(arr["xu"]).all() and np.isfinite(arr["xo"]).all() and np.isfinite(arr["y0"]).all()
    np.savez_compressed(OUT / f"{name}.npz", **arr,
                        damage_features=np.array(DAMAGE_FEATURES), recovery_features=np.array(RECOVERY_FEATURES),
                        occurrence_features=np.array(HAZARD), geo_features=np.array(geo_cols),
                        weather_channels=np.array(CH))
    f = OUT / f"{name}.npz"
    rec = dict(file=str(f.relative_to(ROOT)), sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
               units=int(len(arr["fips"])), counties=int(len(set(arr["fips"]))),
               d_u=int(arr["xu"].shape[-1]), d_r=int(arr["xr"].shape[-1]), d_occ=int(arr["xo"].shape[-1]),
               G=int(arr["geo"].shape[-1]))
    (HERE / "data_provenance" / f"{name}_checksum.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(rec)


if __name__ == "__main__":
    main(geo_ext="--geo-ext" in sys.argv[1:])
