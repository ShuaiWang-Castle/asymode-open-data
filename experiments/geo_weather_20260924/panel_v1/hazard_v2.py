"""Hazard dictionary v2 of the designed panel (idea I12; physical review of the panel design, section 5): one set of
weather-derived hazard series for every regime, computed at the finest weather grid first and then integrated over
each county's 3 km population nodes, from ERA5 (the cropped archive of each system) and from HRRR (wrfsfcf01, 3 km).

Per grid cell and hour (derive):
  g, g_exc      gust (m/s); Klawa-Ulbrich exceedance of the local 98th percentile, max(0, g / g98 - 1) ** 3
                (g98: ERA5 fg10, one random hour a day 2008-2017, fetch_gust_climatology.py; HRRR cells take the g98 of
                their ERA5 cell)
  rain, frz, sleet, snow   precipitation by phase (mm/h liquid): ERA5 by ptype (rain 1 and 7, freezing rain 3 and
                freezing drizzle 12, ice pellets 8) and snowfall sf; HRRR by the categorical types CRAIN, CFRZR,
                CICEP, CSNOW applied to APCP, with FRZR (freezing-rain accumulation) as frz
  ice, wetsnow  accretion loads (mm), temperature-gated states (not linear filters): ice += frz, melts 0.5 mm per
                degree-hour above 0 C and sheds 10 % per hour with gust > 15 m/s (Jones 1998; ISO 12494); wet snow
                accumulates snowfall at -1..+3 C, melts 1 mm per degree-hour above 3 C, sheds 10 % per hour with gust
                > 12 m/s or below -3 C (dry snow does not stick)
  ice_wind, snow_wind, wet_wind   compound loads formed before integration: ice x g^2, wetsnow x g^2 (/1000), and
                antecedent wetness x g_exc (soil water at the origin: ERA5 swvl1 / 0.5, HRRR MSTAV / 100)
  conv          HRRR only: 0-6 km shear magnitude (read every 6 h and held), 2-5 km updraft helicity, composite
                reflectivity, lightning
  profile       warm layer aloft (HRRR: TMP 850 hPa > 0 C over a surface below 0 C); ERA5: leaf area (lai_hv)
Everything is computed at the node (county x 3 km cell): ERA5 fields are taken from the node's 0.25-degree cell with
the 2 m temperature moved to the node's elevation (6.5 C per km against the ERA5 orography), so the phase and the loads
near 0 C follow the terrain; HRRR fields are the node's own cell.
Per county and hour (aggregate): the population-weighted mean over the county's nodes, the maximum over its nodes, the
population shares above fixed thresholds (g > 25 m/s, g_exc > 0, ice > 6 mm, wetsnow > 10 mm, uh > 75 m2/s2,
refc > 50 dBZ), and geography x weather formed at the node before integration: canopy x g_exc, canopy x ice_wind,
canopy x snow_wind, poorly drained share x wet_wind (node modulators in 0..1). Missing HRRR hours
enter as zeros and are listed (DATASET_DESIGN amendment 2 S9)."""
from __future__ import annotations

import numpy as np

T0_ICE_MELT, ICE_SHED_G, SHED = 0.5, 15.0, 0.10
WS_LO, WS_HI, WS_MELT, WS_SHED_G, WS_COLD = -1.0, 3.0, 1.0, 12.0, -3.0


def loads(t2m, frz, snow, g):
    """Temperature-gated accretion states, hour by hour (arrays [T, N], t2m in C)."""
    T, N = t2m.shape
    ice = np.zeros((T, N), np.float32); ws = np.zeros((T, N), np.float32)
    I = np.zeros(N, np.float32); S = np.zeros(N, np.float32)
    for t in range(T):
        I = I + frz[t]
        I = np.maximum(0.0, I - T0_ICE_MELT * np.maximum(t2m[t], 0.0))
        I = np.where(g[t] > ICE_SHED_G, I * (1 - SHED), I)
        wet = (t2m[t] >= WS_LO) & (t2m[t] <= WS_HI)
        S = S + np.where(wet, snow[t], 0.0)
        S = np.maximum(0.0, S - WS_MELT * np.maximum(t2m[t] - WS_HI, 0.0))
        S = np.where((g[t] > WS_SHED_G) | (t2m[t] < WS_COLD), S * (1 - SHED), S)
        ice[t], ws[t] = I, S
    return ice, ws


def derive(f: dict, g98: np.ndarray) -> dict:
    """f: hourly cell fields [T, N] in physical units (g m/s, t2m C, rain/frz/sleet/snow mm/h, wet [N] 0..1, optional
    conv and profile fields). Returns the derived cell series."""
    g = f["g"]
    out = dict(g=g, g_exc=np.maximum(0.0, g / np.maximum(g98, 5.0)[None, :] - 1.0) ** 3,
               rain=f["rain"], frz=f["frz"], sleet=f["sleet"], snow=f["snow"])
    ice, ws = loads(f["t2m"], f["frz"], f["snow"], g)
    out.update(ice=ice, wetsnow=ws, ice_wind=ice * g ** 2 / 1000.0, snow_wind=ws * g ** 2 / 1000.0,
               wet_wind=f["wet"][None, :] * out["g_exc"])
    for k in ("shear", "uh", "refc", "ltng", "warm_nose", "lai"):
        if k in f:
            out[k] = f[k]
    return out


MEAN = ("g", "g_exc", "rain", "frz", "sleet", "snow", "ice", "wetsnow", "ice_wind", "snow_wind", "wet_wind",
        "shear", "uh", "refc", "ltng", "warm_nose", "lai")
MAX = ("g", "g_exc", "ice", "wetsnow", "ice_wind", "snow_wind", "uh", "refc")
SHARE = (("g", 25.0), ("g_exc", 0.0), ("ice", 6.0), ("wetsnow", 10.0), ("uh", 75.0), ("refc", 50.0))
# geography x weather, formed at the node before integration: modulator x hazard
MOD = (("canopy", "g_exc"), ("canopy", "ice_wind"), ("canopy", "snow_wind"), ("drain", "wet_wind"))


def aggregate(d: dict, node_county: np.ndarray, w: np.ndarray, n_counties: int, geo: dict):
    """County series [C, T, K] and names from node series [T, N] (w: node population shares, summing to 1 in each
    county; geo: node modulators in 0..1)."""
    T = next(iter(d.values())).shape[0]
    cols, names = [], []

    def cmean(v):
        out = np.zeros((n_counties, T), np.float32)
        for t in range(T):
            out[:, t] = np.bincount(node_county, v[t] * w, n_counties)
        return out

    def cmax(v):
        out = np.full((n_counties, T), -np.inf, np.float32)
        for t in range(T):
            np.maximum.at(out[:, t], node_county, v[t])
        return out

    for k in MEAN:
        if k in d:
            cols.append(cmean(d[k])); names.append(f"{k}_mean")
    for k in MAX:
        if k in d:
            cols.append(cmax(d[k])); names.append(f"{k}_max")
    for k, thr in SHARE:
        if k in d:
            cols.append(cmean((d[k] > thr).astype(np.float32))); names.append(f"{k}_gt{thr:g}_share")
    for m, k in MOD:
        if k in d and m in geo:
            cols.append(cmean(d[k] * geo[m][None, :])); names.append(f"{m}*{k}")
    return np.stack(cols, -1).astype(np.float32), names
