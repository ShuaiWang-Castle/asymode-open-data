"""Three process-specific feature blocks, built causally.

The competition dimensions do not transfer, but the roles do. What is preserved is
that interruption magnitude, interruption occurrence and recovery read genuinely
different blocks, and that the occurrence block is neither the magnitude block nor
any hidden representation of it.

Causality. The forecast is open loop: weather over the horizon is given, the outage
state after the origin is not. So a feature may read weather at or before the step
it is used at, and outage only from strictly before the origin. Accumulated hazard
is accumulated *from the origin forward*, which is legal for the same reason the
horizon weather is.

Nothing here reads the simulated state, a county identifier, a post-origin outage
observation, or any target-derived quantity.
"""
from __future__ import annotations

import warnings

import numpy as np

# instantaneous weather actually present in the open-data driver block
WX = ["cape", "cloud", "gust", "precip", "pressure", "rh", "snowfall",
      "soil_moisture", "t2m_c", "u10", "v10", "wind_speed"]
STATIC = ["log_area", "rucc", "log_pop", "log_pop_density", "n_neighbours",
          "n_utilities", "coop_share", "saidi", "saifi", "log_cust",
          "log_cust_density", "lat", "lon"]
# occurrence reads a deliberately narrow and differently composed block
OCC_WX = ["gust", "precip"]
OCC_STATIC = ["log_cust_density", "rucc", "n_utilities", "saidi"]
# recovery reads conditions that plausibly gate crew work, and no clock
R_WX = ["t2m_c", "wind_speed", "precip", "snowfall"]
GUST_FOOTPRINT_THRESHOLD = 15.0        # m/s, fixed before the pilot, never tuned


def _cols(channels: list[str], names: list[str]) -> list[int]:
    return [channels.index(n) for n in names]


def build_blocks(X: np.ndarray, channels: list[str], statics: np.ndarray,
                 y_hist: np.ndarray, hist_obs: np.ndarray, origin: int,
                 horizon: int, hour_of_day: np.ndarray, adjacency: np.ndarray | None):
    """Return (x_u, x_occ, x_r) each of shape (n_counties, horizon, d).

    `X` is the full hourly driver cube for the panel, `origin` the forecast origin
    index, `y_hist`/`hist_obs` the pre-origin outage window, `hour_of_day` the UTC
    hour of each forecast step, `adjacency` a row-normalised county adjacency
    matrix or None when it is unavailable.
    """
    C = X.shape[0]
    sl = slice(origin + 1, origin + 1 + horizon)
    W = X[:, sl, :].astype(np.float64)                     # (C, H, len(channels))
    H = W.shape[1]

    # ---- causal accumulated hazard, accumulated from the origin forward -------
    gi, pi_, wi, ci = (channels.index(c) for c in ("gust", "precip", "wind_speed", "cape"))
    gust_max = np.maximum.accumulate(W[:, :, gi], axis=1)
    wind_max = np.maximum.accumulate(W[:, :, wi], axis=1)
    cape_max = np.maximum.accumulate(W[:, :, ci], axis=1)
    precip_cum = np.cumsum(W[:, :, pi_], axis=1)
    path = np.stack([gust_max, wind_max, cape_max, precip_cum], axis=-1)

    # ---- exogenous storm footprint: share of the panel above a gust threshold --
    foot = np.broadcast_to((W[:, :, gi] > GUST_FOOTPRINT_THRESHOLD).mean(0)[None, :, None],
                           (C, H, 1))

    clock = np.stack([np.sin(2 * np.pi * hour_of_day / 24.0),
                      np.cos(2 * np.pi * hour_of_day / 24.0)], axis=-1)
    clock = np.broadcast_to(clock[None, :, :], (C, H, 2))
    S = np.broadcast_to(statics[:, None, :], (C, H, statics.shape[1]))

    x_u = np.concatenate([W[:, :, _cols(channels, WX)], path, foot, clock, S], axis=-1)

    x_occ = np.concatenate([
        W[:, :, _cols(channels, OCC_WX)],
        np.broadcast_to(statics[:, None, [STATIC.index(s) for s in OCC_STATIC]],
                        (C, H, len(OCC_STATIC)))], axis=-1)

    # ---- pre-origin outage history, strictly before the origin ---------------
    hv = np.where(hist_obs, y_hist, np.nan)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN county history
        h_last = np.nan_to_num(np.where(np.isnan(hv[:, -1]), 0.0, hv[:, -1]))
        h_max = np.nan_to_num(np.nanmax(hv, axis=1)) if hv.shape[1] else np.zeros(C)
        h_mean = np.nan_to_num(np.nanmean(hv, axis=1)) if hv.shape[1] else np.zeros(C)
        h_trend = np.nan_to_num(hv[:, -1] - hv[:, 0]) if hv.shape[1] > 1 else np.zeros(C)
    hist = np.stack([h_last, h_max, h_mean, h_trend], axis=-1)
    hist = np.broadcast_to(hist[:, None, :], (C, H, 4))

    r_parts = [W[:, :, _cols(channels, R_WX)], S, hist]
    if adjacency is not None:                              # legal neighbour weather
        nb = np.stack([adjacency @ W[:, :, gi], adjacency @ W[:, :, pi_]], axis=-1)
        r_parts.append(nb)
    x_r = np.concatenate(r_parts, axis=-1)                 # no clock, no state
    return (np.ascontiguousarray(x_u), np.ascontiguousarray(x_occ),
            np.ascontiguousarray(x_r))


def block_names(channels: list[str], with_neighbour: bool) -> dict:
    u = WX + ["gust_max_since_origin", "wind_max_since_origin",
              "cape_max_since_origin", "precip_cum_since_origin",
              "panel_gust_footprint_share", "clock_sin", "clock_cos"] + STATIC
    occ = OCC_WX + OCC_STATIC
    r = R_WX + STATIC + ["hist_last", "hist_max", "hist_mean", "hist_trend"]
    if with_neighbour:
        r = r + ["neighbour_gust_mean", "neighbour_precip_mean"]
    return {"x_u": u, "x_occ": occ, "x_r": r}


def row_normalised_adjacency(fips: np.ndarray, adj: dict) -> np.ndarray | None:
    idx = {f: i for i, f in enumerate(fips)}
    A = np.zeros((len(fips), len(fips)))
    for f, nbrs in adj.items():
        i = idx.get(f)
        if i is None:
            continue
        for g in nbrs:
            j = idx.get(g)
            if j is not None and j != i:
                A[i, j] = 1.0
    s = A.sum(1, keepdims=True)
    if not (s > 0).any():
        return None
    return A / np.maximum(s, 1.0)
