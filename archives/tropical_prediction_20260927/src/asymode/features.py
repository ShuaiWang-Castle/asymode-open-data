"""Covariate families, each built here from primary public sources.

The families exist to test `docs/PREREGISTRATION_asymmetry.md`. Each one is
registered with the side it is hypothesised to belong to, so an experiment can
move a family across the state equation and measure what happens rather than
asserting where it belongs.

Causality is enforced per family and declared in `FAMILIES`:

  `pre`     computed strictly before the forecast origin; constant over the window
  `causal`  at step t uses only steps <= t within the window
  `exog`    a driver at step t, given for the window under the forecast stand-in
  `static`  no time index at all

Nothing here aggregates *across* the forecast window. That would be a stronger
assumption than the forecast stand-in already made, and the pre-registration
excludes it.
"""

from __future__ import annotations

import contextlib
import warnings

import numpy as np
import pandas as pd


@contextlib.contextmanager
def _quiet_nan():
    """All-NaN rows are expected here -- a county with no observed lead-in has no
    history, and the caller zeroes those explicitly. The warning is noise."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"(All-NaN|Mean of empty)", RuntimeWarning)
        yield

# --------------------------------------------------------------------------
# pre-origin history — H-A1 (level, on restoration) and H-B (clearance, on damage)
# --------------------------------------------------------------------------

def history_level(y: np.ndarray, obs: np.ndarray, origin: int,
                  lookback: int = 24) -> tuple[np.ndarray, list[str]]:
    """Where the county stood before the window opened. H-A1.

    y and obs are (C, T) hourly. Only steps strictly before `origin` are read.
    """
    lo = max(0, origin - lookback)
    w, m = y[:, lo:origin], obs[:, lo:origin]
    v = np.where(m, w, np.nan)
    empty = ~np.isfinite(v).any(axis=1)          # counties with no observed lead-in
    with np.errstate(all="ignore"), _quiet_nan():
        last = _last_valid(v)
        mx = np.nanmax(v, axis=1)
        mean = np.nanmean(v, axis=1)
        short = np.nanmean(v[:, -6:], axis=1)
        # Trend over the lead-in: slope of a line through the observed points,
        # scaled so it reads as "fraction per hour".
        t = np.arange(v.shape[1], dtype=float)
        trend = _nanslope(v, t)
        active = np.nansum((v > 0).astype(float), axis=1) / max(v.shape[1], 1)
    out = np.stack([last, mx, mean, short, trend, active], axis=1)
    out[empty] = 0.0        # explicit, not an accident of nan_to_num
    return np.nan_to_num(out).astype(np.float32), [
        "hist_last", "hist_max", "hist_mean", "hist_mean6", "hist_trend", "hist_active"]


def history_clearance(y: np.ndarray, obs: np.ndarray, origin: int,
                      lookback: int = 48) -> tuple[np.ndarray, list[str]]:
    """How fast the county was clearing outages before the window. H-B.

    The same source as `history_level` and a different summary of it. The
    hypothesis is that the *level* belongs on restoration while the *rate at
    which the county clears* belongs on interruption, as a proxy for how fragile
    the county's plant is. Registered as a hypothesis, not a finding.

    Clearance is measured on downward steps only, in log space, so it is a
    relative decay rate rather than an absolute one and does not simply restate
    the level.
    """
    lo = max(0, origin - lookback)
    w, m = y[:, lo:origin], obs[:, lo:origin]
    mm = m[:, :-1] & m[:, 1:]
    a = np.where(mm, w[:, :-1], np.nan)
    b = np.where(mm, w[:, 1:], np.nan)
    with np.errstate(all="ignore"), _quiet_nan():
        # -dlog p on steps that fell, from a floor so zeros do not blow up
        fl = 1e-6
        drop = np.log(np.maximum(a, fl)) - np.log(np.maximum(b, fl))
        drop = np.where((b < a) & np.isfinite(drop), drop, np.nan)
        rate = np.nanmedian(drop, axis=1)
        n = np.nansum(np.isfinite(drop).astype(float), axis=1)
        frac_down = n / np.maximum(np.nansum(mm.astype(float), axis=1), 1)
    out = np.stack([np.nan_to_num(rate), np.log1p(np.nan_to_num(n)), frac_down], axis=1)
    return out.astype(np.float32), ["clr_rate", "clr_log_n", "clr_frac_down"]


def _last_valid(a: np.ndarray) -> np.ndarray:
    idx = np.where(np.isfinite(a), np.arange(a.shape[1])[None, :], -1).max(axis=1)
    r = np.full(a.shape[0], np.nan)
    ok = idx >= 0
    r[ok] = a[np.arange(a.shape[0])[ok], idx[ok]]
    return r


def _nanslope(v: np.ndarray, t: np.ndarray) -> np.ndarray:
    m = np.isfinite(v)
    n = m.sum(axis=1)
    tt = np.where(m, t[None, :], 0.0)
    vv = np.where(m, v, 0.0)
    st, sv = tt.sum(1), vv.sum(1)
    stt, stv = (tt * tt).sum(1), (tt * vv).sum(1)
    den = n * stt - st * st
    with np.errstate(all="ignore"):
        return np.where(den > 0, (n * stv - st * sv) / den, 0.0)


# --------------------------------------------------------------------------
# hazard composites — interruption side, and the gate under H-C
# --------------------------------------------------------------------------

HAZARD_ROLL = (6, 12)


def hazard_composites(X: np.ndarray, channels: list[str]) -> tuple[np.ndarray, list[str]]:
    """Damage-mechanism composites of the raw fields. Causal within the window.

    Each one names a mechanism rather than a correlation: wind acting on
    saturated ground, the super-linear part of wind loading, frozen accretion,
    and precipitation near the freezing point. Rolling sums are cumulative from
    the window's start and never look forward.
    """
    c = {n: i for i, n in enumerate(channels)}
    g = X[..., c["gust"]]
    ws = X[..., c["wind_speed"]]
    tp = X[..., c["precip"]]
    sf = X[..., c["snowfall"]]
    t2 = X[..., c["t2m_c"]]
    sm = X[..., c["soil_moisture"]]

    feats, names = [], []

    def add(a, n):
        feats.append(a.astype(np.float32)); names.append(n)

    add(g * sm, "hz_wet_wind")
    # Wind loading is roughly quadratic in speed above a threshold; below it the
    # term is zero rather than negative.
    add(np.maximum(g - 15.0, 0.0) ** 2, "hz_wind_energy")
    add(sf * np.maximum(1.0 - np.abs(t2) / 5.0, 0.0), "hz_snow_ice")
    add(np.exp(-((t2 - 0.0) ** 2) / 8.0), "hz_near_freeze")
    add(tp * (t2 < 2.0), "hz_cold_precip")
    add(g * ws, "hz_wind_product")

    base = np.stack(feats, axis=-1)
    roll = [base]
    for w in HAZARD_ROLL:
        cs = np.cumsum(base, axis=1)
        lag = np.concatenate([np.zeros_like(cs[:, :w]), cs[:, :-w]], axis=1)
        roll.append(cs - lag)
        names += [f"{n}_r{w}" for n in names[:base.shape[-1]]]
    out = np.concatenate(roll, axis=-1)
    names = ([n for n in names[:base.shape[-1]]]
             + [f"{n}_r{w}" for w in HAZARD_ROLL for n in names[:base.shape[-1]]])
    return out.astype(np.float32), names


def hazard_path(X: np.ndarray, channels: list[str]) -> tuple[np.ndarray, list[str]]:
    """Cumulative hazard since the forecast origin, and time since its peak."""
    c = {n: i for i, n in enumerate(channels)}
    g = X[..., c["gust"]]
    tp = X[..., c["precip"]]
    run_max = np.maximum.accumulate(g, axis=1)
    T = g.shape[1]
    idx = np.argmax(g[:, None, :] >= run_max[:, :, None] - 1e-9, axis=2)
    since = (np.arange(T)[None, :] - idx).astype(np.float32)
    out = np.stack([run_max, np.cumsum(g, axis=1), np.cumsum(tp, axis=1), since], axis=-1)
    return out.astype(np.float32), ["path_gust_max", "path_gust_sum",
                                    "path_precip_sum", "path_hours_since_peak"]


def freeze_cycle(X: np.ndarray, channels: list[str]) -> tuple[np.ndarray, list[str]]:
    """Freeze-thaw cycling over a trailing day. Causal within the window.

    Identically zero on a summer panel, which is correct rather than broken: an
    ablation that moves this family can only be informative on the winter and
    cold-season panels, and pooling it across all events will dilute any effect
    it has. Report it stratified or not at all.
    """
    c = {n: i for i, n in enumerate(channels)}
    t2 = X[..., c["t2m_c"]]
    below = (t2 < 0).astype(np.float32)
    cs = np.cumsum(below, axis=1)
    lag = np.concatenate([np.zeros_like(cs[:, :24]), cs[:, :-24]], axis=1)
    hours24 = cs - lag
    cross = np.abs(np.diff(below, axis=1, prepend=below[:, :1]))
    cc = np.cumsum(cross, axis=1)
    lag2 = np.concatenate([np.zeros_like(cc[:, :24]), cc[:, :-24]], axis=1)
    out = np.stack([hours24, cc - lag2], axis=-1)
    return out.astype(np.float32), ["frz_hours24", "frz_cross24"]


def direction_shift(X: np.ndarray, channels: list[str], lags=(3, 6)
                    ) -> tuple[np.ndarray, list[str]]:
    """Change in wind direction over the last few hours, wrapped to [0, 180].

    Requires the wind components; the magnitude alone cannot supply it. Returns
    an empty block if they were not carried, rather than silently substituting
    something else.
    """
    c = {n: i for i, n in enumerate(channels)}
    if "u10" not in c or "v10" not in c:
        return np.zeros(X.shape[:2] + (0,), dtype=np.float32), []
    ang = np.arctan2(X[..., c["v10"]], X[..., c["u10"]])
    feats, names = [], []
    for L in lags:
        prev = np.concatenate([ang[:, :1].repeat(L, axis=1), ang[:, :-L]], axis=1)
        d = np.abs(np.degrees(np.arctan2(np.sin(ang - prev), np.cos(ang - prev))))
        feats.append(d); names.append(f"dir_shift{L}")
    return np.stack(feats, axis=-1).astype(np.float32), names


# --------------------------------------------------------------------------
# neighbouring counties — H-A2, hypothesised on the restoration side
# --------------------------------------------------------------------------

def neighbour_drivers(X: np.ndarray, channels: list[str], fips: list[str],
                      adjacency: dict, use=("gust", "precip", "soil_moisture", "wind_speed")
                      ) -> tuple[np.ndarray, list[str]]:
    """Mean and max of a neighbour's drivers, over adjacent counties in the panel.

    Exogenous: it reads weather, never a neighbour's outage state. Reading the
    neighbour's state would make the panel's counties mutually dependent and
    change what the held-out fold means.
    """
    idx = {f: i for i, f in enumerate(fips)}
    c = {n: i for i, n in enumerate(channels)}
    C, T, _ = X.shape
    out = np.zeros((C, T, 2 * len(use)), dtype=np.float32)
    for f, i in idx.items():
        nb = [idx[g] for g in adjacency.get(f, ()) if g in idx]
        if not nb:
            nb = [i]                      # isolated: fall back to self, flagged below
        sub = X[nb]
        for k, ch in enumerate(use):
            out[i, :, 2 * k] = sub[..., c[ch]].mean(axis=0)
            out[i, :, 2 * k + 1] = sub[..., c[ch]].max(axis=0)
    names = [f"nb_{ch}_{s}" for ch in use for s in ("mean", "max")]
    return out, names


def load_adjacency(path) -> dict:
    """Census county adjacency file -> {fips: [neighbour fips]}."""
    d = pd.read_csv(path, sep="|", dtype=str, encoding="latin-1")
    cols = list(d.columns)
    src = [c for c in cols if "GEOID" in c.upper()][0]
    dst = [c for c in cols if "GEOID" in c.upper()][-1]
    d = d[[src, dst]].dropna()
    d.columns = ["a", "b"]
    d = d[d.a != d.b]
    return d.groupby("a")["b"].apply(list).to_dict()


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------
# `side` is the *hypothesised* home under the pre-registration, not a finding.
FAMILIES = {
    "hist_level":   dict(side="restoration", causality="pre",    hypothesis="H-A1"),
    "clearance":    dict(side="interruption", causality="pre",   hypothesis="H-B"),
    "neighbour":    dict(side="restoration", causality="exog",   hypothesis="H-A2"),
    "hazard":       dict(side="interruption", causality="causal", hypothesis="H-C"),
    "hazard_path":  dict(side="interruption", causality="causal", hypothesis=None),
    "freeze_cycle": dict(side="interruption", causality="causal", hypothesis=None,
                         note="identically zero on warm-season panels"),
    "dir_shift":    dict(side="interruption", causality="causal", hypothesis=None,
                         note="needs u10/v10; empty until drivers are rebuilt"),
    "statics":      dict(side="gate",        causality="static", hypothesis="H-C"),
    # Ambient is the registered NEGATIVE case: expected to help both sides. It
    # must contain only fields with no strong prior on either side, or the
    # negative case is rigged. Soil moisture is deliberately NOT here -- saturated
    # ground driving root failure is a damage mechanism, so it sits with hazard.
    "ambient":      dict(side="both",        causality="exog",   hypothesis="H-A3",
                         channels=("pressure", "cloud", "rh", "t2m_c")),
}
