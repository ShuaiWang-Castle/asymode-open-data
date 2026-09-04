"""Transition construction and the outcome-blind exogenous active-48 window.

Every function here is deliberately blind to outage state when it defines a
*design*. ``active_window`` accepts no outage array, no target and no residual;
this is enforced by signature and asserted in the test-suite.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from preflight_lib import to_hourly, stable_hash

# The five public weather channels used only to break exact footprint ties.
TIE_CHANNELS = ("gust", "wind_speed", "precip", "snowfall", "cape")

ACTIVE_HALF = 24          # 24 transitions before, 24 after -> 48 transitions
DESIGNS = ("full", "active48")


@dataclass
class EventPanel:
    event: str
    fips: np.ndarray            # (C,)
    y_hourly: np.ndarray        # (C, H)  NaN where unobserved
    obs_hourly: np.ndarray      # (C, H)  bool
    X: np.ndarray               # (C, Hd, 12) hourly weather
    channels: list[str]
    ts_hourly: pd.DatetimeIndex  # (Hd,)
    denominator: np.ndarray     # (C,)


def load_event(interim, event: str) -> EventPanel:
    pz = np.load(interim / f"panel_{event}.npz", allow_pickle=True)
    dz = np.load(interim / f"drivers_{event}.npz", allow_pickle=True)
    if not np.array_equal(pz["fips"], dz["fips"]):
        raise ValueError(f"{event}: panel/driver county order differs")
    yh, oh = to_hourly(pz["y"], pz["observed"])
    ts = pd.to_datetime([str(t) for t in dz["ts"]])
    return EventPanel(
        event=event,
        fips=np.asarray(pz["fips"]).astype(str),
        y_hourly=yh, obs_hourly=oh,
        X=np.asarray(dz["X"], dtype=np.float64),
        channels=[str(c) for c in dz["channels"]],
        ts_hourly=ts,
        denominator=np.asarray(pz["denominator"], dtype=np.float64),
    )


# --------------------------------------------------------------------------
# outcome-blind exogenous storm window
# --------------------------------------------------------------------------

def noaa_hourly_footprint(events_df: pd.DataFrame, fips: np.ndarray,
                          ts_hourly: pd.DatetimeIndex, n_hours: int) -> np.ndarray:
    """Fraction of panel counties with an active NOAA event record at each hour.

    A county is active at hour h when some NOAA county-coded record for that
    county satisfies t_begin_utc <= h <= t_end_utc.  Uses only the public
    NOAA catalogue; no outage array is consulted.
    """
    want = set(fips.tolist())
    sub = events_df[events_df["fips"].isin(want)]
    hours = ts_hourly[:n_hours]
    counts = np.zeros(n_hours, dtype=np.int64)
    if sub.empty:
        return counts.astype(np.float64)
    code = {f: i for i, f in enumerate(fips.tolist())}
    active = np.zeros((len(fips), n_hours), dtype=bool)
    hv = hours.values.astype("datetime64[ns]")
    b = sub["t_begin_utc"].values.astype("datetime64[ns]")
    e = sub["t_end_utc"].values.astype("datetime64[ns]")
    idx = sub["fips"].map(code).to_numpy()
    for ci, bb, ee in zip(idx, b, e):
        if ee < hv[0] or bb > hv[-1]:
            continue
        lo = int(np.searchsorted(hv, bb, side="left"))
        hi = int(np.searchsorted(hv, ee, side="right"))
        if hi > lo:
            active[ci, lo:hi] = True
    counts = active.sum(axis=0)
    return counts.astype(np.float64) / float(len(fips))


def tie_break_composite(panel: EventPanel, n_hours: int) -> np.ndarray:
    """Per-hour positive-part standardized public-weather composite.

    Standardization is over all county-hour cells within the event, per channel.
    Only public weather enters; no outage value, target or residual.
    """
    idx = [panel.channels.index(c) for c in TIE_CHANNELS if c in panel.channels]
    if not idx:
        return np.zeros(n_hours)
    Z = panel.X[:, :n_hours, idx].astype(np.float64)        # (C, H, k)
    mean = Z.mean(axis=(0, 1), keepdims=True)
    std = Z.std(axis=(0, 1), keepdims=True)
    std = np.where(std > 0, std, 1.0)
    pos = np.maximum((Z - mean) / std, 0.0)
    return pos.mean(axis=(0, 2))                            # average counties+channels


def active_window(footprint: np.ndarray, composite: np.ndarray,
                  n_states: int) -> dict:
    """Choose the fixed 48-transition exogenous window.

    Deliberately accepts ONLY exogenous arrays: an outcome-blind NOAA footprint
    and a public-weather tie-break composite. No outage array, target, residual
    or prior gain can reach this function.

    Returns the peak hour, the transition index range and availability. An
    unavailable window is reported as unavailable and is never clipped into
    validity.
    """
    if footprint.size == 0:
        return dict(available=False, reason="no_hours", peak=None,
                    t_start=None, t_end=None, n_transitions=0, peak_footprint=np.nan)

    best = float(np.max(footprint))
    tied = np.flatnonzero(footprint >= best - 0.0)
    if tied.size > 1:
        comp = composite[tied]
        cbest = float(np.max(comp))
        tied = tied[comp >= cbest - 0.0]
    peak = int(tied[0])                       # remaining ties -> earliest hour

    t_start = peak - ACTIVE_HALF              # first current-state index
    t_end = peak + ACTIVE_HALF - 1            # last  current-state index
    # transitions t -> t+1 for t in [t_start, t_end]; needs state index t_end+1
    if t_start < 0 or (t_end + 1) > (n_states - 1):
        return dict(available=False, reason="window_outside_panel", peak=peak,
                    t_start=t_start, t_end=t_end, n_transitions=0,
                    peak_footprint=best)
    return dict(available=True, reason="", peak=peak, t_start=t_start, t_end=t_end,
                n_transitions=2 * ACTIVE_HALF, peak_footprint=best)


# --------------------------------------------------------------------------
# transitions
# --------------------------------------------------------------------------

def build_transitions(panel: EventPanel) -> pd.DataFrame:
    """All legal adjacent observed hourly transitions for one event.

    A transition t -> t+1 is retained only when BOTH hourly states are observed
    and finite. No unobserved current state is ever zero-filled. The exogenous
    weather vector is taken at t+1, and only a deterministic UTC clock
    sine/cosine pair is appended.
    """
    y, obs = panel.y_hourly, panel.obs_hourly
    C, H = y.shape
    Hd = panel.X.shape[1]
    n_states = min(H, Hd)                     # hourly states usable with weather

    cur = np.arange(0, n_states - 1)
    ok = (obs[:, cur] & obs[:, cur + 1]
          & np.isfinite(y[:, cur]) & np.isfinite(y[:, cur + 1]))
    ci, ti = np.nonzero(ok)                   # county index, current-state index
    t_cur = cur[ti]

    y0 = y[ci, t_cur]
    y1 = y[ci, t_cur + 1]
    W = panel.X[ci, t_cur + 1, :]             # weather aligned at t+1

    hod = panel.ts_hourly[t_cur + 1].hour.to_numpy().astype(np.float64)
    clock = np.stack([np.sin(2 * np.pi * hod / 24.0),
                      np.cos(2 * np.pi * hod / 24.0)], axis=-1)

    den = panel.denominator[ci]
    with np.errstate(divide="ignore", invalid="ignore"):
        ocf = np.where(den > 0, 1.0 / den, np.nan)

    df = pd.DataFrame({
        "event": panel.event,
        "county": panel.fips[ci],
        "t_cur": t_cur.astype(np.int32),
        "y": y0, "y_next": y1, "delta": y1 - y0,
        "one_customer_fraction": ocf,
    })
    for j, name in enumerate(panel.channels):
        df[f"w_{name}"] = W[:, j]
    df["w_clock_sin"] = clock[:, 0]
    df["w_clock_cos"] = clock[:, 1]
    df["physical_hour"] = panel.ts_hourly[t_cur].astype("int64") // 10 ** 9
    return df


def feature_columns(channels: list[str]) -> list[str]:
    return [f"w_{c}" for c in channels] + ["w_clock_sin", "w_clock_cos"]


def deterministic_subsample(df: pd.DataFrame, cap: int, salt: str) -> pd.DataFrame:
    """Keep at most `cap` rows, chosen by a deterministic hash of the row key."""
    if len(df) <= cap:
        return df
    keys = list(zip(df["event"], df["county"], df["physical_hour"]))
    h = np.array([stable_hash(*k, salt=salt) for k in keys], dtype=np.uint64)
    order = np.lexsort((np.arange(len(df)), h))
    return df.iloc[np.sort(order[:cap])]
