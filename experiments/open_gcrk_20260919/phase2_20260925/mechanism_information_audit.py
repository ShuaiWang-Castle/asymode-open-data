#!/usr/bin/env python3
"""Synthetic information audit for the proposed bounded process states.

This is not a fitted W+Cin comparison and contains no outage observations. It
constructs two weather histories whose 42-dimensional R2 damage feature vector
is identical at a later wind hour. The proposed wetness state distinguishes
them, but the existing host's learned scalar logit smoother can also retain the
timing of earlier inputs in principle. Therefore order sensitivity alone is not
evidence that a new process graph is necessary.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
sys.path.insert(0, str(EXPERIMENT))

import build_features as bf  # noqa: E402
import build_features_r2 as r2  # noqa: E402
from process_graph_prototype import Geography, rollout  # noqa: E402


T = bf.T
RAIN_EARLY = 80
RAIN_LATE = 98
WIND_HOUR = 144


def _weather(rain_hour: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return X/Hg/S plus rain/gust for one deliberately simplified history."""
    rain = np.zeros(T, dtype=np.float64)
    gust = np.full(T, 5.0, dtype=np.float64)
    rain[rain_hour] = 20.0
    gust[WIND_HOUR] = 25.0
    x = np.zeros((1, T, len(bf.CH)), dtype=np.float64)
    for name, value in {"gust": gust, "precip": rain, "wind_speed": gust,
                        "soil_moisture": np.full(T, 0.25), "t2m_c": np.full(T, 10.0),
                        "rh": np.full(T, 60.0)}.items():
        x[0, :, bf.CH.index(name)] = value
    # The R2 builder supplies cell-first hazard composites. For this audit the
    # only nonzero relevant composite is gust exceedance at the response hour;
    # rain occurs with sub-threshold gust, so wet_wind is zero in both histories.
    hg = np.zeros((1, T, 5), dtype=np.float64)
    hg[0, :, 0] = np.clip(gust - 15.0, 0.0, None) ** 2
    s = np.zeros((1, T, 2), dtype=np.float64)
    s[0, :, 0] = gust
    s[0, :, 1] = (gust > 15.0).astype(float)
    return x, hg, s, rain, gust


def _damage_vector(x: np.ndarray, hg: np.ndarray, support: np.ndarray) -> np.ndarray:
    f = r2.weather_features_r2(x, hg, support)
    return np.concatenate(
        [x, np.stack([f[k] for k in r2.DAMAGE_R2[len(bf.CH):]], axis=-1)], axis=-1
    )[0, WIND_HOUR]


def _scalar_smoother(signal: np.ndarray, forget: float) -> np.ndarray:
    out = np.empty_like(signal, dtype=np.float64)
    out[0] = signal[0]
    for t in range(1, signal.size):
        out[t] = forget * out[t - 1] + (1.0 - forget) * signal[t]
    return out


def run_audit() -> dict:
    e = _weather(RAIN_EARLY)
    l = _weather(RAIN_LATE)
    xe, xl = _damage_vector(*e[:3]), _damage_vector(*l[:3])
    geo = Geography(poor_drainage=0.8, canopy_proxy=0.8, terrain_exposure=0.6)
    base_u, base_r = np.full(T, 0.04), np.full(T, 0.03)
    pe = rollout(e[3], e[4], geo, base_u, base_r)
    pl = rollout(l[3], l[4], geo, base_u, base_r)
    # This is a constructive capacity witness, not a trained W+Cin weight:
    # its damage MLP could emit a pulse from precipitation and set a constant
    # smoother gate. One scalar leaky state then retains input timing.
    # The real host starts this recurrence at forecast hour 72, so use only
    # the legal post-origin sequence for the capacity witness.
    se = _scalar_smoother(e[3][bf.ORIGIN:] / 20.0, forget=0.95)
    sl = _scalar_smoother(l[3][bf.ORIGIN:] / 20.0, forget=0.95)
    response_index = WIND_HOUR - bf.ORIGIN
    return {
        "synthetic_only": True,
        "hours": {"rain_early": RAIN_EARLY, "rain_late": RAIN_LATE, "wind": WIND_HOUR},
        "same_total_rain": bool(e[3].sum() == l[3].sum()),
        "same_current_weather_at_wind": bool(np.array_equal(e[0][0, WIND_HOUR], l[0][0, WIND_HOUR])),
        "r2_damage_feature_dimension": int(xe.size),
        "max_abs_r2_damage_feature_difference_at_wind": float(np.max(np.abs(xe - xl))),
        "bounded_wetness_before_wind": {"early": float(pe.wet_before[WIND_HOUR]),
                                         "late": float(pl.wet_before[WIND_HOUR])},
        "bounded_wetness_absolute_difference": float(abs(pe.wet_before[WIND_HOUR] - pl.wet_before[WIND_HOUR])),
        "host_scalar_smoother_capacity_witness": {"forget": 0.95,
                                                   "early": float(se[response_index]),
                                                   "late": float(sl[response_index]),
                                                   "absolute_difference": float(abs(se[response_index] - sl[response_index]))},
        "interpretation": (
            "The response-hour R2 feature vector collides while the bounded wetness state separates the histories. "
            "However, the existing host's scalar learned smoother can also separate them from the earlier input sequence. "
            "This is a necessity/falsification audit, not a performance result."
        ),
    }


def _self_test() -> None:
    result = run_audit()
    assert result["same_total_rain"] and result["same_current_weather_at_wind"]
    assert result["r2_damage_feature_dimension"] == 42
    assert result["max_abs_r2_damage_feature_difference_at_wind"] == 0.0
    assert result["bounded_wetness_absolute_difference"] > 1e-3
    assert result["host_scalar_smoother_capacity_witness"]["absolute_difference"] > 1e-3
    print("self-test passed: R2 response-hour collision; process state and host smoother both retain order")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
    else:
        print(json.dumps(run_audit(), indent=2))
