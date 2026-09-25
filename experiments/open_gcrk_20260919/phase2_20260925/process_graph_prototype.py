"""Small synthetic-only, bounded weather/geography process-graph prototype.

This is an executable mathematical feasibility check, not an outage model fit.
All weather and rates must be explicitly supplied by the caller; no files or
third-party packages beyond NumPy are read. Geography and states are proxies.

Usage: python process_graph_prototype.py
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Geography:
    """Unit-interval *proxy* descriptors, not measured grid component exposure."""

    poor_drainage: float
    canopy_proxy: float
    terrain_exposure: float

    def __post_init__(self) -> None:
        for name in ("poor_drainage", "canopy_proxy", "terrain_exposure"):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")


@dataclass(frozen=True)
class Trajectory:
    p: np.ndarray  # length T+1, including the observed forecast origin
    damage: np.ndarray
    recovery: np.ndarray
    residual: np.ndarray
    wet_before: np.ndarray
    wet_after: np.ndarray
    fatigue_before: np.ndarray
    fatigue_after: np.ndarray


def _input_vector(name: str, values: np.ndarray, *, upper: float | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite one-dimensional array")
    if np.any(array < 0.0) or (upper is not None and np.any(array > upper)):
        interval = f"[0, {upper}]" if upper is not None else "[0, infinity)"
        raise ValueError(f"{name} must lie in {interval}")
    return array


def _scalar(name: str, value: float) -> float:
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return float(value)


def baseline_path(damage: np.ndarray, recovery: np.ndarray, p0: float) -> np.ndarray:
    """AsymODE stock update using *provided* W+Cin rates, with no model fitting."""
    u = _input_vector("damage", damage, upper=1.0)
    r = _input_vector("recovery", recovery, upper=1.0)
    if u.shape != r.shape:
        raise ValueError("damage and recovery must have equal length")
    initial = _scalar("p0", p0)
    p = np.empty(u.size + 1, dtype=np.float64)
    p[0] = initial
    for t in range(u.size):
        p[t + 1] = p[t] + u[t] * (1.0 - p[t]) - r[t] * p[t]
    return p


def rollout(
    rain_mm: np.ndarray,
    gust_ms: np.ndarray,
    geography: Geography,
    baseline_damage: np.ndarray,
    baseline_recovery: np.ndarray,
    *,
    p0: float = 0.0,
    wet0: float = 0.10,
    fatigue0: float = 0.0,
    alpha: float = 0.8,
    residual_open: bool = True,
) -> Trajectory:
    """Forward causal recursion with bounded weather/geography intermediate states.

    At hour t, current gust and *pre-hour* wetness/fatigue affect damage.
    Rain and wind at hour t then change states for hour t+1. Alpha is the
    bounded residual scale. With residual_open=False or alpha=0, damage
    rates are copied directly from W+Cin; baseline_path is bitwise identical.

    The gust threshold uses a smooth softplus, so the map is differentiable
    inside the input/geography bounds. At exact domain boundaries, gradients
    are understood as one-sided. This is not a trained or physical PINN.
    """
    rain = _input_vector("rain_mm", rain_mm)
    gust = _input_vector("gust_ms", gust_ms)
    base_u = _input_vector("baseline_damage", baseline_damage, upper=1.0)
    base_r = _input_vector("baseline_recovery", baseline_recovery, upper=1.0)
    if not (rain.shape == gust.shape == base_u.shape == base_r.shape):
        raise ValueError("weather and rate arrays must have the same length")
    initial_p = _scalar("p0", p0)
    wet = _scalar("wet0", wet0)
    fatigue = _scalar("fatigue0", fatigue0)
    scale = _scalar("alpha", alpha)
    if not isinstance(geography, Geography):
        raise TypeError("geography must be a Geography instance")

    t_count = rain.size
    p = np.empty(t_count + 1, dtype=np.float64)
    p[0] = initial_p
    u, residual = base_u.copy(), np.zeros(t_count, dtype=np.float64)
    wet_before = np.empty(t_count, dtype=np.float64)
    wet_after = np.empty(t_count, dtype=np.float64)
    fatigue_before = np.empty(t_count, dtype=np.float64)
    fatigue_after = np.empty(t_count, dtype=np.float64)

    for t in range(t_count):
        wet_before[t], fatigue_before[t] = wet, fatigue
        # Smooth positive gust exceedance; threshold 12 m/s is an illustrative
        # synthetic constant, not an empirical damage threshold.
        gust_excess = np.logaddexp(0.0, 4.0 * (gust[t] - 12.0)) / 4.0
        wind_load = -np.expm1(-np.square(gust_excess / 10.0))
        exposure_proxy = (
            0.20 + 0.50 * geography.canopy_proxy
            + 0.30 * geography.terrain_exposure
        )
        # A prior wetness state changes the response to a *later* wind.
        # The 0.15 center permits signed correction to an informed W+Cin host.
        correction = scale * np.tanh(
            exposure_proxy * wind_load * (wet - 0.15) * (1.0 + fatigue)
        )
        if residual_open and scale != 0.0:
            residual[t] = correction
            # |correction| <= 1 means the rate remains in [0,1] without clip.
            u[t] = base_u[t] + correction * base_u[t] * (1.0 - base_u[t])

        p[t + 1] = p[t] + u[t] * (1.0 - p[t]) - base_r[t] * p[t]

        wet_inflow = -np.expm1(
            -0.06 * rain[t] * (0.40 + 0.60 * geography.poor_drainage)
        )
        wet_decay = 0.02 + 0.18 * (1.0 - geography.poor_drainage)
        wet = wet + wet_inflow * (1.0 - wet) - wet_decay * wet
        fatigue_inflow = -np.expm1(-0.12 * gust_excess / 10.0)
        fatigue = fatigue + fatigue_inflow * (1.0 - fatigue) - 0.08 * fatigue
        wet_after[t], fatigue_after[t] = wet, fatigue

    return Trajectory(
        p=p, damage=u, recovery=base_r.copy(), residual=residual,
        wet_before=wet_before, wet_after=wet_after,
        fatigue_before=fatigue_before, fatigue_after=fatigue_after,
    )


if __name__ == "__main__":
    geo = Geography(poor_drainage=0.8, canopy_proxy=0.8, terrain_exposure=0.6)
    base_u = np.full(4, 0.04)
    base_r = np.full(4, 0.03)
    early_rain = rollout([15, 0, 0, 0], [0, 25, 0, 0], geo, base_u, base_r)
    late_rain = rollout([0, 15, 0, 0], [25, 0, 0, 0], geo, base_u, base_r)
    print(f"rain-before-wind final p: {early_rain.p[-1]:.6f}")
    print(f"wind-before-rain final p: {late_rain.p[-1]:.6f}")
