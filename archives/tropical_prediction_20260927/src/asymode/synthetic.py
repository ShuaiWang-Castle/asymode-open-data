"""Synthetic trajectories with known rate functions, for identifiability study.

The generator draws exogenous drivers, evaluates two *known* rate functions on
them, and rolls the same dynamics the model assumes. Because the truth is known
in closed form, recovery error is measurable rather than inferred, and the
regimes where recovery must fail can be produced on purpose.

Driver channels (all standardised to roughly unit scale):
  0  hazard   -- non-negative, spiky; drives interruption only
  1  daylight -- diurnal cycle; drives restoration only
  2  nuisance -- correlated noise that drives neither, to detect false attribution

Identifiability, stated exactly. At a single step the data supply

    dy = u(x) (1 - y) - r(x) y,

one equation in the two unknowns u(x), r(x). Two steps sharing the same driver x
but sitting at different states y1 != y2 give a 2x2 system whose determinant is
(y1 - y2). The split is therefore identified only through variation of the state
under comparable drivers, and the conditioning degrades as |y1 - y2| -> 0. Two
saturation regimes follow: at y == 0 the restoration term vanishes and r is
unidentified; at y == 1 the interruption term vanishes and u is unidentified.
`make_dataset` exposes `pulse_scale`, which moves trajectories along exactly this
axis, so the predicted degradation can be traced out empirically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .dynamics import InflowForm


@dataclass
class TrueRates:
    """Ground-truth rate functions. Channel selection encodes input asymmetry."""

    cap_u: float = 0.30
    cap_r: float = 0.15
    a_u: float = 3.0      # hazard slope (channel 0)
    b_u: float = -3.5     # hazard intercept: near-zero interruption in calm weather
    a_r: float = 1.5      # daylight slope (channel 1)
    b_r: float = -0.5

    def u(self, x: np.ndarray | torch.Tensor):
        h = x[..., 0]
        return self.cap_u / (1.0 + np.exp(-(self.a_u * h + self.b_u)))

    def r(self, x: np.ndarray | torch.Tensor):
        c = x[..., 1]
        return self.cap_r / (1.0 + np.exp(-(self.a_r * c + self.b_r)))


def make_drivers(n: int, T: int, rng: np.random.Generator, pulse_scale: float = 1.0,
                 n_pulses: tuple[int, int] = (1, 3)) -> np.ndarray:
    """(n, T, 3) drivers: spiky hazard, diurnal daylight, correlated nuisance."""
    hazard = np.zeros((n, T), dtype=np.float64)
    for i in range(n):
        for _ in range(rng.integers(n_pulses[0], n_pulses[1] + 1)):
            t0 = rng.integers(0, T)
            width = rng.integers(4, 18)
            amp = pulse_scale * rng.gamma(shape=2.5, scale=0.45)
            tt = np.arange(T)
            hazard[i] += amp * np.exp(-0.5 * ((tt - t0) / width) ** 2)
    hazard += 0.05 * np.abs(rng.normal(size=(n, T)))

    hours = np.arange(T)[None, :] + rng.integers(0, 24, size=(n, 1))
    daylight = np.sin(2 * np.pi * (hours - 6) / 24.0)

    # Nuisance shares the diurnal envelope but carries independent noise, so a
    # model can be tempted to attribute restoration to it.
    nuisance = 0.6 * daylight + 0.8 * rng.normal(size=(n, T))
    return np.stack([hazard, daylight, nuisance], axis=-1).astype(np.float32)


def rollout_truth(drivers: np.ndarray, y0: np.ndarray, rates: TrueRates,
                  inflow: InflowForm = InflowForm.SUSCEPTIBLE,
                  kappa: float = 1.0) -> np.ndarray:
    """Roll the known dynamics forward. Returns (n, T) states y_1..y_T.

    `kappa` bends the served-pool exponent to (1 - y)**kappa. kappa == 1 is the
    form this work assumes; kappa != 1 is a *neutral* generator that neither the
    susceptible nor the transmission arm implements exactly, so a comparison run
    on it cannot be dismissed as the proposed model grading its own homework.
    """
    n, T, _ = drivers.shape
    y = y0.astype(np.float64).copy()
    out = np.empty((n, T), dtype=np.float64)
    for t in range(T):
        x = drivers[:, t]
        u, r = rates.u(x), rates.r(x)
        if inflow is InflowForm.TRANSMISSION:
            u = u * y
        y = np.clip(y + u * (1.0 - y) ** kappa - r * y, 0.0, 1.0)
        out[:, t] = y
    return out


@dataclass
class SynthDataset:
    drivers: np.ndarray      # (n, T, 3)
    y: np.ndarray            # (n, T)
    y0: np.ndarray           # (n,)
    rates: TrueRates
    pulse_scale: float
    obs_noise: float
    kappa: float = 1.0

    @property
    def state_spread(self) -> float:
        """Mean within-trajectory std of the state.

        This is the empirical stand-in for |y1 - y2| in the identifiability
        argument: it is how much the state moves while the model is being asked
        to separate the two rates.
        """
        return float(np.std(self.y, axis=1).mean())

    def tensors(self, device="cpu"):
        t = lambda a, d=torch.float32: torch.tensor(a, dtype=d, device=device)
        return t(self.y0), t(self.drivers), t(self.y)


def make_dataset(n: int = 512, T: int = 96, seed: int = 0, pulse_scale: float = 1.0,
                 obs_noise: float = 0.0, y0_mode: str = "zero",
                 rates: TrueRates | None = None,
                 inflow: InflowForm = InflowForm.SUSCEPTIBLE,
                 kappa: float = 1.0,
                 n_pulses: tuple[int, int] = (1, 3)) -> SynthDataset:
    rng = np.random.default_rng(seed)
    rates = rates or TrueRates()
    drivers = make_drivers(n, T, rng, pulse_scale=pulse_scale, n_pulses=n_pulses)
    if y0_mode == "zero":
        y0 = np.zeros(n)
    elif y0_mode == "uniform":
        y0 = rng.uniform(0.0, 0.4, size=n)
    elif y0_mode == "mixed":              # half start dark, half start lit
        y0 = np.where(rng.random(n) < 0.5, 0.0, rng.uniform(0.02, 0.4, size=n))
    else:
        raise ValueError(f"unknown y0_mode: {y0_mode}")
    y = rollout_truth(drivers, y0, rates, inflow=inflow, kappa=kappa)
    if obs_noise > 0:
        y = np.clip(y + rng.normal(scale=obs_noise, size=y.shape), 0.0, 1.0)
    return SynthDataset(drivers, y.astype(np.float32), y0.astype(np.float32),
                        rates, pulse_scale, obs_noise, kappa)
