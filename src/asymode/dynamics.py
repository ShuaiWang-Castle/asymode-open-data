"""Two-rate compartmental dynamics on a bounded fraction.

The state y_t in [0, 1] is the fraction of tracked customers currently without
power in a county. It is driven by two *independent, separately parameterised*
rates: an interruption rate u_t that converts served customers to interrupted
ones, and a restoration rate r_t that converts them back.

    y_{t+1} = clip( y_t + u_t * (1 - y_t) - r_t * y_t ,  0, 1 )

The inflow term is proportional to the *served* fraction (1 - y_t) only. This is
the modelling commitment the paper is about, and it is what separates the model
from the epidemic/diffusion family, whose inflow is proportional to y_t(1 - y_t)
and therefore cannot leave the state y = 0. Both forms are implemented here so
the difference can be measured rather than asserted; see `InflowForm`.

Three axes of asymmetry are exposed independently, so each can be switched off
on its own in an ablation:

  * dynamical -- InflowForm.SUSCEPTIBLE vs InflowForm.TRANSMISSION
  * input     -- `idx_u` / `idx_r` select different driver channels per rate
  * capacity  -- `hidden_u` / `hidden_r` size the two rate networks separately

Rates are bounded by construction, `cap * sigmoid(logit)`. Every structural
pathway that a variant adds is summed into the *logit*, never into the rate.
Composing in rate space would require `clamp(rate + bump, 0, cap)`, and the
clamp zeroes the gradient wherever it binds; a pathway that drifts negative then
pins the rate at zero and can never be learned back on. In logit space the bound
holds automatically and no gradient is destroyed.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn


class InflowForm(str, enum.Enum):
    """Which pool the interruption term is proportional to."""

    SUSCEPTIBLE = "susceptible"      # u * (1 - y)          -- this work
    TRANSMISSION = "transmission"    # u * y * (1 - y)      -- epidemic/diffusion family
    # Steelmanned epidemic form. A learnable, state-independent seed eps >= 0 is
    # added to the infectious pool so the arm is *able* to leave y = 0 at all:
    # u * (y + eps) * (1 - y). Without it the comparison is decided by algebra
    # rather than by fit, which no reviewer should accept. Note the tension the
    # arm is then under: eps large enough to ignite from zero also swamps the
    # y-dependence that makes the form epidemic in the first place.
    TRANSMISSION_SEED = "transmission_seed"


class RateNet(nn.Module):
    """A rate in [0, cap], parameterised in logit space.

    `hidden == 0` degenerates to a logistic GLM with directly readable
    per-channel coefficients. That is the low-capacity end of the capacity
    asymmetry probe, not a fallback.
    """

    def __init__(self, d_in: int, cap: float, hidden: int = 32, depth: int = 2):
        super().__init__()
        if hidden == 0:
            layers: list[nn.Module] = [nn.Linear(d_in, 1)]
        else:
            layers = [nn.Linear(d_in, hidden), nn.ReLU()]
            for _ in range(depth - 1):
                layers += [nn.Linear(hidden, hidden), nn.ReLU()]
            layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)
        self.cap = float(cap)

    def logit(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cap * torch.sigmoid(self.logit(x))


@dataclass
class TwoRateConfig:
    d_in: int
    cap_u: float
    cap_r: float
    hidden_u: int = 32
    hidden_r: int = 32
    inflow: InflowForm = InflowForm.SUSCEPTIBLE
    # None means "every channel". Lists select a per-rate subset of the drivers,
    # which is how the input-asymmetry ablation is switched off (pass the same
    # list to both) or on (pass different lists).
    idx_u: list[int] | None = None
    idx_r: list[int] | None = None
    # Feed the current state into each rate network as an extra channel. Off by
    # default: the dynamics already carry y, and letting a rate read y lets it
    # imitate the other inflow form, which would confound the ablation.
    state_in_u: bool = False
    state_in_r: bool = False
    clip_state: tuple[float, float] = (0.0, 1.0)
    seed_init: float = 1e-3          # initial eps for TRANSMISSION_SEED
    # Initial rates, in rate units, not logits. A rate network initialised at
    # cap/2 starts three orders of magnitude above the base rate of this process,
    # and the susceptible arm then saturates the state before it can learn its way
    # back down. Callers should set these from the data; see `calibrate_init`.
    u_init: float | None = None
    r_init: float | None = None
    tags: dict = field(default_factory=dict)


class TwoRateODE(nn.Module):
    """Open-loop rollout of the two-rate dynamics under an exogenous driver."""

    def __init__(self, cfg: TwoRateConfig):
        super().__init__()
        self.cfg = cfg
        d_u = (len(cfg.idx_u) if cfg.idx_u is not None else cfg.d_in) + int(cfg.state_in_u)
        d_r = (len(cfg.idx_r) if cfg.idx_r is not None else cfg.d_in) + int(cfg.state_in_r)
        self.phi_u = RateNet(d_u, cap=cfg.cap_u, hidden=cfg.hidden_u)
        self.phi_r = RateNet(d_r, cap=cfg.cap_r, hidden=cfg.hidden_r)
        self.register_buffer("_iu", torch.tensor(cfg.idx_u if cfg.idx_u is not None else [], dtype=torch.long))
        self.register_buffer("_ir", torch.tensor(cfg.idx_r if cfg.idx_r is not None else [], dtype=torch.long))
        for net, val in ((self.phi_u, cfg.u_init), (self.phi_r, cfg.r_init)):
            if val is None:
                continue
            q = min(max(float(val) / net.cap, 1e-9), 1 - 1e-9)
            with torch.no_grad():
                net.net[-1].bias.fill_(float(np.log(q / (1 - q))))
        if cfg.inflow is InflowForm.TRANSMISSION_SEED:
            # softplus keeps the seed non-negative without a clamp, so the
            # gradient survives at eps -> 0.
            inv = float(torch.log(torch.expm1(torch.tensor(max(cfg.seed_init, 1e-6)))))
            self.seed_raw = nn.Parameter(torch.tensor(inv))

    @property
    def seed(self) -> torch.Tensor | None:
        if self.cfg.inflow is not InflowForm.TRANSMISSION_SEED:
            return None
        return torch.nn.functional.softplus(self.seed_raw)

    def _slice(self, x: torch.Tensor, idx: torch.Tensor, use: bool) -> torch.Tensor:
        return x.index_select(-1, idx) if use else x

    def rates(self, x_t: torch.Tensor, y_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Rates at one step. x_t is (B, d_in); y_t is (B,)."""
        cfg = self.cfg
        xu = self._slice(x_t, self._iu, cfg.idx_u is not None)
        xr = self._slice(x_t, self._ir, cfg.idx_r is not None)
        if cfg.state_in_u:
            xu = torch.cat([xu, y_t.unsqueeze(-1)], dim=-1)
        if cfg.state_in_r:
            xr = torch.cat([xr, y_t.unsqueeze(-1)], dim=-1)
        return self.phi_u(xu), self.phi_r(xr)

    def forward(self, y0: torch.Tensor, drivers: torch.Tensor) -> torch.Tensor:
        """y0 is (B,); drivers is (B, T, d_in). Returns (B, T) states y_1..y_T."""
        lo, hi = self.cfg.clip_state
        y = y0
        out = []
        for t in range(drivers.shape[1]):
            u, r = self.rates(drivers[:, t], y)
            if self.cfg.inflow is InflowForm.TRANSMISSION:
                u = u * y
            elif self.cfg.inflow is InflowForm.TRANSMISSION_SEED:
                u = u * (y + torch.nn.functional.softplus(self.seed_raw))
            y = torch.clamp(y + u * (1.0 - y) - r * y, lo, hi)
            out.append(y)
        return torch.stack(out, dim=1)

    def rate_curves(self, drivers: torch.Tensor, y_ref: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        """Rates evaluated on a driver grid at a fixed state, for recovery plots."""
        with torch.no_grad():
            y = torch.full((drivers.shape[0],), y_ref, device=drivers.device)
            return self.rates(drivers, y)


def rollout_mse(model: TwoRateODE, y0, drivers, y_true, mask=None) -> torch.Tensor:
    """Plain mean squared error on the state trajectory. No derived index."""
    pred = model(y0, drivers)
    se = (pred - y_true) ** 2
    if mask is not None:
        return (se * mask).sum() / mask.sum().clamp_min(1.0)
    return se.mean()


def calibrate_init(y: np.ndarray, mask: np.ndarray, inflow: InflowForm,
                   seed_init: float = 1e-3) -> tuple[float, float]:
    """Initial rates matching the observed one-step flows, per arm.

    One rule, applied identically to every arm: set each rate so that the arm's
    *initial inflow and outflow terms* reproduce the average one-step rise and fall
    seen in the training data. Because the arms multiply the interruption rate by
    different factors, the same target flow implies different initial rates -- which
    is the point. Giving every arm the same initial *rate* would hand an advantage
    to whichever arm's multiplier happens to be near one, and that is an artefact
    of parameterisation, not evidence about dynamics.
    """
    m = mask[:, :-1] & mask[:, 1:]
    if not m.any():
        return 1e-4, 1e-2
    d = (y[:, 1:] - y[:, :-1])[m]
    y_bar = float(np.clip(y[mask].mean(), 1e-6, 1.0))
    up = float(np.clip(np.maximum(d, 0).mean(), 1e-9, None))
    dn = float(np.clip(np.maximum(-d, 0).mean(), 1e-9, None))
    # inflow: u*(1-y) ~ u ; u*y*(1-y) ~ u*y_bar ; u*(y+eps)*(1-y) ~ u*(y_bar+eps)
    if inflow is InflowForm.SUSCEPTIBLE:
        u0 = up
    elif inflow is InflowForm.TRANSMISSION:
        u0 = up / y_bar
    else:
        u0 = up / (y_bar + seed_init)
    r0 = dn / y_bar                      # outflow: r*y ~ r*y_bar, same for all arms
    return float(u0), float(r0)
