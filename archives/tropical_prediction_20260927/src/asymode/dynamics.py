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
    # One signed net rate from a single network: n = cap * tanh(f(x)), y <- y + n.
    # This is the fully unstructured end of the comparison. It drops *two* things
    # at once -- the separate parameterisation of the two directions, and the
    # state-dependent scaling -- so it cannot on its own attribute a difference to
    # either. NET_SCALED is the intermediate that isolates the first: one signed
    # network, but the inflow still acts on the served pool and the outflow on the
    # interrupted one.
    #
    # The structural commitment the single-rate forms give up is concurrency. Two
    # non-negative rates are both active at every step, so a county can be losing
    # customers and restoring them at the same time; one signed rate forces the
    # two directions to be mutually exclusive. That is the claim these arms test.
    NET = "net"
    NET_SCALED = "net_scaled"


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


class NetRate(nn.Module):
    """A signed rate in (-cap, cap), from one network.

    `tanh` rather than a pair of sigmoids: the whole point of the arm is that a
    single function has to serve both directions, so it must be able to change
    sign. Bounded by construction like the two-rate form, and for the same reason
    -- an unbounded net rate would differ from the arms it is compared against in
    a second way that has nothing to do with structure.
    """

    def __init__(self, d_in: int, cap: float, hidden: int = 32, depth: int = 2):
        super().__init__()
        self.inner = RateNet(d_in, cap=1.0, hidden=hidden, depth=depth)
        self.cap = float(cap)

    def logit(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner.logit(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cap * torch.tanh(self.inner.logit(x))

    def calibrate(self, n_init: float) -> None:
        q = float(np.clip(n_init / self.cap, -0.999, 0.999))
        with torch.no_grad():
            self.inner.net[-1].bias.fill_(float(np.arctanh(q)))


class GatedRate(nn.Module):
    """A rate composed from two independent failure channels.

        rate = cap * [ 1 - (1 - p_bkg)(1 - p_pulse) ]
        p_pulse = sigmoid(gate_logit) * sigmoid(pulse_logit)
        p_bkg   = sigmoid(bkg_logit)

    The composition is the probability that *either* channel fires, which is the
    standard way two independent hazards combine and is additive in the
    complementary-log link. Two properties matter here. The bound is `cap`, the
    same ceiling every ungated rate has, so a gated arm cannot win by being
    allowed to emit a larger rate than the arm it is compared against. And no
    clamp appears anywhere: composing in rate space and truncating would zero the
    gradient wherever the truncation binds, which is the failure this module
    exists to avoid.

    The gate is *multiplicative* on purpose. An additive gate -- extra inputs to
    the pulse logit -- would remove the pathology described below, but it would
    also stop being a gate, and the product of two sigmoids is precisely the
    structure under test.

    That pathology: the pulse network's gradient is proportional to the gate, and
    the gate's own gradient is proportional to the pulse. A gate that reaches zero
    is therefore absorbing -- both branches stop learning together and neither can
    reopen. The gate is initialised half-open for that reason, and the fraction of
    closed gates is accumulated during the rollout rather than inspected
    afterwards, because for the input-width hypothesis that fraction is not a
    hygiene check but the dependent variable.

    The background term is a scalar: "always on" in the sense of not gated. A
    county-varying background is a different variant and is not this one.
    """

    def __init__(self, d_pulse: int, d_gate: int, cap: float,
                 hidden: int = 32, hidden_gate: int = 0, depth: int = 2):
        super().__init__()
        self.cap = float(cap)
        # cap=1 on both: these produce probabilities, and the cap is applied once
        # at the composition rather than twice inside it.
        self.pulse = RateNet(d_pulse, cap=1.0, hidden=hidden, depth=depth)
        self.gate = RateNet(d_gate, cap=1.0, hidden=hidden_gate, depth=depth)
        self.bkg_logit = nn.Parameter(torch.zeros(()))
        self.reset_gate_stats()

    def reset_gate_stats(self) -> None:
        self._g_n = 0
        self._g_sum = 0.0
        self._g_sq = 0.0
        self._g_closed = 0
        self._g_open = 0

    def gate_stats(self) -> dict:
        """Mean, spread, and the two saturation fractions.

        The spread is the load-bearing one. The gate's *level* is not identified:
        the rate depends on the product `g * sigmoid(pulse_logit)`, so the pulse's
        bias can absorb any constant factor in the gate, and a gate sitting at its
        initial value is therefore not evidence that the gate is doing nothing.
        Only its variation across inputs distinguishes an active gate from an
        inert one, and reporting the mean alone cannot tell them apart.
        """
        n = max(self._g_n, 1)
        mean = self._g_sum / n
        var = max(self._g_sq / n - mean * mean, 0.0)
        return {"gate_mean": mean, "gate_sd": var ** 0.5,
                "frac_gate_closed": self._g_closed / n,
                "frac_gate_open": self._g_open / n}

    def calibrate(self, u_init: float, pulse_share: float = 0.5) -> None:
        """Solve both biases so the composed rate at initialisation *is* `u_init`.

        This is what makes a gated arm comparable to an ungated one: the two start
        from the same rate and diverge only through structure. Splitting the
        calibrated flow between the channels rather than fixing a bias constant
        also keeps the initialisation tied to the data -- a constant that suits one
        operating point silently misplaces the pulse by orders of magnitude at
        another, and this process runs at a base rate near 1e-4.
        """
        q = min(max(float(u_init) / self.cap, 1e-9), 1 - 1e-9)
        with torch.no_grad():
            self.gate.net[-1].bias.zero_()                       # g = 0.5, half open
            p_pulse = float(pulse_share) * q
            sp = min(max(p_pulse / 0.5, 1e-12), 1 - 1e-12)       # sigma(pulse) * g = p_pulse
            self.pulse.net[-1].bias.fill_(float(np.log(sp / (1 - sp))))
            p_bkg = min(max((q - p_pulse) / (1.0 - p_pulse), 1e-12), 1 - 1e-12)
            self.bkg_logit.fill_(float(np.log(p_bkg / (1 - p_bkg))))

    def forward(self, x_pulse: torch.Tensor, x_gate: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate.logit(x_gate))
        p_pulse = g * torch.sigmoid(self.pulse.logit(x_pulse))
        p_bkg = torch.sigmoid(self.bkg_logit)
        with torch.no_grad():
            self._g_n += g.numel()
            self._g_sum += float(g.sum())
            self._g_sq += float((g * g).sum())
            self._g_closed += int((g < 0.01).sum())
            self._g_open += int((g > 0.99).sum())
        # a + b - ab, algebraically identical to 1 - (1-a)(1-b) but not
        # catastrophically cancelling: at this process's operating point both
        # probabilities are ~1e-3, and the product form loses most of the
        # mantissa in float32 before the subtraction.
        return self.cap * (p_bkg + p_pulse - p_bkg * p_pulse)


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
    # Gated-pulse-plus-background form for the interruption rate. Off by default:
    # every arm that does not name it keeps the plain bounded sigmoid.
    gate_u: bool = False
    idx_gate: list[int] | None = None
    # Held at a logistic GLM. The registered hypothesis is about the gate's
    # *input width*; letting capacity vary alongside would mix two axes into one
    # profile, and neither a flat one nor a peaked one could then be attributed.
    hidden_gate: int = 0
    gate_pulse_share: float = 0.5    # share of the calibrated initial flow given to the pulse
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
        d_g = len(cfg.idx_gate) if cfg.idx_gate is not None else cfg.d_in
        self.is_net = cfg.inflow in (InflowForm.NET, InflowForm.NET_SCALED)
        if self.is_net:
            self.phi_u = NetRate(d_u, cap=cfg.cap_u, hidden=cfg.hidden_u)
            self.phi_r = None            # one network, by construction
        elif cfg.gate_u:
            self.phi_u = GatedRate(d_u, d_g, cap=cfg.cap_u, hidden=cfg.hidden_u,
                                   hidden_gate=cfg.hidden_gate)
        else:
            self.phi_u = RateNet(d_u, cap=cfg.cap_u, hidden=cfg.hidden_u)
        if not self.is_net:
            self.phi_r = RateNet(d_r, cap=cfg.cap_r, hidden=cfg.hidden_r)
        self.register_buffer("_iu", torch.tensor(cfg.idx_u if cfg.idx_u is not None else [], dtype=torch.long))
        self.register_buffer("_ir", torch.tensor(cfg.idx_r if cfg.idx_r is not None else [], dtype=torch.long))
        self.register_buffer("_ig", torch.tensor(cfg.idx_gate if cfg.idx_gate is not None else [], dtype=torch.long))
        if cfg.u_init is not None and self.is_net:
            self.phi_u.calibrate(cfg.u_init)
        elif cfg.u_init is not None:
            if cfg.gate_u:
                self.phi_u.calibrate(cfg.u_init, cfg.gate_pulse_share)
            else:
                q = min(max(float(cfg.u_init) / self.phi_u.cap, 1e-9), 1 - 1e-9)
                with torch.no_grad():
                    self.phi_u.net[-1].bias.fill_(float(np.log(q / (1 - q))))
        if cfg.r_init is not None and not self.is_net:
            q = min(max(float(cfg.r_init) / self.phi_r.cap, 1e-9), 1 - 1e-9)
            with torch.no_grad():
                self.phi_r.net[-1].bias.fill_(float(np.log(q / (1 - q))))
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
        if self.is_net:
            n = self.phi_u(xu)
            return n, torch.zeros_like(n)
        if cfg.gate_u:
            xg = self._slice(x_t, self._ig, cfg.idx_gate is not None)
            return self.phi_u(xu, xg), self.phi_r(xr)
        return self.phi_u(xu), self.phi_r(xr)

    def forward(self, y0: torch.Tensor, drivers: torch.Tensor) -> torch.Tensor:
        """y0 is (B,); drivers is (B, T, d_in). Returns (B, T) states y_1..y_T."""
        lo, hi = self.cfg.clip_state
        y = y0
        out = []
        for t in range(drivers.shape[1]):
            u, r = self.rates(drivers[:, t], y)
            if self.is_net:
                if self.cfg.inflow is InflowForm.NET_SCALED:
                    # positive net flow acts on the served pool, negative on the
                    # interrupted one -- the scaling the two-rate form has, without
                    # the separate parameterisation.
                    u = torch.where(u > 0, u * (1.0 - y), u * y)
                y = torch.clamp(y + u, lo, hi)
                out.append(y)
                continue
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
    if inflow in (InflowForm.NET, InflowForm.NET_SCALED):
        # A single signed rate cannot reproduce the mean rise *and* the mean fall
        # -- that inability is the arm's defining property, not a defect in its
        # initialisation. The same rule still applies as far as it can: match the
        # arm's initial *net* flow to the observed mean one-step change.
        net = up - dn
        if inflow is InflowForm.NET_SCALED and net < 0:
            net = net / y_bar            # the negative branch acts on y, not (1-y)
        return float(net), 0.0
    if inflow is InflowForm.SUSCEPTIBLE:
        u0 = up
    elif inflow is InflowForm.TRANSMISSION:
        u0 = up / y_bar
    else:
        u0 = up / (y_bar + seed_init)
    r0 = dn / y_bar                      # outflow: r*y ~ r*y_bar, same for all arms
    return float(u0), float(r0)
