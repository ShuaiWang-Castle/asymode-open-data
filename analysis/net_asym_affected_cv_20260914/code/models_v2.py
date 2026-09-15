"""NET and split ASYM for the affected-county CV round (PI decisions of 2026-09-14).

NET (function class unchanged): y[k+1] = y[k] + scale * f(c, x[k], y[k] / scale), with f a residual MLP.
  Structural candidates: 2, 3 or 4 hidden layers.
ASYM (split): two independent residual MLPs, a damage network D(c, x[k]) and a recovery network R(c, x[k]);
  (u, r, stay) = softmax(D, R, 0) and y[k+1] = y[k] + u (1 - y[k]) - r y[k]. The rates never read the
  recursive state; the state enters only through the source pools. Structural candidates: damage:recovery
  parameter ratio 1:1, 2:1 or 3:1.

Both models, every candidate: total parameters within 1% of the common target. Both start near persistence.
NET's output layer starts near zero (as in the audited core). ASYM's rate biases start at u = eps * pi,
r = eps * (1 - pi) and stay = 1 - eps with eps = 0.002, so an untrained ASYM relaxes towards pi by less than
5% over 24 hours (the audited core used eps = 0.05).
"""
from __future__ import annotations
import math
import torch
from torch import nn

KINDS = ('NET', 'ASYM')
NET_DEPTHS = {'d2': 2, 'd3': 3, 'd4': 4}
ASYM_RATIOS = {'r1': 1, 'r2': 2, 'r3': 3}
ASYM_INIT_EPS = 0.002
PARAM_TOL = 0.01


def mlp_params(din: int, width: int, dout: int, n_hidden: int) -> int:
    return din * width + width + (n_hidden - 1) * (width * width + width) + width * dout + dout + din * dout


class ResidualMLPn(nn.Module):
    """n_hidden nonlinear layers plus a learned linear input-output skip (the audited core has n_hidden = 2)."""

    def __init__(self, din: int, width: int, dout: int, n_hidden: int):
        super().__init__()
        layers = [nn.Linear(din, width), nn.SiLU()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(width, width), nn.SiLU()]
        layers.append(nn.Linear(width, dout))
        self.body = nn.Sequential(*layers)
        self.linear = nn.Linear(din, dout, bias=False)
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.body[-1].weight, std=1e-3)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.body(x)


def match_width(din: int, dout: int, n_hidden: int, target: int) -> int:
    w = min(range(2, 1025), key=lambda v: (abs(mlp_params(din, v, dout, n_hidden) - target), v))
    if abs(mlp_params(din, w, dout, n_hidden) / target - 1) > PARAM_TOL:
        raise ValueError(f'no width within {PARAM_TOL:.0%} of {target} for n_hidden={n_hidden}')
    return w


def match_split(din: int, ratio: float, target: int) -> tuple[int, int]:
    """Damage and recovery widths: total within PARAM_TOL of target, parameter ratio as close to `ratio` as possible."""
    best = None
    for wd in range(2, 513):
        pd_ = mlp_params(din, wd, 1, 2)
        if pd_ > target * (1 + PARAM_TOL):
            break
        for wr in range(2, 513):
            tot = pd_ + mlp_params(din, wr, 1, 2)
            if tot > target * (1 + PARAM_TOL):
                break
            if tot < target * (1 - PARAM_TOL):
                continue
            key = (round(abs(math.log(pd_ / (tot - pd_) / ratio)), 9), abs(tot - target), -wd, wr)
            if best is None or key < best[0]:
                best = (key, wd, wr)
    if best is None:
        raise ValueError(f'no damage/recovery split within {PARAM_TOL:.0%} of {target}')
    return best[1], best[2]


def n_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


class NetModel(nn.Module):
    def __init__(self, context_dim: int, clock_dim: int, horizon: int, structure: str, target: int, state_scale: float):
        super().__init__()
        self.kind, self.structure, self.horizon, self.clock_dim = 'NET', structure, horizon, clock_dim
        self.n_hidden = NET_DEPTHS[structure]
        din = context_dim + clock_dim + 1
        self.width = match_width(din, 1, self.n_hidden, target)
        self.net = ResidualMLPn(din, self.width, 1, self.n_hidden)
        self.register_buffer('state_scale', torch.tensor(float(state_scale)))

    def widths(self): return {'net': self.width, 'n_hidden': self.n_hidden}
    def params_by_part(self): return {'net': n_params(self.net)}

    def forward(self, c: torch.Tensor, y0: torch.Tensor, known_clock: torch.Tensor) -> torch.Tensor:
        if c.ndim != 2 or y0.shape != (c.shape[0],) or known_clock.shape != (c.shape[0], self.horizon, self.clock_dim):
            raise ValueError('expected c[B,D], y0[B], known_clock[B,H,clock_dim]')
        y, out = y0, []
        for k in range(self.horizon):
            f = self.net(torch.cat([c, known_clock[:, k], y[:, None] / self.state_scale], dim=-1)).squeeze(-1)
            y = y + self.state_scale * f
            out.append(y)
        return torch.stack(out, dim=1)


class AsymSplitModel(nn.Module):
    def __init__(self, context_dim: int, clock_dim: int, horizon: int, structure: str, target: int,
                 state_scale: float, source_mean: float, eps: float = ASYM_INIT_EPS):
        super().__init__()
        self.kind, self.structure, self.horizon, self.clock_dim = 'ASYM', structure, horizon, clock_dim
        din = context_dim + clock_dim
        self.width_damage, self.width_recovery = match_split(din, ASYM_RATIOS[structure], target)
        self.damage = ResidualMLPn(din, self.width_damage, 1, 2)
        self.recovery = ResidualMLPn(din, self.width_recovery, 1, 2)
        self.register_buffer('state_scale', torch.tensor(float(state_scale)))
        pi = min(max(float(source_mean), 1e-4), 1 - 1e-4)
        u, r = eps * pi, eps * (1 - pi); stay = 1.0 - u - r
        with torch.no_grad():
            self.damage.body[-1].bias.fill_(math.log(u / stay))
            self.recovery.body[-1].bias.fill_(math.log(r / stay))

    def widths(self): return {'damage': self.width_damage, 'recovery': self.width_recovery}
    def params_by_part(self): return {'damage': n_params(self.damage), 'recovery': n_params(self.recovery)}

    def rates(self, c: torch.Tensor, known_clock: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([c[:, None, :].expand(-1, self.horizon, -1), known_clock], dim=-1)
        logits = torch.cat([self.damage(x), self.recovery(x), torch.zeros_like(x[..., :1])], dim=-1)
        w = torch.softmax(logits, dim=-1)
        return w[..., 0], w[..., 1]

    def forward(self, c: torch.Tensor, y0: torch.Tensor, known_clock: torch.Tensor) -> torch.Tensor:
        if c.ndim != 2 or y0.shape != (c.shape[0],) or known_clock.shape != (c.shape[0], self.horizon, self.clock_dim):
            raise ValueError('expected c[B,D], y0[B], known_clock[B,H,clock_dim]')
        u, r = self.rates(c, known_clock)
        y, out = y0, []
        for k in range(self.horizon):
            y = y + u[:, k] * (1 - y) - r[:, k] * y
            out.append(y)
        return torch.stack(out, dim=1)


def build_model(kind: str, structure: str, context_dim: int, clock_dim: int, horizon: int, target: int,
                state_scale: float, source_mean: float) -> nn.Module:
    if kind == 'NET':
        return NetModel(context_dim, clock_dim, horizon, structure, target, state_scale)
    if kind == 'ASYM':
        return AsymSplitModel(context_dim, clock_dim, horizon, structure, target, state_scale, source_mean)
    raise ValueError(f'{kind!r} is not a model of this round; only NET and ASYM exist')
