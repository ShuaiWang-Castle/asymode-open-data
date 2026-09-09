"""Locked, small forecasting models. No target state or direction labels enter forward.
This is a tested model core, not a completed campaign runner.
"""
from __future__ import annotations
import math
import torch
from torch import nn


class ResidualMLP(nn.Module):
    """Two nonlinear hidden layers plus a learned linear input-output skip."""
    def __init__(self, din: int, width: int, dout: int):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(din, width), nn.SiLU(),
                                  nn.Linear(width, width), nn.SiLU(), nn.Linear(width, dout))
        self.linear = nn.Linear(din, dout, bias=False)
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.body[-1].weight, std=1e-3)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.body(x)


def count_for_width(kind: str, context_dim: int, clock_dim: int, horizon: int, width: int) -> int:
    din = context_dim + (1 if kind == 'DIRECT' else clock_dim + (1 if kind == 'NET' else 0))
    dout = horizon if kind == 'DIRECT' else (2 if kind == 'ASYM' else 1)
    return din * width + width + width * width + width + width * dout + dout + din * dout


def match_width(kind: str, context_dim: int, clock_dim: int, horizon: int, target: int) -> int:
    return min(range(2, 1025), key=lambda w: (abs(count_for_width(kind, context_dim, clock_dim, horizon, w) - target), w))


class TrajectoryModel(nn.Module):
    """DIRECT, NET, ASYM or SR. c excludes current y0; physical update uses raw states.

    c: [B, context_dim], history/scale + group one-hot + all KNOWN future clocks.
    y0: [B], current raw state; known_clock: [B,H,clock_dim].
    No forecast-period state observation is accepted by this interface.
    """
    def __init__(self, kind: str, context_dim: int, horizon: int = 24,
                 clock_dim: int = 5, parameter_target: int = 32768,
                 state_scale: float = .01, source_mean: float = .01,
                 signed_init: float = -.05):
        super().__init__()
        if kind not in {'DIRECT', 'NET', 'ASYM', 'SR'}:
            raise ValueError(kind)
        if state_scale <= 0 or not math.isfinite(state_scale):
            raise ValueError('state_scale must be positive and finite')
        self.kind, self.horizon, self.clock_dim = kind, horizon, clock_dim
        self.context_dim = context_dim
        self.register_buffer('state_scale', torch.tensor(float(state_scale)))
        din = context_dim + (1 if kind == 'DIRECT' else clock_dim + (1 if kind == 'NET' else 0))
        dout = horizon if kind == 'DIRECT' else (2 if kind == 'ASYM' else 1)
        width = match_width(kind, context_dim, clock_dim, horizon, parameter_target)
        self.net = ResidualMLP(din, width, dout)
        self.width, self.parameter_target = width, parameter_target
        if kind == 'ASYM':
            # An interior, net-state-only equilibrium initialization. No starts/ends.
            pi = min(max(float(source_mean), 1e-4), 1-1e-4)
            u, r = .05*pi, .05*(1-pi)
            with torch.no_grad():
                self.net.body[-1].bias.copy_(torch.tensor([math.log(u/.95), math.log(r/.95)]))
        elif kind == 'SR':
            if not -1 < signed_init < 1 or signed_init == 0:
                raise ValueError('Nonzero signed initialization inside (-1,1) required')
            nn.init.constant_(self.net.body[-1].bias, math.atanh(signed_init))

    def schedule(self, c: torch.Tensor, known_clock: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.kind not in {'ASYM', 'SR'}:
            raise ValueError('No native source-pool rates for DIRECT/NET')
        x = torch.cat([c[:, None, :].expand(-1, self.horizon, -1), known_clock], dim=-1)
        q = self.net(x)
        if self.kind == 'ASYM':
            weights = torch.softmax(torch.cat([q, torch.zeros_like(q[..., :1])], dim=-1), dim=-1)
            return weights[..., 0], weights[..., 1]
        s = torch.tanh(q[..., 0])
        return torch.relu(s), torch.relu(-s)

    def forward(self, c: torch.Tensor, y0: torch.Tensor, known_clock: torch.Tensor) -> torch.Tensor:
        if c.ndim != 2 or y0.shape != (c.shape[0],):
            raise ValueError('Expected c[B,D], y0[B]')
        if known_clock.shape != (c.shape[0], self.horizon, self.clock_dim):
            raise ValueError('Bad known_clock shape')
        if self.kind == 'DIRECT':
            # One network call, all H outputs jointly; NOT a one-step model.
            return y0[:, None] + self.state_scale*self.net(torch.cat([c, y0[:, None]/self.state_scale], dim=-1))
        y, predictions = y0, []
        if self.kind in {'ASYM', 'SR'}:
            u, r = self.schedule(c, known_clock)
        for k in range(self.horizon):
            if self.kind == 'NET':
                f = self.net(torch.cat([c, known_clock[:, k], y[:, None]/self.state_scale], dim=-1)).squeeze(-1)
                y = y + self.state_scale*f
            else:
                y = y + u[:, k]*(1-y) - r[:, k]*y
            predictions.append(y)
        return torch.stack(predictions, dim=1)


def path_loss(prediction: torch.Tensor, target: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError('Prediction and full H-step target must match')
    return ((prediction-target)/scale).square().mean()


def one_step_ablation_loss(model: TrajectoryModel, c: torch.Tensor, true_states: torch.Tensor,
                           known_clock: torch.Tensor) -> torch.Tensor:
    """Named ASYM_STEP_ONLY ablation only. True states used HERE, never in inference."""
    if model.kind != 'ASYM' or true_states.shape[1] != model.horizon+1:
        raise ValueError('This ablation is ASYM only and requires H+1 states')
    u, r = model.schedule(c, known_clock)
    y = true_states[:, :-1]
    teacher_one_step = y + u*(1-y) - r*y
    return path_loss(teacher_one_step, true_states[:, 1:], model.state_scale)
