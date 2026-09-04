#!/usr/bin/env python3
"""Reference implementation for the proposed rescue model.

This file is intentionally isolated from the historical experiment zoo. It
implements only:

1. exact constant-class least-squares initialization;
2. one shared causal weather encoder;
3. a nested signed-direction/concurrency parameterization;
4. a bounded transition that preserves [0, 1] without clipping.

It is a reference specification, not yet the canonical training harness.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class ConstantClassFit:
    u: float
    r: float
    signed: float
    loss: float
    branch: str


def _weighted_sse(delta: np.ndarray, pred: np.ndarray, weight: np.ndarray) -> float:
    return float(np.sum(weight * (delta - pred) ** 2))


def _clip_ratio(num: float, den: float, lo: float, hi: float) -> float:
    if den <= 0:
        return lo
    return float(np.clip(num / den, lo, hi))


def fit_constant_classes(
    y: np.ndarray,
    delta: np.ndarray,
    *,
    weight: np.ndarray | None = None,
    cap: float = 0.25,
) -> tuple[ConstantClassFit, ConstantClassFit]:
    """Fit the exact bounded constant two-flow and one-flow classes.

    The two-flow class is

        delta_hat = U * (1-y) - R * y,  0 <= U,R <= cap.

    The one-flow class is the union of the two coordinate rays. The function
    returns `(two_flow_fit, one_flow_fit)`.
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    delta = np.asarray(delta, dtype=float).reshape(-1)
    if y.shape != delta.shape:
        raise ValueError("y and delta must have the same shape")
    if weight is None:
        weight = np.ones_like(y)
    weight = np.asarray(weight, dtype=float).reshape(-1)
    if weight.shape != y.shape or np.any(weight < 0):
        raise ValueError("weight must be nonnegative and match y")
    keep = np.isfinite(y) & np.isfinite(delta) & np.isfinite(weight) & (weight > 0)
    if not np.any(keep):
        raise ValueError("no finite positive-weight transitions")
    y, delta, weight = y[keep], delta[keep], weight[keep]
    if np.any((y < 0) | (y > 1)):
        raise ValueError("state must lie in [0,1]")

    x_u = 1.0 - y
    x_r = -y

    def loss(u: float, r: float) -> float:
        return _weighted_sse(delta, u * x_u + r * x_r, weight)

    candidates: list[tuple[float, float, str]] = [(0.0, 0.0, "zero")]

    # Interior unconstrained least-squares candidate.
    X = np.column_stack([x_u, x_r])
    sw = np.sqrt(weight)
    beta, *_ = np.linalg.lstsq(X * sw[:, None], delta * sw, rcond=None)
    if np.all(beta >= 0.0) and np.all(beta <= cap):
        candidates.append((float(beta[0]), float(beta[1]), "interior"))

    # Exact minimizers on all four box edges. Clipping the free coordinate also
    # includes the four corners.
    for u_fixed, label in ((0.0, "u=0"), (cap, "u=cap")):
        residual = delta - u_fixed * x_u
        r = _clip_ratio(np.sum(weight * x_r * residual),
                        np.sum(weight * x_r * x_r), 0.0, cap)
        candidates.append((u_fixed, r, label))
    for r_fixed, label in ((0.0, "r=0"), (cap, "r=cap")):
        residual = delta - r_fixed * x_r
        u = _clip_ratio(np.sum(weight * x_u * residual),
                        np.sum(weight * x_u * x_u), 0.0, cap)
        candidates.append((u, r_fixed, label))

    u2, r2, branch2 = min(candidates, key=lambda z: loss(z[0], z[1]))
    two = ConstantClassFit(
        u=u2,
        r=r2,
        signed=u2 - r2,
        loss=loss(u2, r2),
        branch=branch2,
    )

    # Exact one-flow union of rays.
    u1 = _clip_ratio(np.sum(weight * x_u * delta),
                     np.sum(weight * x_u * x_u), 0.0, cap)
    r1 = _clip_ratio(np.sum(weight * x_r * delta),
                     np.sum(weight * x_r * x_r), 0.0, cap)
    one_candidates = [
        (0.0, 0.0, "zero"),
        (u1, 0.0, "interruption"),
        (0.0, r1, "restoration"),
    ]
    u1, r1, branch1 = min(one_candidates, key=lambda z: loss(z[0], z[1]))
    one = ConstantClassFit(
        u=u1,
        r=r1,
        signed=u1 - r1,
        loss=loss(u1, r1),
        branch=branch1,
    )
    return two, one


class SharedContextEncoder(nn.Module):
    """Causal weather-history encoder shared by both model classes.

    `weather_sequence` contains L history steps followed by T forecast-weather
    steps. GRU output at forecast step t depends only on weather up to that step.
    The current outage state is deliberately absent from this encoder.
    """

    def __init__(
        self,
        weather_dim: int,
        static_dim: int,
        *,
        gru_hidden: int = 16,
        mlp_hidden: int = 32,
    ) -> None:
        super().__init__()
        self.weather_dim = int(weather_dim)
        self.static_dim = int(static_dim)
        self.gru = nn.GRU(weather_dim, gru_hidden, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(gru_hidden + weather_dim + static_dim, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.SiLU(),
        )
        self.output_dim = mlp_hidden

    def forward(
        self,
        weather_sequence: torch.Tensor,
        statics: torch.Tensor,
        *,
        history_length: int,
    ) -> torch.Tensor:
        if weather_sequence.ndim != 3:
            raise ValueError("weather_sequence must have shape (B,L+T,D)")
        if statics.ndim != 2:
            raise ValueError("statics must have shape (B,S)")
        if history_length < 1 or history_length >= weather_sequence.shape[1]:
            raise ValueError("history_length must leave at least one forecast step")
        h, _ = self.gru(weather_sequence)
        h_future = h[:, history_length:, :]
        x_future = weather_sequence[:, history_length:, :]
        static_future = statics[:, None, :].expand(-1, h_future.shape[1], -1)
        return self.mlp(torch.cat([h_future, x_future, static_future], dim=-1))


class NestedFlowHeads(nn.Module):
    """Nested one-flow/two-flow heads on a common representation.

    The two-flow model uses

        U = relu(s) + c,
        R = relu(-s) + c.

    Setting `learn_concurrency=False` fixes c=0 and recovers the exact one-flow
    class. Both rates are bounded by `cap`, and U+R <= 2*cap.
    """

    def __init__(self, d_in: int, *, cap: float = 0.25,
                 learn_concurrency: bool = True) -> None:
        super().__init__()
        if not (0.0 < cap <= 0.5):
            raise ValueError("cap must lie in (0,0.5]")
        self.cap = float(cap)
        self.learn_concurrency = bool(learn_concurrency)
        self.signed_head = nn.Linear(d_in, 1)
        self.concurrent_head = nn.Linear(d_in, 1)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s = self.cap * torch.tanh(self.signed_head(h).squeeze(-1))
        if self.learn_concurrency:
            room = torch.clamp(self.cap - torch.abs(s), min=0.0)
            c = room * torch.sigmoid(self.concurrent_head(h).squeeze(-1))
        else:
            c = torch.zeros_like(s)
        u = torch.relu(s) + c
        r = torch.relu(-s) + c
        return u, r, s, c

    def initialize_constant(self, fit: ConstantClassFit) -> None:
        """Set zero output weights and biases to a constant class optimum."""
        if max(fit.u, fit.r) > self.cap + 1e-10:
            raise ValueError("constant fit exceeds head cap")
        s0 = float(fit.u - fit.r)
        q_s = np.clip(s0 / self.cap, -0.999999, 0.999999)
        with torch.no_grad():
            self.signed_head.weight.zero_()
            self.signed_head.bias.fill_(float(np.arctanh(q_s)))
            self.concurrent_head.weight.zero_()
            if self.learn_concurrency:
                c0 = float(min(fit.u, fit.r))
                room = max(self.cap - abs(s0), 1e-12)
                q_c = float(np.clip(c0 / room, 1e-6, 1.0 - 1e-6))
                self.concurrent_head.bias.fill_(float(np.log(q_c / (1.0 - q_c))))
            else:
                self.concurrent_head.bias.zero_()


class NestedFlowModel(nn.Module):
    """Causal neural rollout for the nested flow classes."""

    def __init__(
        self,
        weather_dim: int,
        static_dim: int,
        *,
        learn_concurrency: bool,
        cap: float = 0.25,
        gru_hidden: int = 16,
        mlp_hidden: int = 32,
    ) -> None:
        super().__init__()
        self.encoder = SharedContextEncoder(
            weather_dim,
            static_dim,
            gru_hidden=gru_hidden,
            mlp_hidden=mlp_hidden,
        )
        self.heads = NestedFlowHeads(
            self.encoder.output_dim,
            cap=cap,
            learn_concurrency=learn_concurrency,
        )

    def forward(
        self,
        y0: torch.Tensor,
        weather_sequence: torch.Tensor,
        statics: torch.Tensor,
        *,
        history_length: int,
        return_rates: bool = False,
    ):
        context = self.encoder(
            weather_sequence,
            statics,
            history_length=history_length,
        )
        y = y0
        states, rates = [], []
        for t in range(context.shape[1]):
            u, r, s, c = self.heads(context[:, t, :])
            y = y + u * (1.0 - y) - r * y
            states.append(y)
            rates.append(torch.stack([u, r, s, c], dim=-1))
        state = torch.stack(states, dim=1)
        if return_rates:
            return state, torch.stack(rates, dim=1)
        return state


def _self_test() -> None:
    rng = np.random.default_rng(7)

    # The nested coordinates exactly recover arbitrary bounded flow pairs.
    for _ in range(10_000):
        u = rng.uniform(0.0, 0.25)
        r = rng.uniform(0.0, 0.25)
        s = u - r
        c = min(u, r)
        assert abs(max(s, 0.0) + c - u) < 1e-12
        assert abs(max(-s, 0.0) + c - r) < 1e-12
        assert c <= 0.25 - abs(s) + 1e-12

    # Exact constant fits.
    y = np.linspace(0.0, 0.8, 200)
    u_true, r_true = 0.08, 0.035
    delta = u_true * (1.0 - y) - r_true * y
    two, one = fit_constant_classes(y, delta, cap=0.25)
    assert abs(two.u - u_true) < 1e-10
    assert abs(two.r - r_true) < 1e-10
    assert two.loss < 1e-18
    assert one.loss > two.loss

    # State preservation without clipping.
    torch.manual_seed(3)
    model = NestedFlowModel(
        weather_dim=4,
        static_dim=3,
        learn_concurrency=True,
    )
    B, L, T = 32, 24, 48
    weather = torch.randn(B, L + T, 4)
    static = torch.randn(B, 3)
    y0 = torch.rand(B)
    pred, rate = model(y0, weather, static, history_length=L, return_rates=True)
    assert torch.all(pred >= -1e-7) and torch.all(pred <= 1.0 + 1e-7)
    assert torch.all(rate[..., 0] >= 0) and torch.all(rate[..., 0] <= 0.25 + 1e-7)
    assert torch.all(rate[..., 1] >= 0) and torch.all(rate[..., 1] <= 0.25 + 1e-7)

    # The one-flow restriction has zero concurrency at every step.
    one_model = NestedFlowModel(
        weather_dim=4,
        static_dim=3,
        learn_concurrency=False,
    )
    _, one_rate = one_model(y0, weather, static, history_length=L, return_rates=True)
    assert torch.count_nonzero(one_rate[..., 3]) == 0
    assert torch.all((one_rate[..., 0] == 0) | (one_rate[..., 1] == 0))

    print("PASS: nested-flow reference self-test")


if __name__ == "__main__":
    _self_test()
