"""NET and ASYM only, built from the audited package core without changing its function classes.

NET:  y_{k+1} = y_k + scale * net(concat(context, step_x_k, y_k / scale))
ASYM: (u, r, stay) = softmax(concat(rate_logits(context, step_x_k), 0))
      y_{k+1} = y_k + u * (1 - y_k) - r * y_k
The package class also defines DIRECT and SR; this round refuses to build them.
"""
from __future__ import annotations
import torch
from core_models_base import TrajectoryModel

KINDS = ('NET', 'ASYM')


def build_model(kind: str, context_dim: int, clock_dim: int, horizon: int,
                parameter_target: int, state_scale: float, source_mean: float) -> TrajectoryModel:
    if kind not in KINDS:
        raise ValueError(f'{kind!r} is not a model of this round; only NET and ASYM exist')
    return TrajectoryModel(kind, context_dim, horizon, clock_dim, parameter_target, state_scale, source_mean)


def n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def clip_output(prediction):
    """Scoring-only output-domain sensitivity. Never fed back into a recursion."""
    return prediction.clip(0.0, 1.0)
