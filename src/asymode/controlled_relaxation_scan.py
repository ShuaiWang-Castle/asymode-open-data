"""Exact first-order adjoint for the I20 controlled-relaxation recurrence.

This changes differentiation/computation only, not the recurrence or controls.
Each factor is Q=(I-S)^-1(I+S), S=eta*(a*b.T-b*a.T), followed by
state=rho*Q4*Q3*Q2*Q1*previous+(1-rho)*deposit. Factor states are saved,
never recovered by inverting a dissipative time update. Shared plane gradients
sum across both batch and time. Higher-order differentiation is not supported.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch.autograd.function import once_differentiable


@torch.jit.script
def _cayley(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
            eta: torch.Tensor, aa: torch.Tensor, bb: torch.Tensor,
            ab: torch.Tensor) -> torch.Tensor:
    ax = (a * x).sum(-1)
    bx = (b * x).sum(-1)
    det = 1.0 + eta.square() * (aa * bb - ab.square())
    s1 = ((1.0 + eta * ab) * bx - eta * bb * ax) / det
    s2 = (eta * aa * bx + (1.0 - eta * ab) * ax) / det
    return x + (2.0 * eta).unsqueeze(-1) * (a * s1.unsqueeze(-1) - b * s2.unsqueeze(-1))


@torch.jit.script
def _forward(deposit: torch.Tensor, rho: torch.Tensor, eta: torch.Tensor,
             a: torch.Tensor, b: torch.Tensor,
             initial: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    batch, steps, modes, dim = deposit.shape
    planes = a.shape[1]
    # Time-major factor cache gives contiguous BxMxD tensors inside the loop.
    factors = deposit.new_empty((steps, planes + 1, batch, modes, dim))
    states = torch.empty_like(deposit)
    aa, bb, ab = a.square().sum(-1), b.square().sum(-1), (a * b).sum(-1)
    state = initial
    for t in range(steps):
        factors[t, 0].copy_(state)
        for r in range(planes):
            state = _cayley(state, a[:, r], b[:, r], eta[:, t, :, r],
                            aa[:, r], bb[:, r], ab[:, r])
            factors[t, r + 1].copy_(state)
        keep = rho[:, t].unsqueeze(-1)
        state = keep * state + (1.0 - keep) * deposit[:, t]
        states[:, t].copy_(state)
    return states, factors


@torch.jit.script
def _backward(grad_states: torch.Tensor, deposit: torch.Tensor, rho: torch.Tensor,
              eta: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
              factors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                              torch.Tensor, torch.Tensor, torch.Tensor]:
    steps = deposit.shape[1]
    planes = a.shape[1]
    gd, grho, geta = torch.empty_like(deposit), torch.empty_like(rho), torch.empty_like(eta)
    ga, gb = torch.zeros_like(a), torch.zeros_like(b)
    carry = torch.zeros_like(deposit[:, 0])
    aa, bb, ab = a.square().sum(-1), b.square().sum(-1), (a * b).sum(-1)
    for rev_t in range(steps):
        t = steps - 1 - rev_t
        total = carry + grad_states[:, t]
        keep = rho[:, t].unsqueeze(-1)
        grho[:, t].copy_((total * (factors[t, planes] - deposit[:, t])).sum(-1))
        gd[:, t].copy_((1.0 - keep) * total)
        gy = keep * total
        for rev_r in range(planes):
            r = planes - 1 - rev_r
            ar, br, et = a[:, r], b[:, r], eta[:, t, :, r]
            gx = _cayley(gy, ar, br, -et, aa[:, r], bb[:, r], ab[:, r])
            # Q^T*gy=2*(I+S)^-1*gy-gy, hence z=(gy+gx)/2.
            # This applies the transpose Jacobian to a gradient, not an inverse
            # reconstruction of the saved forward state.
            z = 0.5 * (gy + gx)
            v = factors[t, r] + factors[t, r + 1]
            av, bv = (ar * v).sum(-1), (br * v).sum(-1)
            za, zb = (z * ar).sum(-1), (z * br).sum(-1)
            geta[:, t, :, r].copy_(za * bv - zb * av)
            scale = et.unsqueeze(-1)
            ga[:, r].add_((scale * (z * bv.unsqueeze(-1) - v * zb.unsqueeze(-1))).sum(0))
            gb[:, r].add_((scale * (v * za.unsqueeze(-1) - z * av.unsqueeze(-1))).sum(0))
            gy = gx
        carry = gy
    return gd, grho, geta, ga, gb, carry


class _ControlledSequence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, deposit, rho, eta, a, b, initial):
        states, factors = _forward(deposit, rho, eta, a, b, initial)
        ctx.save_for_backward(deposit, rho, eta, a, b, factors)
        return states

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_states):
        return _backward(grad_states, *ctx.saved_tensors)


def controlled_sequence_adjoint(deposit: torch.Tensor, rho: torch.Tensor, eta: torch.Tensor,
                                a: torch.Tensor, b: torch.Tensor,
                                initial: torch.Tensor | None = None) -> torch.Tensor:
    """Return states [B,T,M,D]; exact first-order gradients for every input.

    I20 uses deposit[B,T,4,8], rho[B,T,4], eta[B,T,4,4], a/b[4,4,8].
    Other positive dimensions are accepted for synthetic mathematical checks.
    ``a`` and ``b`` are already bounded vectors; their bounding transform stays
    outside this function and receives these gradients through normal autograd.
    initial=None means an independent zero state; a supplied [B,M,D] initial
    state also receives its full gradient. The function performs no clipping,
    detaching, parameter updates, random draws, or changes in precision.

    Cached factors require B*T*M*D*(R+1) scalar elements, approximately 17.7 MB
    for B=128,T=216,M=4,D=8,R=4 in float32, in addition to inputs and outputs.
    Non-reentrant checkpointing outside this function may rematerialize that
    cache; higher-order derivatives must use the ordinary-autograd reference.
    """
    if deposit.ndim != 4 or any(size < 1 for size in deposit.shape):
        raise ValueError('deposit must have nonempty [B,T,M,D] dimensions')
    batch, steps, modes, dim = deposit.shape
    if (a.ndim != 3 or a.shape[0] != modes or a.shape[2] != dim or a.shape[1] < 1
            or b.shape != a.shape or rho.shape != (batch, steps, modes)
            or eta.shape != (batch, steps, modes, a.shape[1])):
        raise ValueError('Inconsistent controlled-relaxation input dimensions')
    if deposit.dtype not in (torch.float32, torch.float64):
        raise TypeError('The verified adjoint supports float32 and float64')
    tensors = (rho, eta, a, b) + (() if initial is None else (initial,))
    if any(x.dtype != deposit.dtype or x.device != deposit.device for x in tensors):
        raise ValueError('All inputs must share dtype and device')
    if initial is None:
        initial = torch.zeros_like(deposit[:, 0])
    elif initial.shape != (batch, modes, dim):
        raise ValueError('initial must have shape [B,M,D]')
    return _ControlledSequence.apply(deposit, rho, eta, a, b, initial)


# Explicitly named alias for integrations that keep the reference separately.
controlled_sequence = controlled_sequence_adjoint
