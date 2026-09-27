"""Spatio-temporal GCRK (asymode.gcrk._ResponseST) on synthetic inputs only.

  * the hand-written adjoint matches finite differences (gradcheck, float64) for f, lam, a and both couplings;
  * the forward matches the plain autograd recurrence;
  * with zero coupling it is the temporal kernel exactly;
  * the state keeps GCRK's unit bound under the Markov mixing of neighbour states;
  * a layer with the coupling attached equals plain GCRK at step 0 (all three new scalars zero).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from asymode.gcrk import (GCRKLayer, response_sequence, response_sequence_st,  # noqa: E402
                          response_sequence_st_loop)


def problem(B=5, T=9, D=6, k=3, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    f = 0.2 * torch.randn(B, T, D, generator=g, dtype=dtype)
    lam = 0.02 + 0.5 * torch.rand(B, D, generator=g, dtype=dtype)
    a = 0.2 * torch.randn(B, D, generator=g, dtype=dtype)
    nbr = torch.randint(0, B, (B, k), generator=g)
    cw = 0.4 * torch.rand(B, k, generator=g, dtype=dtype) / k
    cu = 0.4 * torch.rand(B, T, k, generator=g, dtype=dtype) / k
    return f, lam, a, cw, cu, nbr


def test_adjoint_matches_finite_differences():
    f, lam, a, cw, cu, nbr = problem()
    args = [x.requires_grad_(True) for x in (f, lam, a, cw, cu)] + [nbr]
    assert torch.autograd.gradcheck(response_sequence_st, tuple(args))


def test_forward_matches_loop():
    f, lam, a, cw, cu, nbr = problem(B=7, T=30, D=8, k=4, seed=1)
    assert torch.allclose(response_sequence_st(f, lam, a, cw, cu, nbr), response_sequence_st_loop(f, lam, a, cw, cu, nbr),
                          atol=1e-13)


def test_zero_coupling_is_temporal_kernel():
    f, lam, a, cw, cu, nbr = problem(B=4, T=20, D=5, seed=2)
    e = response_sequence_st(f, lam, a, torch.zeros_like(cw), torch.zeros_like(cu), nbr)
    assert torch.equal(e, response_sequence(f, lam, a))


def test_unit_bound_holds_with_coupling():
    g = torch.Generator().manual_seed(3)
    B, T, D, k = 12, 216, 16, 5
    d = torch.randn(B, T, D, generator=g, dtype=torch.float64)
    d = 0.999 * d / d.norm(dim=-1, keepdim=True)                      # |d| < 1, the largest deposits the gate allows
    lam = 0.0105 + 0.64 * torch.rand(B, D, generator=g, dtype=torch.float64)
    a = torch.randn(B, D, generator=g, dtype=torch.float64)
    a = 0.5 * a / torch.sqrt(1 + a.square().sum(-1, keepdim=True))
    nu = lam.amin(-1, keepdim=True)
    nbr = torch.randint(0, B, (B, k), generator=g)
    cw = torch.full((B, k), 0.5 / k, dtype=torch.float64)
    cu = torch.full((B, T, k), 0.5 / k, dtype=torch.float64)          # the largest shares allowed
    e = response_sequence_st(nu[:, None] * d, lam, a, cw, cu, nbr)
    assert float(e.norm(dim=-1).max()) < 1.0


def test_layer_with_coupling_equals_gcrk_at_start():
    gen = torch.Generator().manual_seed(5)
    lin = nn.Linear(32, 32).double()
    center = torch.tanh(torch.randn(7, 20, generator=gen, dtype=torch.float64) / 3).mean(0)
    plain, st = GCRKLayer(lin, center).double(), GCRKLayer(lin, center).double()
    st.attach_space()
    h_fit = torch.relu(torch.randn(7, 216, 32, generator=gen, dtype=torch.float64))
    h = torch.relu(torch.randn(6, 216, 32, generator=gen, dtype=torch.float64))
    geo = torch.randn(6, 20, generator=gen, dtype=torch.float64)
    space = dict(nbr=torch.randint(0, 6, (6, 4), generator=gen), wd=torch.rand(6, 4, generator=gen, dtype=torch.float64),
                 up=torch.rand(6, 216, 4, generator=gen, dtype=torch.float64) / 4)
    for lay in (plain, st):
        lay.calibrate_(h_fit, 0); lay.eval(); lay.training_step.fill_(300)
        with torch.no_grad():
            lay.alpha.fill_(0.7)
    assert torch.allclose(plain(h, geo), st(h, geo, space=space), atol=1e-14)
    with torch.no_grad():
        st.kappa_s.fill_(0.3); st.kappa_a.fill_(0.2)
    assert not torch.allclose(plain(h, geo), st(h, geo, space=space), atol=1e-6)
