"""Local mechanism layer: the recurrence adjoint, bounded parameters, and the zero start of the arm."""
import numpy as np
import torch

from asymode.geo_mech import ATTR, MECH, NODE_CH, LocalMechanisms, linear_recurrence, linear_recurrence_loop, wet_bulb
from asymode.asym_host import AsymODE


def test_linear_recurrence_matches_loop():
    torch.manual_seed(0)
    x = torch.randn(3, 4, 30, dtype=torch.float64, requires_grad=True)
    g = torch.rand(3, 4, 30, dtype=torch.float64, requires_grad=True)
    w = torch.randn(3, 4, 30, dtype=torch.float64)
    a = (linear_recurrence(x, g) * w).sum()
    gx, gg = torch.autograd.grad(a, (x, g))
    b = (linear_recurrence_loop(x, g) * w).sum()
    hx, hg = torch.autograd.grad(b, (x, g))
    assert torch.allclose(a, b) and torch.allclose(gx, hx) and torch.allclose(gg, hg)


def test_wet_bulb_reference():
    # Stull (2011): T = 20 C, RH = 50 % gives Tw = 13.7 C
    assert abs(float(wet_bulb(torch.tensor(20.0), torch.tensor(50.0))) - 13.7) < 0.1


def _toy(B=5, K=4, T=216, seed=0):
    rng = np.random.default_rng(seed)
    w = np.zeros((B, K, T, len(NODE_CH)), np.float32)
    w[..., 0] = rng.normal(0, 5, (B, K, T)); w[..., 1] = w[..., 0] - np.abs(rng.normal(2, 1, (B, K, T)))
    w[..., 2] = np.abs(rng.normal(0, 1, (B, K, T))); w[..., 3] = np.abs(rng.normal(12, 6, (B, K, T)))
    w[..., 4] = np.abs(rng.normal(500, 400, (B, K, T)))
    a = np.zeros((B, K, len(ATTR)), np.float32)
    a[..., 0] = rng.normal(0, 200, (B, K)); a[..., 2] = rng.normal(0, 10, (B, K)); a[..., 4] = rng.uniform(0, 90, (B, K))
    a[..., 7] = rng.uniform(0, 1, (B, K))
    r = rng.dirichlet(np.ones(K), B).astype(np.float32)
    return torch.from_numpy(w), torch.from_numpy(a), torch.from_numpy(r)


def test_mechanisms_shape_and_gradients():
    w, a, r = _toy()
    m = LocalMechanisms()
    lam = m(w, a, r)
    assert lam.shape == (5, 216, len(MECH)) and torch.isfinite(lam).all() and (lam[:, :72] == 0).all()
    lam.sum().backward()
    for n, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), n


def test_mechanism_arm_starts_at_its_base():
    w, a, r = _toy()
    torch.manual_seed(3)
    b = dict(xu=torch.randn(5, 216, 7), xr=torch.randn(5, 216, 6), xo=torch.randn(5, 216, 3),
             y0=torch.rand(5) * 0.1, ctx=torch.randn(5, 2), nw=w, na=a, nr=r)
    torch.manual_seed(11); base = AsymODE(7, 6, 3); base.attach_context_input(2)
    torch.manual_seed(11); arm = AsymODE(7, 6, 3); arm.attach_context_input(2)
    arm.attach_mechanisms(LocalMechanisms(), len(MECH)); arm.set_mechanism_scale(b)
    with torch.no_grad():
        base.ctx_in.weight.normal_(); arm.ctx_in.weight.copy_(base.ctx_in.weight)
    assert torch.equal(base(b)["P"], arm(b)["P"])


def test_hazard_arm_starts_at_its_base_and_gets_gradient():
    torch.manual_seed(3)
    b = dict(xu=torch.randn(5, 216, 7), xr=torch.randn(5, 216, 6), xo=torch.randn(5, 216, 3),
             y0=torch.rand(5) * 0.1, ctx=torch.randn(5, 2), phi=torch.rand(5, 144, 4))
    torch.manual_seed(11); base = AsymODE(7, 6, 3); base.attach_context_input(2)
    torch.manual_seed(11); arm = AsymODE(7, 6, 3); arm.attach_context_input(2); arm.attach_hazard(4)
    assert torch.equal(base(b)["P"], arm(b)["P"])
    y = torch.rand(5, 144) * 0.5
    ((arm(b)["P"] - y) ** 2).mean().backward()
    assert arm.haz_beta.grad is not None and (arm.haz_beta.grad.abs() > 0).all()


def test_hazard_slots_start_at_base_and_get_gradient():
    torch.manual_seed(3)
    b = dict(xu=torch.randn(5, 216, 7), xr=torch.randn(5, 216, 6), xo=torch.randn(5, 216, 3),
             y0=torch.rand(5) * 0.1, ctx=torch.randn(5, 2), phi=torch.rand(5, 144, 6))
    torch.manual_seed(11); base = AsymODE(7, 6, 3); base.attach_context_input(2)
    torch.manual_seed(11); arm = AsymODE(7, 6, 3); arm.attach_context_input(2); arm.attach_hazard(6)
    arm.attach_hazard_slots([0, 1, 2], [3, 4], 2)
    assert torch.equal(base(b)["P"], arm(b)["P"])
    ((arm(b)["P"] - torch.rand(5, 144) * 0.5) ** 2).mean().backward()
    assert (arm.haz_a.grad.abs() > 0).all()


def test_expanded_damage_inputs_start_at_base():
    torch.manual_seed(3)
    b = dict(xu=torch.randn(5, 216, 7), xr=torch.randn(5, 216, 6), xo=torch.randn(5, 216, 3),
             y0=torch.rand(5) * 0.1, ctx=torch.randn(5, 2))
    b2 = dict(b, xu=torch.cat([b["xu"], torch.rand(5, 216, 3)], -1))
    torch.manual_seed(11); base = AsymODE(7, 6, 3); base.attach_context_input(2)
    torch.manual_seed(11); arm = AsymODE(7, 6, 3); arm.expand_damage_inputs(3); arm.attach_context_input(2)
    assert torch.equal(base(b)["P"], arm(b2)["P"])
