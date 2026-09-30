"""Algebra and actual layer checks for the D09 write-only structural comparison."""
from pathlib import Path
import sys
import math
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from d09_write import (IncrementalWriteRelaxationLayer, radius_map, spectral_cap,
                       WEATHER_LIPSCHITZ, GEOGRAPHY_FEATURE_LIPSCHITZ)
from asymode.controlled_relaxation import ControlledRelaxationLayer, controlled_sequence

torch.set_num_threads(2)


def layer(cls=IncrementalWriteRelaxationLayer):
    generator = torch.Generator().manual_seed(456)
    geo = torch.randn(40, 40, generator=generator, dtype=torch.float64)
    original = nn.Linear(32, 32, dtype=torch.float64)
    return cls(original, geo, county_ids=[str(i) for i in range(40)],
               checkpoint_steps=0, use_adjoint=False, geo_checkpoint=False)


@pytest.mark.parametrize('radius', [1., 8.])
def test_radial_norm_and_increment_bound(radius):
    torch.manual_seed(17)
    x, z = torch.randn(400, 32, dtype=torch.float64) * 10, torch.randn(400, 32, dtype=torch.float64) * 10
    y = radius_map(x, radius)
    assert torch.linalg.vector_norm(y, dim=-1).max() < radius
    assert torch.all(torch.linalg.vector_norm(y-radius_map(z, radius), dim=-1) <=
                     torch.linalg.vector_norm(x-z, dim=-1) + 1e-12)
    assert torch.equal(radius_map(torch.zeros_like(x), radius), torch.zeros_like(x))


def test_radius_supported_ball_radial_derivative():
    x = torch.zeros(64, dtype=torch.float64);x[0] = 8.
    jac = torch.func.jacrev(lambda v: radius_map(v, 8.))(x)
    eig = torch.linalg.eigvalsh(jac)
    assert torch.allclose(eig.min(), torch.tensor(2**-1.5, dtype=x.dtype))
    assert torch.allclose(eig.max(), torch.tensor(2**-.5, dtype=x.dtype))


def test_spectral_cap_and_gradient():
    matrix = (torch.randn(8, 11, dtype=torch.float64) * .2).requires_grad_()
    assert torch.linalg.matrix_norm(spectral_cap(matrix), ord=2) <= 1+1e-12
    assert torch.autograd.gradcheck(spectral_cap, (matrix,), fast_mode=True)


def test_exact_zero_weather_even_with_saturated_geography():
    k = layer()
    with torch.no_grad():
        k.head_bias[:32].fill_(100)
        k.fusion_bias.fill_(-80)
    e = torch.randn(3, 72, dtype=torch.float64) * 50
    fg, hg, anchor = k._geographic_terms(e)
    controls = k._controls(torch.zeros(3, 7, 64, dtype=torch.float64), fg, hg, anchor)
    assert torch.equal(controls['deposit'], torch.zeros(3, 7, 4, 8, dtype=torch.float64))


def test_same_parameters_and_nonwrite_controls():
    k, old = layer(), layer(ControlledRelaxationLayer)
    old.load_state_dict(k.state_dict())
    assert sum(p.numel() for p in k.parameters()) == sum(p.numel() for p in old.parameters())
    assert set(dict(k.named_parameters())) == set(dict(old.named_parameters()))
    e, u = torch.randn(3, 72, dtype=torch.float64), torch.randn(3, 5, 64, dtype=torch.float64)
    args = k._geographic_terms(e)
    new_controls, old_controls = k._controls(u, *args), old._controls(u, *args)
    for key in ['tau', 'rho', 'eta', 'gain']:
        assert torch.equal(new_controls[key], old_controls[key])


@pytest.mark.parametrize('parameter_scale', [.1, 2., 20.])
def test_both_increment_bounds_and_unit_modes(parameter_scale):
    k = layer()
    with torch.no_grad():
        for name,p in k.named_parameters():
            if name.startswith(('fusion_', 'head_')): p.mul_(parameter_scale)
    torch.manual_seed(40)
    e = torch.randn(32, 72, dtype=torch.float64)
    de = torch.randn_like(e) * .03
    u = torch.randn(32, 5, 64, dtype=torch.float64) * 5
    du = torch.randn_like(u) * .04
    d = k._controls(u, *k._geographic_terms(e))['deposit']
    assert torch.linalg.vector_norm(d, dim=-1).max() < 1
    dw = k._controls(u+du, *k._geographic_terms(e))['deposit'] - d
    dg = k._controls(u, *k._geographic_terms(e+de))['deposit'] - d
    assert torch.all(torch.linalg.vector_norm(dw.flatten(-2), dim=-1) <=
                     WEATHER_LIPSCHITZ * torch.linalg.vector_norm(du, dim=-1) + 1e-12)
    assert torch.all(torch.linalg.vector_norm(dg.flatten(-2), dim=-1) <=
                     GEOGRAPHY_FEATURE_LIPSCHITZ * torch.linalg.vector_norm(de, dim=-1)[:,None] + 1e-12)


def test_geographic_offset_does_not_limit_one_weather_direction():
    k = layer()
    with torch.no_grad():
        k.head_weather[:32].zero_();k.head_weather[0,0] = 1
        k.head_fusion[:32].zero_();k.head_bias[:32].fill_(100)
    e = torch.zeros(1,72,dtype=torch.float64)
    u = torch.zeros(1,2,64,dtype=torch.float64);u[0,0,0]=2;u[0,1,0]=-2
    d = k._controls(u,*k._geographic_terms(e))['deposit']
    assert d[0,0,0,0] > .1 and d[0,1,0,0] < -.1
    assert torch.allclose(d[:,0],-d[:,1])


def test_write_parameter_gradients_finite():
    k = layer();e = torch.randn(3,72,dtype=torch.float64)
    u = torch.randn(3,6,64,dtype=torch.float64)
    loss = k._controls(u,*k._geographic_terms(e))['deposit'].square().sum()
    loss.backward()
    for name,p in k.named_parameters():
        if name.startswith(('fusion_', 'head_')):
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert k.head_geo.grad[:32].abs().sum()>0


def test_bounded_state_and_initial_host_identity():
    k = layer();h = torch.randn(3,12,32,dtype=torch.float64);g=torch.randn(3,40,dtype=torch.float64)
    k.calibrate_(h, 900);k.training_step.fill_(900);k.eval()
    out, detail = k(h,g,diagnostics=True)
    assert torch.equal(out, torch.nn.functional.linear(h,k.weight,k.bias))
    assert torch.linalg.vector_norm(detail['state'],dim=-1).max() <= 1+1e-12
    k.alpha.data.fill_(.2)
    opened = k(h,g)
    opened.square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in k.parameters())


@pytest.mark.parametrize('checkpoint_steps', [0, 3])
def test_full_new_write_adjoint_matches_reference(checkpoint_steps):
    reference, production = layer(), layer()
    production.load_state_dict(reference.state_dict())
    production.use_adjoint = True
    production.checkpoint_steps = checkpoint_steps
    h = torch.randn(2,7,32,dtype=torch.float64)
    g = torch.randn(2,40,dtype=torch.float64)
    probe = torch.randn_like(h)
    for k in (reference, production):
        k.calibrate_(h,900);k.training_step.fill_(900);k.eval();k.alpha.data.fill_(.2)
    outputs, gradients = [], []
    for k in (reference, production):
        hi,gi=h.clone().requires_grad_(),g.clone().requires_grad_()
        out=k(hi,gi);((out*probe).sum()+.1*out.square().sum()).backward()
        outputs.append(out.detach());gradients.append(dict(h=hi.grad,g=gi.grad,
            **{name:p.grad for name,p in k.named_parameters()}))
    assert torch.allclose(*outputs,atol=1e-11,rtol=1e-10)
    for name in gradients[0]:
        a,b=gradients[0][name],gradients[1][name]
        assert (a is None)==(b is None),name
        if a is not None:assert torch.allclose(a,b,atol=1e-9,rtol=1e-7),name
