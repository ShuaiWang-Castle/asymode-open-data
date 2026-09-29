"""Synthetic implementation checks for I20; no panel data or training fits."""
import copy
import itertools
import math
import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asymode.controlled_relaxation import (
    ControlledRelaxationLayer, all_subset_kernel, bounded_vector,
    bounded_readout, cayley_rank2_apply, controlled_sequence,
    HEAD_SLICES, TAU_INITIAL, LENGTH_FLOOR, ORDER_FLOOR,
)
from asymode.gcrk import GCRKLayer, prefix_reference


@pytest.fixture(autouse=True)
def bounded_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def make_layer(checkpoint_steps=0, dtype=torch.float64):
    gen = torch.Generator().manual_seed(28)
    geo = torch.randn(36, 40, generator=gen, dtype=dtype)
    fit_geo = torch.cat((geo, geo[:4]))
    ids = [f"{i:05d}" for i in range(36)] + [f"{i:05d}" for i in range(4)]
    base = nn.Linear(32, 12, dtype=dtype)
    layer = ControlledRelaxationLayer(base, fit_geo, ids, checkpoint_steps=checkpoint_steps)
    h = torch.randn(5, 9, 32, generator=gen, dtype=dtype)
    layer.calibrate_(h, 0)
    layer.eval()
    return layer, h, geo[:5], base, fit_geo, ids


def test_all_subset_matches_enumeration_and_psd():
    gen = torch.Generator().manual_seed(51)
    x = torch.randn(6, 4, generator=gen, dtype=torch.float64)
    lengths = torch.rand(4, generator=gen, dtype=torch.float64) + 0.3
    masses = torch.rand(4, generator=gen, dtype=torch.float64).softmax(0)
    base = torch.exp(-0.5 * ((x[:, None] - x[None]) / lengths).square())
    expected = torch.zeros(6, 6, dtype=x.dtype)
    for order in range(1, 5):
        terms = [base[..., list(subset)].prod(-1) for subset in itertools.combinations(range(4), order)]
        expected += masses[order - 1] * torch.stack(terms).mean(0)
    actual = all_subset_kernel(x, x, lengths, masses)
    torch.testing.assert_close(actual, expected, atol=2e-15, rtol=2e-15)
    assert torch.linalg.eigvalsh(actual).min() >= -1e-12
    forty = torch.randn(5, 40, generator=gen, dtype=x.dtype)
    result = all_subset_kernel(forty, forty, torch.ones(40, dtype=x.dtype), torch.full((40,), 1 / 40, dtype=x.dtype))
    assert bool((result >= 0).all() and (result <= 1 + 1e-14).all())
    torch.testing.assert_close(result.diag(), torch.ones(5, dtype=x.dtype))


def test_all_subset_gradients():
    gen = torch.Generator().manual_seed(1)
    args = (torch.randn(2, 3, generator=gen, dtype=torch.float64).requires_grad_(),
            torch.randn(2, 3, generator=gen, dtype=torch.float64).requires_grad_(),
            torch.ones(3, dtype=torch.float64, requires_grad=True),
            torch.full((3,), 1 / 3, dtype=torch.float64, requires_grad=True))
    assert torch.autograd.gradcheck(all_subset_kernel, args, fast_mode=True)


def test_rank_two_cayley_dense_equivalence_and_gradients():
    gen = torch.Generator().manual_seed(11)
    x = torch.randn(2, 8, generator=gen, dtype=torch.float64)
    a = bounded_vector(torch.randn(2, 8, generator=gen, dtype=x.dtype))
    b = bounded_vector(torch.randn(2, 8, generator=gen, dtype=x.dtype))
    eta = torch.tensor([0.22, -0.17], dtype=x.dtype)
    j = a[..., :, None] * b[..., None, :] - b[..., :, None] * a[..., None, :]
    eye = torch.eye(8, dtype=x.dtype)
    q = torch.linalg.solve(eye - eta[:, None, None] * j, eye + eta[:, None, None] * j)
    expected = (q @ x[..., None]).squeeze(-1)
    torch.testing.assert_close(cayley_rank2_apply(x, a, b, eta), expected, atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(q.mT @ q, eye.expand_as(q), atol=1e-14, rtol=1e-14)
    det = 1 + eta.square() * (a.square().sum(-1) * b.square().sum(-1) - (a * b).sum(-1).square())
    assert bool((det >= 1).all())
    assert torch.autograd.gradcheck(cayley_rank2_apply, tuple(value.requires_grad_() for value in (x, a, b, eta)), fast_mode=True)


def test_sequence_bounds_contraction_and_constant_equilibrium():
    gen = torch.Generator().manual_seed(19)
    a = bounded_vector(torch.randn(4, 4, 8, generator=gen, dtype=torch.float64))
    b = bounded_vector(torch.randn(4, 4, 8, generator=gen, dtype=torch.float64))
    d = bounded_vector(torch.randn(2, 40, 4, 8, generator=gen, dtype=torch.float64))
    rho = torch.exp(-1 / (2 + 90 * torch.rand(2, 40, 4, generator=gen, dtype=d.dtype)))
    eta = 0.25 * torch.tanh(torch.randn(2, 40, 4, 4, generator=gen, dtype=d.dtype))
    first = bounded_vector(torch.randn(2, 4, 8, generator=gen, dtype=d.dtype))
    second = bounded_vector(torch.randn(2, 4, 8, generator=gen, dtype=d.dtype))
    xs = controlled_sequence(d, rho, eta, a, b, first)
    ys = controlled_sequence(d, rho, eta, a, b, second)
    assert float(xs.norm(dim=-1).max()) <= 1 + 1e-13
    expected_difference = rho.cumprod(1) * (first - second).norm(dim=-1)[:, None]
    torch.testing.assert_close((xs - ys).norm(dim=-1), expected_difference, atol=1e-13, rtol=1e-12)
    # No rotation: closed-form constant-input relaxation at all four time scales.
    constant = d[:, :1].expand(-1, 40, -1, -1)
    rr = torch.exp(-1 / torch.tensor(TAU_INITIAL, dtype=d.dtype))[None, None].expand(2, 40, -1)
    value = controlled_sequence(constant, rr, torch.zeros_like(eta), a, b)
    expected = (1 - rr.cumprod(1))[..., None] * constant
    torch.testing.assert_close(value, expected, atol=1e-13, rtol=1e-12)


def test_fit_only_geography_landmarks_initialization_and_parameter_count():
    layer, _, geo, base, fit, ids = make_layer()
    assert isinstance(layer, GCRKLayer)
    assert layer.weight is base.weight and layer.bias is base.bias
    assert layer.n_new_parameters() == 15209
    center = torch.tanh(fit / 3).mean(0)
    rms = torch.sqrt(0.1 ** 2 + (torch.tanh(fit / 3) - center).square().sum(-1).mean())
    torch.testing.assert_close(layer.geo_center, center)
    torch.testing.assert_close(layer.geo_rms_scale, rms)
    assert layer.fit_metadata["n_unique_counties"] == 36
    assert layer.fit_metadata["repeated_landmarks"] == 0
    order = torch.arange(len(fit) - 1, -1, -1)
    other = ControlledRelaxationLayer(copy.deepcopy(base), fit[order], [ids[i] for i in order])
    assert layer.fit_metadata["landmark_county_ids"] == other.fit_metadata["landmark_county_ids"]
    torch.testing.assert_close(layer.landmarks, other.landmarks)
    assert bool((layer.length_scales() > LENGTH_FLOOR).all())
    assert bool((layer.order_masses() >= ORDER_FLOOR).all())
    torch.testing.assert_close(layer.order_masses().sum(), torch.tensor(1., dtype=geo.dtype))
    control = layer.condition(geo, torch.randn(5, 3, 64, dtype=geo.dtype))
    torch.testing.assert_close(control["tau"], torch.tensor(TAU_INITIAL, dtype=geo.dtype).expand(5, 3, 4))
    for value in (layer.head_geo, layer.head_weather, layer.head_fusion):
        assert torch.count_nonzero(value[HEAD_SLICES["tau"]]) == 0
    bad = fit.clone()
    bad[-1, 0] += 0.01
    with pytest.raises(ValueError, match="inconsistent"):
        ControlledRelaxationLayer(copy.deepcopy(base), bad, ids)
    tiny = ControlledRelaxationLayer(copy.deepcopy(base), fit[:2], ids[:2])
    assert tiny.fit_metadata["repeated_landmarks"] == 30
    constant = fit.clone()
    constant[:, 39] = 7
    fallback = ControlledRelaxationLayer(copy.deepcopy(base), constant, ids)
    assert 39 in fallback.fit_metadata["lengthscale_fallback_columns"]
    torch.testing.assert_close(fallback.length_scales()[39], torch.tensor(1., dtype=geo.dtype))


def test_anchored_zero_value_and_total_geo_parameter_derivatives():
    layer, _, geo, *_ = make_layer()
    geo = geo[:2].clone().requires_grad_()
    u = torch.zeros(2, 4, 64, dtype=geo.dtype, requires_grad=True)
    deposit = layer.condition(geo, u)["deposit"]
    assert torch.count_nonzero(deposit) == 0
    parameters = tuple(layer.parameters())
    grads = torch.autograd.grad(deposit.sum(), (geo, u) + parameters, allow_unused=True)
    assert torch.count_nonzero(grads[0]) == 0
    assert float(grads[1].abs().sum()) > 0
    for grad in grads[2:]:
        assert grad is None or float(grad.abs().max()) < 1e-14


def test_host_recovery_warmup_and_branch_opening():
    layer, h, geo, base, *_ = make_layer()
    layer.train()
    layer.set_drop_override(1)
    expected = F.linear(h, base.weight, base.bias)
    layer.training_step.fill_(0)
    assert torch.equal(layer(h, geo), expected)
    layer(h, geo).square().sum().backward()
    assert float(layer.alpha.grad) == 0
    layer.zero_grad(set_to_none=True)
    layer.training_step.fill_(1)
    assert torch.equal(layer(h, geo), expected)
    layer(h, geo).square().sum().backward()
    assert abs(float(layer.alpha.grad)) > 0
    assert float(layer.head_weather.grad.abs().sum()) == 0
    with torch.no_grad():
        layer.alpha.fill_(0.2)
    layer.zero_grad(set_to_none=True)
    layer(h, geo).square().sum().backward()
    assert float(layer.head_weather.grad.abs().sum()) > 0
    assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in layer.parameters())
    assert torch.equal(layer(h, geo, exit_open=False), expected)


def test_streamed_calibration_matches_full_fit_and_preserves_failure():
    full, h, *_ = make_layer()
    streamed = copy.deepcopy(full)
    full.calibrate_(h, 10)
    streamed.calibrate_chunks_((h[:2], h[2:4], h[4:]), 10)
    for name in ("scale", "level_scale", "departure_scale"):
        torch.testing.assert_close(getattr(full, name), getattr(streamed, name), atol=1e-14, rtol=1e-14)
    assert streamed.last_calibration["n_counties"] == len(h)
    assert streamed.last_calibration["n_values"] == len(h) * (h.shape[1] - 1)
    before = streamed.level_scale.clone()
    invalid = h[:1].clone()
    invalid[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        streamed.calibrate_chunks_((h[:2], invalid), 20)
    torch.testing.assert_close(before, streamed.level_scale, atol=0, rtol=0)
    assert int(streamed.calibration_step) == 10


def test_checkpoint_and_county_microbatch_outputs_and_gradients():
    full, h, geo, *_ = make_layer(checkpoint_steps=0)
    full.use_adjoint = False
    full.geo_checkpoint = False
    with torch.no_grad():
        full.alpha.fill_(0.25)
        full.training_step.fill_(200)
        full.readout.copy_(torch.diag(torch.linspace(0.7, 1.4, 32, dtype=h.dtype)))
    full.train()
    full.set_drop_override(1)
    micro = copy.deepcopy(full)
    micro.checkpoint_steps = 3
    micro.use_adjoint = True
    micro.geo_checkpoint = True
    # Repeated counties must accumulate geographic parameter gradients correctly.
    geo = geo.clone()
    geo[4] = geo[0]
    h1, h2 = h.clone().requires_grad_(), h.clone().requires_grad_()
    output = full(h1, geo)
    target = torch.linspace(-1, 1, output.numel(), dtype=h.dtype).reshape_as(output)
    (output * target).sum().backward()
    chunks = []
    for start in range(0, len(h), 2):
        result = micro(h2[start:start + 2], geo[start:start + 2])
        (result * target[start:start + 2]).sum().backward()
        chunks.append(result.detach())
    torch.testing.assert_close(torch.cat(chunks), output, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(h1.grad, h2.grad, atol=3e-11, rtol=3e-10)
    for (name, p), (other_name, q) in zip(full.named_parameters(), micro.named_parameters()):
        assert name == other_name
        assert p.grad is not None and q.grad is not None, name
        torch.testing.assert_close(p.grad, q.grad, atol=3e-10, rtol=3e-9, msg=name)


def test_all_parameter_and_input_directional_gradcheck():
    layer, h, geo, *_ = make_layer(checkpoint_steps=0)
    with torch.no_grad():
        layer.alpha.fill_(0.31)
        layer.training_step.fill_(200)
        # P=I has repeated singular values at the clipping kink. Test an active,
        # simple top singular value instead, as required by the design.
        layer.readout.copy_(torch.diag(torch.linspace(0.6, 1.5, 32, dtype=h.dtype)))
    names, parameters = zip(*layer.named_parameters())
    hh = h[:1, :2].clone().requires_grad_()
    gg = geo[:1].clone().requires_grad_()

    def evaluate(*args):
        return torch.func.functional_call(layer, dict(zip(names, args[:-2])), (args[-2], args[-1]))

    assert torch.autograd.gradcheck(evaluate, parameters + (hh, gg), fast_mode=True,
                                    eps=1e-6, atol=2e-6, rtol=2e-4)


def test_causality_finite_extremes_diagnostics_and_shared_drop():
    layer, h, geo, *_ = make_layer(checkpoint_steps=3)
    with torch.no_grad():
        layer.alpha.fill_(0.4)
        layer.training_step.fill_(200)
    first, details = layer(h, geo, diagnostics=True)
    changed = h.clone()
    changed[:, 5:] += 100
    second = layer(changed, geo)
    torch.testing.assert_close(first[:, :5], second[:, :5], atol=1e-13, rtol=1e-13)
    assert float(details["state"].detach().norm(dim=-1).max()) <= 1 + 1e-12
    assert float(details["response"].detach().norm(dim=-1).max()) <= 2 + 1e-12
    assert float(torch.linalg.matrix_norm(bounded_readout(layer.readout).detach(), ord=2)) <= 1 + 1e-12
    extreme = h.sign() * 1e100
    out, dd = layer(extreme, geo, diagnostics=True)
    assert bool(torch.isfinite(out).all())
    assert all(bool(torch.isfinite(value).all()) for value in dd.values())
    layer.train()
    layer.set_drop_override(None)
    before = layer._drop.get_state().clone()
    layer(h[:2], geo[:2])
    after = layer._drop.get_state().clone()
    mask = layer.last_mask
    assert not torch.equal(before, after)
    layer(h[2:], geo[2:])
    assert layer.last_mask == mask and torch.equal(layer._drop.get_state(), after)
    layer.eval()
    layer(h, geo)
    assert torch.equal(layer._drop.get_state(), after)
    layer.train()
    layer.training_step.add_(1)
    layer(h, geo)
    assert not torch.equal(layer._drop.get_state(), after)


def test_checkpoint_state_roundtrip():
    layer, h, geo, _, fit, ids = make_layer(checkpoint_steps=3)
    layer.geo_checkpoint = False
    with torch.no_grad():
        layer.alpha.fill_(0.1)
        layer.training_step.fill_(44)
    other = ControlledRelaxationLayer(nn.Linear(32, 12, dtype=h.dtype), fit, ids)
    other.load_state_dict(copy.deepcopy(layer.state_dict()))
    other.eval()
    assert other.fit_metadata == layer.fit_metadata
    assert other.checkpoint_steps == 3
    assert other.geo_checkpoint is False
    torch.testing.assert_close(layer(h, geo), other(h, geo), atol=0, rtol=0)
