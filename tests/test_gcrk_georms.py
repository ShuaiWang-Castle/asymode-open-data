"""I18 fixed fitting-fold RMS geography normalisation, synthetic inputs only.

These checks cover the complete host wiring, the geographic map and response
adjoint, bounded states, fitting-set isolation, checkpoint compatibility, and
the radial information that the new normalisation is intended to preserve.
No optimiser step or observational data is used.
"""
from __future__ import annotations

import io
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.func import functional_call
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from asymode.gcrk import (  # noqa: E402
    GAIN_LOG_BOUND,
    GATE_SLOPE,
    INTERACTION_BOUND,
    GCRKLayer,
    prefix_reference,
    response_sequence_loop,
)
from asymode.gcrk_train import Engine  # noqa: E402


def _features(seed=41):
    rng = np.random.default_rng(seed)
    n, t = 8, 216
    features = {
        "xu": rng.normal(size=(n, t, 16)).astype(np.float32),
        "xr": rng.normal(size=(n, t, 23)).astype(np.float32),
        "xo": rng.normal(size=(n, t, 3)).astype(np.float32),
        "geo": rng.normal(size=(n, 5)).astype(np.float32),
        "y0": rng.uniform(0.0, 0.2, size=n).astype(np.float32),
        "y": rng.uniform(0.0, 0.3, size=(n, 144)).astype(np.float32),
        "m": np.ones((n, 144), dtype=np.float32),
    }
    # The host's six context inputs represent static county quantities.
    features["xr"][:, :, 14:20] = features["xr"][:, :1, 14:20]
    return features


def _layer(dtype=torch.float64):
    gen = torch.Generator().manual_seed(112)
    fit_geo = torch.randn(7, 3, generator=gen, dtype=dtype)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(113)
        original = nn.Linear(5, 5).to(dtype)
    layer = GCRKLayer(original, torch.tanh(fit_geo / 3.0).mean(0)).to(dtype)
    layer.attach_geo_rms(fit_geo)
    return layer, gen


@torch.no_grad()
def _open_geographic_heads(layer, gen):
    for name in ("Vl", "Va", "Vg"):
        p = getattr(layer, name)
        p.copy_(0.4 * torch.randn(p.shape, generator=gen, dtype=p.dtype))
    # Distinct damping coordinates avoid a minimum tie in the adjoint comparison.
    layer.l0.add_(torch.linspace(-0.3, 0.3, layer.in_features, dtype=layer.l0.dtype))
    layer.g0.fill_(0.2)
    layer.alpha.fill_(0.7)
    layer.training_step.fill_(300)
    layer.eval()


def test_full_host_paired_step_zero_and_legacy_checkpoint():
    features = _features()
    fit_idx, val_idx = np.arange(6), np.arange(6, 8)
    host = Engine(features, fit_idx, val_idx, seed=0, arm="W+Cin")
    legacy = Engine(features, fit_idx, val_idx, seed=0, arm="GCRK+Cin")
    georms = Engine(features, fit_idx, val_idx, seed=0, arm="GCRK+Cin-georms")

    host_params = dict(host.model.named_parameters())
    for engine in (legacy, georms):
        params = dict(engine.model.named_parameters())
        assert engine.model.ctx_in is not None
        for name, value in host_params.items():
            assert torch.equal(params[name], value), name
    # The new option adds only a persistent buffer, with the same trainable recipe.
    legacy_state = legacy.model.state_dict()
    rms_state = georms.model.state_dict()
    assert set(rms_state) - set(legacy_state) == {"damage.2.geo_rms_scale"}
    assert set(legacy_state) - set(rms_state) == set()
    for name, value in legacy_state.items():
        assert torch.equal(rms_state[name], value), name
    assert "geo_rms_scale" not in dict(georms.model.kernel.named_parameters())
    assert not hasattr(legacy.model.kernel, "geo_rms_scale")
    fresh_legacy = Engine(features, fit_idx, val_idx, seed=9, arm="GCRK+Cin")
    fresh_legacy.model.load_state_dict(legacy_state, strict=True)

    for engine in (host, legacy, georms, fresh_legacy):
        engine.model.eval()
    with torch.no_grad():
        reference = host.model(host.val, diagnostics=True)
        for engine in (legacy, georms, fresh_legacy):
            actual = engine.model(host.val, diagnostics=True)
            for name, expected in reference.items():
                assert torch.equal(actual[name], expected), name


class _CodeMap(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, geography):
        return self.layer.code_of(geography)


def test_geographic_map_gradcheck_includes_projection_and_input():
    layer, gen = _layer()
    module = _CodeMap(layer)
    geography = (0.6 * torch.randn(2, 3, generator=gen, dtype=torch.float64)).requires_grad_()
    projection = layer.U.detach().clone().requires_grad_()

    def code(u, g):
        return functional_call(module, {"layer.U": u}, (g,))

    expected_z = (torch.tanh(geography / 3.0) - layer.geo_center) / layer.geo_rms_scale
    torch.testing.assert_close(code(projection, geography), torch.tanh(F.linear(expected_z, projection)))
    assert torch.autograd.gradcheck(code, (projection, geography), eps=1e-6, atol=1e-6, rtol=1e-4)
    grads = torch.autograd.grad(code(projection, geography).square().sum(), (projection, geography))
    assert all(torch.isfinite(g).all() and g.norm() > 1e-5 for g in grads)


def _plain_layer(layer, h, geography):
    """Independent fixed-RMS conditioning followed by the ordinary autograd loop."""
    z = (torch.tanh(geography / 3.0) - layer.geo_center) / layer.geo_rms_scale
    code = torch.tanh(F.linear(z, layer.U))
    lam = layer.lambda_min + (layer.lambda_max - layer.lambda_min) * torch.sigmoid(
        layer.l0 + F.linear(code, layer.Vl)
    )
    w = layer.a0 + F.linear(code, layer.Va)
    interaction = INTERACTION_BOUND * w / torch.sqrt(1.0 + w.square().sum(-1, keepdim=True))
    gain = torch.exp(GAIN_LOG_BOUND * torch.tanh(layer.g0 + F.linear(code, layer.Vg)))
    departure = h - prefix_reference(h)
    norm = departure.norm(dim=-1, keepdim=True)
    deposit = torch.sigmoid(GATE_SLOPE * (norm / layer.scale - layer.threshold)) * (
        departure / torch.sqrt(layer.scale.square() + norm.square())
    )
    state = response_sequence_loop(lam.amin(-1, keepdim=True)[:, None] * deposit, lam, interaction)
    effect = layer.ramp() * torch.tanh(layer.alpha) * layer.scale * gain[:, None] * state
    return F.linear(h + effect, layer.weight, layer.bias)


def test_complete_new_mapping_adjoint_matches_plain_forward_and_gradients():
    layer, gen = _layer()
    _open_geographic_heads(layer, gen)
    fit_h = torch.randn(4, 12, 5, generator=gen, dtype=torch.float64)
    layer.calibrate_(fit_h, 300)
    h = torch.randn(2, 9, 5, generator=gen, dtype=torch.float64).requires_grad_()
    geography = torch.randn(2, 3, generator=gen, dtype=torch.float64).requires_grad_()
    probe = torch.randn(h.shape, generator=gen, dtype=h.dtype)
    actual, expected = layer(h, geography), _plain_layer(layer, h, geography)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    named_variables = [("h", h), ("geography", geography), *layer.named_parameters()]
    variables = [p for _, p in named_variables]
    actual_grad = torch.autograd.grad((actual * probe).sum(), variables)
    expected_grad = torch.autograd.grad((expected * probe).sum(), variables)
    for (name, _), got, want in zip(named_variables, actual_grad, expected_grad):
        assert torch.isfinite(got).all(), name
        torch.testing.assert_close(got, want, atol=1e-10, rtol=1e-9, msg=lambda message: f"{name}: {message}")
    gradients = dict(zip((name for name, _ in named_variables), actual_grad))
    # An all-zero geographic head would make this a vacuous map-gradient check.
    for name in ("geography", "U", "Vl", "Va", "Vg"):
        assert gradients[name].norm() > 1e-8, name


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_large_geography_preserves_parameter_and_unit_state_bounds(dtype):
    layer, gen = _layer(dtype)
    _open_geographic_heads(layer, gen)
    layer.calibrate_(torch.randn(4, 216, 5, generator=gen, dtype=dtype), 300)
    with torch.no_grad():
        # Push all conditioning maps well outside their ordinary initial range.
        for name in ("U", "Vl", "Va", "Vg", "l0", "a0", "g0"):
            getattr(layer, name).mul_(100.0)
        geography = 1e6 * torch.randn(6, 3, generator=gen, dtype=dtype)
        h = 1e4 * torch.randn(6, 216, 5, generator=gen, dtype=dtype)
        output, diagnostics = layer(h, geography, diagnostics=True)
        code = layer.code_of(geography)
    assert torch.isfinite(output).all() and torch.isfinite(code).all()
    assert code.abs().max() <= 1.0
    for name in ("state", "damping", "interaction", "coordinate_gain"):
        assert torch.isfinite(diagnostics[name]).all(), name
    assert diagnostics["damping"].min() >= layer.lambda_min - 1e-7
    assert diagnostics["damping"].max() <= layer.lambda_max + 1e-7
    assert diagnostics["interaction"].norm(dim=-1).max() <= INTERACTION_BOUND + 1e-6
    assert diagnostics["coordinate_gain"].min() >= 0.5 - 1e-6
    assert diagnostics["coordinate_gain"].max() <= 2.0 + 1e-6
    assert diagnostics["deposit"].norm(dim=-1).max() <= 1.0 + 1e-6
    assert diagnostics["state"].norm(dim=-1).max() <= 1.0 + 2e-6


def test_scale_is_fitted_only_once_from_fit_rows_and_checkpoint_reproduces():
    features = _features()
    fit_idx, val_idx = np.arange(6), np.arange(6, 8)
    changed = {name: value.copy() for name, value in features.items()}
    changed["geo"][val_idx] = np.array([1e7, -1e7], dtype=np.float32)[:, None]
    original = Engine(features, fit_idx, val_idx, seed=0, arm="GCRK+Cin-georms")
    heldout_changed = Engine(changed, fit_idx, val_idx, seed=0, arm="GCRK+Cin-georms")
    layer = original.model.kernel
    expected = torch.sqrt(0.1 ** 2 + (
        torch.tanh(original.fit["geo"] / 3.0) - layer.geo_center
    ).square().sum(-1).mean())
    torch.testing.assert_close(layer.geo_rms_scale, expected, atol=0.0, rtol=0.0)
    assert not layer.geo_rms_scale.requires_grad
    for name, value in original.model.state_dict().items():
        assert torch.equal(value, heldout_changed.model.state_dict()[name]), name
    for mean_or_sd, other in zip(original.stats["geo"], heldout_changed.stats["geo"]):
        np.testing.assert_array_equal(mean_or_sd, other)

    fixed_scale = layer.geo_rms_scale.clone()
    gen = torch.Generator().manual_seed(119)
    _open_geographic_heads(layer, gen)
    original.step = 300
    original.refresh()
    # Evaluation and hidden-state calibration must never refit geographic scale.
    original.evaluate()
    layer.code_of(torch.full_like(original.val["geo"], 1e9))
    assert torch.equal(layer.geo_rms_scale, fixed_scale)

    storage = io.BytesIO()
    torch.save(original.snapshot()["model_state"], storage)
    storage.seek(0)
    saved = torch.load(storage, weights_only=True)
    reloaded = Engine(features, fit_idx, val_idx, seed=17, arm="GCRK+Cin-georms")
    reloaded.model.load_state_dict(saved, strict=True)
    reloaded.model.eval()
    original.model.eval()
    with torch.no_grad():
        before = original.model(original.val, diagnostics=True)
        after = reloaded.model(original.val, diagnostics=True)
    for name, expected_output in before.items():
        assert torch.equal(after[name], expected_output), name
    assert torch.equal(reloaded.model.kernel.geo_rms_scale, fixed_scale)


def test_attached_scale_is_detached_with_a_positive_floor():
    layer, _ = _layer()
    # A degenerate fitting fold still has a valid fixed denominator of 0.1.
    zero_geo = torch.zeros(4, 3, dtype=torch.float64, requires_grad=True)
    plain = GCRKLayer(nn.Linear(5, 5).double(), torch.zeros(3, dtype=torch.float64))
    plain.attach_geo_rms(zero_geo)
    assert float(plain.geo_rms_scale) == 0.1
    assert plain.geo_rms_scale.grad_fn is None
    assert not plain.geo_rms_scale.requires_grad
    assert "geo_rms_scale" in dict(plain.named_buffers())
    assert layer.geo_rms_scale.ndim == 0


def test_centered_collinear_radial_amplitude_is_retained():
    # Construct vectors in the bounded, centred coordinates used by the kernel.
    center = torch.zeros(3, dtype=torch.float64)
    fit_z = torch.tensor([[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]], dtype=torch.float64)
    fit_geo = 3.0 * torch.atanh(fit_z)
    z = torch.tensor([[0.4, 0.0, 0.0], [0.8, 0.0, 0.0]], dtype=torch.float64)
    geography = 3.0 * torch.atanh(z)
    legacy = GCRKLayer(nn.Linear(5, 5).double(), center)
    fixed = GCRKLayer(nn.Linear(5, 5).double(), center)
    fixed.attach_geo_rms(fit_geo)
    with torch.no_grad():
        for layer in (legacy, fixed):
            layer.U.zero_()
            layer.U[0, 0] = 0.2  # Keep the outer tanh away from saturation.
        fixed_precode = torch.atanh(fixed.code_of(geography)[:, 0])
        legacy_precode = torch.atanh(legacy.code_of(geography)[:, 0])
    torch.testing.assert_close(fixed_precode[1] / fixed_precode[0], torch.tensor(2.0, dtype=z.dtype))
    expected_legacy_ratio = 2.0 * math.sqrt(0.1 ** 2 + 0.4 ** 2) / math.sqrt(0.1 ** 2 + 0.8 ** 2)
    assert float(legacy_precode[1] / legacy_precode[0]) == pytest.approx(expected_legacy_ratio)
    assert float(legacy_precode[1] / legacy_precode[0]) < 1.05
    torch.testing.assert_close(fixed_precode, 0.2 * z[:, 0] / fixed.geo_rms_scale)
