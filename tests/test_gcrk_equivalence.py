"""GCRK re-implementation vs the frozen reference layer, on synthetic inputs only.

tests/fixtures/gcrk_reference_synthetic.npz holds seeded random hidden sequences,
geography and layer parameters together with the frozen reference layer's outputs
on them (built by tests/fixtures/make_gcrk_fixture.py). Nothing in it comes from
observations. Checked element by element, float32 tolerance 1e-6:

  calibration (scale c, threshold theta), departure direction q, amplitude gate,
  deposit d, response state e, dissipation, interaction, readout gain, the layer
  output W2 (h + beta_s c Om e) + b2 at two opening stages, drop-path masks and
  outputs in training mode, and gradients with respect to inputs and parameters.

If GCRK_REFERENCE_DIR points at the reference source (model.py, kernel.py), the
live test additionally re-runs the reference on fresh random draws.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests" / "fixtures"))
from asymode.gcrk import GCRKLayer, response_sequence, response_sequence_loop  # noqa: E402

FIX = np.load(ROOT / "tests/fixtures/gcrk_reference_synthetic.npz")
REFERENCE_PRIVATE_SEED = 91723          # the reference layer's own initialisation seed
TOL = {"f32": dict(atol=1e-6, rtol=0.0), "f64": dict(atol=1e-12, rtol=0.0)}
DT = {"f32": torch.float32, "f64": torch.float64}
KERNEL_PARAMS = ("U", "Vl", "Va", "Vg", "l0", "a0", "g0", "alpha")


def t(name, tag):
    return torch.from_numpy(FIX[f"{tag}_{name}"])


def layer_from_fixture(tag: str) -> GCRKLayer:
    d = DT[tag]
    lin = nn.Linear(32, 32).to(d)
    with torch.no_grad():
        lin.weight.copy_(t("param_weight", tag)); lin.bias.copy_(t("param_bias", tag))
    lay = GCRKLayer(lin, t("in_center", tag)).to(d)
    with torch.no_grad():
        for n in KERNEL_PARAMS:
            getattr(lay, n).copy_(t(f"param_{n}", tag))
    return lay


def close(a, b, tag, what):
    a = a.detach().numpy() if torch.is_tensor(a) else np.asarray(a)
    b = np.asarray(b)
    assert a.shape == b.shape, (what, a.shape, b.shape)
    err = float(np.max(np.abs(a - b))) if a.size else 0.0
    assert np.allclose(a, b, **TOL[tag]), f"{what} [{tag}] max |diff| = {err:.3e}"
    return err


@pytest.mark.parametrize("tag", ["f32", "f64"])
def test_initialisation_recipe(tag):
    d = DT[tag]
    lin = nn.Linear(32, 32).to(d)
    lay = GCRKLayer(lin, t("in_center", tag), private_seed=REFERENCE_PRIVATE_SEED).to(d)
    # the recipe draws and scales in float32 before casting, so float32 rounding (which can
    # differ across platforms and BLAS builds) bounds the agreement for both dtypes
    for n in ("U", "a0", "l0"):
        close(getattr(lay, n), FIX[f"{tag}_init_{n}"], "f32", f"initial {n}")
    for n in ("Vl", "Va", "Vg", "g0", "alpha"):
        assert float(getattr(lay, n).detach().abs().max()) == 0.0


@pytest.mark.parametrize("tag", ["f32", "f64"])
def test_calibration_and_forward(tag):
    lay = layer_from_fixture(tag)
    lay.calibrate_(t("in_h_fit", tag), int(FIX[f"{tag}_calibration_step"]))
    close(lay.scale, FIX[f"{tag}_scale"], tag, "scale c")
    close(lay.threshold, FIX[f"{tag}_threshold"], tag, "threshold theta")
    lay.eval()
    h, g = t("in_h", tag), t("in_geo", tag)
    lay.training_step.fill_(150)
    with torch.no_grad():
        close(lay(h, g), FIX[f"{tag}_eval150_out"], tag, "output at opening 0.75")
        lay.training_step.fill_(400)
        out, di = lay(h, g, diagnostics=True)
    close(out, FIX[f"{tag}_eval400_out"], tag, "output at opening 1")
    for k in ("q", "deposit", "gate", "state", "damping", "interaction", "coordinate_gain"):
        close(di[k], FIX[f"{tag}_eval400_{k}"], tag, k)
    assert float(di["state"].norm(dim=-1).max()) <= 1.0 + 1e-6            # unit-state bound
    assert float((di["state"] * di["interaction"][:, None]).sum(-1).abs().max()) <= 0.5 + 1e-6


@pytest.mark.parametrize("tag", ["f32", "f64"])
def test_gradients(tag):
    lay = layer_from_fixture(tag)
    lay.calibrate_(t("in_h_fit", tag), int(FIX[f"{tag}_calibration_step"]))
    lay.eval(); lay.training_step.fill_(400)
    h = t("in_h", tag).clone().requires_grad_(True)
    g = t("in_geo", tag).clone().requires_grad_(True)
    (lay(h, g) * t("wproj", tag)).sum().backward()
    tol = dict(atol=2e-5, rtol=1e-5) if tag == "f32" else dict(atol=1e-10, rtol=1e-10)
    for name, got in [("h", h.grad), ("geo", g.grad)] + [(f"param_{n}", p.grad) for n, p in lay.named_parameters()]:
        ref = FIX[f"{tag}_grad_{name}"]
        assert np.allclose(got.numpy(), ref, **tol), f"grad {name} [{tag}] max |diff| = {np.abs(got.numpy() - ref).max():.3e}"


@pytest.mark.parametrize("tag", ["f32", "f64"])
def test_training_mode_drop_path(tag):
    lay = layer_from_fixture(tag)
    lay.calibrate_(t("in_h_fit", tag), int(FIX[f"{tag}_calibration_step"]))
    lay.train(); lay.training_step.fill_(120)
    lay._drop.set_state(torch.from_numpy(FIX[f"{tag}_rng_state"]))
    h, g = t("in_h", tag), t("in_geo", tag)
    masks = []
    with torch.no_grad():
        for i in range(len(FIX[f"{tag}_train_masks"])):
            y = lay(h, g)
            masks.append(lay.last_mask)
            close(y, FIX[f"{tag}_train{i}_out"], tag, f"training call {i}")
    assert masks == list(FIX[f"{tag}_train_masks"])


def test_solver_matches_dense_equation():
    """(I + D - nu S(d)) e_t = e_{t-1} + nu d_t solved densely, float64."""
    g = torch.Generator().manual_seed(3)
    B, T, D = 3, 40, 32
    d = torch.randn(B, T, D, generator=g, dtype=torch.float64)
    d = 0.9 * d / (1 + d.norm(dim=-1, keepdim=True))
    lam = 0.01 + 0.6 * torch.rand(B, D, generator=g, dtype=torch.float64)
    a = torch.randn(B, D, generator=g, dtype=torch.float64)
    a = 0.5 * a / torch.sqrt(1 + a.square().sum(-1, keepdim=True))
    nu = lam.amin(-1, keepdim=True)
    e = response_sequence(nu[:, None] * d, lam, a)
    prev = torch.zeros(B, D, dtype=torch.float64)
    for s in range(T):
        f = nu * d[:, s]
        S = a[:, :, None] * f[:, None, :] - f[:, :, None] * a[:, None, :]
        M = torch.diag_embed(1 + lam) - S
        prev = torch.linalg.solve(M, (prev + f)[..., None])[..., 0]
        assert torch.allclose(e[:, s], prev, atol=1e-12)
    assert torch.allclose(e, response_sequence_loop(nu[:, None] * d, lam, a), atol=1e-13)


def test_adjoint_matches_autograd():
    g = torch.Generator().manual_seed(4)
    f = (0.2 * torch.randn(2, 7, 6, generator=g, dtype=torch.float64)).requires_grad_(True)
    lam = (0.02 + 0.5 * torch.rand(2, 6, generator=g, dtype=torch.float64)).requires_grad_(True)
    a = (0.2 * torch.randn(2, 6, generator=g, dtype=torch.float64)).requires_grad_(True)
    assert torch.autograd.gradcheck(response_sequence, (f, lam, a))


@pytest.mark.skipif(not os.environ.get("GCRK_REFERENCE_DIR"), reason="reference source not provided")
def test_live_against_reference():
    from make_gcrk_fixture import load_reference
    cls, pref, _ = load_reference(Path(os.environ["GCRK_REFERENCE_DIR"]))
    gen = torch.Generator().manual_seed(11)
    for dtype, atol in ((torch.float32, 1e-6), (torch.float64, 1e-12)):
        lin = nn.Linear(32, 32).to(dtype)
        geo_fit = torch.randn(7, 20, generator=gen, dtype=dtype)
        ref = cls(lin, torch.tanh(geo_fit / 3).mean(0)).to(dtype)
        mine = GCRKLayer(lin, torch.tanh(geo_fit / 3).mean(0)).to(dtype)
        with torch.no_grad():
            for n in KERNEL_PARAMS:
                v = getattr(ref, n) + 0.3 * torch.randn(getattr(ref, n).shape, generator=gen, dtype=dtype)
                getattr(ref, n).copy_(v); getattr(mine, n).copy_(v)
        h_fit = torch.relu(torch.randn(7, 216, 32, generator=gen, dtype=dtype))
        h = torch.relu(torch.randn(5, 216, 32, generator=gen, dtype=dtype) + torch.linspace(-1, 1, 216, dtype=dtype)[None, :, None])
        geo = torch.randn(5, 20, generator=gen, dtype=dtype)
        ref.calibrate_(h_fit, 10); mine.calibrate_(h_fit, 10)
        for lay in (ref, mine):
            lay.eval(); lay.training_step.fill_(333)
        with torch.no_grad():
            assert torch.allclose(pref(h), __import__("asymode.gcrk", fromlist=["x"]).prefix_reference(h), atol=atol)
            assert torch.allclose(ref.compute(h, geo, None, fast=False), mine(h, geo), atol=atol)
