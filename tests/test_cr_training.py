"""I20 Engine checks on eight synthetic units, never observational data.

These checks compare a complete-fit update with 3+3+2 county accumulation.
The loss has nonuniform observation weights and a zero-weight county. All 216
hours are retained so the 72-hour origin and full recurrent path are exercised.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from asymode.gcrk_train import Engine, export  # noqa: E402
from asymode.gcrk import prefix_reference  # noqa: E402
from asymode.asym_host import masked_se, trajectory_loss  # noqa: E402


@pytest.fixture(autouse=True)
def two_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def features():
    rng = np.random.default_rng(20260928)
    n, t = 8, 216
    f = dict(
        xu=rng.normal(size=(n, t, 16)).astype(np.float32),
        xr=rng.normal(size=(n, t, 23)).astype(np.float32),
        xo=rng.normal(size=(n, t, 3)).astype(np.float32),
        geo=rng.normal(size=(n, 40)).astype(np.float32),
        fips=np.array([f"{i:05d}" for i in range(n)]),
        y0=rng.uniform(0.01, 0.2, n).astype(np.float32),
        y=rng.uniform(0.0, 0.3, (n, 144)).astype(np.float32),
        m=(rng.uniform(size=(n, 144)) > 0.15).astype(np.float32),
    )
    f["xr"][:, :, 14:20] = f["xr"][:, :1, 14:20]
    f["m_train"] = f["m"] * rng.uniform(0.2, 3.0, (n, 144)).astype(np.float32)
    f["m_train"][4] = 0  # one county contributes calibration but no training loss
    # Repeat one county across a chunk boundary to exercise geographic deduplication.
    f["fips"][6], f["geo"][6] = f["fips"][0], f["geo"][0]
    return f


def engine(f, microbatch, arm="CRK+Cin"):
    e = Engine(f, np.arange(8), np.arange(8), seed=0, arm=arm)
    e.microbatch_size = microbatch
    return e


def open_branch(e):
    e.step = 200
    k = e.model.kernel
    with torch.no_grad():
        k.alpha.fill_(0.65)
        # A simple active top singular value avoids P=I's clipping kink.
        k.readout.copy_(torch.diag(torch.linspace(0.7, 1.3, 32)))
    e.refresh()
    # The first draw is open; subsequent microbatches must reuse it, not redraw.
    k._drop.manual_seed(1)
    k._drop_step = None
    assert float(torch.rand((), generator=torch.Generator().manual_seed(1))) >= 0.2


def test_shared_host_initialization_and_alpha_zero_equivalence():
    f = features()
    host, candidate = engine(f, None, "W+Cin"), engine(f, 3)
    host_parameters = dict(host.model.named_parameters())
    candidate_parameters = dict(candidate.model.named_parameters())
    for name, expected in host_parameters.items():
        assert torch.equal(candidate_parameters[name], expected), name
    assert sum(p.numel() for p in candidate.model.parameters()) - sum(
        p.numel() for p in host.model.parameters()) == 15209
    candidate.model.kernel.training_step.fill_(200)  # test alpha=0, not only ramp=0
    candidate.model.eval(); host.model.eval()
    with torch.no_grad():
        expected = host.model(host.fit)
        actual = candidate.model(candidate.fit)
    assert expected.keys() == actual.keys()
    for name in expected:
        assert torch.equal(actual[name], expected[name]), name


def test_full_gradient_accumulation_matches_loss_gradients_and_one_adam_step():
    f = features()
    full, micro = engine(f, None), engine(f, 3)
    open_branch(full); open_branch(micro)
    full.model.eval()
    with torch.no_grad():
        before = full.model(full.fit)["P"]
        expected_loss = trajectory_loss(before, full.fit["y"], full.fit["m_train"])
        unweighted_loss = trajectory_loss(before, full.fit["y"], full.fit["m"])
    assert abs(float(expected_loss - unweighted_loss)) > 1e-6
    full_loss, micro_loss = full.train_step(), micro.train_step()
    assert full_loss == pytest.approx(float(expected_loss), abs=2e-8, rel=2e-6)
    assert micro_loss == pytest.approx(full_loss, abs=2e-8, rel=2e-6)
    assert full.model.kernel.last_mask == micro.model.kernel.last_mask == 1.0
    assert torch.equal(full.model.kernel._drop.get_state(), micro.model.kernel._drop.get_state())
    assert full.step == micro.step == 201
    full_parameters, micro_parameters = dict(full.model.named_parameters()), dict(micro.model.named_parameters())
    max_gradient_difference = max_parameter_difference = 0.0
    for name, p in full_parameters.items():
        q = micro_parameters[name]
        assert p.grad is not None and q.grad is not None, name
        assert bool(torch.isfinite(p.grad).all() & torch.isfinite(q.grad).all()), name
        torch.testing.assert_close(p.grad, q.grad, atol=3e-9, rtol=3e-4, msg=name)
        torch.testing.assert_close(p, q, atol=3e-6, rtol=3e-5, msg=name)
        max_gradient_difference = max(max_gradient_difference, float((p.grad - q.grad).abs().max()))
        max_parameter_difference = max(max_parameter_difference, float((p - q).detach().abs().max()))
        fp, mp = full.opt.state[p], micro.opt.state[q]
        assert int(fp["step"]) == int(mp["step"]) == 1
        for key in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(fp[key], mp[key], atol=5e-10, rtol=6e-4, msg=f"{name}/{key}")
    for name in ("alpha", "head_geo", "head_weather", "head_fusion", "length_raw", "order_logits", "plane_a", "readout"):
        assert float(full_parameters[f"damage.2.{name}"].grad.norm()) > 1e-12, name
    print(f"full/3+3+2: loss_difference={abs(full_loss - micro_loss):.3g}, "
          f"max_gradient_difference={max_gradient_difference:.3g}, "
          f"max_one_adam_parameter_difference={max_parameter_difference:.3g}")


def test_chunked_calibration_matches_full_fit_and_refresh_bounds_hidden_batches(monkeypatch):
    f = features()
    e = engine(f, 3)
    with torch.no_grad():
        h = e.model.hidden(e.fit["xu"], e.fit["ctx"])
    whole, chunked = copy.deepcopy(e.model.kernel), copy.deepcopy(e.model.kernel)
    whole.calibrate_(h, 210)
    chunked.calibrate_chunks_((h[j:j + 3] for j in range(0, 8, 3)), 210)
    departure = h - prefix_reference(h)
    torch.testing.assert_close(whole.scale, departure.norm(dim=-1)[:, 1:].reshape(-1).median().clamp_min(1e-4))
    for name in ("scale", "level_scale", "departure_scale"):
        torch.testing.assert_close(getattr(whole, name), getattr(chunked, name), atol=1e-7, rtol=1e-6)
    assert whole.last_calibration["n_counties"] == chunked.last_calibration["n_counties"] == 8
    assert whole.last_calibration["n_values"] == chunked.last_calibration["n_values"] == 8 * 215
    seen, original = [], e.model.hidden

    def counted(xu, ctx=None, mech=None):
        seen.append(len(xu))
        return original(xu, ctx, mech)

    monkeypatch.setattr(e.model, "hidden", counted)
    e.step = 210
    e.refresh()
    assert seen == [3, 3, 2]
    for name in ("scale", "level_scale", "departure_scale"):
        torch.testing.assert_close(getattr(e.model.kernel, name), getattr(whole, name), atol=1e-7, rtol=1e-6)


def test_chunked_evaluation_and_export_keep_row_order_and_raw_masks():
    f = features()
    e = engine(f, None)
    open_branch(e)
    s_full, n_full = e.evaluate()
    idx = np.array([7, 1, 6, 0, 4, 2, 5, 3])
    all_export = export(e, f, idx)
    e.microbatch_size = 3
    s_micro, n_micro = e.evaluate()
    chunk_export = export(e, f, idx)
    np.testing.assert_allclose(s_micro, s_full, rtol=3e-6, atol=1e-7)
    np.testing.assert_array_equal(n_micro, n_full)
    np.testing.assert_array_equal(n_micro, f["m"].sum(1))  # evaluation never uses m_train
    assert n_micro[4] > 0
    assert all_export.keys() == chunk_export.keys()
    for name in all_export:
        np.testing.assert_allclose(chunk_export[name], all_export[name], rtol=3e-6, atol=1e-7, err_msg=name)
    np.testing.assert_array_equal(chunk_export["idx"], idx)
    assert {"P_closed", "raw_logit_closed"} <= set(chunk_export)
