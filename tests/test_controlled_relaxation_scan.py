"""Synthetic exactness tests for the I20 custom adjoint; never read panel data."""
from pathlib import Path
import sys

import pytest
import torch
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from asymode.controlled_relaxation import controlled_sequence as reference
from asymode.controlled_relaxation_scan import controlled_sequence_adjoint as scan


def inputs(batch=2, steps=5, modes=4, dim=8, planes=4, dtype=torch.float64, seed=17):
    gen = torch.Generator().manual_seed(seed)
    rand = lambda *shape: torch.randn(*shape, generator=gen, dtype=dtype)
    deposit = .2 * rand(batch, steps, modes, dim)
    rho = torch.sigmoid(rand(batch, steps, modes))
    eta = .25 * torch.tanh(rand(batch, steps, modes, planes))
    a, b = rand(modes, planes, dim), rand(modes, planes, dim)
    a = a / torch.sqrt(1 + a.square().sum(-1, keepdim=True))
    b = b / torch.sqrt(1 + b.square().sum(-1, keepdim=True))
    initial = .2 * rand(batch, modes, dim)
    return tuple(x.requires_grad_() for x in (deposit, rho, eta, a, b, initial))


@pytest.mark.parametrize('initial', [True, False])
def test_forward_and_all_input_gradients_match_reference(initial):
    args = inputs()
    if not initial: args = args[:-1]
    actual, expected = scan(*args), reference(*args)
    torch.testing.assert_close(actual, expected, atol=2e-14, rtol=2e-14)
    gen = torch.Generator().manual_seed(91)
    weight = torch.randn(actual.shape, generator=gen, dtype=actual.dtype)
    # External gradients at every hour, not merely terminal loss.
    ga = torch.autograd.grad((actual * weight).sum(), args)
    gr = torch.autograd.grad((expected * weight).sum(), args)
    for x, y in zip(ga, gr): torch.testing.assert_close(x, y, atol=3e-13, rtol=3e-12)


def test_double_gradcheck_all_six_inputs():
    args = inputs(batch=1, steps=3, modes=2, dim=3, planes=2)
    assert torch.autograd.gradcheck(scan, args, eps=1e-6, atol=2e-6, rtol=1e-5, fast_mode=True)


def test_shared_planes_sum_batch_and_time_gradients():
    args = inputs(batch=3, steps=4)
    output = scan(*args)
    plane_grad = torch.autograd.grad(output.square().sum(), args[3:5])
    totals = [torch.zeros_like(args[3]), torch.zeros_like(args[4])]
    for i in range(3):
        part = (args[0][i:i+1], args[1][i:i+1], args[2][i:i+1],
                args[3], args[4], args[5][i:i+1])
        grads = torch.autograd.grad(reference(*part).square().sum(), args[3:5])
        for j in range(2): totals[j] += grads[j]
    for x, y in zip(plane_grad, totals): torch.testing.assert_close(x, y, atol=2e-13, rtol=2e-12)


@pytest.mark.parametrize('rho_value', [0., 1., .999])
def test_retention_edges_no_inverse_state_reconstruction(rho_value):
    args = list(inputs(steps=8))
    args[1] = torch.full_like(args[1], rho_value, requires_grad=True)
    actual, expected = scan(*args), reference(*args)
    ga = torch.autograd.grad(actual[:, -1].square().sum(), args)
    gr = torch.autograd.grad(expected[:, -1].square().sum(), args)
    torch.testing.assert_close(actual, expected, atol=2e-14, rtol=2e-14)
    for x, y in zip(ga, gr): torch.testing.assert_close(x, y, atol=4e-13, rtol=3e-12)


def test_zero_angles_and_collinear_planes_have_exact_gradients():
    args = list(inputs(steps=3))
    args[2] = torch.zeros_like(args[2], requires_grad=True)
    args[4] = args[3].detach().clone().requires_grad_()
    actual, expected = scan(*args), reference(*args)
    ga = torch.autograd.grad(actual.sum(), args)
    gr = torch.autograd.grad(expected.sum(), args)
    for x, y in zip(ga, gr): torch.testing.assert_close(x, y, atol=2e-13, rtol=2e-12)


def test_float32_long_sequence_and_causal_prefix():
    args = inputs(batch=2, steps=216, dtype=torch.float32)
    actual, expected = scan(*args), reference(*args)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    ga = torch.autograd.grad(actual.square().sum(), args)
    gr = torch.autograd.grad(expected.square().sum(), args)
    for x, y in zip(ga, gr): torch.testing.assert_close(x, y, atol=2e-5, rtol=2e-4)
    short = tuple(x[:, :37] if i < 3 else x for i, x in enumerate(args))
    torch.testing.assert_close(scan(*short), actual[:, :37], atol=0, rtol=0)


def test_noncontiguous_inputs_and_external_output_gradient():
    args = inputs(batch=2, steps=6)
    sliced = tuple(x[:, ::2] if j < 3 else x for j, x in enumerate(args))
    actual, expected = scan(*sliced), reference(*sliced)
    grad = torch.ones_like(actual).transpose(0, 1).contiguous().transpose(0, 1)
    ga = torch.autograd.grad(actual, args, grad)
    gr = torch.autograd.grad(expected, args, grad)
    for x, y in zip(ga, gr): torch.testing.assert_close(x, y, atol=3e-13, rtol=3e-12)


def test_checkpoint_cold_and_warm_recomputation():
    # Checkpoint sees a fixed save_for_backward tuple from the custom Function;
    # JIT-internal profiling must not change saved-tensor metadata ordering.
    for _ in range(3):
        args = inputs(batch=2, steps=7)
        actual = checkpoint(scan, *args, use_reentrant=False, preserve_rng_state=False)
        expected = reference(*args)
        ga = torch.autograd.grad(actual.square().sum(), args)
        gr = torch.autograd.grad(expected.square().sum(), args)
        for x, y in zip(ga, gr): torch.testing.assert_close(x, y, atol=3e-13, rtol=3e-12)


def test_chained_blocks_initial_gradient():
    args = inputs(steps=9)
    first = scan(*(x[:, :4] if i < 3 else x for i, x in enumerate(args)))
    second_args = (args[0][:, 4:], args[1][:, 4:], args[2][:, 4:], args[3], args[4], first[:, -1])
    actual = torch.cat((first, scan(*second_args)), 1)
    expected = reference(*args)
    ga = torch.autograd.grad(actual.square().sum(), args)
    gr = torch.autograd.grad(expected.square().sum(), args)
    for x, y in zip(ga, gr): torch.testing.assert_close(x, y, atol=3e-13, rtol=3e-12)


def test_rejects_wrong_shapes_and_mixed_dtypes():
    args = inputs()
    with pytest.raises(ValueError): scan(args[0], args[1][:, :2], *args[2:])
    with pytest.raises(ValueError): scan(args[0].float(), *args[1:])
    with pytest.raises(ValueError): scan(*args[:-1], args[-1][:1])
    with pytest.raises(TypeError): scan(*(x.half() for x in args))
