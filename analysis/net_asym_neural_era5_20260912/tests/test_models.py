"""Correctness gates for the NET/ASYM core (protocol section 7.3)."""
import copy, inspect
import numpy as np, pytest, torch
from core_models_base import TrajectoryModel
from models import KINDS, build_model, n_params

DIMS = {'us': (600, 15, 24), 'aneel': (168, 5, 24), 'synthetic': (32, 2, 32)}


def inputs(ctx, clk, H, B=16, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(B, ctx, generator=g, dtype=dtype),
            torch.rand(B, generator=g, dtype=dtype) * .3,
            torch.randn(B, H, clk, generator=g, dtype=dtype))


def test_only_net_and_asym_can_be_built():
    assert KINDS == ('NET', 'ASYM')
    for bad in ('DIRECT', 'SR', 'net', ''):
        with pytest.raises(ValueError):
            build_model(bad, 32, 2, 32, 8192, .01, .01)


def test_forward_interface_accepts_no_target():
    assert list(inspect.signature(TrajectoryModel.forward).parameters) == ['self', 'c', 'y0', 'known_clock']


@pytest.mark.parametrize('kind', KINDS)
def test_changing_target_does_not_change_forward(kind):
    m = build_model(kind, 32, 2, 32, 8192, .05, .1).double()
    c, y0, kc = inputs(32, 2, 32)
    batch = {'c': c, 'y0': y0, 'step': kc, 'target': torch.rand(16, 32, dtype=torch.float64)}
    p1 = m(batch['c'], batch['y0'], batch['step'])
    batch['target'] = batch['target'] * -7 + 3
    assert torch.equal(p1, m(batch['c'], batch['y0'], batch['step']))


def test_net_represents_a_general_affine_net_response_that_signed_single_rate_cannot():
    # scale 0.0625 is exact in float32: the model stores state_scale as a float32 buffer before
    # .double(), so an inexact scale would add a spurious ~1e-9 per-step constant error
    s, a, b = .0625, .03, -.12                  # increment a + b*y changes sign at y = .25
    m = build_model('NET', 32, 2, 32, 8192, s, .1).double()
    with torch.no_grad():
        for p in m.parameters():
            p.zero_()
        m.net.body[-1].bias.fill_(a / s)         # constant part
        m.net.linear.weight[0, -1] = b           # last NET input column is y/scale
    c, _, kc = inputs(32, 2, 32)
    for y0v in (0.0, .1, .6, .95):
        y0 = torch.full((16,), y0v, dtype=torch.float64)
        y, ref = y0.clone(), []
        for _ in range(32):
            y = y + a + b * y; ref.append(y)
        assert torch.allclose(m(c, y0, kc), torch.stack(ref, 1), atol=1e-12, rtol=0)
    lo = m(c, torch.full((16,), .1, dtype=torch.float64), kc)[:, 0] - .1
    hi = m(c, torch.full((16,), .9, dtype=torch.float64), kc)[:, 0] - .9
    assert (lo > 0).all() and (hi < 0).all()     # both signs under the same context and step
    for sv in np.linspace(-1, 1, 401):           # signed single-rate: one sign for every y in (0,1)
        i_lo = max(sv, 0) * (1 - .1) - max(-sv, 0) * .1
        i_hi = max(sv, 0) * (1 - .9) - max(-sv, 0) * .9
        assert not (i_lo > 0 and i_hi < 0)


def test_net_reads_its_own_recursive_state_and_asym_rates_do_not():
    torch.manual_seed(1)
    m = build_model('NET', 32, 2, 32, 8192, .05, .1).double()
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, .2)
    c, y0, kc = inputs(32, 2, 32)
    d1 = m(c, y0, kc)[:, 1] - m(c, y0, kc)[:, 0]
    d2 = m(c, y0 + .2, kc)[:, 1] - m(c, y0 + .2, kc)[:, 0]
    assert not torch.allclose(d1, d2)            # increment depends on the state it reached
    a = build_model('ASYM', 32, 2, 32, 8192, .05, .1).double()
    assert list(inspect.signature(a.schedule).parameters) == ['c', 'known_clock']


@pytest.mark.parametrize('H', (24, 32))
@pytest.mark.parametrize('std,tol', ((.5, 0.0), (3.0, 1e-12)))
def test_asym_recursion_stays_in_unit_interval_from_legal_start(H, std, tol):
    torch.manual_seed(3)
    m = build_model('ASYM', 32, 2, H, 8192, .05, .2).double()
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, std)
    g = torch.Generator().manual_seed(4)
    c = torch.randn(512, 32, generator=g, dtype=torch.float64) * 3
    kc = torch.randn(512, H, 2, generator=g, dtype=torch.float64) * 3
    for y0 in (torch.zeros(512), torch.ones(512), torch.rand(512, generator=g)):
        p = m(c, y0.double(), kc)
        assert p.min() >= -tol and p.max() <= 1 + tol
    u, r = m.schedule(c, kc)
    assert (u >= 0).all() and (r >= 0).all() and ((u + r) <= 1 + 1e-15).all()


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('name', DIMS)
def test_cpu_fp64_reference_matches_fp32_device(kind, name):
    ctx, clk, H = DIMS[name]
    torch.manual_seed(11)
    m32 = build_model(kind, ctx, clk, H, 32768, .0188, .003)
    with torch.no_grad():
        for p in m32.parameters():
            p.add_(torch.randn_like(p) * .05)
    m64 = copy.deepcopy(m32).double()
    c, y0, kc = inputs(ctx, clk, H, B=64, seed=12)
    tgt = torch.rand(64, H, dtype=torch.float64) * .3
    p64 = m64(c, y0, kc); l64 = (((p64 - tgt) / .0188) ** 2).mean(); l64.backward()
    p32 = m32(c.float(), y0.float(), kc.float()); l32 = (((p32 - tgt.float()) / .0188) ** 2).mean(); l32.backward()
    assert torch.allclose(p32.double(), p64, atol=1e-6, rtol=1e-4)
    assert abs(float(l32) - float(l64)) <= 1e-4 * abs(float(l64)) + 1e-6
    rel = [float((a.grad.double() - b.grad).norm() / b.grad.norm())
           for a, b in zip(m32.parameters(), m64.parameters()) if float(b.grad.norm()) > 0]
    assert max(rel) <= 1e-3


@pytest.mark.parametrize('name', DIMS)
def test_shapes_time_alignment_and_endpoint_indices(name):
    ctx, clk, H = DIMS[name]
    for kind in KINDS:
        m = build_model(kind, ctx, clk, H, 8192, .02, .05).double()
        c, y0, kc = inputs(ctx, clk, H, B=8)
        p = m(c, y0, kc)
        assert p.shape == (8, H)
        if kind == 'NET':
            f0 = m.net(torch.cat([c, kc[:, 0], (y0 / m.state_scale)[:, None]], -1)).squeeze(-1)
            assert torch.allclose(p[:, 0], y0 + m.state_scale * f0, atol=1e-14)
        else:
            u, r = m.schedule(c, kc)
            assert torch.allclose(p[:, 0], y0 + u[:, 0] * (1 - y0) - r[:, 0] * y0, atol=1e-14)
    ends = {'1h': 0, '6h': 5, '24h': 23} | ({'32h': 31} if H == 32 else {})
    assert all(j == int(k[:-1]) - 1 < H for k, j in ends.items())


@pytest.mark.parametrize('kind', KINDS)
def test_gradient_from_last_step_reaches_first_step_input(kind):
    m = build_model(kind, 32, 2, 32, 8192, .05, .1).double()
    c, y0, kc = inputs(32, 2, 32, B=8)
    kc.requires_grad_(True)
    m(c, y0, kc)[:, -1].sum().backward()
    assert float(kc.grad[:, 0].abs().sum()) > 0


@pytest.mark.parametrize('name', DIMS)
@pytest.mark.parametrize('target', (8192, 32768))
def test_parameter_budget_within_five_percent(name, target):
    ctx, clk, H = DIMS[name]
    for kind in KINDS:
        n = n_params(build_model(kind, ctx, clk, H, target, .02, .05))
        assert abs(n - target) / target < .05, (kind, name, target, n)
