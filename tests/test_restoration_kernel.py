"""Restoration kernel (asymode.asym_host.RestorationKernel) on synthetic inputs only.

  * before the warm-up opens it, and with all weights zero after, the model equals its host;
  * a positive own-state or regional weight can only slow the restoration: the stock never falls below the host's;
  * the stock stays inside [0, 1];
  * the weights receive finite, non-zero gradients at zero once the kernel is open;
  * project_ keeps the weights non-negative and the disabled part at zero.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from asymode.asym_host import AsymODE, stock_path_capacity, stock_path_loop, trajectory_loss  # noqa: E402


def batch(B=6, seed=0, rings=3):
    g = torch.Generator().manual_seed(seed)
    return dict(xu=torch.randn(B, 216, 10, generator=g), xr=torch.randn(B, 216, 8, generator=g), xo=torch.randn(B, 216, 3, generator=g),
                ctx=torch.randn(B, 4, generator=g), geo=torch.randn(B, 5, generator=g), y0=torch.rand(B, generator=g) * 0.4,
                burden=torch.rand(B, rings, generator=g) * 0.2, y=torch.rand(B, 144, generator=g) * 0.3, m=torch.ones(B, 144))


def model(rest: bool, **kw):
    torch.manual_seed(3)
    net = AsymODE(10, 8, 3)
    net.attach_context_input(4)
    if rest:
        net.attach_restoration(3, **kw)
    return net.eval()


def test_recursion_equals_host_at_zero():
    g = torch.Generator().manual_seed(1)
    u, r, y0 = torch.rand(5, 144, generator=g) * 0.05, torch.rand(5, 144, generator=g) * 0.5, torch.rand(5, generator=g)
    assert torch.equal(stock_path_capacity(u, r, y0, torch.zeros(()), torch.zeros(5)), stock_path_loop(u, r, y0))


def test_equals_host_at_start():
    b = batch()
    base, net = model(False)(b), model(True)
    assert torch.equal(base["P"], net(b)["P"])
    net.rest.open = True
    assert torch.allclose(base["P"], net(b)["P"], atol=1e-7, rtol=0)


def test_slower_and_bounded():
    b = batch()
    net = model(True); net.rest.open = True
    base = net(b)["P"]
    with torch.no_grad():
        net.rest.theta_l.fill_(1.0)
    own = net(b)["P"]
    assert (own >= base - 1e-6).all() and (own > base + 1e-6).any()
    with torch.no_grad():
        net.rest.theta_l.zero_(); net.rest.theta_b.fill_(0.5)
    reg = net(b)["P"]
    assert (reg >= base - 1e-6).all() and (reg > base + 1e-6).any()
    with torch.no_grad():
        net.rest.theta_l.fill_(50.0); net.rest.theta_b.fill_(50.0)
    big = net(b)["P"]
    assert float(big.min()) >= 0.0 and float(big.max()) <= 1.0 and torch.isfinite(big).all()


def test_gradients_and_projection():
    b = batch()
    net = model(True).train(); net.rest.open = True
    trajectory_loss(net(b)["P"], b["y"], b["m"]).backward()
    for p in (net.rest.theta_l, net.rest.theta_b):
        assert p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.abs().sum()) > 0
    for n, p in net.named_parameters():
        assert p.grad is None or torch.isfinite(p.grad).all(), n
    with torch.no_grad():
        net.rest.theta_l.fill_(-1.0); net.rest.theta_b.fill_(-1.0)
    net.rest.project_()
    assert float(net.rest.theta_l) == 0.0 and float(net.rest.theta_b.abs().sum()) == 0.0
    local = model(True, spatial=False)
    with torch.no_grad():
        local.rest.theta_l.fill_(0.3); local.rest.theta_b.fill_(0.3)
    local.rest.project_()
    assert float(local.rest.theta_l) > 0 and float(local.rest.theta_b.abs().sum()) == 0.0


def test_damage_side_and_recovery_inputs():
    b = batch()
    base = model(False)(b)["P"]
    net = model(True, side="damage"); net.rest.open = True
    assert torch.allclose(base, net(b)["P"], atol=1e-7, rtol=0)
    with torch.no_grad():
        net.rest.theta_b.fill_(1.0)
    up = net(b)
    assert (up["u"] >= model(False)(b)["u"] - 1e-7).all() and (up["P"] >= base - 1e-6).all() and (up["P"] > base + 1e-6).any()
    assert float(up["u"].max()) <= 0.515 + 1e-6
    wide = model(False); wide.expand_recovery_inputs(3)
    bb = dict(b); bb["xr"] = torch.cat([b["xr"], torch.randn(b["xr"].shape[0], 216, 3)], -1)
    assert torch.equal(wide(bb)["P"], base)
