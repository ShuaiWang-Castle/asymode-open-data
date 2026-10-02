"""Dose-fragility kernel (asymode.asym_host.DoseKernel) on synthetic inputs only.

  * with beta = eta = 0 the model equals its host exactly (paired initialisation);
  * a positive beta can only raise the damage rate and the stock, a positive eta can only lower the restoration rate;
  * the rates stay inside their caps and the stock inside [0, 1];
  * gradients of every dose parameter are finite, and beta and eta receive a non-zero gradient at the start;
  * project_ keeps beta and eta non-negative and, for the restoration-only variant, beta at zero.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from asymode.asym_host import AsymODE, trajectory_loss  # noqa: E402


def batch(B=6, seed=0, m=7):
    g = torch.Generator().manual_seed(seed)
    return dict(xu=torch.randn(B, 216, 10, generator=g), xr=torch.randn(B, 216, 8, generator=g), xo=torch.randn(B, 216, 3, generator=g),
                ctx=torch.randn(B, 4, generator=g), geo=torch.randn(B, 5, generator=g), y0=torch.rand(B, generator=g) * 0.05,
                vuln=torch.randn(B, m, generator=g), y=torch.rand(B, 144, generator=g) * 0.3, m=torch.ones(B, 144))


def model(dose: bool, m=7, damage_side=True):
    torch.manual_seed(3)
    net = AsymODE(10, 8, 3)
    net.attach_context_input(4)
    if dose:
        net.attach_dose(m, damage_side)
        net.dose.calibrate_(net.hidden(batch()["xu"], batch()["ctx"]))
    return net.eval()


def test_equals_host_at_start():
    b = batch()
    assert torch.equal(model(False)(b)["P"], model(True)(b)["P"])
    assert torch.equal(model(False)(b)["u"], model(True)(b)["u"])


def test_monotone_and_bounded():
    b = batch()
    net = model(True)
    base = net(b)
    with torch.no_grad():
        net.dose.beta.fill_(0.7)
    up = net(b)
    assert (up["u"] >= base["u"] - 1e-7).all() and (up["P"] >= base["P"] - 1e-6).all() and (up["u"] > base["u"]).any()
    with torch.no_grad():
        net.dose.beta.zero_(); net.dose.eta.fill_(0.7)
    slow = net(b)
    assert (slow["r"] <= base["r"] + 1e-7).all() and (slow["r"] < base["r"]).any() and (slow["P"] >= base["P"] - 1e-6).all()
    with torch.no_grad():
        net.dose.beta.fill_(50.0); net.dose.eta.fill_(50.0)
    big = net(b)
    assert float(big["u"].max()) <= 0.515 + 1e-6 and float(big["r"].min()) >= 0.0
    assert float(big["P"].min()) >= 0.0 and float(big["P"].max()) <= 1.0 + 1e-6


def test_gradients_and_projection():
    b = batch()
    net = model(True).train()
    loss = trajectory_loss(net(b)["P"], b["y"], b["m"])
    loss.backward()
    assert float(net.dose.beta.grad.abs().sum()) > 0 and float(net.dose.eta.grad.abs().sum()) > 0
    with torch.no_grad():
        net.dose.beta.fill_(0.3); net.dose.eta.fill_(0.3)
    net.zero_grad()
    trajectory_loss(net(b)["P"], b["y"], b["m"]).backward()
    for n, p in net.dose.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), n
    with torch.no_grad():
        net.dose.beta.fill_(-1.0); net.dose.eta.fill_(-1.0)
    net.dose.project_()
    assert float(net.dose.beta.min()) == 0.0 and float(net.dose.eta.min()) == 0.0
    r_only = model(True, damage_side=False)
    with torch.no_grad():
        r_only.dose.beta.fill_(0.4)
    r_only.dose.project_()
    assert float(r_only.dose.beta.abs().sum()) == 0.0
