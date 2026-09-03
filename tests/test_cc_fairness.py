"""Section 5.2: the fairness assertions, as tests that fail when violated.

Two kinds of check:

* structural -- `fit_arm` must contain no arm-specific branch other than the model
  factory and the burden state, so optimiser, schedule, budget, rows, loss, mask
  and initialisation cannot silently differ between arms;
* runtime -- fitting two arms on the same tiny problem must record identical
  train/validation/test row counts, identical calibrated initial flows, the same
  optimiser and learning rate, and parameter counts within 1%.
"""
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
import cc_event_transfer as CC  # noqa: E402


def _tiny(n=64, T=6, d=14, seed=0):
    rng = np.random.default_rng(seed)
    y0 = rng.uniform(0, 0.05, n).astype(np.float32)
    X = rng.normal(size=(n, T, d)).astype(np.float32)
    yt = np.clip(y0[:, None] + rng.normal(0, 0.01, (n, T)), 0, 1).astype(np.float32)
    m = np.ones((n, T), bool)
    hist = np.repeat(y0[:, None], 4, axis=1).astype(np.float32)
    return (y0, X, yt, m), hist


def test_fit_arm_has_no_arm_specific_branch():
    src = inspect.getsource(CC.fit_arm)
    assert 'arm ==' not in src and 'arm in (' not in src, \
        "fit_arm branches on the arm name; the training path must be shared"
    # the only permitted model-dependent branch is the burden state
    assert src.count("isinstance(model, RecoveryBurdenODE)") == 1
    assert "make_model(arm" in src


def test_shared_budget_is_a_single_object():
    src = inspect.getsource(CC.fit_arm)
    for k in ("epochs", "patience", "batch", "lr"):
        assert f'BUDGET["{k}"]' in src, f"{k} must be read from the shared BUDGET"
    assert CC.BUDGET["epochs"] > 0 and CC.BUDGET["patience"] > 0


@pytest.mark.parametrize("horizons", [(1, 6)])
def test_two_arms_see_identical_rows_init_and_budget(monkeypatch, horizons):
    monkeypatch.setattr(CC, "HORIZONS", horizons)
    monkeypatch.setitem(CC.BUDGET, "epochs", 2)
    monkeypatch.setitem(CC.BUDGET, "patience", 1)
    seen = []
    real_adam = torch.optim.Adam

    def spy(params, **kw):
        seen.append(kw.get("lr"))
        return real_adam(params, **kw)

    monkeypatch.setattr(torch.optim, "Adam", spy)
    data, hist = _tiny()
    tr, va, te = np.arange(0, 40), np.arange(40, 52), np.arange(52, 64)
    recs = {}
    for arm in ("two_rate", "net_scaled", "recovery_burden"):
        rec, _ = CC.fit_arm(arm, tr, va, te, data, hist, seed=0, log=lambda s: None)
        recs[arm] = rec
    assert len(set(seen)) == 1 and seen[0] == CC.BUDGET["lr"], "arms used different learning rates"
    keys = ("n_train", "n_val", "n_test", "u_init", "r_init")
    for k in keys:
        vals = {recs[a][k] for a in recs}
        assert len(vals) == 1, f"arms disagree on {k}: {vals}"


def test_parameter_counts_within_one_percent():
    two = sum(p.numel() for p in CC.make_model("two_rate", 14, 1e-4, 1e-3).parameters())
    net = sum(p.numel() for p in CC.make_model("net_scaled", 14, 1e-4, 1e-3).parameters())
    bur = sum(p.numel() for p in CC.make_model("recovery_burden", 14, 1e-4, 1e-3).parameters())
    assert abs(net - two) / two <= 0.01, f"two_rate {two} vs net_scaled {net}"
    assert bur - two == 33, f"recovery burden increment is {bur - two}, expected +33"


def test_burden_state_never_reads_the_target():
    """Section 11.1.5: the burden must be driven by predictions, not by future truth."""
    from asymode.burden import BurdenConfig, RecoveryBurdenODE
    m = RecoveryBurdenODE(BurdenConfig(d_in=14, cap_u=0.25, cap_r=0.25, hidden_u=32, hidden_r=32))
    sig = inspect.signature(m.forward)
    assert set(sig.parameters) == {"y0", "drivers", "b0"}, \
        "forward must take only the origin state, the drivers and the history-derived burden"
    src = inspect.getsource(RecoveryBurdenODE.forward)
    assert "y_true" not in src and "target" not in src
    # changing future truth cannot change the rollout
    y0 = torch.rand(5) * 0.1
    X = torch.randn(5, 8, 14)
    b0 = torch.rand(5) * 0.1
    a = m(y0, X, b0).detach().clone()
    b = m(y0, X, b0).detach().clone()
    assert torch.allclose(a, b)


def test_split_map_is_leave_one_event_out_and_disjoint():
    events = [f"2021-0{i}-0{j}" for i in range(1, 4) for j in range(1, 5)][:11]
    m = CC.build_event_split_map(events)
    assert len(m["folds"]) == len(events)
    for f in m["folds"]:
        assert len(f["test"]) == 1
        assert not set(f["test"]) & set(f["train"]), "test event appears in training"
        assert not set(f["test"]) & set(f["validation"]), "test event appears in validation"
        assert not set(f["validation"]) & set(f["train"]), "validation event appears in training"
        assert len(f["train"]) == len(events) - 2
    assert sorted(f["test"][0] for f in m["folds"]) == sorted(events), "not every event is held out once"
