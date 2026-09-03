from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "unified_aistats_v2", ROOT / "experiments/unified_aistats_v2.py"
)
V2 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(V2)


def test_split_map_is_exactly_one_test_two_validation_eight_train():
    raw = json.loads((ROOT / "configs/event_split_map_g2_two_validation.json").read_text())
    events = raw["events"]
    smap = V2.load_split_map(events)
    assert len(smap["folds"]) == 11
    for fold in smap["folds"]:
        assert len(fold["test"]) == 1
        assert len(fold["validation"]) == 2
        assert len(fold["train"]) == 8
        assert not (set(fold["test"]) & set(fold["validation"]))
        assert not (set(fold["test"]) & set(fold["train"]))
        assert not (set(fold["validation"]) & set(fold["train"]))
        assert set(fold["test"] + fold["validation"] + fold["train"]) == set(events)


def test_parameter_matched_structural_comparator():
    two = V2.make_model("two_rate_v2", 14, 1e-3, 2e-3)
    one = V2.make_model("net_scaled_v2", 14, 1e-3, 2e-3)
    n_two = sum(p.numel() for p in two.parameters())
    n_one = sum(p.numel() for p in one.parameters())
    assert abs(n_one - n_two) / n_two <= 0.01


def test_step_mask_requires_current_and_next_observation():
    mask = torch.tensor([[1.0, 0.0, 1.0, 1.0]])
    got = V2.step_mask(mask)
    want = torch.tensor([[1.0, 0.0, 0.0, 1.0]])
    assert torch.equal(got, want)


def test_teacher_forced_two_rate_matches_manual_transition():
    model = V2.make_model("two_rate_v2", 14, 1e-3, 2e-3)
    model.eval()
    y0 = torch.tensor([0.1, 0.2])
    drivers = torch.zeros(2, 3, 14)
    truth = torch.tensor([[0.11, 0.12, 0.13], [0.19, 0.18, 0.17]])
    got = V2.teacher_forced_prediction(model, y0, drivers, truth)
    current = torch.cat([y0[:, None], truth[:, :-1]], dim=1)
    manual = []
    for t in range(3):
        u, r = model.rates(drivers[:, t], current[:, t])
        manual.append(torch.clamp(current[:, t] + u * (1-current[:, t]) - r * current[:, t], 0, 1))
    want = torch.stack(manual, dim=1)
    assert torch.allclose(got, want, atol=1e-7, rtol=1e-7)


def test_teacher_forced_net_scaled_matches_manual_transition():
    model = V2.make_model("net_scaled_v2", 14, 1e-3, 2e-3)
    model.eval()
    y0 = torch.tensor([0.1, 0.2])
    drivers = torch.zeros(2, 3, 14)
    truth = torch.tensor([[0.11, 0.12, 0.13], [0.19, 0.18, 0.17]])
    got = V2.teacher_forced_prediction(model, y0, drivers, truth)
    current = torch.cat([y0[:, None], truth[:, :-1]], dim=1)
    manual = []
    for t in range(3):
        n, _ = model.rates(drivers[:, t], current[:, t])
        delta = torch.where(n > 0, n * (1-current[:, t]), n * current[:, t])
        manual.append(torch.clamp(current[:, t] + delta, 0, 1))
    want = torch.stack(manual, dim=1)
    assert torch.allclose(got, want, atol=1e-7, rtol=1e-7)


def test_event_objective_is_invariant_to_row_duplication_within_event():
    model = V2.make_model("two_rate_v2", 14, 1e-3, 2e-3)
    model.eval()
    rng = np.random.default_rng(0)
    y0 = torch.tensor(rng.uniform(0, 0.1, size=4), dtype=torch.float32)
    x = torch.tensor(rng.normal(size=(4, 24, 14)), dtype=torch.float32)
    y = torch.tensor(rng.uniform(0, 0.1, size=(4, 24)), dtype=torch.float32)
    m = torch.ones_like(y)
    base = V2.event_objective(model, np.arange(4), (y0, x, y, m), 0.5, 0.5, False)[0]
    y0d = torch.cat([y0, y0])
    xd = torch.cat([x, x])
    yd = torch.cat([y, y])
    md = torch.cat([m, m])
    dup = V2.event_objective(model, np.arange(8), (y0d, xd, yd, md), 0.5, 0.5, False)[0]
    assert np.isclose(base, dup, rtol=1e-6, atol=1e-8)
