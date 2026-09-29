"""D07 C synthetic contracts: weighting, sliced steps, exact restoration/support."""
from __future__ import annotations
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments/geo_weather_20260924'))
import d07_gradients as D


class SmallKernel(nn.Module):
    def __init__(self, dtype=torch.float64):
        super().__init__()
        for name, shape in {
            'weight': (2, 2), 'bias': (2,), 'fusion_geo': (2, 72),
            'head_geo': (3, 72), 'fusion_weather': (2, 4), 'head_weather': (3, 4),
            'head_fusion': (3, 2), 'fusion_bias': (2,), 'head_bias': (3,),
            'length_raw': (2,), 'order_logits': (2,), 'plane_a': (2, 2),
            'plane_b': (2, 2), 'readout': (2, 2), 'alpha': ()}.items():
            self.register_parameter(name, nn.Parameter(torch.randn(shape, dtype=dtype)))
        self.register_buffer('frozen_scale', torch.tensor(1.7, dtype=dtype))
        self.metadata = {'landmarks': ['a', 'b'], 'step': 900}

    def get_extra_state(self):
        return self.metadata.copy()

    def set_extra_state(self, state):
        self.metadata = state.copy()


class SmallModel(nn.Module):
    def __init__(self, dtype=torch.float64):
        super().__init__()
        self.recovery = nn.Linear(4, 1, dtype=dtype)
        self.damage = nn.Sequential(nn.Linear(4, 2, dtype=dtype), nn.ReLU(),
                                    SmallKernel(dtype), nn.ReLU(), nn.Linear(2, 1, dtype=dtype))
        self.ctx_in = nn.Linear(2, 2, bias=False, dtype=dtype)


def test_original_float32_weighting_and_group_additivity_not_group_denominators():
    torch.manual_seed(6)
    x = torch.randn(9, 4, 3, dtype=torch.float64)
    y = torch.randn(9, 4, dtype=torch.float64)
    mask = torch.tensor([[1, 1, 0, 1], [1, 0, 0, 0], [1, 1, 1, 1],
                         [0, 1, 1, 0], [1, 1, 1, 1], [1, 0, 1, 0],
                         [1, 1, 0, 1], [0, 0, 1, 1], [1, 1, 1, 0]], dtype=torch.float64)
    design = torch.tensor([1., 4., 2., 8., 1.5, 6., 3., 9., 2.], dtype=torch.float64)
    regime = torch.tensor([0, 1, 0, 1, 2, 0, 2, 1, 2])
    z = torch.stack([(design[:, None]*mask*y.square()*(regime == r)[:, None]).sum()
                     for r in range(3)])
    rw = design/z[regime]
    scale = mask.sum()/(mask*rw[:, None]).sum()
    denominator = (mask*rw[:, None]).sum()
    theta = torch.randn(3, dtype=torch.float64, requires_grad=True)
    original = D.weighted_objective(x@theta, y, mask, rw, denominator)
    total = torch.autograd.grad(original, theta)[0]
    parts, losses, wrong = [], [], []
    for ids in ([0, 2], [1, 3, 7], [4], [5, 6, 8]):
        p = x[ids]@theta
        loss = D.weighted_objective(p, y[ids], mask[ids], rw[ids], denominator)
        parts.append(torch.autograd.grad(loss, theta)[0]); losses.append(float(loss.detach()))
        other = D.weighted_objective(x[ids]@theta, y[ids], mask[ids], rw[ids],
                                    (mask[ids]*rw[ids, None]).sum())
        wrong.append(torch.autograd.grad(other, theta)[0])
    torch.testing.assert_close(sum(parts), total, atol=1e-12, rtol=1e-12)
    assert np.isclose(sum(losses), float(original.detach()), rtol=1e-12)
    assert not torch.allclose(sum(wrong), total)
    # Verify the native training weight multiplication/order in float32.
    P32, y32, m32 = (x@theta).detach().float(), y.float(), mask.float()
    rw32 = (rw*scale).float()
    mtrain = m32*rw32[:, None]
    native = ((P32-y32).square()*mtrain).sum()/mtrain.sum()
    torch.testing.assert_close(D.weighted_objective(P32, y32, m32, rw32, mtrain.sum()),
                               native, rtol=0, atol=0)


def test_blocks_cover_every_coordinate_once_and_keep_reused_W2_in_host():
    torch.manual_seed(3)
    model = SmallModel(); spec = D.layout(model)
    blocks, offsets = spec['blocks'], spec['offsets']
    assigned = np.concatenate(list(blocks.values()))
    assert np.array_equal(np.sort(assigned), np.arange(spec['size']))
    for field in ('fusion_geo', 'head_geo'):
        ix = offsets['damage.2.'+field]
        assert set(ix[:, :40].reshape(-1)) <= set(blocks['raw_geography'])
        assert set(ix[:, 40:].reshape(-1)) <= set(blocks['kernel_basis'])
        assert not set(ix[:, :40].reshape(-1)) & set(blocks['kernel_basis'])
    for field in ('weight', 'bias'):
        assert set(offsets['damage.2.'+field].reshape(-1)) <= set(blocks['host'])


def test_exactly_six_independent_column_perturbations_and_exact_state_restoration():
    torch.manual_seed(19)
    model = SmallModel(); spec = D.layout(model); frozen = D.snapshot(model)
    frozen_hash = D.state_hash(frozen)
    gradient = np.random.default_rng(9).normal(size=spec['size'])
    for block in D.PERTURB_BLOCKS:
        for epsilon in D.EPSILONS:
            info = D.apply_perturbation(model, spec, gradient, block, epsilon)
            assert np.isclose(info['achieved_relative_step'], epsilon, rtol=1e-12)
            assert np.isclose(info['descent_cosine'], 1., rtol=1e-12)
            assert info['first_order_full_fit_objective_delta'] < 0
            assert D.state_hash(model.state_dict()) != frozen_hash
            D.restore(model, frozen)
            assert D.state_hash(model.state_dict()) == frozen_hash
    assert torch.equal(model.damage[2].frozen_scale, frozen['damage.2.frozen_scale'])
    with pytest.raises(ValueError, match='unregistered'):
        D.apply_perturbation(model, spec, gradient, 'host', .001)
    with pytest.raises(ValueError, match='unregistered'):
        D.apply_perturbation(model, spec, gradient, 'weather_control', .02)
    D.apply_perturbation(model, spec, gradient, 'raw_geography', .001)
    with pytest.raises(AssertionError, match='frozen baseline'):
        D.apply_perturbation(model, spec, gradient, 'raw_geography', .01)
    D.restore(model, frozen)


def test_zero_norm_parameter_floor_and_undefined_zero_gradient_direction():
    model = SmallModel()
    with torch.no_grad():
        model.damage[2].fusion_weather.zero_(); model.damage[2].head_weather.zero_()
    spec = D.layout(model); frozen = D.snapshot(model)
    grad = np.ones(spec['size'])
    result = D.apply_perturbation(model, spec, grad, 'weather_control', .001)
    assert result['baseline_parameter_l2'] == 0
    assert result['parameter_norm_for_step'] == .001
    assert np.isclose(result['achieved_step_l2'], 1e-6)
    D.restore(model, frozen)
    grad[spec['blocks']['kernel_basis']] = 0
    with pytest.raises(ValueError, match='zero/undefined'):
        D.apply_perturbation(model, spec, grad, 'kernel_basis', .001)


def test_cancellation_and_zero_cosines_are_missing():
    a = np.array([1., 2., -3.])
    result = D.direction_report(np.stack([a, -a]), ['S', 'nonS'], a*0)
    assert result['cancellation_fraction'] == 1
    assert result['top_norm_contributors'][0]['cosine_to_total'] is None
    assert result['pairwise_cosines'][0][1] == pytest.approx(-1)
    assert D.cosine(a*0, a) is None
    good, reconstructed = D.additivity_report(np.stack([a, -a]), a*0)
    assert good['passed'] and np.array_equal(reconstructed, a*0)
    bad, _ = D.additivity_report(np.stack([a, -a+.1]), a*0)
    assert not bad['passed']


def test_partitions_use_original_systems_even_when_merged_group_is_shared():
    data = dict(y=np.zeros((6, 3)),
                meta={'system': np.array(['a', 'a', 'b', 'b', 'c', 'c'])},
                group=np.array(['merged_ab', 'merged_ab', 'merged_ab', 'merged_ab', 'merged_c', 'merged_c']))
    S = np.array([True, False, True, False, False, True])
    systems, rows = D.system_partitions(data, [0, 1, 2, 3], {'S': S, 'nonS': ~S})
    assert systems.tolist() == ['a', 'b']
    assert [(s, c, g) for s, c, g, ids in rows] == [
        ('a', 'S', 'merged_ab'), ('a', 'nonS', 'merged_ab'),
        ('b', 'S', 'merged_ab'), ('b', 'nonS', 'merged_ab')]
    assert np.array_equal(np.sort(np.concatenate([ids for s, c, g, ids in rows])), np.arange(4))


def test_observed_support_false_alarms_and_paired_deltas_are_explicit():
    y = np.zeros((3, 3), float)
    m = np.array([[1, 0, 1], [1, 1, 1], [1, 1, 0]], bool)
    baseline = np.array([[.04, .95, .03], [.11, .02, .04], [.01, .03, .98]])
    P = np.array([[.05, .99, .02], [.09, .02, .04], [.11, .03, .98]])
    w = np.array([2., 3., 5.]); rw = np.array([.5, 2., 1.])
    data = dict(y=y, m=m, meta={'w': w, 'system': np.array(['a1', 'a2', 'b1'])},
                group=np.array(['a', 'a', 'b']))
    masks = dict(all=np.ones(3, bool), nonS=np.ones(3, bool),
                 S=np.zeros(3, bool), J=np.array([False, False, True]))
    result = D.paired_metrics(data, P, baseline, np.arange(3), masks, rw, 17.)
    row = result['nonS']
    assert row['observed_severe_false_alarms']['n'] == 1
    assert row['baseline_observed_severe_false_alarms']['n'] == 1
    assert row['paired_severe_false_alarms'] == dict(new=1, removed=1, retained=0, weighted_rate_delta=.2)
    assert row['observed_hours'] == 7
    assert row['design_observed_hour_mass'] == 23
    expected = np.sum(rw[:, None]*m*(P**2-baseline**2))/17
    assert row['paired_objective_delta'] == pytest.approx(expected)
    assert not result['S']['supported']
