"""Independent D05 arithmetic/coverage tests using only small synthetic data."""
from __future__ import annotations

from pathlib import Path
import sys
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments/geo_weather_20260924'))
import d05_frozen_inputs as frozen  # noqa: E402
import d05_weather_features as weather  # noqa: E402
import run_d05_forensics as report  # noqa: E402


def _panel(y, weights=None):
    y = np.asarray(y)
    n = len(y)
    w = np.ones(n) if weights is None else np.asarray(weights, float)
    return {'y_full': y, 'obs_full': np.ones_like(y, bool),
            'merged_group': np.array([f'G{i}' for i in range(n)]),
            'meta': {'w': w, 'w_raw': w*1.3,
                     'regime': np.resize(np.array(['tropical', 'winter']), n)}}


def test_frozen_helpers_33_synthetic_contract_checks():
    # Covers shuffled indices, duplicate/missing/fold mismatch, optional fields,
    # 42-column streaming, cache county order/times, both candidate Hg mappings.
    frozen.self_test()


def test_weather_helpers_synthetic_schemas_and_order_checks():
    weather._self_test()


def test_masks_preserve_double_precision_threshold_and_do_not_bridge_gaps():
    y = np.zeros((2, 216), np.float32)
    y[0, 71] = .01
    y[0, 72:] = .02
    y[1, 72:76] = [.01, .05, .07, .11]
    panel = _panel(y)
    panel['obs_full'][1, 73] = False
    panel['y_full'][1, 77] = np.inf
    p, delta = report.masks(panel)
    assert p.dtype == delta.dtype == np.float64
    assert delta[0, 72] == float(y[0, 72])-float(y[0, 71])
    assert not (delta[0, 72] >= .01)
    assert np.isnan(p[1, 73]) and np.isnan(p[1, 77])
    assert np.isnan(delta[1, [73, 74, 77, 78]]).all()
    assert delta[1, 75] == float(y[1, 75])-float(y[1, 74])
    assert np.isnan(delta[:, 0]).all()


def test_aggregate_matches_direct_average_and_feature_specific_support():
    panel = _panel(np.zeros((5, 216)), [1, 2, 5, 3, 9])
    panel['merged_group'] = np.array(['A', 'A', 'B', 'C', 'C'])
    unit = np.array([4, 0, 3, 1, 2])
    x = np.array([[4, 2, np.nan, np.nan], [1, np.nan, 7, np.nan],
                  [8, np.nan, np.nan, np.nan], [3, np.nan, 9, np.nan],
                  [9, 6, np.nan, np.nan]])
    result = report.aggregate(x, unit, panel, boot=False)
    for j in range(x.shape[1]):
        ok = np.isfinite(x[:, j])
        w = panel['meta']['w'][unit][ok]
        groups = panel['merged_group'][unit][ok]
        assert result['n'][j] == ok.sum()
        assert result['events'][j] == len(np.unique(groups))
        assert result['weight'][j] == w.sum()
        if not ok.any():
            assert np.isnan(result['mean'][j])
            continue
        np.testing.assert_allclose(result['mean'][j], np.average(x[ok, j], weights=w))
        mass = np.array([w[groups == g].sum() for g in np.unique(groups)])
        np.testing.assert_allclose(result['event_kish'][j], mass.sum()**2/(mass@mass))
        np.testing.assert_allclose(result['max_event_share'][j], mass.max()/mass.sum())


def test_bootstrap_equals_direct_row_reweighting_and_suppresses_single_event_ci():
    panel = _panel(np.zeros((5, 216)), [1, 2, 5, 3, 9])
    panel['merged_group'] = np.array(['A', 'A', 'B', 'C', 'C'])
    unit = np.arange(5)
    x = np.array([[1, 2, 7, np.nan], [3, np.nan, 9, np.nan],
                  [9, 6, np.nan, np.nan], [8, np.nan, np.nan, np.nan],
                  [4, np.nan, np.nan, np.nan]])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        result = report.aggregate(x, unit, panel)
    levels, codes = np.unique(panel['merged_group'], return_inverse=True)
    counts = np.random.default_rng(20260928).multinomial(
        len(levels), np.full(len(levels), 1/len(levels)), size=999)
    for j in range(x.shape[1]):
        finite = np.isfinite(x[:, j])
        samples = []
        for row in counts:
            w = panel['meta']['w']*row[codes]
            use = finite & (w > 0)
            if use.any():
                samples.append(np.average(x[use, j], weights=w[use]))
        assert result['valid_bootstrap_draws'][j] == len(samples)
        assert result['zero_support_draws'][j] == 999-len(samples)
        if result['events'][j] >= 2:
            np.testing.assert_allclose(result['ci95'][:, j], np.quantile(samples, [.025, .975]))
        else:
            assert np.isnan(result['ci95'][:, j]).all()


def test_single_total_event_has_no_bootstrap_precision_claim():
    panel = _panel(np.zeros((2, 216)), [1, 3])
    panel['merged_group'][:] = 'G0'
    result = report.aggregate(np.array([[2., np.nan], [4., np.nan]]), np.arange(2), panel)
    assert np.all(result['valid_bootstrap_draws'] == 0)
    if 'ci95' in result:
        assert np.isnan(result['ci95']).all()
    np.testing.assert_allclose(result['mean'][0], 3.5)


def test_jump_stock_cross_table_and_reversal_followup_support():
    y = np.zeros((7, 216), dtype=np.float32)
    y[0, 100:] = .03125                       # jump only, sustained
    y[1, :] = .25                            # high stock with no new jump
    y[2, 90:] = .25                          # both
    y[4, 100] = .0625                        # immediate reversal
    y[5, 215] = .03125                       # endpoint: no future support
    y[6, 100:] = .03125; y[6, 110:] = .0625  # tied positive jumps
    panel = _panel(y)
    p, d = report.masks(panel)
    result, anchors = report.phenotypes(panel, p, d)
    counts = result['counts']
    assert [counts[k] for k in ('all', 'J', 'S', 'J_and_S', 'J_only', 'S_only', 'neither')] == [7, 5, 2, 1, 4, 1, 1]
    assert counts['origin_already_10pct'] == 1 and counts['S_no_upcross'] == 1
    np.testing.assert_array_equal(anchors['jump_max'][0], [0, 2, 4, 5, 6])
    np.testing.assert_array_equal(anchors['jump_max'][1], [100, 90, 100, 215, 100])
    np.testing.assert_array_equal(anchors['stock_cross'][0], [2])
    np.testing.assert_array_equal(anchors['fixed_clock'][1], np.full(7, 120))
    persistence = result['persistence']
    assert persistence['raw_counts'] == {'immediate_reversal': 1, 'immediate_supported': 4,
                                         'six_hour_sustained': 3, 'six_hour_supported': 4}
    assert persistence['max_jump_boundary'] == 1 and persistence['max_jump_ties'] == 1


def test_model_peak_amplitude_timing_and_common_support_are_distinct():
    y = np.zeros((3, 216), np.float32); y[:, 100:] = .25
    panel = _panel(y, [1, 3, 5]); panel['obs_full'][2, 101] = False
    p, d = report.masks(panel)
    prediction = np.zeros((3, 144), np.float32)
    prediction[1, 104-72:] = .125              # correct amplitude, four hours late
    prediction[2, 101-72] = .875               # peak at missing observed hour
    prediction[2, 110-72:] = .0625             # lower valid-support peak
    bundle = {'meta': {}, 'models': {'synthetic': {'P': prediction,
               'available_fields': ['P'], 'missing_folds': {}}}}
    anchors = {key: (np.arange(3), np.full(3, 100)) for key in ('jump_max', 'stock_max')}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        result = report.model_audit(panel, p, d, anchors, bundle)['models']['synthetic']
    for kind in ('jump_max', 'stock_max'):
        summary = result[kind]
        fields = summary['fields']; means = summary['mean_ci']['mean']
        j = fields.index('peak_time_error_hours')
        assert summary['mean_ci']['n'][j] == 2  # all-zero prediction has no peak time
        np.testing.assert_allclose(means[j], np.average([4., 10.], weights=[3, 5]))
        np.testing.assert_allclose(means[fields.index('at_true_anchor_ratio')], 0.)
        np.testing.assert_allclose(means[fields.index('max_on_observed_support_ratio')],
                                   np.average([0., .5, .25], weights=[1, 3, 5]))
        np.testing.assert_allclose(means[fields.index('max_within_3h_ratio')], 0.)
        np.testing.assert_allclose(means[fields.index('max_within_6h_ratio')],
                                   np.average([0., .5, 0.], weights=[1, 3, 5]))
        np.testing.assert_allclose(means[fields.index('max_within_12h_ratio')],
                                   means[fields.index('max_on_observed_support_ratio')])
        assert summary['all144_max_ratio_unweighted_q10_50_90'][2] > .5
    assert result['jump_max']['no_positive_predicted_increment'] == 1
    assert result['stock_max']['no_positive_predicted_stock'] == 1


def test_negative_predicted_changes_keep_signed_anchor_but_zero_positive_peaks():
    y = np.full((1, 216), .25); y[:, 72:] = .5
    panel = _panel(y)
    p, d = report.masks(panel)
    prediction = np.linspace(.24, .1, 144)[None]
    frozen = {'meta': {}, 'models': {'decline': {'P': prediction,
                'available_fields': ['P'], 'missing_folds': {}}}}
    anchors = {key: (np.array([0]), np.array([72])) for key in ('jump_max', 'stock_max')}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        s = report.model_audit(panel, p, d, anchors, frozen)['models']['decline']['jump_max']
    means = dict(zip(s['fields'], s['mean_ci']['mean']))
    assert means['at_true_anchor_ratio'] < 0
    assert means['max_on_observed_support_ratio'] == 0
    assert np.isnan(means['peak_time_error_hours'])
    assert all(means[f'max_within_{h}h_ratio'] == 0 for h in (3, 6, 12))


def test_within_county_averages_controls_before_weighting_without_quiet_filter(monkeypatch):
    y = np.zeros((2, 216)); y[0, 72:] = .03125; y[0, 84:] = .0625; y[1, 204:] = .03125
    panel = _panel(y, [1, 3])
    panel['feature_names'] = {'weather': list(weather.WEATHER_NAMES)}
    unit, time = np.arange(2), np.array([100, 180])
    anchors = {key: (unit, time) for key in ('jump_max', 'stock_max')}

    def summarize(_panel, u, t, _reference):
        return {'unit': u, 'time': t, 'near_hours': {'standardized': np.zeros((len(u), 3, 12))}}

    def flatten(s, **_kwargs):
        return {'feature_matrix': (1000*s['unit']+s['time'])[:, None].astype(float),
                'feature_names': ['synthetic_time'], 'feature_strict_past': np.array([True])}

    monkeypatch.setattr(report, 'summarize_anchors', summarize)
    monkeypatch.setattr(report, 'flatten_summaries', flatten)
    monkeypatch.setattr(report, 'progress', lambda _stage: None)
    extra = {'values': np.broadcast_to(np.arange(216)[None, :, None], (2, 216, 28)).copy(),
             'feature_names': [f'e{k}' for k in range(28)]}
    result = report.within_county(panel, None, anchors, extra)['cohorts']['jump_max']
    clocks = [report.CLOCK[abs(report.CLOCK-t) > 12] for t in time]
    expected = np.average([t-c.mean() for t, c in zip(time, clocks)], weights=[1, 3])
    np.testing.assert_allclose(result['groups']['all']['mean'][0], expected)
    # The two positive control hours in unit 0 and one in unit 1 remain included.
    jump_rate = np.average([2/len(clocks[0]), 1/len(clocks[1])], weights=[1, 3])
    np.testing.assert_allclose(result['control_jump_ge1pp_fraction']['mean'][0], jump_rate)
    assert result['eligible_controls_min_max'] == [9, 10]
