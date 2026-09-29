"""D04 county-event net-increment peaks from verified frozen OOF exports.

This is a reporting program, not a training dependency. The default command is
for use only after all five folds finish. --self-test uses synthetic arrays only.
No cumulative prediction is interpreted as an autonomous stock trajectory.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '2'

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from d04_data import HERE, load_panel, make_rows
from d04_features import REGIMES
from report_d04_response import clean, load_oof
from run_d04_response import ARMS, DEFAULT_RUN, source_hashes

LARGE_PEAK = .01
FALSE_PEAK = .001
PROBABILITIES = np.array([0., .1, .25, .5, .75, .9, 1.])


def distribution(values, weights, quantiles=False):
    """Unweighted linear quantiles and inverse-weighted-CDF quantiles."""
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    x, w = values[valid], weights[valid]
    if not len(x):
        result = {'units': 0, 'weight_sum': 0., 'unweighted_mean': None,
                  'weighted_mean': None, 'unweighted_median': None, 'weighted_median': None}
        if quantiles:
            result.update(unweighted_quantiles=[None]*len(PROBABILITIES),
                          weighted_quantiles=[None]*len(PROBABILITIES))
        return result
    order = np.argsort(x, kind='stable')
    sorted_x, cum = x[order], np.cumsum(w[order])
    pos = np.minimum(np.searchsorted(cum, PROBABILITIES*cum[-1], side='left'), len(x)-1)
    unweighted, weighted = np.quantile(x, PROBABILITIES), sorted_x[pos]
    result = {'units': len(x), 'weight_sum': float(w.sum()),
              'unweighted_mean': float(x.mean()), 'weighted_mean': float(np.average(x, weights=w)),
              'unweighted_median': float(unweighted[3]), 'weighted_median': float(weighted[3])}
    if quantiles:
        result.update(unweighted_quantiles=unweighted, weighted_quantiles=weighted)
    return result


def share(flag, selected, weights):
    selected, flag = np.asarray(selected, bool), np.asarray(flag, bool)
    count = int(selected.sum())
    return {'units': count, 'numerator_units': int((selected & flag).sum()),
            'unweighted_fraction': float(flag[selected].mean()) if count else None,
            'weighted_fraction': float(np.average(flag[selected], weights=weights[selected])) if count else None}


def _peaks(values, unit, time, valid_stock, n_units):
    """Maximum positive increment; earliest exact tied maximum is the time convention."""
    count = np.bincount(unit, minlength=n_units)
    peak = np.zeros(n_units, dtype=float)
    np.maximum.at(peak, unit, values)
    positive = peak > 0
    tied = (values > 0) & (values == peak[unit])
    peak_time = np.full(n_units, -1, dtype=np.int16)
    first = np.full(n_units, 216, dtype=np.int16)
    np.minimum.at(first, unit[tied], time[tied])
    peak_time[positive] = first[positive]
    tie_count = np.bincount(unit[tied], minlength=n_units)
    boundary = np.zeros(n_units, bool)
    boundary[unit[tied & ((time == 72) | (time == 215))]] = True
    first_valid = np.full(n_units, 216, dtype=np.int16)
    last_valid = np.full(n_units, -1, dtype=np.int16)
    np.minimum.at(first_valid, unit, time)
    np.maximum.at(last_valid, unit, time)
    support_boundary = np.zeros(n_units, bool)
    support_boundary[unit[tied & ((time == first_valid[unit]) | (time == last_valid[unit]))]] = True
    adjacent_missing = np.zeros(n_units, bool)
    # Candidate neighboring increments must use the same adjacent-hour mask.
    # An out-of-window neighbor is a boundary, not a fabricated missing hour.
    previous_missing = (time > 72) & ~valid_stock[unit, time-2]
    next_missing = (time < 215) & ~valid_stock[unit, np.minimum(time+1, 215)]
    adjacent_missing[unit[tied & (previous_missing | next_missing)]] = True
    order = np.lexsort((time, unit))
    u, t, maxima = unit[order], time[order], tied[order]
    plateau = np.zeros(n_units, bool)
    consecutive = maxima[1:] & maxima[:-1] & (u[1:] == u[:-1]) & (t[1:] == t[:-1]+1)
    plateau[u[1:][consecutive]] = True
    pos_sum = np.bincount(unit, weights=np.maximum(values, 0), minlength=n_units)
    neg_sum = np.bincount(unit, weights=np.minimum(values, 0), minlength=n_units)
    for array in (peak, pos_sum, neg_sum):
        array[count == 0] = np.nan
    return {'peak': peak, 'time': peak_time, 'positive_sum': pos_sum, 'negative_sum': neg_sum,
            'tied_peak': tie_count > 1, 'consecutive_peak_plateau': plateau,
            'window_boundary_peak': boundary, 'valid_support_boundary_peak': support_boundary,
            'adjacent_missing_peak': adjacent_missing}


def unit_summaries(panel, rows, pred):
    """Keep all input units, including ones with no valid adjacent-hour pair."""
    y, obs = np.asarray(panel['y_full']), np.asarray(panel['obs_full'], bool)
    n = len(y)
    if y.shape != (n, 216) or obs.shape != y.shape:
        raise ValueError('Expected 216-hour stock and mask')
    unit, time = np.asarray(rows['unit'], int), np.asarray(rows['time'], int)
    values, pred = np.asarray(rows['y'], float), np.asarray(pred, float)
    if (pred.shape != (len(unit), len(ARMS)) or not np.isfinite(pred).all()
            or not np.isfinite(values).all() or np.any(unit < 0) or np.any(unit >= n)
            or np.any(time < 72) or np.any(time > 215)):
        raise ValueError('Invalid OOF row layout')
    if len(np.unique(unit*216+time)) != len(unit):
        raise ValueError('Duplicate county-event/hour')
    valid_stock = obs & np.isfinite(y)
    if not np.all(valid_stock[unit, time] & valid_stock[unit, time-1]):
        raise ValueError('Increment crosses a missing hour')
    expected = y[unit, time].astype(float)-y[unit, time-1].astype(float)
    if not np.array_equal(expected, values):
        raise ValueError('Reported increment disagrees with observed adjacent stocks')
    valid_rows = np.bincount(unit, minlength=n)
    observed = _peaks(values, unit, time, valid_stock, n)
    modeled = []
    for j in range(len(ARMS)):
        item = _peaks(pred[:, j], unit, time, valid_stock, n)
        at_observed = np.full(n, np.nan)
        chosen = (observed['time'][unit] == time) & (observed['peak'][unit] > 0)
        at_observed[unit[chosen]] = pred[chosen, j]
        item['at_observed_peak'] = at_observed
        modeled.append(item)
    observed_count = valid_stock.sum(1)
    all_zero = (observed_count > 0) & ((~valid_stock) | (y == 0)).all(1)
    return {'n_units': n, 'valid_rows': valid_rows, 'observed': observed, 'models': modeled,
            'all_observed_stock_zero': all_zero,
            'observed_stock_hours': observed_count,
            'missing_forecast_hours': (~valid_stock[:, 72:]).sum(1),
            'weights': np.asarray(panel['meta']['w'], float),
            'regime': np.asarray(panel['meta']['regime'])}


def summarize_stratum(units, selected):
    selected = np.asarray(selected, bool)
    weights, observed = units['weights'], units['observed']
    supported = selected & (units['valid_rows'] > 0)
    positive = supported & (observed['peak'] > 0)
    no_positive = supported & (observed['peak'] == 0)
    flags = ('window_boundary_peak', 'valid_support_boundary_peak', 'tied_peak',
             'consecutive_peak_plateau', 'adjacent_missing_peak')
    out = {'units': int(selected.sum()), 'unit_design_mass': float(weights[selected].sum()),
           'supported_units': int(supported.sum()), 'units_without_valid_increments': int((selected & ~supported).sum()),
           'observed_positive_peak_units': int(positive.sum()),
           'no_observed_positive_increment_units': int(no_positive.sum()),
           'units_with_missing_forecast_hours': int((selected & (units['missing_forecast_hours'] > 0)).sum()),
           'missing_forecast_hours': int(units['missing_forecast_hours'][selected].sum()),
           'missing_adjacent_increment_candidates': int((144-units['valid_rows'][selected]).sum()),
           'all_observed_stock_zero_units': int((selected & units['all_observed_stock_zero']).sum()),
           'observed_positive_peak_flag_counts': {name: int((observed[name] & positive).sum()) for name in flags},
           'observed': {name: distribution(observed[name][supported], weights[supported])
                        for name in ('peak', 'positive_sum', 'negative_sum')}, 'arms': {}}
    for name, item in zip(ARMS, units['models']):
        both_positive = positive & (item['peak'] > 0)
        ratio = np.full(units['n_units'], np.nan)
        np.divide(item['peak'], observed['peak'], out=ratio, where=positive)
        at_ratio = np.full_like(ratio, np.nan)
        np.divide(item['at_observed_peak'], observed['peak'], out=at_ratio, where=positive)
        arm = {key: distribution(item[key][supported], weights[supported])
               for key in ('peak', 'positive_sum', 'negative_sum')}
        arm.update({'prediction_at_observed_peak': distribution(item['at_observed_peak'][positive], weights[positive]),
                    'predicted_to_observed_peak_ratio': distribution(ratio[positive], weights[positive]),
                    'at_observed_peak_ratio': distribution(at_ratio[positive], weights[positive]),
                    'reaches_half_observed_peak': share(item['peak'] >= .5*observed['peak'], positive, weights),
                    'at_observed_time_reaches_half_peak': share(item['at_observed_peak'] >= .5*observed['peak'], positive, weights),
                    'peak_lag_hours_predicted_minus_observed': distribution(
                        (item['time']-observed['time'])[both_positive], weights[both_positive], quantiles=True),
                    'false_peak_gt_0p1pp_among_no_positive': share(item['peak'] > FALSE_PEAK, no_positive, weights),
                    'predicted_positive_peak_units': int((supported & (item['peak'] > 0)).sum()),
                    'predicted_positive_peak_flag_counts': {
                        key: int((item[key] & supported & (item['peak'] > 0)).sum()) for key in flags},
                    'both_positive_peak_units': int(both_positive.sum()),
                    'both_positive_observed_flag_counts': {key: int((observed[key] & both_positive).sum()) for key in flags}})
        out['arms'][name] = arm
    return out


def summarize(panel, rows, pred):
    units = unit_summaries(panel, rows, pred)
    supported = units['valid_rows'] > 0
    masks = {'all': np.ones(units['n_units'], bool),
             'observed_positive_peak_ge_1pp': supported & (units['observed']['peak'] >= LARGE_PEAK),
             'no_observed_positive_increment': supported & (units['observed']['peak'] == 0),
             'all_observed_stock_zero': units['all_observed_stock_zero']}
    return {'all_regimes': {label: summarize_stratum(units, mask) for label, mask in masks.items()},
            'by_regime': {regime: {label: summarize_stratum(units, mask & (units['regime'] == regime))
                                   for label, mask in masks.items()} for regime in REGIMES}}


def self_test():
    """Boundary, plateau, missing, false peaks, signed sums and weighted medians."""
    n = 6
    y = np.zeros((n, 216), float)
    obs = np.ones_like(y, bool)
    y[0, 72], y[0, 73:] = .02, .01
    y[2, 90:] = .03
    obs[2, 88] = False
    y[2, 88] = np.nan
    y[3, 80], y[3, 81:] = .01, .02
    obs[4] = False
    y[4] = np.nan
    y[5, :72], y[5, 72:] = .03, .01
    panel = {'y_full': y, 'obs_full': obs, 'fold': np.arange(n) % 5+1,
             'merged_group': np.asarray([f'g{i}' for i in range(n)]),
             'meta': {'w': np.array([1., 2., 3., 5., 7., 6.]), 'w_raw': np.ones(n),
                      'county': np.asarray([f'{i:05d}' for i in range(n)]),
                      'regime': np.asarray(REGIMES)[np.arange(n) % 5]}}
    rows = make_rows(panel)
    pred = np.zeros((len(rows['y']), len(ARMS)))
    for unit, time, value in ((0, 72, .005), (0, 74, .03), (1, 100, .002),
                              (2, 90, .015), (3, 81, .02), (3, 82, .02), (5, 215, .005)):
        pred[(rows['unit'] == unit) & (rows['time'] == time), 0] = value
    units = unit_summaries(panel, rows, pred)
    np.testing.assert_allclose(units['observed']['peak'], [.02, 0, .03, .01, np.nan, 0], equal_nan=True)
    assert units['observed']['window_boundary_peak'][0]
    assert units['observed']['adjacent_missing_peak'][2]
    assert units['observed']['tied_peak'][3] and units['observed']['consecutive_peak_plateau'][3]
    np.testing.assert_allclose(units['observed']['positive_sum'][3], .02)
    np.testing.assert_allclose(units['observed']['negative_sum'][[0, 5]], [-.01, -.02])
    assert units['all_observed_stock_zero'].tolist() == [False, True, False, False, False, False]
    assert units['models'][0]['time'][3] == 81
    result = summarize(panel, rows, pred)
    total = result['all_regimes']['all']
    assert total['units'] == n and total['units_without_valid_increments'] == 1
    large = result['all_regimes']['observed_positive_peak_ge_1pp']
    assert large['units'] == 3
    for flag in ('window_boundary_peak', 'consecutive_peak_plateau', 'adjacent_missing_peak'):
        assert large['observed_positive_peak_flag_counts'][flag] == 1
    arm = large['arms'][next(iter(ARMS))]
    assert arm['predicted_to_observed_peak_ratio']['unweighted_median'] == 1.5
    assert arm['predicted_to_observed_peak_ratio']['weighted_median'] == 2
    assert arm['reaches_half_observed_peak']['unweighted_fraction'] == 1
    assert arm['peak_lag_hours_predicted_minus_observed']['unweighted_median'] == 1
    assert arm['peak_lag_hours_predicted_minus_observed']['units'] == 3
    np.testing.assert_allclose(arm['peak_lag_hours_predicted_minus_observed']['unweighted_quantiles'],
                               np.quantile([2, 0, 1], PROBABILITIES))
    assert total['arms'][next(iter(ARMS))]['false_peak_gt_0p1pp_among_no_positive']['numerator_units'] == 2
    assert total['arms'][list(ARMS)[1]]['peak_lag_hours_predicted_minus_observed']['units'] == 0
    assert sum(result['by_regime'][r]['all']['units'] for r in REGIMES) == n
    encoded = json.dumps(clean(result), allow_nan=False, indent=1)
    return {'synthetic_units': n, 'all_checks_passed': True,
            'synthetic_summary_bytes': len(encoded.encode('utf-8'))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--out', type=Path, default=HERE/'results/v1/d04_increment_peaks.json')
    args = parser.parse_args()
    os.nice(max(0, 15-os.getpriority(os.PRIO_PROCESS, 0)))
    with threadpool_limits(limits=2):
        if args.self_test:
            print(json.dumps(self_test()))
            return
        if args.out.exists():
            raise FileExistsError('Preserve existing D04 increment-peak report')
        panel = load_panel()
        rows = make_rows(panel, phase=None)
        pred, _bundles, manifests = load_oof(args.run_dir, panel, rows)
        result = {'meta': {'analysis': 'D04 frozen OOF county-event net-increment diagnostics',
                   'source_hashes': source_hashes(),
                   'score_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   'validated_folds': [m['fold'] for m in manifests],
                   'county_events_retained': len(panel['y_full']), 'row_support': len(rows['y']),
                   'units': 'hourly fraction differences; 0.01 is 1 percentage point',
                   'large_positive_increment_peak_threshold': LARGE_PEAK,
                   'false_predicted_peak_threshold_strictly_greater_than': FALSE_PEAK,
                   'weight': 'one existing clipped design weight meta.w per county-event',
                   'quantile_probabilities': PROBABILITIES,
                   'weighted_quantile_method': 'inverse empirical weighted CDF; first cumulative mass >= probability',
                   'unweighted_quantile_method': 'numpy linear interpolation',
                   'peak_definition': 'max(max(net increments on valid adjacent observed hours), 0)',
                   'peak_time': 'earliest exact positive maximum; no-positive or unsupported has no peak time',
                   'positive_sum': 'sum of positive one-hour net increments on retained support',
                   'negative_sum': 'signed sum of negative one-hour net increments on retained support',
                   'all_zero_group': 'at least one observed stock hour, and every observed stock in all 216 hours equals zero',
                   'boundary_flag': 'any exact positive peak at forecast target time 72 or 215',
                   'support_boundary_flag': 'any exact positive peak at first or last valid target hour',
                   'adjacent_missing_flag': 'a neighboring increment within 72..215 unavailable because its stock endpoint is missing',
                   'plateau_flag': 'at least two consecutive target hours tied at the exact positive maximum',
                   'time_lag_support': 'both observed and predicted peaks positive; boundary/tie/missing flags never excluded',
                   'interpretation': 'net-change diagnostics, not physical impact rates, stock burden, or autonomous trajectory forecasts',
                   'no_bootstrap_or_physical_impact_ci': True,
                   'threshold_comparison_limit': 'the 1pp increment threshold differs from any previous stock-peak case threshold; counts are not comparable'},
                  'summaries': summarize(panel, rows, pred)}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open('x') as handle:
            json.dump(clean(result), handle, ensure_ascii=False, indent=1, allow_nan=False)
            handle.write('\n')
        print(json.dumps({'out': args.out.name, 'units': len(panel['y_full'])}))


if __name__ == '__main__':
    main()
