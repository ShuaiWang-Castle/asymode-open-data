"""Fixed D09 endpoint scoring; no model import, forward, gradient or fitting.

Read self-contained terminal exports only after all six jobs have DONE receipts.
The complete original event-fold 2/3 county populations are the primary surface.
The input-selected D08 panel is secondary: its HT totals use fixed full-cohort
denominators, while ratios, quantiles and RMSEs remain descriptive estimators.
"""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '2'

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SPLITS = HERE / 'splits_v1D.json'
FEATURES = ROOT / 'data/interim/panel_v1/features_v1D.npz'
D08 = ROOT / 'runs/geo_weather_20260924/d08_panel_20260929_iofix1'
DATA_SHA = 'f043bb39e8abd48183670e2acc3c0cecebd7107ea9be0e28771912cfee358c48'
ARMS = ('host', 'crk', 'new')
FOLDS = (2, 3)
REGIMES = ('tropical', 'winter', 'synoptic_wind', 'convective', 'heavy_rain')
GUARD_REGIMES = REGIMES[2:]
BOOTSTRAP = 1999
SEED = 20260929
META_KEYS = ('idx', 'y', 'm', 'y0', 'w', 'regime', 'fips', 'system', 'family',
             'origin', 'merged_group', 'original_fold', 'panel', 'pi', 'origin_observed')
COMPARISONS = (('new_vs_host', 'host', 'new'), ('crk_vs_host', 'host', 'crk'),
               ('new_vs_crk', 'crk', 'new'),
               ('crk_vs_closed', 'crk_closed', 'crk'),
               ('new_vs_closed', 'new_closed', 'new'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def under_root(value):
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    path.relative_to(ROOT)
    return path


def strict_json(value):
    if isinstance(value, dict):
        return {str(k): strict_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return strict_json(value.tolist())
    if isinstance(value, np.generic):
        return strict_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Nonfinite public result')
    return value


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(strict_json(value), stream, ensure_ascii=False, indent=1,
                  allow_nan=False)
        stream.write('\n')


def git_bytes(commit, path):
    return subprocess.check_output(['git', 'show', f'{commit}:{relative(path)}'],
                                   cwd=ROOT, stderr=subprocess.PIPE)


def verify_registration(commit, scope):
    if len(commit) < 7 or any(c not in '0123456789abcdef' for c in commit):
        raise ValueError('An explicit hexadecimal registration commit is required')
    full = subprocess.check_output(['git', 'rev-parse', commit], cwd=ROOT,
                                   text=True).strip()
    for path in (Path(__file__), Path(scope)):
        if hashlib.sha256(git_bytes(full, path)).hexdigest() != sha(path):
            raise ValueError(f'Registered file changed: {relative(path)}')
    return full


def distribution(values, weights):
    """D06/D07A cumulative-mass knots, with explicitly missing peak ratios."""
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return dict(n=0, missing=int(len(values)), mean=None,
                    q10_median_q90=None, minimum=None, maximum=None)
    x, w = values[valid], weights[valid]
    order = np.argsort(x, kind='stable')
    cumulative = np.cumsum(w[order])
    return dict(n=int(valid.sum()), missing=int((~valid).sum()),
                mean=float(w @ x / w.sum()),
                q10_median_q90=np.interp(np.array([.1, .5, .9]) * cumulative[-1],
                                        cumulative, x[order]).tolist(),
                minimum=float(x.min()), maximum=float(x.max()))


def phenotype(y, m, y0):
    """Observed count differs from contiguous duration; hour71 is the origin."""
    y, m, y0 = np.asarray(y, float), np.asarray(m, bool), np.asarray(y0, float)
    if y.ndim != 2 or y.shape[1] != 144 or m.shape != y.shape or y0.shape != (len(y),):
        raise ValueError('Expected aligned 144-hour outcomes and observed origin')
    if not np.isfinite(y[m]).all() or not np.isfinite(y0).all():
        raise ValueError('Nonfinite observed outcomes/origin')
    valid = m.any(1)
    peak = np.max(np.where(m, y, -np.inf), axis=1)
    hour = np.argmax(np.where(m, y, -np.inf), axis=1) + 72
    peak[~valid], hour[~valid] = np.nan, -1
    severe = m & (y >= .1)
    running, longest = np.zeros(len(y), int), np.zeros(len(y), int)
    for t in range(144):
        running = np.where(severe[:, t], running + 1, 0)
        longest = np.maximum(longest, running)
    adjacent = m & np.column_stack((np.ones(len(y), bool), m[:, :-1]))
    difference = y - np.column_stack((y0, y[:, :-1]))
    rise = np.max(np.where(adjacent, difference, -np.inf), axis=1)
    rise[~adjacent.any(1)] = np.nan
    return dict(observed_hours=m.sum(1), true_peak=peak, true_peak_hour=hour,
                severe_hours=severe.sum(1), max_contiguous_run=longest,
                maximum_adjacent_rise=rise, adjacent_observed_pairs=adjacent.sum(1),
                complete144=m.all(1))


def cohort_masks(ph):
    valid = ph['observed_hours'] > 0
    S = valid & (ph['true_peak'] >= .1)
    J = valid & (ph['maximum_adjacent_rise'] >= .01)
    result = dict(all=valid, S=S, nonS=valid & ~S, J=J, S_and_J=S & J,
                  any_positive=valid & (ph['true_peak'] > 0),
                  all_observed_zero=valid & (ph['true_peak'] == 0),
                  complete144=valid & ph['complete144'],
                  incomplete=valid & ~ph['complete144'],
                  S_complete144=S & ph['complete144'],
                  S_incomplete=S & ~ph['complete144'])
    for name, x in (('severe_hour_count', ph['severe_hours']),
                    ('max_contiguous_run', ph['max_contiguous_run'])):
        result[name + '/1'] = S & (x == 1)
        result[name + '/2..6'] = S & (x >= 2) & (x <= 6)
        result[name + '/>=7'] = S & (x >= 7)
    return result


def model_peak(P, m, ph):
    valid = m.any(1)
    peak = np.max(np.where(m, P, -np.inf), axis=1)
    hour = np.argmax(np.where(m, P, -np.inf), axis=1) + 72
    peak[~valid], hour[~valid] = np.nan, -1
    ratio = np.full(len(P), np.nan)
    np.divide(peak, ph['true_peak'], out=ratio,
              where=valid & (ph['true_peak'] > 0))
    lag = (hour - ph['true_peak_hour']).astype(float)
    lag[~valid] = np.nan
    return dict(predicted_peak=peak, predicted_peak_hour=hour,
                peak_ratio=ratio, signed_peak_lag_hours=lag)


def bootstrap_plan(regime, weight, labels, draws=BOOTSTRAP, seed=SEED):
    """Whole-cluster paired draws, stratified by dominant held-pool regime mass.

    Levels/strata use this fixed complete two-fold heldout population, before
    selecting S or any subgroup. All model/cohort rows reuse the same counts.
    """
    if not isinstance(draws, int) or draws < 1:
        raise ValueError('Positive bootstrap draw count required')
    labels, regime, weight = np.asarray(labels).astype(str), np.asarray(regime).astype(str), np.asarray(weight, float)
    if not len(labels) or len(labels) != len(regime) or len(weight) != len(labels):
        raise ValueError('Invalid cluster population')
    levels, code = np.unique(labels, return_inverse=True)
    rid = np.array([REGIMES.index(r) for r in regime])
    mass = np.zeros((len(levels), len(REGIMES)))
    np.add.at(mass, (code, rid), weight)
    strata = mass.argmax(1)
    counts = np.zeros((draws, len(levels)), np.int32)
    rng = np.random.default_rng(seed)
    for r in range(len(REGIMES)):
        js = np.flatnonzero(strata == r)
        if len(js):
            counts[:, js] = rng.multinomial(len(js), np.full(len(js), 1 / len(js)), size=draws)
    return dict(levels=levels, code=code, counts=counts, strata=strata)


def improvement(base_sse, candidate_sse):
    return 1. - math.sqrt(candidate_sse / base_sse) if base_sse > 0 else None


def bootstrap_comparison(base_rows, candidate_rows, selected, plan):
    base_rows, candidate_rows = np.asarray(base_rows, float), np.asarray(candidate_rows, float)
    selected = np.asarray(selected, bool)
    code, nlevels = plan['code'], len(plan['levels'])
    b = np.bincount(code, weights=np.where(selected, base_rows, 0.), minlength=nlevels)
    c = np.bincount(code, weights=np.where(selected, candidate_rows, 0.), minlength=nlevels)
    support = int(len(np.unique(code[selected])))
    bs, cs = plan['counts'] @ b, plan['counts'] @ c
    valid = (bs > 0) & np.isfinite(bs) & np.isfinite(cs) & (cs >= 0)
    reductions = 1. - np.sqrt(cs[valid] / bs[valid])
    # A singleton-stratum plan can repeat identical cluster counts in every
    # draw. Such repetitions are not confidence support, even with two labels.
    # Check count variation in risk-contributing clusters, not reduction
    # variance: proportional model risks can legitimately yield a zero-width
    # interval despite genuine whole-cluster resampling.
    effective = (b > 0) | (c > 0)
    counts = plan['counts'][:, effective]
    varies = bool(counts.shape[1] and np.any(np.max(counts, axis=0) != np.min(counts, axis=0)))
    empirical = np.quantile(reductions, [.025, .975]).tolist() if valid.any() else None
    supported = support >= 2 and valid.any() and varies
    ci = empirical if supported else None
    status = ('insufficient_support' if support < 2 else 'no_valid_draws' if not valid.any()
              else 'no_effective_cluster_count_variation' if not varies else 'supported')
    return dict(supporting_clusters=support, draws=len(bs), valid_draws=int(valid.sum()),
                invalid_draws=int((~valid).sum()), interval_supported=bool(supported),
                effective_risk_contribution_clusters=int(effective.sum()),
                resample_counts_vary=varies, interval_status=status,
                empirical_relative_RMSE_reduction_percentiles95=empirical,
                relative_RMSE_reduction_ci95=ci,
                ci_excludes_zero_in_beneficial_direction=bool(ci and ci[0] > 0),
                ci_lower_bound_at_least_final_10pct=bool(ci and ci[0] >= .1))


def gain_summary(y, m, base, candidate, selected, weight):
    weight = np.asarray(weight, float)
    ids = np.flatnonzero(selected)
    if not len(ids):
        return dict(units=0, alignment=0., modification_energy=0., net_gain=0.,
                    positive_gain=0., negative_loss=0., positive_units=0,
                    negative_units=0, zero_units=0, positive_design_share=None,
                    top_10pct_all_cohort_units=0, top_10pct_share_of_positive_gain=None)
    residual, delta = y[ids] - base[ids], candidate[ids] - base[ids]
    align = weight[ids] * np.sum(np.where(m[ids], 2 * residual * delta, 0.), axis=1)
    energy = weight[ids] * np.sum(np.where(m[ids], delta ** 2, 0.), axis=1)
    gain = align - energy
    direct = weight[ids] * np.sum(np.where(m[ids], (y[ids] - base[ids]) ** 2 -
                                                  (y[ids] - candidate[ids]) ** 2, 0.), axis=1)
    if not np.allclose(gain, direct, rtol=2e-10, atol=2e-10):
        raise AssertionError('Paired SSE identity failed')
    positive, negative = gain > 0, gain < 0
    positive_gain = float(gain[positive].sum())
    top = max(1, math.ceil(.1 * len(ids)))
    return dict(units=len(ids), alignment=float(align.sum()), modification_energy=float(energy.sum()),
                net_gain=float(gain.sum()), positive_gain=positive_gain,
                negative_loss=float(-gain[negative].sum()), positive_units=int(positive.sum()),
                negative_units=int(negative.sum()), zero_units=int((gain == 0).sum()),
                positive_design_share=float(weight[ids][positive].sum() / weight[ids].sum()),
                top_10pct_all_cohort_units=top,
                top_10pct_share_of_positive_gain=float(np.sort(gain[positive])[::-1][:top].sum() / positive_gain)
                if positive_gain else None)


def alarm_summary(base_peak, candidate_peak, selected, weights, threshold):
    base_peak, candidate_peak, weights = (np.asarray(x) for x in (base_peak, candidate_peak, weights))
    ids = np.flatnonzero(selected)
    if not len(ids):
        return dict(units=0, threshold=threshold, base_count=0, candidate_count=0,
                    new_count=0, removed_count=0, retained_count=0,
                    base_design_rate=None, candidate_design_rate=None,
                    new_design_rate=None, removed_design_rate=None)
    # Severe alarms use >=.1; zero-unit tiny false peaks use >.001.
    base = base_peak[ids] >= threshold if threshold == .1 else base_peak[ids] > threshold
    candidate = candidate_peak[ids] >= threshold if threshold == .1 else candidate_peak[ids] > threshold
    w, denom = weights[ids], float(weights[ids].sum())
    return dict(units=len(ids), threshold=threshold, base_count=int(base.sum()),
                candidate_count=int(candidate.sum()), new_count=int((candidate & ~base).sum()),
                removed_count=int((base & ~candidate).sum()), retained_count=int((base & candidate).sum()),
                base_design_rate=float(w[base].sum() / denom),
                candidate_design_rate=float(w[candidate].sum() / denom),
                new_design_rate=float(w[candidate & ~base].sum() / denom),
                removed_design_rate=float(w[base & ~candidate].sum() / denom))


def score_row(data, predictions, peaks, ph, selected, full_selected, plans, panel=False):
    """Original-design local scores plus panel HT additive totals, if requested."""
    ids = np.flatnonzero(selected)
    y, m, w = data['y'], data['m'], data['w']
    full_mass = float(w @ (m.sum(1) * full_selected))
    local_mass = float(w @ (m.sum(1) * selected))
    weighted_rows = {name: w * np.sum(np.where(m, (P - y) ** 2, 0.), axis=1)
                     for name, P in predictions.items()}
    row = dict(units=len(ids), observed_hours=int(m[ids].sum()),
               counties=len(np.unique(data['fips'][ids])), systems=len(np.unique(data['system'][ids])),
               families=len(np.unique(data['family'][ids])), merged_groups=len(np.unique(data['merged_group'][ids])),
               complete144_units=int(ph['complete144'][ids].sum()),
               incomplete_units=int((~ph['complete144'][ids]).sum()),
               design_unit_mass=float(w[ids].sum()), design_hour_mass=local_mass,
               original_full_held_cohort_hour_mass=full_mass,
               unit_design_kish=float(w[ids].sum() ** 2 / (w[ids] ** 2).sum()) if len(ids) else 0.,
               models={}, comparisons={}, outcome_shapes={})
    for key in ('true_peak', 'severe_hours', 'max_contiguous_run', 'maximum_adjacent_rise'):
        row['outcome_shapes'][key] = distribution(ph[key][ids], w[ids])
    ht = w / data['pi'] if panel else None
    if panel:
        row['HT_design_unit_mass'] = float(ht[ids].sum())
        row['HT_design_hour_mass'] = float(ht @ (m.sum(1) * selected))
        row['HT_unit_design_kish'] = float(ht[ids].sum() ** 2 / (ht[ids] ** 2).sum()) if len(ids) else 0.
    for name, P in predictions.items():
        sse = float(weighted_rows[name][ids].sum())
        model = dict(design_SSE=sse, design_RMSE=math.sqrt(sse / local_mass) if local_mass else None,
                     peak_ratio=distribution(peaks[name]['peak_ratio'][ids], w[ids]),
                     predicted_peak=distribution(peaks[name]['predicted_peak'][ids], w[ids]),
                     signed_peak_lag_hours=distribution(peaks[name]['signed_peak_lag_hours'][ids], w[ids]))
        if panel:
            ht_sse = float((weighted_rows[name] / data['pi'])[ids].sum())
            model.update(HT_design_SSE_numerator=ht_sse,
                         HT_MSE_fixed_full_held_cohort_denominator=ht_sse / full_mass if full_mass else None,
                         HT_RMSE_fixed_full_held_cohort_denominator=math.sqrt(ht_sse / full_mass) if full_mass else None,
                         HT_reweighted_peak_ratio=distribution(peaks[name]['peak_ratio'][ids], ht[ids]))
        row['models'][name] = model
    for label, base, candidate in COMPARISONS:
        if base not in predictions or candidate not in predictions:
            continue
        comp = dict(relative_RMSE_reduction=improvement(row['models'][base]['design_SSE'],
                                                      row['models'][candidate]['design_SSE']),
                    paired_gain=gain_summary(y, m, predictions[base], predictions[candidate], selected, w),
                    bootstrap={key: bootstrap_comparison(weighted_rows[base], weighted_rows[candidate], selected, plan)
                               for key, plan in plans.items()})
        if panel:
            comp['HT_relative_RMSE_reduction'] = improvement(row['models'][base]['HT_design_SSE_numerator'],
                                                           row['models'][candidate]['HT_design_SSE_numerator'])
            comp['HT_paired_gain'] = gain_summary(y, m, predictions[base], predictions[candidate], selected, ht)
        row['comparisons'][label] = comp
    return row


def score_surface(data, predictions, panel=False, draws=BOOTSTRAP):
    ph = phenotype(data['y'], data['m'], data['y0'])
    cohorts = cohort_masks(ph)
    peaks = {name: model_peak(P, data['m'], ph) for name, P in predictions.items()}
    plans = {key: bootstrap_plan(data['regime'], data['w'], data[label], draws)
             for key, label in (('merged_group', 'merged_group'), ('family_sensitivity', 'family'))}
    surface = dict(surface='fixed_D08_panel_secondary' if panel else 'complete_heldout_primary',
                   bootstrap=dict(draws=draws, seed=SEED, paired=True,
                                  strata='dominant original-design regime mass in complete two-fold heldout pool',
                                  confidence=.95,
                                  plans={key: dict(clusters=len(plan['levels']),
                                             stratum_cluster_counts={r: int(np.sum(plan['strata'] == j))
                                                                    for j, r in enumerate(REGIMES)},
                                             singleton_strata=[r for j, r in enumerate(REGIMES)
                                                               if np.sum(plan['strata'] == j) == 1])
                                         for key, plan in plans.items()}), rows={}, alarms={})
    sampled = np.asarray(data['panel'], bool) if panel else np.ones(len(data['idx']), bool)
    strata = {'pooled': np.ones(len(sampled), bool)}
    strata.update({f'fold{fold}': data['original_fold'] == fold for fold in FOLDS})
    strata.update({'regime/' + r: data['regime'] == r for r in REGIMES})
    strata.update({f'fold{fold}/regime/{r}': (data['original_fold'] == fold) & (data['regime'] == r)
                   for fold in FOLDS for r in REGIMES})
    for label, stratum in strata.items():
        surface['rows'][label] = {cohort: score_row(data, predictions, peaks, ph,
                                                  stratum & select & sampled, stratum & select,
                                                  plans, panel)
                                 for cohort, select in cohorts.items()}
        alarms = {}
        for comp, base, candidate in COMPARISONS:
            if base in peaks and candidate in peaks:
                alarms[comp] = dict(nonS_severe=alarm_summary(peaks[base]['predicted_peak'], peaks[candidate]['predicted_peak'],
                                                            stratum & cohorts['nonS'] & sampled, data['w'], .1),
                                    zero_tiny=alarm_summary(peaks[base]['predicted_peak'], peaks[candidate]['predicted_peak'],
                                                           stratum & cohorts['all_observed_zero'] & sampled, data['w'], .001))
        surface['alarms'][label] = alarms
    surface['observational_support'] = dict(total_exported_units=len(data['idx']),
                                            no_observed_forecast_units=int((~cohorts['all']).sum()),
                                            total_observed_hours=int(data['m'].sum()))
    if panel:
        surface['interpretation'] = ['Secondary input-selected panel, never the advancement gate.',
            'Local scores use original design weights. HT additive SSE totals use w/pi and fixed full-held cohort denominators.',
            'HT totals have a sampling-design interpretation only conditional on fixed predictors; RMSE, ratios and quantiles are nonlinear descriptive estimators.',
            'Bootstrap uses original local design scores; it does not add uncertainty from panel selection or fitting.']
    return surface


def verdict(primary):
    rows, alarms = primary['rows'], primary['alarms']
    def reduction(stratum, cohort, comparison):
        return rows[stratum][cohort]['comparisons'][comparison]['relative_RMSE_reduction']
    def finite_pass(value, predicate):
        return value is not None and math.isfinite(value) and predicate(value)
    checks = {
        'pooled_S_new_vs_host_at_least_5pct': finite_pass(reduction('pooled', 'S', 'new_vs_host'), lambda v: v >= .05),
        'pooled_S_new_vs_crk_positive': finite_pass(reduction('pooled', 'S', 'new_vs_crk'), lambda v: v > 0),
        'fold2_S_new_vs_host_positive': finite_pass(reduction('fold2', 'S', 'new_vs_host'), lambda v: v > 0),
        'fold3_S_new_vs_host_positive': finite_pass(reduction('fold3', 'S', 'new_vs_host'), lambda v: v > 0),
        'pooled_all_new_vs_host_not_worse': finite_pass(reduction('pooled', 'all', 'new_vs_host'), lambda v: v >= 0)}
    alarm = alarms['pooled']['new_vs_host']['nonS_severe']
    checks['nonS_severe_alarm_count_not_higher'] = alarm['candidate_count'] <= alarm['base_count']
    # If there are no non-S cases the count/rate guard is vacuous, not undefined benefit.
    checks['nonS_severe_alarm_design_rate_not_higher'] = (alarm['units'] == 0 or
        (alarm['candidate_design_rate'] is not None and alarm['base_design_rate'] is not None and
         alarm['candidate_design_rate'] <= alarm['base_design_rate']))
    class_guards = {}
    for r in GUARD_REGIMES:
        row = rows['regime/' + r]['all']
        value = reduction('regime/' + r, 'all', 'new_vs_host')
        applicable = row['merged_groups'] >= 2
        passed = finite_pass(value, lambda v: v >= -.02) if applicable else None
        class_guards[r] = dict(supporting_merged_groups=row['merged_groups'], applicable=applicable,
                               relative_RMSE_reduction=value, maximum_allowed_RMSE_increase=.02, passed=passed)
        if applicable:
            checks['class_' + r + '_all_RMSE_not_worse_than_2pct'] = passed
    passed = all(checks.values())
    return dict(schema='d09_exploratory_gate_v1', point_gate_pass=passed, checks=checks,
                nonheadline_class_guards=class_guards,
                S_new_vs_host_relative_RMSE_reduction=reduction('pooled', 'S', 'new_vs_host'),
                S_new_vs_crk_relative_RMSE_reduction=reduction('pooled', 'S', 'new_vs_crk'),
                S_new_vs_host_bootstrap=rows['pooled']['S']['comparisons']['new_vs_host']['bootstrap'],
                decision='may_propose_full_D_five_fold_comparison' if passed else 'do_not_advance_this_candidate_by_registered_gate',
                automatically_start_full_D=False, original_final_S_target_relative_RMSE_reduction=.1,
                interpretation='Point gate and 95% interval support are separate. One seed and repeatedly explored D do not establish independent confirmation, causal attribution, or net geographic information.')


def validate_export(data, fold):
    missing = set(META_KEYS + ('P', 'u', 'r', 'raw_logit')) - set(data)
    if missing:
        raise ValueError('Export keys missing: ' + ', '.join(sorted(missing)))
    idx = np.asarray(data['idx'])
    n = len(idx)
    if idx.ndim != 1 or not np.issubdtype(idx.dtype, np.integer) or len(np.unique(idx)) != n or (n and np.any(np.diff(idx) <= 0)):
        raise ValueError('Original unit IDs must be unique sorted integers')
    if data['y'].shape != (n, 144) or data['m'].shape != (n, 144):
        raise ValueError('144-hour forecast interface required')
    if not np.isin(data['m'], [False, True]).all():
        raise ValueError('m must be original binary observation mask')
    data['m'] = data['m'].astype(bool)
    for key in META_KEYS:
        if key not in ('y', 'm') and data[key].shape != (n,):
            raise ValueError('Metadata row alignment: ' + key)
    for key in ('P', 'u', 'r', 'raw_logit', 'P_closed', 'raw_logit_closed'):
        if key in data and (data[key].shape != (n, 144) or not np.isfinite(data[key]).all()):
            raise ValueError('Nonfinite or unaligned predictions: ' + key)
    for key in ('P', 'P_closed'):
        if key in data and ((data[key] < 0).any() or (data[key] > 1).any()):
            raise ValueError('Predicted outage proportion outside [0,1]')
    if not np.isfinite(data['y'][data['m']]).all() or not np.isfinite(data['y0']).all():
        raise ValueError('Nonfinite observed outcomes')
    if np.any(data['y'][data['m']] < 0) or np.any(data['y'][data['m']] > 1) or np.any(data['y0'] < 0) or np.any(data['y0'] > 1):
        raise ValueError('Observed outage proportion outside [0,1]')
    if not np.isfinite(data['w']).all() or np.any(data['w'] <= 0):
        raise ValueError('Invalid original design weight')
    if not np.isfinite(data['pi']).all() or np.any(data['pi'] <= 0) or np.any(data['pi'] > 1):
        raise ValueError('Invalid fixed inclusion probability')
    if not np.isin(data['panel'], [False, True]).all() or not np.all(data['original_fold'] == fold):
        raise ValueError('Wrong heldout population')
    if not np.isin(data['origin_observed'], [False, True]).all() or not data['origin_observed'].all():
        raise ValueError('Observed hour71 origin must be certified for J')
    data['panel'] = data['panel'].astype(bool)
    if not np.isin(data['regime'].astype(str), REGIMES).all():
        raise ValueError('Unknown weather class')
    phenotype(data['y'], data['m'], data['y0'])


def assert_same_metadata(left, right):
    for key in META_KEYS:
        numeric = np.issubdtype(left[key].dtype, np.number)
        same = np.array_equal(left[key], right[key], equal_nan=True) if numeric else np.array_equal(left[key], right[key])
        if not same:
            raise ValueError('Arm/export metadata differ: ' + key)


def read_verified_jobs(jobs_path, scope, commit):
    """All completion/hash/roster checks precede opening the first NPZ outcome."""
    manifest = json.loads(Path(jobs_path).read_text())
    if manifest.get('schema') != 'd09_jobs_v1':
        raise ValueError('Expected d09_jobs_v1 manifest')
    if manifest.get('registration_commit') != commit:
        raise ValueError('Job inventory registration mismatch')
    jobs = manifest.get('jobs', [])
    expected = {(arm, fold) for arm in ARMS for fold in FOLDS}
    if len(jobs) != 6 or {(j['arm'], int(j['heldout_fold'])) for j in jobs} != expected:
        raise ValueError('Exactly three arms and two heldout folds required')
    split = json.loads(SPLITS.read_text())
    roster = json.loads((D08 / 'roster.json').read_text())
    frozen = json.loads((D08 / 'FROZEN.json').read_text())
    if frozen.get('status') != 'input_roster_frozen':
        raise ValueError('D08 input roster not frozen')
    hashes = {relative(path): sha(path) for path in (Path(__file__), Path(jobs_path), Path(scope), SPLITS,
                                                        D08 / 'FROZEN.json', D08 / 'manifest.json', D08 / 'roster.json')}
    if hashes[relative(D08 / 'manifest.json')] != frozen['manifest_sha256'] or hashes[relative(D08 / 'roster.json')] != frozen['roster_sha256']:
        raise ValueError('D08 frozen input metadata changed')
    # File-level streaming hash does not deserialize any raw-D outcome/member.
    if sha(FEATURES) != DATA_SHA:
        raise ValueError('Frozen public-D data bytes changed')
    hashes[relative(FEATURES)] = DATA_SHA
    original_fit = np.sort(np.asarray(split['event']['1']['dev'], int))
    if not np.array_equal(original_fit, np.sort(roster['original_fit'])):
        raise ValueError('D08 original FIT disagrees with frozen event split')
    records = {int(r['unit']): r for r in roster['records']}
    receipts, fit_ids, initial_hashes, protocols, source_maps = {}, {}, set(), [], []
    for job in jobs:
        key = (job['arm'], int(job['heldout_fold']))
        folder = under_root(job['path'])
        done_path = folder / 'DONE.json'
        receipt = json.loads(done_path.read_text())
        if receipt.get('schema') != 'd09_job_done_v1' or receipt.get('arm') != key[0] or receipt.get('heldout_fold') != key[1] or receipt.get('steps') != 900 or receipt.get('seed') != 0:
            raise ValueError('Invalid terminal receipt: ' + relative(done_path))
        if receipt.get('registration_commit') != commit or receipt.get('scope_sha256') != hashes[relative(scope)]:
            raise ValueError('D09 registration mismatch')
        protocol = receipt['protocol']
        fixed = dict(steps=900, seed=0, private_seed=1729, microbatch=512,
                     numerical_threads=2, early_stopping=False,
                     checkpoints=[0, 100, 300, 900], heldout_full_export_step=900,
                     host_learning_rate=.003, recovery_learning_rate=.0003,
                     calibration_every=10,
                     training_weights='original panel w / inner-FIT regime zero-predictor SSE; not HT')
        if any(protocol.get(k) != v for k, v in fixed.items()):
            raise ValueError('Registered training protocol mismatch')
        if not receipt.get('optimizer_covers_all_trainable') or receipt.get('preflight') or receipt['resources']['nice'] < 15 or receipt['resources']['threads'] != 2:
            raise ValueError('Unverified optimizer/formal job/resource protocol')
        for field, path in (('data_sha256', FEATURES), ('split_sha256', SPLITS),
                            ('d08_frozen_sha256', D08 / 'FROZEN.json'),
                            ('d08_manifest_sha256', D08 / 'manifest.json'),
                            ('d08_roster_sha256', D08 / 'roster.json')):
            if receipt[field] != hashes[relative(path)]:
                raise ValueError('Provenance mismatch: ' + field)
        source_map = receipt['source_sha256']
        if not source_map:
            raise ValueError('Missing frozen training sources')
        for name, digest in source_map.items():
            path = under_root(name)
            if relative(path) != name or sha(path) != digest or hashlib.sha256(git_bytes(commit, path)).hexdigest() != digest:
                raise ValueError('Frozen source changed: ' + name)
            hashes[name] = digest
        outputs = receipt['outputs']
        for name in ('step0900_held_full.npz', 'step0900_held_panel.npz', 'step0900_fit_panel.npz', 'final.pt', 'stats.json'):
            if name not in outputs:
                raise ValueError('Incomplete endpoint export: ' + name)
        for name, digest in outputs.items():
            if Path(name).name != name or sha(folder / name) != digest:
                raise ValueError('Training output hash mismatch: ' + name)
            hashes[relative(folder / name)] = digest
        hashes[relative(done_path)] = sha(done_path)
        held = np.sort(np.intersect1d(original_fit, split['event'][str(key[1])]['outer']))
        panel_held = np.array(sorted(set(held.tolist()) & set(records)), int)
        train = np.array(sorted(i for i, r in records.items() if int(r['original_fold']) != key[1]), int)
        if not np.array_equal(held, receipt['held_full_unit_ids']) or not np.array_equal(panel_held, receipt['held_panel_unit_ids']) or not np.array_equal(train, receipt['fit_unit_ids']):
            raise ValueError('Full heldout/selected FIT identity mismatch')
        if not np.array_equal(train, receipt['model_statistics_fit_unit_ids']):
            raise ValueError('Model statistics were not fit only on inner FIT')
        fit_ids[key] = train
        initial_hashes.add(receipt['initial_host_parameter_sha256'])
        protocols.append(receipt['protocol'])
        source_maps.append(source_map)
        receipts[key] = (folder, receipt, held, panel_held)
    if len(initial_hashes) != 1 or any(p != protocols[0] for p in protocols[1:]) or any(s != source_maps[0] for s in source_maps[1:]):
        raise ValueError('Three-arm training initialization/protocol/source mismatch')
    for fold in FOLDS:
        old = receipts[('crk', fold)][1]['initial_kernel_parameter_sha256']
        new = receipts[('new', fold)][1]['initial_kernel_parameter_sha256']
        if not old or old != new:
            raise ValueError('Old/new kernel parameters were not paired at initialization')
    by_fold, predictions = {}, {arm: [] for arm in ARMS}
    predictions.update(crk_closed=[], new_closed=[])
    for fold in FOLDS:
        for arm in ARMS:
            folder, receipt, held, panel_held = receipts[(arm, fold)]
            with np.load(folder / 'step0900_held_full.npz', allow_pickle=False) as archive:
                full = {k: archive[k].copy() for k in archive.files}
            validate_export(full, fold)
            if not np.array_equal(full['idx'], held):
                raise ValueError('Actual full export unit IDs differ from receipt')
            expected_panel = np.isin(full['idx'], panel_held)
            if not np.array_equal(full['panel'], expected_panel):
                raise ValueError('Panel flag differs from fixed input-only roster')
            for j in np.flatnonzero(expected_panel):
                record = records[int(full['idx'][j])]
                if full['pi'][j] != record['pi']:
                    raise ValueError('Panel inclusion probability changed')
                for key in ('fips', 'system', 'family', 'origin', 'regime', 'merged_group', 'original_fold', 'w'):
                    if full[key][j] != record[key]:
                        raise ValueError('Fixed roster metadata changed: ' + key)
            with np.load(folder / 'step0900_held_panel.npz', allow_pickle=False) as archive:
                panel = {k: archive[k].copy() for k in archive.files}
            validate_export(panel, fold)
            subset = {k: v[expected_panel] for k, v in full.items()}
            assert_same_metadata(subset, panel)
            for key in ('P', 'u', 'r', 'raw_logit', 'P_closed', 'raw_logit_closed'):
                if key in subset and (key not in panel or not np.array_equal(subset[key], panel[key])):
                    raise ValueError('Full/panel prediction exports disagree: ' + key)
            if fold in by_fold:
                assert_same_metadata(by_fold[fold], full)
            else:
                by_fold[fold] = {k: full[k] for k in META_KEYS}
            predictions[arm].append(full['P'])
            if arm != 'host':
                if 'P_closed' not in full:
                    raise ValueError('Missing closed-exit comparison')
                predictions[arm + '_closed'].append(full['P_closed'])
    data = {k: np.concatenate([by_fold[f][k] for f in FOLDS]) for k in META_KEYS}
    preds = {k: np.concatenate(v) for k, v in predictions.items()}
    if len(np.unique(data['idx'])) != len(data['idx']):
        raise ValueError('Heldout units covered more than once')
    for group in np.unique(data['merged_group']):
        if len(np.unique(data['original_fold'][data['merged_group'] == group])) != 1:
            raise ValueError('Merged group split across heldout folds')
    for fold in FOLDS:
        train_groups = {records[int(i)]['merged_group'] for i in fit_ids[('host', fold)]}
        held_groups = set(by_fold[fold]['merged_group'])
        if train_groups & held_groups:
            raise ValueError('Merged group crossed training/heldout boundary')
    return data, preds, hashes, dict(registration_commit=commit, data_sha256=DATA_SHA,
        scope_sha256=sha(scope), original_fit_units=len(original_fit),
        full_heldout_units=len(data['idx']), panel_heldout_units=int(data['panel'].sum()),
        original_heldout_folds=list(FOLDS), terminal_steps=900, seed=0,
        initial_host_parameter_sha256=next(iter(initial_hashes)), protocol=protocols[0],
        source_sha256=source_maps[0], files_sha256=hashes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=Path, required=True)
    parser.add_argument('--scope', type=Path, required=True)
    parser.add_argument('--scope-commit', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    jobs, scope, out = (under_root(p) for p in (args.jobs, args.scope, args.out))
    if out.exists():
        raise FileExistsError('Preserve existing scoring output: ' + relative(out))
    commit = verify_registration(args.scope_commit, scope)
    os.nice(max(0, 15 - os.getpriority(os.PRIO_PROCESS, 0)))
    data, predictions, before, provenance = read_verified_jobs(jobs, scope, commit)
    print('All six frozen 900-step receipts verified; endpoint arithmetic started.', flush=True)
    primary = score_surface(data, predictions)
    secondary = score_surface(data, predictions, panel=True)
    result = dict(schema='d09_endpoint_scores_v1', provenance=provenance,
                  primary=primary, panel_secondary=secondary, verdict=verdict(primary),
                  interpretations=[
                    'Two event-heldout folds inside original fold1 FIT. Original OUTER1 is excluded.',
                    'The input-only D08 roster used original fold1 FIT covariates, including these inner-heldout covariates; model preprocessing used inner FIT only. This is input-transductive exploration, not strict wholly unseen-event validation.',
                    'Scores use only terminal900 exports, original observation masks and original design weights; no step selection.',
                    'S means observed forecast-window peak >=.1; J means valid observed adjacent rise >=.01 including observed hour71 origin.',
                    'Severe false peaks are non-S observed-mask predicted peaks >=.1; zero-unit tiny false peaks use >.001.',
                    'Paired improvement versus host/old includes all jointly trained model changes; same-model closed comparisons are conditional exit interventions, not causal geographic contributions.',
                    'Peak ratios are dimensionless, and quantiles use cumulative design-mass knots; peak timing uses the first observed tied peak.',
                    'Missing observations break severe runs and adjacent pairs. Count and contiguous duration strata are distinct.',
                    'Bootstrap samples whole merged groups/families in fixed dominant-regime strata; invalid draws are counted. Fewer than two supporting clusters or no resampling-count variation in risk-contributing clusters gives no supported interval.',
                    'Singleton bootstrap strata have no within-stratum resampling variation. Empirical percentiles without supported resampling are descriptive repetitions, never CI evidence; intervals do not cover training-seed or panel-selection uncertainty.',
                    'Repeatedly explored D and one seed cannot establish independent confirmation, net geographic information or causal mechanisms.',
                    'Publish this registered candidate regardless of success; passing only permits proposing full-D evaluation, never launching it automatically.'])
    for name, digest in before.items():
        if sha(ROOT / name) != digest:
            raise ValueError('Input/source/output changed during score: ' + name)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_new(out, result)
    print(json.dumps(dict(result=relative(out), sha256=sha(out),
                          point_gate_pass=result['verdict']['point_gate_pass']), allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
