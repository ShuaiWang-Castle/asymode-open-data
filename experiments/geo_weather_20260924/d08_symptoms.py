"""D08 fixed-roster symptom audit using existing fold1 FIT predictions only.

The input roster must be frozen and verified before any outcome is opened.
This does not load a model, run a forward pass, fit a model, or score an inner
holdout. Outcome-defined shapes are descriptive and never prediction inputs.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS = ROOT / 'runs/geo_weather_20260924'
RUNTIME = RUNS / 'd08_panel_20260929'
D07 = RUNS / 'd07_selectivity_20260929'
FEATURES = ROOT / 'data/interim/panel_v1/features_v1D.npz'
SPLITS = HERE / 'splits_v1D.json'
RESULT = HERE / 'results/v1/d08_panel_symptoms.json'
DATA_SHA = 'f043bb39e8abd48183670e2acc3c0cecebd7107ea9be0e28771912cfee358c48'
REGIMES = ('tropical', 'winter', 'synoptic_wind', 'convective', 'heavy_rain')
MODELS = ('host', 'crk', 'crk_closed', 'weather_control_0p001')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def strict_json(value):
    if isinstance(value, dict):
        return {str(k): strict_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [strict_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return strict_json(value.tolist())
    if isinstance(value, np.generic):
        return strict_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Nonfinite result cannot be silently replaced')
    return value


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(strict_json(value), stream, indent=1, ensure_ascii=False,
                  allow_nan=False)
        stream.write('\n')


def distribution(values, weights, midpoint=False):
    x, w = np.asarray(values, float), np.asarray(weights, float)
    valid = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not valid.any():
        return dict(n=0, missing=int(len(x)), mean=None, q10_median_q90=None,
                    minimum=None, maximum=None)
    x, w = x[valid], w[valid]
    order = np.argsort(x, kind='stable')
    cumulative = np.cumsum(w[order])
    knots = cumulative - .5*w[order] if midpoint else cumulative
    return dict(n=int(len(x)), missing=int((~valid).sum()),
                mean=float(np.sum(x * w) / w.sum()),
                q10_median_q90=np.interp(np.array([.1, .5, .9]) * cumulative[-1],
                                        knots, x[order]).tolist(),
                minimum=float(x.min()), maximum=float(x.max()))


def phenotype(y, mask, y0, full_y=None, full_mask=None):
    """Observed-only count, contiguous run and adjacent rise/fall at 72..215.

    Hour71 is used solely for the first forecast rise/fall. Missing observations
    break runs and adjacent pairs. First observed tied peak is chosen. Rise/fall
    are extrema of valid adjacent differences, not fitted physical rates.
    """
    y, mask = np.asarray(y, float), np.asarray(mask, bool)
    y0 = np.asarray(y0, float)
    if y.ndim != 2 or y.shape[1] != 144 or mask.shape != y.shape:
        raise ValueError('Expected aligned 144-hour arrays')
    if y0.shape != (len(y),) or not np.isfinite(y[mask]).all():
        raise ValueError('Invalid observed values')
    count = (mask & (y >= .1)).sum(1)
    running, longest = np.zeros(len(y), int), np.zeros(len(y), int)
    for t in range(144):
        running = np.where(mask[:, t] & (y[:, t] >= .1), running + 1, 0)
        longest = np.maximum(longest, running)
    valid = mask.any(1)
    peak = np.max(np.where(mask, y, -np.inf), axis=1)
    peak[~valid] = np.nan
    peak_time = np.argmax(np.where(mask, y, -np.inf), axis=1) + 72
    peak_time[~valid] = -1
    previous = np.column_stack((y0, y[:, :-1]))
    pair_mask = mask & np.column_stack((np.ones(len(y), bool), mask[:, :-1]))
    if full_y is not None:
        fy, fm = np.asarray(full_y), np.asarray(full_mask, bool)
        if fy.shape != (len(y), 216) or fm.shape != fy.shape:
            raise ValueError('Expected aligned full trajectories')
        if not np.array_equal(fm[:, 72:], mask) or not fm[:, 71].all():
            raise ValueError('Origin/forecast observation masks disagree')
        if not np.array_equal(fy[:, 71], y0):
            raise ValueError('Origin stock differs')
    delta = y - previous
    maximum_rise = np.max(np.where(pair_mask, delta, -np.inf), axis=1)
    maximum_fall = np.min(np.where(pair_mask, delta, np.inf), axis=1)
    maximum_rise[~pair_mask.any(1)] = np.nan
    maximum_fall[~pair_mask.any(1)] = np.nan
    before_peak = np.arange(72, 216)[None, :] <= peak_time[:, None]
    after_peak = np.arange(72, 216)[None, :] > peak_time[:, None]
    pre = pair_mask & before_peak
    post = pair_mask & after_peak
    pre_rise = np.max(np.where(pre, delta, -np.inf), axis=1)
    post_fall = np.min(np.where(post, delta, np.inf), axis=1)
    pre_rise[~pre.any(1)] = np.nan
    post_fall[~post.any(1)] = np.nan
    half_rise, half_fall = np.full(len(y), np.nan), np.full(len(y), np.nan)
    half_left, half_right = np.zeros(len(y), bool), np.zeros(len(y), bool)
    for i in np.flatnonzero(valid & (peak > 0)):
        t = int(peak_time[i] - 72)
        left = right = t
        while left > 0 and mask[i, left-1] and y[i, left-1] >= .5 * peak[i]:
            left -= 1
        while right < 143 and mask[i, right+1] and y[i, right+1] >= .5 * peak[i]:
            right += 1
        half_rise[i], half_fall[i] = t-left, right-t
        half_left[i] = left == 0 or not mask[i, left-1]
        half_right[i] = right == 143 or not mask[i, right+1]
    return dict(observed_hours=mask.sum(1), true_peak=peak, true_peak_hour=peak_time,
                severe_hours=count, maximum_contiguous_severe_run=longest,
                maximum_adjacent_rise=maximum_rise,
                minimum_adjacent_change=maximum_fall,
                maximum_rise_through_peak=pre_rise,
                minimum_change_after_peak=post_fall,
                peak_centered_half_height_rise_hours=half_rise,
                peak_centered_half_height_fall_hours=half_fall,
                half_height_left_truncated=half_left,
                half_height_right_truncated=half_right,
                adjacent_observed_pairs=pair_mask.sum(1),
                complete144=mask.all(1))


def model_peak(P, mask, true_peak, true_peak_hour):
    P, mask = np.asarray(P, float), np.asarray(mask, bool)
    pp = np.max(np.where(mask, P, -np.inf), axis=1)
    hour = np.argmax(np.where(mask, P, -np.inf), axis=1) + 72
    none = ~mask.any(1)
    pp[none], hour[none] = np.nan, -1
    ratio = np.full(len(P), np.nan)
    np.divide(pp, true_peak, out=ratio, where=np.isfinite(true_peak) & (true_peak > 0))
    lag = (hour - true_peak_hour).astype(float)
    lag[none] = np.nan
    return dict(predicted_peak=pp, predicted_peak_hour=hour,
                peak_ratio=ratio, signed_peak_lag_hours=lag)


def comparison(y, mask, base, candidate, selected, weight):
    """Whole-trajectory paired gains; positive concentration excludes losses."""
    ids = np.flatnonzero(selected)
    if not len(ids):
        return dict(units=0, net_gain=0., positive_gain=0., negative_loss=0.)
    residual, delta = y[ids] - base[ids], candidate[ids] - base[ids]
    alignment = (weight[ids, None] * mask[ids] * 2 * residual * delta).sum(1)
    energy = (weight[ids, None] * mask[ids] * delta ** 2).sum(1)
    gain = alignment - energy
    direct = (weight[ids, None] * mask[ids] *
              ((base[ids] - y[ids]) ** 2 - (candidate[ids] - y[ids]) ** 2)).sum(1)
    if not np.allclose(gain, direct, rtol=2e-10, atol=2e-10):
        raise AssertionError('Paired SSE identity failed')
    positive, negative = gain > 0, gain < 0
    positive_gain = float(gain[positive].sum())
    top_count = max(1, math.ceil(.1 * len(ids)))
    sorted_positive = np.sort(gain[positive])[::-1]
    return dict(units=int(len(ids)), alignment=float(alignment.sum()),
                modification_energy=float(energy.sum()), net_gain=float(gain.sum()),
                positive_gain=positive_gain, negative_loss=float(-gain[negative].sum()),
                positive_units=int(positive.sum()), negative_units=int(negative.sum()),
                zero_units=int((gain == 0).sum()),
                positive_design_share=float(weight[ids][positive].sum() / weight[ids].sum()),
                top_10pct_all_cohort_units=top_count,
                top_10pct_share_of_positive_gain=float(sorted_positive[:top_count].sum() / positive_gain)
                if positive_gain else None)


def cohort_masks(ph):
    valid = ph['observed_hours'] > 0
    S = valid & (ph['true_peak'] >= .1)
    result = dict(all=valid, S=S, nonS=valid & ~S)
    for name, values in [('severe_hour_count', ph['severe_hours']),
                         ('max_contiguous_run', ph['maximum_contiguous_severe_run'])]:
        for label, select in [('1', values == 1), ('2..6', (values >= 2) & (values <= 6)),
                              ('>=7', values >= 7)]:
            result[name + '/' + label] = S & select
    return result


def score_cohort(data, models, peaks, selected, pi, full_cohort, ph):
    """Local ratios and HT numerators; fixed original full-FIT denominators."""
    y, m, w, rw = (data[k] for k in ('y', 'm', 'w', 'rw'))
    ids = np.flatnonzero(selected)
    cohort_full_mass = float(np.sum(w[full_cohort, None] * m[full_cohort]))
    full_mass = float(np.sum(w[:, None] * m))
    full_denom = data['original_full_FIT_objective_denominator']
    if not len(ids):
        return dict(units=0, original_full_FIT_cohort_hour_mass=cohort_full_mass)
    local_mass = float(np.sum(w[ids, None] * m[ids]))
    local_w, ht_w = w[ids], w[ids] / pi[ids]
    row = dict(units=int(len(ids)), observed_hours=int(m[ids].sum()),
               complete144_units=int(ph['complete144'][ids].sum()),
               incomplete_units=int((~ph['complete144'][ids]).sum()),
               design_unit_mass=float(local_w.sum()), design_hour_mass=local_mass,
               HT_design_unit_mass=float(ht_w.sum()),
               HT_design_hour_mass=float(np.sum(ht_w[:, None] * m[ids])),
               original_full_FIT_cohort_hour_mass=cohort_full_mass,
               original_full_FIT_all_hour_mass=full_mass,
               original_full_FIT_objective_denominator=full_denom,
               outcome_shapes={}, models={}, comparisons={})
    for name in ('severe_hours', 'maximum_contiguous_severe_run',
                 'maximum_adjacent_rise', 'minimum_adjacent_change',
                 'maximum_rise_through_peak', 'minimum_change_after_peak',
                 'peak_centered_half_height_rise_hours',
                 'peak_centered_half_height_fall_hours',
                 'true_peak_hour'):
        row['outcome_shapes'][name] = dict(local=distribution(ph[name][ids], local_w),
                                          HT_reweighted=distribution(ph[name][ids], ht_w))
    for side in ('left', 'right'):
        indicator = ph['half_height_' + side + '_truncated'][ids]
        row['outcome_shapes']['half_height_' + side + '_truncated'] = dict(
            units=int(indicator.sum()), local_design_fraction=float(local_w[indicator].sum()/local_w.sum()),
            HT_design_fraction=float(ht_w[indicator].sum()/ht_w.sum()))
    for name, P in models.items():
        error = m[ids] * (P[ids] - y[ids]) ** 2
        numerator = float(np.sum(w[ids, None] * error))
        ht_num = float(np.sum(ht_w[:, None] * error))
        objective_num = float(np.sum(rw[ids, None] * error))
        ht_objective = float(np.sum((rw[ids] / pi[ids])[:, None] * error))
        model = dict(local_design_SSE=numerator,
                     local_design_RMSE=math.sqrt(numerator / local_mass) if local_mass else None,
                     HT_design_SSE_numerator=ht_num,
                     HT_MSE_full_FIT_cohort_denominator=ht_num / cohort_full_mass if cohort_full_mass else None,
                     HT_SSE_contribution_full_FIT_all_denominator=ht_num / full_mass,
                     local_original_objective_contribution=objective_num / full_denom,
                     HT_original_objective_contribution=ht_objective / full_denom,
                     local_peak_ratio=distribution(peaks[name]['peak_ratio'][ids], local_w),
                     HT_reweighted_peak_ratio=distribution(peaks[name]['peak_ratio'][ids], ht_w),
                     local_peak_lag=distribution(peaks[name]['signed_peak_lag_hours'][ids], local_w))
        nonS = selected & (ph['true_peak'] < .1)
        alarm = nonS & (peaks[name]['predicted_peak'] >= .1)
        model['nonS_severe_alarms'] = dict(units=int(alarm.sum()),
                  local_design_mass=float(w[alarm].sum()),
                  local_design_rate=float(w[alarm].sum()/w[nonS].sum()) if nonS.any() else None,
                  HT_design_mass=float(np.sum(w[alarm] / pi[alarm])),
                  HT_design_rate=float(np.sum(w[alarm] / pi[alarm]) /
                                       np.sum(w[nonS] / pi[nonS])) if nonS.any() else None)
        row['models'][name] = model
    for base, candidate in [('host', 'crk'), ('crk_closed', 'crk'),
                             ('crk', 'weather_control_0p001')]:
        key = candidate + '_vs_' + base
        row['comparisons'][key] = dict(
            local=comparison(y, m, models[base], models[candidate], selected, w),
            HT_reweighted=comparison(y, m, models[base], models[candidate], selected, w / pi))
    return row


def verify_roster(runtime):
    """The final freeze marker is mandatory; nothing outcome-bearing is read here."""
    marker = json.loads((runtime / 'FROZEN.json').read_text())
    if marker['status'] != 'input_roster_frozen':
        raise ValueError('Input roster is not frozen')
    for name, key in [('manifest.json', 'manifest_sha256'),
                      ('roster.json', 'roster_sha256'),
                      ('panel_inputs.npz', 'panel_inputs_sha256')]:
        if sha(runtime / name) != marker[key]:
            raise ValueError('Frozen roster hash differs: ' + name)
    manifest = json.loads((runtime / 'manifest.json').read_text())
    roster = json.loads((runtime / 'roster.json').read_text())
    provenance = manifest['provenance']
    if provenance['data_sha256'] != DATA_SHA or sha(SPLITS) != provenance['split_sha256']:
        raise ValueError('Frozen selector input provenance differs')
    for path_key, digest_key in [('source_path', 'source_sha256'), ('scope_path', 'scope_sha256')]:
        path = Path(provenance[path_key])
        if path.is_absolute() or '..' in path.parts or sha(ROOT / path) != provenance[digest_key]:
            raise ValueError('Frozen selector source/scope provenance differs')
    for path, digest in manifest['artifacts'].items():
        rel = Path(path)
        if rel.is_absolute() or '..' in rel.parts or sha(ROOT / rel) != digest:
            raise ValueError('Frozen selector artifact differs')
    with np.load(runtime / 'panel_inputs.npz', allow_pickle=False) as z:
        panel = {k: z[k].copy() for k in ('unit', 'original_fit', 'w', 'pi', 'core',
                                         'merged_group', 'original_fold', 'regime')}
    fit = panel['original_fit'].astype(np.int64)
    ids = panel['unit'].astype(np.int64)
    if len(fit) != 6350 or len(ids) != 1219:
        raise ValueError('Registered roster sizes differ')
    if len(np.unique(fit)) != len(fit) or len(np.unique(ids)) != len(ids):
        raise ValueError('Duplicate frozen unit IDs')
    if not np.isin(ids, fit).all():
        raise ValueError('Panel must be within original fold1 FIT')
    if not np.isfinite(panel['pi']).all() or np.any((panel['pi'] <= 0) | (panel['pi'] > 1)):
        raise ValueError('Invalid inclusion probabilities')
    if not np.isfinite(panel['w']).all() or np.any(panel['w'] <= 0):
        raise ValueError('Invalid original design weights')
    if not np.array_equal(np.asarray(roster['original_fit']), fit):
        raise ValueError('Roster/array original FIT differs')
    records = roster['records']
    for key in ('unit', 'w', 'pi', 'core', 'merged_group', 'original_fold', 'regime'):
        if not np.array_equal(np.asarray([r[key] for r in records]), panel[key]):
            raise ValueError('Roster/array field differs: ' + key)
    if marker['scope_commit'] != manifest['scope_commit']:
        raise ValueError('Freeze marker scope registration differs')
    return panel, manifest, roster, marker


def verified_package(name, audit):
    path = HERE / 'results/v1' / name
    record = audit['packages'][name]
    if sha(path) != record['gzip_sha256']:
        raise ValueError('Prior summary gzip hash differs')
    raw = gzip.decompress(path.read_bytes())
    if hashlib.sha256(raw).hexdigest() != record['uncompressed_sha256']:
        raise ValueError('Prior summary plain hash differs')
    return json.loads(raw)


def load_verified_fit(panel):
    """Read only registered public-D features; retain only the fold1 FIT outcomes."""
    if sha(FEATURES) != DATA_SHA:
        raise ValueError('Public D provenance differs')
    split = json.loads(SPLITS.read_text())
    fit = np.sort(panel['original_fit'].astype(np.int64))
    if not np.array_equal(fit, np.sort(np.asarray(split['event']['1']['dev']))):
        raise ValueError('Roster original FIT does not match frozen split')
    with np.load(FEATURES, allow_pickle=False) as z:
        data = {k: z[k][fit].copy() for k in ('y', 'm', 'y_full', 'obs_full', 'y0',
                                            'w', 'regime', 'system', 'family', 'fips', 'origin')}
    data['y'] = data['y'].astype(float)
    data['m'] = data['m'].astype(bool)
    data['w'] = data['w'].astype(float)
    if not np.isfinite(data['y']).all() or not np.isfinite(data['w']).all():
        raise ValueError('Invalid outcome or weight')
    if not np.array_equal(data['m'], data['obs_full'][:, 72:].astype(bool)):
        raise ValueError('Forecast/full observation masks differ')
    from d04_data import _merged_groups
    data['merged_group'] = _merged_groups(dict(system=data['system'], family=data['family'],
                                             origin=data['origin'], regime=data['regime'],
                                             county=data['fips']))
    data['original_fold'] = np.asarray([split['event_groups'][str(s)] for s in data['system']])
    audit = json.loads((HERE / 'results/v1/d07_final_audit.json').read_text())
    if not audit['passed'] or audit['public_D_hash'] != DATA_SHA:
        raise ValueError('Prior D07 audit is not successful')
    controls = verified_package('d07_controllers.json.gz', audit)
    gradients = verified_package('d07_gradients.json.gz', audit)
    cache_hash = dict(controls['local_artifact_hashes'], **gradients['local_artifacts'])
    for filename in ('controller_baseline_scalars.npz', 'host_fold1_scalars.npz',
                     'fold1_closed.npz', 'gradients_parameter_predictions.npz'):
        if sha(D07 / filename) != cache_hash[filename]:
            raise ValueError('Prior prediction cache differs: ' + filename)
    models, working = {}, {}
    for filename, model_key, value_key in [
            ('controller_baseline_scalars.npz', 'crk', 'P'),
            ('host_fold1_scalars.npz', 'host', 'P'),
            ('fold1_closed.npz', 'crk_closed', 'P_closed')]:
        with np.load(D07 / filename, allow_pickle=False) as z:
            if not np.array_equal(z['idx'], np.arange(8457)):
                raise ValueError('Cached prediction index order differs')
            models[model_key] = z[value_key][fit].astype(float)
            if model_key == 'crk':
                for name in ('jacobian_frobenius', 'jacobian_direction_norm',
                             'deposit_norm', 'temporal_deposit_cosine'):
                    working[name] = z[name][fit].astype(float)
    with np.load(D07 / 'gradients_parameter_predictions.npz', allow_pickle=False) as z:
        models['weather_control_0p001'] = z['weather_control_0p001'][fit].astype(float)
        if not np.allclose(z['baseline'][fit], models['crk'], atol=2e-8, rtol=0):
            raise ValueError('Weather perturbation baseline differs from CRK')
    for name, P in models.items():
        if P.shape != (6350, 144) or not np.isfinite(P).all() or np.any((P < 0) | (P > 1)):
            raise ValueError('Invalid cached FIT prediction: ' + name)
    Z = gradients['objective']['Z_regime']
    raw_weight = data['w'] / np.array([Z[r] for r in data['regime']])
    scale = float(data['m'].sum() / np.sum(data['m'] * raw_weight[:, None]))
    data['rw'] = (scale * raw_weight).astype(np.float32).astype(float)
    data['original_full_FIT_objective_denominator'] = gradients['objective']['common_full_FIT_denominator']
    data['original_regime_normalizers'] = Z
    data['fit_unit'] = fit
    return data, models, working, cache_hash, sha(HERE / 'results/v1/d07_final_audit.json')


def working_points(working, peaks, ph, selected, w, mask):
    result = {}
    clocks = np.arange(72, 216, 12) - 72
    subsets = dict(S=selected & (ph['true_peak'] >= .1),
                   nonS_alarm=selected & (ph['true_peak'] < .1) &
                   (peaks['crk']['predicted_peak'] >= .1))
    for cohort, select in subsets.items():
        ids = np.flatnonzero(select)
        result[cohort] = dict(units=len(ids), points={})
        for name, index in [('observed_true_peak', ph['true_peak_hour'] - 72),
                             ('crk_predicted_peak', peaks['crk']['predicted_peak_hour'] - 72)]:
            values = {}
            for key, arr in working.items():
                vals = arr[ids, index[ids]]
                if vals.ndim == 2:
                    vals = vals.mean(1)
                values[key] = distribution(vals, w[ids], midpoint=True)
            result[cohort]['points'][name] = values
        validclock = mask[ids][:, clocks]
        values = {}
        for key, arr in working.items():
            vals = arr[ids][:, clocks]
            if vals.ndim == 3:
                vals = vals.mean(2)
            values[key] = distribution(vals[validclock],
                                       np.broadcast_to(w[ids, None], validclock.shape)[validclock], midpoint=True)
        result[cohort]['points']['fixed_clock_72_every12_to204'] = values
    return result


def pair_descriptives(records, fit, panel_select, pi, data, models, peaks, ph):
    """Keep the selector's exact accepted pairs; no outcome-driven re-pairing."""
    selected = [r for r in records if r['status'] == 'selected']
    mapping = {int(unit): j for j, unit in enumerate(fit)}
    arrays = {key: [] for key in ('anchor_unit', 'partner_unit', 'kind',
                                  'weather_distance', 'geography_distance',
                                  'common_observed_hours', 'same_merged_group',
                                  'true_peak_difference', 'true_peak_hour_difference',
                                  'true_trajectory_RMS_difference')}
    for name in models:
        for key in ('predicted_peak_difference', 'peak_lag_difference',
                    'trajectory_RMS_difference', 'contrast_reproduction_RMSE'):
            arrays[name + '_' + key] = []
    y, m = data['y'], data['m']
    for record in selected:
        a, b = mapping[int(record['anchor_unit'])], mapping[int(record['partner_unit'])]
        if not panel_select[a] or not panel_select[b] or pi[a] != 1 or pi[b] != 1:
            raise ValueError('Accepted input-pair endpoints must be retained certainty units')
        common = m[a] & m[b]
        for key in ('anchor_unit', 'partner_unit', 'kind', 'weather_distance', 'geography_distance'):
            arrays[key].append(record[key])
        arrays['common_observed_hours'].append(int(common.sum()))
        arrays['same_merged_group'].append(bool(data['merged_group'][a] == data['merged_group'][b]))
        arrays['true_peak_difference'].append(ph['true_peak'][a] - ph['true_peak'][b])
        arrays['true_peak_hour_difference'].append(float(ph['true_peak_hour'][a] - ph['true_peak_hour'][b])
                                                  if ph['true_peak_hour'][a] >= 0 and ph['true_peak_hour'][b] >= 0 else np.nan)
        delta_y = y[a, common] - y[b, common]
        arrays['true_trajectory_RMS_difference'].append(float(np.sqrt(np.mean(delta_y ** 2))) if common.any() else np.nan)
        for name, P in models.items():
            contrast = P[a, common] - P[b, common]
            arrays[name + '_predicted_peak_difference'].append(
                peaks[name]['predicted_peak'][a] - peaks[name]['predicted_peak'][b])
            arrays[name + '_peak_lag_difference'].append(
                peaks[name]['signed_peak_lag_hours'][a] - peaks[name]['signed_peak_lag_hours'][b])
            arrays[name + '_trajectory_RMS_difference'].append(
                float(np.sqrt(np.mean(contrast ** 2))) if common.any() else np.nan)
            arrays[name + '_contrast_reproduction_RMSE'].append(
                float(np.sqrt(np.mean((contrast-delta_y) ** 2))) if common.any() else np.nan)
    arrays = {k: np.asarray(v) for k, v in arrays.items()}
    summary = {}
    for kind in ('near_weather_far_geo', 'near_geo_far_weather'):
        take = arrays['kind'] == kind
        summary[kind] = dict(accepted_pairs=int(take.sum()),
                            zero_common_observation_pairs=int((arrays['common_observed_hours'][take] == 0).sum()),
                            same_merged_group_pairs=int(arrays['same_merged_group'][take].sum()),
                            unweighted_descriptive_distributions={})
        for key, values in arrays.items():
            if key in ('kind', 'anchor_unit', 'partner_unit', 'same_merged_group'):
                continue
            summary[kind]['unweighted_descriptive_distributions'][key] = distribution(values[take], np.ones(take.sum()))
    return dict(selection='Exactly the input-only accepted pairs; all failures retained in frozen input manifest; no outcome reselection',
                definitions='Endpoint peaks use each endpoint observed support; trajectory contrasts use common observed hours at corresponding forecast-relative phases, not necessarily the same absolute clock. Signed differences are anchor minus partner. Model peak_lag_difference is the error in predicted timing contrast, not the true timing contrast.',
                interpretation='Dependent deterministic input pairs, not matched causal comparisons or population risk; infrastructure/background not balanced',
                accepted_pairs=len(selected), rows=summary), arrays


def compact_stratum(row):
    if row['units'] == 0:
        return row
    out = {k: row[k] for k in ('units', 'observed_hours', 'complete144_units', 'incomplete_units',
                               'design_unit_mass', 'design_hour_mass', 'HT_design_unit_mass')}
    out['models'] = {name: {k: value[k] for k in ('local_design_SSE', 'local_design_RMSE',
                        'HT_design_SSE_numerator', 'local_original_objective_contribution',
                        'HT_original_objective_contribution', 'nonS_severe_alarms')}
                     for name, value in row['models'].items()}
    out['comparisons'] = row['comparisons']
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', type=Path, default=RUNTIME)
    args = parser.parse_args()
    runtime = args.runtime
    if RESULT.exists() or (runtime / 'symptoms_arrays.npz').exists():
        raise FileExistsError('Preserve earlier symptom results')
    stable_paths = [Path(__file__), SPLITS, HERE / 'results/v1/d07_final_audit.json',
                    HERE / 'results/v1/d07_controllers.json.gz',
                    HERE / 'results/v1/d07_gradients.json.gz']
    stable_hashes = {path: sha(path) for path in stable_paths}
    panel, manifest, roster, marker = verify_roster(runtime)
    data, models, working, cache_hash, audit_hash = load_verified_fit(panel)
    fit = data['fit_unit']
    local = np.searchsorted(fit, panel['unit'])
    if not np.array_equal(fit[local], panel['unit']):
        raise ValueError('Panel IDs do not map exactly to full FIT')
    if not np.array_equal(data['w'][local], panel['w']):
        raise ValueError('Frozen panel original weights differ')
    for key in ('merged_group', 'regime', 'original_fold'):
        if not np.array_equal(data[key][local], panel[key]):
            raise ValueError('Frozen panel FIT metadata differs: ' + key)
    ph = phenotype(data['y'], data['m'], data['y0'], data['y_full'], data['obs_full'])
    peaks = {name: model_peak(P, data['m'], ph['true_peak'], ph['true_peak_hour'])
             for name, P in models.items()}
    masks = cohort_masks(ph)
    panel_select = np.zeros(len(fit), bool)
    panel_select[local] = True
    pi = np.ones(len(fit), float)
    pi[local] = panel['pi']
    results = {}
    for label, selected, probabilities in [('full_original_fold1_FIT', np.ones(len(fit), bool), np.ones(len(fit))),
                                            ('frozen_input_panel', panel_select, pi)]:
        results[label] = {}
        for support, supportmask in [('common_observed', ph['observed_hours'] > 0),
                                     ('complete144', ph['complete144']),
                                     ('incomplete', ~ph['complete144'])]:
            results[label][support] = {}
            for cohort, cmask in masks.items():
                results[label][support][cohort] = score_cohort(
                    data, models, peaks, selected & supportmask & cmask,
                    probabilities, supportmask & cmask, ph)
    eq_weight = np.empty(len(fit), float)
    full_group_mass = {}
    for group in np.unique(data['merged_group']):
        ii = data['merged_group'] == group
        full_group_mass[str(group)] = float(data['w'][ii].sum())
        eq_weight[ii] = data['w'][ii] / full_group_mass[str(group)]
    eq_data = dict(data, w=eq_weight)
    equal_group = {}
    for label, selected, probs in [('full_original_fold1_FIT', np.ones(len(fit), bool), np.ones(len(fit))),
                                   ('frozen_input_panel', panel_select, pi)]:
        equal_group[label] = {c: score_cohort(eq_data, models, peaks, selected & cmask,
                                             probs, cmask, ph)
                              for c, cmask in masks.items()}
    joint = {}
    S = masks['S']
    for label, selected in [('full_original_fold1_FIT', S), ('frozen_input_panel', S & panel_select)]:
        rows = []
        for count in (1, 2, 3):
            ci = np.where(ph['severe_hours'] == 1, 1, np.where(ph['severe_hours'] <= 6, 2, 3))
            ri = np.where(ph['maximum_contiguous_severe_run'] == 1, 1,
                          np.where(ph['maximum_contiguous_severe_run'] <= 6, 2, 3))
            for run in (1, 2, 3):
                ids = np.flatnonzero(selected & (ci == count) & (ri == run))
                rows.append(dict(count_bin=['1', '2..6', '>=7'][count-1],
                                 run_bin=['1', '2..6', '>=7'][run-1], units=len(ids),
                                 local_design_mass=float(data['w'][ids].sum()),
                                 HT_design_mass=float(np.sum(data['w'][ids]/pi[ids])) if label == 'frozen_input_panel' else float(data['w'][ids].sum())))
        joint[label] = rows
    support = {}
    for name in ('merged_group', 'regime', 'original_fold'):
        values = panel[name]
        support[name] = [dict(label=str(value), units=int((values == value).sum()),
                             S_units=int(S[local][values == value].sum()),
                             design_mass=float(panel['w'][values == value].sum()),
                             HT_design_mass=float(np.sum(panel['w'][values == value]/panel['pi'][values == value])))
                         for value in np.unique(values)]
    within_fit = {}
    for stratum in ('regime', 'merged_group'):
        within_fit[stratum] = {}
        for value in np.unique(data[stratum]):
            take = data[stratum] == value
            entry = {}
            for label, selected, probs in [('full_original_fold1_FIT', np.ones(len(fit), bool), np.ones(len(fit))),
                                           ('frozen_input_panel', panel_select, pi)]:
                entry[label] = {c: compact_stratum(score_cohort(data, models, peaks, selected & take & masks[c],
                                               probs, take & masks[c], ph))
                                for c in ('all', 'S', 'nonS')}
            within_fit[stratum][str(value)] = entry
    pairs, pair_arrays = pair_descriptives(manifest['pairs'], fit, panel_select, pi,
                                          data, models, peaks, ph)
    save = dict(unit=fit, panel=panel_select, pi=pi, y=data['y'], m=data['m'])
    save.update(ph)
    save.update({'P_' + name: P for name, P in models.items()})
    save.update({'pair_' + name: values for name, values in pair_arrays.items()})
    with (runtime / 'symptoms_arrays.npz').open('xb') as stream:
        np.savez_compressed(stream, **save)
    result = dict(phase='frozen_FIT_symptom_audit_complete',
                  input_roster_frozen_before_outcome_access=True,
                  no_new_forward_gradient_optimizer_or_training=True,
                  all_scores_are_original_fold1_FIT_descriptive_not_inner_holdout=True,
                  source_sha256=sha(__file__),
                  provenance=dict(input_manifest_sha256=marker['manifest_sha256'],
                                  roster_sha256=marker['roster_sha256'],
                                  panel_inputs_sha256=marker['panel_inputs_sha256'],
                                  public_D_sha256=DATA_SHA, splits_sha256=sha(SPLITS),
                                  D07_audit_sha256=audit_hash, D07_cache_sha256=cache_hash,
                                  local_arrays_sha256=sha(runtime/'symptoms_arrays.npz'),
                                  unchanged_source_receipts={str(path.relative_to(ROOT)): digest for path, digest in stable_hashes.items()}),
                  definitions=dict(forecast_hours=[72, 215], threshold=.1,
                    severe_hour_count='Count of observed hours >=0.1; may be disjoint',
                    max_contiguous_run='Longest consecutive observed >=0.1 run; missing hours break runs',
                    rise_fall='Observed adjacent difference with hour71 origin included for hour72; no wrap or interpolation',
                    peak_lag='First observed predicted peak hour minus first observed true peak hour',
                    half_height='Contiguous observed >=half of true peak episode containing the first peak; missing/forecast edges flag truncation',
                    HT='Panel inclusion-probability numerator; fixed original FIT denominators, no inference interval',
                    medians='Sampling-reweighted distributions are ratio estimators, not unbiased quantiles',
                    quantile_convention='Peaks/outcome shapes: interpolate cumulative weight knots as D07 A/D06; cached controller points: midpoint knots as D07 B. Cached four-mode values are averaged per working point before distribution.',
                    closed='Same frozen CRK exit closed, not independently trained host',
                    parameter_step='Existing D07 weather block .001 relative parameter norm intervention, not small output guarantee'),
                  original_regime_normalizers=data['original_regime_normalizers'],
                  scores=results, severe_count_vs_contiguous_run=joint,
                  equal_merged_group_descriptives=dict(
                    definition='Replace design weights by w_i / original full-FIT group total unit design mass; never normalize sampled groups. Original class objective stays unchanged.',
                    full_FIT_group_design_mass=full_group_mass, scores=equal_group),
                  panel_support_only=support,
                  within_FIT_regime_and_group_descriptives=within_fit,
                  frozen_input_pair_outcome_descriptives=pairs,
                  cached_control_working_points=dict(
                      full_original_fold1_FIT=working_points(working, peaks, ph, np.ones(len(fit), bool), data['w'], data['m']),
                      frozen_input_panel=working_points(working, peaks, ph, panel_select, data['w'], data['m'])),
                  limitations=['Panel was selected using inputs only, but these diagnostics use future outcomes and frozen FIT-trained predictions.',
                               'This cannot show inner event generalization, geography net information or causal impact.',
                               'No OUTER outcomes are analyzed; original folds2..5 are identity support labels only.'])
    for name, digest in cache_hash.items():
        if name in ('controller_baseline_scalars.npz', 'host_fold1_scalars.npz', 'fold1_closed.npz', 'gradients_parameter_predictions.npz'):
            if sha(D07 / name) != digest:
                raise AssertionError('Original cache changed during audit')
    if sha(FEATURES) != DATA_SHA:
        raise AssertionError('Original public D changed during audit')
    for path, digest in stable_hashes.items():
        if sha(path) != digest:
            raise AssertionError('Diagnostic source or prior receipt changed during audit')
    verify_roster(runtime)
    write_new(RESULT, result)
    print(json.dumps(dict(stage='complete', units_full_FIT=len(fit), units_panel=len(local),
                          result_sha256=sha(RESULT)), ensure_ascii=False))


if __name__ == '__main__':
    main()
