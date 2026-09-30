"""D10-A: paired allocation replay on every common D09 FIT case.

Only existing, hash-verified step0/100/300/900 caches are used. Registration and
all six original DONE/source/output guards precede cache arithmetic. Original
fold1 OUTER outcomes, models, forwards, gradients and optimizers are never used.
The two fits are training configurations, not independent heldout replicates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

import d09_score as frozen  # Fixes numerical threads before NumPy loads.
import d09_fit_replay as fit_replay
import numpy as np

TRAIN_COMMIT = 'eb91994fd124bc7178e7edc0b175c84113406b68'
TRAIN_SCOPE = frozen.HERE / 'notes/D09_INCREMENTAL_WRITE_DESIGN_20260929_ZH.md'
D10_SCOPE = frozen.HERE / 'notes/D10_RESPONSE_CHAIN_SCOPE_20260929_ZH.md'
JOBS = frozen.ROOT / 'runs/geo_weather_20260924/d09_write_screen_20260929/JOBS.json'
STEPS = (0, 100, 300, 900)
COHORTS = ('all', 'S', 'nonS')
COMPARISONS = tuple(c for c in frozen.COMPARISONS if c[1] in frozen.ARMS and c[2] in frozen.ARMS)
EXPECTED_COUNTS = {2: 871, 3: 883, 'common': 535, 'exclusive2': 336, 'exclusive3': 348}
NEW_SOURCES = tuple(frozen.HERE / name for name in
                    ('d10_common_fit.py', 'd10_replay.py', 'd10_downstream_oracle.py'))


def verify_registration(commit, scope):
    if len(commit) < 7 or any(c not in '0123456789abcdef' for c in commit):
        raise ValueError('Explicit hexadecimal D10 registration commit required')
    full = subprocess.check_output(['git', 'rev-parse', '--verify', commit + '^{commit}'],
                                   cwd=frozen.ROOT, text=True, stderr=subprocess.PIPE).strip()
    if Path(scope).resolve() != D10_SCOPE.resolve():
        raise ValueError('Use the fixed D10 response-chain scope')
    # The replay helper is reused as source, never as an existing result.
    for path in (*NEW_SOURCES, Path(fit_replay.__file__), Path(scope)):
        if hashlib.sha256(frozen.git_bytes(full, path)).hexdigest() != frozen.sha(path):
            raise ValueError('Registered diagnostic/helper/scope changed: ' + frozen.relative(path))
    return full


def require_close(left, right, label):
    if not np.allclose(left, right, rtol=2e-10, atol=2e-10):
        raise AssertionError('D10 additive identity failed: ' + label)


def optional_array(values):
    values = np.asarray(values, dtype=np.float64)
    if np.isinf(values).any():
        raise ValueError('Infinite scalar diagnostic')
    return [None if np.isnan(x) else float(x) for x in values]


def promote(metadata, predictions):
    data = dict(metadata)
    for key in ('y', 'y0', 'w', 'pi'):
        data[key] = np.asarray(metadata[key], dtype=np.float64)
    data['m'] = np.asarray(metadata['m'], dtype=bool)
    models = {arm: np.asarray(predictions[arm], dtype=np.float64) for arm in frozen.ARMS}
    if any(not np.isfinite(P).all() for P in models.values()):
        raise ValueError('Nonfinite cached predictions')
    return data, models


def row_gains(data, base, candidate):
    residual, delta = data['y'] - base, candidate - base
    alignment = data['w'] * np.sum(np.where(data['m'], 2 * residual * delta, 0.), axis=1)
    energy = data['w'] * np.sum(np.where(data['m'], delta ** 2, 0.), axis=1)
    gain = alignment - energy
    direct = data['w'] * np.sum(np.where(data['m'], residual ** 2 - (data['y'] - candidate) ** 2, 0.), axis=1)
    require_close(gain, direct, 'per-case 2rDelta-Delta^2')
    return gain, alignment, energy


def average_ranks(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind='stable')
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2 + 1
        start = end
    return ranks


def rank_correlation(left, right, weight):
    """Ordinary midranks; report ordinary and design-weighted Pearson of ranks."""
    left, right, weight = (np.asarray(x, dtype=np.float64) for x in (left, right, weight))
    valid = np.isfinite(left) & np.isfinite(right) & np.isfinite(weight) & (weight > 0)
    x, y, w = left[valid], right[valid], weight[valid]
    result = dict(units=int(valid.sum()), missing=int((~valid).sum()), supported=False,
                  ordinary_spearman=None, design_weighted_rank_correlation=None,
                  rank_convention='Ordinary average ranks for exact ties; weighted Pearson of these ranks uses original w, not weighted empirical ranks.')
    if len(x) < 3:
        return dict(result, reason='Fewer than three paired finite cases')
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return dict(result, reason='At least one paired variable is constant')
    a, b = average_ranks(x), average_ranks(y)
    def corr(weights):
        weights = weights / weights.sum()
        ac, bc = a - weights @ a, b - weights @ b
        denominator = math.sqrt(float(weights @ (ac ** 2)) * float(weights @ (bc ** 2)))
        return float(np.clip((weights @ (ac * bc)) / denominator, -1., 1.))
    return dict(result, supported=True, reason=None, ordinary_spearman=corr(np.ones(len(x))),
                design_weighted_rank_correlation=corr(w))


def transition_table(state2, state3, selected, weight, levels):
    """Exact disjoint case and original-design-mass table; no uncertainty claim."""
    state2, state3, selected = np.asarray(state2), np.asarray(state3), np.asarray(selected, bool)
    w = np.asarray(weight, dtype=np.float64)
    if not np.isin(state2[selected], levels).all() or not np.isin(state3[selected], levels).all():
        raise ValueError('Unrecognized transition state')
    counts = np.zeros((len(levels), len(levels)), dtype=np.int64)
    masses = np.zeros((len(levels), len(levels)), dtype=np.float64)
    for i, a in enumerate(levels):
        for j, b in enumerate(levels):
            mask = selected & (state2 == a) & (state3 == b)
            counts[i, j], masses[i, j] = int(mask.sum()), w[mask].sum()
    denom = float(w[selected].sum())
    if counts.sum() != selected.sum():
        raise AssertionError('Transition case conservation failed')
    require_close(masses.sum(), denom, 'transition design mass conservation')
    return dict(units=int(selected.sum()), design_unit_mass=denom,
                row_configuration=2, column_configuration=3, levels=list(levels),
                count_matrix=counts.tolist(), design_mass_matrix=masses.tolist(),
                design_share_matrix=(masses / denom).tolist() if denom else None,
                descriptive_only=True)


def positive_switch(gain2, gain3, selected, weight):
    state2, state3 = np.sign(gain2).astype(int), np.sign(gain3).astype(int)
    result = transition_table(state2, state3, selected, weight, (-1, 0, 1))
    w, denom = np.asarray(weight, float), result['design_unit_mass']
    enter, leave = selected & (gain2 <= 0) & (gain3 > 0), selected & (gain2 > 0) & (gain3 <= 0)
    mass2, mass3 = float(w[selected & (gain2 > 0)].sum()), float(w[selected & (gain3 > 0)].sum())
    require_close(mass3 - mass2, float(w[enter].sum() - w[leave].sum()), 'benefit coverage switches')
    result.update(configuration2_positive_units=int((selected & (gain2 > 0)).sum()),
                  configuration3_positive_units=int((selected & (gain3 > 0)).sum()),
                  enter_positive_units=int(enter.sum()), leave_positive_units=int(leave.sum()),
                  positive_design_share2=mass2 / denom if denom else None,
                  positive_design_share3=mass3 / denom if denom else None,
                  positive_design_share_difference=(mass3 - mass2) / denom if denom else None,
                  enter_positive_design_mass=float(w[enter].sum()), leave_positive_design_mass=float(w[leave].sum()),
                  net_gain2=float(np.asarray(gain2)[selected].sum()), net_gain3=float(np.asarray(gain3)[selected].sum()),
                  net_gain_difference3_minus2=float((np.asarray(gain3) - np.asarray(gain2))[selected].sum()))
    require_close(result['net_gain_difference3_minus2'], result['net_gain3'] - result['net_gain2'], 'common gain differences')
    return result


def recovery_switch(ratio2, ratio3, selected, weight):
    eligible = selected & np.isfinite(ratio2) & np.isfinite(ratio3)
    state2, state3 = (ratio2 >= .5).astype(int), (ratio3 >= .5).astype(int)
    result = transition_table(state2, state3, eligible, weight, (0, 1))
    result.update(threshold=.5, eligible_definition='Finite peak ratios: observed true peak >0; both predicted peaks use the same original observation mask.',
                  cohort_units=int(selected.sum()), ratio_missing_units=int((selected & ~eligible).sum()))
    w, denom = np.asarray(weight, float), result['design_unit_mass']
    enter, leave = eligible & (state2 == 0) & (state3 == 1), eligible & (state2 == 1) & (state3 == 0)
    recovered2, recovered3 = float(w[eligible & (state2 == 1)].sum()), float(w[eligible & (state3 == 1)].sum())
    require_close(recovered3 - recovered2, float(w[enter].sum() - w[leave].sum()), 'peak recovery switches')
    result.update(enter_recovery_units=int(enter.sum()), leave_recovery_units=int(leave.sum()),
                  recovered_design_share2=recovered2 / denom if denom else None,
                  recovered_design_share3=recovered3 / denom if denom else None,
                  recovered_design_share_difference=(recovered3 - recovered2) / denom if denom else None)
    return result


def subset(data, models, selected):
    return ({key: value[selected] for key, value in data.items()},
            {key: value[selected] for key, value in models.items()})


def partition_scores(data, models):
    ph = frozen.phenotype(data['y'], data['m'], data['y0'])
    cohorts = frozen.cohort_masks(ph)
    peaks = {arm: frozen.model_peak(P, data['m'], ph) for arm, P in models.items()}
    rows = {}
    for name in COHORTS:
        selected = cohorts[name]
        row = frozen.score_row(data, models, peaks, ph, selected, selected, plans={})
        row['original_FIT_partition_cohort_hour_mass'] = row.pop('original_full_held_cohort_hour_mass')
        row['descriptive_support'] = dict(units=row['units'], merged_groups=row['merged_groups'],
                                          at_least_two_groups=row['merged_groups'] >= 2,
                                          no_confidence_interval=True)
        rows[name] = row
    return dict(rows=rows, confidence_intervals=False, arithmetic_dtype='float64')


def mixture_decomposition(pool2, pool3, cohort, arm):
    r2 = {p: pool2[p]['rows'][cohort] for p in ('whole', 'common', 'exclusive')}
    r3 = {p: pool3[p]['rows'][cohort] for p in ('whole', 'common', 'exclusive')}
    s2 = {p: r2[p]['models'][arm]['design_SSE'] for p in r2}
    s3 = {p: r3[p]['models'][arm]['design_SSE'] for p in r3}
    h2 = {p: r2[p]['design_hour_mass'] for p in r2}
    h3 = {p: r3[p]['design_hour_mass'] for p in r3}
    for fold, s, h in ((2, s2, h2), (3, s3, h3)):
        require_close(s['whole'], s['common'] + s['exclusive'], f'SSE partition fit{fold}')
        require_close(h['whole'], h['common'] + h['exclusive'], f'observed-hour mass partition fit{fold}')
    require_close(h2['common'], h3['common'], 'same-case observed-hour mass')
    result = dict(configuration2_SSE=s2, configuration3_SSE=s3,
                  configuration2_hour_mass=h2, configuration3_hour_mass=h3,
                  total_SSE_difference3_minus2=s3['whole'] - s2['whole'],
                  common_SSE_difference3_minus2=s3['common'] - s2['common'],
                  exclusive_SSE_difference3_minus2=s3['exclusive'] - s2['exclusive'],
                  total_hour_mass_difference3_minus2=h3['whole'] - h2['whole'],
                  exclusive_hour_mass_difference3_minus2=h3['exclusive'] - h2['exclusive'],
                  MSE_decomposition=None)
    require_close(result['total_SSE_difference3_minus2'], result['common_SSE_difference3_minus2'] + result['exclusive_SSE_difference3_minus2'], 'total SSE difference partition')
    require_close(result['total_hour_mass_difference3_minus2'], result['exclusive_hour_mass_difference3_minus2'], 'total hour mass difference partition')
    if h2['whole'] and h3['whole']:
        risk2, risk3 = s2['whole'] / h2['whole'], s3['whole'] / h3['whole']
        common_term = s3['common'] / h3['whole'] - s2['common'] / h2['whole']
        exclusive_term = s3['exclusive'] / h3['whole'] - s2['exclusive'] / h2['whole']
        components = dict(configuration2_MSE=risk2, configuration3_MSE=risk3,
                          difference3_minus2=risk3 - risk2,
                          normalized_common_term=common_term, normalized_exclusive_term=exclusive_term,
                          symmetric_common_response_contrast=None,
                          symmetric_common_mass_mixture_contrast=None,
                          common_hourmass_share2=h2['common'] / h2['whole'],
                          common_hourmass_share3=h3['common'] / h3['whole'])
        if h2['common']:
            a2, a3 = components['common_hourmass_share2'], components['common_hourmass_share3']
            c2, c3 = s2['common'] / h2['common'], s3['common'] / h3['common']
            response, mix = .5 * (a2 + a3) * (c3 - c2), .5 * (c2 + c3) * (a3 - a2)
            components.update(symmetric_common_response_contrast=response,
                              symmetric_common_mass_mixture_contrast=mix)
            require_close(common_term, response + mix, 'symmetric common MSE decomposition')
        require_close(risk3 - risk2, common_term + exclusive_term, 'MSE mixture difference')
        result['MSE_decomposition'] = components
    return result


def gain_mixture_decomposition(pool2, pool3, cohort, label):
    r2 = {p: pool2[p]['rows'][cohort] for p in ('whole', 'common', 'exclusive')}
    r3 = {p: pool3[p]['rows'][cohort] for p in ('whole', 'common', 'exclusive')}
    gains = [{p: rows[p]['comparisons'][label]['paired_gain'] for p in rows} for rows in (r2, r3)]
    masses = [{p: rows[p]['design_unit_mass'] for p in rows} for rows in (r2, r3)]
    for fold, g, w in zip((2, 3), gains, masses):
        for field in ('net_gain', 'positive_gain', 'negative_loss', 'positive_units', 'negative_units', 'zero_units'):
            require_close(g['whole'][field], g['common'][field] + g['exclusive'][field], f'gain partition fit{fold}: {field}')
        require_close(w['whole'], w['common'] + w['exclusive'], f'case weight partition fit{fold}')
    require_close(masses[0]['common'], masses[1]['common'], 'same-case weight mass')
    result = dict(total_gain_difference3_minus2=gains[1]['whole']['net_gain'] - gains[0]['whole']['net_gain'],
                  common_gain_difference3_minus2=gains[1]['common']['net_gain'] - gains[0]['common']['net_gain'],
                  exclusive_gain_difference3_minus2=gains[1]['exclusive']['net_gain'] - gains[0]['exclusive']['net_gain'],
                  coverage_decomposition=None)
    require_close(result['total_gain_difference3_minus2'], result['common_gain_difference3_minus2'] + result['exclusive_gain_difference3_minus2'], 'gain difference mixture')
    if masses[0]['whole'] and masses[1]['whole']:
        positive_mass = []
        for g, w in zip(gains, masses):
            p = {part: w[part] * g[part]['positive_design_share'] if w[part] else 0. for part in w}
            require_close(p['whole'], p['common'] + p['exclusive'], 'positive case weight partition')
            positive_mass.append(p)
        rates = [positive_mass[k]['whole'] / masses[k]['whole'] for k in (0, 1)]
        common_term = positive_mass[1]['common'] / masses[1]['whole'] - positive_mass[0]['common'] / masses[0]['whole']
        exclusive_term = positive_mass[1]['exclusive'] / masses[1]['whole'] - positive_mass[0]['exclusive'] / masses[0]['whole']
        require_close(rates[1] - rates[0], common_term + exclusive_term, 'positive coverage difference mixture')
        result['coverage_decomposition'] = dict(configuration2_positive_design_share=rates[0],
                                               configuration3_positive_design_share=rates[1],
                                               difference3_minus2=rates[1] - rates[0],
                                               normalized_common_term=common_term,
                                               normalized_exclusive_term=exclusive_term,
                                               configuration2_positive_design_mass=positive_mass[0],
                                               configuration3_positive_design_mass=positive_mass[1])
    return result


def paired_common(data2, models2, data3, models3):
    frozen.assert_same_metadata(data2, data3)
    data, m, w = data2, data2['m'], data2['w']
    ph = frozen.phenotype(data['y'], m, data['y0'])
    cohorts = frozen.cohort_masks(ph)
    peaks = [{arm: frozen.model_peak(P, m, ph) for arm, P in models.items()} for models in (models2, models3)]
    sse = [{arm: w * np.sum(np.where(m, (P - data['y']) ** 2, 0.), axis=1) for arm, P in models.items()} for models in (models2, models3)]
    gains = [{label: row_gains(data, models[base], models[candidate]) for label, base, candidate in COMPARISONS} for models in (models2, models3)]
    row_arrays = dict(unit_ids=data['idx'].tolist(), observed_hours=ph['observed_hours'].tolist(),
                      original_w=w.tolist(), true_peak=optional_array(ph['true_peak']), S=cohorts['S'].tolist(),
                      models={}, comparisons={})
    model_changes = {}
    for arm in frozen.ARMS:
        delta = models3[arm] - models2[arm]
        hours = ph['observed_hours']
        per_rms, per_mean, per_max = np.full(len(w), np.nan), np.full(len(w), np.nan), np.full(len(w), np.nan)
        valid = hours > 0
        square = np.sum(np.where(m, delta ** 2, 0.), axis=1)
        per_rms[valid] = np.sqrt(square[valid] / hours[valid])
        per_mean[valid] = np.sum(np.where(m, delta, 0.), axis=1)[valid] / hours[valid]
        per_max[valid] = np.max(np.where(m, np.abs(delta), -np.inf), axis=1)[valid]
        row_arrays['models'][arm] = dict(configuration2_weighted_SSE=sse[0][arm].tolist(),
            configuration3_weighted_SSE=sse[1][arm].tolist(),
            weighted_SSE_difference3_minus2=(sse[1][arm] - sse[0][arm]).tolist(),
            configuration2_peak_ratio=optional_array(peaks[0][arm]['peak_ratio']),
            configuration3_peak_ratio=optional_array(peaks[1][arm]['peak_ratio']),
            configuration2_predicted_peak=optional_array(peaks[0][arm]['predicted_peak']),
            configuration3_predicted_peak=optional_array(peaks[1][arm]['predicted_peak']),
            configuration2_peak_hour=peaks[0][arm]['predicted_peak_hour'].tolist(),
            configuration3_peak_hour=peaks[1][arm]['predicted_peak_hour'].tolist(),
            prediction_difference_observed_RMS=optional_array(per_rms),
            prediction_difference_observed_mean3_minus2=optional_array(per_mean),
            prediction_difference_observed_max_absolute=optional_array(per_max))
        model_changes[arm] = {}
        for cohort in COHORTS:
            select = cohorts[cohort]
            hourmass = float(w @ (hours * select))
            delta_energy = float((w * square)[select].sum())
            model_changes[arm][cohort] = dict(units=int(select.sum()),
                original_design_hour_mass=hourmass,
                design_weighted_prediction_difference_RMS=math.sqrt(delta_energy / hourmass) if hourmass else None,
                prediction_difference_observed_mean3_minus2=frozen.distribution(per_mean[select], w[select]),
                prediction_difference_observed_RMS=frozen.distribution(per_rms[select], w[select]),
                maximum_absolute_observed_prediction_difference=float(per_max[select].max()) if select.any() else None,
                peak_ratio_rank_correlation=rank_correlation(peaks[0][arm]['peak_ratio'][select], peaks[1][arm]['peak_ratio'][select], w[select]),
                SSE_rank_correlation=rank_correlation(sse[0][arm][select], sse[1][arm][select], w[select]),
                peak_recovery_switch=recovery_switch(peaks[0][arm]['peak_ratio'], peaks[1][arm]['peak_ratio'], select, w),
                configuration3_vs_configuration2=frozen.gain_summary(data['y'], m, models2[arm], models3[arm], select, w))
    comparison_changes = {}
    for label, base, candidate in COMPARISONS:
        g2, a2, e2 = gains[0][label]
        g3, a3, e3 = gains[1][label]
        row_arrays['comparisons'][label] = dict(configuration2_gain=g2.tolist(), configuration3_gain=g3.tolist(),
            gain_difference3_minus2=(g3 - g2).tolist(),
            configuration2_alignment=a2.tolist(), configuration3_alignment=a3.tolist(),
            configuration2_modification_energy=e2.tolist(), configuration3_modification_energy=e3.tolist())
        comparison_changes[label] = {}
        ratio2, ratio3 = peaks[0][candidate]['peak_ratio'], peaks[1][candidate]['peak_ratio']
        for cohort in COHORTS:
            select = cohorts[cohort]
            eligible = select & np.isfinite(ratio2) & np.isfinite(ratio3)
            state2 = 2 * (g2 > 0).astype(int) + (ratio2 >= .5).astype(int)
            state3 = 2 * (g3 > 0).astype(int) + (ratio3 >= .5).astype(int)
            joint = transition_table(state2, state3, eligible, w, (0, 1, 2, 3))
            joint.update(state_definition={'0': 'No positive trajectory gain; peak ratio <.5',
                                           '1': 'No positive trajectory gain; peak ratio >=.5',
                                           '2': 'Positive trajectory gain; peak ratio <.5',
                                           '3': 'Positive trajectory gain; peak ratio >=.5'},
                         ratio_missing_units=int((select & ~eligible).sum()),
                         zero_trajectory_gain_included_in_nonpositive=True)
            comparison_changes[label][cohort] = dict(benefit_switch=positive_switch(g2, g3, select, w),
                gain_rank_correlation=rank_correlation(g2[select], g3[select], w[select]),
                joint_benefit_peak_recovery_switch=joint)
    return dict(model_configuration_changes=model_changes, comparison_allocation_changes=comparison_changes,
                common_per_case_scalars=row_arrays, scalar_array_order='All arrays follow common unit_ids exactly; no predicted or observed trajectory is published.')


def analyze_step(pools):
    common_ids = np.intersect1d(pools[2][0]['idx'], pools[3][0]['idx'])
    partitions, common = {}, {}
    for fold in frozen.FOLDS:
        data, models = promote(*pools[fold])
        selected = np.isin(data['idx'], common_ids)
        common[fold] = subset(data, models, selected)
        partitions[str(fold)] = dict(whole=partition_scores(data, models),
                                    common=partition_scores(*common[fold]),
                                    exclusive=partition_scores(*subset(data, models, ~selected)))
    frozen.assert_same_metadata(common[2][0], common[3][0])
    decompositions = {cohort: dict(models={arm: mixture_decomposition(partitions['2'], partitions['3'], cohort, arm) for arm in frozen.ARMS},
                                   comparisons={label: gain_mixture_decomposition(partitions['2'], partitions['3'], cohort, label) for label, _, _ in COMPARISONS})
                      for cohort in COHORTS}
    return dict(configurations=partitions, additive_decompositions=decompositions,
                common_paired=paired_common(*common[2], *common[3]))


def run(jobs_path, scope, registration_commit):
    registration = verify_registration(registration_commit, scope)
    training_registration = frozen.verify_registration(TRAIN_COMMIT, TRAIN_SCOPE)
    own = Path(__file__).resolve()
    local_sources = {frozen.relative(path): frozen.sha(path) for path in (*NEW_SOURCES, Path(fit_replay.__file__), Path(scope))}
    if Path(jobs_path).resolve() != JOBS.resolve():
        raise ValueError('Only the fixed six-job D09 inventory is permitted')
    # The original reader first completes all receipt/source/output guards. Its
    # original folds2/3 terminal outcomes are opened only for inherited identity
    # validation and immediately discarded, never scored or selected by D10-A.
    held, predictions, verified, provenance = frozen.read_verified_jobs(jobs_path, TRAIN_SCOPE, training_registration)
    del held, predictions
    jobs = json.loads(Path(jobs_path).read_text())['jobs']
    inventory = {(job['arm'], int(job['heldout_fold'])): frozen.under_root(job['path']) for job in jobs}
    receipts = {key: json.loads((folder / 'DONE.json').read_text()) for key, folder in inventory.items()}
    roster = {int(record['unit']): record for record in json.loads((frozen.D08 / 'roster.json').read_text())['records']}
    # Before opening any FIT outcome or prediction: exact geometry, all24
    # previously declared hashes, and all training-arm ID sets must agree.
    ids = {}
    for key, receipt in receipts.items():
        ids[key] = np.asarray(receipt['fit_unit_ids'], int)
        if len(ids[key]) != EXPECTED_COUNTS[key[1]] or not np.array_equal(ids[key], np.sort(np.unique(ids[key]))):
            raise ValueError('Unexpected original FIT identity geometry')
        for step in STEPS:
            path = inventory[key] / f'step{step:04d}_fit_panel.npz'
            name = frozen.relative(path)
            digest = receipt['outputs'].get(path.name)
            if not digest or verified.get(name) != digest or frozen.sha(path) != digest:
                raise ValueError('FIT cache absent from verified immutable inventory: ' + name)
    for fold in frozen.FOLDS:
        if any(not np.array_equal(ids[(arm, fold)], ids[('host', fold)]) for arm in frozen.ARMS):
            raise ValueError('Training arms use different FIT cases')
    common_ids = np.intersect1d(ids[('host', 2)], ids[('host', 3)])
    exclusive_ids = {fold: np.setdiff1d(ids[('host', fold)], common_ids) for fold in frozen.FOLDS}
    if len(common_ids) != EXPECTED_COUNTS['common'] or any(len(exclusive_ids[f]) != EXPECTED_COUNTS['exclusive' + str(f)] for f in frozen.FOLDS):
        raise ValueError('Unexpected common/exclusive FIT partition')
    if any(int(roster[int(i)]['original_fold']) not in (4, 5) for i in common_ids):
        raise ValueError('Common FIT must contain original folds4/5 only')
    for fold, other in ((2, 3), (3, 2)):
        if any(int(roster[int(i)]['original_fold']) != other for i in exclusive_ids[fold]):
            raise ValueError('Wrong original-fold exclusive partition')
    result, references, original_dtypes = {}, {}, {}
    for step in STEPS:
        pools = {}
        for fold in frozen.FOLDS:
            metadata, predictions = None, {}
            for arm in frozen.ARMS:
                path = inventory[(arm, fold)] / f'step{step:04d}_fit_panel.npz'
                with np.load(path, allow_pickle=False) as archive:
                    export = {key: archive[key].copy() for key in archive.files}
                fit_replay.validate_fit_export(export, receipts[(arm, fold)], roster)
                current = {key: export[key] for key in frozen.META_KEYS}
                if metadata is None:
                    metadata = current
                else:
                    frozen.assert_same_metadata(metadata, current)
                if fold not in references:
                    references[fold] = current
                else:
                    frozen.assert_same_metadata(references[fold], current)
                predictions[arm] = export['P']
                original_dtypes[f'{arm}_fit{fold}_step{step}'] = dict(P=str(export['P'].dtype), y=str(export['y'].dtype), w=str(export['w'].dtype))
            pools[fold] = (metadata, predictions)
        result[str(step)] = analyze_step(pools)
    all_files = dict(verified, **local_sources)
    for name, digest in all_files.items():
        if frozen.sha(frozen.ROOT / name) != digest:
            raise ValueError('Verified source/scope/input/output changed during D10: ' + name)
    population = dict(configuration2_units=len(ids[('host', 2)]), configuration3_units=len(ids[('host', 3)]),
                      common_units=len(common_ids), common_unit_ids=common_ids.tolist(),
                      configuration2_exclusive_unit_ids=exclusive_ids[2].tolist(),
                      configuration3_exclusive_unit_ids=exclusive_ids[3].tolist(),
                      common_original_folds=[4, 5], configuration2_exclusive_original_fold=3,
                      configuration3_exclusive_original_fold=2,
                      common_merged_groups=len({roster[int(i)]['merged_group'] for i in common_ids}),
                      common_unit_ids_sha256=hashlib.sha256(np.asarray(common_ids, dtype='<i8').tobytes()).hexdigest())
    return dict(schema='d10_common_FIT_v1', population=population, steps=result,
        provenance=dict(original_D09=provenance, files_sha256=all_files,
                        D10_registration_commit=registration, D10_scope_sha256=local_sources[frozen.relative(scope)],
                        source_path=frozen.relative(own), source_sha256=local_sources[frozen.relative(own)],
                        original_dtypes=original_dtypes, arithmetic_dtype='float64',
                        steps=list(STEPS), cached_FIT_exports=24, original_endpoint_reader_unmodified=True),
        decomposition_formulas=[
            'For each fixed cohort and arm: SSE_k = SSE_common,k + SSE_exclusive,k; H_k = H_common + H_exclusive,k. H is original w times original observed-hour count.',
            'MSE_k = a_k c_k + (1-a_k)e_k, with a_k=H_common/H_k, c_k=SSE_common,k/H_common, e_k=SSE_exclusive,k/H_exclusive,k; empty partitions contribute zero additive numerator, undefined local means stay null.',
            'MSE_3-MSE_2 = ((a_2+a_3)/2)(c_3-c_2) + ((c_2+c_3)/2)(a_3-a_2) + SSE_exclusive,3/H_3 - SSE_exclusive,2/H_2. This is an exact symmetric accounting identity, not a causal decomposition.',
            'Trajectory gain_k = sum_i w_i sum_t m_it(2(y-P_base,k)(P_candidate,k-P_base,k)-(P_candidate,k-P_base,k)^2); total/common/exclusive gains add exactly.',
            'Positive benefit coverage uses case-weight mass, not observed-hour mass: Wpositive_k=Wpositive_common,k+Wpositive_exclusive,k. The common-case coverage change equals entering-positive weight minus leaving-positive weight, divided by the same common cohort weight.',
            'Peak-ratio medians are separately reported on whole/common/exclusive sets. Quantiles and RMSE reductions have no additive decomposition.'],
        interpretations=[
            'All three arms and all four fixed cache steps are reported; no response-based county selection, step selection, refit or budget extension.',
            'Both configurations trained on every common original-fold4/5 case. Same-case contrasts diagnose sensitivity to the remaining training cases, not generalization or causal effects of training-set composition.',
            'Exclusive terms involve different cases and differently trained models. Existing caches do not identify how either model would predict the other exclusive pool; they are not pure causal composition effects.',
            'The two overlapping training configurations are not pooled as independent cases. No CI, bootstrap, new heldout coverage or advancement gate is produced.',
            'Observation mask, true S threshold >=.1, first observed tied peak, original county-event weights and cumulative-mass quantiles match frozen D09 arithmetic; no HT or class-normalized training loss is substituted.',
            'Configuration3-minus-configuration2 gain uses residual alignment minus modification energy in float64; comparison changes include all jointly trained model parameters, not isolated kernel effects.',
            'Peak ratio .5 means half the observed true peak, not fifty percentage points of outage. Peak recovery and positive trajectory gain can disagree.',
            'Rank correlations are null below three finite pairs or under constant ranks. Small-group descriptive summaries do not provide inferential support. Missing peak ratios from zero true peaks remain explicit nulls.',
            'S and peak-recovery labels are retrospective diagnostics only, never permitted future labels for routing or model inputs.',
            'Finite cached FIT evidence cannot establish geographic net information, complex-system causality, global effects or independent out-of-event confirmation.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration-commit', required=True)
    parser.add_argument('--scope', type=Path, required=True)
    parser.add_argument('--jobs', type=Path, default=JOBS)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    jobs, scope, out = (frozen.under_root(path) for path in (args.jobs, args.scope, args.out))
    if out.exists():
        raise FileExistsError('Preserve existing D10 common-FIT result: ' + frozen.relative(out))
    os.nice(max(0, 15 - os.getpriority(os.PRIO_PROCESS, 0)))
    result = run(jobs, scope, args.registration_commit)
    out.parent.mkdir(parents=True, exist_ok=True)
    frozen.write_new(out, result)
    print(json.dumps(dict(result=frozen.relative(out), sha256=frozen.sha(out),
                          common_cases=result['population']['common_units'], steps=list(STEPS),
                          no_training=True, no_forward=True, confidence_intervals=False), allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
