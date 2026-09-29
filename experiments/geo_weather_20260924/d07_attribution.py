"""D07 A: exact paired frozen public-D error attribution; no model fitting.

Only public-D outcome/mask/identity arrays and frozen prediction exports are
loaded. Unit details remain in an ignored local NPZ; the checked-in JSON uses
aggregate tables, including every registered regime, fold, and merged group.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'

import argparse
import json
from pathlib import Path
import numpy as np

from evaluate_cr_tail import (ROOT, HERE, RUNS, FEATURES, SPLITS, REGIMES,
    sha, clean, load_outcomes, load_predictions, cohorts, bootstrap_plan,
    interval, indices, write_new)

DATA_SHA = 'f043bb39e8abd48183670e2acc3c0cecebd7107ea9be0e28771912cfee358c48'
COHORTS = ('all', 'S', 'J', 'nonS')
RESULT = HERE / 'results/v1/d07_attribution.json'
D06 = RUNS / 'd06_host_interface_20260929'


def divide(a, b):
    return float(a / b) if b > 0 else None


def weighted_quantiles(x, w):
    x, w = np.asarray(x, float), np.asarray(w, float)
    if not len(x):
        return None
    if not np.isfinite(x).all() or not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError('Invalid weighted quantile support')
    order = np.argsort(x, kind='stable')
    mass = np.cumsum(w[order])
    return np.interp(np.array([.1, .5, .9]) * mass[-1], mass, x[order])


def paired_arrays(y, m, base, candidate):
    """Mask before arithmetic, then verify the identity for every unit."""
    y, m = np.asarray(y, float), np.asarray(m, bool)
    base, candidate = np.asarray(base, float), np.asarray(candidate, float)
    if not (y.shape == m.shape == base.shape == candidate.shape) or y.ndim != 2:
        raise ValueError('Paired prediction shape differs')
    if not np.isfinite(y[m]).all() or not np.isfinite(base).all() or not np.isfinite(candidate).all():
        raise ValueError('Nonfinite observed target or prediction')
    residual, delta = np.zeros_like(y), np.zeros_like(y)
    np.subtract(y, base, out=residual, where=m)
    np.subtract(candidate, base, out=delta, where=m)
    base_sse = np.square(residual).sum(1)
    candidate_sse = np.square(residual - delta).sum(1)
    alignment = 2 * np.einsum('ij,ij->i', residual, delta)
    energy = np.square(delta).sum(1)
    gain = base_sse - candidate_sse
    identity_error = gain - (alignment - energy)
    if not np.allclose(gain, alignment - energy, rtol=2e-12, atol=2e-13):
        raise AssertionError('Paired decomposition fails at unit level')
    n = m.sum(1)
    target_peak = np.max(np.where(m, y, -np.inf), 1)
    base_peak = np.max(np.where(m, base, -np.inf), 1)
    candidate_peak = np.max(np.where(m, candidate, -np.inf), 1)
    no_support = n == 0
    for peak in (target_peak, base_peak, candidate_peak):
        peak[no_support] = np.nan
    ratio_valid = (n > 0) & (target_peak > 0)
    base_ratio, candidate_ratio = np.full(len(n), np.nan), np.full(len(n), np.nan)
    np.divide(base_peak, target_peak, out=base_ratio, where=ratio_valid)
    np.divide(candidate_peak, target_peak, out=candidate_ratio, where=ratio_valid)
    # Positive is improvement toward the observed target peak, including true zero.
    peak_accuracy_gain = abs(base_peak - target_peak) - abs(candidate_peak - target_peak)
    return dict(n=n, base_sse=base_sse, candidate_sse=candidate_sse,
                alignment=alignment, energy=energy, gain=gain,
                identity_error=identity_error, target_peak=target_peak,
                base_peak=base_peak, candidate_peak=candidate_peak,
                ratio_valid=ratio_valid, base_ratio=base_ratio,
                candidate_ratio=candidate_ratio, peak_accuracy_gain=peak_accuracy_gain)


def concentration(magnitude, labels=None):
    """Shares of nonnegative weighted gains; ceiling cardinality is explicit."""
    magnitude = np.asarray(magnitude, float)
    if magnitude.ndim != 1 or not np.isfinite(magnitude).all() or np.any(magnitude < 0):
        raise ValueError('Concentration requires finite nonnegative magnitude')
    if labels is None:
        values = magnitude
    else:
        labels = np.asarray(labels)
        if labels.shape != magnitude.shape:
            raise ValueError('Concentration identity shape differs')
        _, code = np.unique(labels, return_inverse=True)
        values = np.bincount(code, weights=magnitude)
    total, positive = float(values.sum()), values[values > 0]
    ordered = np.sort(values)[::-1]
    top_k = int(np.ceil(.1 * len(values)))
    positive_k = int(np.ceil(.1 * len(positive)))
    return dict(entities=len(values), contributing_entities=len(positive), total=total,
                top10pct_all_entity_count=top_k,
                top10pct_all_entity_share=divide(ordered[:top_k].sum(), total),
                top10pct_contributing_entity_count=positive_k,
                top10pct_contributing_entity_share=divide(ordered[:positive_k].sum(), total),
                largest_entity_share=divide(ordered[0], total) if len(ordered) else None,
                effective_entities=divide(total * total, np.square(values).sum()))


def weighted_transition(eligible, a, b, w, a_levels, b_levels):
    eligible = np.asarray(eligible, bool)
    w = np.asarray(w, float)
    mass = float(w[eligible].sum())
    rows = []
    for aa in a_levels:
        for bb in b_levels:
            chosen = eligible & (a == aa) & (b == bb)
            rows.append([str(aa), str(bb), int(chosen.sum()), float(w[chosen].sum()),
                         divide(w[chosen].sum(), mass)])
    if sum(row[2] for row in rows) != int(eligible.sum()):
        raise AssertionError('Joint transition cells are not a complete partition')
    return dict(eligible_units=int(eligible.sum()), design_unit_mass=mass,
                columns=['trajectory_gain_sign', 'peak_change', 'units', 'design_unit_mass', 'design_unit_share'],
                rows=rows)


def sign_values(x):
    return np.where(x > 0, 'positive', np.where(x < 0, 'negative', 'zero'))


def bootstrap_summary(arrays, selected, w, plan):
    code, counts = plan['code'], plan['counts']
    size = len(plan['levels'])
    sw = w * selected
    gm = np.bincount(code, weights=sw * arrays['n'], minlength=size)
    gb = np.bincount(code, weights=sw * arrays['base_sse'], minlength=size)
    gc = np.bincount(code, weights=sw * arrays['candidate_sse'], minlength=size)
    ga = np.bincount(code, weights=sw * arrays['alignment'], minlength=size)
    ge = np.bincount(code, weights=sw * arrays['energy'], minlength=size)
    bm, bb, bc, ba, be = (counts @ z for z in (gm, gb, gc, ga, ge))
    groups = int((gm > 0).sum())
    result = {}
    good = (bm > 0) & (bb > 0)
    rmse_fraction = np.full(len(counts), np.nan)
    rmse_fraction[good] = 1 - np.sqrt(bc[good] / bb[good])
    result['rmse_improvement_fraction'] = interval(rmse_fraction, good, groups)
    result['rmse_improvement_fraction'].update(zero_support_draws=int((bm <= 0).sum()),
                                              zero_base_error_draws=int(((bm > 0) & (bb <= 0)).sum()))
    for key, numerator in [('mse_gain', bb - bc), ('alignment_per_weighted_hour', ba),
                            ('energy_per_weighted_hour', be)]:
        good = bm > 0
        values = np.full(len(counts), np.nan)
        values[good] = numerator[good] / bm[good]
        result[key] = interval(values, good, groups)
    positive = selected & (arrays['gain'] > 0)
    denominator = np.bincount(code, weights=sw, minlength=size)
    numerator = np.bincount(code, weights=w * positive, minlength=size)
    bd, bn = counts @ denominator, counts @ numerator
    good = bd > 0
    values = np.full(len(counts), np.nan)
    values[good] = bn[good] / bd[good]
    result['positive_unit_design_weight_share'] = interval(values, good, int((denominator > 0).sum()))
    return result


def aggregate(arrays, selected, w, meta, group, plan=None, rich=True, fit_objective=None):
    selected = np.asarray(selected, bool) & (arrays['n'] > 0)
    sw, n = w[selected], arrays['n'][selected]
    mass, den = float(sw.sum()), float(sw @ n)
    totals = {key: float(sw @ arrays[key][selected])
              for key in ('base_sse', 'candidate_sse', 'alignment', 'energy', 'gain')}
    error = totals['base_sse'] - totals['candidate_sse'] - (totals['alignment'] - totals['energy'])
    if abs(error) > 2e-11 * max(1., totals['base_sse'], totals['candidate_sse']):
        raise AssertionError('Weighted paired identity fails')
    base_mse, candidate_mse = divide(totals['base_sse'], den), divide(totals['candidate_sse'], den)
    base_rmse = np.sqrt(base_mse) if base_mse is not None else None
    candidate_rmse = np.sqrt(candidate_mse) if candidate_mse is not None else None
    positive = selected & (arrays['gain'] > 0)
    negative = selected & (arrays['gain'] < 0)
    tied = selected & (arrays['gain'] == 0)
    gain = w * arrays['gain']
    support = dict(units=int(selected.sum()), observed_hours=int(n.sum()), design_unit_mass=mass,
                   weighted_observed_hour_mass=den, counties=len(np.unique(meta['county'][selected])),
                   systems=len(np.unique(meta['system'][selected])), merged_groups=len(np.unique(group[selected])))
    signs = {name: dict(units=int(sel.sum()), design_unit_mass=float(w[sel].sum()),
                       unit_fraction=divide(sel.sum(), selected.sum()),
                       design_unit_share=divide(w[sel].sum(), mass),
                       weighted_gain=float(gain[sel].sum()))
             for name, sel in [('positive', positive), ('negative', negative), ('zero', tied)]}
    out = dict(support=support, base_rmse=base_rmse, candidate_rmse=candidate_rmse,
               rmse_improvement_fraction=(1 - candidate_rmse / base_rmse) if base_rmse else None,
               mse_gain=divide(totals['gain'], den), totals=totals,
               weighted_identity_abs_error=abs(error), unit_signs=signs)
    if fit_objective is not None:
        rw, full_fit_denominator = fit_objective
        objective = {key: float(rw[selected] @ arrays[key][selected]) / full_fit_denominator
                     for key in ('base_sse', 'candidate_sse', 'alignment', 'energy', 'gain')}
        objective['denominator'] = full_fit_denominator
        objective['selected_normalized_hour_mass'] = float(rw[selected] @ arrays['n'][selected])
        objective['definition'] = 'Subset numerator divided by the single original all-FIT sum_i rw_i N_i; rw is the original scaled float32 w_i/Z_regime row weight'
        out['original_fit_objective_contribution'] = objective
    if plan is not None:
        out['merged_event_bootstrap'] = bootstrap_summary(arrays, selected, w, plan)
    if not rich:
        return out
    out['concentration'] = {direction: {kind: concentration(magnitude[selected], labels[selected] if labels is not None else None)
        for kind, labels in [('unit', None), ('system', meta['system']), ('merged_group', group), ('county', meta['county'])]}
        for direction, magnitude in [('positive_gain', np.maximum(gain, 0)), ('negative_loss', np.maximum(-gain, 0))]}
    traj_sign = sign_values(arrays['gain'])
    peak_sign = sign_values(arrays['peak_accuracy_gain'])
    out['trajectory_by_peak_accuracy'] = weighted_transition(selected, traj_sign, peak_sign, w,
          ('positive', 'negative', 'zero'), ('positive', 'negative', 'zero'))
    valid = selected & arrays['ratio_valid']
    ratio_sign = sign_values(arrays['candidate_ratio'] - arrays['base_ratio'])
    out['trajectory_by_peak_ratio_direction'] = weighted_transition(valid, traj_sign, ratio_sign, w,
          ('positive', 'negative', 'zero'), ('positive', 'negative', 'zero'))
    out['peak_ratio'] = dict(eligible_units=int(valid.sum()), undefined_true_zero_units=int((selected & ~arrays['ratio_valid']).sum()),
                            base_q10_median_q90=weighted_quantiles(arrays['base_ratio'][valid], w[valid]),
                            candidate_q10_median_q90=weighted_quantiles(arrays['candidate_ratio'][valid], w[valid]))
    if valid.any():
        state = lambda x: np.where(x < 1, 'below', np.where(x > 1, 'above', 'equal'))
        base_state, candidate_state = state(arrays['base_ratio']), state(arrays['candidate_ratio'])
        transition = weighted_transition(valid, base_state, candidate_state, w,
                                       ('below', 'equal', 'above'), ('below', 'equal', 'above'))
        transition['columns'][0:2] = ['base_peak_ratio_state', 'candidate_peak_ratio_state']
        out['peak_ratio_state_transition'] = transition
    # Same observed mask for each arm; severe false peaks are defined by true non-S.
    non_s = selected & (arrays['target_peak'] < .1)
    ba, ca = arrays['base_peak'] >= .1, arrays['candidate_peak'] >= .1
    false_transition = weighted_transition(non_s, ba.astype(str), ca.astype(str), w,
                                           ('False', 'True'), ('False', 'True'))
    false_transition['columns'][0:2] = ['base_severe_false_peak', 'candidate_severe_false_peak']
    out['nonS_severe_false_peak_joint'] = false_transition
    return out


STRATIFIED_COLUMNS = ['partition', 'comparison', 'support', 'cohort', 'stratum',
    'units', 'observed_hours', 'design_unit_mass', 'weighted_observed_hour_mass',
    'base_rmse', 'candidate_rmse', 'rmse_improvement_fraction', 'alignment',
    'modification_energy', 'net_gain', 'positive_units', 'negative_units', 'zero_units',
    'positive_design_unit_mass', 'negative_design_unit_mass', 'positive_gain', 'negative_loss']


def compact_row(partition, comparison, sensitivity, cohort, stratum, summary):
    s, t, signs = summary['support'], summary['totals'], summary['unit_signs']
    return [partition, comparison, sensitivity, cohort, stratum, s['units'], s['observed_hours'],
            s['design_unit_mass'], s['weighted_observed_hour_mass'], summary['base_rmse'],
            summary['candidate_rmse'], summary['rmse_improvement_fraction'], t['alignment'],
            t['energy'], t['gain'], signs['positive']['units'], signs['negative']['units'],
            signs['zero']['units'], signs['positive']['design_unit_mass'],
            signs['negative']['design_unit_mass'], signs['positive']['weighted_gain'],
            -signs['negative']['weighted_gain']]


def load_closed_oof(data):
    n = len(data['y'])
    predictions, seen = np.empty((n, 144), float), np.zeros(n, np.int16)
    for fold in range(1, 6):
        path = RUNS / 'v1_crk_s0' / f'fold{fold:02d}' / 'outer.npz'
        with np.load(path, allow_pickle=False) as z:
            idx = indices(z['idx'], n, f'CRK closed fold {fold}')
            if not np.array_equal(np.sort(idx), np.sort(data['expected'][fold])):
                raise ValueError('Closed OOF membership differs')
            p = z['P_closed'].astype(float)
            if p.shape != (len(idx), 144) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
                raise ValueError('Invalid frozen closed prediction')
            predictions[idx] = p
            np.add.at(seen, idx, 1)
    if np.any(seen != 1):
        raise ValueError('Closed coverage must be exactly once')
    return predictions


def load_d06_scalars(data):
    receipt = json.loads((HERE / 'results/v1/d06_checkpoint_audit.json').read_text())
    outputs, hashes = {}, {}
    for label in ('v1_host_s0', 'v1_crk_s0'):
        name = label + '_fold1_scalars.npz'
        path = D06 / name
        digest = sha(path)
        if receipt['local_artifact_hashes'][name] != digest:
            raise AssertionError('D06 scalar receipt differs')
        if sha(RUNS / label / 'fold01/final.pt') != receipt['checkpoint_provenance'][label]['checkpoint_sha256']:
            raise AssertionError('D06 checkpoint differs')
        with np.load(path, allow_pickle=False) as z:
            p = z['P'].astype(float)
        if p.shape != data['y'].shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError('Invalid D06 fold1 P')
        with np.load(RUNS / label / 'fold01/outer.npz', allow_pickle=False) as z:
            replay = float(abs(p[z['idx']] - z['P']).max())
        if replay > 2e-6:
            raise AssertionError('D06 scalar OOF replay differs')
        outputs[label], hashes[name] = p, dict(sha256=digest, outer_replay_max_abs=replay)
    return outputs, dict(receipt_sha256=sha(HERE / 'results/v1/d06_checkpoint_audit.json'), artifacts=hashes)


def fit_weights(data):
    fit = np.asarray(data['split']['event']['1']['dev'], int)
    w = data['meta']['w'].astype(float)
    zero = np.square(data['y']).sum(1)  # load_outcomes zeroes missing targets.
    z = {regime: float(np.sum(w[fit] * zero[fit] * (data['meta']['regime'][fit] == regime)))
         for regime in REGIMES}
    if any(value <= 0 for value in z.values()):
        raise ValueError('Original FIT class zero-SSE denominator is nonpositive')
    rw = w / np.array([z[r] for r in data['meta']['regime']])
    return rw, float(rw[fit] @ data['m'][fit].sum(1)), z


def outcome_phenotypes(data):
    """Fixed outcome-only S phenotype cutpoints, registered before D07 results."""
    y, m = data['y'], data['m']
    severe_hours = np.sum(m & (y >= .1), axis=1)
    true_peak = np.max(np.where(m, y, -np.inf), axis=1)
    zero_sse = np.where(m, y * y, 0).sum(1)
    duration = [('severe_hours/1', severe_hours == 1),
                ('severe_hours/2..6', (severe_hours >= 2) & (severe_hours <= 6)),
                ('severe_hours/>=7', severe_hours >= 7)]
    peaks = [('true_peak/[.1,.2)', (true_peak >= .1) & (true_peak < .2)),
             ('true_peak/[.2,1]', (true_peak >= .2) & (true_peak <= 1))]
    joint = [(a + '|' + b, aa & bb) for a, aa in duration for b, bb in peaks]
    return duration + peaks + joint, dict(severe_hours=severe_hours, true_peak=true_peak, zero_sse=zero_sse)


def load_fold1_closed(path, data, opened):
    n = len(data['y'])
    with np.load(path, allow_pickle=False) as z:
        key = 'P_closed' if 'P_closed' in z.files else 'P'
        p = z[key].astype(float)
        if 'idx' in z.files:
            idx = indices(z['idx'], n, 'fold1 closed local export')
            if len(idx) != n or p.shape != (n, 144):
                raise ValueError('Fold1 closed must cover all FIT and OUTER units')
            ordered = np.empty_like(p)
            ordered[idx] = p
            p = ordered
        if p.shape != data['y'].shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError('Invalid fold1 closed local export')
        if 'P_open' in z.files:
            local_open = z['P_open'].astype(float)
            if 'idx' in z.files:
                local_open = local_open[np.argsort(idx)]
            if np.max(abs(local_open - opened)) > 2e-6:
                raise AssertionError('Fold1 local open replay differs')
    with np.load(RUNS / 'v1_crk_s0/fold01/outer.npz', allow_pickle=False) as z:
        error = float(abs(p[z['idx']] - z['P_closed']).max())
    if error > 2e-6:
        raise AssertionError('Fold1 closed OUTER replay differs')
    return p, dict(sha256=sha(path), outer_replay_max_abs=error)


def report(data, comparisons, original_objective=None):
    w, meta = data['meta']['w'].astype(float), data['meta']
    masks = cohorts(data)
    plan = bootstrap_plan(meta, data['group'], draws=1999, seed=20260928)
    rw, denominator, z = original_objective or fit_weights(data)
    phenotypes, phenotype_arrays = outcome_phenotypes(data)
    fit = np.zeros(len(w), bool)
    fit[data['split']['event']['1']['dev']] = True
    outer = np.zeros(len(w), bool)
    outer[data['expected'][1]] = True
    partitions = {'OOF': np.ones(len(w), bool), 'FIT_fold1': fit, 'OUTER_fold1': outer}
    strata = [('all', np.ones(len(w), bool))]
    strata += [('regime/' + regime, meta['regime'] == regime) for regime in REGIMES]
    strata += [('fold/' + str(fold), data['fold'] == fold) for fold in range(1, 6)]
    strata += [('group/' + str(group), data['group'] == group) for group in np.unique(data['group'])]
    summaries, rows, local = {}, [], {}
    for tag, comparison in comparisons.items():
        partition, base, candidate, description = comparison
        arrays = paired_arrays(data['y'], data['m'], base, candidate)
        selected_partition = partitions[partition]
        local[tag] = {key: value for key, value in arrays.items() if isinstance(value, np.ndarray)}
        summaries[tag] = dict(partition=partition, description=description,
                              maximum_unit_identity_abs_error=float(abs(arrays['identity_error']).max()), supports={})
        for sensitivity, window in [('common_observed', np.ones(len(w), bool)), ('complete144', data['m'].all(1))]:
            block = {}
            summaries[tag]['supports'][sensitivity] = block
            for cohort in COHORTS:
                selected = selected_partition & masks[cohort] & window
                objective = (rw, denominator) if partition == 'FIT_fold1' else None
                block[cohort] = aggregate(arrays, selected, w, meta, data['group'],
                                         plan if partition == 'OOF' else None, fit_objective=objective)
                for stratum, stratum_mask in strata:
                    ss = selected & stratum_mask
                    summary = aggregate(arrays, ss, w, meta, data['group'], rich=False)
                    rows.append(compact_row(partition, tag, sensitivity, cohort, stratum, summary))
                block[cohort]['by_regime_intervals'] = {
                    regime: bootstrap_summary(arrays, selected & (meta['regime'] == regime), w, plan)
                    for regime in REGIMES} if partition == 'OOF' else None
            phenotype_summaries = {}
            for phenotype, phenotype_mask in phenotypes:
                selected = selected_partition & masks['S'] & window & phenotype_mask
                summary = aggregate(arrays, selected, w, meta, data['group'],
                    plan if partition == 'OOF' else None, rich=False,
                    fit_objective=(rw, denominator) if partition == 'FIT_fold1' else None)
                summary['weighted_zero_SSE'] = float(w[selected] @ phenotype_arrays['zero_sse'][selected])
                summary['unweighted_zero_SSE'] = float(phenotype_arrays['zero_sse'][selected].sum())
                summary['base_peak_ratio_q10_median_q90'] = weighted_quantiles(arrays['base_ratio'][selected], w[selected])
                summary['candidate_peak_ratio_q10_median_q90'] = weighted_quantiles(arrays['candidate_ratio'][selected], w[selected])
                phenotype_summaries[phenotype] = summary
            block['S_outcome_phenotypes'] = phenotype_summaries
    return dict(summaries=summaries, stratified_table=dict(columns=STRATIFIED_COLUMNS, rows=rows),
                original_fold1_fit_objective=dict(regime_zero_SSE=z, denominator=denominator),
                bootstrap=dict(draws=1999, seed=20260928, groups=len(plan['levels']),
                   group_strata={str(level): REGIMES[plan['strata'][i]] for i, level in enumerate(plan['levels'])})), local


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=RESULT)
    parser.add_argument('--local-dir', type=Path)
    parser.add_argument('--fold1-closed', type=Path)
    args = parser.parse_args()
    if os.getpriority(os.PRIO_PROCESS, 0) < 15:
        os.nice(15 - os.getpriority(os.PRIO_PROCESS, 0))
    from d07_common import OUT, provenance, fit_weights as original_fit_weights
    local_dir = args.local_dir or OUT
    closed_path = args.fold1_closed or OUT / 'fold1_closed.npz'
    if args.out.exists():
        raise FileExistsError(args.out)
    if sha(FEATURES) != DATA_SHA:
        raise AssertionError('Public D data hash differs')
    shared_provenance = provenance()
    data = load_outcomes()
    n = len(data['y'])
    print('Validated public D outcomes and exactly-once fivefold OOF coverage', flush=True)
    candidate, crk_receipts = load_predictions('v1_crk_s0', 'CRK+Cin', data['expected'], n)
    host, host_receipts = load_predictions('v1_host_s0', 'W+Cin', data['expected'], n)
    if host_receipts != json.loads((RUNS / 'v1_crk_s0/HOST_REFERENCE.json').read_text()):
        raise AssertionError('Independent frozen host reference differs')
    closed = load_closed_oof(data)
    fold1, d06_receipts = load_d06_scalars(data)
    h1, c1 = fold1['v1_host_s0'], fold1['v1_crk_s0']
    comparisons = {
        'oof_independent_host': ('OOF', host, candidate, 'Candidate CRK open versus separately trained frozen host, all five OOF folds'),
        'oof_same_crk_exit': ('OOF', closed, candidate, 'Same frozen CRK: candidate open exit versus its exported closed exit, all five OOF folds'),
        'fit1_independent_host': ('FIT_fold1', h1, c1, 'Frozen fold1 CRK open versus separately trained frozen host on FIT; descriptive fitting residuals'),
        'outer1_independent_host': ('OUTER_fold1', h1, c1, 'Frozen fold1 CRK open versus separately trained frozen host on OUTER')}
    print('Computing primary host and frozen-exit attribution', flush=True)
    original_objective = original_fit_weights(data)
    result, local = report(data, comparisons, original_objective)
    primary = result['summaries']['oof_independent_host']['supports']['common_observed']
    print(json.dumps(clean({key: {field: primary[key][field] for field in ('base_rmse', 'candidate_rmse', 'rmse_improvement_fraction', 'unit_signs')}
                            for key in COHORTS}), allow_nan=False), flush=True)
    if not closed_path.is_file():
        print(f'DEPENDENCY_PENDING {closed_path.name}: primary computations complete; final exclusive output withheld', flush=True)
        return
    f1closed, closed_receipt = load_fold1_closed(closed_path, data, c1)
    comparisons.update({
        'fit1_same_crk_exit': ('FIT_fold1', f1closed, c1, 'Same frozen fold1 CRK open versus closed exit on FIT; descriptive fitting residuals'),
        'outer1_same_crk_exit': ('OUTER_fold1', f1closed, c1, 'Same frozen fold1 CRK open versus closed exit on OUTER')})
    print('Computing fold1 same-CRK exit attribution', flush=True)
    result, local = report(data, comparisons, original_objective)
    unit_path = local_dir / 'attribution_unit_rows.npz'
    unit_values = {'idx': np.arange(n), 'group': data['group'], 'cohort_S': cohorts(data)['S'],
                   'cohort_J': cohorts(data)['J'], 'fold': data['fold']}
    _, phenotype_arrays = outcome_phenotypes(data)
    unit_values.update(phenotype_arrays)
    for tag, arrays in local.items():
        unit_values.update({tag + '__' + key: value for key, value in arrays.items()})
    with unit_path.open('xb') as stream:
        np.savez_compressed(stream, **unit_values)
    scope = HERE / 'notes/D07_RESPONSE_SELECTIVITY_SCOPE_20260929.md'
    result.update(scope_commit='8f8ddef', phase='A', exploratory=True, seed=0, steps=900,
        definitions=dict(
            paired_identity='r=y-P_base; Delta=P_candidate-P_base; SSE_base-SSE_candidate=sum_i w_i sum_t m_it (2 r_it Delta_it-Delta_it^2)',
            alignment='sum_i w_i sum_t m_it 2r_it Delta_it; may be negative',
            modification_energy='sum_i w_i sum_t m_it Delta_it^2; nonnegative',
            pooled_rmse='sqrt(sum_i w_i SSE_i / sum_i w_i N_i); never mean unit RMSE',
            signs='Strict unit net-SSE sign (>0, <0, ==0), computed jointly rather than inferred from quantiles',
            common_observed='Both finite exported arms evaluated on the identical authoritative observed target mask, hours 72..215',
            complete144='Keep original outcome-defined cohorts and restrict to units with all 144 forecast hours observed',
            peak='Each arm and target peak is its maximum on that same observed forecast mask; ratio undefined if true peak is zero',
            peak_accuracy_gain='abs(base_peak-target_peak)-abs(candidate_peak-target_peak); positive means amplitude closer to target',
            ratio_direction='Candidate observed peak ratio minus base ratio; increase does not necessarily improve peak accuracy',
            severe_false_peak='For true non-S units (observed target peak <0.10), predicted observed-mask peak >=0.10',
            concentration='Absolute positive unit design-weighted SSE gains and absolute negative losses. Top10% cardinality uses ceil, independently of values; county aggregation can span multiple systems',
            bootstrap='1999 paired merged-event draws stratified by largest full-D design unit regime mass, seed20260928; conditional on frozen models and observed cohorts, no refitting',
            S='Observed forecast target maximum >=0.10',
            J='Maximum adjacent-observed target increment >=0.01; forecast hours72..215 with hour71 eligible as prior',
            nonS='Observed forecast target maximum <0.10',
            fit='FIT quantities are descriptive; original class-normalized contributions use original scaled float32 row weights and one all-FIT denominator even for subsets',
            S_phenotypes='Fixed before D07 results: observed severe hours y>=.10 bins 1,2..6,>=7 and observed true peak bins [.1,.2),[.2,1], with all six joint cells; no imputation'),
        provenance=dict(data_sha256=sha(FEATURES), splits_sha256=sha(SPLITS), scope_sha256=sha(scope),
            attribution_source_sha256=sha(__file__), common_source_sha256=sha(HERE / 'd07_common.py'),
            safe_loader_source_sha256=sha(HERE / 'evaluate_cr_tail.py'), shared_frozen_provenance=shared_provenance,
            host_oof_receipts=host_receipts, crk_oof_receipts=crk_receipts, d06_scalars=d06_receipts,
            fold1_closed=closed_receipt, local_unit_rows_sha256=sha(unit_path)))
    write_new(args.out, result)
    print(json.dumps(dict(output=args.out.name, sha256=sha(args.out), unit_rows_sha256=sha(unit_path),
                          strata_rows=len(result['stratified_table']['rows'])), allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
