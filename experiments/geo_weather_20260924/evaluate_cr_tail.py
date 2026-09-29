"""Strict public-D I20 OOF evaluation, with outcome-defined tail diagnostics.

No model loading, fitting, weather loading, or paper imports. The main score is
pooled masked 144-hour RMSE, not peak error or an average of unit RMSEs.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'

import argparse
import hashlib
import json
from pathlib import Path
import re
import numpy as np

from d04_data import _merged_groups

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS = ROOT / 'runs/geo_weather_20260924'
FEATURES = ROOT / 'data/interim/panel_v1/features_v1D.npz'
SPLITS = HERE / 'splits_v1D.json'
REGIMES = ['tropical', 'winter', 'synoptic_wind', 'convective', 'heavy_rain']
LABEL, ARM, HOST = 'v1_crk_s0', 'CRK+Cin', 'v1_host_s0'
BOOTSTRAP, SEED = 1999, 20260928
PRIMARY_CHOICES = ('S', 'any_positive', 'J')


def clean(x):
    if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, np.ndarray): return clean(x.tolist())
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    if isinstance(x, (bool, np.bool_)): return bool(x)
    if isinstance(x, (int, np.integer)): return int(x)
    if isinstance(x, (float, np.floating)): return float(x) if np.isfinite(x) else None
    return x


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(clean(value), stream, indent=1, allow_nan=False)
        stream.write('\n')


def indices(value, n, description):
    idx = np.asarray(value)
    if (idx.ndim != 1 or not np.issubdtype(idx.dtype, np.integer)
            or np.any(idx < 0) or np.any(idx >= n) or len(np.unique(idx)) != len(idx)):
        raise ValueError(f'Invalid/duplicate indices: {description}')
    return idx


def validate_split(split, n, meta=None, group=None):
    if split['n_units'] != n or set(split['event']) != set(map(str, range(1, 6))):
        raise ValueError('Expected complete public-D five-event-fold split')
    expected, seen = {}, np.zeros(n, dtype=np.int16)
    fold = np.zeros(n, dtype=np.int8)
    for k in range(1, 6):
        entry = split['event'][str(k)]
        outer, dev = (indices(entry[key], n, f'fold {k} {key}') for key in ('outer', 'dev'))
        if not len(outer) or not np.array_equal(np.sort(np.r_[outer, dev]), np.arange(n)):
            raise ValueError('Outer/dev must be a disjoint complete partition')
        np.add.at(seen, outer, 1)
        fold[outer] = k
        expected[k] = outer
    if np.any(seen != 1): raise ValueError('Split coverage must be exactly once')
    if group is not None:
        for label in np.unique(group):
            if len(np.unique(fold[group == label])) != 1:
                raise ValueError('Merged group crosses original folds')
    if meta is not None:
        if set(split['event_groups']) != set(meta['system']):
            raise ValueError('System-to-fold split inventory differs')
        for system, k in split['event_groups'].items():
            if np.any(fold[meta['system'] == system] != k):
                raise ValueError('System-to-fold mapping differs')
    return expected, fold


def load_outcomes(path=FEATURES, split_path=SPLITS, expected_units=8457):
    """Only outcome/mask and small identity/weight arrays; never xu/xr/geo."""
    with np.load(path, allow_pickle=False) as z:
        y, raw_m = z['y'].astype('f8'), z['m']
        yf, raw_obs = z['y_full'].astype('f8'), z['obs_full']
        y0 = z['y0'].astype('f8')
        meta = {key: z[key].copy() for key in ('system', 'family', 'origin', 'regime', 'w', 'w_raw')}
        meta['county'] = z['fips'].astype(str)
        if not np.array_equal(z['event'], meta['system']): raise ValueError('Event/system mismatch')
    n = len(y)
    if (n != expected_units or y.shape != (n, 144) or raw_m.shape != y.shape
            or yf.shape != (n, 216) or raw_obs.shape != yf.shape or y0.shape != (n,)):
        raise ValueError('Unexpected public-D outcomes layout')
    if not np.isfinite(y).all():
        raise ValueError('Stored forecast y must be finite for the legacy evaluator; m remains authoritative')
    for name, a in [('m', raw_m), ('obs_full', raw_obs)]:
        if not np.isin(a, [0, 1]).all(): raise ValueError(f'{name} must be a binary mask')
    m, obs = raw_m.astype(bool), raw_obs.astype(bool)
    if not np.array_equal(m, obs[:, 72:]): raise ValueError('Forecast/full masks differ')
    if (not np.isfinite(yf[obs]).all() or np.any((yf[obs] < 0) | (yf[obs] > 1))
            or not np.array_equal(y[m], yf[:, 72:][m])):
        raise ValueError('Observed outcomes disagree or lie outside [0,1]')
    if not obs[:, 71].all() or not np.array_equal(y0, yf[:, 71]):
        raise ValueError('Origin must equal observed hour 71')
    for key, a in meta.items():
        if np.asarray(a).shape != (n,): raise ValueError(f'Bad metadata shape: {key}')
    for key in ('w', 'w_raw'):
        if not np.isfinite(meta[key]).all() or np.any(meta[key] <= 0):
            raise ValueError(f'Invalid design weights: {key}')
    if not np.isin(meta['regime'], REGIMES).all(): raise ValueError('Unknown regime')
    if len(np.unique(np.column_stack([meta['system'], meta['county']]), axis=0)) != n:
        raise ValueError('Duplicate county-system row')
    origin = np.asarray(meta['origin']).astype('datetime64[s]')
    if np.isnat(origin).any() or np.any(origin != origin.astype('datetime64[h]')):
        raise ValueError('Origins must be finite whole hours')
    group = _merged_groups(meta)
    if n == 8457 and len(np.unique(group)) != 68: raise ValueError('Expected 68 merged D groups')
    split = json.loads(Path(split_path).read_text())
    expected, fold = validate_split(split, n, meta, group)
    return dict(y=np.where(m, y, 0), m=m, y_full=yf, obs_full=obs, y0=y0,
                meta=meta, group=group, expected=expected, fold=fold, split=split)


def check_done(folder, label, arm, k, steps=900):
    d = json.loads((Path(folder) / 'DONE.json').read_text())
    expected = dict(label=label, arm=arm, data='v1D', design='event', design_weights=True,
                    fold=k, seed=0, steps=steps)
    for key, value in expected.items():
        if key not in d or type(d[key]) is not type(value) or d[key] != value:
            raise ValueError(f'{label} fold {k}: incompatible DONE field {key}')
    for key in ('phi', 'keep', 'ctx_geo', 'xu_phi'):
        if d.get(key) is not None: raise ValueError(f'{label}: unexpected {key}')
    for key in ('train_mask', 'mask_placebo'):
        if d.get(key, False) is not False: raise ValueError(f'{label}: unexpected {key}')
    if not (Path(folder) / 'final.pt').is_file(): raise ValueError('DONE without final checkpoint')
    return {key: d[key] for key in expected}


def load_predictions(label, arm, expected, n, runs=RUNS, steps=900):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', label): raise ValueError('Unsafe run label')
    P, seen, receipts = np.empty((n, 144), dtype='f8'), np.zeros(n, dtype=np.int16), {}
    for k in range(1, 6):
        folder = Path(runs) / label / f'fold{k:02d}'
        d = check_done(folder, label, arm, k, steps)
        path = folder / 'outer.npz'
        with np.load(path, allow_pickle=False) as z:
            idx = indices(z['idx'], n, f'{label} fold {k}')
            if not np.array_equal(np.sort(idx), np.sort(expected[k])):
                raise ValueError(f'{label} fold {k}: incorrect held-out membership')
            p = z['P']
            if (p.shape != (len(idx), 144) or p.dtype.kind not in 'fiu'
                    or not np.isfinite(p).all() or np.any((p < 0) | (p > 1))):
                raise ValueError(f'{label} fold {k}: invalid P')
            np.add.at(seen, idx, 1)
            P[idx] = p
        receipts[str(k)] = dict(metadata=d, outer_sha256=sha(path), done_sha256=sha(folder / 'DONE.json'))
    if np.any(seen != 1): raise ValueError(f'{label}: held-out coverage is not exactly once')
    return P, receipts


def cohorts(data):
    m, y = data['m'], data['y']
    N = m.sum(1)
    peak = np.max(np.where(m, y, -np.inf), axis=1)
    yf, obs = data['y_full'], data['obs_full']
    pair = obs[:, 72:] & obs[:, 71:215]
    dy = np.where(pair, yf[:, 72:] - yf[:, 71:215], -np.inf)
    jump = np.max(dy, axis=1)
    S, J = (N > 0) & (peak >= .1), pair.any(1) & (jump >= .01)
    return {'all': N > 0, 'S': S, 'J': J, 'S_and_J': S & J,
            'any_positive': (N > 0) & (peak > 0),
            'all_observed_zero': (N > 0) & (peak == 0),
            'nonS': (N > 0) & (peak < .1)}


def bootstrap_plan(meta, labels, draws=BOOTSTRAP, seed=SEED):
    """Sample whole clusters; stratum is largest full-D design-weight regime mass."""
    if not isinstance(draws, int) or draws < 1: raise ValueError('Positive bootstrap draw count required')
    levels, code = np.unique(labels, return_inverse=True)
    mass = np.zeros((len(levels), len(REGIMES)))
    rid = np.array([REGIMES.index(r) for r in meta['regime']])
    np.add.at(mass, (code, rid), np.asarray(meta['w'], dtype='f8'))
    strata = mass.argmax(1)
    count = np.zeros((draws, len(levels)), dtype=np.int32)
    rng = np.random.default_rng(seed)
    for r in range(len(REGIMES)):
        js = np.flatnonzero(strata == r)
        if len(js): count[:, js] = rng.multinomial(len(js), np.full(len(js), 1 / len(js)), size=draws)
    return dict(levels=levels, code=code, counts=count, strata=strata)


def interval(values, good, supporting_clusters):
    valid = good & np.isfinite(values)
    return dict(ci95=np.quantile(values[valid], [.025, .975]) if supporting_clusters >= 2 and valid.any() else None,
                valid_draws=int(valid.sum()), invalid_draws=int((~valid).sum()),
                supporting_clusters=int(supporting_clusters),
                interval_supported=bool(supporting_clusters >= 2 and valid.any()))


def support(data, selected, N, w):
    idx = selected & (N > 0)
    meta = data['meta']
    mass = w[idx] * N[idx]
    _, gc = np.unique(data['group'][idx], return_inverse=True)
    group_mass = np.bincount(gc, weights=mass)
    return dict(county_events=int(idx.sum()), counties=len(np.unique(meta['county'][idx])),
                systems=len(np.unique(meta['system'][idx])), families=len(np.unique(meta['family'][idx])),
                merged_groups=len(np.unique(data['group'][idx])), observed_hours=int(N[idx].sum()),
                weighted_observed_hours=float(mass.sum()), design_unit_mass=float(w[idx].sum()),
                unit_mass_kish=float(w[idx].sum() ** 2 / np.square(w[idx]).sum()) if idx.any() else 0,
                max_merged_group_weighted_hour_share=float(group_mass.max() / mass.sum()) if mass.sum() > 0 else None,
                merged_group_hour_mass_kish=float(mass.sum() ** 2 / np.square(group_mass).sum()) if mass.sum() > 0 else 0,
                full_144h_observed_count=int((idx & (data['m'].sum(1) == 144)).sum()))


def score_block(data, P, H, selected, w, plans=None, columns=None):
    y, m = data['y'], data['m']
    if columns is not None: y, m, P, H = (x[:, columns] for x in (y, m, P, H))
    N = m.sum(1)
    selected = np.asarray(selected, dtype=bool) & (N > 0)
    N = np.where(selected, N, 0)
    sw = np.where(selected, w, 0)
    se = np.column_stack([np.where(m, (p - y) ** 2, 0).sum(1) for p in (P, H, np.zeros_like(y))])
    ae = np.column_stack([np.where(m, abs(p - y), 0).sum(1) for p in (P, H, np.zeros_like(y))])
    den = float((sw * N).sum())
    mse = (sw @ se) / den if den > 0 else np.full(3, np.nan)
    rmse = np.sqrt(mse)
    mae = (sw @ ae) / den if den > 0 else np.full(3, np.nan)
    improvement = 1 - rmse[0] / rmse[1] if rmse[1] > 0 else np.nan
    out = dict(support=support(data, selected, N, w),
               models={name: dict(mse=mse[j], rmse=rmse[j], mae=mae[j])
                       for j, name in enumerate(('candidate', 'host', 'zero'))},
               rmse_improvement_fraction=improvement,
               rmse_change_fraction=-improvement, intervals={})
    for name, plan in (plans or {}).items():
        g = plan['code']; size = len(plan['levels'])
        gm = np.bincount(g, weights=sw * N, minlength=size)
        ga = np.bincount(g, weights=sw * se[:, 0], minlength=size)
        gh = np.bincount(g, weights=sw * se[:, 1], minlength=size)
        counts = plan['counts']
        bm, ba, bh = counts @ gm, counts @ ga, counts @ gh
        good = (bm > 0) & (bh > 0) & np.isfinite(ba) & np.isfinite(bh)
        values = np.full(len(counts), np.nan)
        values[good] = 1 - np.sqrt(ba[good] / bh[good])
        result = interval(values, good, int((gm > 0).sum()))
        result.update(zero_support_draws=int((bm <= 0).sum()),
                      zero_host_error_draws=int(((bm > 0) & (bh <= 0)).sum()))
        out['intervals'][name] = result
    return out


def alarm_block(data, P, H, selected, w, threshold, inclusive, plans):
    m = data['m']; selected = selected & m.any(1)
    peaks = [np.max(np.where(m, p, -np.inf), axis=1) for p in (P, H)]
    alarm = [p >= threshold if inclusive else p > threshold for p in peaks]
    sw = np.where(selected, w, 0)
    point = np.array([sw @ a / sw.sum() for a in alarm]) if sw.sum() else np.full(2, np.nan)
    out = dict(threshold=threshold, comparison='>=' if inclusive else '>',
               definition='Maximum P over observed forecast hours only; one binary event per county-event',
               support=support(data, selected, m.sum(1), w),
               candidate_rate=point[0], host_rate=point[1], difference=point[0] - point[1],
               candidate_count=int((alarm[0] & selected).sum()), host_count=int((alarm[1] & selected).sum()),
               intervals={})
    for name, plan in plans.items():
        g, counts = plan['code'], plan['counts']; size = len(plan['levels'])
        gm = np.bincount(g, weights=sw, minlength=size)
        gs = [np.bincount(g, weights=sw * a, minlength=size) for a in alarm]
        bm = counts @ gm; good = bm > 0
        values = np.full(len(counts), np.nan)
        values[good] = (counts @ (gs[0] - gs[1]))[good] / bm[good]
        out['intervals'][name] = interval(values, good, int((gm > 0).sum()))
    return out


def report(data, P, H, primary='S', draws=BOOTSTRAP):
    if primary not in PRIMARY_CHOICES: raise ValueError('Unknown primary cohort')
    if P.shape != data['y'].shape or H.shape != P.shape: raise ValueError('Prediction dimensions differ')
    for p in (P, H):
        if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)): raise ValueError('Invalid predictions')
    meta = data['meta']; subsets = cohorts(data)
    plans = {name: bootstrap_plan(meta, labels, draws) for name, labels in
             [('merged_event', data['group']), ('original_family', meta['family'])]}
    out = dict(definitions={
        'S': 'max observed stock over hours 72..215 >= 0.10',
        'J': 'max adjacent-observed stock increase at hours 72..215 >= 0.01; hour 71 is eligible prior',
        'any_positive': 'At least one observed forecast stock > 0',
        'S_and_J': 'Intersection of S and J, one county-event per row',
        'all_observed_zero': 'All observed forecast stocks zero and at least one observed hour; missing hours are unknown',
        'nonS': 'All observed forecast stocks below 0.10 and at least one observed hour',
        'primary_metric': 'sqrt(sum_i w_i SSE_i / sum_i w_i N_i), all observed hours 72..215 in selected units',
        'improvement': '1 - candidate RMSE / host RMSE; +0.10 means 10% lower RMSE',
        'horizons': 'Fixed-origin snapshots: +h is column h-1, panel hour 71+h; no shift or rolling reforecast',
        'interval_scope': 'Paired bootstrap of frozen OOF errors; conditional on observed cohorts and fitted models, not refit/seed uncertainty',
        'bootstrap_strata': 'Each whole cluster assigned to largest full-D clipped design unit weight mass regime; ties follow REGIMES order',
        'selection_limit': 'Outcome-defined evaluation subsets; not a prospective affected-county detector or causal estimand'},
        meta=dict(primary_cohort=primary, target_rmse_improvement=.10, draws=draws, seed=SEED,
                  regimes=REGIMES, units=len(P), no_observed_forecast_units=int((~data['m'].any(1)).sum()),
                  bootstrap_groups={name: len(plan['levels']) for name, plan in plans.items()},
                  bootstrap_stratum_groups={name: {r: int((plan['strata'] == j).sum())
                                                  for j, r in enumerate(REGIMES)}
                                            for name, plan in plans.items()}), weights={})
    for weight in ('w', 'w_raw', 'unweighted'):
        w = np.ones(len(P)) if weight == 'unweighted' else np.asarray(meta[weight], dtype='f8')
        summaries = {}
        for name in ('all', 'S', 'J', 'S_and_J', 'any_positive'):
            selected = subsets[name]
            summaries[name] = {'full_window': score_block(data, P, H, selected, w, plans),
                'by_regime': {r: score_block(data, P, H, selected & (meta['regime'] == r), w, plans) for r in REGIMES},
                'by_fold': {str(k): score_block(data, P, H, selected & (data['fold'] == k), w) for k in range(1, 6)},
                'horizon_snapshots': {str(h): score_block(data, P, H, selected, w, columns=[h-1]) for h in (1, 6, 24, 48)}}
        false_alarms = {}
        for name, threshold, inclusive in [('all_observed_zero', .001, False), ('nonS', .10, True)]:
            selected = subsets[name]
            false_alarms[name] = alarm_block(data, P, H, selected, w, threshold, inclusive, plans)
            false_alarms[name]['complete_144h'] = alarm_block(
                data, P, H, selected & data['m'].all(1), w, threshold, inclusive, plans)
            false_alarms[name]['by_regime'] = {r: alarm_block(data, P, H, selected & (meta['regime'] == r),
                                                             w, threshold, inclusive, plans) for r in REGIMES}
        out['weights'][weight] = dict(cohorts=summaries, false_alarms=false_alarms)
    p = out['weights']['w']['cohorts'][primary]['full_window']
    ci = p['intervals']['merged_event']['ci95']
    gain = p['rmse_improvement_fraction']
    out['primary_target'] = dict(cohort=primary, rmse_improvement_fraction=gain,
        point_target_met=bool(gain >= .10) if np.isfinite(gain) else None,
        ci95=ci, ci_supports_positive=bool(ci[0] > 0) if ci is not None else None,
        ci_supports_at_least_10pct=bool(ci[0] >= .10) if ci is not None else None,
        scope='Single seed, five held-out event folds. This target does not replace legacy scores or diagnose causality.')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--primary-cohort', choices=PRIMARY_CHOICES, default='S')
    ap.add_argument('--out', type=Path, default=HERE / 'results/v1/i20_cr_tail_s0.json')
    a = ap.parse_args()
    if a.out.exists(): raise FileExistsError(a.out)
    if os.getpriority(os.PRIO_PROCESS, 0) < 15: os.nice(15 - os.getpriority(os.PRIO_PROCESS, 0))
    data = load_outcomes()
    P, rp = load_predictions(LABEL, ARM, data['expected'], len(data['y']))
    H, rh = load_predictions(HOST, 'W+Cin', data['expected'], len(data['y']))
    result = report(data, P, H, a.primary_cohort)
    result['provenance'] = dict(candidate=LABEL, host=HOST, folds=[1, 2, 3, 4, 5], seed=0, steps=900,
                                exports={LABEL: rp, HOST: rh}, split_sha256=sha(SPLITS),
                                evaluator_sha256=sha(__file__))
    write_new(a.out, result)
    print(json.dumps(clean(result['primary_target']), allow_nan=False), flush=True)


if __name__ == '__main__': main()
