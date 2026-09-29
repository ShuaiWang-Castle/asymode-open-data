"""D04 public-D loader and adjacent-hour rows, without features or fitting.

The only accepted panel is the existing public D panel. Weather lag zero for a
strictly past predictor is weather[unit, time - 1]; this module does not expand
any history tensor. Stored y_full missing values remain NaN and obs_full remains
authoritative. Phase-zero rows sample one-hour differences every six hours.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '2'

import json
from pathlib import Path
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FEATURES = ROOT / 'data/interim/panel_v1/features_v1D.npz'
SPLITS = HERE / 'splits_v1D.json'
FIRST_FORECAST = 72
LAST_EXCLUSIVE = 216
WEATHER_NAMES = ['cape', 'cloud', 'gust', 'precip', 'pressure', 'rh', 'snowfall',
                 'soil_moisture', 't2m_c', 'u10', 'v10', 'wind_speed']
CONTEXT_NAMES = ['log_cust', 'rucc', 'log_pop_density', 'coop_share',
                 'log1p_n_utilities', 'log1p_saidi']


def _stream_columns(archive, member, columns, expected_units, static=False):
    """Read one county-event at a time from a C-ordered compressed NPY member."""
    with archive.open(member + '.npy') as stream:
        version = np.lib.format.read_magic(stream)
        shape, fortran, dtype = np.lib.format._read_array_header(stream, version)
        if (fortran or dtype.hasobject or len(shape) != 3 or
                shape[:2] != (expected_units, LAST_EXCLUSIVE) or max(columns) >= shape[2]):
            raise ValueError(f'Unexpected {member} NPY layout')
        out_shape = (shape[0], len(columns)) if static else (shape[0], shape[1], len(columns))
        out = np.empty(out_shape, dtype=np.float32)
        nbytes = shape[1] * shape[2] * dtype.itemsize
        for unit in range(shape[0]):
            buf = stream.read(nbytes)
            if len(buf) != nbytes:
                raise ValueError(f'Truncated {member} unit {unit}')
            block = np.frombuffer(buf, dtype=dtype).reshape(shape[1:])[:, columns]
            if static:
                if not np.array_equal(block, np.broadcast_to(block[0], block.shape), equal_nan=True):
                    raise ValueError(f'County context varies across hours: unit {unit}')
                out[unit] = block[0]
            else:
                out[unit] = block
        if stream.read(1):
            raise ValueError(f'Trailing bytes in {member}')
    return out


def _merged_groups(meta):
    """Original family plus shared-county/origin-within-16-days connected groups."""
    system = meta['system']
    names = np.unique(system)
    parent = {s: s for s in names}

    def find(s):
        while parent[s] != s:
            parent[s] = parent[parent[s]]
            s = parent[s]
        return s

    def union(a, b):
        aa, bb = find(a), find(b)
        parent[max(aa, bb)] = min(aa, bb)

    info = {}
    for s in names:
        rows = np.flatnonzero(system == s)
        for field in ('family', 'origin', 'regime'):
            if len(np.unique(meta[field][rows])) != 1:
                raise ValueError(f'Inconsistent {field} within system {s}')
        info[s] = (meta['family'][rows[0]], np.datetime64(meta['origin'][rows[0]]),
                   set(meta['county'][rows]))
    for i, a in enumerate(names):
        fa, oa, ca = info[a]
        for b in names[i + 1:]:
            fb, ob, cb = info[b]
            if fa == fb or (abs(oa - ob) <= np.timedelta64(16, 'D') and ca & cb):
                union(a, b)
    return np.array([find(s) for s in system])


def _original_folds(n_units, meta, group):
    split = json.loads(SPLITS.read_text())
    if split['n_units'] != n_units:
        raise ValueError('Original split and public D unit counts disagree')
    fold = np.zeros(n_units, dtype=np.int8)
    all_units = np.arange(n_units)
    for k in range(1, 6):
        entry = split['event'][str(k)]
        outer = np.asarray(entry['outer'], dtype=int)
        dev = np.asarray(entry['dev'], dtype=int)
        if (len(np.unique(outer)) != len(outer) or len(np.unique(dev)) != len(dev) or
                np.any(outer < 0) or np.any(outer >= n_units) or
                np.any(dev < 0) or np.any(dev >= n_units)):
            raise ValueError('Invalid or duplicate original event-fold indices')
        if not np.array_equal(np.sort(np.concatenate([outer, dev])), all_units):
            raise ValueError('Original outer/dev split is not a disjoint partition')
        if np.any(fold[outer]):
            raise ValueError('Unit belongs to more than one original outer fold')
        fold[outer] = k
    if np.any(fold == 0):
        raise ValueError('An original unit is never held out')
    for label in np.unique(group):
        if len(np.unique(fold[group == label])) != 1:
            raise ValueError('Merged group crosses original event folds')
    for system, expected in split['event_groups'].items():
        actual = fold[meta['system'] == system]
        if not len(actual) or np.any(actual != expected):
            raise ValueError('Original system-to-fold mapping disagrees')
    return fold


def load_panel():
    """Return unscaled public D arrays; never load another tranche or impute geo.

    Keys: weather [U,216,12] float32; geo [U,40]; context [U,6]; y_full
    [U,216]; obs_full bool; meta; fold int8; merged_group; feature_names.
    Meta retains all units, including used=False. No learned preprocessing is
    applied, so any later imputation/scaling must be fitted on training units.
    """
    with np.load(FEATURES, allow_pickle=False) as data:
        names = {'weather': data['damage_features'][:12].astype(str).tolist(),
                 'geo': data['geo_features'].astype(str).tolist(),
                 'context': data['recovery_features'][14:20].astype(str).tolist()}
        if names['weather'] != WEATHER_NAMES or names['context'] != CONTEXT_NAMES:
            raise ValueError('Public D channel order changed')
        geo = data['geo'].astype(np.float32, copy=False)
        y = data['y_full'].astype(np.float32, copy=False)
        obs = data['obs_full'].astype(bool, copy=False)
        meta = {k: data[k] for k in ('w', 'w_raw', 'system', 'family', 'origin', 'regime', 'used')}
        meta['county'] = data['fips'].astype(str)
        n = len(meta['system'])
        if n != 8457 or geo.shape != (n, 40) or y.shape != (n, 216) or obs.shape != y.shape:
            raise ValueError('Unexpected public D panel dimensions')
        if not np.array_equal(data['event'], meta['system']):
            raise ValueError('Event and system identity disagree')
        if not np.array_equal(data['y0'], y[:, 71]) or not np.all(obs[:, 71]):
            raise ValueError('Origin state must be the observed final prefix hour')
    if not np.isfinite(y[obs]).all() or np.any((y[obs] < 0) | (y[obs] > 1)):
        raise ValueError('Observed outage fraction must be finite in [0,1]')
    if np.any(~np.isfinite(meta['w'])) or np.any(meta['w'] <= 0):
        raise ValueError('Invalid clipped design weights')
    if np.any(~np.isfinite(meta['w_raw'])) or np.any(meta['w_raw'] <= 0):
        raise ValueError('Invalid raw design weights')
    if len(np.unique(np.column_stack([meta['system'], meta['county']]), axis=0)) != n:
        raise ValueError('Repeated county within one system')
    with zipfile.ZipFile(FEATURES) as archive:
        weather = _stream_columns(archive, 'xu', list(range(12)), n)
        context = _stream_columns(archive, 'xr', list(range(14, 20)), n, static=True)
    if not np.isfinite(weather).all() or not np.isfinite(context).all():
        raise ValueError('Nonfinite raw weather/context')
    group = _merged_groups(meta)
    fold = _original_folds(n, meta, group)
    return {'weather': weather, 'geo': geo, 'context': context,
            'y_full': y, 'obs_full': obs, 'meta': meta,
            'fold': fold, 'merged_group': group, 'feature_names': names}


def make_rows(panel, phase=None):
    """Adjacent observed one-hour differences on the forecast time grid.

    phase=0 selects 72,78,...,210; phase=1..5 selects that six-hour phase;
    phase=None retains all 72..215. Units with no valid adjacent pair have no
    rows and are explicitly visible in audit_counts. A unit's design mass is
    divided equally over its actually retained rows, separately for each phase.
    The response is a fraction difference, not percentage points or gross damage.
    """
    if phase is not None and (isinstance(phase, bool) or not isinstance(phase, (int, np.integer)) or not 0 <= phase < 6):
        raise ValueError('phase must be None or an integer in 0..5')
    y, obs = panel['y_full'], panel['obs_full']
    if y.shape != obs.shape or y.ndim != 2 or y.shape[1] != LAST_EXCLUSIVE:
        raise ValueError('Expected matching [unit,216] stock and observation arrays')
    times = np.arange(FIRST_FORECAST, LAST_EXCLUSIVE) if phase is None else np.arange(FIRST_FORECAST + phase, LAST_EXCLUSIVE, 6)
    valid = (obs[:, times] & obs[:, times - 1] &
             np.isfinite(y[:, times]) & np.isfinite(y[:, times - 1]))
    unit, column = np.nonzero(valid)
    time = times[column]
    n_units = len(y)
    count = np.bincount(unit, minlength=n_units)
    meta = panel['meta']
    if len(unit) != len(np.unique(unit.astype(np.int64) * LAST_EXCLUSIVE + time)):
        raise ValueError('Duplicate county-event/time row')
    if not np.all(obs[unit, time] & obs[unit, time - 1]):
        raise ValueError('Response crosses an unobserved adjacent hour')
    if not np.all((panel['fold'][unit] >= 1) & (panel['fold'][unit] <= 5)):
        raise ValueError('Row has no original outer fold')
    denominator = count[unit]
    rows = {'unit': unit.astype(np.int32), 'time': time.astype(np.int16),
            'y': y[unit, time].astype(np.float64) - y[unit, time - 1].astype(np.float64),
            'p_prev': y[unit, time - 1].copy(),
            'w': meta['w'][unit].astype(np.float64) / denominator,
            'w_raw': meta['w_raw'][unit].astype(np.float64) / denominator,
            'fold': panel['fold'][unit].copy(), 'group': panel['merged_group'][unit].copy(),
            'county': meta['county'][unit].copy(), 'regime': meta['regime'][unit].copy()}
    for key in ('w', 'w_raw'):
        restored = np.bincount(unit, weights=rows[key], minlength=n_units)
        present = count > 0
        if not np.allclose(restored[present], meta[key][present], rtol=1e-12, atol=1e-12):
            raise ValueError('Per-unit design weight not conserved')
    # No used filter: every valid row, regardless of prior-use status, is retained.
    if 'used' in meta:
        expected_unused = int(valid[~meta['used']].sum())
        if int((~meta['used'][unit]).sum()) != expected_unused:
            raise ValueError('Prior-use flag incorrectly excluded a valid row')
    return rows


def _row_audit(rows, panel, times):
    meta = panel['meta']
    unit = rows['unit']
    counts = np.bincount(unit, minlength=len(panel['y_full']))
    present = counts > 0
    y = rows['y']
    used = meta.get('used', np.zeros(len(counts), dtype=bool))
    out = {'candidate_rows': int(len(counts) * len(times)), 'retained_rows': len(unit),
           'excluded_rows': int(len(counts) * len(times) - len(unit)),
           'county_events': int(present.sum()), 'units_without_valid_rows': int((~present).sum()),
           'counties': len(np.unique(rows['county'])),
           'systems': len(np.unique(meta['system'][unit])),
           'families': len(np.unique(meta['family'][unit])),
           'merged_groups': len(np.unique(rows['group'])),
           'delta_sign_counts': {'negative': int((y < 0).sum()), 'zero': int((y == 0).sum()), 'positive': int((y > 0).sum())},
           'retained_rows_per_unit_quantiles': np.quantile(counts, [0, .01, .1, .5, .9, 1]).tolist(),
           'quantile_probabilities': [0, .01, .1, .5, .9, 1],
           'used_false_units_retained': int((present & ~used).sum()),
           'used_false_rows_retained': int((~used[unit]).sum()),
           'design_weight_sum': float(rows['w'].sum()), 'raw_weight_sum': float(rows['w_raw'].sum()),
           'fold_rows': {str(k): int((rows['fold'] == k).sum()) for k in range(1, 6)},
           'regime_rows': {str(r): int((rows['regime'] == r).sum()) for r in np.unique(meta['regime'])},
           'time_rows': {str(t): int((rows['time'] == t).sum()) for t in times}}
    if len(y):
        out['delta_fraction_quantiles'] = np.quantile(y, [0, .01, .1, .5, .9, .99, 1]).tolist()
        out['delta_quantile_probabilities'] = [0, .01, .1, .5, .9, .99, 1]
    return out


def audit_counts(trainrows, allrows, panel):
    """Compact count/weight audit for phase-zero training and all-hour evaluation."""
    if np.any((trainrows['time'] - FIRST_FORECAST) % 6):
        raise ValueError('Training rows must use phase zero')
    train_key = trainrows['unit'].astype(np.int64) * LAST_EXCLUSIVE + trainrows['time']
    all_key = allrows['unit'].astype(np.int64) * LAST_EXCLUSIVE + allrows['time']
    if not np.all(np.isin(train_key, all_key, assume_unique=True)):
        raise ValueError('Training rows are not a subset of all-hour rows')
    out = {'unit_total': len(panel['y_full']), 'prefix_hours': [0, 71],
           'forecast_hours': [72, 215], 'stock_unit': 'fraction in [0,1]',
           'response_definition': 'y_full[t] - y_full[t-1], both adjacent hours observed and finite',
           'weather_lag_zero_for_strict_past': 'weather[t-1]',
           'training_sampling': 'one-hour differences sampled at t=72,78,...,210',
           'evaluation_sampling': 'every adjacent one-hour difference at t=72,...,215',
           'weight_definition': 'existing county-event design weight divided by that unit\'s retained row count in this row set',
           'geo_missing_entries': int((~np.isfinite(panel['geo'])).sum()),
           'outage_unobserved_entries': int((~panel['obs_full']).sum()),
           'outage_nonfinite_entries': int((~np.isfinite(panel['y_full'])).sum()),
           'train': _row_audit(trainrows, panel, np.arange(72, 216, 6)),
           'all_hours': _row_audit(allrows, panel, np.arange(72, 216)),
           'limits': ['A reported hour may average only some of its four quarter-hour observations.',
                      'Adjacent-hour changes are net stock changes, not identified damage or restoration.',
                      'Reanalysis history indexes are strictly past, but do not guarantee real-time data availability.',
                      'Repeated windows and counties are dependent; row counts are not independent event counts.']}
    return out
