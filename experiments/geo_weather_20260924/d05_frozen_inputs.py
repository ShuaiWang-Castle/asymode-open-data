"""Read existing public-D neural exports and stored host inputs for D05.

No model is reconstructed, no prediction is recomputed, and no rate is converted
to a physical failure or restoration quantity. The command-line self-test uses
only temporary synthetic files. Real reads are available only through the two
public functions below.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import tempfile
import zipfile

import numpy as np

from d04_data import FEATURES, ROOT, SPLITS, WEATHER_NAMES, _stream_columns

RUNS = ROOT / 'runs/geo_weather_20260924'
PANELS = ROOT / 'data/interim/panel_v1/panels'
LABEL_ARMS = {'v1_host_s0': 'W+Cin', 'v1_gcrk_s0': 'GCRK+Cin',
              'v1_gcrk_georms_s0': 'GCRK+Cin-georms'}
DEFAULT_LABELS = tuple(LABEL_ARMS)
FIELDS = ('P', 'u', 'r', 'raw_logit', 'P_closed', 'raw_logit_closed')
EXTRA_NAMES = [
    'gust_excess_energy', 'wet_wind', 'snow_ice_load', 'near_freeze', 'cold_precip',
    'gust_excess_energy_sum6', 'wet_wind_sum12', 'gust_max72',
    'gust_excess_energy_sum72', 'wet_wind_sum72', 'snow_ice_load_sum72',
    'hours_since_gust_max72', 'gust_max6', 'gust_max12', 'gust_max24',
    'precip_sum6', 'precip_sum12', 'precip_sum24', 'wind_speed_mean12',
    'soil_moisture_mean24', 'subzero_hours24', 'zero_crossings24',
    'dir_change3', 'dir_change6', 'dir_sin', 'dir_cos', 'gust_cell_max',
    'gust_exceed15_share',
]
SEMANTIC_WARNING = {
    'status': 'Stored feature labels and values are preserved; cache correspondence requires the separate audit',
    'producer': 'experiments/geo_weather_20260924/build_v3p.py:88-94',
    'producer_Hg_order': ['gust_excess_energy', 'wet_wind', 'near_freeze', 'snow_ice_load', 'cold_precip'],
    'consumer': 'experiments/open_gcrk_20260919/build_features_r2.py:43-53',
    'consumer_mapping': 'Hg is read in the same order into a name-keyed dictionary; xu is then assembled by feature name',
    'xu_assembly': 'experiments/geo_weather_20260924/build_v3p.py:109-110',
    'source_expected_columns': 'xu[16] snow_ice_load = Hg[3]; xu[17] near_freeze = Hg[2]; xu[24] snow_ice_load_sum72 = trailing sum of Hg[3]',
    'source_definitions': 'experiments/open_gcrk_20260919/build_panel216_r2.py:67-78: nf=exp(-(t2m_c/3.5)^2); snow_ice_load=(snowfall+precip*I[t2m_c<=0.5])*nf*(0.25+rh/100)',
    'aggregation_limit': 'These nonlinear quantities are computed per grid cell before county weighting; nonlinear functions of county-mean raw weather are not an equality test',
    'interpretation_limit': 'Cache agreement verifies positional correspondence, not the physical formula, source weather correctness, or an outage mechanism',
}


def _layout(panel, expected_units=None):
    meta = panel['meta']
    n = len(meta['system'])
    if expected_units is not None and n != expected_units:
        raise ValueError('Only the existing 8457-unit public D panel is accepted')
    for key in ('system', 'county', 'origin'):
        if np.asarray(meta[key]).shape != (n,):
            raise ValueError(f'Invalid panel metadata shape: {key}')
    identity = np.column_stack((meta['system'], meta['county']))
    if len(np.unique(identity, axis=0)) != n:
        raise ValueError('Duplicate county-event in panel metadata')
    origin = np.asarray(meta['origin']).astype('datetime64[s]')
    if np.isnat(origin).any() or np.any(origin != origin.astype('datetime64[h]')):
        raise ValueError('Panel origins must be finite, whole-hour timestamps')
    for system in np.unique(meta['system']):
        if len(np.unique(origin[np.asarray(meta['system']) == system])) != 1:
            raise ValueError('Inconsistent origin within a system')
    fold = np.asarray(panel['fold'])
    if (fold.shape != (n,) or not np.issubdtype(fold.dtype, np.integer)
            or np.any((fold < 1) | (fold > 5))):
        raise ValueError('Invalid panel fold vector')
    return n


def _indices(values, n, description):
    idx = np.asarray(values)
    if (idx.ndim != 1 or not np.issubdtype(idx.dtype, np.integer)
            or np.any(idx < 0) or np.any(idx >= n)
            or len(np.unique(idx)) != len(idx)):
        raise ValueError(f'Invalid, duplicate or out-of-range indices: {description}')
    return idx


def _expected_folds(panel, split):
    n = _layout(panel)
    if split['n_units'] != n or set(split['event']) != set(map(str, range(1, 6))):
        raise ValueError('Expected all five public-D event folds')
    expected, seen = {}, np.zeros(n, dtype=np.int16)
    for k in range(1, 6):
        entry = split['event'][str(k)]
        idx = _indices(entry['outer'], n, f'split fold {k}')
        dev = _indices(entry['dev'], n, f'development fold {k}')
        if not len(idx) or not np.array_equal(np.sort(np.r_[idx, dev]), np.arange(n)):
            raise ValueError(f'Fold {k} is not a complete disjoint partition')
        if not np.array_equal(np.sort(idx), np.flatnonzero(panel['fold'] == k)):
            raise ValueError(f'Panel fold vector disagrees with split fold {k}')
        expected[k] = idx
        np.add.at(seen, idx, 1)
    if np.any(seen != 1):
        raise ValueError('Split coverage must be exactly once per county-event')
    return expected


def _metadata(folder, label, k):
    metadata = json.loads((folder / 'DONE.json').read_text())
    expected = dict(label=label, arm=LABEL_ARMS[label], data='v1D', design='event',
                    design_weights=True, fold=k, seed=0, steps=900)
    for key, value in expected.items():
        if (key not in metadata or type(metadata[key]) is not type(value)
                or metadata[key] != value):
            raise ValueError(f'{label} fold {k}: incompatible DONE field {key}')
    # Extra legacy metadata is neither required nor copied into public reports.
    return {key: metadata[key] for key in expected}


def _load_frozen(panel, labels, runs, split):
    n = _layout(panel)
    labels = tuple(labels)
    if (not labels or len(set(labels)) != len(labels)
            or any(label not in LABEL_ARMS for label in labels)):
        raise ValueError('Choose distinct labels from DEFAULT_LABELS')
    expected = _expected_folds(panel, split)
    models = {}
    for label in labels:
        arrays = {field: None for field in FIELDS}
        missing = {field: [] for field in FIELDS[1:]}
        seen = np.zeros(n, dtype=np.int16)
        metadata = {}
        for k in range(1, 6):
            folder = runs / label / f'fold{k:02d}'
            metadata[str(k)] = _metadata(folder, label, k)
            with np.load(folder / 'outer.npz', allow_pickle=False) as archive:
                if 'idx' not in archive or 'P' not in archive:
                    raise ValueError(f'{label} fold {k}: required idx/P missing')
                idx = _indices(archive['idx'], n, f'{label} fold {k}')
                if not np.array_equal(np.sort(idx), np.sort(expected[k])):
                    raise ValueError(f'{label} fold {k}: held-out indices disagree')
                np.add.at(seen, idx, 1)
                for field in FIELDS:
                    if field not in archive:
                        missing[field].append(k)
                        continue
                    values = archive[field]
                    if (values.shape != (len(idx), 144) or values.dtype.kind not in 'fiu'
                            or not np.isfinite(values).all()):
                        raise ValueError(f'{label} fold {k}: invalid {field} array')
                    if field in ('P', 'P_closed') and np.any((values < 0) | (values > 1)):
                        raise ValueError(f'{label} fold {k}: {field} outside [0,1]')
                    if arrays[field] is None:
                        arrays[field] = np.empty((n, 144), dtype=values.dtype)
                    elif arrays[field].dtype != values.dtype:
                        raise ValueError(f'{label}: {field} dtype differs across folds')
                    arrays[field][idx] = values
        if np.any(seen != 1):
            raise ValueError(f'{label}: held-out coverage must be exactly once')
        # A partially available optional field is not a usable all-unit export.
        # Return None, never uninitialized values or silent zero imputation.
        missing = {field: folds for field, folds in missing.items() if folds}
        for field in missing:
            arrays[field] = None
        models[label] = {**arrays,
                         'available_fields': [field for field in FIELDS if arrays[field] is not None],
                         'missing_folds': missing, 'fold_metadata': metadata}
    return {'models': models, 'time': np.arange(72, 216, dtype=np.int16),
            'meta': {'units': n, 'folds': [1, 2, 3, 4, 5], 'labels': list(labels),
                     'source_pattern': 'runs/geo_weather_20260924/{label}/fold{fold:02d}/outer.npz',
                     'row_identity': 'Original public-D unit index, aligned to panel.meta',
                     'time_axis': 'Column j is panel hour t=72+j, j=0..143',
                     'absolute_time': 'panel.meta.origin[unit] + (t-72) hours, UTC',
                     'origin_source': 'Only panel.meta.origin; no label-based date inference',
                     'P_units': 'Hourly customer-outage fraction; 0.01 is one percentage point',
                     'initial_state': 'Original open-loop export initialized from observed p(71)',
                     'other_fields': 'Unchanged stored model outputs; no physical-rate interpretation',
                     'optional_fields': 'A field missing in any fold is None; missing_folds identifies those folds',
                     'inference_or_training': False}}


def load_frozen(panel, labels=DEFAULT_LABELS):
    """Load the three frozen D seed-0 labels, or a subset, without inference.

    Returns models[label][field], all arrays aligned to the input panel and 144
    forecast hours. P is mandatory. Optional fields not present in all five folds
    are None with explicit missing_folds. Input panel comes from d04_data.load_panel.
    DONE identity, exact split indices, integer exactly-once coverage, finite
    arrays and bounded P/P_closed are checked with exceptions, not assertions.
    """
    _layout(panel, expected_units=8457)
    return _load_frozen(panel, labels, RUNS, json.loads(SPLITS.read_text()))


def _load_host_extra(panel, path):
    n = _layout(panel)
    names_expected = WEATHER_NAMES + ['clock_sin', 'clock_cos'] + EXTRA_NAMES
    with np.load(path, allow_pickle=False) as archive:
        names = archive['damage_features'].astype(str).tolist()
        if names != names_expected:
            raise ValueError('Stored xu names/order must be the original 42 host features')
        for stored, panel_key in (('system', 'system'), ('fips', 'county'), ('origin', 'origin')):
            if not np.array_equal(archive[stored].astype(str), np.asarray(panel['meta'][panel_key]).astype(str)):
                raise ValueError(f'Stored feature identity disagrees with panel: {stored}')
    columns = np.arange(14, 42, dtype=np.int16)
    with zipfile.ZipFile(path) as archive:
        if archive.namelist().count('xu.npy') != 1:
            raise ValueError('Expected exactly one stored xu array')
        with archive.open('xu.npy') as stream:
            version = np.lib.format.read_magic(stream)
            shape, fortran, dtype = np.lib.format._read_array_header(stream, version)
            if shape != (n, 216, 42) or fortran or dtype.kind != 'f':
                raise ValueError('Stored xu must be a C-order floating [unit,216,42] array')
        values = _stream_columns(archive, 'xu', columns.tolist(), n)
    if values.shape != (n, 216, 28) or not np.isfinite(values).all():
        raise ValueError('Nonfinite or malformed stored host extra inputs')
    return {'values': values, 'feature_names': names[14:42], 'columns': columns,
            'time': np.arange(216, dtype=np.int16),
            'meta': {'units': n, 'source': 'data/interim/panel_v1/features_v1D.npz',
                     'member': 'xu', 'column_slice': [14, 42],
                     'row_identity': 'Verified system, county and origin equal panel.meta',
                     'absolute_time': 'panel.meta.origin[unit] + (t-72) hours, UTC',
                     'input_semantics': 'Existing raw host-derived weather coordinates, unchanged; units vary by feature',
                     'information_boundary': 'Stored coordinate at t may use target-hour weather and trailing hours <=t',
                     'computation': 'Read existing columns only; no features rebuilt or rescaled',
                     'semantic_warning': SEMANTIC_WARNING.copy()}}


def load_host_extra(panel):
    """Stream stored xu[14:42] into [unit,216,28], without loading full xu/xr.

    Also returns feature_names, original columns/time, and identity/time metadata.
    The compressed member is consumed one unit at a time by d04_data's existing
    reader; only the 28 requested coordinates are retained in memory.
    """
    _layout(panel, expected_units=8457)
    return _load_host_extra(panel, FEATURES)


def _audit_extra_semantics(panel, extra, panels):
    n = _layout(panel)
    values = np.asarray(extra['values'])
    if values.shape != (n, 216, 28) or list(extra['feature_names']) != EXTRA_NAMES:
        raise ValueError('Audit requires unmodified load_host_extra output')
    # Fixed tolerance allows stored float32 sums; exact equality is also counted.
    rtol, atol = 1e-6, 1e-7
    cases = [(name, j, column, j) for j, (name, column) in enumerate(
        zip(EXTRA_NAMES[:5], [0, 1, 3, 2, 4]))]
    cases.append(('snow_ice_load_sum72', EXTRA_NAMES.index('snow_ice_load_sum72'), 3, 2))
    stats = {name: {mapping: {'values': 0, 'matching_values': 0, 'exact_values': 0,
                              'max_abs_difference': 0.}
                    for mapping in ('source_expected', 'swapped_candidate')}
             for name, _, _, _ in cases}
    missing, audited_systems, audited_units, distinguishable = [], 0, 0, 0
    metadata = panel['meta']
    for system in np.unique(metadata['system']):
        system = str(system)
        if re.fullmatch(r'[A-Za-z0-9_-]+', system) is None:
            raise ValueError('Unsafe system identity in panel metadata')
        ids = np.flatnonzero(np.asarray(metadata['system']) == system)
        path = panels / f'panel216w_{system}.npz'
        if not path.exists():
            missing.append(system)
            continue
        with np.load(path, allow_pickle=False) as archive:
            county = archive['fips'].astype(str)
            if (county.ndim != 1 or len(np.unique(county)) != len(county)
                    or set(county) != set(np.asarray(metadata['county'])[ids])):
                raise ValueError(f'{system}: cached counties do not equal the D system counties')
            if str(archive['event'].item()) != system:
                raise ValueError(f'{system}: cached event identity disagrees')
            ts = archive['ts'].astype('datetime64[s]')
            origin = np.datetime64(metadata['origin'][ids[0]], 's')
            expected_ts = origin + (np.arange(216)-72).astype('timedelta64[h]')
            if not np.array_equal(ts, expected_ts):
                raise ValueError(f'{system}: cached timestamps disagree with panel.meta.origin')
        with zipfile.ZipFile(path) as archive:
            if archive.namelist().count('Hg.npy') != 1:
                raise ValueError(f'{system}: expected exactly one Hg array')
            with archive.open('Hg.npy') as stream:
                version = np.lib.format.read_magic(stream)
                shape, fortran, dtype = np.lib.format._read_array_header(stream, version)
                if shape != (len(county), 216, 5) or fortran or dtype.kind != 'f':
                    raise ValueError(f'{system}: unexpected cached Hg layout')
            hg = _stream_columns(archive, 'Hg', list(range(5)), len(county))
        where = {fips: j for j, fips in enumerate(county)}
        hg = hg[[where[str(fips)] for fips in np.asarray(metadata['county'])[ids]]]
        if not np.isfinite(hg).all() or not np.isfinite(values[ids]).all():
            raise ValueError(f'{system}: nonfinite values in semantic audit')
        distinguishable += int((~np.isclose(hg[:, :, 2], hg[:, :, 3], rtol=rtol, atol=atol)).sum())
        for name, extra_col, correct_col, swapped_col in cases:
            actual = values[ids, :, extra_col].astype(np.float64)
            for mapping, column in (('source_expected', correct_col), ('swapped_candidate', swapped_col)):
                target = hg[:, :, column].astype(np.float64)
                if name == 'snow_ice_load_sum72':
                    target = np.stack([target[:, max(0, t-71):t+1].sum(axis=1)
                                       for t in range(216)], axis=1).astype(np.float32).astype(np.float64)
                item = stats[name][mapping]
                item['values'] += actual.size
                item['matching_values'] += int(np.isclose(actual, target, rtol=rtol, atol=atol).sum())
                item['exact_values'] += int((actual == target).sum())
                item['max_abs_difference'] = max(item['max_abs_difference'], float(np.max(np.abs(actual-target))))
        audited_systems += 1
        audited_units += len(ids)
    complete = audited_units == n
    for comparisons in stats.values():
        for item in comparisons.values():
            item['audited_values_all_match'] = (item['matching_values'] == item['values']) if item['values'] else None
    matched = {mapping: (all(stats[name][mapping]['audited_values_all_match'] for name in stats)
                         if audited_units else None)
               for mapping in ('source_expected', 'swapped_candidate')}
    if not complete:
        status = 'incomplete_cache_coverage'
    elif matched['source_expected'] and matched['swapped_candidate']:
        status = 'both_mappings_match_not_distinguishable'
    elif matched['source_expected']:
        status = 'cache_matches_current_source_mapping'
    elif matched['swapped_candidate']:
        status = 'cache_matches_swapped_candidate_mapping'
    else:
        status = 'cache_matches_neither_mapping'
    return {'status': status, 'units_expected': n, 'units_audited': audited_units,
            'systems_audited': audited_systems, 'missing_systems': missing,
            'complete_cache_coverage': complete,
            'cache_matches_source_mapping': matched['source_expected'] if complete else None,
            'cache_matches_swapped_mapping': matched['swapped_candidate'] if complete else None,
            'audited_subset_matches': matched, 'Hg2_Hg3_distinguishable_values': distinguishable,
            'comparisons': stats, 'rtol': rtol, 'atol': atol,
            'identity_and_timestamps_verified_for_audited_systems': True,
            'physical_definition_verified': False, 'mechanism_identified': False,
            'source_pattern': 'data/interim/panel_v1/panels/panel216w_{D_system}.npz',
            'read_fields': ['event', 'fips', 'ts', 'Hg'],
            'semantic_warning': SEMANTIC_WARNING.copy()}


def audit_extra_semantics(panel, extra):
    """Compare stored host labels against D-only county Hg caches, without grids.

    Reads only systems listed in the supplied D panel, one cache at a time, with
    county identity and origin-derived timestamps checked before Hg comparison.
    Tests both the current-source name mapping and a reversed near-freeze/snow
    mapping. Even exact agreement verifies positions, not physical mechanisms.
    Missing caches produce incomplete coverage, never a successful full audit.
    """
    _layout(panel, expected_units=8457)
    return _audit_extra_semantics(panel, extra, PANELS)


def self_test():
    """Synthetic shuffled indices, malformed coverage, optional fields and stream."""
    n = 10
    fold = np.tile(np.arange(1, 6), 2)
    panel = {'fold': fold, 'meta': {'system': np.array([f'S{x}' for x in range(n)]),
                                  'county': np.array([f'{x:05d}' for x in range(n)]),
                                  'origin': np.full(n, '2020-01-01 12:00:00')}}
    split = {'n_units': n, 'event': {str(k): {'outer': np.flatnonzero(fold == k).tolist(),
                                            'dev': np.flatnonzero(fold != k).tolist()}
                                       for k in range(1, 6)}}
    checks = 0

    def reject(call, text):
        nonlocal checks
        try:
            call()
        except (ValueError, FileNotFoundError):
            checks += 1
        else:
            raise AssertionError(f'Invalid synthetic input accepted: {text}')

    with tempfile.TemporaryDirectory(prefix='d05_frozen_test_') as tmp:
        run = Path(tmp)
        label = DEFAULT_LABELS[0]
        expected = np.arange(n*144, dtype=np.float32).reshape(n, 144) / (n*144)
        for k in range(1, 6):
            folder = run / label / f'fold{k:02d}'
            folder.mkdir(parents=True)
            idx = np.flatnonzero(fold == k)[::-1]
            np.savez_compressed(folder / 'outer.npz', idx=idx, P=expected[idx],
                                u=expected[idx], raw_logit=-expected[idx])
            (folder / 'DONE.json').write_text(json.dumps(dict(
                label=label, arm=LABEL_ARMS[label], data='v1D', design='event',
                design_weights=True, fold=k, seed=0, steps=900)))
        load = lambda: _load_frozen(panel, [label], run, split)
        result = load()
        assert np.array_equal(result['models'][label]['P'], expected)
        assert result['models'][label]['r'] is None
        assert result['models'][label]['missing_folds']['r'] == [1, 2, 3, 4, 5]
        assert np.array_equal(result['time'], np.arange(72, 216))
        checks += 4
        file = run / label / 'fold01/outer.npz'
        original = file.read_bytes()
        idx = np.flatnonzero(fold == 1)
        for bad_idx, bad_p, name in [
            (np.repeat(idx[:1], 2), expected[idx], 'duplicate'),
            (idx[:1], expected[idx[:1]], 'missing'),
            (np.flatnonzero(fold == 2), expected[idx], 'wrong fold'),
            (idx, expected[idx, :143], 'wrong column count'),
            (idx, np.full((2, 144), np.nan), 'nonfinite'),
            (idx, np.full((2, 144), 1.1), 'out of range'),
        ]:
            np.savez_compressed(file, idx=bad_idx, P=bad_p)
            reject(load, name)
        np.savez_compressed(file, idx=idx, P=expected[idx])
        partial = load()['models'][label]
        assert partial['u'] is None and partial['missing_folds']['u'] == [1]
        checks += 1
        file.write_bytes(original)
        done = file.with_name('DONE.json')
        saved = done.read_text()
        wrong = json.loads(saved); wrong['data'] = 'unexpected'
        done.write_text(json.dumps(wrong))
        reject(load, 'metadata mismatch')
        done.write_text(saved)
        file.unlink()
        reject(load, 'absent fold export')
        file.write_bytes(original)
        reject(lambda: _load_frozen(panel, [label, label], run, split), 'duplicate labels')
        malformed = json.loads(json.dumps(split)); malformed['event']['1']['outer'] = [0]
        reject(lambda: _load_frozen(panel, [label], run, malformed), 'invalid split coverage')
        path = run / 'synthetic_features.npz'
        xu = np.arange(n*216*42, dtype=np.float32).reshape(n, 216, 42)
        extras = dict(damage_features=np.array(WEATHER_NAMES + ['clock_sin', 'clock_cos'] + EXTRA_NAMES),
                      system=panel['meta']['system'], fips=panel['meta']['county'],
                      origin=panel['meta']['origin'])
        np.savez_compressed(path, xu=xu, **extras)
        host = _load_host_extra(panel, path)
        assert np.array_equal(host['values'], xu[:, :, 14:42])
        assert host['feature_names'] == EXTRA_NAMES
        checks += 2
        for bad_xu in (xu[:, :, :41], xu[:, :215], np.asfortranarray(xu),
                       np.full_like(xu, np.nan)):
            np.savez_compressed(path, xu=bad_xu, **extras)
            reject(lambda: _load_host_extra(panel, path), 'malformed stored xu')
        bad = dict(extras); bad['damage_features'] = extras['damage_features'][::-1]
        np.savez_compressed(path, xu=xu, **bad)
        reject(lambda: _load_host_extra(panel, path), 'feature order')
        bad = dict(extras); bad['fips'] = extras['fips'][::-1]
        np.savez_compressed(path, xu=xu, **bad)
        reject(lambda: _load_host_extra(panel, path), 'county order')
        audit_panel = {**panel, 'meta': {**panel['meta'],
                       'system': np.array([f'S{x//2}' for x in range(n)])}}
        hg = np.arange(n*216*5, dtype=np.float32).reshape(n, 216, 5) / 1000
        ts = np.datetime64('2020-01-01T12:00:00') + (np.arange(216)-72).astype('timedelta64[h]')
        for system in np.unique(audit_panel['meta']['system']):
            ids = np.flatnonzero(audit_panel['meta']['system'] == system)[::-1]
            np.savez_compressed(run / f'panel216w_{system}.npz',
                                event=system, fips=panel['meta']['county'][ids],
                                ts=ts.astype(str), Hg=hg[ids])

        def fake_extra(swap=False):
            result = np.zeros((n, 216, 28), dtype=np.float32)
            result[:, :, :5] = hg[:, :, [0, 1, 2, 3, 4] if swap else [0, 1, 3, 2, 4]]
            column = 2 if swap else 3
            result[:, :, EXTRA_NAMES.index('snow_ice_load_sum72')] = np.stack([
                hg[:, max(0, t-71):t+1, column].astype(np.float64).sum(axis=1)
                for t in range(216)], axis=1)
            return {'values': result, 'feature_names': EXTRA_NAMES}

        right, swapped = fake_extra(), fake_extra(True)
        audit = _audit_extra_semantics(audit_panel, right, run)
        assert audit['cache_matches_source_mapping'] is True
        assert audit['cache_matches_swapped_mapping'] is False
        assert audit['Hg2_Hg3_distinguishable_values'] == n*216
        assert audit['physical_definition_verified'] is False
        backwards = _audit_extra_semantics(audit_panel, swapped, run)
        assert backwards['status'] == 'cache_matches_swapped_candidate_mapping'
        broken = fake_extra(); broken['values'][:, :, 0] += 1
        assert _audit_extra_semantics(audit_panel, broken, run)['status'] == 'cache_matches_neither_mapping'
        checks += 6
        cache = run / 'panel216w_S0.npz'
        original_cache = cache.read_bytes()
        cache.unlink()
        missing = _audit_extra_semantics(audit_panel, right, run)
        assert missing['cache_matches_source_mapping'] is None and missing['units_audited'] == n-2
        checks += 1
        np.savez_compressed(cache, event='S0', fips=panel['meta']['county'][:2],
                            ts=(ts+np.timedelta64(1, 'h')).astype(str), Hg=hg[:2])
        reject(lambda: _audit_extra_semantics(audit_panel, right, run), 'cache time shift')
        cache.write_bytes(original_cache)
        # An equal-column cache must not be called evidence distinguishing labels.
        hg[:, :, 3] = hg[:, :, 2]
        for system in np.unique(audit_panel['meta']['system']):
            ids = np.flatnonzero(audit_panel['meta']['system'] == system)
            np.savez_compressed(run / f'panel216w_{system}.npz', event=system,
                                fips=panel['meta']['county'][ids], ts=ts.astype(str), Hg=hg[ids])
        ambiguous = _audit_extra_semantics(audit_panel, fake_extra(), run)
        assert ambiguous['status'] == 'both_mappings_match_not_distinguishable'
        assert ambiguous['Hg2_Hg3_distinguishable_values'] == 0
        checks += 2
    print(f'D05 frozen-input synthetic checks passed: {checks}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if not args.self_test:
        parser.error('Import load_frozen/load_host_extra from the D05 analysis; CLI is self-test only')
    self_test()
