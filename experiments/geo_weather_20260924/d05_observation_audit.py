"""Read-only checks of the already packaged public-D hourly observations.

The public API takes the output of d04_data.load_panel(). It never calls that
loader itself, reads annual outage records, loads a tranche mask, or downloads
data. Only explicit D event files are opened, after checking the input identities
against features_v1D. --self-test uses temporary synthetic files only.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FEATURES = ROOT / 'data/interim/panel_v1/features_v1D.npz'
PANELS = ROOT / 'data/interim/panel_v1/panels'
HOURS = 216
PREFIX = 72

TIME_AND_OBSERVATION_METADATA = {
    'scope': 'Already packaged public D hourly observations and explicit D event metadata only.',
    'outage_timestamp': (
        'Release run_start_time is documented as the GMT collection-run start; '
        'the existing ingest preserves it as a timezone-naive UTC timestamp.'),
    'hourly_label': (
        'p[T] is the mean of available clipped fractions at UTC T, T+15min, '
        'T+30min and T+45min; it is not an end-of-hour stock.'),
    'observation_mask': (
        'An hourly observed=True requires at least one finite quarter-hour '
        'fraction. The exact number and timing of contributing quarters were not saved.'),
    'source_denominator': 'Fixed EAGLE-I 2024 modeled county customer count in every year.',
    'source_clipping': (
        'attach_denominator clips each quarter-hour count/denominator to [0,1] '
        'before hourly averaging. A saved hourly value of 1 does not identify '
        'an original over-denominator count; a value below 1 does not exclude clipping.'),
    'zero_and_blank_handling': (
        'build_panels.eaglei_window drops explicit zero rows. build_panel starts '
        'counts at zero and assigns retained records, including blank counts. '
        'Missing rows become inferred zeros only when pre-window service and the '
        'collection-run rule allow observation; blank counts remain nonfinite.'),
    'collection_rule_source_boundary': {
        'registered_text': (
            'DATASET_DESIGN amendment 2 M4 says the national collection-run set '
            'counts all rows, including explicit zeros and blanks.'),
        'implemented_path': (
            'build_panels drops explicit zeros and applies its existing mask '
            'before build_panel calls collection_timestamps. That function uses '
            'group size >=5, not distinct-county count.'),
        'empirical_effect': 'Not evaluated here; a source-definition difference is not proof of changed masks.'},
    'quarter_hour_reconstruction': {
        'available': False,
        'reason': (
            'Safe D hourly artifacts omit quarter-hour counts, contributing '
            'quarter masks and original over-denominator flags. This audit '
            'does not read annual raw rows or construct a sealed-tranche mask.'),
        'not_inferred_from': 'D event identity, an hourly observation mask, or a hourly zero/one value.'},
    'weather_timestamp': {
        'instantaneous': 'ERA5 instantaneous channels refer to their UTC valid time, not an hourly temporal mean.',
        'accumulated_and_maximum': (
            'ERA5 HRES total precipitation, snowfall and gust since previous '
            'post-processing at T describe the preceding hourly interval ending at T.'),
        'transform': (
            'The panel multiplies precipitation/snowfall metres by 1000 without '
            'time differencing. Round 2 replaces instantaneous i10fg with fg10 '
            'and aligns exact hourly timestamps; missing weather raises an error.'),
        'd04_latest_weather': (
            'For target p[T]-p[T-1], D04 uses weather through T-1: its latest '
            'precipitation/gust interval ends at T-1, whereas p[T] includes '
            'collection starts through T+45min. This is the frozen strict-past '
            'design, not an established implementation error.'),
        'official_reference': 'https://confluence.ecmwf.int/spaces/CKB/pages/85402030/',
        'clock_controls': (
            'Original panel X clock channels use UTC hour. D04 nuisance sin/cos '
            'uses window index t modulo 24; it is an event-relative periodic control.')},
    'weather_spatial_and_archive_boundary': (
        'The v1 builder passes area weights into build_v3p.panel. Its Hg and S '
        'include cell-level hazards/support, which D04 raw-12 weather excludes. '
        'The separate cropped ERA5 archive stores most fields as float16 and '
        'contains both i10fg and fg10; it is not read by this audit.'),
    'interpretation': (
        'Hourly missingness, exact zeros/ones and adjacent changes are observable '
        'shape flags, not adjudications of reporting errors, physical damage or recovery.'),
}

SOURCE_FILES = (
    'experiments/geo_weather_20260924/d04_data.py',
    'experiments/geo_weather_20260924/d04_features.py',
    'experiments/geo_weather_20260924/panel_v1/build_panels.py',
    'experiments/geo_weather_20260924/panel_v1/features_v1.py',
    'experiments/geo_weather_20260924/panel_v1/draw.py',
    'experiments/geo_weather_20260924/build_v3p.py',
    'experiments/geo_weather_20260924/fetch_arco_windows.py',
    'experiments/open_gcrk_20260919/build_panel216.py',
    'experiments/open_gcrk_20260919/build_panel216_r2.py',
    'src/asymode/panel.py',
    'src/asymode/weather.py',
    'experiments/geo_weather_20260924/DATASET_DESIGN.md',
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _hour_counts(y, obs):
    """Unweighted counts; no inferred sub-hour coverage and no imputation."""
    out = {'county_events': int(len(y))}
    for label, sl in [('all', slice(None)), ('prefix', slice(0, PREFIX)),
                      ('forecast', slice(PREFIX, HOURS))]:
        a, m = y[:, sl], obs[:, sl]
        out[label] = {
            'hours': int(m.size),
            'observed_hours': int(m.sum()),
            'missing_hours': int((~m).sum()),
            'unobserved_but_finite_hours': int((~m & np.isfinite(a)).sum()),
            'exact_zero_observed_hours': int((m & (a == 0)).sum()),
            'exact_one_observed_hours': int((m & (a == 1)).sum()),
            'county_events_with_missing_hours': int((~m).any(1).sum()),
            'county_events_with_exact_one': int((m & (a == 1)).any(1).sum()),
        }
    valid = obs[:, PREFIX:] & obs[:, PREFIX-1:-1]
    previous, current = y[:, PREFIX-1:-1], y[:, PREFIX:]
    delta = current - previous
    out['forecast_adjacent_differences'] = {
        'candidate_pairs': int(valid.size),
        'valid_pairs': int(valid.sum()),
        'excluded_pairs': int((~valid).sum()),
        'positive_pairs': int((valid & (delta > 0)).sum()),
        'negative_pairs': int((valid & (delta < 0)).sum()),
        'zero_pairs': int((valid & (delta == 0)).sum()),
        'positive_stock_to_exact_zero_pairs': int((valid & (previous > 0) & (current == 0)).sum()),
        'exact_zero_to_positive_stock_pairs': int((valid & (previous == 0) & (current > 0)).sum()),
        'pairs_with_either_hour_exact_one': int((valid & ((previous == 1) | (current == 1))).sum()),
    }
    return out


def _audit(panel, features_path, panel_dir, require_complete_d):
    """Private path-injected implementation, so tests never need real data."""
    meta = panel['meta']
    system = np.asarray(meta['system']).astype(str)
    county = np.asarray(meta['county']).astype(str)
    origin = np.asarray(meta['origin']).astype('datetime64[ns]')
    y, obs = np.asarray(panel['y_full']), np.asarray(panel['obs_full'])
    n = len(system)
    _require(system.shape == county.shape == origin.shape == (n,), 'Invalid D identity shapes')
    _require(y.shape == obs.shape == (n, HOURS), 'Expected [county-event,216] observations')
    _require(obs.dtype.kind == 'b', 'Observation mask must be boolean')
    _require(np.isfinite(y[obs]).all(), 'Nonfinite observed hourly stock')
    _require(np.all((y[obs] >= 0) & (y[obs] <= 1)), 'Observed stock outside [0,1]')
    _require(not np.isnat(origin).any(), 'Missing forecast origin')
    _require(np.all(origin == origin.astype('datetime64[h]').astype('datetime64[ns]')),
             'Forecast origins must be on the hourly UTC grid')
    systems = sorted(set(system))
    _require(all(re.fullmatch(r'S\d{5}', s) for s in systems), 'Unsafe or unrecognized system identifier')
    _require(all(re.fullmatch(r'\d{5}', f) for f in county), 'Invalid county FIPS')
    _require(len(set(zip(system, county))) == n, 'Duplicate county-event identities')
    if require_complete_d:
        _require(n == 8457 and len(systems) == 81, 'Expected the complete frozen public D panel')

    # Small metadata members only: no xu/xr, new target reconstruction, or other tranche.
    _require(features_path.name == 'features_v1D.npz', 'Only the explicit public D feature archive is allowed')
    with np.load(features_path, allow_pickle=False) as z:
        fs, fc = z['system'].astype(str), z['fips'].astype(str)
        fo = z['origin'].astype('datetime64[ns]')
        cust = np.asarray(z['cust']).copy()
        _require(np.array_equal(fs, system) and np.array_equal(fc, county)
                 and np.array_equal(fo, origin), 'Input identities/order differ from features_v1D')
        _require(np.array_equal(z['event'].astype(str), fs), 'Feature event/system mismatch')
    _require(cust.shape == (n,) and np.isfinite(cust).all() and (cust > 0).all(),
             'Invalid feature denominators')

    reports = []
    seen = np.zeros(n, dtype=np.int8)
    denominator_by_county = {}
    max_denom_roundoff = 0.0
    for s in systems:
        idx = np.flatnonzero(system == s)
        _require(len(np.unique(origin[idx])) == 1, f'Inconsistent origin within D system {s}')
        path = panel_dir / f'panel216w_{s}.npz'  # no glob or tranche-wide table read
        with np.load(path, allow_pickle=False) as z:
            expected_keys = {'event', 'fips', 'ts', 'y', 'observed', 'denominator'}
            _require(expected_keys.issubset(z.files), f'Missing event-panel keys for {s}')
            event = np.asarray(z['event'])
            _require(event.ndim == 0 and str(event.item()) == s, f'Event filename/content mismatch for {s}')
            ids = z['fips'].astype(str)
            _require(ids.ndim == 1 and len(set(ids)) == len(ids), f'Duplicate/invalid event FIPS for {s}')
            _require(set(ids) == set(county[idx]), f'Event FIPS differ from public D identities for {s}')
            pos = {f: j for j, f in enumerate(ids)}
            order = np.asarray([pos[f] for f in county[idx]], dtype=int)
            ts = z['ts'].astype('datetime64[ns]')
            t0 = origin[idx[0]] - np.timedelta64(PREFIX, 'h')
            expected_ts = t0 + np.arange(HOURS) * np.timedelta64(1, 'h')
            _require(ts.shape == (HOURS,) and np.array_equal(ts, expected_ts),
                     f'Hourly timestamps disagree with D forecast origin for {s}')
            py, pm, pd = z['y'], z['observed'], z['denominator']
            _require(py.shape == pm.shape == (len(ids), HOURS) and pm.dtype.kind == 'b',
                     f'Unexpected event observation shape/mask for {s}')
            _require(pd.shape == (len(ids),) and np.isfinite(pd).all() and (pd > 0).all(),
                     f'Invalid event denominators for {s}')
            _require(np.array_equal(py[order], y[idx], equal_nan=True), f'Hourly stock mismatch for {s}')
            _require(np.array_equal(pm[order], obs[idx]), f'Hourly observation-mask mismatch for {s}')
            d = pd[order].astype('f8')
            # The feature builder stores cust as float32; the event panels retain
            # the original denominator precision. Compare at the recorded feature precision.
            _require(np.array_equal(d.astype(cust.dtype), cust[idx]), f'Feature denominator mismatch for {s}')
            max_denom_roundoff = max(max_denom_roundoff, float(np.max(np.abs(d-cust[idx].astype('f8')))))
            for f, value in zip(county[idx], d):
                if f in denominator_by_county:
                    _require(denominator_by_county[f] == float(value), f'Denominator changes across D systems for {f}')
                denominator_by_county[f] = float(value)
            reports.append({
                'system': s,
                'event_file_order_matches_features': bool(np.array_equal(ids, county[idx])),
                'first_hour_utc': str(ts[0].astype('datetime64[s]')),
                'last_hour_utc': str(ts[-1].astype('datetime64[s]')),
                'forecast_origin_utc': str(origin[idx[0]].astype('datetime64[s]')),
                'denominator_min': float(d.min()), 'denominator_max': float(d.max()),
                'counts': _hour_counts(y[idx], obs[idx]),
            })
        seen[idx] += 1
    _require(np.all(seen == 1), 'Every feature unit must match exactly one explicit D event file')
    by_regime = {}
    if 'regime' in meta:
        regime = np.asarray(meta['regime']).astype(str)
        _require(regime.shape == (n,), 'Invalid regime metadata shape')
        by_regime = {r: _hour_counts(y[regime == r], obs[regime == r]) for r in sorted(set(regime))}
    return {
        'analysis': 'D05 packaged-D hourly observation/source audit',
        'metadata': copy.deepcopy(TIME_AND_OBSERVATION_METADATA),
        'validation': {
            'explicit_d_feature_identity_match': True,
            'event_fips_set_match': True, 'event_timestamp_match': True,
            'event_y_match_including_nan': True, 'event_observed_match': True,
            'denominator_match_at_feature_precision': True,
            'repeated_county_denominator_constant': True,
            'unit_event_file_coverage_exactly_once': True,
            'systems': len(systems), 'county_events': n, 'counties': len(set(county)),
            'maximum_denominator_feature_precision_difference': max_denom_roundoff,
            'denominator_comparison': 'Exact equality after casting event denominator to stored feature cust dtype.',
            'raw_quarter_hour_records_read': False, 'sealed_mask_loaded': False,
            'full_event_file_checksums_recomputed': False,
        },
        'count_definition': 'Unweighted retained county-event/hour counts; no design-weighted rates or fitting.',
        'all': _hour_counts(y, obs), 'by_regime': by_regime, 'by_system': reports,
    }


def audit_d_panels(panel):
    """Validate the complete d04_data.load_panel() result against explicit D files.

    Returns a JSON-serializable compact dictionary. No output is written and no
    arrays for weather, other tranches, or sub-hour raw observations are loaded.
    Any identity, time, value, mask or denominator mismatch raises ValueError.
    Call this from the owning process with its existing nice/thread limits.
    """
    result = _audit(panel, FEATURES, PANELS, require_complete_d=True)
    paths = (*SOURCE_FILES, str(Path(__file__).resolve().relative_to(ROOT)))
    result['source_sha256'] = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
    return result


def _self_test():
    with tempfile.TemporaryDirectory(prefix='d05-observation-test-') as td:
        root = Path(td)
        features = root/'features_v1D.npz'
        panels = root/'panels'
        panels.mkdir()
        system = np.array(['S00001', 'S00001', 'S00002'])
        county = np.array(['01001', '01003', '01001'])
        origin = np.array(['2020-01-04T03:00:00', '2020-01-04T03:00:00', '2021-06-04T17:00:00'])
        y = np.zeros((3, HOURS), dtype='f4')
        obs = np.ones_like(y, dtype=bool)
        y[0, 73] = np.nan; obs[0, 73] = False
        y[1, 85] = 1.0
        y[2, 72:74] = .25
        raw_cust = np.array([10000000.25, 12345.5, 10000000.25], dtype='f8')
        panel = {'y_full': y, 'obs_full': obs,
                 'meta': {'system': system, 'county': county, 'origin': origin,
                          'regime': np.array(['winter', 'winter', 'convective'])}}
        np.savez(features, system=system, event=system, fips=county, origin=origin, cust=raw_cust.astype('f4'))

        def write_event(s, bad=None):
            idx = np.flatnonzero(system == s)[::-1]  # exercise FIPS-based alignment
            ts = (np.datetime64(origin[idx[0]], 'ns') - np.timedelta64(PREFIX, 'h')
                  + np.arange(HOURS)*np.timedelta64(1, 'h')).astype(str)
            payload = dict(event=s, fips=county[idx], ts=ts, y=y[idx].copy(),
                           observed=obs[idx].copy(), denominator=raw_cust[idx].copy())
            if bad == 'time': payload['ts'] = (payload['ts'].astype('datetime64[ns]') + np.timedelta64(1, 'h')).astype(str)
            if bad == 'denominator': payload['denominator'][0] += 100
            if bad == 'value': payload['y'][0, 80] = .125
            if bad == 'mask': payload['observed'][0, 80] = False
            if bad == 'fips': payload['fips'] = np.array(['99999']*len(idx))
            np.savez(panels/f'panel216w_{s}.npz', **payload)

        for s in sorted(set(system)): write_event(s)
        result = _audit(panel, features, panels, False)
        assert result['all']['all']['missing_hours'] == 1
        assert result['all']['forecast']['exact_one_observed_hours'] == 1
        assert result['all']['forecast_adjacent_differences']['valid_pairs'] == 3*144-2
        assert result['all']['forecast_adjacent_differences']['positive_pairs'] == 2
        assert result['validation']['maximum_denominator_feature_precision_difference'] == .25
        json.dumps(result, allow_nan=False)
        for kind in ['time', 'denominator', 'value', 'mask', 'fips']:
            write_event('S00001', kind)
            try: _audit(panel, features, panels, False)
            except ValueError: pass
            else: raise AssertionError(f'Accepted mismatched {kind}')
            write_event('S00001')
        bad_panel = copy.deepcopy(panel)
        bad_panel['meta']['system'][0] = 'S99999'
        try: _audit(bad_panel, features, panels, False)
        except ValueError: pass
        else: raise AssertionError('Accepted an identity outside the explicit D allowlist')
    return {'synthetic_only': True, 'checks': [
        'FIPS reordering', 'missing adjacent support', 'exact-one counts',
        'denominator feature precision', 'time/value/mask/denominator/FIPS rejection',
        'unknown D identity rejection', 'strict JSON serialization']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if not args.self_test:
        parser.error('No data-loading CLI: import audit_d_panels(panel) from the owning D05 process.')
    print(json.dumps(_self_test(), indent=2))
