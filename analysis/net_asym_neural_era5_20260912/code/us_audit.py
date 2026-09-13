#!/usr/bin/env python3
"""Byte-level audit of the 26 EAGLE-I panels and matched ERA5 driver files.

Opens every binary. A prior manifest PASS is not accepted as this round's PASS.
Writes a per-event audit, the inventory rows for these files, and the
input-legality summary used by the window builder.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np, pandas as pd

RAW_CHANNELS = ['cape', 'cloud', 'gust', 'precip', 'pressure', 'rh', 'snowfall',
                'soil_moisture', 't2m_c', 'u10', 'v10', 'wind_speed']


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(a):
    root = Path(a.root); I = root / 'data/interim'
    man = json.loads((root / 'configs/panel_manifest_g3-all-26.json').read_text())
    events, inv, allzero_fips = [], [], {}
    for e in man['panels']:
        pf, df = I / f'panel_{e}.npz', I / f'drivers_{e}.npz'
        p = np.load(pf, allow_pickle=True); d = np.load(df, allow_pickle=True)
        y, obs, den = p['y'], p['observed'], p['denominator']
        pfips, dfips = p['fips'].astype(str), d['fips'].astype(str)
        pts = pd.to_datetime([str(t) for t in p['ts']]); dts = pd.to_datetime([str(t) for t in d['ts']])
        X = d['X']; ch = [str(c) for c in d['channels']]
        r = {'event': e, 'year': int(e[:4])}
        r['panel_keys'] = sorted(p.files); r['driver_keys'] = sorted(d.files)
        r['y_shape'] = list(y.shape); r['observed_shape'] = list(obs.shape); r['X_shape'] = list(X.shape)
        r['y_dtype'] = str(y.dtype); r['observed_dtype'] = str(obs.dtype); r['X_dtype'] = str(X.dtype)
        r['channels'] = ch; r['channels_match_expected'] = ch == RAW_CHANNELS
        r['n_counties'] = int(len(pfips))
        r['fips_identical_order'] = bool(np.array_equal(pfips, dfips))
        r['fips_same_set'] = set(pfips) == set(dfips)
        r['fips_unique'] = len(set(pfips)) == len(pfips)
        dstep = np.diff(pts.values).astype('timedelta64[m]').astype(int)
        r['panel_ts_len'] = len(pts); r['panel_ts_start_utc'] = str(pts[0]); r['panel_ts_end_utc'] = str(pts[-1])
        r['panel_ts_uniform_15min'] = bool((dstep == 15).all())
        r['panel_starts_on_hour'] = bool(pts[0].minute == 0)
        r['driver_ts_len'] = len(dts)
        r['driver_ts_uniform_hourly'] = bool((np.diff(dts.values).astype('timedelta64[m]').astype(int) == 60).all())
        snap = pts[::4]
        r['driver_ts_equals_panel_hourly_snapshots'] = bool(len(snap) == len(dts) and (snap == dts).all())
        yo = np.where(obs, y, np.nan)
        r['observed_fraction_15min'] = float(obs.mean())
        r['observed_nonfinite_cells'] = int((obs & ~np.isfinite(y)).sum())
        r['unobserved_cells_with_value'] = int((~obs & np.isfinite(y) & (y != 0)).sum())
        r['observed_y_min'] = float(np.nanmin(yo)); r['observed_y_max'] = float(np.nanmax(yo))
        r['observed_y_below0'] = int((obs & (y < 0)).sum()); r['observed_y_above1'] = int((obs & (y > 1)).sum())
        r['observed_y_exact_zero_fraction'] = float(((y == 0) & obs).sum() / max(obs.sum(), 1))
        r['denominator_finite_positive'] = bool(np.isfinite(den).all() and (den > 0).all())
        r['denominator_min'] = float(np.nanmin(den))
        ys, os_ = y[:, ::4], obs[:, ::4]
        r['hourly_snapshot_count'] = int(ys.shape[1]); r['hourly_snapshot_observed_fraction'] = float(os_.mean())
        r['X_nonfinite_values'] = int((~np.isfinite(X)).sum())
        zero_cell = (X == 0).all(axis=2)                     # (C, H): all 12 channels exactly 0
        allzero = zero_cell.all(axis=1)
        partial = zero_cell.any(axis=1) & ~allzero
        r['weather_all_zero_counties'] = int(allzero.sum())
        r['weather_all_zero_fraction'] = float(allzero.mean())
        r['weather_partial_zero_hour_counties'] = int(partial.sum())
        ti, pi_ = ch.index('t2m_c'), ch.index('pressure')
        sentinel = ((X[:, :, ti] == 0).all(axis=1) & (X[:, :, pi_] == 0).all(axis=1))
        r['temp_and_pressure_identically_zero_counties'] = int(sentinel.sum())
        r['sentinel_equals_allzero'] = bool(np.array_equal(sentinel, allzero))
        good = ~allzero
        if good.any():
            r['channel_ranges_covered_counties'] = {
                c: [float(X[good, :, k].min()), float(X[good, :, k].mean()), float(X[good, :, k].max())]
                for k, c in enumerate(ch)}
        for f in pfips[allzero]: allzero_fips.setdefault(f, []).append(e)
        events.append(r)
        for path, role in ((pf, 'EAGLE-I outage panel (target y)'), (df, 'ERA5 county drivers (input x)')):
            inv.append({'dataset': role, 'repo_path': str(path.relative_to(root)), 'event': e,
                        'bytes': path.stat().st_size, 'sha256': sha(path)})
    # counties whose status changes across events
    appear = {}
    for r_, e in zip(events, man['panels']):
        pass
    status = {}
    for e in man['panels']:
        p = np.load(I / f'panel_{e}.npz', allow_pickle=True); d = np.load(I / f'drivers_{e}.npz', allow_pickle=True)
        az = (d['X'] == 0).all(axis=(1, 2))
        for f, z in zip(p['fips'].astype(str), az): status.setdefault(f, set()).add(bool(z))
    switch = sum(1 for v in status.values() if len(v) > 1)
    summ = {'events': len(events), 'unique_panel_counties': len(status),
            'counties_all_zero_whenever_present': sum(1 for v in status.values() if v == {True}),
            'counties_nonzero_whenever_present': sum(1 for v in status.values() if v == {False}),
            'counties_switching_status': switch,
            'all_channels_match_expected_order': all(r['channels_match_expected'] for r in events),
            'all_fips_identical_order': all(r['fips_identical_order'] for r in events),
            'all_driver_ts_equal_panel_snapshots': all(r['driver_ts_equals_panel_hourly_snapshots'] for r in events),
            'all_panels_start_on_hour': all(r['panel_starts_on_hour'] for r in events),
            'all_denominators_finite_positive': all(r['denominator_finite_positive'] for r in events),
            'total_observed_nonfinite_cells': sum(r['observed_nonfinite_cells'] for r in events),
            'total_observed_y_below0': sum(r['observed_y_below0'] for r in events),
            'total_observed_y_above1': sum(r['observed_y_above1'] for r in events),
            'total_X_nonfinite': sum(r['X_nonfinite_values'] for r in events),
            'total_partial_zero_hour_counties': sum(r['weather_partial_zero_hour_counties'] for r in events),
            'sentinel_equals_allzero_everywhere': all(r['sentinel_equals_allzero'] for r in events)}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / 'US_DATA_AUDIT.json').write_text(json.dumps({'summary': summ, 'events': events}, indent=1))
    pd.DataFrame(inv).to_csv(Path(a.out) / 'us_file_inventory.csv', index=False)
    pd.DataFrame([{'event': r['event'], 'n_counties': r['n_counties'],
                   'weather_all_zero_counties': r['weather_all_zero_counties'],
                   'weather_all_zero_fraction': r['weather_all_zero_fraction'],
                   'partial_zero_hour_counties': r['weather_partial_zero_hour_counties'],
                   'hourly_snapshot_observed_fraction': r['hourly_snapshot_observed_fraction'],
                   'observed_y_above1': r['observed_y_above1'], 'observed_y_max': r['observed_y_max']}
                  for r in events]).to_csv(Path(a.out) / 'us_weather_coverage_by_event.csv', index=False)
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--root', required=True); ap.add_argument('--out', required=True)
    main(ap.parse_args())
