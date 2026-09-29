"""Outcome-value-blind D03 geometry audit; no outcome/prediction arrays are read."""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '2'
import argparse
import hashlib
import json
from pathlib import Path
import time
import zipfile

import numpy as np
from scipy.fft import dct
from threadpoolctl import threadpool_limits
import analyze_data_relationships_v1 as D01

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LOOKUP = ROOT / 'runs/geo_weather_20260924/county_structure_d02/county_structure_d02_lookup.npz'
SUPPORT = ROOT / 'runs/geo_weather_20260924/data_first_d01/aligned_exposure_oof.npz'
FEATURES = ROOT / 'data/interim/panel_v1/features_v1D.npz'
SCOPE = HERE / 'notes/D03_HIGH_DIM_STRUCTURE_SCOPE_20260928.md'
OUT = HERE / 'results/v1/highdim_structure_d03.json'
REGIMES = ['all', *D01.REG]
WEIGHTINGS = ['design', 'equal_merged_group']
CHECKS = {'max_parseval_relative_error': 0., 'max_projection_normal_error': 0.,
          'min_covariance_eigenvalue': 0.}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spectrum(cov):
    eig = np.linalg.eigvalsh((cov + cov.T) / 2)[::-1]
    CHECKS['min_covariance_eigenvalue'] = min(CHECKS['min_covariance_eigenvalue'], float(eig[-1]))
    if eig[-1] < -1e-8 * max(1., float(eig[0])):
        raise ValueError('Non-PSD covariance')
    eig = np.maximum(eig, 0.)
    total = eig.sum()
    if total <= 1e-15:
        return {'dimensions': len(eig), 'degenerate': True}
    p = eig / total
    cum = np.cumsum(p)
    return {'dimensions': len(eig), 'degenerate': False,
            'variance_fractions': p.tolist(), 'cumulative_variance_fractions': cum.tolist(),
            'participation_rank': float(1 / np.square(p).sum()),
            'entropy_rank': float(np.exp(-np.sum(p[p > 0] * np.log(p[p > 0])))),
            'dimensions_for_variance': {str(q): int(np.searchsorted(cum, q) + 1) for q in [.8, .9, .95]}}


def concentration(x, w, codes, groups):
    """Event shares of a centered or residual coordinate's weighted energy."""
    energy = np.bincount(codes, weights=w * x * x, minlength=len(groups))
    total = energy.sum()
    if total <= 1e-20:
        return {'variance': float(total), 'informative_groups': 0,
                'effective_groups': None, 'max_group_share': None, 'max_group': None}
    shares = energy / total
    return {'variance': float(total), 'informative_groups': int((shares > 1e-8).sum()),
            'effective_groups': float(1 / np.square(shares).sum()),
            'max_group_share': float(shares.max()), 'max_group': str(groups[shares.argmax()])}


def county_geometry():
    with np.load(LOOKUP, allow_pickle=False) as f:
        names = f['feature_names'].astype(str).tolist()
        standardized = f['feature_z'].astype(float)
        blocks = f['blocks'].astype(str)
        raw = f['features'].astype(float)
        counties = f['county'].astype(str)
    result = {'counties': len(counties), 'feature_names': names, 'geometries': {}}
    for dim in [40, 46]:
        bs = blocks[:dim]
        unique, count = np.unique(bs, return_counts=True)
        sizes = dict(zip(unique, count))
        div = np.array([np.sqrt(sizes[b] * len(unique)) for b in bs])
        z = standardized[:, :dim] / div
        z -= z.mean(0)
        result['geometries'][str(dim)] = {
            'semantic_blocks': {b: int(v) for b, v in sizes.items()},
            'spectrum': spectrum(z.T @ z / len(z))}
    angle = np.arange(8) * np.pi / 4
    aspect = raw[:, [names.index('aspect_' + a) for a in ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']]]
    if not np.allclose(aspect.sum(1), 1, atol=1e-6):
        raise ValueError('Aspect shares do not sum to one')
    first = abs(aspect @ np.exp(1j * angle))
    second = abs(aspect @ np.exp(2j * angle))
    probs = [0, .1, .25, .5, .75, .9, 1]
    result['aspect_harmonics'] = {
        'quantile_probabilities': probs,
        'first_magnitude_quantiles': np.quantile(first, probs).tolist(),
        'second_magnitude_quantiles': np.quantile(second, probs).tolist(),
        'cutoff_counts': {str(c): {'first_at_least': int((first >= c).sum()),
                                  'second_at_least': int((second >= c).sum()),
                                  'first_below_second_at_least': int(((first < c) & (second >= c)).sum())}
                          for c in [.05, .1, .15]},
        'meaning': 'First is directed orientation; second is axial orientation modulo pi. Neither is an outage effect.'}
    return result


def read_weather_stream():
    """Read only the weather member and retain channels 0:12; bounded peak memory."""
    with zipfile.ZipFile(FEATURES) as archive:
        with archive.open('xu.npy') as f:
            version = np.lib.format.read_magic(f)
            shape, fortran, dtype = np.lib.format._read_array_header(f, version)
            if fortran or len(shape) != 3 or shape[1] != 216 or shape[2] < 12 or dtype.hasobject:
                raise ValueError('Unexpected xu layout')
            keep = np.empty((shape[0], shape[1], 12), dtype=np.float64)
            for i in range(shape[0]):
                nbytes = shape[1] * shape[2] * dtype.itemsize
                buf = f.read(nbytes)
                if len(buf) != nbytes:
                    raise ValueError('Truncated xu row')
                keep[i] = np.frombuffer(buf, dtype=dtype).reshape(shape[1:])[:, :12]
            if f.read(1):
                raise ValueError('Trailing xu data')
    return keep


def load_supported_weather():
    with np.load(SUPPORT, allow_pickle=False) as f:
        # NPZ members are lazy: do not read target or prediction members.
        unit = f['unit'].astype(int)
        anchor = f['anchor'].astype(int)
    with np.load(FEATURES, allow_pickle=False) as f:
        meta = {k: f[k] for k in ['system', 'family', 'fips', 'origin', 'regime', 'w']}
        names = f['damage_features'][:12].astype(str).tolist()
    n = len(meta['system'])
    if len(unit) != 50302 or len(np.unique(unit)) != 8457 or n != 8457:
        raise ValueError('Existing support differs from D01')
    if len(np.unique(np.column_stack([unit, anchor]), axis=0)) != len(unit):
        raise ValueError('Duplicate supported window')
    if not set(np.unique(anchor)).issubset(set(D01.ANCHORS)):
        raise ValueError('Unexpected anchor')
    groups = D01.merged_groups(meta['system'], meta['family'], meta['origin'], meta['fips'])
    split = json.loads((HERE / 'splits_v1D.json').read_text())
    fold = np.zeros(n, dtype=int)
    for k in range(1, 6):
        ids = split['event'][str(k)]['outer']
        if np.any(fold[ids]):
            raise ValueError('Repeated outer unit')
        fold[ids] = k
    for group in np.unique(groups):
        if len(np.unique(fold[groups == group])) != 1:
            raise ValueError('Merged group crosses original folds')
    count = np.bincount(unit, minlength=n)
    w = meta['w'][unit] / count[unit]
    raw = read_weather_stream()
    for key in ['cape', 'precip', 'snowfall']:
        j = names.index(key)
        raw[:, :, j] = np.log1p(np.maximum(raw[:, :, j], 0))
    wx = np.empty((len(unit), 24, 12), dtype=np.float64)
    for a in D01.ANCHORS:
        rows = np.flatnonzero(anchor == a)
        wx[rows] = raw[unit[rows], a:a + 24]
    del raw
    if not np.isfinite(wx).all() or not np.all(w > 0):
        raise ValueError('Nonfinite weather or invalid weight')
    audit = {'windows': len(unit), 'county_events': len(np.unique(unit)),
             'counties': len(np.unique(meta['fips'][unit])),
             'systems': len(np.unique(meta['system'][unit])),
             'families': len(np.unique(meta['family'][unit])),
             'merged_groups': len(np.unique(groups)), 'weather_channels': names,
             'anchors': D01.ANCHORS,
             'outage_or_prediction_members_read': [],
             'D01_support_members_read': ['unit', 'anchor'],
             'D_panel_members_read': ['xu', 'damage_features', *meta.keys()]}
    return wx, names, w, groups[unit], meta['regime'][unit], unit, meta['fips'][unit], audit


def weather_summaries(wx, names):
    means = wx.mean(1)
    trends = wx[:, 12:].mean(1) - wx[:, :12].mean(1)
    core = wx[:, :, [names.index(c) for c in D01.WEATHER]]
    compound = np.stack([(core[:, :, i] * core[:, :, j]).mean(1) for i, j in D01.PAIRS], 1)
    _, _, order = D01.moments(core)
    labels = ['mean:' + c for c in names] + ['late_minus_early:' + c for c in names]
    labels += ['compound:' + D01.WEATHER[i] + '*' + D01.WEATHER[j] for i, j in D01.PAIRS]
    labels += ['order:' + D01.WEATHER[i] + '->' + D01.WEATHER[j] for i, j in D01.PAIRS]
    return np.column_stack([means, trends, compound, order]), labels


def summary_audit(x, w, codes, groups):
    centered = x - w @ x
    variance = w @ (centered * centered)
    sd = np.sqrt(variance)
    active = sd > 1e-10
    z = centered / np.where(active, sd, 1.)
    z[:, ~active] = 0.
    corr = z.T @ (w[:, None] * z)
    out = {'active_driver_mask': active.tolist(), 'raw_driver_standard_deviation': sd.tolist(),
           'correlation_spectrum': spectrum(corr),
           'variance_event_concentration': [concentration(centered[:, j], w, codes, groups) for j in range(x.shape[1])],
           'order_linear_projections': {}}
    y = z[:, 39:54]
    for width, label in [(12, 'means'), (24, 'means_and_trends'), (39, 'means_trends_compound')]:
        a = z[:, :width]
        gram = a.T @ (w[:, None] * a)
        eigen, vec = np.linalg.eigh(gram)
        cutoff = max(1e-12, 1e-10 * float(eigen[-1]))
        keep = eigen > cutoff
        rhs = a.T @ (w[:, None] * y)
        coefficient = vec[:, keep] @ ((vec[:, keep].T @ rhs) / eigen[keep, None])
        residual = y - a @ coefficient
        normal_error = float(np.max(np.abs(a.T @ (w[:, None] * residual))))
        CHECKS['max_projection_normal_error'] = max(CHECKS['max_projection_normal_error'], normal_error)
        if normal_error > 1e-7:
            raise ValueError('Projection normal equation failed')
        fractions = w @ (residual * residual)
        if np.any(fractions < -1e-10) or np.any(fractions > 1 + 1e-8):
            raise ValueError('Invalid residual fraction')
        fractions = np.clip(fractions, 0, 1)
        entries = []
        for j in range(15):
            entries.append({'remaining_variance_fraction': float(fractions[j]) if active[39 + j] else None,
                            'event_concentration': concentration(residual[:, j], w, codes, groups)})
        out['order_linear_projections'][label] = {'projector_columns': width,
            'projector_rank': int(keep.sum()), 'order_residuals': entries}
    return out


def path_audit(wx, w, names):
    channel_mean = np.einsum('n,ntc->c', w, wx) / 24
    channel_second = np.einsum('n,ntc->c', w, wx * wx) / 24
    scale = np.sqrt(np.maximum(channel_second - channel_mean ** 2, 0))
    active = scale > 1e-10
    scale[~active] = 1.
    hour_mean = np.einsum('n,ntc->tc', w, wx)
    centered = (wx - hour_mean[None]) / scale[None, None]
    full = np.einsum('n,ntc,ntc->c', w, centered, centered)
    coeff = dct(centered, type=2, axis=1, norm='ortho', workers=1)
    energy = np.einsum('n,ntc,ntc->tc', w, coeff, coeff)
    err = float(np.max(np.abs(energy.sum(0) - full) / np.maximum(full, 1e-12)))
    CHECKS['max_parseval_relative_error'] = max(CHECKS['max_parseval_relative_error'], err)
    if err > 1e-8:
        raise ValueError('DCT Parseval check failed')
    per_channel = []
    for j in range(12):
        if full[j] <= 1e-15:
            per_channel.append({'name': names[j], 'degenerate': True})
        else:
            per_channel.append({'name': names[j], 'degenerate': False,
                'mean_only_fraction': float(energy[0, j] / full[j]),
                'first8_fraction': float(energy[:8, j].sum() / full[j])})
    total = full.sum()
    z = coeff[:, :8].reshape(len(wx), 96)
    cov = z.T @ (w[:, None] * z)
    fractions = np.cumsum(energy.sum(1)) / total
    if np.any(np.diff(fractions) < -1e-12) or not np.isclose(fractions[-1], 1):
        raise ValueError('Invalid cumulative DCT energy')
    return {'channel_scales': scale.tolist(), 'active_channel_mask': active.tolist(),
            'mean_only_energy_fraction': float(fractions[0]),
            'first8_energy_fraction': float(fractions[7]),
            'all24_cumulative_energy_fractions': fractions.tolist(),
            'per_channel': per_channel,
            'first8_per_channel_96D_covariance_spectrum': spectrum(cov)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Preserve existing D03 result; inspect it before any rerun.')
    os.nice(max(0, 15 - os.getpriority(os.PRIO_PROCESS, 0)))
    started = time.monotonic()
    with threadpool_limits(limits=2):
        geography = county_geometry()
        wx, names, weights, group, reg, unit, county, audit = load_supported_weather()
        x, drivers = weather_summaries(wx, names)
        views = {}
        for regime in REGIMES:
            idx = np.flatnonzero(np.ones(len(wx), dtype=bool) if regime == 'all' else reg == regime)
            labels, codes = np.unique(group[idx], return_inverse=True)
            views[regime] = {}
            for weighting in WEIGHTINGS:
                w = weights[idx].copy()
                if weighting == 'equal_merged_group':
                    totals = np.bincount(codes, weights=w, minlength=len(labels))
                    w /= totals[codes]
                w /= w.sum()
                views[regime][weighting] = {
                    'support': {'windows': len(idx), 'county_events': len(np.unique(unit[idx])),
                                'counties': len(np.unique(county[idx])), 'merged_groups': len(labels)},
                    'summary54': summary_audit(x[idx], w, codes, labels),
                    'path': path_audit(wx[idx], w, names)}
                print(f'{regime} {weighting}: completed', flush=True)
        out = {'meta': {'analysis': 'D03 outcome-value-blind high-dimensional information audit',
                'exploratory': True, 'elapsed_seconds': time.monotonic() - started,
                'resources': {'nice': os.getpriority(os.PRIO_PROCESS, 0), 'numerical_threads': 2},
                'source_hashes': {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), SCOPE, LOOKUP]},
                'support_cache': str(SUPPORT.relative_to(ROOT)), 'D_panel': str(FEATURES.relative_to(ROOT)),
                'driver_names': drivers,
                'interpretation_limits': [
                    'Descriptor/weather rank is not useful outage-response rank or proof of a nonlinear manifold.',
                    'Order residual variance is in-sample linear novelty, not incremental predictive utility.',
                    'DCT residual energy may be irrelevant noise; temporal information loss is not demonstrated outcome loss.',
                    'First and second aspect harmonics do not identify damage direction or transmission topology.',
                    'Shared event and county dependence means the window count is not independent sample size.',
                    'The existing D01 row support was selected using outage observation availability, but no outage values are read.']},
               'audit': audit, 'county_geometry': geography, 'weather_views': views, 'checks': CHECKS}
        payload = json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open('x') as f:
            f.write(payload)
        print(json.dumps({'out': str(args.out.relative_to(ROOT)), 'elapsed_seconds': out['meta']['elapsed_seconds'],
                          'checks': CHECKS}), flush=True)


if __name__ == '__main__':
    main()
