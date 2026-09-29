"""D08: freeze a public-D FIT-only, outcome-blind mechanism panel.

The selector reads raw weather, geo40 and identity/design-weight metadata only.
It neither imports a model nor reads targets, observation masks or predictions.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import time
import zipfile

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FEATURES = ROOT / 'data/interim/panel_v1/features_v1D.npz'
SPLITS = HERE / 'splits_v1D.json'
DEFAULT_OUT = ROOT / 'runs/geo_weather_20260924/d08_panel_20260929'
SUMMARY_OUT = HERE / 'results/v1/d08_panel_input.json'
ROSTER_OUT = HERE / 'results/v1/d08_panel_roster.json.gz'
DATA_SHA = 'f043bb39e8abd48183670e2acc3c0cecebd7107ea9be0e28771912cfee358c48'
SEED = 20260929
CAP, CORE, TAIL, ANCHORS, PAIR_SAMPLE = 24, 16, 8, 4, 10000
WEATHER_NAMES = ['cape', 'cloud', 'gust', 'precip', 'pressure', 'rh', 'snowfall',
                 'soil_moisture', 't2m_c', 'u10', 'v10', 'wind_speed']
METADATA_MEMBERS = ('system', 'family', 'origin', 'fips', 'regime', 'w')
READ_MEMBERS = ('damage_features', 'geo_features', *METADATA_MEMBERS, 'xu', 'geo')
SETTINGS = dict(seed=SEED, cap_per_group=CAP, deterministic_core_per_group=CORE,
                uniform_tail_per_large_group=TAIL, anchors_per_group=ANCHORS,
                cutoff_pair_sample=PAIR_SAMPLE, near_percentile=10,
                far_percentile=75, weather_hours=216, weather_channels=12,
                geography_channels=40, outer_fold=1)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False,
                       separators=(',', ':'), allow_nan=False) + '\n').encode()


def write_json_new(path, value):
    with Path(path).open('xb') as f:
        f.write(json_bytes(value))


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def verify_registration(commit, scope):
    """Read-only Git check: registered scope and selector must match working files."""
    try:
        full = subprocess.check_output(['git', 'rev-parse', '--verify', commit + '^{commit}'],
                                       cwd=ROOT, stderr=subprocess.DEVNULL, text=True).strip()
        for path in (Path(__file__), Path(scope)):
            blob = subprocess.check_output(['git', 'show', full + ':' + relative(path)],
                                           cwd=ROOT, stderr=subprocess.DEVNULL)
            if hashlib.sha256(blob).hexdigest() != sha(path):
                raise ValueError('Registered source/scope differs from current file')
    except subprocess.CalledProcessError:
        raise ValueError('Registration commit/source/scope is not available') from None
    return full


def merged_groups(meta):
    """Original family/shared-county within-16-day connected event groups."""
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
                raise ValueError(f'Inconsistent {field} within system')
        info[s] = (meta['family'][rows[0]], np.datetime64(meta['origin'][rows[0]]),
                   set(meta['fips'][rows]))
    for j, a in enumerate(names):
        fa, oa, ca = info[a]
        for b in names[j + 1:]:
            fb, ob, cb = info[b]
            if fa == fb or (abs(oa - ob) <= np.timedelta64(16, 'D') and ca & cb):
                union(a, b)
    return np.asarray([find(s) for s in system])


def read_fit_member(archive, member, fit, n, expected_tail, columns=None):
    """Stream bytes, interpreting and retaining FIT rows only; no OUTER values."""
    fit = np.asarray(fit, dtype=int)
    if len(np.unique(fit)) != len(fit) or not np.array_equal(fit, np.sort(fit)):
        raise ValueError('FIT indices must be unique and sorted')
    with archive.open(member + '.npy') as stream:
        version = np.lib.format.read_magic(stream)
        shape, fortran, dtype = np.lib.format._read_array_header(stream, version)
        if (fortran or dtype.hasobject or shape[0] != n or
                tuple(shape[1:-1]) != tuple(expected_tail[:-1]) or
                shape[-1] < expected_tail[-1]):
            raise ValueError(f'Unexpected {member} layout')
        tail = shape[1:] if columns is None else (*shape[1:-1], len(columns))
        out = np.empty((len(fit), *tail), dtype=np.float32)
        nbytes = int(np.prod(shape[1:])) * dtype.itemsize
        row = 0
        for unit in range(n):
            buf = stream.read(nbytes)
            if len(buf) != nbytes:
                raise ValueError(f'Truncated {member} row')
            if row < len(fit) and unit == fit[row]:
                block = np.frombuffer(buf, dtype=dtype).reshape(shape[1:])
                out[row] = block if columns is None else block[..., columns]
                row += 1
        if row != len(fit) or stream.read(1):
            raise ValueError(f'Invalid {member} row count/trailing data')
    return out


def load_inputs():
    """Lazy schema/metadata reads, then FIT-only streamed weather and geography."""
    if sha(FEATURES) != DATA_SHA:
        raise ValueError('Public D feature hash differs from frozen source')
    split = json.loads(SPLITS.read_text())
    with np.load(FEATURES, allow_pickle=False) as f:
        weather_names = f['damage_features'][:12].astype(str).tolist()
        geo_names = f['geo_features'].astype(str).tolist()
        meta = {k: f[k].copy() for k in METADATA_MEMBERS}
    if weather_names != WEATHER_NAMES or len(geo_names) != 40:
        raise ValueError('Public D input schema changed')
    n = len(meta['system'])
    if n != 8457 or split['n_units'] != n:
        raise ValueError('Public D population differs')
    if not np.isfinite(meta['w']).all() or np.any(meta['w'] <= 0):
        raise ValueError('Invalid public design weights')
    group = merged_groups(meta)
    folds = np.zeros(n, dtype=np.int8)
    for k in range(1, 6):
        e = split['event'][str(k)]
        outside = np.asarray(e['outer'], dtype=int)
        inside = np.asarray(e['dev'], dtype=int)
        if (not np.array_equal(np.sort(np.concatenate([outside, inside])), np.arange(n))
                or len(np.unique(outside)) != len(outside)
                or len(np.unique(inside)) != len(inside) or np.any(folds[outside])):
            raise ValueError('Invalid original event fold partition')
        folds[outside] = k
    for label in np.unique(group):
        if len(np.unique(folds[group == label])) != 1:
            raise ValueError('Merged group crosses an original fold')
    fit = np.sort(np.asarray(split['event']['1']['dev'], dtype=int))
    if len(fit) != 6350 or np.any(folds[fit] == 1):
        raise ValueError('Unexpected original fold1 FIT')
    with zipfile.ZipFile(FEATURES) as archive:
        weather = read_fit_member(archive, 'xu', fit, n, (216, 12), list(range(12)))
        geo = read_fit_member(archive, 'geo', fit, n, (40,))
    if weather.shape != (6350, 216, 12) or geo.shape != (6350, 40):
        raise ValueError('Unexpected retained FIT input shapes')
    if not np.isfinite(weather).all():
        raise ValueError('Raw FIT weather is nonfinite')
    retained = {k: v[fit] for k, v in meta.items()}
    return dict(unit=fit, weather=weather, geo=geo, meta=retained,
                group=group[fit], fold=folds[fit], geo_names=geo_names)


def population_weights(group, w):
    """Equal-group mass; retain original design proportions within each group."""
    labels, codes = np.unique(group, return_inverse=True)
    total = np.bincount(codes, weights=w, minlength=len(labels))
    weight = w / total[codes] / len(labels)
    assert np.isclose(weight.sum(), 1)
    return weight


def moments_paths(x, weight):
    mean = np.sum(weight[:, None] * x.mean(1, dtype=np.float64), axis=0)
    second = np.sum(weight[:, None] * np.square(x.astype(np.float64)).mean(1), axis=0)
    scale = np.sqrt(np.maximum(second - mean * mean, 0))
    active = scale > 1e-10
    scale[~active] = 1
    return mean, scale, active


def fit_scales(weather, geo, county, group, w):
    """Input-only scales: all FIT paths, unique FIT counties for geo40."""
    weight = population_weights(group, w)
    wx = weather.astype(np.float64, copy=True)
    for name in ('cape', 'precip', 'snowfall'):
        j = WEATHER_NAMES.index(name)
        wx[..., j] = np.log1p(np.maximum(wx[..., j], 0))
    mu, sd, active = moments_paths(wx, weight)
    changes = np.diff(wx, axis=1)
    dmu, dsd, dactive = moments_paths(changes, weight)
    z = ((wx - mu) / sd).astype(np.float32)
    dz = ((changes - dmu) / dsd).astype(np.float32)
    z[..., ~active] = 0
    dz[..., ~dactive] = 0
    counties, first = np.unique(county, return_index=True)
    unique = geo[first].astype(np.float64)
    for c, j in zip(counties, first):
        if not np.array_equal(geo[county == c], np.broadcast_to(geo[j], geo[county == c].shape), equal_nan=True):
            raise ValueError('Geo40 differs between the same FIT county')
    median = np.zeros(geo.shape[1], dtype=np.float64)
    for j in range(geo.shape[1]):
        valid = np.isfinite(unique[:, j])
        if valid.any():
            median[j] = np.median(unique[valid, j])
    imputed = np.where(np.isfinite(unique), unique, median)
    gmu, gsd = imputed.mean(0), imputed.std(0)
    gactive = gsd > 1e-10
    gsd[~gactive] = 1
    gs = ((np.where(np.isfinite(geo), geo, median) - gmu) / gsd).astype(np.float32)
    gs[:, ~gactive] = 0
    scales = dict(weather_level_mean=mu.tolist(), weather_level_scale=sd.tolist(),
                  weather_level_active=active.tolist(),
                  weather_difference_mean=dmu.tolist(), weather_difference_scale=dsd.tolist(),
                  weather_difference_active=dactive.tolist(),
                  geography_imputation_median=median.tolist(), geography_mean=gmu.tolist(),
                  geography_scale=gsd.tolist(), geography_active=gactive.tolist(),
                  unique_fit_counties=int(len(counties)),
                  geography_missing_entries=int((~np.isfinite(geo)).sum()),
                  weather_scale_weighting='equal merged group; design proportions within group; all hours',
                  geography_scale_weighting='equal unique FIT county',
                  weather_transforms={'cape': 'log1p(max(x,0))', 'precip': 'log1p(max(x,0))',
                                      'snowfall': 'log1p(max(x,0))'})
    return Geometry(z, dz, gs), scales


class Geometry:
    def __init__(self, level, difference, geo):
        self.level = np.asarray(level)
        self.difference = np.asarray(difference)
        self.geo = np.asarray(geo)
        n = len(self.geo)
        assert self.level.shape[0] == self.difference.shape[0] == n
        assert all(np.isfinite(x).all() for x in (self.level, self.difference, self.geo))

    def distances(self, anchor, rows):
        rows = np.asarray(rows, dtype=int)
        level = np.square(self.level[rows].astype(np.float64) - self.level[anchor]).mean(axis=(1, 2))
        difference = np.square(self.difference[rows].astype(np.float64) - self.difference[anchor]).mean(axis=(1, 2))
        geo = np.square(self.geo[rows].astype(np.float64) - self.geo[anchor]).mean(axis=1)
        return np.sqrt(.5 * (level + difference)), np.sqrt(geo), np.sqrt(difference)

    def medoid(self, rows):
        rows = np.asarray(rows, dtype=int)
        level = self.level[rows].astype(np.float64)
        difference = self.difference[rows].astype(np.float64)
        geo = self.geo[rows].astype(np.float64)
        # Squared-distance medoid equals the closest observed point to the mean.
        cost = (.5 * np.square(level - level.mean(0)).mean(axis=(1, 2))
                + .5 * np.square(difference - difference.mean(0)).mean(axis=(1, 2))
                + np.square(geo - geo.mean(0)).mean(axis=1))
        return int(rows[np.argmin(cost)])


def cutoff_pairs(geometry, n, seed=SEED, count=PAIR_SAMPLE):
    """Fixed distinct-unit Monte Carlo pairs; no outcome or pair winner search."""
    if n < 2:
        raise ValueError('At least two input rows required')
    rng = np.random.default_rng(seed)
    left = rng.integers(0, n, size=count)
    right = rng.integers(0, n - 1, size=count)
    right += right >= left
    dw, dg = np.empty(count), np.empty(count)
    for start in range(0, count, 64):
        a, b = left[start:start + 64], right[start:start + 64]
        level = np.square(geometry.level[a].astype(float) - geometry.level[b]).mean(axis=(1, 2))
        diff = np.square(geometry.difference[a].astype(float) - geometry.difference[b]).mean(axis=(1, 2))
        dw[start:start + len(a)] = np.sqrt(.5 * (level + diff))
        dg[start:start + len(a)] = np.sqrt(np.square(geometry.geo[a].astype(float) - geometry.geo[b]).mean(axis=1))
    thresholds = dict(weather_near=float(np.percentile(dw, 10)),
                      weather_far=float(np.percentile(dw, 75)),
                      geography_near=float(np.percentile(dg, 10)),
                      geography_far=float(np.percentile(dg, 75)))
    return thresholds, np.column_stack([left, right])


def farthest_add(geometry, rows, selected):
    rows = np.asarray(rows, dtype=int)
    nearest = np.full(len(rows), np.inf)
    for anchor in sorted(selected):
        dw, dg, _ = geometry.distances(anchor, rows)
        nearest = np.minimum(nearest, dw * dw + dg * dg)
    nearest[np.isin(rows, list(selected))] = -np.inf
    if not np.isfinite(nearest).any():
        raise ValueError('No unselected row for farthest-point extension')
    return int(rows[np.argmax(nearest)])


def group_seed(group, seed):
    token = int.from_bytes(hashlib.sha256(str(group).encode()).digest()[:8], 'little')
    return np.random.SeedSequence([seed, token])


def select_panel(geometry, group, unit, thresholds, seed=SEED):
    """Cap24: core16 input contrasts/coverage plus tail8 SRSWOR per large group."""
    group, unit = np.asarray(group), np.asarray(unit, dtype=int)
    if not np.array_equal(unit, np.sort(unit)) or len(np.unique(unit)) != len(unit):
        raise ValueError('Population IDs must be sorted and unique for tie-breaking')
    labels = np.unique(group)
    rows_by_group = {g: np.flatnonzero(group == g) for g in labels}
    cores, anchors = {}, []
    for g in labels:
        rows = rows_by_group[g]
        chosen = {geometry.medoid(rows)}
        while len(chosen) < min(ANCHORS, len(rows)):
            chosen.add(farthest_add(geometry, rows, chosen))
        anchors.extend((g, a) for a in sorted(chosen))
        cores[g] = set(map(int, rows)) if len(rows) <= CAP else chosen
    all_rows = np.arange(len(unit))
    pairs = []

    def partner_record(kind, source_group, anchor, candidates):
        dw, dg, _ = geometry.distances(anchor, candidates)
        if kind == 'near_weather_far_geo':
            eligible = (dw <= thresholds['weather_near']) & (dg >= thresholds['geography_far'])
            contrast = dg
        else:
            eligible = (dg <= thresholds['geography_near']) & (dw >= thresholds['weather_far'])
            contrast = dw
        qualified = candidates[eligible]
        available = np.asarray([int(p) in cores[group[p]] or
                                len(cores[group[p]]) < CORE or
                                len(rows_by_group[group[p]]) <= CAP for p in qualified], dtype=bool)
        record = dict(kind=kind, anchor_unit=int(unit[anchor]), anchor_group=str(source_group),
                      qualified_candidates=int(eligible.sum()), quota_rejections=int((~available).sum()))
        if not eligible.any():
            record['status'] = 'no_qualified_input_partner'
        elif not available.any():
            record['status'] = 'qualified_but_core_quota_exhausted'
        else:
            allowed_pos = np.flatnonzero(eligible)[available]
            pos = int(allowed_pos[np.argmax(contrast[allowed_pos])])
            partner = int(candidates[pos])
            cores[group[partner]].add(partner)
            record.update(status='selected', partner_unit=int(unit[partner]),
                          partner_group=str(group[partner]), weather_distance=float(dw[pos]),
                          geography_distance=float(dg[pos]))
        pairs.append(record)

    # Ordering is fixed before labels: all same-group comparisons, then cross-group.
    for g, anchor in anchors:
        candidates = rows_by_group[g]
        partner_record('near_weather_far_geo', g, anchor, candidates[candidates != anchor])
    for g, anchor in anchors:
        partner_record('near_geo_far_weather', g, anchor, all_rows[group != g])
    selected, core_mask, pi = set(), np.zeros(len(unit), dtype=bool), np.empty(len(unit), dtype=float)
    for g in labels:
        rows, chosen = rows_by_group[g], cores[g]
        if len(rows) > CAP:
            if len(chosen) > CORE:
                raise ValueError('Cross-group insertions exceeded core quota')
            while len(chosen) < CORE:
                chosen.add(farthest_add(geometry, rows, chosen))
            remaining = rows[~np.isin(rows, list(chosen))]
            tail = np.random.default_rng(group_seed(g, seed)).choice(remaining, TAIL, replace=False)
            pi[rows] = TAIL / len(remaining)
            selected.update(map(int, tail))
        else:
            pi[rows] = 1
        core_mask[list(chosen)] = True
        pi[list(chosen)] = 1
        selected.update(chosen)
    selected = np.asarray(sorted(selected), dtype=int)
    assert len(selected) == sum(min(CAP, len(rows)) for rows in rows_by_group.values())
    assert np.all((pi > 0) & (pi <= 1)) and np.all(pi[core_mask] == 1)
    for g, rows in rows_by_group.items():
        assert np.sum(group[selected] == g) == min(CAP, len(rows))
        if len(rows) > CAP:
            assert core_mask[rows].sum() == CORE
    return dict(selected=selected, population_core=core_mask, population_pi=pi,
                pairs=pairs, anchors=[int(unit[a]) for _, a in anchors])


def quantiles(x, weight=None):
    x = np.asarray(x, dtype=float)
    if weight is None:
        q = np.quantile(x, [.5, .9, 1])
        mean = x.mean()
    else:
        order = np.argsort(x, kind='stable')
        v, w = x[order], np.asarray(weight, dtype=float)[order]
        c = (np.cumsum(w) - .5 * w) / w.sum()
        q = np.interp([.5, .9, 1], c, v)
        mean = np.sum(x * weight) / np.sum(weight)
    return dict(mean=float(mean), median_q90_max=q.tolist())


def coverage_audit(geometry, group, selected, w):
    """Every full FIT row versus selected inputs in its own merged group."""
    n = len(group)
    nearest_w, nearest_g, nearest_diff, nearest_joint = [np.full(n, np.inf) for _ in range(4)]
    for g in np.unique(group):
        rows = np.flatnonzero(group == g)
        anchors = selected[group[selected] == g]
        for a in anchors:
            dw, dg, dd = geometry.distances(int(a), rows)
            nearest_w[rows] = np.minimum(nearest_w[rows], dw)
            nearest_g[rows] = np.minimum(nearest_g[rows], dg)
            nearest_diff[rows] = np.minimum(nearest_diff[rows], dd)
            nearest_joint[rows] = np.minimum(nearest_joint[rows], np.sqrt(dw * dw + dg * dg))
    assert all(np.isfinite(x).all() for x in (nearest_w, nearest_g, nearest_diff, nearest_joint))
    weight = population_weights(group, w)
    return {key: dict(equal_unit=quantiles(x), equal_group_design=quantiles(x, weight))
            for key, x in [('weather_full216_level_and_change', nearest_w),
                           ('hourly_change_full215', nearest_diff),
                           ('geography_all40', nearest_g), ('joint', nearest_joint)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope', type=Path, required=True)
    parser.add_argument('--scope-commit', required=True)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if not args.scope.is_file() or not set(args.scope_commit).issubset(set('0123456789abcdef')) or len(args.scope_commit) < 7:
        raise ValueError('An existing scope and registration commit are required')
    args.scope_commit = verify_registration(args.scope_commit, args.scope)
    for path in (args.out, SUMMARY_OUT, ROSTER_OUT):
        if path.exists():
            raise FileExistsError(f'Preserve existing artifact: {relative(path)}')
    args.out.resolve().relative_to(ROOT)
    args.scope.resolve().relative_to(ROOT)
    args.out.mkdir(parents=True, exist_ok=False)
    os.nice(max(0, 15 - os.getpriority(os.PRIO_PROCESS, 0)))
    start = time.monotonic()
    source_hash, scope_hash, split_hash = sha(__file__), sha(args.scope), sha(SPLITS)
    print('D08 input-only selection started', flush=True)
    with threadpool_limits(limits=2):
        data = load_inputs()
        group, unit, meta = data['group'], data['unit'], data['meta']
        if len(np.unique(group)) != 53:
            raise ValueError('Expected 53 original FIT merged groups')
        geometry, scales = fit_scales(data['weather'], data['geo'], meta['fips'], group, meta['w'])
        cutoffs, sampled_pairs = cutoff_pairs(geometry, len(unit))
        panel = select_panel(geometry, group, unit, cutoffs)
        selected = panel['selected']
        if len(selected) != 1219:
            raise ValueError('Expected 1219 input-only panel units')
        coverage = coverage_audit(geometry, group, selected, meta['w'])
    core, pi = panel['population_core'], panel['population_pi']
    effective = meta['w'][selected] / pi[selected]
    populations = {str(g): dict(n=int(np.sum(group == g)), design_mass=float(meta['w'][group == g].sum()),
                                selected=int(np.sum(group[selected] == g)),
                                selected_w_over_pi_mass=float(effective[group[selected] == g].sum()))
                   for g in np.unique(group)}
    records = []
    for j in selected:
        records.append(dict(unit=int(unit[j]), fips=str(meta['fips'][j]), system=str(meta['system'][j]),
                            family=str(meta['family'][j]), origin=str(meta['origin'][j]), regime=str(meta['regime'][j]),
                            merged_group=str(group[j]), original_fold=int(data['fold'][j]),
                            w=float(meta['w'][j]), pi=float(pi[j]), core=bool(core[j]),
                            census=populations[str(group[j])]['n'] <= CAP))
    roster = dict(schema='d08_input_panel_roster_v1', original_fit=unit.tolist(), records=records,
                  group_population=populations, settings=SETTINGS,
                  inclusion_probability_definition='core/census pi=1; remainder of groups N>24 pi=8/(N-16)')
    roster_path = args.out / 'roster.json'
    write_json_new(roster_path, roster)
    raw_geo = data['geo'][selected]
    geo_imputed = np.where(np.isfinite(raw_geo), raw_geo,
                           np.asarray(scales['geography_imputation_median'], dtype=np.float32))
    assert np.isfinite(geo_imputed).all()
    arrays = dict(unit=unit[selected], original_fit=unit, weather=data['weather'][selected], geo=raw_geo,
                  geo_imputed=geo_imputed, geo_missing=~np.isfinite(raw_geo),
                  census=np.asarray([populations[str(g)]['n'] <= CAP for g in group[selected]], dtype=bool),
                  merged_group=group[selected], original_fold=data['fold'][selected], core=core[selected], pi=pi[selected],
                  population_core=core, population_pi=pi, population_group=group, population_w=meta['w'],
                  cutoff_sample_unit_pairs=unit[sampled_pairs],
                  **{k: meta[k][selected] for k in METADATA_MEMBERS})
    inputs_path = args.out / 'panel_inputs.npz'
    with inputs_path.open('xb') as f:
        np.savez_compressed(f, **arrays)
    group_effective = np.asarray([p['selected_w_over_pi_mass'] for p in populations.values()])
    pair_summary = {kind: {status: sum(p['kind'] == kind and p['status'] == status for p in panel['pairs'])
                          for status in ('selected', 'no_qualified_input_partner', 'qualified_but_core_quota_exhausted')}
                    for kind in ('near_weather_far_geo', 'near_geo_far_weather')}
    for kind, summary in pair_summary.items():
        accepted = [p for p in panel['pairs'] if p['kind'] == kind and p['status'] == 'selected']
        summary['distinct_unordered_pairs'] = len({tuple(sorted((p['anchor_unit'], p['partner_unit']))) for p in accepted})
        summary['distinct_participants'] = len({u for p in accepted for u in (p['anchor_unit'], p['partner_unit'])})
        summary['distinct_unordered_group_pairs'] = len({tuple(sorted((p['anchor_group'], p['partner_group']))) for p in accepted})
    result = dict(schema='d08_input_panel_v1', scope_commit=args.scope_commit,
                  provenance=dict(data_sha256=DATA_SHA, split_sha256=split_hash, source_sha256=source_hash,
                                  scope_sha256=scope_hash, scope_path=relative(args.scope), source_path=relative(__file__),
                                  D_panel=relative(FEATURES), splits=relative(SPLITS)),
                  settings=SETTINGS, members_read=list(READ_MEMBERS), outcome_members_read=[],
                  model_or_prediction_imports=[],
                  population=dict(units=len(unit), counties=len(np.unique(meta['fips'])),
                                  systems=len(np.unique(meta['system'])), merged_groups=len(populations)),
                  panel=dict(units=len(selected), counties=len(np.unique(meta['fips'][selected])),
                             systems=len(np.unique(meta['system'][selected])), merged_groups=len(populations),
                             core_units=int(core[selected].sum()), tail_units=int((~core[selected]).sum()),
                             original_fold_counts={str(k): int(np.sum(data['fold'][selected] == k)) for k in range(2, 6)},
                             regime_counts={str(r): int(np.sum(meta['regime'][selected] == r)) for r in np.unique(meta['regime'])}),
                  scales=scales, cutoffs=cutoffs, pair_summary=pair_summary, pairs=panel['pairs'],
                  group_population=populations, coverage=coverage,
                  inverse_probability_weight_audit=dict(minimum_population_pi=float(pi.min()),
                      selected_w_over_pi_sum=float(effective.sum()), population_w_sum=float(meta['w'].sum()),
                      selected_w_over_pi_kish=float(effective.sum() ** 2 / np.square(effective).sum()),
                      largest_selected_w_over_pi_share=float(effective.max() / effective.sum()),
                      merged_group_w_over_pi_kish=float(group_effective.sum() ** 2 / np.square(group_effective).sum()),
                      largest_merged_group_w_over_pi_share=float(group_effective.max() / group_effective.sum())),
                  interpretation_limits=[
                      'Input-only coverage is not proof of outage mechanisms or geography utility.',
                      'All40 geography coordinates remain accessible; finite support does not cover every possible combination.',
                      'This manifest contains original fold1 FIT only; frozen fold1 models trained on all those rows.',
                      'Replay on original folds2-5 is not an independent event-heldout prediction result.',
                      'HT inverse-probability risk requires a fixed predictor or independently sampled heldout-group predictions; training risk is not unbiased.',
                      'Pair cutoff failures and exhausted quotas are preserved without relaxation.',
                      'Future inner-model preprocessing must fit inner training rows only.',
                      'Observation support and outcome strata may be inspected only after this roster is frozen.'],
                  resources=dict(nice=os.getpriority(os.PRIO_PROCESS, 0), numerical_threads=2, numpy_version=np.__version__),
                  elapsed_seconds=time.monotonic() - start)
    # Source/data cannot change between selection and freeze.
    if (sha(__file__) != source_hash or sha(args.scope) != scope_hash or
            sha(SPLITS) != split_hash or sha(FEATURES) != DATA_SHA):
        raise ValueError('Source/input/scope changed during selection; do not consume unsealed outputs')
    manifest = dict(result, artifacts={relative(inputs_path): sha(inputs_path), relative(roster_path): sha(roster_path)})
    manifest_path = args.out / 'manifest.json'
    write_json_new(manifest_path, manifest)
    result['roster_sha256'] = sha(roster_path)
    result['panel_inputs_sha256'] = sha(inputs_path)
    SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
    write_json_new(SUMMARY_OUT, result)
    with ROSTER_OUT.open('xb') as f:
        with gzip.GzipFile(filename='', mode='wb', fileobj=f, mtime=0) as g:
            g.write(roster_path.read_bytes())
    frozen = dict(status='input_roster_frozen', manifest_sha256=sha(manifest_path),
                  roster_sha256=sha(roster_path), panel_inputs_sha256=sha(inputs_path),
                  summary_sha256=sha(SUMMARY_OUT), public_roster_sha256=sha(ROSTER_OUT),
                  scope_commit=args.scope_commit)
    write_json_new(args.out / 'FROZEN.json', frozen)
    print(json.dumps(dict(status=frozen['status'], panel=result['panel'], pair_summary=pair_summary,
                          elapsed_seconds=result['elapsed_seconds'], out=relative(args.out)), allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
