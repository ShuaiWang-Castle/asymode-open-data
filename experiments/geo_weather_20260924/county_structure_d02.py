"""Outcome-blind, finite D02 county structure map; no outage/weather array loading."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import time
import zipfile

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_samples
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FEAT = ROOT / 'data/interim/panel_v1/features_v1D.npz'
RUN = ROOT / 'runs/geo_weather_20260924/county_structure_d02'
OUT = HERE / 'results/v1/county_structure_d02.json'
LOOKUP = RUN / 'county_structure_d02_lookup.npz'
SEED = 20260928
REGIMES = ['tropical', 'winter', 'synoptic_wind', 'convective', 'heavy_rain']
CONTEXT = ['log_cust', 'rucc', 'log_pop_density', 'coop_share',
           'log1p_n_utilities', 'log1p_saidi']
BLOCKS = {
    'terrain_extent': ['elev_mean', 'relief_p95_p5', 'slope_mean_deg', 'steep_frac',
        'ruggedness_tri', 'aspect_N', 'aspect_NE', 'aspect_E', 'aspect_SE', 'aspect_S',
        'aspect_SW', 'aspect_W', 'aspect_NW', 'log_land_area_km2', 's50_relief_p95_p5',
        'elev_mean5', 'relief5'],
    'canopy_land_cover': ['canopy_mean', 'canopy_dense_frac', 'forest_frac',
        'developed_frac', 'wetland_frac', 's50_canopy_mean', 'fia_forest_land_share'],
    'soil_drainage': ['poorly_drained_share', 'hydric_share', 'shallow_share',
        'high_water_table_share', 'root_limiting_share', 'windthrow_susceptibility',
        's50_windthrow_susceptibility', 's50_poorly_drained_share', 'soil_wet_share',
        'soil_windthrow_hazard'],
    'spatial_colocation': ['forest_on_steep_frac', 'canopy_in_developed',
        'forest_wet_coloc', 'wet_in_forest', 'hazard_in_forest', 'forest_near_developed'],
    'county_service_context': CONTEXT,
}


def static_columns_streamed(path: Path, n: int) -> np.ndarray:
    """Decompress one unit at a time; expose only six static columns as an array.

    Compressed NPZ cannot directly memory-map columns. A <30 KB byte buffer holds
    one unit transiently; its weather/history columns never become numpy arrays.
    All 216 copies of the statics are checked before retaining the first copy.
    """
    with zipfile.ZipFile(path) as archive, archive.open('xr.npy') as f:
        version = np.lib.format.read_magic(f)
        shape, fortran, dtype = np.lib.format._read_array_header(f, version)
        if fortran or shape != (n, 216, 33) or dtype != np.dtype('float32'):
            raise ValueError(f'Unexpected xr layout: {shape}, {dtype}, F={fortran}')
        stride = shape[2] * dtype.itemsize
        size = shape[1] * stride
        values = np.empty((n, len(CONTEXT)), dtype=dtype)
        for i in range(n):
            buffer = f.read(size)
            if len(buffer) != size:
                raise ValueError(f'Truncated xr at unit {i}')
            statics = np.ndarray((shape[1], len(CONTEXT)), dtype=dtype, buffer=buffer,
                                 offset=14*dtype.itemsize, strides=(stride, dtype.itemsize))
            if not np.array_equal(statics, np.broadcast_to(statics[0], statics.shape), equal_nan=True):
                raise ValueError(f'Nonconstant context over time at unit {i}')
            values[i] = statics[0]
        if f.read(1):
            raise ValueError('Unexpected trailing xr payload')
    return values


def merged_groups(meta: dict) -> np.ndarray:
    systems = np.unique(meta['system'])
    parent = {s: s for s in systems}
    def find(s):
        while parent[s] != s:
            parent[s] = parent[parent[s]]
            s = parent[s]
        return s
    info = {}
    for s in systems:
        idx = np.flatnonzero(meta['system'] == s)
        info[s] = (meta['family'][idx[0]], np.datetime64(meta['origin'][idx[0]]),
                   set(meta['fips'][idx]))
    for a, b in itertools.combinations(systems, 2):
        fa, oa, ca = info[a]; fb, ob, cb = info[b]
        if fa == fb or (abs(oa-ob) <= np.timedelta64(16, 'D') and ca & cb):
            aa, bb = find(a), find(b)
            parent[max(aa, bb)] = min(aa, bb)
    return np.array([find(s) for s in meta['system']])


def canonical_labels(labels: np.ndarray, county: np.ndarray) -> np.ndarray:
    order = sorted(np.unique(labels), key=lambda k: min(county[labels == k]))
    mapping = {int(k): j for j, k in enumerate(order)}
    return np.array([mapping[int(k)] for k in labels], dtype=np.int16)


def fit(x: np.ndarray, county: np.ndarray, k: int, seed: int):
    km = KMeans(n_clusters=k, random_state=seed, n_init=50, max_iter=500,
                tol=1e-4, algorithm='lloyd').fit(x)
    labels = canonical_labels(km.labels_, county)
    centers = np.stack([x[labels == k].mean(0) for k in range(k)])
    return labels, centers, {'inertia': float(km.inertia_), 'iterations_best_start': int(km.n_iter_)}


def kish(values: np.ndarray) -> float:
    return float(values.sum()**2 / np.square(values).sum()) if np.square(values).sum() else 0.


def finite(value):
    if isinstance(value, dict):
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(v) for v in value]
    if isinstance(value, np.ndarray):
        return finite(value.tolist())
    if isinstance(value, np.generic):
        return finite(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def add_continuous_geometry(summary: dict, lookup: dict):
    """Add a continuous view without changing clustering, preprocessing or inputs."""
    x, labels = lookup['geometry'], lookup['type']
    county, names = lookup['county'], lookup['feature_names'].tolist()
    pca = PCA(n_components=6, svd_solver='full', whiten=False)
    scores = pca.fit_transform(x)
    # Fix arbitrary component signs using the largest absolute loading.
    for j in range(6):
        if pca.components_[j, np.argmax(np.abs(pca.components_[j]))] < 0:
            pca.components_[j] *= -1
            scores[:, j] *= -1
    distance = np.sqrt(((x[:, None, :]-lookup['centroids_geometry'][None, :, :])**2).sum(2))
    nearest = np.argsort(distance, axis=1, kind='stable')[:, :2]
    d1 = distance[np.arange(len(x)), nearest[:, 0]]
    d2 = distance[np.arange(len(x)), nearest[:, 1]]
    own = distance[np.arange(len(x)), labels]
    margin = (d2-d1)/np.maximum(d2, 1e-12)
    np.testing.assert_allclose(margin, lookup['assignment_margin'], rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(scores, (x-pca.mean_) @ pca.components_.T, rtol=1e-12, atol=1e-12)
    axes = {}
    for j, coefficients in enumerate(pca.components_):
        order = np.argsort(coefficients, kind='stable')
        pos = [int(k) for k in order[::-1] if coefficients[k] > 0][:6]
        neg = [int(k) for k in order if coefficients[k] < 0][:6]
        axes[f'PC{j+1}'] = {'explained_variance_ratio': float(pca.explained_variance_ratio_[j]),
            'positive_loadings': [{'feature': names[k], 'coefficient': float(coefficients[k])} for k in pos],
            'negative_loadings': [{'feature': names[k], 'coefficient': float(coefficients[k])} for k in neg]}
    # Compact public rows retain county-level heterogeneity without repeating profiles.
    rows = [[str(county[i]), int(labels[i]), round(float(scores[i, 0]), 6),
             round(float(scores[i, 1]), 6), round(float(own[i]), 6), round(float(d2[i]), 6),
             round(float(margin[i]), 6), int(lookup['unit_count'][i]),
             int(lookup['system_count'][i]), int(lookup['merged_group_count'][i])]
            for i in range(len(x))]
    summary['continuous_geometry'] = {
        'method': 'Unweighted county PCA of the same imputed, standardized, five-block-equalized 46D geometry; full SVD, no whitening.',
        'purpose': 'Continuous navigation of within-type and between-type variation; this is not a geographic map or a set of natural categories.',
        'component_sign': 'Largest absolute component coefficient is positive; sign has no causal meaning.',
        'loading_definition': 'Unit eigenvector coefficients on the block-equalized columns, not feature-outcome effects or feature-PC correlations.',
        'explained_variance_ratio_first6': pca.explained_variance_ratio_,
        'cumulative_explained_variance_ratio_first6': np.cumsum(pca.explained_variance_ratio_),
        'two_dimensional_explained_variance_ratio': float(pca.explained_variance_ratio_[:2].sum()),
        'axes': axes,
        'distance_definition': 'Euclidean distance in the full 46D geometry, not in the 2D PCA display.',
        'margin_definition': '(second-nearest distance - nearest distance)/second-nearest distance; near zero indicates a weak boundary. It is not a probability.',
        'assigned_type_differs_from_nearest_centroid_count': int((labels != nearest[:, 0]).sum()),
        'county_table': {'columns': ['county', 'type', 'pc1', 'pc2', 'centroid_distance',
            'second_nearest_centroid_distance', 'assignment_margin', 'county_events', 'systems', 'merged_groups'],
            'rounding_decimals': 6, 'rows': rows},
        'projection_limit': 'Two PCs omit variation outside their span. Inspect the 46D descriptors, remaining PCs and full-space distances; visual proximity is not a matched-exposure or causal comparison.'}
    lookup.update(pca_scores=scores, pca_coordinates=scores[:, :2], pca_components=pca.components_,
        pca_explained_variance_ratio=pca.explained_variance_ratio_, pca_mean=pca.mean_,
        centroid_distance=own, nearest_centroid_distance=d1, second_nearest_centroid_distance=d2,
        nearest_centroid_type=nearest[:, 0], second_nearest_centroid_type=nearest[:, 1])
    summary['meta']['continuous_geometry_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    summary['meta']['continuous_geometry_scope_sha256'] = hashlib.sha256(
        (HERE/'notes/D02_COUNTY_STRUCTURE_SCOPE_20260928.md').read_bytes()).hexdigest()


def augment_existing_geometry():
    """Explicit additive update: back up existing artifacts and preserve every old field."""
    summary = json.loads(OUT.read_text())
    if 'continuous_geometry' in summary:
        raise RuntimeError('Continuous geometry already exists; preserve it and inspect before rerunning')
    with np.load(LOOKUP, allow_pickle=False) as f:
        lookup = {k: f[k] for k in f.files}
    old_arrays = {k: v.copy() for k, v in lookup.items()}
    expected = hashlib.sha256(lookup['features'].tobytes()+lookup['county'].tobytes()).hexdigest()
    if expected != summary['meta']['selected_array_sha256']:
        raise ValueError('Existing lookup does not match the saved selected-array hash')
    add_continuous_geometry(summary, lookup)
    for k, old in old_arrays.items():
        if old.dtype.kind in 'fc':
            np.testing.assert_array_equal(old, lookup[k])
        elif not np.array_equal(old, lookup[k]):
            raise ValueError(f'Existing lookup field changed: {k}')
    for path in [LOOKUP, OUT]:
        backup = RUN/f'{path.stem}_before_continuous{path.suffix}'
        if backup.exists():
            raise RuntimeError(f'Backup already exists: {backup.name}')
    for path in [LOOKUP, OUT]:
        shutil.copy2(path, RUN/f'{path.stem}_before_continuous{path.suffix}')
    summary['meta']['continuous_geometry_update'] = 'Added after initial structure characterization, without rereading outcomes or changing any original lookup array.'
    tmp_lookup = RUN/'county_structure_d02_lookup_continuous_tmp.npz'
    tmp_result = RUN/'county_structure_d02_continuous_tmp.json'
    np.savez_compressed(tmp_lookup, **lookup)
    tmp_result.write_text(json.dumps(finite(summary), separators=(',', ':'), allow_nan=False)+'\n')
    os.replace(tmp_lookup, LOOKUP)
    os.replace(tmp_result, OUT)
    print(json.dumps({'lookup_sha256': hashlib.sha256(LOOKUP.read_bytes()).hexdigest(),
        'json_bytes': OUT.stat().st_size, 'explained_variance_ratio_first6': lookup['pca_explained_variance_ratio'].tolist(),
        'two_pc_variance': float(lookup['pca_explained_variance_ratio'][:2].sum()),
        'original_lookup_arrays_unchanged': True}, indent=2), flush=True)


def run():
    if OUT.exists() or LOOKUP.exists():
        raise RuntimeError('Existing D02 output: preserve and inspect before rerunning')
    start = time.time()
    with np.load(FEAT, allow_pickle=False) as f:
        meta = {k: f[k] for k in ['fips', 'system', 'family', 'origin', 'regime', 'w', 'w_raw']}
        geo = f['geo']
        names = f['geo_features'].tolist() + CONTEXT
        if f['recovery_features'].tolist()[14:20] != CONTEXT:
            raise ValueError('Unexpected context field order')
    n = len(geo)
    static = static_columns_streamed(FEAT, n)
    values = np.column_stack([geo, static]).astype(float)
    if values.shape != (n, 46) or np.isinf(values).any():
        raise ValueError('Unexpected feature dimension or infinite features')
    county, first, inverse = np.unique(meta['fips'], return_index=True, return_inverse=True)
    features = values[first]
    unequal = ~((values == features[inverse]) | (np.isnan(values) & np.isnan(features[inverse])))
    if unequal.any():
        bad = {names[j]: int(unequal[:, j].sum()) for j in np.flatnonzero(unequal.any(0))}
        raise ValueError(f'County values differ across events: {bad}')
    declared = [name for block in BLOCKS.values() for name in block]
    if len(declared) != 46 or len(set(declared)) != 46 or set(declared) != set(names):
        raise ValueError('The fixed blocks must partition all 46 features')
    blocks = np.array([next(b for b, columns in BLOCKS.items() if name in columns) for name in names])
    missing = ~np.isfinite(features)
    if missing.all(0).any():
        raise ValueError('A feature is missing in every county')
    med = np.nanmedian(features, axis=0)
    imputed = np.where(missing, med, features)
    mean, sd = imputed.mean(0), imputed.std(0)
    scale = np.where(sd > 0, sd, 1.)
    feature_z = (imputed-mean)/scale
    block_scale = np.array([np.sqrt(len(BLOCKS[b])*len(BLOCKS)) for b in blocks])
    geometry = feature_z / block_scale
    labels, centers, fit_record = fit(geometry, county, 6, SEED)
    silhouette = silhouette_samples(geometry, labels)
    distance = np.sqrt(((geometry[:, None, :]-centers[None, :, :])**2).sum(2))
    ordered_distance = np.sort(distance, axis=1)
    margin = (ordered_distance[:, 1]-ordered_distance[:, 0])/np.maximum(ordered_distance[:, 1], 1e-12)
    groups = merged_groups(meta)
    split = json.loads((HERE/'splits_v1D.json').read_text())
    fold = np.zeros(n, dtype=int)
    for k in range(1, 6):
        ids = np.array(split['event'][str(k)]['outer'], dtype=int)
        if fold[ids].any():
            raise ValueError('Duplicate outer-fold membership')
        fold[ids] = k
    if (fold == 0).any():
        raise ValueError('Missing outer-fold membership')
    for group in np.unique(groups):
        if len(np.unique(fold[groups == group])) != 1:
            raise ValueError('Merged event group crosses outer folds')
    unit_count = np.bincount(inverse)
    systems_count = np.array([len(np.unique(meta['system'][inverse == j])) for j in range(len(county))])
    families_count = np.array([len(np.unique(meta['family'][inverse == j])) for j in range(len(county))])
    group_count = np.array([len(np.unique(groups[inverse == j])) for j in range(len(county))])
    weight_sum = np.bincount(inverse, weights=meta['w'])
    unit_type = labels[inverse]
    type_summaries = {}
    for k in range(6):
        c = labels == k; rows = unit_type == k
        zm = feature_z[c].mean(0)
        top = np.argsort(-np.abs(zm), kind='stable')[:8]
        spread = geometry[c]-centers[k]
        type_summaries[f'T{k}'] = {
            'counties': int(c.sum()), 'county_events': int(rows.sum()),
            'systems': len(np.unique(meta['system'][rows])),
            'families': len(np.unique(meta['family'][rows])),
            'merged_groups': len(np.unique(groups[rows])),
            'repeated_counties_by_systems': {str(t): int((systems_count[c] >= t).sum()) for t in [2, 3, 5]},
            'repeated_counties_by_merged_groups': {str(t): int((group_count[c] >= t).sum()) for t in [2, 3, 5]},
            'effective_county_support_event_repetition': kish(unit_count[c]),
            'effective_county_support_accumulated_design_weight': kish(weight_sum[c]),
            'distinct_systems_per_county_quantiles': np.quantile(systems_count[c], [0, .25, .5, .75, 1]),
            'per_regime': {r: {'county_events': int((rows & (meta['regime'] == r)).sum()),
                'counties': len(np.unique(meta['fips'][rows & (meta['regime'] == r)])),
                'systems': len(np.unique(meta['system'][rows & (meta['regime'] == r)])),
                'merged_groups': len(np.unique(groups[rows & (meta['regime'] == r)]))} for r in REGIMES},
            'per_outer_fold_county_events': {str(f): int((rows & (fold == f)).sum()) for f in range(1, 6)},
            'feature_profile': {'raw_mean': np.nanmean(features[c], 0),
                'raw_q25': np.nanquantile(features[c], .25, axis=0), 'raw_median': np.nanmedian(features[c], 0),
                'raw_q75': np.nanquantile(features[c], .75, axis=0), 'standardized_mean': zm,
                'standardized_sd': feature_z[c].std(0), 'missing_count': missing[c].sum(0)},
            'largest_absolute_standardized_means': [{'feature': names[j], 'mean_z': float(zm[j])} for j in top],
            'heterogeneity': {'mean_squared_distance_to_centroid': float(np.square(spread).sum(1).mean()),
                'within_type_mean_squared_distance_by_block': {b: float(np.square(spread[:, blocks == b]).sum(1).mean()) for b in BLOCKS},
                'distance_to_centroid_quantiles': np.quantile(distance[c, k], [0, .25, .5, .75, .95, 1]),
                'silhouette_quantiles': np.quantile(silhouette[c], [0, .25, .5, .75, 1]),
                'negative_silhouette_count': int((silhouette[c] < 0).sum()),
                'assignment_margin_below_0_1_count': int((margin[c] < .1).sum())}}
    stability = {'different_K': {}, 'K6_initializations': {}}
    for k in [4, 8]:
        other, _, fr = fit(geometry, county, k, SEED)
        table = np.zeros((6, k), dtype=int)
        np.add.at(table, (labels, other), 1)
        stability['different_K'][str(k)] = {'adjusted_rand_against_K6': adjusted_rand_score(labels, other),
            'K6_by_other_type_counties': table, 'cluster_counties': np.bincount(other), 'fit': fr}
    for seed in [20260929, 20260930]:
        other, _, fr = fit(geometry, county, 6, seed)
        table = np.zeros((6, 6), dtype=int)
        np.add.at(table, (labels, other), 1)
        stability['K6_initializations'][str(seed)] = {'adjusted_rand_against_designated_seed': adjusted_rand_score(labels, other),
            'designated_by_other_type_counties': table, 'fit': fr}
    systems = np.unique(meta['system'])
    system_type = np.stack([np.bincount(unit_type[meta['system'] == s], minlength=6) for s in systems])
    protocol = HERE/'notes/D02_COUNTY_STRUCTURE_SCOPE_20260928.md'
    summary = {
        'meta': {'analysis': 'D02 outcome-blind county structure map', 'exploratory': True,
            'outage_outcomes_read': False, 'neural_or_outage_predictor_fit': False,
            'clustering_features': '40 geography + 6 preexisting county/service context descriptors',
            'unit_weight_for_clustering': 'one per unique sampled D county',
            'scope': 'Sampled D counties only; outcome-blind transductive characterization, not nationally representative or causal types.',
            'K': 6, 'seed': SEED, 'n_init': 50, 'max_iter': 500, 'tol': 1e-4,
            'algorithm': 'lloyd', 'threads': 2, 'nice': os.getpriority(os.PRIO_PROCESS, 0),
            'type_labels': '0..5, ordered by the smallest FIPS within each cluster; no causal interpretation',
            'source': str(FEAT.relative_to(ROOT)), 'lookup': str(LOOKUP.relative_to(ROOT)),
            'source_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'protocol_sha256': hashlib.sha256(protocol.read_bytes()).hexdigest(),
            'selected_array_sha256': hashlib.sha256(features.tobytes()+county.tobytes()).hexdigest(),
            'elapsed_seconds': time.time()-start},
        'audit': {'county_events': n, 'unique_counties': len(county), 'systems': len(systems),
            'families': len(np.unique(meta['family'])), 'merged_groups': len(np.unique(groups)),
            'county_features_identical_across_events': True, 'six_context_columns_identical_over_all_216_hours': True,
            'missing_feature_entries_county_level': int(missing.sum()),
            'counties_with_any_missing_feature': int(missing.any(1).sum()),
            'constant_columns': [names[j] for j in np.flatnonzero(sd == 0)],
            'negative_silhouette_count': int((silhouette < 0).sum()),
            'mean_silhouette': float(silhouette.mean()),
            'assignment_margin_below_0_1_count': int((margin < .1).sum()),
            'merged_groups_respect_existing_event_folds': True},
        'feature_names': names, 'feature_blocks': blocks, 'blocks': BLOCKS,
        'preprocessing': {'imputation': 'unique-county median', 'median': med, 'mean_after_imputation': mean,
            'sd_after_imputation': scale, 'geometry_divisor_after_standardization': block_scale,
            'clipping': False, 'fit_population': 'all distinct sampled D counties; no weights or outcomes'},
        'fit': fit_record, 'types': type_summaries, 'stability': stability,
        'coverage': {'systems': systems, 'system_by_type_county_events': system_type},
        'interpretation_limits': ['Clusters are coarse partitions of a continuous and correlated descriptor space.',
            'Kish counts quantify concentration only; they are not independent event/county sample sizes.',
            'Repeated county support is observational and does not establish matched weather exposure or causal contrasts.',
            'All D geography defines the map. Downstream outcome analyses need event dependence, sparse-cell and multiplicity controls.',
            'Descriptor scaling and equal-block weights are fixed conventions, not learned causal relevance weights.']}
    RUN.mkdir(parents=True, exist_ok=True)
    lookup = dict(county=county, type=labels, features=features,
        feature_names=np.array(names), blocks=blocks, feature_z=feature_z, geometry=geometry,
        impute_median=med, standard_mean=mean, standard_scale=scale, block_scale=block_scale,
        centroids_geometry=centers, silhouette=silhouette, assignment_margin=margin,
        unit_count=unit_count, system_count=systems_count, family_count=families_count,
        merged_group_count=group_count, accumulated_design_weight=weight_sum)
    add_continuous_geometry(summary, lookup)
    np.savez_compressed(LOOKUP, **lookup)
    OUT.write_text(json.dumps(finite(summary), separators=(',', ':'), allow_nan=False)+'\n')
    print(json.dumps(finite({'result': str(OUT.relative_to(ROOT)), 'lookup': str(LOOKUP.relative_to(ROOT)),
        'audit': summary['audit'], 'type_counties': {t: v['counties'] for t, v in type_summaries.items()},
        'initialization_ARI': {s: v['adjusted_rand_against_designated_seed'] for s, v in stability['K6_initializations'].items()},
        'seconds': time.time()-start}), indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--augment-existing-geometry', action='store_true',
                        help='Back up existing D02 artifacts and add PCA/distances without refitting clusters.')
    args = parser.parse_args()
    if os.getpriority(os.PRIO_PROCESS, 0) < 15:
        os.nice(15-os.getpriority(os.PRIO_PROCESS, 0))
    with threadpool_limits(limits=2):
        augment_existing_geometry() if args.augment_existing_geometry else run()
