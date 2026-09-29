"""Frozen D04 weather-component block summaries on phase-zero held-out rows.

No training. --self-test uses synthetic matrices only. The normal report must
wait for all five OOF exports; it validates them before transforming any rows.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '2'

import argparse
import gc
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from d04_data import ROOT, HERE, load_panel, make_rows
from d04_features import REGIMES, transform_weather
from d04_models import D04Model, explain_components
from report_d04_response import clean, load_oof, sha
from run_d04_response import ARMS, DEFAULT_RUN, source_hashes, subset

BLOCKS = (('main', 0, 144), ('synchronous', 144, 342),
          ('symmetric_cross_band', 342, 576), ('antisymmetric_order', 576, 774))
COMPONENTS = ('shared', 'context', 'geo')
REGION_LABELS = ('all', *REGIMES)
TYPE_LABELS = ('all', 'T0', 'T1', 'T2', 'T3', 'T4', 'T5')
LOOKUP = ROOT/'runs/geo_weather_20260924/county_structure_d02/county_structure_d02_lookup.npz'


def map_county_types(counties, lookup_counties, lookup_types):
    lookup_counties = np.asarray(lookup_counties).astype(str)
    lookup_types = np.asarray(lookup_types)
    if (lookup_counties.ndim != 1 or lookup_types.shape != lookup_counties.shape
            or len(np.unique(lookup_counties)) != len(lookup_counties)
            or np.any(~np.isin(lookup_types, np.arange(6)))):
        raise ValueError('D02 lookup must contain one valid type per unique county')
    mapping = dict(zip(lookup_counties, lookup_types.astype(int)))
    absent = set(np.asarray(counties).astype(str))-set(mapping)
    if absent:
        raise ValueError(f'D04 counties absent from D02 lookup: {len(absent)}')
    return np.asarray([mapping[str(county)] for county in counties], dtype=np.int8)


def decompose_blocks(model, X, N, G, C, county, y_scale):
    """Return [row, shared/context/geo, four blocks], in fraction-change units."""
    if not np.isfinite(y_scale) or y_scale <= 0:
        raise ValueError('Training response scale must be positive and finite')
    p = len(model.beta_x)
    if p not in (144, 774) or X.shape[1] != p:
        raise ValueError('Unexpected D04 weather basis width')
    x, g, c = [np.asarray(a, float) for a in (X, G, C)]
    geo_readout = g@np.asarray(model.v_geo, float)
    context_readout = c@np.asarray(model.v_context, float)
    values = np.zeros((len(x), 3, 4), dtype=float)
    for j, (_, lo, hi) in enumerate(BLOCKS):
        stop = min(hi, p)
        if lo >= stop:
            continue
        block = x[:, lo:stop]
        values[:, 0, j] = block@np.asarray(model.beta_x[lo:stop], float)
        values[:, 1, j] = ((block@np.asarray(model.u_context[lo:stop], float))*context_readout).sum(1)
        values[:, 2, j] = ((block@np.asarray(model.u_geo[lo:stop], float))*geo_readout).sum(1)
    values *= y_scale
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite component block')
    reference = explain_components(model, X, N, G, C, county)
    expected = np.column_stack([reference[name] for name in COMPONENTS]).astype(float)*y_scale
    # Production explain_components uses float32; the decomposition deliberately
    # accumulates in float64. Allow roundoff from reordering its matrix products.
    np.testing.assert_allclose(values.sum(2), expected, rtol=2e-4, atol=2e-5*y_scale)
    return values, np.max(np.abs(values.sum(2)-expected), axis=0)


def operator_norms(model, y_scale):
    """Norms of reconstructed operators, not norms of unidentified factors."""
    operators = {'shared': np.asarray(model.beta_x, float),
                 'context': np.asarray(model.u_context, float)@np.asarray(model.v_context, float).T,
                 'geo': np.asarray(model.u_geo, float)@np.asarray(model.v_geo, float).T}
    p = len(model.beta_x)
    out = {}
    for name, operator in operators.items():
        operator = operator*y_scale
        blocks = {label: float(np.linalg.norm(operator[lo:min(hi, p)]))
                  for label, lo, hi in BLOCKS if lo < p}
        total = float(np.linalg.norm(operator))
        np.testing.assert_allclose(sum(value*value for value in blocks.values()), total*total,
                                   rtol=1e-12, atol=1e-24)
        out[name] = {'block_norms': blocks, 'total_norm': total}
    return out


def _cell_codes(rows, types):
    reg = np.asarray([REGIMES.index(str(r))+1 for r in rows['regime']], int)
    typ = np.asarray(types, int)+1
    return np.column_stack([np.zeros(len(reg), int), reg*7, typ, reg*7+typ])


def add_chunk(sums, squares, values, rows, types):
    """Accumulate disjoint blocks and their summed component using original w."""
    v = np.concatenate([values, values.sum(2, keepdims=True)], axis=2).reshape(len(values), 15)
    codes = _cell_codes(rows, types).ravel()
    w = np.asarray(rows['w'], float)
    for j in range(15):
        sums[:, j] += np.bincount(codes, weights=np.repeat(w*v[:, j], 4), minlength=42)
        squares[:, j] += np.bincount(codes, weights=np.repeat(w*v[:, j]*v[:, j], 4), minlength=42)


def support_table(rows, types):
    cells = []
    for regime in REGION_LABELS:
        reg = np.ones(len(types), bool) if regime == 'all' else rows['regime'] == regime
        for label in TYPE_LABELS:
            use = reg if label == 'all' else reg & (types == int(label[1:]))
            groups, codes = np.unique(rows['group'][use], return_inverse=True)
            w = np.asarray(rows['w'][use], float)
            mass = np.bincount(codes, weights=w) if len(w) else np.array([])
            cells.append({'regime': regime, 'county_type': label, 'rows': int(use.sum()),
                          'counties': len(np.unique(rows['county'][use])),
                          'county_events': len(np.unique(rows['unit'][use])),
                          'merged_groups': len(groups), 'design_weight_sum': float(w.sum()),
                          'event_weight_kish': float(mass.sum()**2/(mass@mass)) if len(mass) else None,
                          'max_event_weight_share': float(mass.max()/mass.sum()) if len(mass) else None})
    return cells


def finish_summary(sums, squares, support, weather_width):
    active = 1 if weather_width == 144 else 4
    output = {}
    for i, cell in enumerate(support):
        mass = cell['design_weight_sum']
        if mass:
            mean = (sums[i]/mass).reshape(3, 5)
            rms = np.sqrt(np.maximum(squares[i]/mass, 0)).reshape(3, 5)
            summary = {'weighted_mean': mean[:, :active], 'weighted_rms': rms[:, :active],
                       'component_total_weighted_mean': mean[:, 4],
                       'component_total_weighted_rms': rms[:, 4]}
        else:
            summary = {'weighted_mean': None, 'weighted_rms': None,
                       'component_total_weighted_mean': None, 'component_total_weighted_rms': None}
        output.setdefault(cell['regime'], {})[cell['county_type']] = summary
    return {'component_axis': COMPONENTS, 'block_axis': [block[0] for block in BLOCKS[:active]],
            'summaries': output}


def self_test():
    rng = np.random.default_rng(40601)
    n = 19
    X = rng.normal(size=(n, 774)).astype('f4')
    N = rng.normal(size=(n, 3)).astype('f4')
    G = rng.normal(size=(n, 7)).astype('f4')
    C = rng.normal(size=(n, 3)).astype('f4')
    county = np.asarray(['00001', '00002', '00003', '00001']*5)[:n]
    rows = {'w': np.linspace(.2, 2, n), 'regime': np.asarray(REGIMES)[np.arange(n) % 5],
            'unit': np.arange(n) % 11, 'county': county,
            'group': np.asarray([f'g{i%4}' for i in range(n)])}
    types = np.arange(n) % 6
    support = support_table(rows, types)
    y_scale = .00321
    errors = []
    for width, rank in ((144, 0), (144, 2), (774, 0), (774, 4)):
        def r(shape):
            return (.05*rng.normal(size=shape)).astype('f4')
        model = D04Model(r((3,)), r((width,)), r((width, rank)), r((7, rank)),
                         r((width, 2)), r((3, 2)), np.asarray(['00001', '00002']),
                         np.array([.1, -.2], dtype='f4'), {}, {})
        xx = X[:, :width]
        values, error = decompose_blocks(model, xx, N, G, C, county, y_scale)
        errors.append(error)
        for j, (_, lo, hi) in enumerate(BLOCKS):
            stop = min(width, hi)
            if lo >= stop:
                np.testing.assert_array_equal(values[:, :, j], 0)
                continue
            block = xx[:, lo:stop].astype(float)
            dense_g = model.u_geo[lo:stop].astype(float)@model.v_geo.astype(float).T
            dense_c = model.u_context[lo:stop].astype(float)@model.v_context.astype(float).T
            independent = np.column_stack([block@model.beta_x[lo:stop].astype(float),
                np.einsum('ni,ij,nj->n', block, dense_c, C.astype(float)),
                np.einsum('ni,ij,nj->n', block, dense_g, G.astype(float))])*y_scale
            np.testing.assert_allclose(values[:, :, j], independent, rtol=1e-11, atol=1e-15)
        sums, squares = np.zeros((42, 15)), np.zeros((42, 15))
        for start, stop in ((0, 7), (7, n)):
            add_chunk(sums, squares, values[start:stop], subset(rows, np.arange(start, stop)), types[start:stop])
        summary = finish_summary(sums, squares, support, width)
        for cell in support:
            mask = np.ones(n, bool)
            if cell['regime'] != 'all':
                mask &= rows['regime'] == cell['regime']
            if cell['county_type'] != 'all':
                mask &= types == int(cell['county_type'][1:])
            assert cell['rows'] == int(mask.sum())
            assert cell['counties'] == len(np.unique(rows['county'][mask]))
            assert cell['county_events'] == len(np.unique(rows['unit'][mask]))
            assert cell['merged_groups'] == len(np.unique(rows['group'][mask]))
            np.testing.assert_allclose(cell['design_weight_sum'], rows['w'][mask].sum())
            actual = summary['summaries'][cell['regime']][cell['county_type']]
            if not mask.any():
                assert actual['weighted_mean'] is None
                continue
            active = 1 if width == 144 else 4
            expected_mean = np.average(values[mask], weights=rows['w'][mask], axis=0)
            expected_rms = np.sqrt(np.average(values[mask]**2, weights=rows['w'][mask], axis=0))
            np.testing.assert_allclose(actual['weighted_mean'], expected_mean[:, :active], rtol=1e-12, atol=1e-15)
            np.testing.assert_allclose(actual['weighted_rms'], expected_rms[:, :active], rtol=1e-12, atol=1e-15)
            total_rms = np.sqrt(np.average(values[mask].sum(2)**2, weights=rows['w'][mask], axis=0))
            np.testing.assert_allclose(actual['component_total_weighted_rms'], total_rms, rtol=1e-12, atol=1e-15)
        norms = operator_norms(model, y_scale)
        if rank:
            q, _ = np.linalg.qr(rng.normal(size=(rank, rank)))
            rotated = D04Model(model.beta_n, model.beta_x, model.u_geo.astype(float)@q,
                               model.v_geo.astype(float)@q, model.u_context, model.v_context,
                               model.county_levels, model.county_intercepts, {}, {})
            np.testing.assert_allclose(operator_norms(rotated, y_scale)['geo']['total_norm'],
                                       norms['geo']['total_norm'], rtol=1e-12)
        json.dumps(clean(summary), allow_nan=False)
    mapped = map_county_types(['00001', '00003', '00001'], ['00003', '00001'], [5, 2])
    np.testing.assert_array_equal(mapped, [2, 5, 2])
    for counties, labels in ((['00001', '00001'], [1, 1]), (['00001'], [6]), (['00003'], [1])):
        try:
            map_county_types(['00001'], counties, labels)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid/missing county lookup accepted')
    return {'all_checks_passed': True, 'synthetic_rows': n, 'model_structures': 4,
            'max_component_reconstruction_error_fraction': float(np.max(errors))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--lookup', type=Path, default=LOOKUP)
    parser.add_argument('--out', type=Path, default=HERE/'results/v1/d04_components.json')
    parser.add_argument('--chunk-size', type=int, default=4096)
    args = parser.parse_args()
    if args.chunk_size < 1:
        raise ValueError('chunk-size must be positive')
    os.nice(max(0, 15-os.getpriority(os.PRIO_PROCESS, 0)))
    with threadpool_limits(limits=2):
        if args.self_test:
            print(json.dumps(self_test()))
            return
        if args.out.exists():
            raise FileExistsError('Preserve existing D04 component report')
        panel = load_panel()
        all_rows = make_rows(panel, phase=None)
        pred, bundles, manifests = load_oof(args.run_dir, panel, all_rows)
        del pred, all_rows
        gc.collect()
        rows = make_rows(panel, phase=0)
        wx = transform_weather(panel)
        with np.load(args.lookup, allow_pickle=False) as lookup:
            label_key = 'type' if 'type' in lookup.files else 'labels'
            types = map_county_types(rows['county'], lookup['county'], lookup[label_key])
        support = support_table(rows, types)
        sums = {arm: np.zeros((42, 15)) for arm in ARMS}
        squares = {arm: np.zeros((42, 15)) for arm in ARMS}
        seen = np.zeros(len(rows['y']), dtype=np.int8)
        validation, norms = [], []
        for bundle in bundles:
            fold, fm, scale = bundle['fold'], bundle['feature_map'], bundle['y_scale']
            test_idx = np.flatnonzero(rows['fold'] == fold)
            seen[test_idx] += 1
            errors = {arm: np.zeros(3) for arm in ARMS}
            for start in range(0, len(test_idx), args.chunk_size):
                idx = test_idx[start:start+args.chunk_size]
                query = subset(rows, idx)
                X, N, G, C = fm.transform(panel, wx, query)
                for arm, specification in ARMS.items():
                    value, error = decompose_blocks(bundle['models'][arm], X[:, :specification['columns']],
                                                   N, G, C, query['county'], scale)
                    add_chunk(sums[arm], squares[arm], value, query, types[idx])
                    errors[arm] = np.maximum(errors[arm], error)
                    del value
                del X, N, G, C, query
            validation.append({'fold': fold, 'phase0_heldout_rows': len(test_idx),
                               'component_axis': COMPONENTS, 'max_abs_error_fraction': errors,
                               'reference': 'd04_models.explain_components multiplied by training y_scale',
                               'relative_tolerance': 2e-4, 'absolute_tolerance_fraction': 2e-5*scale})
            norms.append({'fold': fold, 'y_scale': scale,
                          'arms': {arm: operator_norms(model, scale) for arm, model in bundle['models'].items()}})
            gc.collect()
            print(json.dumps({'phase': 'component_fold_complete', 'fold': fold, 'rows': len(test_idx)}), flush=True)
        if not np.all(seen == 1):
            raise ValueError('Phase-zero heldout rows were not covered exactly once')
        result = {'meta': {'analysis': 'D04 frozen fitted component decomposition on phase-zero OOF rows',
                   'source_hashes': source_hashes(), 'report_script_sha256': sha(__file__),
                   'lookup_sha256': sha(args.lookup), 'lookup_label_field': label_key,
                   'validated_folds': [m['fold'] for m in manifests], 'phase0_rows': len(rows['y']),
                   'units': 'hourly net fraction change; multiply by 100 for percentage points',
                   'component_axis': COMPONENTS, 'weather_block_ranges': {name: [lo, hi] for name, lo, hi in BLOCKS},
                   'weighting': 'original phase-zero county-event design weights conditional on each slice; all-regime is pooled',
                   'county_type_role': 'D02 current-event-outcome-blind navigation partition, not six physical mechanisms',
                   'transform_scope': 'the saved fold feature_map is applied without fitting; all bases are fold-training standardized',
                   'operator_norm_unit': 'Euclidean/Frobenius norm of reconstructed coefficients times y_scale in fold-standardized coordinates',
                   'block_total_rule': 'block predictions sum to each component; block RMS values do not sum to total RMS',
                   'interpretation': 'correlated-basis fitted decomposition; not independent effects, physical contributions, causal effects, or unique order evidence',
                   'nonidentifiability': 'low-rank factors and allocation across correlated inputs are not unique; norms use reconstructed coefficient operators',
                   'uncertainty': 'no model-fitting or selection uncertainty estimated; between-fold differences are descriptive',
                   'no_neural_or_statistical_fit': True, 'phase0_coverage_exactly_once': True},
                  'support_cells': support, 'reconstruction_validation': validation,
                  'fold_operator_norms': norms,
                  'arms': {arm: finish_summary(sums[arm], squares[arm], support, spec['columns'])
                           for arm, spec in ARMS.items()}}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open('x') as handle:
            json.dump(clean(result), handle, ensure_ascii=False, indent=1, allow_nan=False)
            handle.write('\n')
        print(json.dumps({'out': args.out.name, 'rows': len(rows['y'])}))


if __name__ == '__main__':
    main()
