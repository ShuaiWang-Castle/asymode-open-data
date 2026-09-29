"""Synthetic D08 tests; no real features, outcomes or frozen models are loaded."""
import ast
import io
import itertools
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

import d08_panel as D


def toy(n=60, groups=None):
    rng = np.random.default_rng(190)
    level = rng.normal(size=(n, 8, 12)).astype(np.float32)
    difference = np.diff(level, axis=1)
    geo = rng.normal(size=(n, 40)).astype(np.float32)
    if groups is None:
        groups = np.repeat(['A', 'B'], n // 2)
    return D.Geometry(level, difference, geo), np.asarray(groups), np.arange(n)


class InputDistanceChecks(unittest.TestCase):
    def test_identity_and_symmetry(self):
        geometry, _, _ = toy()
        self.assertTrue(all(float(x[0]) == 0 for x in geometry.distances(0, [0])))
        a = geometry.distances(0, [1]); b = geometry.distances(1, [0])
        for x, y in zip(a, b):
            np.testing.assert_allclose(x, y, rtol=0, atol=1e-12)

    def test_difference_retains_high_frequency_changes(self):
        level = np.zeros((2, 8, 12), dtype=np.float32)
        level[1, :, 0] = [-1, 1] * 4
        geometry = D.Geometry(level, np.diff(level, axis=1), np.zeros((2, 40)))
        dw, dg, dd = geometry.distances(0, [1])
        self.assertGreater(dw[0], 0); self.assertGreater(dd[0], 0)
        self.assertEqual(dg[0], 0)

    def test_geography_all_coordinates_contribute(self):
        for coordinate in range(40):
            geo = np.zeros((2, 40)); geo[1, coordinate] = 1
            geometry = D.Geometry(np.zeros((2, 8, 12)), np.zeros((2, 7, 12)), geo)
            self.assertAlmostEqual(geometry.distances(0, [1])[1][0], 1 / np.sqrt(40))

    def test_medoid_and_farthest_tie_break_sorted_ids(self):
        geometry = D.Geometry(np.zeros((5, 8, 12)), np.zeros((5, 7, 12)), np.zeros((5, 40)))
        self.assertEqual(geometry.medoid(np.arange(5)), 0)
        self.assertEqual(D.farthest_add(geometry, np.arange(5), {0}), 1)

    def test_cutoffs_reproducible_distinct_and_fixed_quantiles(self):
        geometry, _, unit = toy()
        first, pairs = D.cutoff_pairs(geometry, len(unit), count=100)
        second, again = D.cutoff_pairs(geometry, len(unit), count=100)
        self.assertEqual(first, second); np.testing.assert_array_equal(pairs, again)
        self.assertTrue(np.all(pairs[:, 0] != pairs[:, 1]))
        self.assertLessEqual(first['weather_near'], first['weather_far'])
        self.assertLessEqual(first['geography_near'], first['geography_far'])

    def test_geo_imputation_unique_counties_and_inactive_dimensions(self):
        rng = np.random.default_rng(181)
        wx = rng.normal(size=(4, 8, 12))
        geo = np.zeros((4, 40)); geo[0, 0] = geo[1, 0] = np.nan; geo[2, 0] = 2; geo[3, 0] = 4
        geometry, scales = D.fit_scales(wx, geo, np.array(['a', 'a', 'b', 'c']), np.array(['A'] * 4), np.ones(4))
        self.assertEqual(scales['unique_fit_counties'], 3)
        self.assertEqual(scales['geography_imputation_median'][0], 3)
        self.assertFalse(scales['geography_active'][1])
        self.assertTrue(np.isfinite(geometry.geo).all())

    def test_changed_same_county_geo_is_rejected(self):
        rng = np.random.default_rng(8)
        with self.assertRaisesRegex(ValueError, 'same FIT county'):
            D.fit_scales(rng.normal(size=(2, 8, 12)), rng.normal(size=(2, 40)),
                         np.array(['a', 'a']), np.array(['A', 'A']), np.ones(2))

    def test_scale_weights_equal_groups_not_unit_counts(self):
        w = D.population_weights(np.array(['A', 'A', 'B']), np.array([1., 3., 8.]))
        np.testing.assert_allclose(w, [.125, .375, .5])


class PanelDesignChecks(unittest.TestCase):
    def test_large_groups_have_core16_tail8_pi(self):
        geometry, group, unit = toy()
        thresholds, _ = D.cutoff_pairs(geometry, len(unit), count=100)
        p = D.select_panel(geometry, group, unit, thresholds)
        self.assertEqual(len(p['selected']), 48)
        for g in np.unique(group):
            rows = group == g
            self.assertEqual(p['population_core'][rows].sum(), 16)
            np.testing.assert_allclose(p['population_pi'][rows & ~p['population_core']], 8 / 14)
        self.assertEqual(p['population_core'][p['selected']].sum(), 32)

    def test_census_at_and_below24(self):
        for n in (1, 6, 16, 23, 24):
            geometry, group, unit = toy(n=n, groups=['A'] * n)
            thresholds = dict(weather_near=0, weather_far=1, geography_near=0, geography_far=1)
            p = D.select_panel(geometry, group, unit, thresholds)
            np.testing.assert_array_equal(p['selected'], unit)
            self.assertTrue(p['population_core'].all()); self.assertTrue((p['population_pi'] == 1).all())

    def test25_has_pi_eight_ninths(self):
        geometry, group, unit = toy(n=25, groups=['A'] * 25)
        p = D.select_panel(geometry, group, unit, dict(weather_near=0, weather_far=9, geography_near=0, geography_far=9))
        self.assertEqual(len(p['selected']), 24)
        np.testing.assert_allclose(p['population_pi'][~p['population_core']], 8 / 9)

    def test_selection_reproducibility(self):
        geometry, group, unit = toy()
        thresholds, _ = D.cutoff_pairs(geometry, len(unit), count=100)
        p = D.select_panel(geometry, group, unit, thresholds)
        q = D.select_panel(geometry, group, unit, thresholds)
        np.testing.assert_array_equal(p['selected'], q['selected'])
        np.testing.assert_array_equal(p['population_core'], q['population_core'])
        self.assertEqual(p['pairs'], q['pairs'])

    def test_no_qualified_pair_is_preserved_without_relaxation(self):
        geometry, group, unit = toy()
        p = D.select_panel(geometry, group, unit,
                           dict(weather_near=-1, weather_far=9, geography_near=-1, geography_far=9))
        self.assertTrue(all(x['status'] == 'no_qualified_input_partner' for x in p['pairs']))
        self.assertEqual(len(p['selected']), 48)

    def test_crossgroup_pairs_and_quota(self):
        geometry, group, unit = toy(n=120, groups=np.repeat(['A', 'B', 'C', 'D'], 30))
        p = D.select_panel(geometry, group, unit,
                           dict(weather_near=99, weather_far=0, geography_near=99, geography_far=0))
        for record in p['pairs']:
            if record['status'] == 'selected' and record['kind'] == 'near_geo_far_weather':
                self.assertNotEqual(record['anchor_group'], record['partner_group'])
        for g in np.unique(group):
            self.assertEqual(p['population_core'][group == g].sum(), 16)

    def test_original_ids_tie_order_and_invalid_ids(self):
        geometry, group, unit = toy()
        with self.assertRaisesRegex(ValueError, 'sorted and unique'):
            D.select_panel(geometry, group, unit[::-1], {})

    def test_exact_ht_expectation_for_uniform_tail(self):
        # Enumerate all 8-of-9 tails: core16 always in; no simulation approximation.
        loss = np.arange(1, 26, dtype=float) ** 2
        design_w = np.linspace(.3, 4, 25)
        base = np.sum(design_w[:16] * loss[:16])
        estimates = [base + np.sum(design_w[list(t)] * loss[list(t)] / (8 / 9))
                     for t in itertools.combinations(range(16, 25), 8)]
        self.assertAlmostEqual(np.mean(estimates), np.sum(design_w * loss), places=10)

    def test_population_coverage_census_zero_and_panel_finite(self):
        geometry, group, unit = toy()
        full = D.coverage_audit(geometry, group, unit, np.ones(len(unit)))
        self.assertTrue(all(v['equal_unit']['mean'] == 0 for v in full.values()))
        p = D.select_panel(geometry, group, unit,
                           dict(weather_near=-1, weather_far=9, geography_near=-1, geography_far=9))
        subset = D.coverage_audit(geometry, group, p['selected'], np.ones(len(unit)))
        self.assertTrue(all(np.isfinite(v['equal_unit']['mean']) for v in subset.values()))


class AccessAndFreezeChecks(unittest.TestCase):
    def test_stream_retains_only_fit_rows(self):
        values = np.arange(4 * 3 * 5, dtype=np.float32).reshape(4, 3, 5)
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / 'test.npz'
            np.savez_compressed(path, xu=values)
            with zipfile.ZipFile(path) as archive:
                result = D.read_fit_member(archive, 'xu', np.array([0, 2]), 4, (3, 2), [0, 1])
        np.testing.assert_array_equal(result, values[[0, 2], :, :2])

    def test_stream_static_geo(self):
        values = np.arange(4 * 40, dtype=np.float32).reshape(4, 40)
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / 'test.npz'; np.savez_compressed(path, geo=values)
            with zipfile.ZipFile(path) as archive:
                result = D.read_fit_member(archive, 'geo', np.array([1, 3]), 4, (40,))
        np.testing.assert_array_equal(result, values[[1, 3]])

    def test_stream_fortran_static_geo_retains_fit_values_exactly(self):
        values = np.asfortranarray(np.arange(8 * 40, dtype=np.float32).reshape(8, 40))
        values[2, 7] = np.nan
        fit = np.array([1, 2, 6])
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / 'test.npz'; np.savez_compressed(path, geo=values)
            with zipfile.ZipFile(path) as archive:
                result = D.read_fit_member(archive, 'geo', fit, 8, (40,))
        np.testing.assert_array_equal(result, values[fit])

    def test_fortran_weather_is_rejected(self):
        values = np.asfortranarray(np.arange(4 * 3 * 5, dtype=np.float32).reshape(4, 3, 5))
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / 'test.npz'; np.savez_compressed(path, xu=values)
            with zipfile.ZipFile(path) as archive:
                with self.assertRaisesRegex(ValueError, 'Fortran'):
                    D.read_fit_member(archive, 'xu', np.array([0, 2]), 4, (3, 2), [0, 1])

    def test_fortran_static_column_subset_is_rejected(self):
        values = np.asfortranarray(np.arange(4 * 40, dtype=np.float32).reshape(4, 40))
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / 'test.npz'; np.savez_compressed(path, geo=values)
            with zipfile.ZipFile(path) as archive:
                with self.assertRaisesRegex(ValueError, 'Fortran'):
                    D.read_fit_member(archive, 'geo', np.array([0, 2]), 4, (2,), [0, 1])

    def test_fortran_unrecognized_member_is_rejected(self):
        values = np.asfortranarray(np.arange(4 * 40, dtype=np.float32).reshape(4, 40))
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / 'test.npz'; np.savez_compressed(path, unsupported=values)
            with zipfile.ZipFile(path) as archive:
                with self.assertRaisesRegex(ValueError, 'Fortran'):
                    D.read_fit_member(archive, 'unsupported', np.array([0, 2]), 4, (40,))

    def test_stream_unsorted_fit_rejected(self):
        with self.assertRaises(ValueError):
            D.read_fit_member(None, 'xu', np.array([1, 0]), 4, (3, 2))

    def test_source_import_and_member_whitelist(self):
        tree = ast.parse(Path(D.__file__).read_text())
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import): imported.extend(x.name for x in node.names)
            elif isinstance(node, ast.ImportFrom): imported.append(node.module)
        forbidden = ('torch', 'asymode', 'd07', 'd04_data', 'evaluate', 'audit', 'paper')
        self.assertFalse(any(any(part in name for part in forbidden) for name in imported))
        self.assertEqual(set(D.READ_MEMBERS), {'damage_features', 'geo_features', 'system', 'family',
                                             'origin', 'fips', 'regime', 'w', 'xu', 'geo'})
        function = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'load_inputs')
        literal_reads = {x.slice.value for x in ast.walk(function) if isinstance(x, ast.Subscript)
                         and isinstance(x.value, ast.Name) and x.value.id == 'f'
                         and isinstance(x.slice, ast.Constant)}
        self.assertTrue(literal_reads.issubset(set(D.READ_MEMBERS)))

    def test_exclusive_write_and_canonical_finite_json(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / 'frozen.json'; D.write_json_new(path, dict(status='input_roster_frozen'))
            with self.assertRaises(FileExistsError): D.write_json_new(path, {})
        self.assertEqual(D.json_bytes(dict(b=2, a=1)), D.json_bytes(dict(a=1, b=2)))
        with self.assertRaises(ValueError): D.json_bytes(dict(value=float('nan')))

    def test_registration_requires_scope_and_source_blobs(self):
        with tempfile.TemporaryDirectory(dir=D.ROOT) as name:
            scope = Path(name) / 'scope.md'; scope.write_text('registered scope\n')
            full = 'a' * 40
            source = Path(D.__file__).read_bytes()
            with patch.object(D.subprocess, 'check_output', side_effect=[full + '\n', source, scope.read_bytes()]):
                self.assertEqual(D.verify_registration('a' * 7, scope), full)
            with patch.object(D.subprocess, 'check_output', side_effect=[full + '\n', b'changed source']):
                with self.assertRaisesRegex(ValueError, 'differs'):
                    D.verify_registration('a' * 7, scope)

    def test_synthetic_group_links_match_family_and_shared_county_dates(self):
        meta = dict(system=np.array(['B', 'A', 'C', 'D']), family=np.array(['f1', 'f1', 'f3', 'f4']),
                    origin=np.array(['2020-01-01', '2020-02-01', '2020-03-01', '2020-03-05']),
                    regime=np.array(['r'] * 4), fips=np.array(['c1', 'c2', 'c3', 'c3']))
        np.testing.assert_array_equal(D.merged_groups(meta), ['A', 'A', 'C', 'C'])


if __name__ == '__main__':
    unittest.main()
