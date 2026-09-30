"""Synthetic-only precision regression; no real export, label or model access."""
import unittest
from unittest.mock import patch

import d09_precision_score as adapter
import numpy as np


def cancellation_fixture():
    rng = np.random.default_rng(20260929)
    y, base, candidate = [rng.uniform(0, .3, (257, 144)).astype(np.float32) for _ in range(3)]
    mask = rng.random((257, 144)) > .1
    weight = rng.uniform(.5, 2, 257).astype(np.float32)
    return y, mask, base, candidate, weight


def small_fixture():
    n = 8
    y = np.zeros((n, 144), np.float64)
    y[::2, :7], y[1::2, :7] = .2, .02
    group = np.array(['g' + str(i // 2) for i in range(n)])
    data = dict(idx=np.arange(n), y=y, m=np.ones((n, 144), bool), y0=np.zeros(n),
                w=np.arange(1., n + 1), regime=np.repeat('tropical', n),
                fips=np.array([str(i) for i in range(n)]), system=group.copy(),
                family=group.copy(), origin=np.repeat('synthetic', n), merged_group=group,
                original_fold=np.repeat([2, 3], 4), panel=np.ones(n, bool),
                pi=np.ones(n), origin_observed=np.ones(n, bool))
    predictions = dict(host=.5 * y, crk=.7 * y, new=.8 * y,
                       crk_closed=.5 * y, new_closed=.5 * y)
    return data, predictions


class D09PrecisionScore(unittest.TestCase):
    def test_artificial_float32_identity_failure_is_reproduced(self):
        y, mask, base, candidate, weight = cancellation_fixture()
        with self.assertRaisesRegex(AssertionError, 'Paired SSE identity'):
            adapter.frozen.gain_summary(y, mask, base, candidate,
                                        np.ones(len(y), bool), weight)

    def test_identical_stored_values_promoted_to_float64_pass_strict_identity(self):
        y, mask, base, candidate, weight = cancellation_fixture()
        result = adapter.frozen.gain_summary(y.astype(np.float64), mask,
            base.astype(np.float64), candidate.astype(np.float64),
            np.ones(len(y), bool), weight.astype(np.float64))
        self.assertAlmostEqual(result['alignment'] - result['modification_energy'], result['net_gain'], places=12)
        residual, delta = y.astype(np.float64) - base, candidate.astype(np.float64) - base
        gain = weight.astype(np.float64) * np.sum(np.where(mask, 2 * residual * delta - delta ** 2, 0.), axis=1)
        direct = weight.astype(np.float64) * np.sum(np.where(mask,
            (y.astype(np.float64) - base) ** 2 - (y.astype(np.float64) - candidate) ** 2, 0.), axis=1)
        self.assertLess(float(np.max(np.abs(gain - direct))), 2e-10)

    def test_promotion_preserves_masks_ids_metadata_and_exact_numeric_values(self):
        data, predictions = small_fixture()
        for key in adapter.DATA_ARRAYS:
            data[key] = data[key].astype(np.float32)
        predictions = {key: value.astype(np.float32) for key, value in predictions.items()}
        promoted, models, original = adapter.promote_inputs(data, predictions)
        for key in adapter.DATA_ARRAYS:
            self.assertEqual(promoted[key].dtype, np.dtype('float64'))
            np.testing.assert_array_equal(promoted[key], data[key])
            self.assertEqual(data[key].dtype, np.dtype('float32'))
            self.assertEqual(original['data'][key], 'float32')
        for key, value in data.items():
            if key not in adapter.DATA_ARRAYS:
                self.assertIs(promoted[key], value)
        for key, value in predictions.items():
            self.assertEqual(models[key].dtype, np.dtype('float64'))
            np.testing.assert_array_equal(models[key], value)
            self.assertEqual(value.dtype, np.dtype('float32'))

    def test_double_input_primary_and_panel_reports_exactly_equal_original(self):
        data, predictions = small_fixture()
        for panel in (False, True):
            original = adapter.ORIGINAL_SCORE_SURFACE(data, predictions, panel=panel, draws=19)
            corrected = adapter.score_surface_float64(data, predictions, panel=panel, draws=19)
            self.assertEqual(original, corrected)
            self.assertEqual(adapter.frozen.verdict(original), adapter.frozen.verdict(corrected))

    def test_adapter_cli_wrapper_keeps_original_main_and_guard_function(self):
        data, predictions = small_fixture()
        prior_surface, prior_write = adapter.frozen.score_surface, adapter.frozen.write_new
        prior_guard = adapter.frozen.read_verified_jobs
        captured = []

        def fake_main():
            adapter.frozen.score_surface(data, predictions, draws=9)
            result = dict(provenance=dict(files_sha256={}, source_sha256={'unchanged': 'frozen'}), interpretations=[])
            adapter.frozen.write_new('unused_synthetic_output', result)

        with patch.object(adapter, 'verify_adapter_registration', return_value='b' * 40), \
             patch.object(adapter.frozen, 'main', side_effect=fake_main), \
             patch.object(adapter, 'ORIGINAL_WRITE_NEW', side_effect=lambda p, r: captured.append(r)):
            adapter.run_with_precision(['--jobs', 'synthetic_only'], 'b' * 40)
        self.assertIs(adapter.frozen.read_verified_jobs, prior_guard)
        self.assertIs(adapter.frozen.score_surface, prior_surface)
        self.assertIs(adapter.frozen.write_new, prior_write)
        metadata = captured[0]['provenance']['precision_adapter']
        self.assertEqual(metadata['arithmetic_dtype'], 'float64')
        self.assertEqual(metadata['registration_commit'], 'b' * 40)
        self.assertEqual(captured[0]['provenance']['source_sha256'], {'unchanged': 'frozen'})
        self.assertIn(metadata['path'], captured[0]['provenance']['files_sha256'])

    def test_wrapper_restores_original_hooks_on_failure(self):
        prior_surface, prior_write = adapter.frozen.score_surface, adapter.frozen.write_new
        with patch.object(adapter, 'verify_adapter_registration', return_value='b' * 40), \
             patch.object(adapter.frozen, 'main', side_effect=RuntimeError('synthetic guard failure')):
            with self.assertRaisesRegex(RuntimeError, 'synthetic guard failure'):
                adapter.run_with_precision([], 'b' * 40)
        self.assertIs(adapter.frozen.score_surface, prior_surface)
        self.assertIs(adapter.frozen.write_new, prior_write)

    def test_unchanged_tiny_threshold_has_explicit_float32_boundary_semantics(self):
        base, candidate = np.zeros(1, np.float32), np.array([.001], np.float32)
        old = adapter.frozen.alarm_summary(base, candidate, [True], [1.], .001)
        promoted = adapter.frozen.alarm_summary(base.astype(np.float64), candidate.astype(np.float64), [True], [1.], .001)
        self.assertEqual(old['threshold'], promoted['threshold'])
        self.assertEqual(old['candidate_count'], 0)
        self.assertEqual(promoted['candidate_count'], 1)
        self.assertGreater(float(candidate[0]), .001)


if __name__ == '__main__':
    unittest.main()
