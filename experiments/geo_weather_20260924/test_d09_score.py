"""Synthetic-only regression checks for fixed-endpoint D09 arithmetic."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import d09_score as scorer  # Sets the two-thread environment before NumPy loads.
import numpy as np

from d09_score import (ARMS, REGIMES, META_KEYS, phenotype, cohort_masks,
    model_peak, distribution, bootstrap_plan, bootstrap_comparison, gain_summary,
    alarm_summary, score_row, score_surface, verdict, validate_export,
    assert_same_metadata, strict_json, improvement, read_verified_jobs)


def synthetic_data(n=8):
    y = np.zeros((n, 144))
    y[::2, :7] = .2
    y[1::2, :7] = .02
    m = np.ones_like(y, bool)
    fold = np.repeat([2, 3], n // 2)
    group = np.array(['g' + str(i // 2) for i in range(n)])
    data = dict(idx=np.arange(n), y=y, m=m, y0=np.zeros(n),
                w=np.arange(1., n + 1), regime=np.array([REGIMES[(i // 2) % 5] for i in range(n)]),
                fips=np.array([str(i) for i in range(n)]),
                system=group.copy(), family=group.copy(), origin=np.repeat('synthetic', n),
                merged_group=group, original_fold=fold, panel=np.ones(n, bool),
                pi=np.ones(n), origin_observed=np.ones(n, bool))
    predictions = dict(host=.5 * y, crk=.7 * y, new=.8 * y,
                       crk_closed=.5 * y, new_closed=.5 * y)
    return data, predictions


def fake_delivery(folder):
    """Only artificial labels and metadata in a temporary repository folder."""
    data, predictions = synthetic_data()
    split_path, scope_path, features_path = folder / 'split.json', folder / 'scope.md', folder / 'features.npz'
    split = dict(event={'1': dict(dev=list(range(8)), outer=[]),
                        '2': dict(dev=list(range(4, 8)), outer=list(range(4))),
                        '3': dict(dev=list(range(4)), outer=list(range(4, 8)))})
    split_path.write_text(json.dumps(split))
    scope_path.write_text('Synthetic scoring-registration fixture\n')
    features_path.write_bytes(b'Synthetic placeholder; no raw NPZ members are read')
    panel_dir = folder / 'd08'
    panel_dir.mkdir()
    records = []
    for i in range(8):
        record = {k: data[k][i].item() for k in ('fips', 'system', 'family', 'origin', 'regime', 'merged_group', 'original_fold', 'w', 'pi')}
        record['unit'] = i
        records.append(record)
    roster_path, manifest_path, frozen_path = [panel_dir / p for p in ('roster.json', 'manifest.json', 'FROZEN.json')]
    roster_path.write_text(json.dumps(dict(original_fit=list(range(8)), records=records)))
    manifest_path.write_text('{}')
    frozen_path.write_text(json.dumps(dict(status='input_roster_frozen', roster_sha256=scorer.sha(roster_path), manifest_sha256=scorer.sha(manifest_path))))
    jobs = []
    for fold in (2, 3):
        held = np.flatnonzero(data['original_fold'] == fold)
        fit = np.flatnonzero(data['original_fold'] != fold)
        for arm in ARMS:
            jobdir = folder / f'{arm}_fold{fold}'
            jobdir.mkdir()
            export = {k: v[held].copy() for k, v in data.items()}
            export.update(P=predictions[arm][held], u=np.zeros((4, 144)), r=np.zeros((4, 144)), raw_logit=np.zeros((4, 144)))
            if arm != 'host':
                export.update(P_closed=predictions[arm + '_closed'][held], raw_logit_closed=np.zeros((4, 144)))
            for name in ('step0900_held_full.npz', 'step0900_held_panel.npz'):
                np.savez_compressed(jobdir / name, **export)
            for name in ('step0900_fit_panel.npz', 'final.pt', 'stats.json'):
                (jobdir / name).write_bytes(b'synthetic artifact only')
            outputs = {p.name: scorer.sha(p) for p in jobdir.iterdir()}
            protocol = dict(steps=900, seed=0, private_seed=1729, microbatch=512,
                            numerical_threads=2, early_stopping=False, checkpoints=[0, 100, 300, 900],
                            heldout_full_export_step=900, host_learning_rate=.003,
                            recovery_learning_rate=.0003, calibration_every=10,
                            training_weights='original panel w / inner-FIT regime zero-predictor SSE; not HT')
            receipt = dict(schema='d09_job_done_v1', arm=arm, heldout_fold=fold, steps=900, seed=0,
                           registration_commit='a' * 40, scope_sha256=scorer.sha(scope_path),
                           data_sha256=scorer.sha(features_path), split_sha256=scorer.sha(split_path),
                           d08_frozen_sha256=scorer.sha(frozen_path), d08_manifest_sha256=scorer.sha(manifest_path),
                           d08_roster_sha256=scorer.sha(roster_path),
                           source_sha256={scorer.relative(Path(scorer.__file__)): scorer.sha(scorer.__file__)},
                           outputs=outputs, held_full_unit_ids=held.tolist(), held_panel_unit_ids=held.tolist(),
                           fit_unit_ids=fit.tolist(), model_statistics_fit_unit_ids=fit.tolist(),
                           initial_host_parameter_sha256='paired_host',
                           initial_kernel_parameter_sha256=None if arm == 'host' else 'paired_kernel',
                           protocol=protocol, optimizer_covers_all_trainable=True, preflight=False,
                           resources=dict(nice=15, threads=2))
            (jobdir / 'DONE.json').write_text(json.dumps(receipt))
            jobs.append(dict(arm=arm, heldout_fold=fold, path=scorer.relative(jobdir)))
    jobs_path = folder / 'jobs.json'
    jobs_path.write_text(json.dumps(dict(schema='d09_jobs_v1', registration_commit='a' * 40, jobs=jobs)))
    patches = dict(SPLITS=split_path, FEATURES=features_path, D08=panel_dir,
                   DATA_SHA=scorer.sha(features_path))
    return jobs_path, scope_path, patches, jobs


class D09Score(unittest.TestCase):
    def arrays(self, n=1):
        return np.zeros((n, 144)), np.ones((n, 144), bool), np.zeros(n)

    def test_count_not_contiguous_duration(self):
        y, m, y0 = self.arrays()
        y[0, [0, 2, 4, 6, 8, 10, 12]] = .1
        ph = phenotype(y, m, y0)
        self.assertEqual(ph['severe_hours'][0], 7)
        self.assertEqual(ph['max_contiguous_run'][0], 1)
        self.assertTrue(cohort_masks(ph)['max_contiguous_run/1'][0])

    def test_missing_breaks_duration_and_adjacent_rise(self):
        y, m, y0 = self.arrays()
        y[0, :7] = .11
        m[0, 3] = False
        y[0, 3] = np.nan
        ph = phenotype(y, m, y0)
        self.assertEqual(ph['severe_hours'][0], 6)
        self.assertEqual(ph['max_contiguous_run'][0], 3)
        self.assertEqual(ph['adjacent_observed_pairs'][0], 142)
        self.assertAlmostEqual(ph['maximum_adjacent_rise'][0], .11)

    def test_j_uses_origin_without_wrap(self):
        y, m, y0 = self.arrays()
        y[0, 0], y[0, -1], y0[0] = .02, .5, .01
        ph = phenotype(y, m, y0)
        self.assertTrue(cohort_masks(ph)['J'][0])
        self.assertAlmostEqual(ph['maximum_adjacent_rise'][0], .5)

    def test_no_observation_not_nonS(self):
        y, m, y0 = self.arrays()
        m[:] = False
        ph = phenotype(y, m, y0)
        self.assertTrue(np.isnan(ph['true_peak'][0]))
        self.assertEqual(ph['true_peak_hour'][0], -1)
        self.assertFalse(cohort_masks(ph)['all'][0])
        self.assertFalse(cohort_masks(ph)['nonS'][0])

    def test_first_observed_tie_and_peak_ratio_lag(self):
        y, m, y0 = self.arrays()
        y[0, 2:4] = .2
        P = np.zeros_like(y)
        P[0, 0], P[0, 5:7] = .9, .1
        m[0, 0] = False
        ph = phenotype(y, m, y0)
        peak = model_peak(P, m, ph)
        self.assertEqual(ph['true_peak_hour'][0], 74)
        self.assertEqual(peak['predicted_peak_hour'][0], 77)
        self.assertEqual(peak['signed_peak_lag_hours'][0], 3)
        self.assertEqual(peak['peak_ratio'][0], .5)

    def test_peak_ratio_zero_truth_explicit_missing(self):
        y, m, y0 = self.arrays()
        ph = phenotype(y, m, y0)
        peak = model_peak(y, m, ph)
        self.assertTrue(np.isnan(peak['peak_ratio'][0]))
        self.assertEqual(distribution(peak['peak_ratio'], [1])['missing'], 1)

    def test_quantiles_use_cumulative_knots(self):
        self.assertEqual(distribution([0., 1.], [1., 1.])['q10_median_q90'][1], 0.)

    def test_gain_identity_coverage_and_concentration(self):
        y, m, _ = self.arrays(10)
        y[:] = .2
        base, candidate = np.zeros_like(y), np.zeros_like(y)
        candidate[0], candidate[1] = .1, -.1
        weights = np.arange(1., 11)
        result = gain_summary(y, m, base, candidate, np.ones(10, bool), weights)
        self.assertAlmostEqual(result['net_gain'], result['alignment'] - result['modification_energy'])
        self.assertAlmostEqual(result['net_gain'], result['positive_gain'] - result['negative_loss'])
        self.assertEqual(result['positive_units'], 1)
        self.assertEqual(result['negative_units'], 1)
        self.assertEqual(result['zero_units'], 8)
        self.assertAlmostEqual(result['positive_design_share'], 1 / 55)
        self.assertEqual(result['top_10pct_all_cohort_units'], 1)
        self.assertEqual(result['top_10pct_share_of_positive_gain'], 1.)

    def test_gain_ignores_missing_nan(self):
        y, m, _ = self.arrays()
        y[0, 0], m[0, 0] = np.nan, False
        result = gain_summary(y, m, np.zeros_like(y), np.full_like(y, .1), [True], [1.])
        self.assertAlmostEqual(result['net_gain'], -143 * .01)

    def test_false_alarm_new_removed_and_design_rate(self):
        result = alarm_summary(np.array([.1, .05, .2]), np.array([.05, .1, .2]),
                               np.ones(3, bool), np.array([1., 4., 5.]), .1)
        self.assertEqual((result['new_count'], result['removed_count'], result['retained_count']), (1, 1, 1))
        self.assertEqual(result['base_count'], result['candidate_count'])
        self.assertAlmostEqual(result['base_design_rate'], .6)
        self.assertAlmostEqual(result['candidate_design_rate'], .9)

    def test_zero_false_peak_strict_threshold(self):
        result = alarm_summary(np.zeros(2), np.array([.001, .001001]), [True, True], [1., 1.], .001)
        self.assertEqual(result['candidate_count'], 1)

    def test_bootstrap_whole_clusters_same_strata_total(self):
        d, _ = synthetic_data()
        plan = bootstrap_plan(d['regime'], d['w'], d['merged_group'], draws=49)
        again = bootstrap_plan(d['regime'], d['w'], d['merged_group'], draws=49)
        np.testing.assert_array_equal(plan['counts'], again['counts'])
        self.assertEqual(plan['code'][0], plan['code'][1])
        for r in range(5):
            select = plan['strata'] == r
            np.testing.assert_array_equal(plan['counts'][:, select].sum(1), np.repeat(select.sum(), 49))

    def test_bootstrap_paired_proportional_reduction(self):
        plan = bootstrap_plan(['tropical'] * 4, np.ones(4), ['a', 'a', 'b', 'b'], 99)
        result = bootstrap_comparison(np.array([1., 2., 3., 4.]), np.array([1., 2., 3., 4.]) * .64,
                                      np.ones(4, bool), plan)
        np.testing.assert_allclose(result['relative_RMSE_reduction_ci95'], [.2, .2])
        self.assertTrue(result['ci_excludes_zero_in_beneficial_direction'])
        self.assertTrue(result['ci_lower_bound_at_least_final_10pct'])
        self.assertTrue(result['resample_counts_vary'])

    def test_singleton_strata_repetition_not_confidence_support(self):
        plan = bootstrap_plan(['tropical', 'winter'], np.ones(2), ['a', 'b'], 99)
        result = bootstrap_comparison(np.array([1., 2.]), np.array([1., 2.]) * .64,
                                      np.ones(2, bool), plan)
        self.assertEqual(result['supporting_clusters'], 2)
        self.assertFalse(result['resample_counts_vary'])
        self.assertFalse(result['interval_supported'])
        self.assertIsNone(result['relative_RMSE_reduction_ci95'])
        np.testing.assert_allclose(result['empirical_relative_RMSE_reduction_percentiles95'], [.2, .2])
        self.assertFalse(result['ci_excludes_zero_in_beneficial_direction'])
        self.assertFalse(result['ci_lower_bound_at_least_final_10pct'])
        self.assertEqual(result['interval_status'], 'no_effective_cluster_count_variation')

    def test_bootstrap_invalid_draws_and_one_cluster_no_interval(self):
        plan = bootstrap_plan(['tropical'] * 2, np.ones(2), ['a', 'b'], 99)
        result = bootstrap_comparison(np.array([1., 0.]), np.zeros(2), [True, False], plan)
        self.assertGreater(result['invalid_draws'], 0)
        self.assertLess(result['valid_draws'], 99)
        self.assertEqual(result['supporting_clusters'], 1)
        self.assertFalse(result['interval_supported'])
        self.assertIsNone(result['relative_RMSE_reduction_ci95'])

    def test_primary_full_scores_use_original_weight(self):
        d, predictions = synthetic_data()
        d['pi'][:] = .01
        surface = score_surface(d, predictions, draws=9)
        row = surface['rows']['pooled']['S']
        manual = np.sum(d['w'][::2, None] * (.5 * d['y'][::2]) ** 2)
        self.assertAlmostEqual(row['models']['host']['design_SSE'], manual)
        self.assertAlmostEqual(row['comparisons']['new_vs_host']['relative_RMSE_reduction'], .6)
        self.assertNotIn('HT_design_SSE_numerator', row['models']['host'])
        self.assertEqual(surface['rows']['fold2/regime/tropical']['S']['units'], 1)
        self.assertEqual(surface['rows']['fold3/regime/tropical']['S']['units'], 0)

    def test_pool_is_not_mean_fold_reduction(self):
        d, predictions = synthetic_data()
        predictions['new'][:4] = .9 * d['y'][:4]
        predictions['new'][4:] = .6 * d['y'][4:]
        surface = score_surface(d, predictions, draws=9)
        rs = [surface['rows'][s]['S']['comparisons']['new_vs_host']['relative_RMSE_reduction'] for s in ('pooled', 'fold2', 'fold3')]
        self.assertNotAlmostEqual(rs[0], (rs[1] + rs[2]) / 2)

    def test_secondary_ht_fixed_full_denominator(self):
        d, predictions = synthetic_data()
        d['panel'][1:] = False
        d['pi'][0] = .5
        surface = score_surface(d, predictions, panel=True, draws=9)
        row = surface['rows']['pooled']['S']
        model = row['models']['host']
        self.assertEqual(row['units'], 1)
        self.assertAlmostEqual(row['original_full_held_cohort_hour_mass'], 144 * (1 + 3 + 5 + 7))
        self.assertAlmostEqual(model['HT_design_SSE_numerator'], 2 * model['design_SSE'])
        self.assertAlmostEqual(model['HT_MSE_fixed_full_held_cohort_denominator'],
                               model['HT_design_SSE_numerator'] / row['original_full_held_cohort_hour_mass'])

    def test_registered_point_gate_pass_and_no_auto_full_D(self):
        d, predictions = synthetic_data()
        result = verdict(score_surface(d, predictions, draws=9))
        self.assertTrue(result['point_gate_pass'])
        self.assertFalse(result['automatically_start_full_D'])
        self.assertEqual(result['original_final_S_target_relative_RMSE_reduction'], .1)

    def test_gate_requires_both_folds_positive(self):
        d, predictions = synthetic_data()
        predictions['new'][:4] = .4 * d['y'][:4]
        result = verdict(score_surface(d, predictions, draws=9))
        self.assertFalse(result['checks']['fold2_S_new_vs_host_positive'])
        self.assertFalse(result['point_gate_pass'])

    def test_gate_uses_count_and_design_false_alarm(self):
        d, predictions = synthetic_data()
        predictions['host'][1, 0] = .11
        predictions['new'][7, 0] = .11
        result = verdict(score_surface(d, predictions, draws=9))
        self.assertTrue(result['checks']['nonS_severe_alarm_count_not_higher'])
        self.assertFalse(result['checks']['nonS_severe_alarm_design_rate_not_higher'])
        self.assertFalse(result['point_gate_pass'])

    def test_nonheadline_class_guard_two_groups_and_2pct(self):
        d, predictions = synthetic_data()
        d['regime'][:] = 'convective'
        predictions['new'][1::2] = .001
        surface = score_surface(d, predictions, draws=9)
        # Replace just the scalar class reduction to test registered boundary.
        class_comp = surface['rows']['regime/convective']['all']['comparisons']['new_vs_host']
        class_comp['relative_RMSE_reduction'] = -.020001
        result = verdict(surface)
        self.assertFalse(result['checks']['class_convective_all_RMSE_not_worse_than_2pct'])
        class_comp['relative_RMSE_reduction'] = -.02
        self.assertTrue(verdict(surface)['checks']['class_convective_all_RMSE_not_worse_than_2pct'])
        self.assertFalse(result['nonheadline_class_guards']['heavy_rain']['applicable'])

    def test_empty_S_fails_gate_without_nan(self):
        d, predictions = synthetic_data()
        d['y'][:] = 0.
        result = verdict(score_surface(d, predictions, draws=9))
        self.assertFalse(result['point_gate_pass'])
        self.assertIsNone(result['S_new_vs_host_relative_RMSE_reduction'])
        strict_json(result)

    def test_empty_support_row_has_explicit_null(self):
        d, predictions = synthetic_data()
        surface = score_surface(d, predictions, draws=9)
        row = surface['rows']['regime/heavy_rain']['all']
        self.assertEqual(row['units'], 0)
        self.assertIsNone(row['models']['host']['design_RMSE'])
        self.assertIsNone(row['comparisons']['new_vs_host']['relative_RMSE_reduction'])
        strict_json(row)

    def test_validate_rejects_bad_endpoint_identity_mask_and_origin(self):
        d, predictions = synthetic_data()
        export = {k: v[:4].copy() for k, v in d.items()}
        export.update(P=predictions['host'][:4], u=np.zeros((4, 144)),
                      r=np.zeros((4, 144)), raw_logit=np.zeros((4, 144)))
        validate_export(copy.deepcopy(export), 2)
        for key, value in [('idx', [0, 0, 2, 3]), ('m', np.full((4, 144), .5)),
                           ('origin_observed', [True, False, True, True]),
                           ('original_fold', [2, 2, 1, 2]), ('P', np.full((4, 144), np.inf))]:
            bad = copy.deepcopy(export)
            bad[key] = np.asarray(value)
            with self.assertRaises(ValueError):
                validate_export(bad, 2)

    def test_labels_and_metadata_exactly_equal_between_arms(self):
        d, _ = synthetic_data()
        assert_same_metadata(d, copy.deepcopy(d))
        changed = copy.deepcopy(d)
        changed['y'][0, 0] += .001
        with self.assertRaises(ValueError):
            assert_same_metadata(d, changed)
        changed = copy.deepcopy(d)
        changed['family'][0] = 'x'
        with self.assertRaises(ValueError):
            assert_same_metadata(d, changed)

    def test_closed_comparison_is_separate(self):
        d, predictions = synthetic_data()
        surface = score_surface(d, predictions, draws=9)
        comps = surface['rows']['pooled']['S']['comparisons']
        self.assertIn('new_vs_closed', comps)
        self.assertIn('crk_vs_closed', comps)
        self.assertEqual(comps['new_vs_closed']['relative_RMSE_reduction'], comps['new_vs_host']['relative_RMSE_reduction'])

    def test_nonfinite_result_not_silently_converted(self):
        with self.assertRaises(ValueError):
            strict_json(dict(value=np.nan))
        self.assertIsNone(improvement(0., 1.))

    def test_all_six_receipts_checked_before_any_outcome_open(self):
        with tempfile.TemporaryDirectory(prefix='d09_synthetic_', dir=scorer.ROOT) as name:
            folder = Path(name)
            jobs_path, scope, patches, jobs = fake_delivery(folder)
            scorer.under_root(jobs[-1]['path']).joinpath('DONE.json').unlink()
            with patch.multiple(scorer, **patches), patch.object(scorer, 'git_bytes', side_effect=lambda c, p: p.read_bytes()), patch.object(scorer.np, 'load') as load:
                with self.assertRaises(FileNotFoundError):
                    read_verified_jobs(jobs_path, scope, 'a' * 40)
                load.assert_not_called()

    def test_output_hash_failure_precedes_outcome_open(self):
        with tempfile.TemporaryDirectory(prefix='d09_synthetic_', dir=scorer.ROOT) as name:
            jobs_path, scope, patches, jobs = fake_delivery(Path(name))
            scorer.under_root(jobs[0]['path']).joinpath('final.pt').write_bytes(b'altered synthetic file')
            with patch.multiple(scorer, **patches), patch.object(scorer, 'git_bytes', side_effect=lambda c, p: p.read_bytes()), patch.object(scorer.np, 'load') as load:
                with self.assertRaisesRegex(ValueError, 'output hash'):
                    read_verified_jobs(jobs_path, scope, 'a' * 40)
                load.assert_not_called()

    def test_synthetic_six_job_delivery_identity_and_scores(self):
        with tempfile.TemporaryDirectory(prefix='d09_synthetic_', dir=scorer.ROOT) as name:
            jobs_path, scope, patches, _ = fake_delivery(Path(name))
            with patch.multiple(scorer, **patches), patch.object(scorer, 'git_bytes', side_effect=lambda c, p: p.read_bytes()):
                data, predictions, hashes, provenance = read_verified_jobs(jobs_path, scope, 'a' * 40)
            self.assertEqual(provenance['full_heldout_units'], 8)
            np.testing.assert_array_equal(data['idx'], np.arange(8))
            self.assertTrue(verdict(score_surface(data, predictions, draws=9))['point_gate_pass'])

    def test_wrong_endpoint_or_initialization_precedes_outcome_open(self):
        for changed_key, changed_value in [('steps', 300), ('initial_host_parameter_sha256', 'different')]:
            with tempfile.TemporaryDirectory(prefix='d09_synthetic_', dir=scorer.ROOT) as name:
                jobs_path, scope, patches, jobs = fake_delivery(Path(name))
                path = scorer.under_root(jobs[-1]['path']) / 'DONE.json'
                receipt = json.loads(path.read_text())
                receipt[changed_key] = changed_value
                path.write_text(json.dumps(receipt))
                with patch.multiple(scorer, **patches), patch.object(scorer, 'git_bytes', side_effect=lambda c, p: p.read_bytes()), patch.object(scorer.np, 'load') as load:
                    with self.assertRaises(ValueError):
                        read_verified_jobs(jobs_path, scope, 'a' * 40)
                    load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
