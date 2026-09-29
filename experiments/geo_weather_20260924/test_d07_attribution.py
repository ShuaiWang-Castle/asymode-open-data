"""Meaningful synthetic checks for the D07 A paired audit."""
import unittest
import numpy as np

from d07_attribution import (paired_arrays, aggregate, concentration,
                            bootstrap_summary, weighted_transition, outcome_phenotypes)


class PairedAttributionTest(unittest.TestCase):
    def fixture(self):
        # Equal residuals and equal-sized changes, but opposite alignment.
        y = np.array([[.5, .2], [.5, .2], [.5, .2], [99., 99.]])
        m = np.array([[1, 1], [1, 1], [1, 0], [0, 0]], bool)
        base = np.array([[.2, .2], [.2, .2], [.2, .3], [.2, .2]])
        candidate = np.array([[.3, .2], [.1, .2], [.3, .9], [.9, .9]])
        w = np.array([2., 3., 5., 7.])
        meta = {'county': np.array(['a', 'b', 'c', 'd']), 'system': np.array(['s1', 's1', 's2', 's2'])}
        group = np.array(['g1', 'g1', 'g2', 'g2'])
        return y, m, base, candidate, w, meta, group

    def test_known_opposite_signed_effects_and_exact_identity(self):
        y, m, base, candidate, *_ = self.fixture()
        a = paired_arrays(y, m, base, candidate)
        np.testing.assert_allclose(a['gain'], [.05, -.07, .05, 0], atol=1e-15)
        np.testing.assert_allclose(a['alignment'], [.06, -.06, .06, 0], atol=1e-15)
        np.testing.assert_allclose(a['energy'], [.01, .01, .01, 0], atol=1e-15)
        np.testing.assert_allclose(a['identity_error'], 0, atol=1e-15)

    def test_masked_target_and_extreme_predictions_do_not_affect_gain_or_peak(self):
        y, m, base, candidate, *_ = self.fixture()
        y[~m] = np.nan
        base[~m], candidate[~m] = 1e100, -1e100
        a = paired_arrays(y, m, base, candidate)
        self.assertAlmostEqual(a['gain'][2], .05)
        self.assertAlmostEqual(a['candidate_peak'][2], .3)
        self.assertTrue(np.isnan(a['base_peak'][3]))
        self.assertFalse(a['ratio_valid'][3])

    def test_weighted_denominator_and_actual_gain_counts(self):
        y, m, base, candidate, w, meta, group = self.fixture()
        a = paired_arrays(y, m, base, candidate)
        s = aggregate(a, np.ones(4, bool), w, meta, group)
        self.assertEqual(s['support']['units'], 3)
        self.assertEqual(s['support']['weighted_observed_hour_mass'], 15)
        self.assertEqual(s['unit_signs']['positive']['units'], 2)
        self.assertEqual(s['unit_signs']['negative']['units'], 1)
        self.assertAlmostEqual(s['unit_signs']['positive']['design_unit_share'], .7)
        self.assertAlmostEqual(s['totals']['gain'], .14)
        self.assertAlmostEqual(s['base_rmse'], np.sqrt(.9 / 15))

    def test_positive_zero_target_peak_ratio_explicitly_missing(self):
        a = paired_arrays(np.array([[0., 0.]]), np.ones((1, 2), bool),
                          np.array([[.2, .1]]), np.array([[.1, .05]]))
        self.assertFalse(a['ratio_valid'][0])
        self.assertTrue(np.isnan(a['base_ratio'][0]))
        self.assertAlmostEqual(a['peak_accuracy_gain'][0], .1)

    def test_group_concentration_uses_absolute_positive_unit_gains(self):
        s = concentration(np.array([8., 1., 1., 0.]), np.array(['a', 'b', 'b', 'c']))
        self.assertEqual(s['entities'], 3)
        self.assertEqual(s['contributing_entities'], 2)
        self.assertEqual(s['top10pct_all_entity_count'], 1)
        self.assertAlmostEqual(s['top10pct_all_entity_share'], .8)
        self.assertAlmostEqual(s['effective_entities'], 100 / 68)
        self.assertIsNone(concentration(np.zeros(3))['largest_entity_share'])

    def test_joint_transitions_are_not_marginal_reconstructions(self):
        t = weighted_transition(np.array([1, 1, 1, 0], bool),
                 np.array(['positive', 'negative', 'positive', 'negative']),
                 np.array(['positive', 'positive', 'negative', 'negative']),
                 np.array([2., 3., 5., 7.]), ('positive', 'negative'), ('positive', 'negative'))
        counts = {(row[0], row[1]): row[2] for row in t['rows']}
        self.assertEqual(counts, {('positive', 'positive'): 1, ('positive', 'negative'): 1,
                                 ('negative', 'positive'): 1, ('negative', 'negative'): 0})
        self.assertEqual(t['design_unit_mass'], 10)

    def test_bootstrap_whole_groups_and_empty_support(self):
        y, m, base, candidate, w, *_ = self.fixture()
        a = paired_arrays(y, m, base, candidate)
        plan = {'code': np.array([0, 0, 1, 1]), 'levels': np.array(['g1', 'g2']),
                'counts': np.array([[1, 1], [2, 0], [0, 2]])}
        b = bootstrap_summary(a, np.ones(4, bool), w, plan)
        self.assertEqual(b['mse_gain']['valid_draws'], 3)
        self.assertEqual(b['mse_gain']['supporting_clusters'], 2)
        empty = bootstrap_summary(a, np.zeros(4, bool), w, plan)
        self.assertIsNone(empty['mse_gain']['ci95'])
        self.assertEqual(empty['mse_gain']['invalid_draws'], 3)

    def test_unified_fit_denominator_and_subset_contributions_add(self):
        y, m, base, candidate, w, meta, group = self.fixture()
        a = paired_arrays(y, m, base, candidate)
        rw, denominator = w / 2, 7.5
        full = aggregate(a, np.ones(4, bool), w, meta, group, fit_objective=(rw, denominator))
        first = aggregate(a, np.array([1, 1, 0, 0], bool), w, meta, group, fit_objective=(rw, denominator))
        second = aggregate(a, np.array([0, 0, 1, 0], bool), w, meta, group, fit_objective=(rw, denominator))
        for key in ('base_sse', 'candidate_sse', 'alignment', 'energy', 'gain'):
            self.assertAlmostEqual(full['original_fit_objective_contribution'][key],
                  first['original_fit_objective_contribution'][key] + second['original_fit_objective_contribution'][key])

    def test_fixed_outcome_phenotypes_ignore_missing_severe_hours(self):
        y = np.array([[.1, .9], [.15, .15], [.2, .3]])
        m = np.array([[1, 0], [1, 1], [1, 1]], bool)
        masks, values = outcome_phenotypes(dict(y=y, m=m))
        chosen = dict(masks)
        np.testing.assert_array_equal(values['severe_hours'], [1, 2, 2])
        np.testing.assert_allclose(values['true_peak'], [.1, .15, .3])
        np.testing.assert_allclose(values['zero_sse'], [.01, .045, .13])
        np.testing.assert_array_equal(chosen['severe_hours/1|true_peak/[.1,.2)'], [1, 0, 0])


if __name__ == '__main__':
    unittest.main()
