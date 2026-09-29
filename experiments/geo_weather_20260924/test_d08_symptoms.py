"""Synthetic checks for mask-aware D08 descriptions and fixed denominators."""
import unittest
import numpy as np

from d08_symptoms import (phenotype, model_peak, comparison, score_cohort,
                          cohort_masks, distribution, strict_json, pair_descriptives)


class D08Symptoms(unittest.TestCase):
    def arrays(self, n=1):
        return np.zeros((n, 144)), np.ones((n, 144), bool), np.zeros(n)

    def test_disjoint_severe_count_is_not_contiguous_run(self):
        y, m, y0 = self.arrays()
        y[0, [0, 2, 4, 6, 8, 10, 12]] = .11
        ph = phenotype(y, m, y0)
        self.assertEqual(ph['severe_hours'][0], 7)
        self.assertEqual(ph['maximum_contiguous_severe_run'][0], 1)
        self.assertTrue(cohort_masks(ph)['severe_hour_count/>=7'][0])
        self.assertTrue(cohort_masks(ph)['max_contiguous_run/1'][0])

    def test_missing_hours_break_runs_and_pairs(self):
        y, m, y0 = self.arrays()
        y[0, :7] = .11
        m[0, 3] = False
        y[0, 3] = .99
        ph = phenotype(y, m, y0)
        self.assertEqual(ph['severe_hours'][0], 6)
        self.assertEqual(ph['maximum_contiguous_severe_run'][0], 3)
        self.assertEqual(ph['adjacent_observed_pairs'][0], 142)
        self.assertAlmostEqual(ph['maximum_adjacent_rise'][0], .11)

    def test_hour71_rise_and_no_forecast_end_wrap(self):
        y, m, y0 = self.arrays()
        y[0, 0], y[0, -1], y0[0] = .2, .9, .1
        ph = phenotype(y, m, y0)
        self.assertEqual(ph['true_peak_hour'][0], 215)
        self.assertAlmostEqual(ph['maximum_adjacent_rise'][0], .9)
        self.assertAlmostEqual(ph['minimum_adjacent_change'][0], -.2)

    def test_peak_masks_and_first_tie(self):
        y, m, y0 = self.arrays()
        y[0, 2:4] = .2
        m[0, 0] = False
        P = np.zeros_like(y)
        P[0, 0], P[0, 5:7] = 1., .1
        ph = phenotype(y, m, y0)
        pp = model_peak(P, m, ph['true_peak'], ph['true_peak_hour'])
        self.assertEqual(ph['true_peak_hour'][0], 74)
        self.assertEqual(pp['predicted_peak_hour'][0], 77)
        self.assertEqual(pp['signed_peak_lag_hours'][0], 3)
        self.assertAlmostEqual(pp['peak_ratio'][0], .5)

    def test_half_height_episode_not_first_unrelated_threshold(self):
        y, m, y0 = self.arrays()
        y[0, 0:2] = .11
        y[0, 10:15] = [.1, .15, .2, .15, .1]
        ph = phenotype(y, m, y0)
        self.assertEqual(ph['peak_centered_half_height_rise_hours'][0], 2)
        self.assertEqual(ph['peak_centered_half_height_fall_hours'][0], 2)
        self.assertFalse(ph['half_height_left_truncated'][0])
        self.assertFalse(ph['half_height_right_truncated'][0])

    def test_half_height_mask_and_window_truncation(self):
        y, m, y0 = self.arrays(2)
        y[0, :3] = [.1, .2, .1]
        y[1, -3:] = [.1, .2, .1]
        m[1, -4] = False
        ph = phenotype(y, m, y0)
        self.assertTrue(ph['half_height_left_truncated'][0])
        self.assertTrue(ph['half_height_right_truncated'][1])
        self.assertTrue(ph['half_height_left_truncated'][1])

    def test_no_observation_is_explicit_missing(self):
        y, m, y0 = self.arrays()
        m[:] = False
        ph = phenotype(y, m, y0)
        self.assertTrue(np.isnan(ph['true_peak'][0]))
        self.assertEqual(ph['true_peak_hour'][0], -1)
        self.assertEqual(distribution(ph['true_peak'], np.ones(1))['missing'], 1)
        self.assertFalse(cohort_masks(ph)['all'][0])

    def test_exact_alignment_and_concentration_denominator(self):
        y, m, _ = self.arrays(2)
        y[:] = .2
        base = np.zeros_like(y)
        candidate = np.zeros_like(y)
        candidate[0] = .1
        candidate[1] = -.1
        r = comparison(y, m, base, candidate, np.ones(2, bool), np.ones(2))
        self.assertAlmostEqual(r['positive_gain'] - r['negative_loss'], r['net_gain'])
        self.assertEqual(r['positive_units'], 1)
        self.assertEqual(r['negative_units'], 1)
        self.assertEqual(r['top_10pct_share_of_positive_gain'], 1.)

    def test_HT_numerator_uses_full_FIT_denominator(self):
        y, m, y0 = self.arrays(2)
        y[:] = .2
        ph = phenotype(y, m, y0)
        models = {k: np.zeros_like(y) for k in ('host', 'crk', 'crk_closed', 'weather_control_0p001')}
        peaks = {k: model_peak(P, m, ph['true_peak'], ph['true_peak_hour']) for k, P in models.items()}
        data = dict(y=y, m=m, w=np.ones(2), rw=np.ones(2),
                    original_full_FIT_objective_denominator=288.)
        selected = np.array([True, False])
        report = score_cohort(data, models, peaks, selected, np.array([.5, 1.]), np.ones(2, bool), ph)
        r = report['models']['crk']
        self.assertAlmostEqual(r['local_design_SSE'], 144 * .04)
        self.assertAlmostEqual(r['HT_design_SSE_numerator'], 288 * .04)
        self.assertAlmostEqual(r['local_original_objective_contribution'], .02)
        self.assertAlmostEqual(r['HT_original_objective_contribution'], .04)
        self.assertAlmostEqual(r['HT_MSE_full_FIT_cohort_denominator'], .04)
        self.assertAlmostEqual(r['local_design_RMSE'], .2)

    def test_nonfinite_public_json_is_rejected(self):
        with self.assertRaises(ValueError):
            strict_json(dict(value=float('nan')))

    def test_registered_two_quantile_conventions_are_explicit(self):
        self.assertEqual(distribution([0, 1], [1, 1])['q10_median_q90'][1], 0.)
        self.assertEqual(distribution([0, 1], [1, 1], midpoint=True)['q10_median_q90'][1], .5)

    def test_pair_reproduction_uses_common_observations_without_reselection(self):
        y, m, y0 = self.arrays(2)
        y[0, :4] = .2
        y[1, :4] = .1
        m[0, 0] = False
        ph = phenotype(y, m, y0)
        models = {name: y.copy() for name in ('host', 'crk', 'crk_closed', 'weather_control_0p001')}
        peaks = {name: model_peak(P, m, ph['true_peak'], ph['true_peak_hour']) for name, P in models.items()}
        records = [dict(status='selected', kind='near_weather_far_geo', anchor_unit=3, partner_unit=9,
                        weather_distance=.1, geography_distance=2),
                   dict(status='no_qualified_input_partner', kind='near_geo_far_weather', anchor_unit=3)]
        report, arr = pair_descriptives(records, np.array([3, 9]), np.ones(2, bool),
                                       np.ones(2), dict(y=y, m=m, merged_group=np.array(['a', 'b'])),
                                       models, peaks, ph)
        self.assertEqual(report['accepted_pairs'], 1)
        self.assertEqual(arr['common_observed_hours'][0], 143)
        self.assertEqual(arr['crk_contrast_reproduction_RMSE'][0], 0.)
        self.assertEqual(arr['true_peak_difference'][0], .1)
        self.assertEqual(report['rows']['near_geo_far_weather']['accepted_pairs'], 0)
        with self.assertRaises(ValueError):
            pair_descriptives(records, np.array([3, 9]), np.ones(2, bool), np.array([1., .5]),
                              dict(y=y, m=m, merged_group=np.array(['a', 'b'])), models, peaks, ph)


if __name__ == '__main__':
    unittest.main()
