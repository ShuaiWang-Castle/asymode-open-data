"""Synthetic falsification checks for process_graph_prototype.py.

Run: python -m unittest -v test_process_graph_prototype.py
No real outages, climate data, checkpoints, or third-party test runner required.
"""

import unittest

import numpy as np

from process_graph_prototype import Geography, baseline_path, rollout


HIGH = Geography(poor_drainage=0.9, canopy_proxy=0.9, terrain_exposure=0.9)
LOW = Geography(poor_drainage=0.1, canopy_proxy=0.1, terrain_exposure=0.1)


class ProcessGraphPrototypeTests(unittest.TestCase):
    def test_states_rates_and_customer_stock_are_bounded_under_random_extremes(self):
        rng = np.random.default_rng(3701)
        for _ in range(50):
            n = 144
            rain = rng.exponential(15.0, n)
            gust = rng.uniform(0.0, 65.0, n)
            rates_u = rng.uniform(0.0, 1.0, n)
            rates_r = rng.uniform(0.0, 1.0, n)
            geography = Geography(*rng.uniform(0.0, 1.0, 3))
            path = rollout(
                rain, gust, geography, rates_u, rates_r,
                p0=rng.uniform(), wet0=rng.uniform(), fatigue0=rng.uniform(),
                alpha=rng.uniform(),
            )
            for values in (
                path.p, path.damage, path.recovery,
                path.wet_before, path.wet_after,
                path.fatigue_before, path.fatigue_after,
            ):
                self.assertTrue(np.isfinite(values).all())
                self.assertGreaterEqual(values.min(), -1e-14)
                self.assertLessEqual(values.max(), 1.0 + 1e-14)
            self.assertTrue((np.abs(path.residual) <= 1.0).all())

    def test_ordered_rain_and_wind_change_later_damage_with_identical_totals(self):
        base_u, base_r = np.full(4, 0.04), np.full(4, 0.03)
        rain_first = rollout([15, 0, 0, 0], [0, 25, 0, 0], HIGH, base_u, base_r)
        wind_first = rollout([0, 15, 0, 0], [25, 0, 0, 0], HIGH, base_u, base_r)
        self.assertGreater(rain_first.wet_before[1], wind_first.wet_before[0])
        # Compare each ordering at its strong-wind hour, independent of phase.
        self.assertGreater(rain_first.damage[1], wind_first.damage[0])
        self.assertGreater(rain_first.p[-1], wind_first.p[-1])
        self.assertAlmostEqual(sum([15, 0, 0, 0]), sum([0, 15, 0, 0]))

    def test_geography_swap_is_reversible_without_county_identity(self):
        rain, gust = [16, 0, 0, 0], [0, 27, 0, 0]
        base_u, base_r = np.full(4, 0.06), np.full(4, 0.02)
        high = rollout(rain, gust, HIGH, base_u, base_r)
        low = rollout(rain, gust, LOW, base_u, base_r)
        high_again = rollout(rain, gust, HIGH, base_u, base_r)
        self.assertGreater(high.wet_before[1], low.wet_before[1])
        self.assertGreater(high.damage[1], low.damage[1])
        self.assertGreater(high.p[-1], low.p[-1])
        np.testing.assert_array_equal(high.p, high_again.p)
        np.testing.assert_array_equal(high.residual, high_again.residual)

    def test_closed_residual_is_exactly_the_supplied_w_cin_baseline(self):
        rng = np.random.default_rng(927)
        n = 80
        rain, gust = rng.exponential(7, n), rng.uniform(0, 48, n)
        u, r = rng.uniform(0, 0.6, n), rng.uniform(0, 0.5, n)
        reference = baseline_path(u, r, p0=0.23)
        for geo in (HIGH, LOW):
            closed = rollout(rain, gust, geo, u, r, p0=0.23, residual_open=False)
            zero_scale = rollout(rain, gust, geo, u, r, p0=0.23, alpha=0.0)
            np.testing.assert_array_equal(closed.p, reference)
            np.testing.assert_array_equal(zero_scale.p, reference)
            np.testing.assert_array_equal(closed.damage, u)
            np.testing.assert_array_equal(zero_scale.damage, u)
            np.testing.assert_array_equal(closed.residual, np.zeros(n))
            np.testing.assert_array_equal(zero_scale.residual, np.zeros(n))

    def test_no_future_rain_influences_earlier_predictions(self):
        base_u, base_r = np.full(5, 0.04), np.full(5, 0.02)
        gust = [0, 27, 0, 0, 0]
        dry = rollout([0, 0, 0, 0, 0], gust, HIGH, base_u, base_r)
        future_rain = rollout([0, 0, 0, 40, 0], gust, HIGH, base_u, base_r)
        np.testing.assert_array_equal(dry.p[:4], future_rain.p[:4])
        np.testing.assert_array_equal(dry.damage[:3], future_rain.damage[:3])
        np.testing.assert_array_equal(dry.wet_before[:3], future_rain.wet_before[:3])

    def test_synthetic_gradient_has_expected_sign_before_wind(self):
        base_u, base_r = np.full(4, 0.04), np.full(4, 0.03)
        def forecast(first_hour_rain):
            return rollout([first_hour_rain, 0, 0, 0], [0, 25, 0, 0],
                           HIGH, base_u, base_r).p[-1]
        step = 1e-4
        slope = (forecast(15.0 + step) - forecast(15.0 - step)) / (2 * step)
        self.assertTrue(np.isfinite(slope))
        self.assertGreater(slope, 0.0)

    def test_invalid_probability_or_weather_input_fails_loudly(self):
        with self.assertRaises(ValueError):
            Geography(1.1, 0.2, 0.2)
        with self.assertRaises(ValueError):
            rollout([0, -1], [20, 20], HIGH, [0.1, 0.1], [0.1, 0.1])
        with self.assertRaises(ValueError):
            rollout([0], [20], HIGH, [1.1], [0.1])
        with self.assertRaises(ValueError):
            rollout([0], [20, 30], HIGH, [0.1], [0.1])


if __name__ == "__main__":
    unittest.main()
