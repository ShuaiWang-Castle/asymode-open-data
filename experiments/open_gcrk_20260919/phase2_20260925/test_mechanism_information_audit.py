import unittest

from mechanism_information_audit import run_audit


class MechanismInformationAuditTests(unittest.TestCase):
    def test_response_hour_features_collide_but_both_memories_distinguish(self):
        result = run_audit()
        self.assertEqual(result["r2_damage_feature_dimension"], 42)
        self.assertEqual(result["max_abs_r2_damage_feature_difference_at_wind"], 0.0)
        self.assertGreater(result["bounded_wetness_absolute_difference"], 1e-3)
        self.assertGreater(result["host_scalar_smoother_capacity_witness"]["absolute_difference"], 1e-3)

    def test_pair_matches_current_weather_and_total_rain(self):
        result = run_audit()
        self.assertTrue(result["same_total_rain"])
        self.assertTrue(result["same_current_weather_at_wind"])


if __name__ == "__main__":
    unittest.main()
