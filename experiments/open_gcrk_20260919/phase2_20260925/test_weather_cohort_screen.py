import json
import unittest

import pandas as pd

import weather_cohort_screen as screen


class WeatherCohortScreenTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rule = json.loads(screen.DEFAULT_RULE.read_text())
        cls.protocol = json.loads(screen.DEFAULT_PROTOCOL.read_text())

    def test_weather_screened_window_is_excluded_not_only_exact_anchor(self):
        windows = screen.excluded_windows(self.protocol, self.rule)
        candidate = screen.window(pd.Timestamp("2024-11-21"), self.rule)
        hits = [event_id for start, stop, role, event_id in windows
                if role == "weather_metadata_screened"
                and screen.overlaps(candidate, (start, stop))]
        self.assertIn("2024-11-20", hits)

    def test_metrics_and_hard_filters(self):
        anchor = pd.Timestamp("2025-05-15")
        rows = []
        for index in range(100):
            rows.append({
                "EVENT_TYPE": "Thunderstorm Wind",
                "t_begin_utc": pd.Timestamp("2025-05-15") + pd.Timedelta(hours=index % 30),
                "fips": f"{index:05d}",
                "STATE": f"STATE{index % 5}",
            })
        metric = screen.family_metrics(
            pd.DataFrame(rows), anchor, "convective", {"Thunderstorm Wind"}, self.rule,
        )
        self.assertIsNotNone(metric)
        self.assertEqual(metric["forecast_family_counties"], 100)
        self.assertEqual(metric["forecast_family_states"], 5)
        self.assertTrue(screen.qualifies(metric, self.rule))

    def test_release_boundary_rejects_window_after_2022_archive_end(self):
        start, stop = screen.window(pd.Timestamp("2022-11-10"), self.rule)
        self.assertFalse(screen.release_contains(start, stop, self.rule))

    def test_tie_break_is_deterministic_and_windows_overlap(self):
        rows = [
            {"event_id": "2025-05-15", "forecast_family_counties": 120,
             "forecast_family_states": 5},
            {"event_id": "2025-05-16", "forecast_family_counties": 120,
             "forecast_family_states": 5},
        ]
        ranked = sorted(rows, key=lambda row: (
            -row["forecast_family_counties"], -row["forecast_family_states"],
            row["event_id"],
        ))
        self.assertEqual(ranked[0]["event_id"], "2025-05-15")
        self.assertTrue(screen.overlaps(
            screen.window(pd.Timestamp(ranked[0]["event_id"]), self.rule),
            screen.window(pd.Timestamp(ranked[1]["event_id"]), self.rule),
        ))


if __name__ == "__main__":
    unittest.main()
