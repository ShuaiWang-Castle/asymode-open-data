import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import fresh_cohort_gate as gate


class FreshCohortGateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads(gate.DEFAULT_PROTOCOL.read_text())

    def valid(self):
        return gate._valid_example()

    def test_tracked_sources_match_protocol(self):
        self.assertEqual(gate.verify_tracked_sources(self.protocol), [])

    def test_valid_manifest_passes(self):
        self.assertEqual(gate.validate_candidate(self.valid(), self.protocol), [])

    def test_rejects_outcome_inspected_event(self):
        candidate = self.valid()
        candidate["events"][0] = {
            "event_id": "2024-05-08",
            "window_start_utc": "2024-05-05T00:00:00Z",
            "window_end_utc": "2024-05-13T23:00:00Z"
        }
        errors = gate.validate_candidate(candidate, self.protocol)
        self.assertTrue(any("outcome-inspected" in error for error in errors))

    def test_rejects_prior_weather_screen(self):
        candidate = self.valid()
        candidate["events"][0] = {
            "event_id": "2024-11-20",
            "window_start_utc": "2024-11-17T00:00:00Z",
            "window_end_utc": "2024-11-25T23:00:00Z"
        }
        errors = gate.validate_candidate(candidate, self.protocol)
        self.assertTrue(any("weather-screened" in error for error in errors))

    def test_rejects_historical_window_overlap_with_different_anchor(self):
        candidate = self.valid()
        candidate["events"][0] = {
            "event_id": "2024-05-10",
            "window_start_utc": "2024-05-07T00:00:00Z",
            "window_end_utc": "2024-05-15T23:00:00Z"
        }
        errors = gate.validate_candidate(candidate, self.protocol)
        self.assertTrue(any("overlaps historical" in error for error in errors))

    def test_rejects_prior_weather_screen_window_overlap_with_different_anchor(self):
        candidate = self.valid()
        candidate["events"][0] = {
            "event_id": "2024-11-21",
            "window_start_utc": "2024-11-18T00:00:00Z",
            "window_end_utc": "2024-11-26T23:00:00Z"
        }
        errors = gate.validate_candidate(candidate, self.protocol)
        self.assertTrue(any("overlaps prior weather screen" in error for error in errors))

    def test_rejects_outcome_field(self):
        candidate = self.valid()
        candidate["selection_fields_used"].append("outage_peak")
        errors = gate.validate_candidate(candidate, self.protocol)
        self.assertTrue(any("allow-list" in error for error in errors))

    def test_rejects_outcome_access_before_lock(self):
        candidate = self.valid()
        candidate["outcome_access_before_lock"] = ["targets.parquet"]
        errors = gate.validate_candidate(candidate, self.protocol)
        self.assertTrue(any("explicit empty" in error for error in errors))

    def test_rejects_result_source_path_even_with_allowed_role(self):
        candidate = self.valid()
        candidate["source_files"][0]["path"] = "results/weather.csv"
        errors = gate.validate_candidate(candidate, self.protocol)
        self.assertTrue(any("forbidden tokens" in error for error in errors))

    def test_rejects_overlap_between_candidates(self):
        candidate = self.valid()
        candidate["events"][1] = {
            "event_id": "2025-02-21",
            "window_start_utc": "2025-02-18T00:00:00Z",
            "window_end_utc": "2025-02-26T23:00:00Z"
        }
        errors = gate.validate_candidate(candidate, self.protocol)
        self.assertTrue(any("candidate windows overlap" in error for error in errors))

    def test_rejects_unmatched_seed_set(self):
        candidate = self.valid()
        candidate["seeds"] = [0]
        errors = gate.validate_candidate(candidate, self.protocol)
        self.assertTrue(any("seeds must equal" in error for error in errors))

    def test_candidate_source_bytes_are_verified(self):
        candidate = self.valid()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "noaa.csv"
            path.write_bytes(b"weather-only\n")
            candidate["source_files"][0]["path"] = str(path)
            candidate["source_files"][0]["sha256"] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            self.assertEqual(gate.verify_candidate_sources(candidate), [])
            candidate["source_files"][0]["sha256"] = "0" * 64
            errors = gate.verify_candidate_sources(candidate)
            self.assertTrue(any("hash mismatch" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
