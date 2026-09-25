import copy
import json
import unittest
from pathlib import Path

from comparator_protocol_gate import ProtocolError, _example_bundle, validate_bundle, validate_protocol


HERE = Path(__file__).resolve().parent
PROTOCOL = json.loads((HERE / "COMPARATOR_PROTOCOL.json").read_text())


class ComparatorProtocolGateTests(unittest.TestCase):
    def test_pilot_and_fresh_confirmation_pass(self):
        self.assertTrue(validate_bundle(_example_bundle(PROTOCOL, "pilot"), PROTOCOL)["valid"])
        self.assertTrue(validate_bundle(_example_bundle(PROTOCOL, "confirmatory"), PROTOCOL)["valid"])

    def test_pilot_refuses_outer_artifact(self):
        b = _example_bundle(PROTOCOL, "pilot")
        b["outer_artifacts"] = ["outer.npz"]
        with self.assertRaises(ProtocolError):
            validate_bundle(b, PROTOCOL)

    def test_confirmation_refuses_historical_panel(self):
        b = _example_bundle(PROTOCOL, "confirmatory")
        b["common"]["panel_sha256"] = PROTOCOL["historical_exploratory_identity"]["panel_sha256"]
        with self.assertRaises(ProtocolError):
            validate_bundle(b, PROTOCOL)

    def test_refuses_missing_arm_cell(self):
        b = _example_bundle(PROTOCOL, "pilot")
        b["cells"].pop()
        with self.assertRaises(ProtocolError):
            validate_bundle(b, PROTOCOL)

    def test_refuses_unmatched_path_state_capacity(self):
        b = _example_bundle(PROTOCOL, "pilot")
        row = next(c for c in b["cells"] if c["arm"] == "structured_process_graph")
        row["new_parameter_count"] += 1
        with self.assertRaises(ProtocolError):
            validate_bundle(b, PROTOCOL)

    def test_refuses_cell_identity_override(self):
        b = _example_bundle(PROTOCOL, "pilot")
        b["cells"][0]["weather_information_set"] = "forecast"
        with self.assertRaises(ProtocolError):
            validate_bundle(b, PROTOCOL)

    def test_refuses_unmatched_input_view(self):
        b = _example_bundle(PROTOCOL, "pilot")
        row = next(c for c in b["cells"] if c["arm"] == "structured_process_graph")
        row["input_view_sha256"] = "f" * 64
        with self.assertRaises(ProtocolError):
            validate_bundle(b, PROTOCOL)

    def test_refuses_missing_expected_fold(self):
        b = _example_bundle(PROTOCOL, "pilot")
        b["common"]["expected_fold_ids"] = ["dev1", "dev2"]
        with self.assertRaises(ProtocolError):
            validate_bundle(b, PROTOCOL)

    def test_source_derived_parameter_counts_are_locked(self):
        validate_protocol(PROTOCOL)
        p = copy.deepcopy(PROTOCOL)
        p["host_budget_audit"]["w_cin_new_parameters"] += 1
        with self.assertRaises(ProtocolError):
            validate_protocol(p)


if __name__ == "__main__":
    unittest.main()
