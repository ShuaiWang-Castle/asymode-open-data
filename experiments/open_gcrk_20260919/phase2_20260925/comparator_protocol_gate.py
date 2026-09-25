#!/usr/bin/env python3
"""Reject mixed or post-selected process-model comparison bundles.

The gate checks metadata only. It never certifies that a cohort is truly fresh,
that a file exists, or that a scientific claim is correct. Those facts require
the separately hashed source manifests and human review documented in the
protocol. A passing result means only that the supplied run bundle is internally
consistent with COMPARATOR_PROTOCOL.json.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = HERE / "COMPARATOR_PROTOCOL.json"


class ProtocolError(ValueError):
    pass


def validate_protocol(protocol: dict) -> None:
    budget = protocol["host_budget_audit"]
    expected_cin = int(budget["context_dimension"]) * int(budget["damage_hidden_width"])
    if int(budget["w_cin_new_parameters"]) != expected_cin:
        raise ProtocolError("frozen W+Cin parameter count does not match the source formula")
    d = int(budget["damage_hidden_width"])
    r = int(budget["gcrk_code_dimension"])
    g = int(protocol["historical_exploratory_identity"]["d_geo"])
    expected_gcrk = r * g + 3 * d * r + 2 * d + 2
    if int(budget["gcrk_new_parameters_at_geo40"]) != expected_gcrk:
        raise ProtocolError("frozen GCRK parameter count does not match the source formula")


def _need(mapping: dict, fields: list[str], where: str) -> None:
    missing = [k for k in fields if k not in mapping or mapping[k] in (None, "")]
    if missing:
        raise ProtocolError(f"{where}: missing required fields {missing}")


def validate_bundle(bundle: dict, protocol: dict) -> dict:
    validate_protocol(protocol)
    if bundle.get("schema_version") != protocol["schema_version"]:
        raise ProtocolError("schema_version does not match the frozen protocol")
    phase = bundle.get("phase")
    if phase not in protocol["phases"]:
        raise ProtocolError(f"unknown phase {phase!r}")
    rule = protocol["phases"][phase]
    common = bundle.get("common", {})
    _need(common, protocol["common_identity_fields"], "common")
    if common.get("cohort_status") != rule["allowed_cohort_status"]:
        raise ProtocolError("cohort_status is not allowed for this phase")
    if common.get("scope") != rule["required_scope"]:
        raise ProtocolError("scope is not allowed for this phase")
    if not common.get("preprocessing_frozen_before_run", False):
        raise ProtocolError("preprocessing was not declared frozen before the run")
    if not common.get("architecture_frozen_before_run", False):
        raise ProtocolError("architecture was not declared frozen before the run")
    if common.get("outer_results_inspected_before_lock", True):
        raise ProtocolError("bundle declares OUTER inspection before lock")
    expected_folds = common.get("expected_fold_ids")
    if not isinstance(expected_folds, list) or not expected_folds or len(expected_folds) != len(set(expected_folds)):
        raise ProtocolError("expected_fold_ids must be a nonempty unique list")

    old = protocol["historical_exploratory_identity"]
    if phase == "pilot":
        for key in ("cohort_id", "panel_sha256", "split_sha256", "weather_information_set"):
            if common[key] != old[key]:
                raise ProtocolError(f"pilot {key} differs from the audited historical identity")
        if bundle.get("outer_artifacts"):
            raise ProtocolError("pilot bundles must not contain OUTER artifacts")
    else:
        if common["panel_sha256"] == old["panel_sha256"]:
            raise ProtocolError("confirmatory bundle reuses the inspected historical panel")
        if common["split_sha256"] == old["split_sha256"]:
            raise ProtocolError("confirmatory bundle reuses the inspected historical split")
        if not common.get("fresh_selection_rule_sha256"):
            raise ProtocolError("confirmatory bundle lacks the locked fresh-event selection rule hash")

    cells = bundle.get("cells", [])
    if not cells:
        raise ProtocolError("bundle contains no cells")
    for i, cell in enumerate(cells):
        _need(cell, protocol["required_cell_fields"], f"cell[{i}]")
        if cell["arm"] not in protocol["required_arms"]:
            raise ProtocolError(f"cell[{i}] has unregistered arm {cell['arm']!r}")
        if cell["new_parameter_count"] < 0:
            raise ProtocolError(f"cell[{i}] has a negative parameter count")
        for key in protocol["common_identity_fields"]:
            if key in cell and cell[key] != common[key]:
                raise ProtocolError(f"cell[{i}] overrides common identity field {key}")

    arms = set(protocol["required_arms"])
    present = {c["arm"] for c in cells}
    if present != arms:
        raise ProtocolError(f"arm set mismatch: missing={sorted(arms-present)}, extra={sorted(present-arms)}")
    cell_sets = {}
    for arm in sorted(arms):
        triples = [(int(c["seed"]), str(c["fold"])) for c in cells if c["arm"] == arm]
        if len(triples) != len(set(triples)):
            raise ProtocolError(f"duplicate seed/fold cell in arm {arm}")
        cell_sets[arm] = set(triples)
    reference = cell_sets[protocol["required_arms"][0]]
    for arm, values in cell_sets.items():
        if values != reference:
            raise ProtocolError(f"seed/fold cells for {arm} do not match the other arms")
    seeds = {s for s, _ in reference}
    folds = {f for _, f in reference}
    if folds != {str(f) for f in expected_folds}:
        raise ProtocolError(f"fold set {sorted(folds)} does not equal expected_fold_ids {sorted(map(str, expected_folds))}")
    expected_seeds = set(rule.get("allowed_seeds", rule.get("required_seeds", [])))
    if seeds != expected_seeds:
        raise ProtocolError(f"seed set {sorted(seeds)} does not equal {sorted(expected_seeds)}")

    by_arm = {arm: [c for c in cells if c["arm"] == arm] for arm in arms}
    for group, members in protocol["matched_capacity_groups"].items():
        for seed_fold in sorted(reference):
            rows = [next(c for c in by_arm[arm] if (int(c["seed"]), str(c["fold"])) == seed_fold)
                    for arm in members]
            counts = {int(c["new_parameter_count"]) for c in rows}
            if len(counts) != 1:
                raise ProtocolError(f"{group} parameter counts differ at seed/fold {seed_fold}: {sorted(counts)}")
            pair_ids = {c["initialization_pair_id"] for c in rows}
            if len(pair_ids) != 1:
                raise ProtocolError(f"{group} initialization_pair_id differs at seed/fold {seed_fold}")
            stopping = {c["stopping_rule_id"] for c in rows}
            if len(stopping) != 1:
                raise ProtocolError(f"{group} stopping rules differ at seed/fold {seed_fold}")
            inputs = {c["input_view_sha256"] for c in rows}
            if len(inputs) != 1:
                raise ProtocolError(f"{group} input views differ at seed/fold {seed_fold}")
            optimizers = {c["optimizer_group_id"] for c in rows}
            if len(optimizers) != 1:
                raise ProtocolError(f"{group} optimizer groups differ at seed/fold {seed_fold}")

    return {
        "valid": True,
        "phase": phase,
        "n_cells": len(cells),
        "arms": sorted(arms),
        "seeds": sorted(seeds),
        "folds": sorted(folds),
        "cohort_id": common["cohort_id"],
        "panel_sha256": common["panel_sha256"]
    }


def _example_bundle(protocol: dict, phase: str = "pilot") -> dict:
    old = protocol["historical_exploratory_identity"]
    common = {
        "cohort_id": old["cohort_id"],
        "cohort_status": "historical_exploratory",
        "cohort_manifest_sha256": "1" * 64,
        "panel_sha256": old["panel_sha256"],
        "split_sha256": old["split_sha256"],
        "weather_information_set": old["weather_information_set"],
        "feature_name_sha256": "2" * 64,
        "observation_support_sha256": "3" * 64,
        "normalization_spec_sha256": "4" * 64,
        "host_source_sha256": protocol["locked_source_hashes"]["asym_host.py"],
        "training_protocol_sha256": protocol["locked_source_hashes"]["gcrk_train.py"],
        "residual_amplitude_cap": 0.25,
        "expected_fold_ids": ["dev1"],
        "scope": "inner_fit_only",
        "preprocessing_frozen_before_run": True,
        "architecture_frozen_before_run": True,
        "outer_results_inspected_before_lock": False
    }
    seeds = [0]
    if phase == "confirmatory":
        common.update(cohort_id="fresh-example", cohort_status=protocol["phases"][phase]["allowed_cohort_status"],
                      panel_sha256="5" * 64, split_sha256="6" * 64,
                      fresh_selection_rule_sha256="7" * 64, scope="outer_once_after_lock")
        seeds = [0, 1, 2, 3, 4]
    cells = []
    for arm in protocol["required_arms"]:
        for seed in seeds:
            count = 300 if arm in protocol["matched_capacity_groups"]["path_state"] else 192
            cells.append({"arm": arm, "seed": seed, "fold": "dev1", "new_parameter_count": count,
                          "candidate_source_sha256": "8" * 64,
                          "input_view_sha256": "a" * 64,
                          "initialization_pair_id": f"seed{seed}-dev1",
                          "optimizer_group_id": "host-3e-3-recovery-3e-4-v1",
                          "stopping_rule_id": "inner-pooled-v1"})
    return {"schema_version": protocol["schema_version"], "phase": phase, "common": common,
            "cells": cells, "outer_artifacts": []}


def self_test(protocol: dict) -> None:
    validate_protocol(protocol)
    validate_bundle(_example_bundle(protocol, "pilot"), protocol)
    validate_bundle(_example_bundle(protocol, "confirmatory"), protocol)
    cases = []
    b = _example_bundle(protocol); b["outer_artifacts"] = ["outer.npz"]; cases.append(b)
    b = _example_bundle(protocol); b["cells"].pop(); cases.append(b)
    b = _example_bundle(protocol); b["cells"][-1]["new_parameter_count"] += 1; cases.append(b)
    b = _example_bundle(protocol); b["cells"][-1]["initialization_pair_id"] = "unpaired"; cases.append(b)
    b = _example_bundle(protocol); b["cells"][-1]["input_view_sha256"] = "b" * 64; cases.append(b)
    b = _example_bundle(protocol); b["common"]["expected_fold_ids"] = ["dev1", "missing"]; cases.append(b)
    b = _example_bundle(protocol, "confirmatory"); b["common"]["panel_sha256"] = protocol["historical_exploratory_identity"]["panel_sha256"]; cases.append(b)
    b = _example_bundle(protocol); b["cells"][0]["panel_sha256"] = "9" * 64; cases.append(b)
    for i, bad in enumerate(cases):
        try:
            validate_bundle(bad, protocol)
        except ProtocolError:
            continue
        raise AssertionError(f"negative self-test {i} did not fail")
    bad_protocol = copy.deepcopy(protocol)
    bad_protocol["host_budget_audit"]["gcrk_new_parameters_at_geo40"] += 1
    try:
        validate_protocol(bad_protocol)
    except ProtocolError:
        pass
    else:
        raise AssertionError("corrupted source-derived parameter count did not fail")
    print(f"self-test passed: 2 valid bundles accepted; {len(cases)} mixed/post-selected bundles rejected")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle", nargs="?")
    ap.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    protocol = json.loads(Path(args.protocol).read_text())
    if args.self_test:
        self_test(protocol)
        return
    if not args.bundle:
        ap.error("bundle is required unless --self-test is used")
    result = validate_bundle(json.loads(Path(args.bundle).read_text()), protocol)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
