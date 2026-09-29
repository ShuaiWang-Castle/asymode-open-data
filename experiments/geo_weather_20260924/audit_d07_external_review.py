"""Independently audit the external D07 review using frozen public summaries.

This program does not extract or execute any code in the supplied archive. It
does not load outcome arrays, checkpoints, model modules, or gradient vectors.
An output receipt must be new; existing receipts are never overwritten.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import random
import zipfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE / "results" / "v1"
PREFIX = "D07_INDEPENDENT_REVIEW_20260929/"
REPORT_BLOB = "3d98dc9818f07747a439088ae32249ef365429a7"
EXPECTED_COMMIT = "34288910cf32769029d7e6c0d764bc9b05a9ac17"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def close(actual: float, expected: float) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=2e-9, abs_tol=2e-8):
        raise AssertionError((actual, expected))


def norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def rotate(v: list[float], angle: float) -> list[float]:
    c, s = math.cos(angle), math.sin(angle)
    return [c * v[0] - s * v[1], s * v[0] + c * v[1]]


def pure_math_checks() -> dict:
    """Synthetic inequalities, separate from real-data/model validation."""
    rng = random.Random(2907)
    maximum_violation = 0.0
    maximum_aggregate_norm = 0.0
    for _ in range(500):
        aggregate_next = []
        for _mode in range(4):
            vec = lambda: rotate([rng.random(), 0.0], rng.uniform(-math.pi, math.pi))
            x, xp, d, dp = vec(), vec(), vec(), vec()
            rho, rhop = rng.uniform(0.0, 0.95), rng.uniform(0.0, 0.95)
            angle, anglep = rng.uniform(-math.pi, math.pi), rng.uniform(-math.pi, math.pi)
            qx, qpxp = rotate(x, angle), rotate(xp, anglep)
            nxt = [rho * qx[j] + (1.0 - rho) * d[j] for j in range(2)]
            nxtp = [rhop * qpxp[j] + (1.0 - rhop) * dp[j] for j in range(2)]
            # In 2D the spectral norm of the rotation difference is exact.
            qdiff = 2.0 * abs(math.sin((angle - anglep) / 2.0))
            rhs = (0.95 * norm([x[j] - xp[j] for j in range(2)])
                   + 2.0 * abs(rho - rhop) + 0.95 * qdiff
                   + norm([d[j] - dp[j] for j in range(2)]))
            maximum_violation = max(maximum_violation, norm([nxt[j] - nxtp[j] for j in range(2)]) - rhs)
            assert norm(nxt) <= 1.0 + 1e-12
            aggregate_next.extend(nxt)
        maximum_aggregate_norm = max(maximum_aggregate_norm, norm(aggregate_next))
        assert norm(aggregate_next) <= 2.0 + 1e-12
    # The four-mode aggregate reaches 2, not 1, when all four unit modes align.
    close(norm([1.0, 0.0] * 4), 2.0)
    # Four common retention changes can require the coefficient 4 rather than 2.
    # x=-d and equal rotations attain ||Delta X_next||=4*|Delta rho|.
    r0, r1 = 0.2, 0.3
    old = [(2.0 * r0 - 1.0), 0.0] * 4
    new = [(2.0 * r1 - 1.0), 0.0] * 4
    aggregate_change = norm([a - b for a, b in zip(old, new)])
    close(aggregate_change, 4.0 * abs(r1 - r0))
    assert aggregate_change > 2.0 * abs(r1 - r0)
    # A negative gradient cosine does not exclude a common descent direction.
    gs, gn, direction = [1.0, 0.0], [-0.5, math.sqrt(0.75)], [-0.5, -math.sqrt(0.75)]
    cosine = sum(a * b for a, b in zip(gs, gn)) / (norm(gs) * norm(gn))
    dots = [sum(a * b for a, b in zip(g, direction)) for g in (gs, gn)]
    close(cosine, -0.5)
    assert max(dots) < 0.0
    return dict(synthetic=True, trials=500, modes=4,
                per_mode_incremental_bound_max_violation=maximum_violation,
                four_mode_concat_bound=2.0,
                maximum_random_four_mode_norm=maximum_aggregate_norm,
                common_retention_change_aggregate_coefficient=4.0,
                negative_cosine=cosine, common_descent_directional_derivatives=dots,
                not_real_model_forward=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-zip", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, default=RESULTS / "d07_external_review_receipt.json")
    args = parser.parse_args()
    if args.receipt.exists():
        raise FileExistsError("Refusing to overwrite an existing audit receipt.")

    def load(name: str) -> dict:
        # The fixed public commit stores the large summaries as gzip archives.
        # Prefer those checked-in bytes; local expanded copies are optional.
        compressed = RESULTS / (name + ".gz")
        if compressed.exists():
            return json.loads(gzip.decompress(compressed.read_bytes()))
        return json.loads((RESULTS / name).read_bytes())
    evidence, attr, grad, audit = map(load, ("d07_evidence.json", "d07_attribution.json", "d07_gradients.json", "d07_final_audit.json"))
    checked_files = []
    with zipfile.ZipFile(args.review_zip) as archive:
        read = lambda name: archive.read(PREFIX + name)
        manifest = json.loads(read("MANIFEST.json"))
        assert len(manifest) == 9
        for name, digest in manifest.items():
            assert sha(read(name)) == digest
        fixed_files = json.loads(read("recomputed/verified_manifest.json"))
        assert len(fixed_files) == 18
        for record in fixed_files:
            relative = Path(record["file"])
            assert not relative.is_absolute() and ".." not in relative.parts
            assert str(relative).startswith("experiments/geo_weather_20260924/") or str(relative) == "tests/test_d07_gradients.py"
            raw = (ROOT / relative).read_bytes()
            assert len(raw) == record["bytes"] and sha(raw) == record["sha256"]
            checked_files.append(record)
        external_audit = json.loads(read("recomputed/independent_audit.json"))
        assert external_audit["scope_commit"] == EXPECTED_COMMIT
        tables = {
            name: list(csv.DictReader(io.StringIO(read("recomputed/" + name).decode("utf-8"))))
            for name in ("oracle_retrospective_only.csv", "finite_intervention_effects.csv", "gradient_scale_and_conflict.csv", "severity_duration_partition.csv", "S_regime_and_fold.csv")
        }

    report = (HERE / "notes" / "D07_REAL_DATA_SELECTIVITY_RESULTS_20260929_ZH.md").read_bytes()
    git_blob = hashlib.sha1(f"blob {len(report)}\0".encode() + report).hexdigest()
    assert git_blob == REPORT_BLOB == external_audit["report_git_blob"]
    gzip_checks = []
    assert len(audit["packages"]) == 3
    for name, record in audit["packages"].items():
        raw = (RESULTS / name).read_bytes()
        plain = gzip.decompress(raw)
        assert sha(raw) == record["gzip_sha256"]
        assert sha(plain) == record["uncompressed_sha256"]
        assert json.loads(plain) == load(name.removesuffix(".gz"))
        expanded = RESULTS / name.removesuffix(".gz")
        if expanded.exists():
            assert json.loads(plain) == json.loads(expanded.read_bytes())
        gzip_checks.append(dict(file=name, gzip_sha256=sha(raw), uncompressed_sha256=sha(plain)))

    identity_errors = {key: 0.0 for key in ("alignment_minus_energy", "positive_minus_negative", "rmse_sse", "unit_count")}
    columns = attr["stratified_table"]["columns"]
    strata = [dict(zip(columns, row)) for row in attr["stratified_table"]["rows"]]
    assert len(strata) == 3792
    positive_support_rows = 0
    for row in strata:
        derived = {
            "alignment_minus_energy": row["alignment"] - row["modification_energy"],
            "positive_minus_negative": row["positive_gain"] - row["negative_loss"],
        }
        if row["weighted_observed_hour_mass"] > 0:
            derived["rmse_sse"] = (row["base_rmse"] ** 2 - row["candidate_rmse"] ** 2) * row["weighted_observed_hour_mass"]
            positive_support_rows += 1
        for key, value in derived.items():
            close(value, row["net_gain"])
            identity_errors[key] = max(identity_errors[key], abs(value - row["net_gain"]))
        count_error = row["positive_units"] + row["negative_units"] + row["zero_units"] - row["units"]
        assert count_error == 0

    def check_csv(rows: list[dict], expected_fn) -> float:
        maximum = 0.0
        for row in rows:
            for key, value in expected_fn(row).items():
                actual = float(row[key])
                close(actual, value)
                maximum = max(maximum, abs(actual - value))
        return maximum

    def oracle_values(row: dict) -> dict:
        value = evidence["attribution"][row["scope"]][row["support"]][row["cohort"]]
        loss = value["base_RMSE"] ** 2 * value["support"]["weighted_observed_hour_mass"]
        alignment, energy = value["alignment"], value["modification_energy"]
        positive = value["unit_signs"]["positive"]["weighted_gain"]
        best = alignment / (2.0 * energy) if energy else 0.0
        convex = min(1.0, max(0.0, best))
        return dict(base_SSE=loss, alignment=alignment, modification_energy=energy,
                    net_gain=value["net_gain"], positive_unit_gains=positive,
                    best_scalar_unconstrained=best, best_scalar_convex=convex,
                    oracle_scalar_convex_RMSE_improvement=1.0 - math.sqrt(max(0.0, 1.0 - (convex * alignment - convex * convex * energy) / loss)),
                    oracle_binary_whole_unit_RMSE_improvement=1.0 - math.sqrt(max(0.0, 1.0 - positive / loss)))

    interventions = {value["name"]: value for value in grad["parameter_interventions"]}

    def intervention_values(row: dict) -> dict:
        arm = interventions[row["intervention"]]
        value = arm["scores"][row["split"]][row["cohort"]]
        return dict(achieved_relative_parameter_step=arm["achieved_relative_step"],
                    RMSE_change=value["design_RMSE"] / value["baseline_design_RMSE"] - 1.0,
                    SSE_change=value["paired_design_SSE_delta"],
                    max_abs_prediction_change=value["prediction_change_max_abs_observed"],
                    objective_delta=value["paired_objective_delta"])

    gradient_values = {}
    for block, value in grad["blocks"].items():
        s, n, cosine = value["S"]["gradient_l2"], value["nonS"]["gradient_l2"], value["S_nonS_cosine"]
        total = math.sqrt(s * s + n * n + 2.0 * s * n * cosine)
        gradient_values[block] = dict(S_norm=s, nonS_norm=n, nonS_to_S=n / s,
                                     cosine_S_nonS=cosine, cosine_total_S=(s + n * cosine) / total,
                                     cosine_total_nonS=(n + s * cosine) / total)
    counts = {"oracle_retrospective_only.csv": 48, "finite_intervention_effects.csv": 36, "gradient_scale_and_conflict.csv": 9}
    csv_errors = {}
    for name, count in counts.items():
        assert len(tables[name]) == count
    csv_errors["oracle"] = check_csv(tables["oracle_retrospective_only.csv"], oracle_values)
    csv_errors["intervention"] = check_csv(tables["finite_intervention_effects.csv"], intervention_values)
    csv_errors["gradient"] = check_csv(tables["gradient_scale_and_conflict.csv"], lambda row: gradient_values[row["block"]])

    core = evidence["attribution"]["oof_independent_host"]["common_observed"]["S"]
    complete = evidence["attribution"]["oof_independent_host"]["complete144"]["S"]
    loss = core["base_RMSE"] ** 2 * core["support"]["weighted_observed_hour_mass"]
    positive = core["unit_signs"]["positive"]["weighted_gain"]
    duration = {}
    for group in ("severe_hours/1", "severe_hours/2..6", "severe_hours/>=7"):
        value = evidence["OOF_S_outcome_phenotypes"]["common_observed"][group]
        duration[group] = dict(units=value["support"]["units"], positive_units=value["unit_signs"]["positive"]["units"],
                               design_unit_share=value["support"]["design_unit_mass"] / core["support"]["design_unit_mass"],
                               baseline_S_SSE_share=value["base_RMSE"] ** 2 * value["support"]["weighted_observed_hour_mass"] / loss,
                               RMSE_change=-value["RMSE_improvement_fraction"], alignment=value["alignment"],
                               modification_energy=value["modification_energy"], alignment_to_energy=value["alignment"] / value["modification_energy"],
                               net_gain=value["net_gain"], positive_design_share_within_group=value["unit_signs"]["positive"]["design_unit_share"])
    assert len(tables["severity_duration_partition.csv"]) == 3
    csv_errors["duration"] = check_csv(tables["severity_duration_partition.csv"], lambda row: duration[row["group"]])
    close(sum(value["net_gain"] for value in duration.values()), core["net_gain"])
    close(sum(value["design_unit_share"] for value in duration.values()), 1.0)
    s_strata = {row["stratum"]: row for row in strata if row["comparison"] == "oof_independent_host" and row["support"] == "common_observed" and row["cohort"] == "S" and row["stratum"].startswith(("regime/", "fold/"))}
    assert len(tables["S_regime_and_fold.csv"]) == len(s_strata) == 10
    for row in tables["S_regime_and_fold.csv"]:
        source = s_strata[row["stratum"]]
        for key, value in source.items():
            if value is None:
                assert row[key] == ""
            elif isinstance(value, (int, float)):
                close(float(row[key]), value)
            else:
                assert row[key] == value

    findings = dict(
        S_base_SSE=loss, S_positive_gains=positive,
        S_negative_losses=-core["unit_signs"]["negative"]["weighted_gain"],
        S_current_net_gain=core["net_gain"],
        S_oracle_binary_whole_unit_improvement=1.0 - math.sqrt(1.0 - positive / loss),
        S_ten_percent_RMSE_requires_SSE_gain=(1.0 - 0.9 ** 2) * loss,
        required_gain_over_existing_positive_gains=(1.0 - 0.9 ** 2) * loss / positive,
        incomplete_S_units=core["support"]["units"] - complete["support"]["units"],
        complete_S_net_gain=complete["net_gain"],
        incomplete_S_net_gain=core["net_gain"] - complete["net_gain"],
        incomplete_S_fraction_of_net_gain=(core["net_gain"] - complete["net_gain"]) / core["net_gain"],
        duration=duration,
        gradient_scale_and_conflict={key: gradient_values[key] for key in ("weather_control", "raw_geography", "kernel_basis")},
        weather_control_0p001_OUTER={key: intervention_values(dict(intervention="weather_control_0p001", split="OUTER", cohort=key)) for key in ("all", "S", "nonS")},
        weather_control_0p001_OUTER_false_alarms=interventions["weather_control_0p001"]["scores"]["OUTER"]["nonS"]["paired_severe_false_alarms"],
    )
    for key in ("S_base_SSE", "S_positive_gains", "S_current_net_gain", "S_oracle_binary_whole_unit_improvement", "S_ten_percent_RMSE_requires_SSE_gain", "required_gain_over_existing_positive_gains", "incomplete_S_units", "incomplete_S_net_gain", "incomplete_S_fraction_of_net_gain"):
        close(findings[key], external_audit[key])

    original_zip = ROOT / "runs" / "geo_weather_20260924" / "d07_selectivity_20260929" / "D07_REAL_DATA_EVIDENCE_20260929.zip"
    receipt = dict(
        passed=True, fixed_D07_commit=EXPECTED_COMMIT,
        external_archive=dict(name=args.review_zip.name, sha256=sha(args.review_zip.read_bytes()), internal_hashes_verified=len(manifest)),
        fixed_local_files_verified=checked_files, report_git_blob=git_blob, gzip_packages=gzip_checks,
        paired_summary_rows_checked=len(strata), positive_support_rows=positive_support_rows,
        maximum_identity_errors=identity_errors,
        external_CSV_rows_checked={**counts, "severity_duration_partition.csv": 3, "S_regime_and_fold.csv": 10},
        external_CSV_maximum_errors=csv_errors, findings=findings,
        pure_math_checks=pure_math_checks(),
        packaging=dict(original_evidence_zip_present_at_audit=original_zip.exists(),
                       issue="Absent original delivery archive requires repackaging; all eighteen frozen scientific files are verified." if not original_zip.exists() else None),
        audit_source_sha256=sha(Path(__file__).read_bytes()),
        execution_boundaries=dict(external_code_executed=False, archive_extracted=False,
                                  raw_outcomes_loaded=False, model_forward_performed=False,
                                  gradients_recomputed=False, new_training_performed=False,
                                  original_nineteen_tests_rerun=False, bootstrap_rerun=False),
        interpretation_bounds=[
            "Binary oracle uses future truths to select one entire frozen W or CRK trajectory per unit, on the same observed support; it does not bound a new model or hourly routing.",
            "The 4.895 factor compares squared-error gains, not RMSE gains.",
            "Prediction-space scalar interpolation is retrospective and is not an equivalent scaling of the model kernel exit.",
            "The partial-observation net-gain fraction diagnoses fragility of a small net gain; it does not establish missingness bias or justify deleting those units.",
            "Severe-hour counts need not be consecutive and are outcome-defined retrospective strata, not inference inputs.",
            "Gradient directions describe the frozen endpoint and current parameter coordinates, not the entire Adam training history.",
            "Equal relative parameter norms do not imply equal functional interventions or comparable geographic-information strength.",
            "The incremental inequality is per unit-norm mode. Four concatenated unit modes have norm bound two, and common retention-change coefficients must be adjusted accordingly.",
            "The synthetic inequalities check algebra only; they do not constitute real-data forecasts or identify a physical geographical mechanism.",
        ],
    )
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(passed=True, paired_rows=len(strata), verified_local_files=18,
                          csv_rows=sum(len(rows) for rows in tables.values()), findings={key: findings[key] for key in ("S_oracle_binary_whole_unit_improvement", "required_gain_over_existing_positive_gains", "incomplete_S_fraction_of_net_gain")}), indent=2))


if __name__ == "__main__":
    main()
