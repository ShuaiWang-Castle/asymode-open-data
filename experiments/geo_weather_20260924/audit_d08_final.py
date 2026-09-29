"""Independent D08 arithmetic audit after the input roster is frozen.

Only fixed public-D FIT labels and existing prediction caches are used. No
model module, checkpoint contents, forward pass, gradient or optimizer is used.
Run only after the selector and symptom replay have completed successfully.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "2"

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS = ROOT / "runs" / "geo_weather_20260924"
RESULTS = HERE / "results" / "v1"
FEATURES = ROOT / "data" / "interim" / "panel_v1" / "features_v1D.npz"
SPLITS = HERE / "splits_v1D.json"
DATA_SHA = "f043bb39e8abd48183670e2acc3c0cecebd7107ea9be0e28771912cfee358c48"
MODEL_NAMES = ("host", "crk", "crk_closed", "weather_control_0p001")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def finite_json(value) -> None:
    if isinstance(value, dict):
        for child in value.values():
            finite_json(child)
    elif isinstance(value, list):
        for child in value:
            finite_json(child)
    elif isinstance(value, float):
        assert math.isfinite(value), "Nonfinite public result"


def compare(actual, expected, errors: dict, category: str) -> None:
    if isinstance(expected, dict):
        for key, value in expected.items():
            assert key in actual, (category, key)
            compare(actual[key], value, errors, category)
    elif isinstance(expected, list):
        assert len(actual) == len(expected), category
        for a, b in zip(actual, expected):
            compare(a, b, errors, category)
    elif expected is None:
        assert actual is None, (category, actual)
    elif isinstance(expected, (int, np.integer)):
        assert actual == int(expected), (category, actual, expected)
    elif isinstance(expected, (float, np.floating)):
        assert math.isclose(actual, float(expected), rel_tol=2e-9, abs_tol=2e-9), (category, actual, expected)
        errors[category] = max(errors.get(category, 0.0), abs(actual - float(expected)))
    else:
        assert actual == expected, (category, actual, expected)


def dist(x, w, midpoint=False) -> dict:
    x, w = np.asarray(x, float), np.asarray(w, float)
    valid = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not valid.any():
        return dict(n=0, missing=len(x), mean=None, q10_median_q90=None,
                    minimum=None, maximum=None)
    values, weights = x[valid], w[valid]
    order = np.argsort(values, kind="stable")
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    knots = cumulative - 0.5 * sorted_weights if midpoint else cumulative
    quantiles = np.interp(np.array([0.1, 0.5, 0.9]) * cumulative[-1], knots, values[order])
    return dict(n=int(valid.sum()), missing=int((~valid).sum()),
                mean=float(weights @ values / weights.sum()),
                q10_median_q90=quantiles.tolist(), minimum=float(values.min()), maximum=float(values.max()))


def observed_shapes(y, mask, origin) -> dict:
    """Per-row reconstruction using explicit observed indices and runs."""
    n = len(y)
    integer_keys = ("observed_hours", "true_peak_hour", "severe_hours",
                    "maximum_contiguous_severe_run", "adjacent_observed_pairs")
    float_keys = ("true_peak", "maximum_adjacent_rise", "minimum_adjacent_change",
                  "maximum_rise_through_peak", "minimum_change_after_peak",
                  "peak_centered_half_height_rise_hours", "peak_centered_half_height_fall_hours")
    answer = {key: np.zeros(n, int) for key in integer_keys}
    answer.update({key: np.full(n, np.nan) for key in float_keys})
    answer.update({key: np.zeros(n, bool) for key in ("half_height_left_truncated", "half_height_right_truncated", "complete144")})
    answer["true_peak_hour"][:] = -1
    for i in range(n):
        observed = np.flatnonzero(mask[i])
        answer["observed_hours"][i] = len(observed)
        answer["complete144"][i] = len(observed) == 144
        if not len(observed):
            continue
        tpeak = int(observed[np.argmax(y[i, observed])])
        peak = float(y[i, tpeak])
        answer["true_peak"][i], answer["true_peak_hour"][i] = peak, tpeak + 72
        exceedances = observed[y[i, observed] >= 0.1]
        answer["severe_hours"][i] = len(exceedances)
        if len(exceedances):
            segments = np.split(exceedances, np.flatnonzero(np.diff(exceedances) != 1) + 1)
            answer["maximum_contiguous_severe_run"][i] = max(map(len, segments))
        pair_times = observed[(observed == 0) | mask[i, np.maximum(observed - 1, 0)]]
        answer["adjacent_observed_pairs"][i] = len(pair_times)
        changes = np.asarray([y[i, t] - (origin[i] if t == 0 else y[i, t - 1]) for t in pair_times])
        if len(changes):
            answer["maximum_adjacent_rise"][i] = changes.max()
            answer["minimum_adjacent_change"][i] = changes.min()
            before, after = changes[pair_times <= tpeak], changes[pair_times > tpeak]
            if len(before):
                answer["maximum_rise_through_peak"][i] = before.max()
            if len(after):
                answer["minimum_change_after_peak"][i] = after.min()
        if peak > 0:
            left_candidates = np.flatnonzero((~mask[i, :tpeak]) | (y[i, :tpeak] < 0.5 * peak))
            left = int(left_candidates[-1] + 1) if len(left_candidates) else 0
            right_candidates = np.flatnonzero((~mask[i, tpeak + 1:]) | (y[i, tpeak + 1:] < 0.5 * peak))
            right = int(tpeak + right_candidates[0]) if len(right_candidates) else 143
            answer["peak_centered_half_height_rise_hours"][i] = tpeak - left
            answer["peak_centered_half_height_fall_hours"][i] = right - tpeak
            answer["half_height_left_truncated"][i] = left == 0 or not mask[i, left - 1]
            answer["half_height_right_truncated"][i] = right == 143 or not mask[i, right + 1]
    return answer


def peak_descriptions(P, mask, shapes) -> dict:
    n = len(P)
    maximum, hour, ratio, lag = np.full(n, np.nan), np.full(n, -1, int), np.full(n, np.nan), np.full(n, np.nan)
    for i in range(n):
        observed = np.flatnonzero(mask[i])
        if not len(observed):
            continue
        time = int(observed[np.argmax(P[i, observed])])
        maximum[i], hour[i] = P[i, time], time + 72
        if shapes["true_peak"][i] > 0:
            ratio[i] = maximum[i] / shapes["true_peak"][i]
        lag[i] = hour[i] - shapes["true_peak_hour"][i]
    return dict(predicted_peak=maximum, predicted_peak_hour=hour,
                peak_ratio=ratio, signed_peak_lag_hours=lag)


def cohort_masks(shapes) -> dict:
    valid = shapes["observed_hours"] > 0
    severe = valid & (shapes["true_peak"] >= 0.1)
    answer = dict(all=valid, S=severe, nonS=valid & ~severe)
    for label, key in (("severe_hour_count", "severe_hours"), ("max_contiguous_run", "maximum_contiguous_severe_run")):
        x = shapes[key]
        answer[label + "/1"] = severe & (x == 1)
        answer[label + "/2..6"] = severe & (x >= 2) & (x <= 6)
        answer[label + "/>=7"] = severe & (x >= 7)
    return answer


def gain_summary(y, mask, baseline, candidate, selected, weights) -> dict:
    ids = np.flatnonzero(selected)
    if not len(ids):
        return dict(units=0, net_gain=0.0, positive_gain=0.0, negative_loss=0.0)
    residual, correction = y[ids] - baseline[ids], candidate[ids] - baseline[ids]
    align_rows = weights[ids] * np.sum(np.where(mask[ids], 2.0 * residual * correction, 0.0), axis=1)
    energy_rows = weights[ids] * np.sum(np.where(mask[ids], correction ** 2, 0.0), axis=1)
    gains = align_rows - energy_rows
    direct = weights[ids] * np.sum(np.where(mask[ids], (y[ids] - baseline[ids]) ** 2 - (y[ids] - candidate[ids]) ** 2, 0.0), axis=1)
    assert np.allclose(gains, direct, rtol=2e-9, atol=2e-9)
    positive, negative = gains > 0, gains < 0
    total_positive = float(gains[positive].sum())
    top = max(1, math.ceil(len(ids) * 0.1))
    return dict(units=len(ids), alignment=float(align_rows.sum()), modification_energy=float(energy_rows.sum()),
                net_gain=float(gains.sum()), positive_gain=total_positive, negative_loss=float(-gains[negative].sum()),
                positive_units=int(positive.sum()), negative_units=int(negative.sum()), zero_units=int((gains == 0).sum()),
                positive_design_share=float(weights[ids][positive].sum() / weights[ids].sum()),
                top_10pct_all_cohort_units=top,
                top_10pct_share_of_positive_gain=float(np.sort(gains[positive])[::-1][:top].sum() / total_positive) if total_positive else None)


def audit_score(row, y, mask, weights, rounded_weights, denominator, predictions,
                peaks, shapes, selected, pi, full_cohort, errors) -> None:
    ids = np.flatnonzero(selected)
    full_cohort_mass = float(np.sum(weights * mask.sum(1) * full_cohort))
    compare(row, dict(units=len(ids), original_full_FIT_cohort_hour_mass=full_cohort_mass), errors, "support")
    if not len(ids):
        return
    local_mass = float(weights[ids] @ mask[ids].sum(1))
    local_w, reweighted = weights[ids], weights[ids] / pi[ids]
    all_mass = float(weights @ mask.sum(1))
    compare(row, dict(observed_hours=int(mask[ids].sum()), complete144_units=int(shapes["complete144"][ids].sum()),
                      incomplete_units=int((~shapes["complete144"][ids]).sum()),
                      design_unit_mass=float(local_w.sum()), design_hour_mass=local_mass,
                      HT_design_unit_mass=float(reweighted.sum()), HT_design_hour_mass=float(reweighted @ mask[ids].sum(1)),
                      original_full_FIT_all_hour_mass=all_mass, original_full_FIT_objective_denominator=denominator), errors, "support")
    for name, values in row["outcome_shapes"].items():
        if name.endswith("_truncated"):
            indicator = shapes[name][ids]
            expected = dict(units=int(indicator.sum()), local_design_fraction=float(local_w[indicator].sum() / local_w.sum()),
                            HT_design_fraction=float(reweighted[indicator].sum() / reweighted.sum()))
        else:
            expected = dict(local=dist(shapes[name][ids], local_w), HT_reweighted=dist(shapes[name][ids], reweighted))
        compare(values, expected, errors, "shape_distributions")
    for name, P in predictions.items():
        squared_rows = np.sum(np.where(mask[ids], (P[ids] - y[ids]) ** 2, 0.0), axis=1)
        numerator, ht_num = float(local_w @ squared_rows), float(reweighted @ squared_rows)
        expected = dict(local_design_SSE=numerator, local_design_RMSE=math.sqrt(numerator / local_mass) if local_mass else None,
                        HT_design_SSE_numerator=ht_num, HT_MSE_full_FIT_cohort_denominator=ht_num / full_cohort_mass if full_cohort_mass else None,
                        HT_SSE_contribution_full_FIT_all_denominator=ht_num / all_mass,
                        local_original_objective_contribution=float(rounded_weights[ids] @ squared_rows / denominator),
                        HT_original_objective_contribution=float((rounded_weights[ids] / pi[ids]) @ squared_rows / denominator),
                        local_peak_ratio=dist(peaks[name]["peak_ratio"][ids], local_w),
                        HT_reweighted_peak_ratio=dist(peaks[name]["peak_ratio"][ids], reweighted),
                        local_peak_lag=dist(peaks[name]["signed_peak_lag_hours"][ids], local_w))
        nonS = selected & (shapes["true_peak"] < 0.1)
        alarm = nonS & (peaks[name]["predicted_peak"] >= 0.1)
        expected["nonS_severe_alarms"] = dict(units=int(alarm.sum()), local_design_mass=float(weights[alarm].sum()),
                                              local_design_rate=float(weights[alarm].sum() / weights[nonS].sum()) if nonS.any() else None,
                                              HT_design_mass=float(np.sum(weights[alarm] / pi[alarm])),
                                              HT_design_rate=float(np.sum(weights[alarm] / pi[alarm]) / np.sum(weights[nonS] / pi[nonS])) if nonS.any() else None)
        # Compact regime/group strata retain only this public subset of keys.
        compare(row["models"][name], {key: value for key, value in expected.items() if key in row["models"][name]}, errors, "model_scores")
    for base, candidate in (("host", "crk"), ("crk_closed", "crk"), ("crk", "weather_control_0p001")):
        key = candidate + "_vs_" + base
        for label, full_weights in (("local", weights), ("HT_reweighted", weights / pi)):
            compare(row["comparisons"][key][label], gain_summary(y, mask, predictions[base], predictions[candidate], selected, full_weights), errors, "paired_gains")


def merged_groups(meta) -> np.ndarray:
    systems = np.unique(meta["system"])
    parent = {str(system): str(system) for system in systems}
    def find(key):
        while key != parent[key]:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key
    info = {}
    for system in systems:
        ids = np.flatnonzero(meta["system"] == system)
        info[str(system)] = (str(meta["family"][ids[0]]), np.datetime64(meta["origin"][ids[0]]), set(meta["fips"][ids]))
    for j, a in enumerate(map(str, systems)):
        for b in map(str, systems[j + 1:]):
            fa, oa, ca = info[a]
            fb, ob, cb = info[b]
            if fa == fb or (abs(oa - ob) <= np.timedelta64(16, "D") and ca.intersection(cb)):
                pa, pb = find(a), find(b)
                parent[max(pa, pb)] = min(pa, pb)
    return np.array([find(str(system)) for system in meta["system"]])


def verify_frozen_originals() -> dict:
    audit = load_json(RESULTS / "d07_final_audit.json")
    packages = {}
    for name in ("d07_attribution.json.gz", "d07_controllers.json.gz", "d07_gradients.json.gz"):
        path = RESULTS / name
        raw = gzip.decompress(path.read_bytes())
        assert sha(path) == audit["packages"][name]["gzip_sha256"]
        assert hashlib.sha256(raw).hexdigest() == audit["packages"][name]["uncompressed_sha256"]
        packages[name] = json.loads(raw)
    frozen = packages["d07_attribution.json.gz"]["provenance"]["shared_frozen_provenance"]
    sources, artifacts = frozen["frozen_source_sha256"], frozen["frozen_artifact_sha256"]
    assert len(sources) == 16
    for name, digest in sources.items():
        rel = Path(name)
        assert not rel.is_absolute() and ".." not in rel.parts
        assert sha(ROOT / rel) == digest, name
    count = 0
    for label, folds in artifacts.items():
        assert label in ("v1_host_s0", "v1_crk_s0") and len(folds) == 5
        for fold, files in folds.items():
            for name, digest in files.items():
                assert name in ("outer.npz", "DONE.json", "final.pt")
                assert sha(RUNS / label / f"fold{int(fold):02d}" / name) == digest
                count += 1
    assert count == 30 and sha(FEATURES) == DATA_SHA
    return dict(original_sources=16, training_artifacts=30, prior_summary_packages=3, public_D_sha256=DATA_SHA)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=RUNS / "d08_panel_20260929_iofix1")
    parser.add_argument("--receipt", type=Path, default=RESULTS / "d08_final_audit.json")
    args = parser.parse_args()
    if args.receipt.exists():
        raise FileExistsError("Preserve earlier final D08 audit.")
    assert os.getpriority(os.PRIO_PROCESS, 0) >= 15, "Run the numeric audit at nice >= 15"
    runtime = args.runtime.resolve()
    runtime.relative_to(ROOT)
    # This gate precedes every outcome-bearing member or prediction-cache read.
    marker = load_json(runtime / "FROZEN.json")
    assert marker["status"] == "input_roster_frozen"
    for filename, key in (("manifest.json", "manifest_sha256"), ("roster.json", "roster_sha256"), ("panel_inputs.npz", "panel_inputs_sha256")):
        assert sha(runtime / filename) == marker[key]
    manifest = load_json(runtime / "manifest.json")
    result_path = RESULTS / "d08_panel_symptoms.json"
    compressed_result = RESULTS / "d08_panel_symptoms.json.gz"
    if compressed_result.exists():
        result_path = compressed_result
    result = load_json(result_path)
    finite_json(result)
    assert result["input_roster_frozen_before_outcome_access"]
    assert result["no_new_forward_gradient_optimizer_or_training"]
    assert result["all_scores_are_original_fold1_FIT_descriptive_not_inner_holdout"]
    assert sha(HERE / "d08_symptoms.py") == result["source_sha256"]
    arrays_path = runtime / "symptoms_arrays.npz"
    assert sha(arrays_path) == result["provenance"]["local_arrays_sha256"]
    originals = verify_frozen_originals()
    initial_paths = [Path(__file__), result_path, arrays_path, SPLITS,
                     RESULTS / "d07_final_audit.json", RESULTS / "d07_gradients.json.gz",
                     RESULTS / "d07_controllers.json.gz", RESULTS / "d07_attribution.json.gz"]
    initial_hashes = {path: sha(path) for path in initial_paths}
    with np.load(runtime / "panel_inputs.npz", allow_pickle=False) as cache:
        panel_ids, panel_pi = cache["unit"].copy(), cache["pi"].copy()
        original_fit = cache["original_fit"].copy()
    fit = np.sort(original_fit.astype(int))
    split = load_json(SPLITS)
    assert np.array_equal(fit, np.sort(split["event"]["1"]["dev"]))
    assert len(fit) == 6350 and len(panel_ids) == 1219 and len(np.unique(panel_ids)) == 1219
    assert sha(FEATURES) == DATA_SHA
    with np.load(FEATURES, allow_pickle=False) as public:
        data = {key: public[key][fit].copy() for key in ("y", "m", "y0", "w", "y_full", "obs_full", "system", "family", "origin", "fips", "regime")}
    y, mask, w = data["y"].astype(float), data["m"].astype(bool), data["w"].astype(float)
    assert y.shape == mask.shape == (6350, 144) and np.isfinite(y).all()
    assert np.array_equal(mask, data["obs_full"][:, 72:].astype(bool))
    assert data["obs_full"][:, 71].all() and np.array_equal(data["y_full"][:, 71], data["y0"])
    with np.load(arrays_path, allow_pickle=False) as cache:
        arrays = {key: cache[key].copy() for key in cache.files}
    assert np.array_equal(arrays["unit"], fit)
    assert np.array_equal(arrays["y"], y) and np.array_equal(arrays["m"], mask)
    selected = np.isin(fit, panel_ids)
    pi = np.ones(len(fit))
    pi[np.searchsorted(fit, panel_ids)] = panel_pi
    assert np.array_equal(arrays["panel"], selected) and np.array_equal(arrays["pi"], pi)
    predictions = {name: arrays["P_" + name].astype(float) for name in MODEL_NAMES}
    for name, P in predictions.items():
        assert P.shape == y.shape and np.isfinite(P).all() and np.all((P >= 0) & (P <= 1)), name
    cache_hashes = result["provenance"]["D07_cache_sha256"]
    controls = json.loads(gzip.decompress((RESULTS / "d07_controllers.json.gz").read_bytes()))
    grad = json.loads(gzip.decompress((RESULTS / "d07_gradients.json.gz").read_bytes()))
    expected_cache_hashes = dict(controls["local_artifact_hashes"], **grad["local_artifacts"])
    prior = RUNS / "d07_selectivity_20260929"
    for name, model, key in (("host_fold1_scalars.npz", "host", "P"), ("controller_baseline_scalars.npz", "crk", "P"), ("fold1_closed.npz", "crk_closed", "P_closed"), ("gradients_parameter_predictions.npz", "weather_control_0p001", "weather_control_0p001")):
        assert sha(prior / name) == cache_hashes[name] == expected_cache_hashes[name]
        with np.load(prior / name, allow_pickle=False) as cache:
            assert np.array_equal(predictions[model], cache[key][fit].astype(float)), model
    shapes = observed_shapes(y, mask, data["y0"])
    for name, values in shapes.items():
        assert np.array_equal(values, arrays[name], equal_nan=True), name
    peaks = {name: peak_descriptions(P, mask, shapes) for name, P in predictions.items()}
    groups = merged_groups(data)
    assert len(np.unique(groups)) == 53
    Z = grad["objective"]["Z_regime"]
    for regime, normalizer in Z.items():
        actual = float(w[data["regime"] == regime] @ np.sum(np.where(mask[data["regime"] == regime], y[data["regime"] == regime] ** 2, 0.0), axis=1))
        assert math.isclose(actual, normalizer, rel_tol=2e-9, abs_tol=2e-9)
    unscaled = w / np.array([Z[regime] for regime in data["regime"]])
    rounded_weights = (unscaled * (mask.sum() / np.sum(mask * unscaled[:, None]))).astype(np.float32).astype(float)
    denominator = grad["objective"]["common_full_FIT_denominator"]
    approximate_float32_denom = float(np.sum(mask.astype(np.float32) * rounded_weights.astype(np.float32)[:, None], dtype=np.float32))
    assert math.isclose(denominator, approximate_float32_denom, rel_tol=3e-7, abs_tol=0.5)
    masks, errors, scored_rows = cohort_masks(shapes), {}, 0
    populations = {"full_original_fold1_FIT": (np.ones(len(fit), bool), np.ones(len(fit))), "frozen_input_panel": (selected, pi)}
    supports = {"common_observed": shapes["observed_hours"] > 0, "complete144": shapes["complete144"], "incomplete": ~shapes["complete144"]}
    for label, (selection, probability) in populations.items():
        for support, support_mask in supports.items():
            for cohort, cohort_mask in masks.items():
                full_cohort = support_mask & cohort_mask
                audit_score(result["scores"][label][support][cohort], y, mask, w, rounded_weights, denominator,
                            predictions, peaks, shapes, selection & full_cohort, probability, full_cohort, errors)
                scored_rows += 1
    equal_weights = np.empty(len(fit))
    for group in np.unique(groups):
        members = groups == group
        equal_weights[members] = w[members] / w[members].sum()
    for label, (selection, probability) in populations.items():
        for cohort, cohort_mask in masks.items():
            audit_score(result["equal_merged_group_descriptives"]["scores"][label][cohort], y, mask, equal_weights,
                        rounded_weights, denominator, predictions, peaks, shapes, selection & cohort_mask, probability, cohort_mask, errors)
            scored_rows += 1
    for stratum, values in (("regime", data["regime"]), ("merged_group", groups)):
        for value in np.unique(values):
            member = values == value
            for label, (selection, probability) in populations.items():
                for cohort in ("all", "S", "nonS"):
                    full_cohort = member & masks[cohort]
                    row = result["within_FIT_regime_and_group_descriptives"][stratum][str(value)][label][cohort]
                    # Compact strata omit denominator/support detail, but additive model metrics remain.
                    augmented = dict(row)
                    augmented["original_full_FIT_cohort_hour_mass"] = float(w @ (mask.sum(1) * full_cohort))
                    if len(np.flatnonzero(selection & full_cohort)):
                        augmented.update(outcome_shapes={}, original_full_FIT_cohort_hour_mass=float(w @ (mask.sum(1) * full_cohort)),
                                         original_full_FIT_all_hour_mass=float(w @ mask.sum(1)), original_full_FIT_objective_denominator=denominator)
                        for key, val in dict(HT_design_hour_mass=float(np.sum((w[selection & full_cohort] / probability[selection & full_cohort]) * mask[selection & full_cohort].sum(1)))).items():
                            augmented[key] = val
                    audit_score(augmented, y, mask, w, rounded_weights, denominator, predictions, peaks, shapes,
                                selection & full_cohort, probability, full_cohort, errors)
                    scored_rows += 1
    # Independently reconstruct continuous-vs-count cross-tabulation.
    count_bin = np.digitize(shapes["severe_hours"], [2, 7])
    run_bin = np.digitize(shapes["maximum_contiguous_severe_run"], [2, 7])
    for label, (selection, probability) in populations.items():
        rows = result["severe_count_vs_contiguous_run"][label]
        for j, row in enumerate(rows):
            count, run = divmod(j, 3)
            members = selection & masks["S"] & (count_bin == count) & (run_bin == run)
            compare(row, dict(units=int(members.sum()), local_design_mass=float(w[members].sum()), HT_design_mass=float(np.sum(w[members] / probability[members]))), errors, "run_cross_tab")
    # The accepted input pairs are dependent deterministic comparisons, with no rematching.
    accepted = [row for row in manifest["pairs"] if row["status"] == "selected"]
    mapping = {int(unit): i for i, unit in enumerate(fit)}
    pair_values = {key: [] for key in ("anchor_unit", "partner_unit", "kind", "weather_distance", "geography_distance", "common_observed_hours", "same_merged_group", "true_peak_difference", "true_peak_hour_difference", "true_trajectory_RMS_difference")}
    for name in predictions:
        for key in ("predicted_peak_difference", "peak_lag_difference", "trajectory_RMS_difference", "contrast_reproduction_RMSE"):
            pair_values[name + "_" + key] = []
    for row in accepted:
        a, b = mapping[row["anchor_unit"]], mapping[row["partner_unit"]]
        assert selected[a] and selected[b] and pi[a] == pi[b] == 1
        common = mask[a] & mask[b]
        for key in ("anchor_unit", "partner_unit", "kind", "weather_distance", "geography_distance"):
            pair_values[key].append(row[key])
        pair_values["common_observed_hours"].append(int(common.sum()))
        pair_values["same_merged_group"].append(groups[a] == groups[b])
        pair_values["true_peak_difference"].append(shapes["true_peak"][a] - shapes["true_peak"][b])
        pair_values["true_peak_hour_difference"].append(float(shapes["true_peak_hour"][a] - shapes["true_peak_hour"][b]) if shapes["true_peak_hour"][a] >= 0 and shapes["true_peak_hour"][b] >= 0 else np.nan)
        true_contrast = y[a, common] - y[b, common]
        pair_values["true_trajectory_RMS_difference"].append(float(np.sqrt(np.mean(true_contrast ** 2))) if common.any() else np.nan)
        for name, P in predictions.items():
            contrast = P[a, common] - P[b, common]
            pair_values[name + "_predicted_peak_difference"].append(peaks[name]["predicted_peak"][a] - peaks[name]["predicted_peak"][b])
            pair_values[name + "_peak_lag_difference"].append(peaks[name]["signed_peak_lag_hours"][a] - peaks[name]["signed_peak_lag_hours"][b])
            pair_values[name + "_trajectory_RMS_difference"].append(float(np.sqrt(np.mean(contrast ** 2))) if common.any() else np.nan)
            pair_values[name + "_contrast_reproduction_RMSE"].append(float(np.sqrt(np.mean((contrast - true_contrast) ** 2))) if common.any() else np.nan)
    for key, values in pair_values.items():
        actual = arrays["pair_" + key]
        values = np.asarray(values)
        if actual.dtype.kind in "fc":
            assert np.allclose(actual, values, rtol=2e-9, atol=2e-9, equal_nan=True), key
        else:
            assert np.array_equal(actual, values), key
    pair_report = result["frozen_input_pair_outcome_descriptives"]
    assert pair_report["accepted_pairs"] == len(accepted)
    for kind, row in pair_report["rows"].items():
        take = np.asarray(pair_values["kind"]) == kind
        compare(row, dict(accepted_pairs=int(take.sum()), zero_common_observation_pairs=int((np.asarray(pair_values["common_observed_hours"])[take] == 0).sum()), same_merged_group_pairs=int(np.asarray(pair_values["same_merged_group"])[take].sum())), errors, "pair_scores")
        for key, values in row["unweighted_descriptive_distributions"].items():
            compare(values, dist(np.asarray(pair_values[key])[take], np.ones(take.sum())), errors, "pair_scores")
    changes = {}
    for label, (selection, probability) in populations.items():
        changes[label] = {}
        for cohort in ("all", "S", "nonS"):
            members = selection & masks[cohort]
            cells = members[:, None] & mask
            delta = predictions["weather_control_0p001"] - predictions["crk"]
            baseline_alarm = members & masks["nonS"] & (peaks["crk"]["predicted_peak"] >= 0.1)
            changed_alarm = members & masks["nonS"] & (peaks["weather_control_0p001"]["predicted_peak"] >= 0.1)
            changes[label][cohort] = dict(maximum_abs_prediction_change=float(np.abs(delta[cells]).max()) if cells.any() else None,
                                         baseline_false_alarms=int(baseline_alarm.sum()), changed_false_alarms=int(changed_alarm.sum()),
                                         new_false_alarms=int((changed_alarm & ~baseline_alarm).sum()), removed_false_alarms=int((baseline_alarm & ~changed_alarm).sum()))
    prior_attr = load_json(RESULTS / "d07_attribution.json.gz")
    prior_S = prior_attr["summaries"]["fit1_independent_host"]["supports"]["common_observed"]["S"]
    current_S = result["scores"]["full_original_fold1_FIT"]["common_observed"]["S"]
    direct_reproduction = dict(S_units=current_S["units"], prior_D07_S_units=prior_S["support"]["units"], models={})
    assert direct_reproduction["S_units"] == direct_reproduction["prior_D07_S_units"] == 559
    for name, prefix in (("host", "base"), ("crk", "candidate")):
        old_rmse, new_rmse = prior_S[prefix + "_rmse"], current_S["models"][name]["local_design_RMSE"]
        old_quantiles = prior_S["peak_ratio"][prefix + "_q10_median_q90"]
        new_quantiles = current_S["models"][name]["local_peak_ratio"]["q10_median_q90"]
        assert old_rmse == new_rmse and old_quantiles == new_quantiles
        direct_reproduction["models"][name] = dict(prior_D07_RMSE=old_rmse, D08_RMSE=new_rmse,
                                                  prior_D07_peak_ratio_quantiles=old_quantiles,
                                                  D08_peak_ratio_quantiles=new_quantiles, exact_match=True)
    kish = lambda weight: float(weight.sum() ** 2 / np.square(weight).sum())
    weight_support = dict(full_FIT_original_design_Kish=kish(w),
                          panel_original_design_Kish=kish(w[selected]),
                          panel_inverse_probability_design_Kish=kish(w[selected] / pi[selected]),
                          interpretation="Kish is a weight-concentration diagnostic, not the number of independent events or a guarantee of generalization.")
    originals_after = verify_frozen_originals()
    assert originals == originals_after
    for path, digest in initial_hashes.items():
        assert sha(path) == digest
    for filename, key in (("manifest.json", "manifest_sha256"), ("roster.json", "roster_sha256"), ("panel_inputs.npz", "panel_inputs_sha256")):
        assert sha(runtime / filename) == marker[key]
    unique_pairs = {tuple(sorted((row["anchor_unit"], row["partner_unit"]))) for row in accepted}
    unique_units = {unit for pair in unique_pairs for unit in pair}
    receipt = dict(passed=True, scope="Independent fixed-roster original-fold1-FIT arithmetic only",
                   data_units=6350, panel_units=1219, merged_groups=53,
                   source_sha256=sha(Path(__file__)), input_roster_frozen_before_outcomes=True,
                   scored_cohort_rows=scored_rows, maximum_arithmetic_errors=errors,
                   all_cached_prediction_arrays_match_D07=True, original_sources_and_training_artifacts_unchanged=originals,
                   continuous_and_hour_count_arrays_independently_recomputed=True,
                   outcome_members_read=["y", "m", "y0", "w", "y_full", "obs_full", "system", "family", "origin", "fips", "regime"],
                   weather_control_0p001_function_changes=changes,
                   fixed_input_pairs=dict(accepted_records=len(accepted), distinct_unordered_pairs=len(unique_pairs), distinct_participants=len(unique_units), all_pair_common_observation_arithmetic_verified=True),
                   prior_D07_full_FIT_severe_symptoms_exact_reproduction=direct_reproduction,
                   design_weight_effective_support=weight_support,
                   original_objective_denominator=denominator, numpy_float32_denominator_sanity_check=approximate_float32_denom,
                   full_result_artifact=dict(path=str(result_path.relative_to(ROOT)),
                                             file_sha256=initial_hashes[result_path],
                                             uncompressed_sha256=hashlib.sha256(gzip.decompress(result_path.read_bytes()) if result_path.suffix == ".gz" else result_path.read_bytes()).hexdigest()),
                   provenance={str(path.relative_to(ROOT)): digest for path, digest in initial_hashes.items()},
                   no_new_model_forward_gradient_or_training=True, no_outer_risk_claim=True,
                   limitations=["HT design-unbiasedness applies to additive fixed-predictor totals; weighted quantiles, proportions, RMSE, raw sample counts and top-selected-decile concentration are not unbiased full-population estimates.",
                                "All frozen predictions here were trained on original fold1 FIT; replay does not test inner event generalization.",
                                "Inverse-probability panel weights reduce Kish support from about 845.99 for full FIT original design weights to about 89.41; this warns of fragile reweighted summaries, not a model effect.",
                                "Dependent input-matched pairs do not establish a causal geographic mechanism or net geographic information."])
    finite_json(receipt)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(passed=True, scored_cohort_rows=scored_rows, maximum_arithmetic_errors=errors,
                          fixed_input_pairs=receipt["fixed_input_pairs"]), indent=2))


if __name__ == "__main__":
    main()
