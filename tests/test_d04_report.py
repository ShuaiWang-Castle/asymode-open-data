"""Independent weighted D04 score checks; only tiny synthetic OOF arrays."""
from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/geo_weather_20260924"))
import report_d04_response as report  # noqa: E402

NAMES = ["zero_change", *report.ARMS]


def _fixture():
    entries = []
    for group in range(9):
        for regime in range(5):
            for local in range(2):
                for phase in range(6):
                    # Deliberately unequal clock-phase coverage within units.
                    if local == 1 and phase == 2 and group % 2:
                        continue
                    entries.append((group, regime, local, phase))
    group, regime, local, phase = np.asarray(entries).T
    unit = group*10+regime*2+local
    count = np.bincount(unit)[unit]
    design_mass = 1+group*.11+regime*.2+local*.6
    y = (.004+.0008*group+.002*regime)*(phase-2.5)
    y[(group+regime+phase) % 11 == 0] = 0
    prediction = np.column_stack([.55*y+.001*group, .7*y-.002*local,
                                  .78*y+.0004*(regime-2), .85*y+.0003*phase])
    rows = dict(unit=unit, time=72+phase, y=y, w=design_mass/count,
                w_raw=design_mass*(1+.4*local+.05*group)/count,
                group=np.array([f"G{x}" for x in group]),
                county=np.array([f"{1001+(x+z)%6:05d}" for x, z in zip(group, local)]),
                regime=np.asarray(report.REGIMES)[regime], fold=group % 5+1,
                system=np.array([f"S{x:02d}_{r}" for x, r in zip(group, regime)]))
    return rows, prediction


def _direct_scores(rows, predictions, weights):
    p = np.column_stack([np.zeros(len(predictions)), predictions])
    error = np.square(p-rows["y"][:, None])
    per = []
    for regime in report.REGIMES:
        idx = (rows["regime"] == regime) & (weights > 0)
        per.append(np.average(error[idx], weights=weights[idx], axis=0))
    per = np.asarray(per)
    pooled = np.average(error, weights=weights, axis=0)
    return dict(per_regime_mse=per, headline_mse=per[:2].mean(0), all5_mse=per.mean(0),
                pooled_mse=pooled, pooled_rmse=np.sqrt(pooled))


def test_grouped_sums_matches_explicit_group_regime_loops():
    rng = np.random.default_rng(40501)
    error = rng.uniform(0, 3, size=(43, 5))
    weights = rng.uniform(.05, 7, size=43)
    regimes = rng.integers(0, 5, size=43)
    labels = np.asarray(["q", "a", "z"])[rng.integers(0, 3, size=43)]
    levels, mass, sums = report.grouped_sums(error, weights, regimes, labels)
    expected_mass = np.zeros((len(levels), 5))
    expected_sums = np.zeros((len(levels), 5, 5))
    for i, label in enumerate(levels):
        for r in range(5):
            idx = (labels == label) & (regimes == r)
            expected_mass[i, r] = sum(weights[idx])
            for model in range(5):
                expected_sums[i, r, model] = sum(weights[idx]*error[idx, model])
    np.testing.assert_allclose(mass, expected_mass, rtol=2e-15, atol=2e-15)
    np.testing.assert_allclose(sums, expected_sums, rtol=2e-15, atol=2e-15)


def test_statistics_batched_and_point_estimates_match_direct_weighted_means():
    rows, pred = _fixture()
    p = np.column_stack([np.zeros(len(pred)), pred])
    error = np.square(p-rows["y"][:, None])
    regime = np.array([report.REGIMES.index(r) for r in rows["regime"]])
    levels, mass, sums = report.grouped_sums(error, rows["w"], regime, rows["group"])
    multipliers = np.vstack([np.ones(len(levels)), np.arange(1, len(levels)+1)])
    actual = report.statistics(multipliers@mass,
                              (multipliers@sums.reshape(len(levels), -1)).reshape(2, 5, 5))
    codes = np.searchsorted(levels, rows["group"])
    for j in range(2):
        expected = _direct_scores(rows, pred, rows["w"]*multipliers[j, codes])
        for key in expected:
            np.testing.assert_allclose(actual[key][j], expected[key], rtol=2e-14, atol=1e-16)
    assert not np.allclose(actual["all5_mse"], actual["pooled_mse"], atol=1e-10)


@pytest.mark.parametrize("weight", ["w", "w_raw"])
def test_score_summary_point_estimates_and_sign_fold_slices_use_direct_average(weight):
    rows, pred = _fixture()
    result = report.score_summary(rows, pred, weight=weight, boot=20)
    expected = _direct_scores(rows, pred, rows[weight])
    for j, name in enumerate(NAMES):
        for key, value in expected.items():
            np.testing.assert_allclose(result["arms"][name][key], value[..., j], rtol=2e-14, atol=1e-16)
    p = np.column_stack([np.zeros(len(pred)), pred])
    masks = {"zero_change": rows["y"] == 0, "positive": rows["y"] > 0,
             "negative": rows["y"] < 0, "positive_at_least_1pp": rows["y"] >= .01,
             "negative_at_most_minus1pp": rows["y"] <= -.01,
             **{f"fold_{k}": rows["fold"] == k for k in range(1, 6)}}
    for label, idx in masks.items():
        summary = result["strata"][label]
        assert summary["rows"] == int(idx.sum())
        errors = p[idx]-rows["y"][idx, None]
        for field, values in (("mse", errors**2), ("mae", abs(errors)), ("mean_prediction", p[idx])):
            np.testing.assert_allclose([summary[field][name] for name in NAMES],
                                       np.average(values, weights=rows[weight][idx], axis=0),
                                       rtol=2e-14, atol=1e-16)


def test_event_and_county_bootstraps_match_independent_row_weight_resampling():
    rows, pred = _fixture()
    nboot = 67
    result = report.score_summary(rows, pred, boot=nboot)
    expected_point = _direct_scores(rows, pred, rows["w"])
    for label, key, seed in (("merged_event", "group", 20260930), ("county", "county", 20260931)):
        levels, codes = np.unique(rows[key], return_inverse=True)
        draws = np.random.default_rng(seed).multinomial(len(levels), np.ones(len(levels))/len(levels), size=nboot)
        direct = [_direct_scores(rows, pred, rows["w"]*draw[codes]) for draw in draws]
        for arm, baseline in (("D_geo_pairs", "B_shared_pairs"), ("A_shared_main", "zero_change")):
            ia, ib = NAMES.index(arm), NAMES.index(baseline)
            comparison = result["comparisons"][arm+"_vs_"+baseline][label]
            for metric in ("per_regime_mse", "headline_mse", "all5_mse", "pooled_rmse"):
                values = np.asarray([100*(d[metric][..., ia]/d[metric][..., ib]-1) for d in direct])
                np.testing.assert_allclose(comparison[metric]["conditional_score_ci95_percent"],
                                           np.quantile(values, [.025, .975], axis=0), rtol=2e-13, atol=2e-12)
                np.testing.assert_allclose(comparison[metric]["change_percent"],
                                           100*(expected_point[metric][..., ia]/expected_point[metric][..., ib]-1),
                                           rtol=2e-13, atol=2e-12)
                assert np.all(np.asarray(comparison[metric]["valid_draws"]) == nboot)


def test_within_system_time_score_is_weighted_residual_contrast_not_forecast_error():
    rows, pred = _fixture()
    result = report.score_summary(rows, pred, boot=20)
    p = np.column_stack([np.zeros(len(pred)), pred])
    residual = p-rows["y"][:, None]
    keys = [(system, int(time)) for system, time in zip(rows["system"], rows["time"])]
    for key in set(keys):
        idx = np.asarray([value == key for value in keys])
        residual[idx] -= np.average(residual[idx], weights=rows["w"][idx], axis=0)
    expected = np.average(residual**2, weights=rows["w"], axis=0)
    np.testing.assert_allclose([result["within_system_time_contrast_mse"][name] for name in NAMES],
                               expected, rtol=2e-13, atol=1e-16)
    for j, name in enumerate(NAMES):
        assert expected[j] <= result["arms"][name]["pooled_mse"]+1e-16
    # An arbitrary common system/time bias is removed by this diagnostic, even
    # though it worsens the actual held-out prediction. This is not an OOF score.
    system_code = np.unique(rows["system"], return_inverse=True)[1]
    shifted = pred + (.08*system_code+.03*rows["time"])[:, None]
    altered = report.score_summary(rows, shifted, boot=20)
    np.testing.assert_allclose(list(altered["within_system_time_contrast_mse"].values()),
                               list(result["within_system_time_contrast_mse"].values()), rtol=2e-10, atol=1e-16)
    assert altered["arms"]["D_geo_pairs"]["pooled_mse"] > result["arms"]["D_geo_pairs"]["pooled_mse"]


def test_zero_denominator_bootstrap_draws_are_excluded_metric_by_metric():
    # Each regime has one active and one all-zero event. Event resampling often
    # retains a regime but misses its active event, making its baseline MSE zero.
    regime = np.repeat(np.arange(5), 2)
    n = len(regime)
    rows = dict(y=np.tile([.02, 0.], 5), w=np.ones(n), w_raw=np.ones(n),
                unit=np.arange(n), time=np.arange(n)+72, fold=regime+1,
                regime=np.asarray(report.REGIMES)[regime],
                county=np.asarray([f"{i:05d}" for i in range(n)]),
                group=np.asarray([f"g{i}" for i in range(n)]))
    pred = np.full((n, 4), .005)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = report.score_summary(rows, pred, boot=211)
    summary = result["comparisons"]["A_shared_main_vs_zero_change"]["merged_event"]["per_regime_mse"]
    ci = np.asarray(summary["conditional_score_ci95_percent"])
    assert np.isfinite(ci).all(), "Undefined relative errors must not poison supported intervals"
    count = np.asarray(summary["valid_draws"])
    assert count.shape == (5,), "Each regime has different positive-denominator bootstrap support"
    assert np.all(count > 0) and np.all(count < 211)


@pytest.mark.parametrize("weight", ["w", "w_raw"])
def test_clock_phase_preserves_each_present_county_event_mass(weight):
    rows, pred = _fixture()
    result = report.score_summary(rows, pred, weight=weight, boot=20)
    full_mass = np.bincount(rows["unit"], weights=rows[weight])
    p = np.column_stack([np.zeros(len(pred)), pred])
    changed_by_reweighting = False
    for phase in range(6):
        selected = (rows["time"]-72) % 6 == phase
        county_events = np.unique(rows["unit"][selected])
        means = []
        for unit in county_events:
            idx = selected & (rows["unit"] == unit)
            means.append(np.mean(np.square(p[idx]-rows["y"][idx, None]), axis=0))
        expected = np.average(means, weights=full_mass[county_events], axis=0)
        summary = result["strata"][f"clock_phase_{phase}"]
        np.testing.assert_allclose([summary["mse"][name] for name in NAMES], expected,
                                   rtol=2e-14, atol=1e-16)
        assert summary["weighting"] == "county-event mass conserved within phase"
        conditional = np.average(np.square(p[selected]-rows["y"][selected, None]),
                                 weights=rows[weight][selected], axis=0)
        changed_by_reweighting |= not np.allclose(expected, conditional, rtol=1e-7, atol=1e-12)
    assert changed_by_reweighting, "The fixture must detect accidental reuse of full-hour weights"


def test_ratio_interval_tracks_denominator_and_nonfinite_support_per_column():
    pa, pb = np.array([2., 4., 5.]), np.array([1., 0., 2.])
    da = np.array([[2., 4., 8.], [3., 5., 4.], [np.nan, 2., 0.],
                   [1., 6., 2.], [5., 2., np.inf]])
    db = np.array([[1., 0., 4.], [1., 1., 0.], [2., 0., 1.],
                   [0., 2., 2.], [2., -1., 1.]])
    result = report.ratio_interval(pa, pb, da, db)
    np.testing.assert_allclose(result["change_percent"], [100., np.nan, 150.], equal_nan=True)
    for j in range(3):
        valid = np.isfinite(da[:, j]) & np.isfinite(db[:, j]) & (db[:, j] > 0)
        values = 100*(da[valid, j]/db[valid, j]-1)
        np.testing.assert_allclose(result["conditional_score_ci95_percent"][:, j],
                                   np.quantile(values, [.025, .975]))
        assert result["valid_draws"][j] == valid.sum()
        assert result["zero_denominator_draws"][j] == (db[:, j] <= 0).sum()
        assert result["nonfinite_draws"][j] == (~np.isfinite(da[:, j]) | ~np.isfinite(db[:, j])).sum()


@pytest.mark.parametrize("shape", [(), (5,)])
def test_no_supported_bootstrap_draws_return_missing_intervals_instead_of_crashing(shape):
    point_a, point_b = np.ones(shape), np.ones(shape)
    empty = np.empty((0,)+shape)
    result = report.ratio_interval(point_a, point_b, empty, empty)
    assert np.asarray(result["conditional_score_ci95_percent"]).shape == (2,)+shape
    assert np.isnan(result["conditional_score_ci95_percent"]).all()
    assert np.all(result["valid_draws"] == 0)


def test_zero_reference_point_is_explicit_json_null_and_scalar_arrays_serialize():
    rows, pred = _fixture()
    rows["y"][:] = 0
    result = report.score_summary(rows, pred, boot=20)
    point = result["comparisons"]["A_shared_main_vs_zero_change"]["merged_event"]["headline_mse"]
    assert np.isnan(point["change_percent"])
    assert point["valid_draws"] == 0
    cleaned = report.clean({"score": point, "scalar": np.asarray(2.), "invalid": np.asarray(np.nan)})
    encoded = json.loads(json.dumps(cleaned, allow_nan=False))
    assert encoded["score"]["change_percent"] is None
    assert encoded["score"]["conditional_score_ci95_percent"] == [None, None]
    assert encoded["scalar"] == 2. and encoded["invalid"] is None
