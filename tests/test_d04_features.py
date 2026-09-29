"""Independent D04 feature checks on synthetic arrays only; no panel loading or fit."""
from __future__ import annotations

import copy
import itertools
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/geo_weather_20260924"))

import d04_features as feature  # noqa: E402
from d04_data import CONTEXT_NAMES, WEATHER_NAMES, make_rows  # noqa: E402


def _raw(wx, times=(72,), units=None, mu=None, sd=None, knots=None, chunk=2):
    times = np.asarray(times)
    rows = {"unit": np.zeros(len(times), dtype=int) if units is None else np.asarray(units),
            "time": times}
    return feature.raw_weather_features(
        wx, rows, np.zeros(12) if mu is None else mu,
        np.ones(12) if sd is None else sd,
        np.zeros((2, 12)) if knots is None else knots, chunk=chunk,
    )


def _reference(wx, rows, mu, sd, knots):
    """Scalar-loop definition, independent of production vectorization/reshaping."""
    values = []
    cross = list(itertools.combinations(range(12), 2))
    self_cross = list(itertools.combinations_with_replacement(range(12), 2))
    for unit, target in zip(rows["unit"], rows["time"]):
        bands = [(wx[unit, target-hi:target-lo]-mu)/sd
                 for lo, hi in ((0, 6), (6, 24), (24, 48))]
        result = []
        for band in bands:
            for channel in range(12):
                x = band[:, channel]
                result.extend([sum(x)/len(x), sum(x*x)/len(x),
                               sum(max(float(v-knots[0, channel]), 0) for v in x)/len(x),
                               sum(max(float(v-knots[1, channel]), 0) for v in x)/len(x)])
        for band in bands:
            result.extend(sum(band[:, a]*band[:, b])/len(band) for a, b in cross)
        means = [np.array([sum(band[:, j])/len(band) for j in range(12)]) for band in bands]
        for near, far in ((0, 1), (0, 2), (1, 2)):
            for a, b in self_cross:
                result.append((means[far][a]*means[near][b] + means[far][b]*means[near][a])/2)
        for near, far in ((0, 1), (0, 2), (1, 2)):
            for a, b in cross:
                result.append((means[far][a]*means[near][b] - means[far][b]*means[near][a])/2)
        values.append(result)
    return np.asarray(values)


def _synthetic_panel():
    rng = np.random.default_rng(40401)
    n = 7
    geo_names = [name for block, names in feature.BLOCKS.items()
                 if block != "county_service_context" for name in names]
    county = np.array(["01001", "01003", "01001", "01005", "02001", "02003", "02005"])
    geo = rng.normal(size=(n, 40)).astype("f4")
    context = rng.normal(size=(n, 6)).astype("f4")
    geo[2], context[2] = geo[0], context[0]
    geo[[0, 2], 3] = np.nan
    y = rng.uniform(0, 0.3, size=(n, 216)).astype("f4")
    panel = {"weather": rng.normal(size=(n, 216, 12)).astype("f4"),
             "geo": geo, "context": context, "y_full": y,
             "obs_full": np.ones_like(y, dtype=bool),
             "feature_names": {"weather": list(WEATHER_NAMES), "geo": geo_names,
                               "context": list(CONTEXT_NAMES)},
             "meta": {"county": county, "origin": np.array(["2020-01-03T12:00:00"]*n),
                      "regime": np.array(["winter"]*n), "w": np.arange(1, n+1, dtype=float),
                      "w_raw": np.arange(1, n+1, dtype=float)*2,
                      "used": np.array([False, True, True, False, True, False, True])},
             "fold": np.array([1, 2, 1, 3, 4, 5, 5], dtype=np.int8),
             "merged_group": np.array(["a", "b", "c", "d", "e", "f", "g"])}
    rows = make_rows(panel, phase=0)
    train = (rows["unit"] < 4) & (rows["time"] <= 90)
    rows = {name: value[train] for name, value in rows.items()}
    return panel, rows


def _assert_nested_equal(actual, expected):
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for name in actual:
            _assert_nested_equal(actual[name], expected[name])
    elif isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            _assert_nested_equal(a, b)
    elif isinstance(actual, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    else:
        assert actual == expected


def test_all_774_columns_match_independent_scalar_definition_and_labels():
    rng = np.random.default_rng(40402)
    wx = rng.normal(size=(3, 216, 12))
    rows = {"unit": np.array([2, 0, 1]), "time": np.array([72, 101, 215])}
    mu, sd = rng.normal(size=12), rng.uniform(.3, 2, size=12)
    knots = np.stack([rng.normal(size=12), rng.normal(size=12)+2])
    actual = feature.raw_weather_features(wx, rows, mu, sd, knots, chunk=1)
    expected = _reference(wx, rows, mu, sd, knots)
    assert actual.shape == (3, 774)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    labels = feature.label_features(WEATHER_NAMES)
    assert len(labels) == len(set(labels)) == 774
    assert [sum(label.startswith(prefix+":") for label in labels)
            for prefix in ("main", "synchronous", "symmetric_history", "ordered_history")] == [144, 198, 234, 198]
    np.testing.assert_array_equal(actual, feature.raw_weather_features(wx, rows, mu, sd, knots, chunk=8))


@pytest.mark.parametrize("age,band", [(1, 0), (6, 0), (7, 1), (24, 1), (25, 2), (48, 2), (0, None), (49, None)])
def test_impulses_verify_strict_past_band_edges(age, band):
    wx = np.zeros((1, 216, 12))
    wx[0, 72-age, 0] = 12
    x = _raw(wx)[0]
    expected = np.zeros(3)
    if band is not None:
        expected[band] = 12/(6, 18, 24)[band]
    np.testing.assert_allclose(x[[0, 48, 96]], expected)
    if band is None:
        np.testing.assert_array_equal(x, 0)


def test_current_future_and_pre_lookback_weather_cannot_change_features():
    rng = np.random.default_rng(40403)
    wx = rng.normal(size=(1, 216, 12))
    baseline = _raw(wx)
    changed = wx.copy()
    changed[:, :24] = 1e5
    changed[:, 72:] = -1e5
    np.testing.assert_array_equal(_raw(changed), baseline)


def test_early_a_late_b_positive_and_channel_swap_flips_order_only():
    wx = np.zeros((1, 216, 12))
    wx[0, 48:66, 0] = 2  # a at ages 7..24
    wx[0, 66:72, 1] = 3  # b at ages 1..6
    before = _raw(wx)[0]
    swapped = wx.copy()
    swapped[:, :, [0, 1]] = wx[:, :, [1, 0]]
    after = _raw(swapped)[0]
    cross_index = list(itertools.combinations(range(12), 2)).index((0, 1))
    self_cross_index = list(itertools.combinations_with_replacement(range(12), 2)).index((0, 1))
    assert before[576+cross_index] == 3
    assert after[576+cross_index] == -3
    assert before[342+self_cross_index] == after[342+self_cross_index] == 3
    assert before[144+cross_index] == after[144+cross_index] == 0


def test_repeated_same_channel_has_symmetric_history_without_fake_self_order():
    wx = np.zeros((1, 216, 12))
    wx[0, 48:66, 4] = 2
    wx[0, 66:72, 4] = 5
    repeated = _raw(wx)[0]
    self_index = list(itertools.combinations_with_replacement(range(12), 2)).index((4, 4))
    assert repeated[342+self_index] == 10
    np.testing.assert_array_equal(repeated[576:], 0)
    wx[0, 48:66, 4] = 0
    assert _raw(wx)[0, 342+self_index] == 0
    assert all("pressure->pressure" not in label for label in feature.label_features(WEATHER_NAMES))


def test_synchronous_is_pointwise_product_and_preserves_within_band_overlap():
    overlap = np.zeros((1, 216, 12))
    overlap[0, 66:69, :2] = 2
    separated = overlap.copy()
    separated[0, 66:69, 1] = 0
    separated[0, 69:72, 1] = 2
    together, apart = _raw(overlap)[0], _raw(separated)[0]
    np.testing.assert_array_equal(together[:144], apart[:144])
    assert together[144] == 2 and apart[144] == 0
    # Both products of means are 1, and would miss this contrast.
    assert together[0]*together[4] == apart[0]*apart[4] == 1


def test_fit_mapping_ignores_heldout_values_unaccessed_weather_and_response_targets():
    panel, train = _synthetic_panel()
    wx = feature.transform_weather(panel)
    baseline_panel, baseline_wx = copy.deepcopy(panel), wx.copy()
    fm, arrays = feature.fit_map(panel, wx, train)
    changed_panel = copy.deepcopy(panel)
    changed_weather = wx.copy()
    accessed = np.zeros(wx.shape[:2], dtype=bool)
    for unit, time in zip(train["unit"], train["time"]):
        accessed[unit, time-48:time] = True
    changed_weather[~accessed] = 1e5
    changed_panel["geo"][4:] = 1e5
    changed_panel["context"][4:] = -1e5
    changed_panel["y_full"][:, 72:] = 1e5
    changed_panel["y_full"][4:, :72] = -1e5
    changed_train = copy.deepcopy(train)
    changed_train["y"][:] = -1e6
    altered, altered_arrays = feature.fit_map(changed_panel, changed_weather, changed_train)
    _assert_nested_equal(vars(fm), vars(altered))
    for before, after in zip(arrays, altered_arrays):
        np.testing.assert_array_equal(before, after)
    _assert_nested_equal(panel, baseline_panel)
    np.testing.assert_array_equal(wx, baseline_wx)
    for fit_array, transformed in zip(arrays, fm.transform(panel, wx, train)):
        np.testing.assert_allclose(transformed, fit_array, rtol=2e-6, atol=2e-6)
    assert arrays[0].shape == (len(train["unit"]), 774)
    assert arrays[2].shape == (len(train["unit"]), 144)
    assert arrays[3].shape == (len(train["unit"]), 12)
    np.testing.assert_array_equal(arrays[1][:, 0], 1)


def test_static_preprocessing_deduplicates_training_counties_and_context_is_separate():
    panel, rows = _synthetic_panel()
    wx = feature.transform_weather(panel)
    fm, arrays = feature.fit_map(panel, wx, rows)
    unique = panel["geo"][[0, 1, 3]].astype(float)
    med = np.nanmedian(unique, axis=0)
    filled = np.where(np.isfinite(unique), unique, med)
    np.testing.assert_allclose(fm.static_params["geo"][0], med, atol=1e-7)
    np.testing.assert_allclose(fm.static_params["geo"][1], filled.mean(0), atol=1e-7)
    changed = copy.deepcopy(panel)
    changed["context"] = (changed["context"]**2+1).astype("f4")
    other, other_arrays = feature.fit_map(changed, wx, rows)
    _assert_nested_equal(fm.static_params["geo"], other.static_params["geo"])
    np.testing.assert_array_equal(arrays[0], other_arrays[0])
    np.testing.assert_array_equal(arrays[2], other_arrays[2])
    assert not np.allclose(arrays[3], other_arrays[3])


def test_history_only_uses_allowed_previous_and_fixed_origin_stock():
    panel, rows = _synthetic_panel()
    wx = feature.transform_weather(panel)
    fm, _ = feature.fit_map(panel, wx, rows)
    G, C = feature.static_basis(panel, fm.static_params)
    with_history = feature.nuisance_raw(panel, rows, G, C, history=True)
    without_history = feature.nuisance_raw(panel, rows, G, C, history=False)
    assert G.shape[1] == 144 and C.shape[1] == 12
    assert with_history.shape[1] == 183 and without_history.shape[1] == 179
    for nuisance in (with_history, without_history):
        # Every arm must control the complete additive geographic basis, including
        # all 64 joint projections later reused by the geographic modifier.
        np.testing.assert_array_equal(nuisance[:, -156:-12], G[rows["unit"]])
        np.testing.assert_array_equal(nuisance[:, -12:], C[rows["unit"]])
    changed_g = G.copy()
    changed_g[:, 80:] += .25
    geo_changed = feature.nuisance_raw(panel, rows, changed_g, C, history=True)
    np.testing.assert_array_equal(geo_changed[:, :-76], with_history[:, :-76])
    np.testing.assert_allclose(geo_changed[:, -76:-12]-with_history[:, -76:-12], .25,
                               rtol=0, atol=1e-7)
    np.testing.assert_array_equal(geo_changed[:, -12:], with_history[:, -12:])
    changed = copy.deepcopy(panel)
    changed["y_full"][:, 72:] = 1e4
    np.testing.assert_array_equal(feature.nuisance_raw(changed, rows, G, C, history=True), with_history)
    changed["y_full"][:, 71] += .1
    assert not np.array_equal(feature.nuisance_raw(changed, rows, G, C, history=True), with_history)
    np.testing.assert_array_equal(feature.nuisance_raw(changed, rows, G, C, history=False), without_history)
    changed_rows = copy.deepcopy(rows)
    changed_rows["p_prev"] += .05
    assert not np.array_equal(feature.nuisance_raw(panel, changed_rows, G, C, history=True), with_history)
    np.testing.assert_array_equal(feature.nuisance_raw(panel, changed_rows, G, C, history=False), without_history)


def test_adjacent_rows_do_not_bridge_missing_hours_and_conserve_each_unit_weight():
    panel, _ = _synthetic_panel()
    panel["y_full"][0] = 0  # A quiet county is retained, including used=False.
    panel["obs_full"][1, 77] = False
    panel["y_full"][2, 81] = np.nan  # A true mask is insufficient for a nonfinite value.
    panel["obs_full"][3, 72:] = False
    all_rows, phase_rows = make_rows(panel), make_rows(panel, phase=0)
    key = set(zip(all_rows["unit"], all_rows["time"]))
    assert (1, 77) not in key and (1, 78) not in key
    assert (2, 81) not in key and (2, 82) not in key
    assert not (all_rows["unit"] == 3).any()
    assert (all_rows["unit"] == 0).sum() == 144
    np.testing.assert_array_equal(all_rows["y"][all_rows["unit"] == 0], 0)
    assert set(zip(phase_rows["unit"], phase_rows["time"])).issubset(key)
    for rows in (all_rows, phase_rows):
        for name in ("w", "w_raw"):
            restored = np.bincount(rows["unit"], weights=rows[name], minlength=7)
            np.testing.assert_allclose(restored[[0, 1, 2, 4, 5, 6]], panel["meta"][name][[0, 1, 2, 4, 5, 6]])
        np.testing.assert_allclose(rows["y"], panel["y_full"][rows["unit"], rows["time"]].astype(float)
                                   - panel["y_full"][rows["unit"], rows["time"]-1].astype(float))
