"""Independent numerical checks of D01 helpers, using synthetic data only."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/geo_weather_20260924"))
from analyze_data_relationships_v1 import (  # noqa: E402
    RidgeCache,
    donor_geography,
    fit_transform,
    moments,
)


def test_time_reversal_preserves_symmetric_moments_and_negates_order():
    rng = np.random.default_rng(911)
    x = rng.normal(size=(5, 24, 6))
    x[:, 5, 0] += 4.0
    x[:, 11, 1] += 5.0
    covariance, symmetric, ordered = moments(x)
    reversed_covariance, reversed_symmetric, reversed_order = moments(x[:, ::-1])
    assert np.max(np.abs(ordered)) > 1e-3  # Avoid a vacuous all-zero order check.
    np.testing.assert_allclose(reversed_covariance, covariance, atol=2e-14, rtol=2e-13)
    np.testing.assert_allclose(reversed_symmetric, symmetric, atol=2e-14, rtol=2e-13)
    np.testing.assert_allclose(reversed_order, -ordered, atol=2e-14, rtol=2e-13)


def test_centered_moments_ignore_individual_channel_level_offsets():
    rng = np.random.default_rng(912)
    x = rng.normal(size=(4, 24, 6))
    offsets = rng.uniform(-100.0, 100.0, size=(4, 1, 6))
    for original, shifted in zip(moments(x), moments(x + offsets)):
        np.testing.assert_allclose(shifted, original, atol=2e-13, rtol=2e-12)


def test_centered_moments_vanish_for_sustained_levels():
    levels = np.array([[12.0, 3.0, -2.0, 0.7, 4.0, 600.0],
                       [18.0, 7.0, 1.0, 0.8, 2.0, 900.0]])
    x = np.repeat(levels[:, None], 24, axis=1)
    for value in moments(x):
        np.testing.assert_allclose(value, 0.0, atol=1e-25)
    # Centered moments intentionally do not represent sustained joint magnitude.
    assert np.all(x.mean(1)[:, 0] * x.mean(1)[:, 1] > 0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_cached_ridge_matches_sklearn_for_prefixes_and_extra_blocks(dtype):
    rng = np.random.default_rng(913)
    base = rng.normal(size=(43, 7)).astype(dtype)
    extra = rng.normal(size=(43, 3)).astype(dtype)
    order = rng.permutation(len(base))
    train, test = order[:31], order[31:]
    y = np.column_stack([
        np.clip(0.45 + 0.20 * base[:, 0] + 0.06 * extra[:, 0], 0, 1),
        np.clip(0.65 - 0.17 * base[:, 0] + 0.04 * extra[:, 1], 0, 1),
    ]).astype(np.float64)
    base[test[0], 0] = 15.0
    base[test[1], 0] = -15.0
    weights = np.linspace(0.25, 3.0, len(train)) ** 2
    weights /= weights.sum()
    lambdas = [0.001, 0.07, 1.0]
    cache = RidgeCache(base, y, train, test, weights)
    original_gram, original_rhs = cache.gram.copy(), cache.rhs.copy()
    clipped_cases = 0
    for prefix, extra_block in [(2, None), (7, None), (3, extra), (7, extra)]:
        x = base[:, :prefix].astype(np.float64)
        if extra_block is not None:
            x = np.concatenate([x, extra_block.astype(np.float64)], axis=1)
        expected = []
        for lam in lambdas:
            independent = Ridge(alpha=lam, fit_intercept=True, solver="cholesky")
            independent.fit(x[train], y[train], sample_weight=weights)
            raw = independent.predict(x[test])
            clipped_cases += int(np.any((raw < 0) | (raw > 1)))
            expected.append(np.clip(raw, 0, 1))
        actual = cache.predict((prefix, extra_block), lambdas)
        np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=2e-11)
    assert clipped_cases > 0
    # Solving extra-feature blocks and multiple penalties must not mutate the cache.
    np.testing.assert_array_equal(cache.gram, original_gram)
    np.testing.assert_array_equal(cache.rhs, original_rhs)
    changed_y = y.copy()
    changed_y[test] = 1e6
    changed_cache = RidgeCache(base, changed_y, train, test, weights)
    np.testing.assert_array_equal(
        cache.predict((3, extra), lambdas), changed_cache.predict((3, extra), lambdas)
    )


def test_transform_uses_only_training_imputation_mean_and_scale():
    x = np.array([
        [1.0, np.nan, 5.0, 7.0],
        [1000.0, 1000.0, np.nan, -1000.0],
        [3.0, 2.0, 5.0, 7.0],
        [2.0, np.nan, 5.0, 7.0],
        [5.0, 6.0, 5.0, 7.0],
    ])
    train = np.array([0, 2, 4])
    actual = fit_transform(x, train)
    median = np.array([3.0, 4.0, 5.0, 7.0])
    mean = np.array([3.0, 4.0, 5.0, 7.0])
    scale = np.array([np.sqrt(8 / 3), np.sqrt(8 / 3), 1.0, 1.0])
    filled = np.where(np.isfinite(x), x, median)
    np.testing.assert_allclose(actual, np.clip((filled - mean) / scale, -8, 8), atol=1e-14)
    changed = x.copy()
    changed[1] = [np.inf, -1e20, 1e30, -1e10]
    transformed = fit_transform(changed, train)
    np.testing.assert_array_equal(transformed[[0, 2, 3, 4]], actual[[0, 2, 3, 4]])
    assert np.isfinite(transformed).all()
    assert np.abs(transformed).max() <= 8


def test_donor_is_stable_per_county_and_comes_from_fitting_counties():
    counties = np.array(["01001", "01003", "02001", "01001",
                         "01005", "03001", "02001", "01003"])
    unique = np.unique(counties)
    geography_of = {county: np.array([i + 1.0, 10.0 * (i + 1), 100.0 * (i + 1)])
                    for i, county in enumerate(unique)}
    d = dict(county=counties, geo=np.stack([geography_of[c] for c in counties]), gidx=[2, 0])
    train = np.array([0, 1, 2])
    actual, fallback = donor_geography(d, train, seed=914)
    fit_counties = set(counties[train])
    for county in unique:
        rows = np.flatnonzero(counties == county)
        np.testing.assert_array_equal(actual[rows], np.repeat(actual[rows[:1]], len(rows), axis=0))
        donors = [c for c in fit_counties
                  if np.array_equal(actual[rows[0]], geography_of[c][d["gidx"]])]
        assert len(donors) == 1 and donors[0] != county
        same_state = [c for c in fit_counties if c[:2] == county[:2] and c != county]
        if same_state:
            assert donors[0][:2] == county[:2]
    assert fallback == 2  # Singleton state 02 and held-out-only state 03.
    changed = dict(d, geo=d["geo"].copy())
    heldout_only = ~np.isin(counties, list(fit_counties))
    changed["geo"][heldout_only] = 1e12
    repeat, repeated_fallback = donor_geography(changed, train, seed=914)
    np.testing.assert_array_equal(repeat, actual)
    assert repeated_fallback == fallback
