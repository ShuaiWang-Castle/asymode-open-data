"""Randomised property tests for the identification geometry (A3 items 1-8).

Every theorem statement in docs/CC_THEORY_CHECK.md has a test here; a statement
without a passing test is not quoted in the paper.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from asymode import information as I  # noqa: E402

RNG = np.random.default_rng(20260902)


def _random_dist(rng, n=None, endpoint_mass=0.3):
    """Weighted empirical distribution on [0,1], often with mass at 0 and 1 and
    sometimes highly imbalanced or degenerate."""
    n = n or int(rng.integers(2, 40))
    kind = rng.integers(0, 5)
    if kind == 0:
        y = rng.uniform(0, 1, n)
    elif kind == 1:                      # mostly zeros, a few small positives
        y = np.where(rng.uniform(size=n) < 0.8, 0.0, rng.uniform(0, 0.05, n))
    elif kind == 2:                      # endpoints only
        y = rng.choice([0.0, 1.0], size=n)
    elif kind == 3:                      # point mass (degenerate)
        y = np.full(n, rng.uniform(0, 1))
    else:                                # beta-shaped
        y = rng.beta(0.3, 3.0, n)
    w = rng.uniform(0.1, 2.0, n)
    return y, w


# 1. determinant identity ----------------------------------------------------
@pytest.mark.parametrize("trial", range(300))
def test_det_equals_conditional_variance(trial):
    y, w = _random_dist(RNG)
    s = I.gram_stats(y, w)
    assert s["det"] == pytest.approx(s["var"], abs=1e-12)
    # A B - C^2 = var as well (Theorem 5's denominator identity)
    assert s["A"] * s["B"] - s["C"] ** 2 == pytest.approx(s["var"], abs=1e-12)


# 2. eigenvalue bounds ----------------------------------------------------------
@pytest.mark.parametrize("trial", range(300))
def test_eigenvalue_bounds(trial):
    y, w = _random_dist(RNG)
    s = I.gram_stats(y, w)
    assert 0.5 - 1e-12 <= s["lam_max"] <= 1.0 + 1e-12
    assert s["var"] - 1e-12 <= s["lam_min"] <= 2 * s["var"] + 1e-12


# 3. centred-coordinate round trip and diagonal Gram ---------------------------
@pytest.mark.parametrize("trial", range(200))
def test_alpha_tau_round_trip_and_orthogonality(trial):
    y, w = _random_dist(RNG)
    s = I.gram_stats(y, w)
    U, R = RNG.uniform(0, 0.5, 2)
    a, t = I.alpha_tau(U, R, s["mu"])
    U2, R2 = I.from_alpha_tau(a, t, s["mu"])
    assert U2 == pytest.approx(U, abs=1e-12) and R2 == pytest.approx(R, abs=1e-12)
    # conditional mean in the two coordinate systems agrees pointwise
    yy = np.asarray(y)
    m_ur = (1 - yy) * U - yy * R
    m_at = a - t * (yy - s["mu"])
    assert np.allclose(m_ur, m_at, atol=1e-12)
    # Gram in (alpha, tau) coordinates is diag(1, var): basis (1, -(y - mu))
    wn = np.asarray(w) / np.sum(w)
    B = np.stack([np.ones_like(yy), -(yy - s["mu"])], -1)
    Qt = (B * wn[:, None]).T @ B
    assert np.allclose(Qt, np.diag([1.0, s["var"]]), atol=1e-12)


# 4. null direction at zero dispersion ------------------------------------------
@pytest.mark.parametrize("mu", [0.0, 0.2, 0.5, 0.9, 1.0])
def test_null_direction_at_point_mass(mu):
    y = np.full(7, mu)
    s = I.gram_stats(y)
    assert s["var"] == pytest.approx(0.0, abs=1e-15)
    assert s["lam_min"] == pytest.approx(0.0, abs=1e-12)
    n = I.concurrency_direction(mu)
    assert float(I.phi(mu) @ n) == pytest.approx(0.0, abs=1e-15)
    # adding c * n_mu changes the drift by c (mu - y) -> zero at the point mass
    for c in (0.1, 1.0, 3.0):
        assert np.allclose(I.phi(y) @ (c * n), c * (mu - y), atol=1e-15)
    # zero-state blind spot: at y = 0 the R column vanishes; at y = 1 the U column
    if mu == 0.0:
        assert np.all(I.phi(y)[:, 1] == 0)
    if mu == 1.0:
        assert np.all(I.phi(y)[:, 0] == 0)


# 5. exact one-rate approximation gap --------------------------------------------
@pytest.mark.parametrize("trial", range(120))
def test_one_rate_gap_matches_bruteforce(trial):
    y, w = _random_dist(RNG, n=int(RNG.integers(2, 25)))
    U, R = RNG.uniform(0, 0.4, 2)
    if trial % 7 == 0:
        R = 0.0            # one true rate zero -> gap must be 0
    if trial % 11 == 0:
        U = 0.0
    exact = I.one_rate_gap(U, R, y, w)
    brute = I.one_rate_gap_bruteforce(U, R, y, w, n_grid=40001)
    assert exact == pytest.approx(brute, abs=2e-7, rel=1e-3)
    s = I.gram_stats(y, w)
    if s["var"] > 1e-12 and U > 0 and R > 0:
        assert exact > 0
    else:
        assert exact == pytest.approx(0.0, abs=1e-12)


# 6. recurrence invariance on [0,1] ---------------------------------------------
@pytest.mark.parametrize("trial", range(300))
def test_recurrence_maps_unit_interval_into_itself(trial):
    u = RNG.uniform(0, 1); r = RNG.uniform(0, 1 - u)      # u + r <= 1
    y = RNG.uniform(0, 1, 50)
    y1 = I.step(y, u, r)
    assert np.all(y1 >= -1e-15) and np.all(y1 <= 1 + 1e-15)
    # endpoints exactly
    assert I.step(0.0, u, r) == pytest.approx(u) and I.step(1.0, u, r) == pytest.approx(1 - r)
    # slope 1 - u - r in [0, 1]: monotone, non-expansive
    assert 0 <= 1 - u - r <= 1


# 7. exact continuous-time equivalence -------------------------------------------
@pytest.mark.parametrize("trial", range(300))
def test_ct_dt_bijection_and_exact_transition(trial):
    u = RNG.uniform(0, 0.6); r = RNG.uniform(0, 0.99 - u)
    if trial % 10 == 0:
        u = 0.0
    if trial % 13 == 0:
        r = 0.0
    lu, lr = I.prob_to_hazard(u, r)
    u2, r2 = I.hazard_to_prob(lu, lr)
    assert u2 == pytest.approx(u, abs=1e-12) and r2 == pytest.approx(r, abs=1e-12)
    y = RNG.uniform(0, 1, 20)
    assert np.allclose(I.ct_step(y, lu, lr), I.step(y, u, r), atol=1e-12)
    # round trip the other way from hazards
    lu0, lr0 = RNG.uniform(0, 3, 2)
    uu, rr = I.hazard_to_prob(lu0, lr0)
    assert uu + rr < 1
    lu1, lr1 = I.prob_to_hazard(uu, rr)
    assert lu1 == pytest.approx(lu0, rel=1e-10) and lr1 == pytest.approx(lr0, rel=1e-10)


# 8. rollout product-sum bound; counterexample to the constant-rate saturation ----
@pytest.mark.parametrize("trial", range(200))
def test_rollout_product_sum_bound_holds(trial):
    h = int(RNG.integers(1, 48))
    u = RNG.uniform(0, 0.3, h); r = RNG.uniform(0, 0.3, h)           # true rates
    du = RNG.normal(0, 0.02, h); dr = RNG.normal(0, 0.02, h)          # misspecification
    uh, rh = np.clip(u + du, 0, 0.6), np.clip(r + dr, 0, 0.6)
    y = yh = RNG.uniform(0, 1)
    e0 = 0.0
    rho = np.abs(1 - uh - rh)
    # delta_t = sup_y |F_hat(y) - F(y)| = max(|du + (dr... )|) over y in [0,1]: affine in y -> endpoints
    delta = np.maximum(np.abs(uh - u), np.abs((1 - rh) - (1 - r)))
    for t in range(h):
        y = I.step(y, u[t], r[t]); yh = I.step(yh, uh[t], rh[t])
    assert abs(yh - y) <= I.rollout_bound(e0, rho, delta) + 1e-12


def test_constant_rate_saturation_bound_fails_without_lower_rate_bound():
    """Old wording: e_h <= eps * min(h, 1/(U+R)). With U+R -> 0 along the path the
    error grows linearly in h and exceeds any bound of the form eps/(U+R) evaluated
    at a rate bound that is not uniform along the path."""
    h = 48
    eps = 1e-3
    u = np.zeros(h); r = np.zeros(h)                 # total intensity 0 along the path
    uh = np.full(h, eps); rh = np.zeros(h)           # per-step misspecification eps
    y = yh = 0.0
    for t in range(h):
        y = I.step(y, u[t], r[t]); yh = I.step(yh, uh[t], rh[t])
    err = abs(yh - y)
    # linear accumulation: err ~ h * eps (up to the tiny (1-eps)^k factors)
    assert err > 0.9 * h * eps
    # a "saturation" bound eps/(U+R) evaluated with any positive cap-level rate,
    # e.g. U+R = 0.25 (the caps), would claim err <= eps/0.25 = 4 eps: violated
    assert err > eps / 0.25
    # the product-sum bound is still valid
    assert err <= I.rollout_bound(0.0, np.abs(1 - uh - rh), np.full(h, eps)) + 1e-12
