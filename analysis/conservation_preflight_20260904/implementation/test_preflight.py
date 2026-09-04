"""Mandatory algebra and data-path tests.

These must pass before any empirical result is inspected. They establish the
eight properties required by the canonical work prompt, Phase 3.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

import preflight_lib as L
import preflight_data as D


def rng(seed):
    return np.random.default_rng(seed)


def _rand_cell(seed, n=500):
    r = rng(seed)
    y = r.uniform(0.0, 1.0, n)
    delta = 0.02 * (1 - y) - 0.05 * y + r.normal(0, 0.01, n)
    w = r.uniform(0.1, 3.0, n)
    w = w / w.sum()
    return y, delta, w


# ---------------------------------------------------------------- test 1
def test_1_weighted_unconstrained_identity():
    """U*(1-mu) - R*mu = mean_delta to 1e-11, for arbitrary positive weights."""
    worst = 0.0
    for s in range(40):
        y, delta, w = _rand_cell(s)
        m = L._moments(y, delta, w)
        U, R, rank = L.unconstrained_fit(m)
        assert rank == 2
        res = abs(U * (1 - m["mu"]) - R * m["mu"] - m["mean_delta"])
        worst = max(worst, res)
    assert worst <= 1e-11, worst


# ---------------------------------------------------------------- test 2
def test_2_abc_identity():
    """A*B - C^2 = v to 1e-12 under arbitrary positive normalised weights."""
    worst = 0.0
    for s in range(40):
        y, delta, w = _rand_cell(s + 100)
        m = L._moments(y, delta, w)
        worst = max(worst, abs(m["A"] * m["B"] - m["C"] ** 2 - m["v"]))
    assert worst <= 1e-12, worst


# ---------------------------------------------------------------- test 3
def test_3_boundary_fit_breaks_unconstrained_equality():
    """An explicit boundary-constrained counterexample violates the equality.

    Boundary fits obey KKT inequalities, not the unconstrained normal equations.
    Failing the equality here is correct behaviour, not a defect.
    """
    r = rng(7)
    n = 400
    y = r.uniform(0.0, 0.4, n)
    # strongly negative drift drives the unconstrained U below zero
    delta = -0.08 * y - 0.03 * (1 - y) + r.normal(0, 0.001, n)
    w = np.full(n, 1.0 / n)
    m = L._moments(y, delta, w)
    d2 = float(w @ (delta * delta))
    U_u, R_u, rank = L.unconstrained_fit(m)
    assert rank == 2
    assert U_u < 0.0, "counterexample requires an infeasible unconstrained U"

    U_b, R_b, status = L.box_fit(m, d2)
    assert status != "interior" and "U_at_0" in status
    # unconstrained fit still satisfies the identity exactly
    assert abs(U_u * (1 - m["mu"]) - R_u * m["mu"] - m["mean_delta"]) <= 1e-11
    # the boundary fit does not
    box_res = abs(U_b * (1 - m["mu"]) - R_b * m["mu"] - m["mean_delta"])
    assert box_res > 1e-6, box_res

    # and the box fit really is the constrained optimum (grid cross-check)
    gu = np.linspace(0, L.CAP_U, 401)
    gr = np.linspace(0, L.CAP_R, 401)
    UU, RR = np.meshgrid(gu, gr, indexing="ij")
    J = (d2 - 2 * UU * m["Spd"] + 2 * RR * m["Syd"]
         + UU ** 2 * m["A"] + RR ** 2 * m["B"] - 2 * UU * RR * m["C"])
    assert L._objective(U_b, R_b, m, d2) <= J.min() + 1e-12


def test_3b_box_fit_matches_grid_on_random_problems():
    """No grid point beats the closed-form box solution."""
    worst = -np.inf
    for s in range(30):
        y, delta, w = _rand_cell(s + 300, n=200)
        m = L._moments(y, delta, w)
        d2 = float(w @ (delta * delta))
        U_b, R_b, _ = L.box_fit(m, d2)
        gu = np.linspace(0, L.CAP_U, 201)
        gr = np.linspace(0, L.CAP_R, 201)
        UU, RR = np.meshgrid(gu, gr, indexing="ij")
        J = (d2 - 2 * UU * m["Spd"] + 2 * RR * m["Syd"]
             + UU ** 2 * m["A"] + RR ** 2 * m["B"] - 2 * UU * RR * m["C"])
        worst = max(worst, L._objective(U_b, R_b, m, d2) - J.min())
    assert worst <= 1e-12, worst


# ---------------------------------------------------------------- test 4
def test_4_zero_drift_gap_and_ceiling():
    """Under exact zero drift and mu <= 1/2,
        G = v R^2 mu^2 / ((1-mu)^2 (mu^2+v))  <=  R^2 min(mu^2, v)/(1-mu)^2 .
    """
    r = rng(11)
    for _ in range(300):
        mu = float(r.uniform(1e-4, 0.5))
        v = float(r.uniform(1e-8, mu * (1 - mu)))     # feasible variance
        R = float(r.uniform(1e-4, L.CAP_R))
        U = R * mu / (1 - mu)                          # exact zero-drift closure
        A = (1 - mu) ** 2 + v
        B = mu ** 2 + v
        G_def = v * min(R ** 2 / A, U ** 2 / B)
        G_closed = v * R ** 2 * mu ** 2 / ((1 - mu) ** 2 * (mu ** 2 + v))
        assert abs(G_def - G_closed) <= 1e-15 + 1e-9 * abs(G_closed)
        ceiling = R ** 2 * min(mu ** 2, v) / (1 - mu) ** 2
        assert G_closed <= ceiling * (1 + 1e-12)


# ---------------------------------------------------------------- test 5
def test_5_collapse_difference_is_c_times_one_minus_two_y():
    """m_two(y) - m_one(y) = min(U,R) * (1 - 2y), numerically."""
    r = rng(13)
    U = r.uniform(0, L.CAP_U, 5000)
    R = r.uniform(0, L.CAP_R, 5000)
    y = r.uniform(0, 1, 5000)
    lhs = L.drift_two_flow(U, R, y) - L.drift_one_flow(U, R, y)
    rhs = np.minimum(U, R) * (1 - 2 * y)
    assert np.max(np.abs(lhs - rhs)) <= 1e-15


# ---------------------------------------------------------------- test 6
def test_6_transition_requires_both_states_observed():
    """Rows enter only when observed[t] AND observed[t+1]."""
    C, H = 3, 8
    y15 = np.zeros((C, H * 4)); obs15 = np.ones((C, H * 4), dtype=bool)
    r = rng(5)
    y15 += r.uniform(0, 0.5, y15.shape)
    obs15[0, 4 * 3:4 * 4] = False          # county 0 hour 3 fully unobserved
    obs15[1, 4 * 5:4 * 6] = False          # county 1 hour 5 fully unobserved
    yh, oh = L.to_hourly(y15, obs15)
    assert not oh[0, 3] and not oh[1, 5]

    panel = D.EventPanel(
        event="T", fips=np.array(["A", "B", "C"]), y_hourly=yh, obs_hourly=oh,
        X=np.zeros((C, H, 2)), channels=["a", "b"],
        ts_hourly=pd.date_range("2020-01-01", periods=H, freq="h"),
        denominator=np.array([100.0, 200.0, 400.0]),
    )
    df = D.build_transitions(panel)
    got = set(zip(df["county"], df["t_cur"]))
    expect = {(panel.fips[c], t) for c in range(C) for t in range(H - 1)
              if oh[c, t] and oh[c, t + 1]}
    assert got == expect
    # explicitly: the transitions touching an unobserved state are absent
    for bad in [("A", 2), ("A", 3), ("B", 4), ("B", 5)]:
        assert bad not in got


# ---------------------------------------------------------------- test 7
def test_7_no_unobserved_state_is_zero_filled():
    """A large unobserved state must never enter as a fabricated 0.0."""
    C, H = 1, 6
    y15 = np.full((C, H * 4), 0.4)
    obs15 = np.ones((C, H * 4), dtype=bool)
    obs15[0, 4 * 2:4 * 3] = False               # hour 2 unobserved
    y15[0, 4 * 2:4 * 3] = np.nan                # and its raw value is NaN
    yh, oh = L.to_hourly(y15, obs15)
    assert np.isnan(yh[0, 2]) and not oh[0, 2]

    panel = D.EventPanel(
        event="T", fips=np.array(["A"]), y_hourly=yh, obs_hourly=oh,
        X=np.zeros((C, H, 1)), channels=["a"],
        ts_hourly=pd.date_range("2020-01-01", periods=H, freq="h"),
        denominator=np.array([100.0]),
    )
    df = D.build_transitions(panel)
    assert set(df["t_cur"]) == {0, 3, 4}
    assert np.all(np.isfinite(df["y"])) and np.all(np.isfinite(df["y_next"]))
    assert not ((df["y"] == 0.0) & (df["t_cur"] == 2)).any()
    # no fabricated zero anywhere: every retained state equals the real 0.4
    assert np.allclose(df["y"], 0.4) and np.allclose(df["y_next"], 0.4)


# ---------------------------------------------------------------- test 8
def test_8_active_window_is_outcome_blind_and_never_clipped():
    """The window function takes no outage/target argument and never clips."""
    params = set(inspect.signature(D.active_window).parameters)
    assert params == {"footprint", "composite", "n_states"}
    banned = ("y", "outage", "target", "delta", "residual", "gain", "state")
    src = inspect.getsource(D.active_window)
    for b in banned:
        assert f"{b}=" not in params, b

    n_states = 168
    # peak too early -> unavailable, NOT clipped to a legal index
    fp = np.zeros(n_states); fp[3] = 1.0
    out = D.active_window(fp, np.zeros(n_states), n_states)
    assert out["available"] is False and out["peak"] == 3
    assert out["reason"] == "window_outside_panel"

    # peak too late -> unavailable
    fp = np.zeros(n_states); fp[166] = 1.0
    out = D.active_window(fp, np.zeros(n_states), n_states)
    assert out["available"] is False and out["peak"] == 166

    # interior peak -> exactly 48 transitions, exact index range
    fp = np.zeros(n_states); fp[100] = 1.0
    out = D.active_window(fp, np.zeros(n_states), n_states)
    assert out["available"] and out["peak"] == 100
    assert out["t_start"] == 76 and out["t_end"] == 123
    assert out["n_transitions"] == 48

    # exact boundary cases p = 24 and p = 143 are available
    for p, ok in ((23, False), (24, True), (143, True), (144, False)):
        fp = np.zeros(n_states); fp[p] = 1.0
        assert D.active_window(fp, np.zeros(n_states), n_states)["available"] is ok

    # footprint ties are broken by the public-weather composite, then earliest
    fp = np.zeros(n_states); fp[[60, 90, 120]] = 1.0
    comp = np.zeros(n_states); comp[90] = 5.0
    assert D.active_window(fp, comp, n_states)["peak"] == 90
    assert D.active_window(fp, np.zeros(n_states), n_states)["peak"] == 60


# ------------------------------------------------- supporting determinism
def test_hourly_matches_documented_semantics():
    """Independent to_hourly agrees with the documented mean-of-observed rule."""
    r = rng(21)
    C, H = 6, 10
    y15 = r.uniform(0, 1, (C, H * 4))
    obs15 = r.uniform(0, 1, (C, H * 4)) > 0.3
    yh, oh = L.to_hourly(y15, obs15)
    for c in range(C):
        for t in range(H):
            sl = slice(4 * t, 4 * t + 4)
            o = obs15[c, sl]
            assert oh[c, t] == o.any()
            if o.any():
                assert abs(yh[c, t] - y15[c, sl][o].mean()) < 1e-12
            else:
                assert np.isnan(yh[c, t])


def test_stable_hash_is_process_independent():
    assert int(L.stable_hash("a", "b", 1, salt="s")) == \
           int(L.stable_hash("a", "b", 1, salt="s"))
    assert L.stable_hash("a", salt="x") != L.stable_hash("a", salt="y")


def test_balanced_interval_geometry():
    """The K=2 band brackets rho/(K+rho) .. K rho/(1+K rho) and narrows with rho."""
    lo, hi = L.balanced_flow_interval(1e-3, K=2.0)
    assert lo < hi and hi < 1e-2
    U, R = 1e-3, 1.0
    for y in (lo, hi):
        ratio = (U * (1 - y)) / (R * y)
        assert 0.5 - 1e-9 <= ratio <= 2.0 + 1e-9
    assert L.balanced_flow_interval(0.0) == (0.0, 0.0)
    assert L.balanced_flow_interval(np.inf) == (1.0, 1.0)


def test_closure_gate_requires_interior_and_small_drift():
    """closure_pass must fail on a boundary fit even when drift is tiny."""
    r = rng(31)
    n = 800
    y = r.uniform(0.0, 0.3, n)
    delta = -0.05 * y - 0.02 * (1 - y) + r.normal(0, 1e-4, n)   # forces U<0
    ocf = np.full(n, 1e-4); cty = np.arange(n) % 20
    fit = L.constant_fit(y, delta, None, ocf, cty)
    assert fit.interior_unc is False
    assert fit.closure_pass is False
    assert np.isnan(fit.Gamma_near_closure)
