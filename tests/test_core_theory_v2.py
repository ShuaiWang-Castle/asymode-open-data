from __future__ import annotations

import numpy as np


def moments(y: np.ndarray, p: np.ndarray):
    a = float(np.sum(p * (1-y)**2))
    b = float(np.sum(p * y**2))
    c = float(np.sum(p * y * (1-y)))
    v = float(np.sum(p * (y-np.sum(p*y))**2))
    return a, b, c, v


def risk_plus(y, p, u, r, a):
    m = u*(1-y)-r*y
    return float(np.sum(p * (m-a*(1-y))**2))


def risk_minus(y, p, u, r, b):
    m = u*(1-y)-r*y
    return float(np.sum(p * (m+b*y)**2))


def test_projection_decomposition_and_identification_randomized():
    rng = np.random.default_rng(20260903)
    max_err = 0.0
    for _ in range(5000):
        k = int(rng.integers(2, 8))
        y = np.sort(rng.uniform(0, 1, size=k))
        pp = rng.dirichlet(np.ones(k))
        pq = rng.dirichlet(np.ones(k))
        u, r = rng.uniform(1e-4, 0.25, size=2)
        ap, bp, cp, vp = moments(y, pp)
        aq, bq, cq, vq = moments(y, pq)

        # Identification determinant.
        Q = np.array([[aq, -cq], [-cq, bq]])
        assert np.isclose(np.linalg.det(Q), vq, rtol=1e-10, atol=1e-12)
        lmin = np.linalg.eigvalsh(Q)[0]
        assert lmin + 1e-12 >= vq
        assert lmin <= 2*vq + 1e-12

        # Source-to-target interruption projection.
        astar = u-r*cp/ap
        lhs = risk_plus(y, pq, u, r, astar)
        rhs = r*r*vq/aq + r*r*aq*(cp/ap-cq/aq)**2
        max_err = max(max_err, abs(lhs-rhs))
        assert np.isclose(lhs, rhs, rtol=1e-9, atol=1e-11)

        # Source-to-target restoration projection.
        bstar = r-u*cp/bp
        lhs = risk_minus(y, pq, u, r, bstar)
        rhs = u*u*vq/bq + u*u*bq*(cp/bp-cq/bq)**2
        max_err = max(max_err, abs(lhs-rhs))
        assert np.isclose(lhs, rhs, rtol=1e-9, atol=1e-11)
    assert max_err < 1e-10


def test_minimax_interruption_regret_formula():
    rng = np.random.default_rng(17)
    for _ in range(1000):
        y = np.sort(rng.uniform(0, 1, size=5))
        pp = rng.dirichlet(np.ones(5))
        pq = rng.dirichlet(np.ones(5))
        u, r = rng.uniform(1e-3, 0.2, size=2)
        ap, _, cp, _ = moments(y, pp)
        aq, _, cq, _ = moments(y, pq)
        a_p = u-r*cp/ap
        a_q = u-r*cq/aq
        theory = ap*aq/(np.sqrt(ap)+np.sqrt(aq))**2 * (a_p-a_q)**2
        # Weighted midpoint equalizes the two regrets.
        a_mm = (np.sqrt(ap)*a_p + np.sqrt(aq)*a_q)/(np.sqrt(ap)+np.sqrt(aq))
        observed = max(ap*(a_mm-a_p)**2, aq*(a_mm-a_q)**2)
        assert np.isclose(observed, theory, rtol=1e-10, atol=1e-12)
        # Dense numerical grid cannot improve it beyond discretization tolerance.
        lo, hi = sorted((a_p, a_q))
        grid = np.linspace(lo, hi, 10001)
        numeric = np.min(np.maximum(ap*(grid-a_p)**2, aq*(grid-a_q)**2))
        assert numeric >= theory-1e-10
        assert numeric <= theory+1e-7


def test_zero_shift_and_zero_omitted_rate_boundaries():
    y = np.array([0.0, 0.2, 0.8])
    p = np.array([0.2, 0.5, 0.3])
    a, b, c, v = moments(y, p)
    u, r = 0.08, 0.03
    astar = u-r*c/a
    assert np.isclose(risk_plus(y, p, u, r, astar), r*r*v/a)
    bstar = r-u*c/b
    assert np.isclose(risk_minus(y, p, u, r, bstar), u*u*v/b)
    assert np.isclose(risk_plus(y, p, u, 0.0, u), 0.0)
    assert np.isclose(risk_minus(y, p, 0.0, r, r), 0.0)
