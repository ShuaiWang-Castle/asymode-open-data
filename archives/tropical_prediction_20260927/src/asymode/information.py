"""Identification geometry of the two-rate transition law.

One step of the two-rate model, on unclipped cells, is the varying-coefficient
regression

    D = (1 - y) U(x) - y R(x) + eps,        phi(y) = (1 - y, -y),  beta = (U, R).

Everything about *whether* and *how precisely* the two rates can be separated at
a driver value is a statement about the conditional design Gram matrix

    Q(x) = E[ phi(Y) phi(Y)^T | X = x ].

This module holds the exact identities and the algebra the tests check. It is
deliberately free of torch and of any data loading: numpy only, so that every
statement can be verified on arbitrary weighted empirical distributions on [0, 1].

Terminology, fixed here and used everywhere else:
* "design information" / "Gram" = Q or A^T A -- no likelihood assumed;
* "Fisher information" is used only under the homoskedastic Gaussian working
  model, where it equals the Gram matrix divided by sigma^2 (determinant x 1/sigma^4);
* "variance ratio" Var(R_hat)/Var(U_hat) = sum (1-y)^2 / sum y^2; the precision
  ratio is its reciprocal. Kish effective sample sizes of the leverage weights are
  a *concentration* diagnostic and are not computed here.
"""
from __future__ import annotations

import numpy as np

__all__ = ["phi", "gram", "gram_stats", "alpha_tau", "from_alpha_tau",
           "one_rate_gap", "one_rate_gap_bruteforce", "ols_covariance",
           "prob_to_hazard", "hazard_to_prob", "step", "rollout_bound",
           "concurrency_direction"]


def phi(y):
    """Row basis phi(y) = (1 - y, -y) for an array of states."""
    y = np.asarray(y, dtype=float)
    return np.stack([1.0 - y, -y], axis=-1)


def _norm_w(y, w):
    y = np.asarray(y, dtype=float)
    if w is None:
        w = np.ones_like(y)
    w = np.asarray(w, dtype=float)
    if (w < 0).any() or w.sum() <= 0:
        raise ValueError("weights must be nonnegative with positive sum")
    return y, w / w.sum()


def gram(y, w=None):
    """Weighted empirical Q = sum_i w_i phi(y_i) phi(y_i)^T with weights normalised to 1."""
    y, w = _norm_w(y, w)
    P = phi(y)
    return (P * w[:, None]).T @ P


def gram_stats(y, w=None):
    """mu, var, m2, A = E[(1-y)^2], B = E[y^2], C = E[y(1-y)], det, eigenvalues, condition."""
    y, w = _norm_w(y, w)
    mu = float(w @ y)
    m2 = float(w @ y ** 2)
    var = m2 - mu ** 2
    A = float(w @ (1 - y) ** 2)
    B = m2
    C = float(w @ (y * (1 - y)))
    Q = gram(y, w)
    ev = np.linalg.eigvalsh(Q)
    lam_min, lam_max = float(ev[0]), float(ev[-1])
    cond = float("inf") if lam_min <= 0 else lam_max / lam_min
    return dict(mu=mu, var=var, m2=m2, A=A, B=B, C=C, det=float(np.linalg.det(Q)),
                lam_min=lam_min, lam_max=lam_max, cond=cond, Q=Q)


def alpha_tau(U, R, mu):
    """(U, R) -> (alpha, tau): alpha = (1-mu)U - mu R is the drift at the mean state,
    tau = U + R the total turnover. The conditional mean is alpha - tau (y - mu)."""
    U, R, mu = (np.asarray(a, dtype=float) for a in (U, R, mu))
    return (1 - mu) * U - mu * R, U + R


def from_alpha_tau(alpha, tau, mu):
    """Inverse: U = alpha + mu tau, R = (1 - mu) tau - alpha."""
    alpha, tau, mu = (np.asarray(a, dtype=float) for a in (alpha, tau, mu))
    return alpha + mu * tau, (1 - mu) * tau - alpha


def concurrency_direction(mu):
    """n_mu = (mu, 1 - mu): the coefficient direction invisible when Y == mu a.s."""
    mu = float(mu)
    return np.array([mu, 1.0 - mu])


def one_rate_gap(U, R, y, w=None):
    """Exact conditional approximation gap of the best one-signed-rate model.

    The state-scaled single signed rate can realise (a, 0) with a >= 0 or (0, b)
    with b >= 0 at a driver value, never both. Against the truth (U, R) >= 0 the
    infimum of E[(phi^T beta0 - phi^T beta)^2 | x] over that union is

        var(y|x) * min( R^2 / A,  U^2 / B ),   A = E[(1-y)^2], B = E[y^2].

    Positive iff var > 0 and both rates are positive.
    """
    s = gram_stats(y, w)
    U, R = float(U), float(R)
    if U < 0 or R < 0:
        raise ValueError("rates must be nonnegative")
    terms = []
    if s["A"] > 0:
        terms.append(R ** 2 / s["A"])
    if s["B"] > 0:
        terms.append(U ** 2 / s["B"])
    if not terms:            # A = B = 0 is impossible on [0,1] since (1-y)^2 + y^2 >= 1/2
        return 0.0
    return s["var"] * min(terms)


def one_rate_gap_bruteforce(U, R, y, w=None, n_grid=20001, a_max=None):
    """Direct constrained minimisation over a dense grid on each axis (test oracle)."""
    y, w = _norm_w(y, w)
    P = phi(y)
    truth = P @ np.array([U, R], dtype=float)
    a_max = a_max if a_max is not None else 4.0 * max(U, R, 1e-3) + 1.0
    grid = np.linspace(0.0, a_max, n_grid)
    # interruption-only (a, 0): error = sum w (truth - (1-y) a)^2
    e_a = ((truth[:, None] - P[:, [0]] * grid[None, :]) ** 2 * w[:, None]).sum(0)
    # restoration-only (0, b): error = sum w (truth - (-y) b)^2
    e_b = ((truth[:, None] - P[:, [1]] * grid[None, :]) ** 2 * w[:, None]).sum(0)
    return float(min(e_a.min(), e_b.min()))


def ols_covariance(y, sigma2=1.0):
    """sigma^2 (A^T A)^{-1} for the fixed design with rows phi(y_i); also returns
    det(A^T A) and the two closed forms sum_{i<j}(y_i-y_j)^2 and N^2 var_hat."""
    y = np.asarray(y, dtype=float)
    A = phi(y)
    G = A.T @ A
    N = len(y)
    det = float(np.linalg.det(G))
    pair = float(((y[:, None] - y[None, :]) ** 2).sum() / 2.0)
    vhat = float(((y - y.mean()) ** 2).mean())
    cov = sigma2 * np.linalg.inv(G) if det > 0 else None
    return dict(G=G, det=det, pairwise=pair, n2var=N ** 2 * vhat, cov=cov, N=N, vhat=vhat)


def prob_to_hazard(u, r, delta=1.0):
    """Transition components (u, r), u + r < 1  ->  constant hazards over an interval.

    p = u + r, Lambda = -log(1 - p)/delta, lambda_u = Lambda u/p, lambda_r = Lambda r/p;
    zero convention: p = 0 -> both hazards 0.
    """
    u, r = np.asarray(u, dtype=float), np.asarray(r, dtype=float)
    p = u + r
    if (p >= 1).any() or (u < 0).any() or (r < 0).any():
        raise ValueError("need u, r >= 0 and u + r < 1")
    Lam = -np.log1p(-p) / delta
    with np.errstate(invalid="ignore", divide="ignore"):
        lu = np.where(p > 0, Lam * u / p, 0.0)
        lr = np.where(p > 0, Lam * r / p, 0.0)
    return lu, lr


def hazard_to_prob(lu, lr, delta=1.0):
    """Inverse of prob_to_hazard: p = 1 - exp(-(lu + lr) delta), u = p lu/(lu+lr), r = p lr/(lu+lr)."""
    lu, lr = np.asarray(lu, dtype=float), np.asarray(lr, dtype=float)
    if (lu < 0).any() or (lr < 0).any():
        raise ValueError("hazards must be nonnegative")
    Lam = lu + lr
    p = -np.expm1(-Lam * delta)
    with np.errstate(invalid="ignore", divide="ignore"):
        u = np.where(Lam > 0, p * lu / Lam, 0.0)
        r = np.where(Lam > 0, p * lr / Lam, 0.0)
    return u, r


def step(y, u, r):
    """One unclipped step y + u(1-y) - r y = u + (1 - u - r) y."""
    return u + (1.0 - u - r) * np.asarray(y, dtype=float)


def ct_step(y, lu, lr, delta=1.0):
    """Exact interval transition of dY/dt = lu (1 - Y) - lr Y with constant hazards."""
    Lam = lu + lr
    q = np.where(Lam > 0, lu / np.where(Lam > 0, Lam, 1.0), 0.0)
    return q + (np.asarray(y, dtype=float) - q) * np.exp(-Lam * delta)


def rollout_bound(e0, rho, delta):
    """Time-varying product-sum bound e_h <= e0 prod rho_k + sum_j delta_j prod_{k>j} rho_k."""
    rho, delta = np.asarray(rho, dtype=float), np.asarray(delta, dtype=float)
    h = len(rho)
    b = float(e0) * float(np.prod(rho))
    for j in range(h):
        b += delta[j] * float(np.prod(rho[j + 1:]))
    return b
