"""Two-flow (source-pool) recursion with state-independent rates, fitted on a design of initial states.

For one forcing path, the rates (u_h, r_h), h = 0..H-1, are shared by every initial state z:
    g_0(z) = z,   g_{h+1}(z) = u_h + s_h g_h(z),   s_h = 1 - u_h - r_h,
with u_h, r_h >= 0 and u_h + r_h <= 1 (the closure of the softmax parameterisation used by the neural model).
Parameterisation used for fitting: s_h in [0,1], w_h in [0,1], u_h = (1-s_h) w_h, r_h = (1-s_h)(1-w_h).
This covers the feasible set exactly, so box-constrained L-BFGS-B fits the constrained least-squares problem.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize


def path(s, w, z):
    """Paths g_1..g_H for initial states z; returns array (len(z), H)."""
    H = len(s)
    u = (1 - s) * w
    g = np.empty((len(z), H + 1)); g[:, 0] = z
    for h in range(H):
        g[:, h + 1] = u[h] + s[h] * g[:, h]
    return g[:, 1:]


def loss_grad(theta, z, T, weight=None):
    """Squared path error sum_{z,h} W (g - T)^2 and its gradient with respect to theta = (s, w)."""
    H = T.shape[1]
    s, w = theta[:H], theta[H:]
    u = (1 - s) * w
    g = np.empty((len(z), H + 1)); g[:, 0] = z
    for h in range(H):
        g[:, h + 1] = u[h] + s[h] * g[:, h]
    W = np.ones_like(T) if weight is None else weight
    res = g[:, 1:] - T
    loss = float(np.sum(W * res ** 2))
    lam = 2 * W[:, H - 1] * res[:, H - 1]           # dL/dg_H
    dL_du = np.empty(H); dL_ds = np.empty(H)
    for h in range(H - 1, -1, -1):
        dL_du[h] = lam.sum()
        dL_ds[h] = (lam * g[:, h]).sum()
        if h > 0:
            lam = s[h] * lam + 2 * W[:, h - 1] * res[:, h - 1]
    grad_s = dL_ds - dL_du * w
    grad_w = dL_du * (1 - s)
    return loss, np.concatenate([grad_s, grad_w])


def rates_from_affine(T, z):
    """Reconstruct (s, u, r) that would reproduce the per-horizon affine fit of T exactly (may be infeasible)."""
    X = np.c_[np.ones_like(z), z]
    coef, *_ = np.linalg.lstsq(X, T, rcond=None)          # (2, H): b_h, a_h
    b, a = coef[0], coef[1]
    a_prev = np.r_[1.0, a[:-1]]; b_prev = np.r_[0.0, b[:-1]]
    with np.errstate(divide='ignore', invalid='ignore'):
        s = np.where(np.abs(a_prev) > 1e-12, a / a_prev, np.nan)
    u = b - s * b_prev
    r = 1 - s - u
    return {'a': a, 'b': b, 's': s, 'u': u, 'r': r}


def to_theta(s, u):
    s = np.clip(s, 0, 1); u = np.clip(u, 0, None)
    with np.errstate(divide='ignore', invalid='ignore'):
        w = np.where(1 - s > 1e-12, u / (1 - s), 0.5)
    return np.concatenate([s, np.clip(w, 0, 1)])


def fit(T, z, starts=None, n_random=6, seed=0, weight=None, tol=1e-15):
    """Constrained least-squares fit of the two-flow recursion to target paths T (len(z) x H)."""
    H = T.shape[1]
    rng = np.random.default_rng(seed)
    cand = [] if starts is None else list(starts)
    aff = rates_from_affine(T, z)
    s0 = np.nan_to_num(aff['s'], nan=0.9); cand.append(to_theta(s0, np.nan_to_num(aff['u'], nan=0.0)))
    for _ in range(n_random):
        cand.append(np.concatenate([rng.uniform(0.5, 1.0, H), rng.uniform(0, 1, H)]))
    best = None
    for th0 in cand:
        res = minimize(loss_grad, th0, args=(z, T, weight), jac=True, method='L-BFGS-B',
                       bounds=[(0.0, 1.0)] * (2 * H), options={'maxiter': 20000, 'ftol': tol, 'gtol': 1e-13, 'maxcor': 30})
        if best is None or res.fun < best.fun:
            best = res
    th = best.x
    s, w = th[:H], th[H:]
    u = (1 - s) * w; r = (1 - s) * (1 - w)
    return {'theta': th, 's': s, 'w': w, 'u': u, 'r': r, 'loss': float(best.fun), 'path': path(s, w, z),
            'converged': bool(best.success), 'message': str(best.message)}


def active_set(fitres, tol=1e-7):
    u, r, s = fitres['u'], fitres['r'], fitres['s']
    return {'u_zero': np.flatnonzero(u < tol).tolist(), 'r_zero': np.flatnonzero(r < tol).tolist(),
            's_zero': np.flatnonzero(s < tol).tolist(), 's_one': np.flatnonzero(s > 1 - tol).tolist()}


def affine_projector(z):
    X = np.c_[np.ones_like(z), z]
    return X @ np.linalg.solve(X.T @ X, X.T)
