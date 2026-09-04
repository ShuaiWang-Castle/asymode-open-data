"""Exact bounded constant-class fits, and their exact mapping into the network.

Three closed-form problems on observed one-step transitions, all with box
constraints, all solved exactly rather than by descent:

    two-flow  min_{0<=U<=CU, 0<=R<=CR}  sum_i w_i (dY_i - U(1-Y_i) + R Y_i)^2
    U-ray     min_{0<=a<=CU}            sum_i w_i (dY_i - a(1-Y_i))^2
    R-ray     min_{0<=b<=CR}            sum_i w_i (dY_i + b Y_i)^2

The two-flow problem is a two-variable box-constrained least squares. Its solution
is the unconstrained stationary point when that point is inside the box, and
otherwise lies on the boundary, so it is obtained exactly by comparing the
interior candidate against the four edge minima and the four corners. No iterative
solver is used and none is needed.

The `u0-r0` heuristic is deliberately absent: it is prohibited for this study.
"""
from __future__ import annotations

import numpy as np


def _wls_1d(num: float, den: float, lo: float, hi: float) -> float:
    """argmin of a one-dimensional convex quadratic on [lo, hi]."""
    if den <= 0:
        return lo
    return float(np.clip(num / den, lo, hi))


def fit_u_ray(y: np.ndarray, dy: np.ndarray, w: np.ndarray, cap_u: float) -> float:
    """Exact bounded interruption-ray optimum: dY ~ a (1-Y), 0 <= a <= cap_u."""
    p = 1.0 - y
    return _wls_1d(float(np.sum(w * dy * p)), float(np.sum(w * p * p)), 0.0, cap_u)


def fit_r_ray(y: np.ndarray, dy: np.ndarray, w: np.ndarray, cap_r: float) -> float:
    """Exact bounded restoration-ray optimum: dY ~ -b Y, 0 <= b <= cap_r."""
    return _wls_1d(float(np.sum(w * (-dy) * y)), float(np.sum(w * y * y)), 0.0, cap_r)


def fit_two_flow(y: np.ndarray, dy: np.ndarray, w: np.ndarray,
                 cap_u: float, cap_r: float) -> tuple[float, float]:
    """Exact bounded two-flow constant optimum over the box [0,cap_u] x [0,cap_r]."""
    p, q = 1.0 - y, -y
    a11 = float(np.sum(w * p * p)); a22 = float(np.sum(w * q * q))
    a12 = float(np.sum(w * p * q))
    b1 = float(np.sum(w * dy * p)); b2 = float(np.sum(w * dy * q))

    def obj(U, R):
        r = dy - U * p - R * q          # note: R enters through q = -y
        return float(np.sum(w * r * r))

    cands: list[tuple[float, float]] = []
    det = a11 * a22 - a12 * a12
    if det > 0:                                   # interior stationary point
        U = (b1 * a22 - b2 * a12) / det
        R = (b2 * a11 - b1 * a12) / det
        if 0 <= U <= cap_u and 0 <= R <= cap_r:
            cands.append((U, R))
    for R in (0.0, cap_r):                        # horizontal edges
        cands.append((_wls_1d(b1 - R * a12, a11, 0.0, cap_u), R))
    for U in (0.0, cap_u):                        # vertical edges
        cands.append((U, _wls_1d(b2 - U * a12, a22, 0.0, cap_r)))
    for U in (0.0, cap_u):                        # corners
        for R in (0.0, cap_r):
            cands.append((U, R))
    U, R = min(cands, key=lambda c: obj(*c))
    return float(U), float(R)


def logit(p: float, eps: float = 1e-8) -> float:
    p = float(np.clip(p, eps, 1.0 - eps))
    return float(np.log(p / (1.0 - p)))


def modular_init(u0: float, r0: float, cap_u_main: float, cap_bkg: float,
                 cap_r: float) -> dict:
    """Biases that make the modular network reproduce a constant (u0, r0) at update 0.

    Every output weight is zero, the occurrence gate sits at g0 = 0.5 with zero
    weight and bias, and the hold gate has zero weights and bias -3.0 so that a
    constant raw logit passes through unchanged. The desired interruption flow is
    split between the two pathways in proportion to their caps, which makes the
    decomposition deterministic rather than a free choice.
    """
    g0 = 0.5
    share = cap_bkg / (cap_u_main + cap_bkg)
    u_bkg = u0 * share
    u_pulse = u0 - u_bkg
    # Feasibility. The amendment fixes the occurrence gate at g0 = 0.5, so the
    # pulse pathway can emit at most g0 * cap_u_main, while the prescribed split
    # hands it u0 * (1 - share). The representable ceiling on the constant is
    # therefore g0 * cap_u_main / (1 - share), which is strictly below the nominal
    # cap_u_main + cap_bkg. Above it, `logit` would silently clip and the update-0
    # model would NOT reproduce the constant class. Fail closed instead: the
    # 1e-6 reproduction requirement is a gate, not a target to be approximated.
    u_max = g0 * cap_u_main / (1.0 - share)
    r_max = cap_r
    if u0 > u_max + 1e-12 or r0 > r_max + 1e-12:
        raise ValueError(
            f"constant ({u0:.6g}, {r0:.6g}) is outside the representable set of the "
            f"modular initialisation: U0 <= {u_max:.6g} (gate fixed at g0={g0}), "
            f"R0 <= {r_max:.6g}. Clipping here would break the update-0 identity.")
    return dict(
        g0=g0,
        background_bias=logit(u_bkg / cap_bkg),
        raw_u_bias=logit(u_pulse / (g0 * cap_u_main)),
        recovery_bias=logit(r0 / cap_r),
        hold_bias=-3.0,
        u_bkg_target=u_bkg, u_pulse_target=u_pulse,
        u_representable_max=u_max, r_representable_max=r_max,
    )
