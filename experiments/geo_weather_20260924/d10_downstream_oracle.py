"""D10 fixed-downstream, masked trajectory oracle; artificial checks by default.

No file, checkpoint, model or label is loaded by this module. ``project`` solves
only the numerical problem passed by an authorized caller. Recovery, occurrence
gate and background stay fixed; a future-truth oracle chooses the conditional
damage rate independently at each hour. This relaxes the damage network and its
logit smoother, and does not identify a single downstream component or a cause.

Both a genuinely feasible primal path and a nonnegative-multiplier, box-minimum
dual lower bound are retained. Optimizer success alone is never a certificate.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'

import argparse
import json
import time

import numpy as np
from scipy.optimize import minimize

CONDITIONAL_CAP = .5
RECOVERY_CAP = .5
BACKGROUND_CAP = .015
GAP_TOLERANCE = 1e-8
FEASIBILITY_TOLERANCE = 1e-10
RECONSTRUCTION_TOLERANCE = 1e-10
PRIMAL_MAXITER = 300
DUAL_MAXITER = 1000


def _outcome(y, m, y0):
    y, m = np.asarray(y, np.float64), np.asarray(m)
    if y.ndim != 1 or not 1 <= len(y) <= 144 or m.shape != y.shape:
        raise ValueError('Expected aligned 1..144-hour vectors')
    if not np.isin(m, [0, 1]).all():
        raise ValueError('Observation mask must be binary')
    m = m.astype(bool)
    if not np.isfinite(y[m]).all() or np.any((y[m] < 0) | (y[m] > 1)):
        raise ValueError('Observed outcomes must be finite proportions')
    y0 = float(y0)
    if not np.isfinite(y0) or not 0 <= y0 <= 1:
        raise ValueError('Origin must be an observed proportion')
    return np.where(m, y, 0.), m, y0


def _band_constraints(y0, lower_offset, upper_offset, lower_coefficient, upper_coefficient):
    """Two bidiagonal bands A p <= b; the [0,1] box is separate."""
    n = len(lower_offset)
    a, b = np.zeros((2*n, n)), np.zeros(2*n)
    for t in range(n):
        a[t, t], a[n+t, t] = -1., 1.
        b[t], b[n+t] = -lower_offset[t], upper_offset[t]
        if t:
            a[t, t-1] = lower_coefficient[t]
            a[n+t, t-1] = -upper_coefficient[t]
        else:
            b[t] -= lower_coefficient[t]*y0
            b[n+t] += upper_coefficient[t]*y0
    return a, b


def _feasible_path(proposal, y0, lo, hi, al, ah):
    path, previous = np.empty_like(proposal, dtype=np.float64), float(y0)
    for t, value in enumerate(proposal):
        lower, upper = lo[t]+al[t]*previous, hi[t]+ah[t]*previous
        if lower > upper+FEASIBILITY_TOLERANCE or lower < -FEASIBILITY_TOLERANCE or upper > 1+FEASIBILITY_TOLERANCE:
            raise ArithmeticError('Invalid feasible band')
        path[t] = np.clip(value if np.isfinite(value) else previous, lower, upper)
        previous = path[t]
    return path


def _solve_band(y, m, y0, lo, hi, al, ah):
    """Independent primal and box-minimum dual for one fixed convex band QP."""
    n = len(y)
    a, b = _band_constraints(y0, lo, hi, al, ah)
    # Missing hours remain state variables and bridge observed hours.
    x0 = _feasible_path(np.where(m, y, np.nan), y0, lo, hi, al, ah)
    objective = lambda x: float(np.dot(np.where(m, x-y, 0.), np.where(m, x-y, 0.)))
    started = time.monotonic()
    fit = minimize(objective, x0, jac=lambda x: 2*np.where(m, x-y, 0.),
        method='SLSQP', bounds=[(0., 1.)]*n,
        constraints={'type': 'ineq', 'fun': lambda x: b-a@x, 'jac': lambda x: -a},
        options={'ftol': 1e-11, 'maxiter': PRIMAL_MAXITER})
    # A solver's nearly feasible output is not a valid upper bound until
    # projected forward through the exact interval defined by its prior state.
    path = _feasible_path(fit.x, y0, lo, hi, al, ah)
    upper = objective(path)

    def dual(multiplier, gradient=False):
        q = a.T@multiplier
        # On missing hours the loss is zero. Minimize q_t*v over [0,1]
        # directly; dividing by the zero observation weight would be invalid.
        v = np.where(m, np.clip(y-.5*q, 0., 1.), (q < 0).astype(float))
        value = objective(v)+float(q@v)-float(multiplier@b)
        return (-value, b-a@v) if gradient else (value, v, q)

    multipliers = np.asarray(getattr(fit, 'multipliers', np.zeros(2*n)), np.float64)
    if multipliers.shape != (2*n,) or not np.isfinite(multipliers).all():
        multipliers = np.zeros(2*n)
    multipliers = np.maximum(multipliers, 0.)
    candidates = [multipliers.copy()]
    initial_value = dual(multipliers)[0]
    refined = upper-max(0., initial_value) > GAP_TOLERANCE
    refinement = None
    if refined:
        refinement = minimize(lambda z: dual(z, True), multipliers, jac=True,
            method='L-BFGS-B', bounds=[(0., None)]*(2*n),
            options={'maxiter': DUAL_MAXITER, 'ftol': 1e-15, 'gtol': 1e-9})
        if np.isfinite(refinement.x).all():
            candidates.append(np.maximum(refinement.x, 0.))
    # Guard cancellation in the evaluated dual, especially on missing-hour
    # box terms. This is a floating-point numerical certificate, not an exact
    # interval-arithmetic proof. It never uses optimizer success as a bound.
    lower, raw_dual, roundoff_guard = 0., 0., 0.
    for candidate in candidates:
        value, v, q = dual(candidate)
        guard = 8*(n+2)*np.finfo(np.float64).eps*(
            1.+objective(v)+float(np.sum(np.abs(q*v)))+float(np.sum(np.abs(candidate*b))))
        safe = max(0., value-guard)
        if safe > lower:
            lower, raw_dual, roundoff_guard = safe, value, guard
    band_violation = max(0., float(np.max(a@path-b)))
    box_violation = max(0., float(-path.min()), float(path.max()-1.))
    if band_violation > FEASIBILITY_TOLERANCE or box_violation > FEASIBILITY_TOLERANCE:
        raise ArithmeticError('Projected primal path is not feasible')
    if lower > upper+FEASIBILITY_TOLERANCE:
        raise ArithmeticError('Primal-dual weak duality failed')
    gap = max(0., upper-lower)
    certificate = dict(lower=lower, upper=upper, gap=gap,
        gap_tolerance=GAP_TOLERANCE, gap_certified=bool(gap <= GAP_TOLERANCE),
        maximum_band_violation=band_violation, maximum_box_violation=box_violation,
        feasibility_tolerance=FEASIBILITY_TOLERANCE, primal_feasible=True,
        primal_success=bool(fit.success), primal_status=int(fit.status),
        primal_iterations=int(fit.nit), primal_iteration_budget=PRIMAL_MAXITER,
        dual_refined=refined, dual_success=None if refinement is None else bool(refinement.success),
        dual_iterations=0 if refinement is None else int(refinement.nit),
        dual_iteration_budget=DUAL_MAXITER, raw_dual_value=raw_dual,
        dual_roundoff_guard=roundoff_guard, numerical_dtype='float64',
        observed_hours=int(m.sum()), hours=n, elapsed_seconds=time.monotonic()-started)
    return path, certificate


def project(y, m, y0, r, gate, background):
    """Return (path, certificate) with fixed r/gate/bg and free c_t in [0,.5].

    The loss is the unweighted observed-hour SSE for this unit. Multiply both
    bounds by its original positive design weight for aggregate summaries.
    Legal stock transitions are convex combinations of u and 1-r, so clipping
    is inactive. Bounds remain linear even if 1-r-u is slightly negative.
    """
    y, m, y0 = _outcome(y, m, y0)
    rates = [np.asarray(value, np.float64) for value in (r, gate, background)]
    for name, rate, cap in zip(('r','gate','background'), rates, (RECOVERY_CAP, 1., BACKGROUND_CAP)):
        if rate.shape != y.shape or not np.isfinite(rate).all() or np.any((rate < 0) | (rate > cap)):
            raise ValueError('Invalid fixed '+name+' trajectory')
    r, gate, background = rates
    lo, hi = background, background+CONDITIONAL_CAP*gate
    al, ah = 1-r-lo, 1-r-hi
    path, certificate = _solve_band(y, m, y0, lo, hi, al, ah)
    # Recover a legal conditional-rate witness and independently replay each
    # stock transition. Degenerate gate=0 or previous stock=1 needs no division.
    previous, residual, c_min, c_max = y0, 0., CONDITIONAL_CAP, 0.
    for t in range(len(path)):
        needed = lo[t] if previous == 1 else (path[t]-(1-r[t])*previous)/(1-previous)
        u = float(np.clip(needed, lo[t], hi[t]))
        c = 0. if gate[t] == 0 else float(np.clip((u-background[t])/gate[t], 0., CONDITIONAL_CAP))
        u = background[t]+gate[t]*c
        replay = previous+u*(1-previous)-r[t]*previous
        residual = max(residual, abs(float(replay-path[t])))
        c_min, c_max = min(c_min,c), max(c_max,c)
        previous = float(path[t])
    if residual > RECONSTRUCTION_TOLERANCE:
        raise ArithmeticError('Legal-rate reconstruction did not reproduce primal path')
    certificate.update(rate_reconstruct_maximum_residual=residual,
        rate_reconstruct_tolerance=RECONSTRUCTION_TOLERANCE,
        conditional_rate_minimum=c_min, conditional_rate_maximum=c_max,
        fixed_downstream=dict(recovery=True, occurrence_gate=True, background=True),
        free_conditional_rate_cap=CONDITIONAL_CAP, design_weight_in_objective=False,
        relaxes_damage_network_and_logit_smoother=True, future_truth_oracle=True,
        bound_interpretation='Minimum masked SSE inside this fixed downstream rate envelope; not a causal component attribution or achievable learned-model score.')
    return path, certificate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(dict(status='no_real_execution',file_reads=False,model_forward=False,
        public_api='project(y,m,y0,r,gate,background)',gap_tolerance=GAP_TOLERANCE)))


if __name__ == '__main__':
    main()
