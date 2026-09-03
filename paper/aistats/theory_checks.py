#!/usr/bin/env python3
"""Numerical unit checks for the one-flow-versus-two-flow theory.

This script verifies algebraic identities and boundary cases. It is not an
empirical forecasting experiment and does not generate paper performance claims.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SEED = 20260903
RNG = np.random.default_rng(SEED)


def moments(y: np.ndarray, w: np.ndarray) -> tuple[float, float, float, float]:
    mu = float(np.dot(w, y))
    a = float(np.dot(w, (1.0 - y) ** 2))
    b = float(np.dot(w, y**2))
    c = float(np.dot(w, y * (1.0 - y)))
    v = float(np.dot(w, (y - mu) ** 2))
    return a, b, c, v


def ray_risk(y: np.ndarray, w: np.ndarray, d: float, r: float) -> float:
    target = d * (1.0 - y) - r * y
    x_d = 1.0 - y
    x_r = y
    a = max(0.0, float(np.dot(w, x_d * target) / max(np.dot(w, x_d**2), 1e-300)))
    b = max(0.0, float(np.dot(w, -x_r * target) / max(np.dot(w, x_r**2), 1e-300)))
    loss_d = float(np.dot(w, (target - a * x_d) ** 2))
    loss_r = float(np.dot(w, (target + b * x_r) ** 2))
    return min(loss_d, loss_r)


def check_population(n_trials: int = 100_000) -> dict[str, float | int]:
    max_moment = 0.0
    max_gap = 0.0
    max_eigen_violation = 0.0
    for _ in range(n_trials):
        k = int(RNG.integers(2, 9))
        y = RNG.uniform(0.0, 1.0, size=k)
        w = RNG.dirichlet(np.ones(k))
        d, r = RNG.uniform(0.0, 0.25, size=2)
        a, b, c, v = moments(y, w)
        max_moment = max(max_moment, abs(a * b - c * c - v))
        direct = ray_risk(y, w, float(d), float(r))
        formula = 0.0 if v <= 1e-15 else v * min(r * r / a, d * d / b)
        max_gap = max(max_gap, abs(direct - formula))
        q = np.array([[a, -c], [-c, b]], dtype=float)
        eig = np.linalg.eigvalsh(q)
        violation = max(0.0, v - eig[0], eig[0] - 2.0 * v, 0.5 - eig[1], eig[1] - 1.0)
        max_eigen_violation = max(max_eigen_violation, violation)
    return {
        "trials": n_trials,
        "max_abs_moment_identity_error": max_moment,
        "max_abs_constrained_gap_error": max_gap,
        "max_eigen_bound_violation": max_eigen_violation,
    }


def check_finite_design(n_trials: int = 20_000) -> dict[str, float | int]:
    max_det = 0.0
    for _ in range(n_trials):
        n = int(RNG.integers(2, 80))
        y = RNG.uniform(0.0, 1.0, size=n)
        phi = np.column_stack([1.0 - y, -y])
        lhs = float(np.linalg.det(phi.T @ phi))
        rhs = float(np.sum((y[:, None] - y[None, :]) ** 2) / 2.0)
        rhs2 = float(n * n * np.var(y))
        max_det = max(max_det, abs(lhs - rhs), abs(lhs - rhs2))
    return {"trials": n_trials, "max_abs_determinant_error": max_det}


def check_oracle_span_mc(n_rep: int = 50_000) -> dict[str, float | int]:
    max_mean_error = 0.0
    max_diff_mean_error = 0.0
    checked = 0
    for n in (16, 32, 64, 128):
        for spread in (0.04, 0.08, 0.16, 0.24):
            lo, hi = 0.5 - spread, 0.5 + spread
            y = np.resize(np.array([lo, hi], dtype=float), n)
            phi = np.column_stack([1.0 - y, -y])
            if np.linalg.matrix_rank(phi) < 2:
                continue
            d = r = 0.05
            sigma = 0.05
            m = phi @ np.array([d, r])
            p2 = phi @ np.linalg.inv(phi.T @ phi) @ phi.T
            cols = [phi[:, [0]], phi[:, [1]]]
            projectors = [x @ np.linalg.inv(x.T @ x) @ x.T for x in cols]
            p1 = min(projectors, key=lambda p: np.linalg.norm((np.eye(n) - p) @ m) ** 2)
            g = float(np.linalg.norm((np.eye(n) - p1) @ m) ** 2 / n)
            eps = RNG.normal(0.0, sigma, size=(n_rep, n))
            err2 = np.sum((eps @ p2.T) ** 2, axis=1) / n
            bias = -(np.eye(n) - p1) @ m
            err1_vec = bias[None, :] + eps @ p1.T
            err1 = np.sum(err1_vec**2, axis=1) / n
            theory1 = g + sigma * sigma / n
            theory2 = 2.0 * sigma * sigma / n
            max_mean_error = max(max_mean_error, abs(float(err1.mean()) - theory1), abs(float(err2.mean()) - theory2))
            max_diff_mean_error = max(max_diff_mean_error, abs(float((err1 - err2).mean()) - (g - sigma * sigma / n)))
            checked += 1
    return {
        "replicates_per_cell": n_rep,
        "cells": checked,
        "max_abs_mc_mean_risk_error": max_mean_error,
        "max_abs_mc_mean_difference_error": max_diff_mean_error,
    }


def check_shift(n_trials: int = 50_000) -> dict[str, float | int]:
    max_interior = 0.0
    max_boundary = 0.0
    n_interior = 0
    for _ in range(n_trials):
        def draw():
            k = int(RNG.integers(2, 8))
            yy = RNG.uniform(0.0, 1.0, size=k)
            ww = RNG.dirichlet(np.ones(k))
            return yy, ww, moments(yy, ww)

        yp, wp, mp = draw()
        yq, wq, mq = draw()
        ap, bp, cp, vp = mp
        aq, bq, cq, vq = mq
        d, r = RNG.uniform(0.0, 0.25, size=2)
        astar_p = float(d - r * cp / ap)
        astar_q = float(d - r * cq / aq)
        aplus_p = max(0.0, astar_p)
        aplus_q = max(0.0, astar_q)
        target_q = d * (1.0 - yq) - r * yq
        risk = lambda a: float(np.dot(wq, (target_q - a * (1.0 - yq)) ** 2))
        direct_boundary = risk(aplus_p)
        formula_boundary = risk(aplus_q) + aq * ((aplus_p - astar_q) ** 2 - (aplus_q - astar_q) ** 2)
        max_boundary = max(max_boundary, abs(direct_boundary - formula_boundary))
        if astar_p >= 0.0 and astar_q >= 0.0:
            direct = risk(astar_p)
            formula = r * r * vq / aq + r * r * aq * (cp / ap - cq / aq) ** 2
            max_interior = max(max_interior, abs(direct - formula))
            n_interior += 1
    return {
        "trials": n_trials,
        "interior_cases": n_interior,
        "max_abs_interior_shift_error": max_interior,
        "max_abs_boundary_identity_error": max_boundary,
    }


def check_state_map(n_trials: int = 100_000) -> dict[str, float | int]:
    max_violation = 0.0
    for _ in range(n_trials):
        y = float(RNG.uniform())
        d = float(RNG.uniform(0.0, 1.0))
        r = float(RNG.uniform(0.0, 1.0 - d))
        nxt = y + d * (1.0 - y) - r * y
        max_violation = max(max_violation, max(0.0, -nxt, nxt - 1.0, d - nxt, nxt - (1.0 - r)))
    return {"trials": n_trials, "max_interval_violation": max_violation}


def main() -> None:
    report = {
        "seed": SEED,
        "population": check_population(),
        "finite_design": check_finite_design(),
        "oracle_span_monte_carlo": check_oracle_span_mc(),
        "event_shift": check_shift(),
        "state_map": check_state_map(),
    }
    out = Path(__file__).with_name("THEORY_CHECKS.json")
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

    assert report["population"]["max_abs_moment_identity_error"] < 1e-12
    assert report["population"]["max_abs_constrained_gap_error"] < 1e-12
    assert report["population"]["max_eigen_bound_violation"] < 1e-12
    assert report["finite_design"]["max_abs_determinant_error"] < 1e-9
    assert report["oracle_span_monte_carlo"]["max_abs_mc_mean_risk_error"] < 2e-5
    assert report["oracle_span_monte_carlo"]["max_abs_mc_mean_difference_error"] < 2e-5
    assert report["event_shift"]["max_abs_interior_shift_error"] < 1e-12
    assert report["event_shift"]["max_abs_boundary_identity_error"] < 1e-12
    assert report["state_map"]["max_interval_violation"] < 1e-12


if __name__ == "__main__":
    main()
