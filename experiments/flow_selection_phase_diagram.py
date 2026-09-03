#!/usr/bin/env python3
"""Controlled ablation for the paper's core one-flow versus two-flow theorem.

No neural network is trained.  The script varies only the quantities in the
flow-selection theory: damage D, restoration R, conditional state variance v,
sample size n, and noise variance sigma^2.

It verifies:
  1. the exact constrained one-flow oracle gap;
  2. zero gap when D=0, R=0, or v=0;
  3. the fixed-design risk crossover at Gamma_n = n G_n / sigma^2 = 1;
  4. the identification relation det(Q)=v and v <= lambda_min(Q) <= 2v.

The Monte Carlo comparison uses an oracle-selected one-dimensional linear
branch.  The exact finite-sample equality is therefore an interior/unconstrained
local benchmark favorable to one flow.  With nonnegative coefficient clipping,
the one-flow risk can only increase when the constraint binds.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def moments(y: np.ndarray) -> dict[str, float]:
    a = float(np.mean((1.0 - y) ** 2))
    b = float(np.mean(y ** 2))
    c = float(np.mean(y * (1.0 - y)))
    v = float(np.var(y))
    q = np.array([[a, -c], [-c, b]], dtype=float)
    eig = np.linalg.eigvalsh(q)
    return {
        "A": a,
        "B": b,
        "C": c,
        "v": v,
        "det_Q": float(np.linalg.det(q)),
        "lambda_min_Q": float(eig[0]),
        "lambda_max_Q": float(eig[1]),
    }


def constrained_oracle_gap(y: np.ndarray, damage: float, restoration: float) -> dict[str, float]:
    z_d = 1.0 - y
    z_r = -y
    mean = damage * z_d + restoration * z_r

    a = max(0.0, float(np.dot(z_d, mean) / np.dot(z_d, z_d)))
    b = max(0.0, float(np.dot(z_r, mean) / np.dot(z_r, z_r)))
    risk_d = float(np.mean((mean - a * z_d) ** 2))
    risk_r = float(np.mean((mean - b * z_r) ** 2))

    mom = moments(y)
    formula = mom["v"] * min(
        restoration ** 2 / mom["A"] if mom["A"] > 0 else float("inf"),
        damage ** 2 / mom["B"] if mom["B"] > 0 else float("inf"),
    )
    return {
        "damage_branch_coefficient": a,
        "restoration_branch_coefficient": b,
        "damage_branch_gap": risk_d,
        "restoration_branch_gap": risk_r,
        "oracle_gap_numeric": min(risk_d, risk_r),
        "oracle_gap_formula": float(formula),
        "oracle_branch": "damage" if risk_d <= risk_r else "restoration",
    }


def monte_carlo_cell(
    y: np.ndarray,
    damage: float,
    restoration: float,
    sigma: float,
    reps: int,
    rng: np.random.Generator,
) -> dict[str, float | int | bool]:
    n = len(y)
    x2 = np.column_stack([1.0 - y, -y])
    beta = np.array([damage, restoration], dtype=float)
    mean = x2 @ beta
    rank = int(np.linalg.matrix_rank(x2))

    branches = [x2[:, 0], x2[:, 1]]
    branch_gaps = []
    branch_projectors = []
    for z in branches:
        p = np.outer(z, z) / float(np.dot(z, z))
        branch_projectors.append(p)
        branch_gaps.append(float(np.mean((p @ mean - mean) ** 2)))
    branch = int(np.argmin(branch_gaps))
    p1 = branch_projectors[branch]
    g_n = branch_gaps[branch]

    p2 = x2 @ np.linalg.pinv(x2)
    noise = rng.normal(0.0, sigma, size=(reps, n))
    observed = mean[None, :] + noise
    fit_one = observed @ p1.T
    fit_two = observed @ p2.T
    risk_one = np.mean((fit_one - mean[None, :]) ** 2, axis=1)
    risk_two = np.mean((fit_two - mean[None, :]) ** 2, axis=1)

    theory_one = g_n + sigma ** 2 / n
    theory_two = rank * sigma ** 2 / n
    gamma = n * g_n / sigma ** 2 if sigma > 0 else float("inf")

    return {
        "n": n,
        "rank_two_flow_design": rank,
        "oracle_linear_branch": "damage" if branch == 0 else "restoration",
        "G_n": g_n,
        "Gamma_n": gamma,
        "risk_one_mc": float(np.mean(risk_one)),
        "risk_two_mc": float(np.mean(risk_two)),
        "risk_one_theory": float(theory_one),
        "risk_two_theory": float(theory_two),
        "two_flow_better_mc": bool(np.mean(risk_two) < np.mean(risk_one)),
        "two_flow_better_theory": bool(rank == 2 and gamma > 1.0),
    }


def two_point_states(n: int, mu: float, variance: float) -> np.ndarray:
    if n % 2:
        raise ValueError("n must be even so the two-point design is exactly balanced")
    if variance < 0:
        raise ValueError("variance must be nonnegative")
    spread = math.sqrt(variance)
    if mu - spread < 0 or mu + spread > 1:
        raise ValueError("requested variance puts the two-point support outside [0,1]")
    if variance == 0:
        return np.full(n, mu, dtype=float)
    return np.tile(np.array([mu - spread, mu + spread], dtype=float), n // 2)


def random_formula_audit(rng: np.random.Generator, trials: int = 10_000) -> float:
    max_error = 0.0
    for _ in range(trials):
        n = int(rng.integers(2, 80))
        y = rng.random(n)
        damage = float(10 ** rng.uniform(-3, -0.1))
        restoration = float(10 ** rng.uniform(-3, -0.1))
        gap = constrained_oracle_gap(y, damage, restoration)
        max_error = max(
            max_error,
            abs(gap["oracle_gap_numeric"] - gap["oracle_gap_formula"]),
        )
    return max_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reps", type=int, default=5000)
    parser.add_argument("--damage", type=float, default=0.05)
    parser.add_argument("--restoration", type=float, default=0.05)
    parser.add_argument("--mu", type=float, default=0.5)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--n-values", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    parser.add_argument(
        "--v-values",
        type=float,
        nargs="+",
        default=[0.0, 0.0004, 0.0016, 0.0064, 0.0256, 0.0625],
    )
    parser.add_argument("--out", default="results/flow_selection_phase_diagram.json")
    args = parser.parse_args()

    if args.sigma <= 0:
        raise ValueError("sigma must be positive")
    rng = np.random.default_rng(args.seed)

    formula_audit_error = random_formula_audit(rng)
    if formula_audit_error > 1e-10:
        raise RuntimeError(f"oracle-gap formula audit failed: {formula_audit_error}")

    rows = []
    max_det_error = 0.0
    max_risk_error = 0.0
    threshold_mismatches = 0
    threshold_cells = 0

    for n in args.n_values:
        for variance in args.v_values:
            y = two_point_states(n, args.mu, variance)
            mom = moments(y)
            gap = constrained_oracle_gap(y, args.damage, args.restoration)
            mc = monte_carlo_cell(
                y,
                args.damage,
                args.restoration,
                args.sigma,
                args.reps,
                rng,
            )
            max_det_error = max(max_det_error, abs(mom["det_Q"] - mom["v"]))
            max_risk_error = max(
                max_risk_error,
                abs(mc["risk_one_mc"] - mc["risk_one_theory"]),
                abs(mc["risk_two_mc"] - mc["risk_two_theory"]),
            )
            if mc["rank_two_flow_design"] == 2 and abs(mc["Gamma_n"] - 1.0) > 0.15:
                threshold_cells += 1
                threshold_mismatches += int(
                    mc["two_flow_better_mc"] != mc["two_flow_better_theory"]
                )
            rows.append(
                {
                    "damage": args.damage,
                    "restoration": args.restoration,
                    "mu": args.mu,
                    "sigma": args.sigma,
                    "requested_v": variance,
                    **mom,
                    **gap,
                    **mc,
                }
            )

    null_checks = {}
    y_null = two_point_states(64, args.mu, 0.0256)
    for name, damage, restoration in [
        ("damage_absent", 0.0, args.restoration),
        ("restoration_absent", args.damage, 0.0),
    ]:
        g = constrained_oracle_gap(y_null, damage, restoration)
        null_checks[name] = g["oracle_gap_numeric"]
    y_degenerate = two_point_states(64, args.mu, 0.0)
    null_checks["state_variance_zero"] = constrained_oracle_gap(
        y_degenerate, args.damage, args.restoration
    )["oracle_gap_numeric"]

    if max(abs(x) for x in null_checks.values()) > 1e-12:
        raise RuntimeError(f"null-regime gap is nonzero: {null_checks}")
    if max_det_error > 1e-12:
        raise RuntimeError(f"det(Q)=v check failed: {max_det_error}")
    if threshold_mismatches:
        raise RuntimeError(
            f"Monte Carlo threshold disagreed in {threshold_mismatches}/{threshold_cells} "
            "cells away from Gamma=1"
        )

    payload = {
        "config": vars(args),
        "theory_checks": {
            "random_constrained_gap_trials": 10_000,
            "max_constrained_gap_formula_error": formula_audit_error,
            "max_det_Q_minus_v_error": max_det_error,
            "max_monte_carlo_absolute_risk_error": max_risk_error,
            "threshold_cells_away_from_one": threshold_cells,
            "threshold_mismatches": threshold_mismatches,
            "null_checks": null_checks,
        },
        "rows": rows,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["theory_checks"], indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
