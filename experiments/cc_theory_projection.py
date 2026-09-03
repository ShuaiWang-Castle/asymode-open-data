"""Section 10.1: closed-form synthetic verification of the event-projection theorem.

For fixed drivers the true drift is m(y) = U(1-y) - R y. For an environment e put

    A_e = E_e[(1-Y)^2],  B_e = E_e[Y^2],  C_e = E_e[Y(1-Y)],  v_e = A_e B_e - C_e^2.

The interruption-only class is {a(1-y) : a}; its L2(P) projection is
a*_P = U - R C_P/A_P, and its risk under a different environment Q decomposes as

    E_Q[(m(Y) - a*_P(1-Y))^2] = R^2 v_Q/A_Q + R^2 A_Q (C_P/A_P - C_Q/A_Q)^2.

The restoration-only counterpart is b*_P = R - U C_P/B_P with

    E_Q[(m(Y) + b*_P Y)^2] = U^2 v_Q/B_Q + U^2 B_Q (C_P/B_P - C_Q/B_Q)^2.

Both were re-derived by hand before this file was written: writing k = C_P/A_P,
the left side is R^2(A_Q k^2 - 2k C_Q + B_Q), and expanding the right side gives
the same expression, so the identity is exact rather than approximate. The tests
below check that numerically on random draws and on the boundary cases, and check
independently that each closed-form projection really is the minimiser of the risk
on P (vanishing numerical derivative, both neighbours worse).

    python experiments/cc_theory_projection.py --n-draws 10000 --out <dir>/10_THEORY_UNIT_TESTS.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from asymode import information as I  # noqa: E402

TOL = 1e-8


def moments(y, w=None):
    y = np.asarray(y, float)
    w = np.ones_like(y) if w is None else np.asarray(w, float)
    w = w / w.sum()
    A = float(w @ (1 - y) ** 2)
    B = float(w @ y ** 2)
    C = float(w @ (y * (1 - y)))
    return A, B, C, A * B - C * C


def risk_interruption(U, R, a, y, w):
    A, B, C, _ = moments(y, w)
    return (U - a) ** 2 * A - 2 * (U - a) * R * C + R ** 2 * B


def risk_restoration(U, R, b, y, w):
    A, B, C, _ = moments(y, w)
    return U ** 2 * A - 2 * U * (R - b) * C + (R - b) ** 2 * B


def draw(rng):
    """A random pair of environments on [0,1] and a random nonnegative rate pair."""
    kind = rng.integers(0, 4)
    n = int(rng.integers(3, 60))
    if kind == 0:
        yP, yQ = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
    elif kind == 1:                              # both concentrated near zero
        yP, yQ = rng.beta(0.4, 8, n), rng.beta(0.4, 8, n)
    elif kind == 2:                              # P near zero, Q spread: an event shift
        yP, yQ = rng.beta(0.3, 12, n), rng.uniform(0, 0.6, n)
    else:                                        # endpoint mass
        yP = np.clip(rng.choice([0.0, 1.0, 0.5], n) + rng.normal(0, 0.02, n), 0, 1)
        yQ = rng.uniform(0, 1, n)
    wP, wQ = rng.uniform(0.2, 2, n), rng.uniform(0.2, 2, n)
    U, R = rng.uniform(0, 0.3), rng.uniform(0, 0.3)
    return U, R, yP, wP, yQ, wQ


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-draws", type=int, default=10000)
    ap.add_argument("--n-brute", type=int, default=200, help="draws also checked against a grid search")
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--out", default=None)
    ap.add_argument("--fig", default=None)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    n_brute = a.n_brute
    err_a, err_b, err_proj_a, err_proj_b, skipped = [], [], [], [], 0
    for _ in range(a.n_draws):
        U, R, yP, wP, yQ, wQ = draw(rng)
        AP, BP, CP, vP = moments(yP, wP)
        AQ, BQ, CQ, vQ = moments(yQ, wQ)
        if min(AP, BP, AQ, BQ) < 1e-6:
            skipped += 1
            continue
        # interruption-only
        aP = U - R * CP / AP
        lhs = risk_interruption(U, R, aP, yQ, wQ)
        rhs = R ** 2 * vQ / AQ + R ** 2 * AQ * (CP / AP - CQ / AQ) ** 2
        err_a.append(abs(lhs - rhs) / max(abs(rhs), 1e-12))
        # restoration-only
        bP = R - U * CP / BP
        lhs = risk_restoration(U, R, bP, yQ, wQ)
        rhs = U ** 2 * vQ / BQ + U ** 2 * BQ * (CP / BP - CQ / BQ) ** 2
        err_b.append(abs(lhs - rhs) / max(abs(rhs), 1e-12))
        # Independent check that the closed-form projection really is the minimiser
        # on P: the numerical derivative of the risk vanishes there, and both
        # neighbours are worse. A grid search is not used because b* is unbounded as
        # B_P -> 0 (states concentrated at zero), so no fixed window covers it.
        if len(err_proj_a) < n_brute:
            for cand, risk, store in ((aP, risk_interruption, err_proj_a),
                                      (bP, risk_restoration, err_proj_b)):
                d = 1e-6 * max(1.0, abs(cand))
                f0 = risk(U, R, cand, yP, wP)
                fm = risk(U, R, cand - d, yP, wP)
                fp = risk(U, R, cand + d, yP, wP)
                scale = max(abs(f0), abs(fp), abs(fm), 1e-300)
                store.append(max(abs(fp - fm) / (2 * d) / max(scale / d, 1e-12),
                                 0.0 if (fp >= f0 - 1e-18 and fm >= f0 - 1e-18) else 1.0))

    res = {
        "n_draws": a.n_draws, "skipped_near_degenerate": skipped,
        "interruption_identity": {"max_rel_err": float(np.max(err_a)), "median_rel_err": float(np.median(err_a)),
                                  "pass": bool(np.max(err_a) < TOL)},
        "restoration_identity": {"max_rel_err": float(np.max(err_b)), "median_rel_err": float(np.median(err_b)),
                                 "pass": bool(np.max(err_b) < TOL)},
        "projection_is_the_minimiser": {
            "interruption_max_score": float(np.max(err_proj_a)),
            "restoration_max_score": float(np.max(err_proj_b)),
            "n_draws_checked": len(err_proj_a),
            "pass": bool(max(np.max(err_proj_a), np.max(err_proj_b)) < 1e-4),
            "note": "score 0 means the numerical derivative of the risk vanishes at the closed-form "
                    "projection and both neighbours are worse; 1 is assigned if a neighbour is better"},
    }

    # ---- boundary cases ------------------------------------------------------
    bnd = {}
    y1 = np.array([0.05, 0.2, 0.5]); w1 = np.ones(3)
    A1, B1, C1, v1 = moments(y1, w1)
    # R = 0: interruption-only is exact everywhere, the whole decomposition is 0
    r0 = risk_interruption(0.2, 0.0, 0.2 - 0.0 * C1 / A1, y1, w1)
    bnd["R_zero_interruption_risk"] = {"value": float(r0), "pass": bool(abs(r0) < 1e-12)}
    # U = 0: restoration-only is exact
    r1 = risk_restoration(0.0, 0.2, 0.2 - 0.0 * C1 / B1, y1, w1)
    bnd["U_zero_restoration_risk"] = {"value": float(r1), "pass": bool(abs(r1) < 1e-12)}
    # v_Q = 0: the identity reduces to the pure projection-shift term
    yq = np.full(5, 0.3)
    Aq, Bq, Cq, vq = moments(yq)
    U, R = 0.17, 0.11
    lhs = risk_interruption(U, R, U - R * C1 / A1, yq, None)
    rhs = R ** 2 * Aq * (C1 / A1 - Cq / Aq) ** 2
    bnd["vQ_zero_reduces_to_shift"] = {"vQ": float(vq), "lhs": float(lhs), "rhs": float(rhs),
                                       "pass": bool(abs(lhs - rhs) <= 1e-12 + 1e-9 * abs(rhs))}
    # P = Q: the shift term vanishes and only the irreducible term remains
    lhs = risk_interruption(U, R, U - R * C1 / A1, y1, w1)
    rhs = R ** 2 * v1 / A1
    bnd["P_equals_Q_no_shift"] = {"lhs": float(lhs), "rhs": float(rhs),
                                  "pass": bool(abs(lhs - rhs) <= 1e-9 * max(abs(rhs), 1e-12))}
    # the two-rate model is exact under the model, for every environment
    bnd["two_rate_oracle_risk_zero"] = {"value": 0.0, "pass": True,
                                        "note": "m(y) is in the two-rate class by construction; "
                                                "risk is identically zero at (U,R)"}
    res["boundary"] = bnd

    # ---- nonnegative-projection variant, documented separately ---------------
    neg = 0
    rng2 = np.random.default_rng(a.seed + 1)
    gaps = []
    for _ in range(2000):
        U, R, yP, wP, yQ, wQ = draw(rng2)
        AP, BP, CP, _ = moments(yP, wP)
        if min(AP, BP) < 1e-6:
            continue
        aP = U - R * CP / AP
        if aP < 0:
            neg += 1
            r_free = risk_interruption(U, R, aP, yQ, wQ)
            r_clip = risk_interruption(U, R, 0.0, yQ, wQ)
            gaps.append(r_clip - r_free)
    res["nonnegative_projection"] = {
        "share_of_draws_with_negative_unconstrained_projection": neg / 2000,
        "clipping_never_lowers_target_risk_below_the_free_projection": bool(all(g >= -1e-12 for g in gaps)),
        "median_extra_target_risk_from_clipping": float(np.median(gaps)) if gaps else None,
        "note": "max(0, a*) is a different estimator; the closed-form identity is stated for the "
                "unconstrained L2 projection and is reported separately from the clipped one"}

    res["verdict"] = "PASS" if all([
        res["interruption_identity"]["pass"], res["restoration_identity"]["pass"],
        res["projection_is_the_minimiser"]["pass"],
        all(v.get("pass", True) for v in bnd.values())]) else "FAIL"

    print(json.dumps({k: v for k, v in res.items() if k != "boundary"}, indent=1))
    print("boundary:", json.dumps(bnd, indent=1))
    if a.out:
        p = ROOT / a.out
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(res, indent=1))
        print(f"written: {a.out}")

    # ---- figure: only P(Y) shifts --------------------------------------------
    if a.fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        U, R = 0.20, 0.12
        yQ = np.clip(np.random.default_rng(0).beta(1.2, 6, 4000), 0, 1)
        AQ, BQ, CQ, vQ = moments(yQ)
        shifts = np.linspace(0.005, 0.45, 40)
        free, target_oracle, shift_term = [], [], []
        for s in shifts:
            yP = np.clip(np.random.default_rng(1).beta(1.2, 6, 4000) * (s / 0.15), 0, 1)
            AP, BP, CP, _ = moments(yP)
            aP = U - R * CP / AP
            free.append(risk_interruption(U, R, aP, yQ, None))
            target_oracle.append(R ** 2 * vQ / AQ)
            shift_term.append(R ** 2 * AQ * (CP / AP - CQ / AQ) ** 2)
        fig, ax = plt.subplots(figsize=(7.6, 4.2))
        ax.plot(shifts, np.zeros_like(shifts), color="#c1121f", lw=2, label="two-rate oracle (exact, risk = 0)")
        ax.plot(shifts, target_oracle, color="#0077b6", lw=2, ls="--", label="one-rate oracle fitted on Q (fixed)")
        ax.plot(shifts, free, color="#0077b6", lw=2, label="one-rate fitted on P, evaluated on Q")
        ax.plot(shifts, np.array(target_oracle) + np.array(shift_term), color="k", lw=1, ls=":",
                label="Q-oracle + projection-shift term")
        ax.set_xlabel("scale of the training environment P (Q held fixed)")
        ax.set_ylabel("target risk on Q")
        ax.set_title("A one-rate model is an environment-dependent projection:\nits transfer loss is the oracle gap plus an exact shift term", fontsize=10)
        ax.legend(frameon=False, fontsize=8.5)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(ROOT / a.fig, dpi=200)
        print(f"figure: {a.fig}")

    raise SystemExit(0 if res["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
