"""Core algebra and data construction for the GitHub-only conservation preflight.

Zero neural training. Nothing here imports or touches model/training code.

State equation under audit
--------------------------
    Y[t+1] = Y[t] + U_t (1 - Y[t]) - R_t Y[t]

For a *constant* two-variable weighted least-squares fit with normalised
positive weights w (sum w = 1), writing p = 1 - y,

    minimise_{U,R}  sum_i w_i (delta_i - U p_i + R y_i)^2 .

The unconstrained optimum satisfies the exact moment identity

    U (1 - mu) - R mu = mean_delta                       (Proposition 1)

with mu = sum w y and mean_delta = sum w delta, because the residual is
orthogonal to both columns and the columns sum to the constant vector.

Second-moment shorthand
    A = sum w p^2 = (1 - mu)^2 + v
    B = sum w y^2 = mu^2 + v
    C = sum w p y = mu - mu^2 - v
    v = sum w y^2 - mu^2   (weighted variance of y)
and A*B - C^2 = v exactly.

Nothing in this module presumes empirical closure; the closure ratio is
measured and reported, never assumed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, asdict, field

import numpy as np

# Fixed rate box from the paper model. Never widened, never searched.
CAP_U = 0.265
CAP_R = 0.25

# Fixed protocol constants.
CLOSURE_TOL = 0.05
K_BALANCE = 2.0
LOCAL_K = 200
LOCAL_QUERIES = 800
EVENT_ROW_CAP = 12_000
PCA_DIMS = 5


# --------------------------------------------------------------------------
# deterministic hashing
# --------------------------------------------------------------------------

def stable_hash(*parts: object, salt: str = "") -> np.uint64:
    """Deterministic 64-bit hash, stable across processes and platforms.

    Python's built-in hash() is salted per process, so it must not be used
    for anything that has to reproduce.
    """
    s = salt + "|" + "|".join(str(p) for p in parts)
    return np.uint64(int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], "big"))


def stable_hash_array(parts: list[tuple], salt: str = "") -> np.ndarray:
    return np.array([stable_hash(*p, salt=salt) for p in parts], dtype=np.uint64)


# --------------------------------------------------------------------------
# hourly aggregation (independent reimplementation of evalproto.to_hourly)
# --------------------------------------------------------------------------

def to_hourly(y15: np.ndarray, obs15: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Collapse 15-minute states to hourly using observed sub-steps only.

    The hourly value is the mean over *observed* sub-steps; an hour counts as
    observed if at least one sub-step was observed. Unobserved hours are NaN
    and are never zero-filled.

    Written independently from the semantics documented in
    ``src/asymode/evalproto.py`` and checked against it numerically in the
    test-suite. The pilot ``pack()`` path is not called.
    """
    C, T = y15.shape
    n = T // 4
    y = np.asarray(y15[:, : n * 4], dtype=np.float64).reshape(C, n, 4)
    o = np.asarray(obs15[:, : n * 4], dtype=bool).reshape(C, n, 4)
    cnt = o.sum(axis=2)
    # NaN/inf inside *unobserved* cells must not leak into the sum.
    contrib = np.where(o, np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    tot = contrib.sum(axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        yh = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    return yh, cnt > 0


# --------------------------------------------------------------------------
# constant-fit algebra
# --------------------------------------------------------------------------

@dataclass
class ConstantFit:
    """All constant-fit diagnostics for one cell (event, fold or neighbourhood)."""

    n: int
    mu: float
    v: float
    A: float
    B: float
    C: float
    abc_residual: float          # A*B - C^2 - v, must be ~0
    mean_delta: float
    rank: int

    U_unc: float
    R_unc: float
    identity_residual_unc: float  # U(1-mu) - R mu - mean_delta, exact ~0

    U_box: float
    R_box: float
    identity_residual_box: float  # generally non-zero on a boundary; not a failure
    boundary_status: str
    n_active_bounds: int

    sigma2: float
    sigma: float
    sse: float

    closure_ratio: float
    closure_pass: bool
    interior_unc: bool

    mean_flow_interrupt: float
    mean_flow_restore: float
    c_common: float
    rms_delivered_treatment: float

    balanced_lo: float
    balanced_hi: float
    balanced_share: float
    rho_box: float

    median_one_customer_fraction: float
    n_unique_counties: int

    G_plugin: float
    Gamma_plugin: float
    noise_threshold: float        # sqrt(n * G)
    Gamma_cap: float
    Gamma_near_closure: float     # NaN unless closure_pass
    n_eff_kish: float

    def as_dict(self) -> dict:
        return asdict(self)


def _moments(y: np.ndarray, delta: np.ndarray, w: np.ndarray) -> dict:
    p = 1.0 - y
    mu = float(w @ y)
    Ey2 = float(w @ (y * y))
    v = Ey2 - mu * mu
    A = float(w @ (p * p))
    B = Ey2
    C = float(w @ (p * y))
    mean_delta = float(w @ delta)
    Spd = float(w @ (p * delta))
    Syd = float(w @ (y * delta))
    return dict(mu=mu, v=v, A=A, B=B, C=C, mean_delta=mean_delta, Spd=Spd, Syd=Syd)


def _objective(U: float, R: float, m: dict, d2: float) -> float:
    return (d2 - 2 * U * m["Spd"] + 2 * R * m["Syd"]
            + U * U * m["A"] + R * R * m["B"] - 2 * U * R * m["C"])


def unconstrained_fit(m: dict) -> tuple[float, float, int]:
    """Solve the 2x2 normal equations.  Returns (U, R, rank)."""
    M = np.array([[m["A"], -m["C"]], [-m["C"], m["B"]]], dtype=np.float64)
    rhs = np.array([m["Spd"], -m["Syd"]], dtype=np.float64)
    det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]   # == A*B - C^2 == v
    scale = max(abs(m["A"]), abs(m["B"]), 1e-300)
    if not np.isfinite(det) or abs(det) <= 1e-14 * scale:
        # rank-deficient design: y is (numerically) constant, U and R are not
        # separately identified.  Report rank 1 and a least-norm solution.
        sol, *_ = np.linalg.lstsq(M, rhs, rcond=None)
        return float(sol[0]), float(sol[1]), 1
    sol = np.linalg.solve(M, rhs)
    return float(sol[0]), float(sol[1]), 2


def box_fit(m: dict, d2: float, cap_u: float = CAP_U, cap_r: float = CAP_R
            ) -> tuple[float, float, str]:
    """Exact minimiser of the convex quadratic over [0,cap_u] x [0,cap_r].

    Enumerates the interior stationary point, the four edge minimisers and the
    four corners, and returns the best feasible candidate. This is exact for a
    two-variable convex quadratic on a box.
    """
    A, B, C, Spd, Syd = m["A"], m["B"], m["C"], m["Spd"], m["Syd"]
    cands: list[tuple[float, float]] = []

    U_u, R_u, rank = unconstrained_fit(m)
    if rank == 2 and 0.0 <= U_u <= cap_u and 0.0 <= R_u <= cap_r:
        cands.append((U_u, R_u))

    def clamp(x, lo, hi):
        return float(min(max(x, lo), hi))

    # edge U = 0  ->  R*B = -Syd
    if B > 0:
        cands.append((0.0, clamp(-Syd / B, 0.0, cap_r)))
        cands.append((cap_u, clamp((cap_u * C - Syd) / B, 0.0, cap_r)))
    # edge R = 0  ->  U*A = Spd
    if A > 0:
        cands.append((clamp(Spd / A, 0.0, cap_u), 0.0))
        cands.append((clamp((Spd + cap_r * C) / A, 0.0, cap_u), cap_r))
    # corners
    for u in (0.0, cap_u):
        for r in (0.0, cap_r):
            cands.append((u, r))

    best, best_j = (0.0, 0.0), np.inf
    for u, r in cands:
        j = _objective(u, r, m, d2)
        if j < best_j - 1e-18:
            best, best_j = (float(u), float(r)), j
    return best[0], best[1], _boundary_status(best[0], best[1], cap_u, cap_r)


def _boundary_status(U: float, R: float, cap_u: float, cap_r: float,
                     tol: float = 1e-12) -> str:
    tags = []
    if abs(U) <= tol:
        tags.append("U_at_0")
    elif abs(U - cap_u) <= tol:
        tags.append("U_at_cap")
    if abs(R) <= tol:
        tags.append("R_at_0")
    elif abs(R - cap_r) <= tol:
        tags.append("R_at_cap")
    return "interior" if not tags else "+".join(tags)


def balanced_flow_interval(rho: float, K: float = K_BALANCE) -> tuple[float, float]:
    """States where interruption and restoration flows are within a factor K.

    U(1-y) and R y are within factor K exactly when
        rho/(K+rho) <= y <= K rho/(1+K rho),  rho = U/R.
    Degenerate rates give a degenerate (single-point) interval.
    """
    if not np.isfinite(rho):
        return (1.0, 1.0)          # R == 0 < U : only y == 1 balances
    if rho <= 0.0:
        return (0.0, 0.0)          # U == 0 : only y == 0 balances
    return (rho / (K + rho), K * rho / (1.0 + K * rho))


def constant_fit(y: np.ndarray, delta: np.ndarray, w: np.ndarray | None,
                 one_customer_fraction: np.ndarray,
                 county_key: np.ndarray,
                 allow_gamma: bool = True) -> ConstantFit:
    """Full diagnostic bundle for one cell.

    ``w`` may be None (uniform). It is normalised internally; the same rows,
    mask and weights drive every reported quantity.
    """
    y = np.asarray(y, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    n = int(y.size)
    if w is None:
        w = np.full(n, 1.0 / n, dtype=np.float64)
    else:
        w = np.asarray(w, dtype=np.float64)
        w = w / w.sum()

    m = _moments(y, delta, w)
    d2 = float(w @ (delta * delta))
    U_u, R_u, rank = unconstrained_fit(m)
    U_b, R_b, status = box_fit(m, d2)

    resid_unc = U_u * (1.0 - m["mu"]) - R_u * m["mu"] - m["mean_delta"]
    resid_box = U_b * (1.0 - m["mu"]) - R_b * m["mu"] - m["mean_delta"]
    abc_res = m["A"] * m["B"] - m["C"] ** 2 - m["v"]

    # residual scale from the exact box fit (the fit actually used downstream)
    e = delta - U_b * (1.0 - y) + R_b * y
    uniform = bool(np.allclose(w, w[0])) if n else True
    if uniform:
        sse = float(e @ e)
        dof = max(n - rank, 1)
        sigma2 = sse / dof
        n_eff = float(n)
    else:
        n_eff = float((w.sum() ** 2) / (w @ w))          # Kish effective size
        sse = float(np.sum(e * e))
        sigma2 = float(w @ (e * e)) * n_eff / max(n_eff - rank, 1.0)

    sigma = float(np.sqrt(max(sigma2, 0.0)))

    denom = abs(U_u) * (1.0 - m["mu"]) + abs(R_u) * m["mu"]
    closure_ratio = float(abs(m["mean_delta"]) / denom) if denom > 0 else np.inf
    interior_unc = bool(rank == 2 and 0.0 < U_u < CAP_U and 0.0 < R_u < CAP_R)
    closure_pass = bool(closure_ratio <= CLOSURE_TOL and 0.0 < m["mu"] < 1.0
                        and rank == 2 and interior_unc)

    c_common = float(min(U_b, R_b))
    rms_treat = float(c_common * np.sqrt(float(w @ ((1.0 - 2.0 * y) ** 2))))

    rho = float(U_b / R_b) if R_b > 0 else (np.inf if U_b > 0 else 0.0)
    lo, hi = balanced_flow_interval(rho)
    share = float(w @ ((y >= lo) & (y <= hi)).astype(np.float64))

    uniq, first = np.unique(county_key, return_index=True)
    med_ocf = float(np.median(one_customer_fraction[first])) if uniq.size else np.nan

    A, B, v, mu = m["A"], m["B"], m["v"], m["mu"]
    if allow_gamma and sigma2 > 0 and A > 0 and B > 0 and np.isfinite(sigma2):
        G = float(v * min(R_b ** 2 / A, U_b ** 2 / B))
        Gamma = float(n * G / sigma2)
        thr = float(np.sqrt(max(n * G, 0.0)))
        Gcap = float((n * v / sigma2) * min(CAP_R ** 2 / A, CAP_U ** 2 / B))
    else:
        G = Gamma = thr = Gcap = np.nan

    # Near-closure diagnostic formula: only where closure and interiority hold.
    # Small measured drift is not exactly zero drift, so this is a diagnostic
    # formula, never an exact empirical upper bound.
    if allow_gamma and closure_pass and sigma2 > 0:
        if mu <= 0.5:
            Gnc = float(n * R_b ** 2 * min(mu ** 2, v) / ((1.0 - mu) ** 2 * sigma2))
        else:
            Gnc = float(n * U_b ** 2 * min((1.0 - mu) ** 2, v) / (mu ** 2 * sigma2))
    else:
        Gnc = np.nan

    return ConstantFit(
        n=n, mu=float(mu), v=float(v), A=float(A), B=float(B), C=float(m["C"]),
        abc_residual=float(abc_res), mean_delta=float(m["mean_delta"]), rank=int(rank),
        U_unc=float(U_u), R_unc=float(R_u), identity_residual_unc=float(resid_unc),
        U_box=float(U_b), R_box=float(R_b), identity_residual_box=float(resid_box),
        boundary_status=status,
        n_active_bounds=0 if status == "interior" else len(status.split("+")),
        sigma2=float(sigma2), sigma=sigma, sse=float(sse),
        closure_ratio=closure_ratio, closure_pass=closure_pass,
        interior_unc=interior_unc,
        mean_flow_interrupt=float(U_b * (1.0 - mu)), mean_flow_restore=float(R_b * mu),
        c_common=c_common, rms_delivered_treatment=rms_treat,
        balanced_lo=float(lo), balanced_hi=float(hi), balanced_share=share,
        rho_box=rho,
        median_one_customer_fraction=med_ocf, n_unique_counties=int(uniq.size),
        G_plugin=G, Gamma_plugin=Gamma, noise_threshold=thr, Gamma_cap=Gcap,
        Gamma_near_closure=Gnc, n_eff_kish=float(n_eff),
    )


# --------------------------------------------------------------------------
# one-flow collapse
# --------------------------------------------------------------------------

def drift_two_flow(U: np.ndarray, R: np.ndarray, y: np.ndarray) -> np.ndarray:
    return U * (1.0 - y) - R * y


def drift_one_flow(U: np.ndarray, R: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Collapsed single signed rate s = U - R, applied to the matching pool."""
    s = U - R
    return np.maximum(s, 0.0) * (1.0 - y) - np.maximum(-s, 0.0) * y
