#!/usr/bin/env python3
"""Local risk of the ideal constrained two-flow learner near the no-feedback boundary (THEORY_T3.md section 4 (f)).

At gamma = 0 the no-forcing hours sit on the boundary u_k = 0, so the constrained least-squares learner A projects the
affine-span noise onto a cone and has lower variance than the affine learner A_aff. In the local linear model
(Gaussian noise with the exact gamma = 0 covariance, paths linearised in the rates) this script computes, per unit
1/n and per entry d = K*H:
  x_P        = E||P eps||^2 / d                       (affine-span variance)
  x_Q        = E||Q eps||^2 / d                       (excluded-direction variance)
  x_B        = x_P - E||Pi_cone(P eps)||^2 / d        (boundary dividend at gamma = 0)
  local risk R(A) - R(A_aff) along gamma = c / sqrt(n) for crowding (c > 0) and mobilisation (c < 0), which moves
  from -x_B to F/gamma^2 * c^2 - (remaining dividend) as |c| grows.
Theory computation only (Monte Carlo over the Gaussian limit); no neural fits and no sampled panels.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import lsq_linear

HERE = Path(__file__).resolve().parent; W = HERE.parent
sys.path.insert(0, str(HERE))
from t3_check import theory, jac_path, H, X1  # noqa: E402

DRAWS = 3000; SEED = 20260926


def cov_stack(x, z, M):
    """Exact gamma = 0 covariance of the stacked paths (K*H), independent sub-populations of M groups."""
    th = theory(x, z, M); lam = th['lam'][:H]; A, B = th['A'][1:], th['B'][1:]
    Bht = np.ones((H, H))                     # B_{h,t} = prod_{l=h+1}^{t} lam_l  (paths indexed 1..H)
    for h in range(H):
        for t in range(h + 1, H):
            Bht[h, t] = Bht[h, t - 1] * lam[t]
    blocks = []
    for zj in z:
        po, ps = A + B, A
        var_h = zj * po * (1 - po) + (1 - zj) * ps * (1 - ps)
        C = np.triu(var_h[:, None] * Bht); C = C + C.T - np.diag(np.diag(C))
        blocks.append(C / M)
    K = len(z); S = np.zeros((K * H, K * H))
    for j, C in enumerate(blocks):
        S[j * H:(j + 1) * H, j * H:(j + 1) * H] = C
    return S, th


def local(x, z, M, cs=(-6, -3, -1.5, 0, 1.5, 3, 6)):
    S, th = cov_stack(x, z, M)
    J = jac_path(th['p'][:H], th['lam'][:H], z)
    Pz = J @ np.linalg.pinv(J)                       # projector onto the affine span (range of J)
    d = len(z) * H
    lo = np.full(2 * H, -np.inf); lo[:H][th['p'][:H] == 0] = 0; hi = np.full(2 * H, np.inf)
    rng = np.random.default_rng(SEED)
    w, V = np.linalg.eigh(S); root = V * np.sqrt(np.clip(w, 0, None))
    eps = rng.standard_normal((DRAWS, len(S))) @ root.T
    Peps = eps @ Pz.T
    xP = float(np.trace(Pz @ S) / d); xQ = float(np.trace(S) / d) - xP
    delta = np.r_[th['du'][:H], th['dl'][:H]]
    Jpinv = np.linalg.pinv(J)
    out = {'xP': xP, 'xQ': xQ, 'active_constraints': int((th['p'][:H] == 0).sum()), 'curve': []}
    for c in cs:
        drift = c * (J @ delta)                      # gamma * sqrt(n) * first-order shift of P m, in sqrt(n) units
        loss = []
        for e in Peps:
            y = drift + e
            t = lsq_linear(J, y, bounds=(lo, hi), method='bvls', tol=1e-12).x
            loss.append(((J @ t - drift) ** 2).sum() - (e ** 2).sum())
        out['curve'].append({'c': c, 'risk_A_minus_Aaff_times_n': float(np.mean(loss) / d),
                             'mc_se': float(np.std(loss) / np.sqrt(DRAWS) / d)})
        if c == 0:
            out['xB'] = -out['curve'][-1]['risk_A_minus_Aaff_times_n']
    return out


def main():
    designs = {'B4 layout: M 16, K {0,4,8,12}/16, forcing 1': (X1, np.array([0, 4, 8, 12]) / 16, 16),
               'narrow: M 64, K {12..20 step 2}/64, forcing 1': (X1, np.array([12, 14, 16, 18, 20]) / 64, 64)}
    res = {'status': 'THEORY_COMPUTATION_LOCAL_GAUSSIAN_LIMIT', 'draws': DRAWS, 'seed': SEED}
    for name, (x, z, M) in designs.items():
        r = local(x, z, M); res[name] = r
        print(f"== {name}: x_P {r['xP']:.4e}  x_Q {r['xQ']:.4e}  x_B {r['xB']:.4e}  (x_B/x_Q {r['xB'] / r['xQ']:.3f}; "
              f"{r['active_constraints']} active constraints)")
        for row in r['curve']:
            print(f"   c = gamma*sqrt(n) {row['c']:+5.1f}: n*(R(A) - R(A_aff)) = {row['risk_A_minus_Aaff_times_n']:+.4e} "
                  f"(MC se {row['mc_se']:.1e})")
    (W / 'results/t2_boundary.json').write_text(json.dumps(res, indent=1, default=float) + '\n')


if __name__ == '__main__':
    main()
