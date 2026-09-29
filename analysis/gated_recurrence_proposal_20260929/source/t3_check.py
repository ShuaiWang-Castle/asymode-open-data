#!/usr/bin/env python3
"""Check of the ideal-theory formulas in refine-logs/THEORY_T3.md against exact finite-population moments.

The formulas were derived first; this script only checks the algebra (no neural fits, no sampled data).
Panel design: one forcing path, K initial states z_j (independent sub-populations of M groups), crowding feedback
gamma > 0 (restoration prob R - gamma*y) or mobilisation gamma < 0. Checks:
  1. S = ||Q m||^2/d  vs  gamma^2 s^4 (kappa - 1 - sk^2) C_H                      (T3, first order)
  2. x_Q = tr(Q Sigma)/d at gamma = 0  vs  the affine-variance formula            (T3, exact at gamma = 0)
  3. environmental rate shocks: Q-trace exactly 0 at gamma = 0; gamma^2 s^4 kappa2 Vtilde_H at small gamma
  4. implied one-step intercepts of the per-horizon affine fit: (u^P_k - p_k)/gamma  vs  delta u_k   (T2)
  5. feasibility gap F = dist(Pm, A)^2/d  vs  the linearised cone projection      (T2)
  6. the closed form on the B4' design (where the controlled neural panels sat)
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np
from scipy.optimize import lsq_linear

HERE = Path(__file__).resolve().parent; W = HERE.parent
sys.path.insert(0, str(HERE))
import dualflow as DF  # noqa: E402

H, U, R = 24, .06, .20
X1 = np.r_[np.ones(8), np.zeros(16)]


def binom(n, p):
    return np.array([math.comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(n + 1)])


def exact(x, M, k0, gamma, R_=R, U_=U):
    """Exact mean and variance of Y_1..Y_H from K_0 = k0 (restore prob R - gamma*K/M, fail prob U*x)."""
    z = np.arange(M + 1) / M; pr = np.zeros(M + 1); pr[k0] = 1; cache = {}
    mu = np.empty(len(x)); var = np.empty(len(x))
    for h, xx in enumerate(x):
        if xx not in cache:
            cache[xx] = np.stack([np.convolve(binom(k, 1 - (R_ - gamma * k / M)), binom(M - k, U_ * xx)) for k in range(M + 1)])
        pr = pr @ cache[xx]; mu[h] = pr @ z; var[h] = pr @ z ** 2 - mu[h] ** 2
    return mu, var


def design_moments(z):
    zb = z.mean(); s = z.std(); w = (z - zb) / s
    return zb, s, float((w ** 3).mean()), float((w ** 4).mean())


def theory(x, z, M, R_=R, U_=U):
    """gamma = 0 affine coefficients and first-order quantities of THEORY_T3.md."""
    p = U_ * x; lam = 1 - p - R_
    A = np.zeros(H + 1); B = np.ones(H + 1); c = np.zeros(H + 1)
    for k in range(H):
        A[k + 1] = p[k] + lam[k] * A[k]; B[k + 1] = lam[k] * B[k]; c[k + 1] = lam[k] * c[k] + B[k] ** 2
    zb, s, sk, ku = design_moments(z)
    mu = A + B * zb; sd = B * s
    du = sd ** 2 - sk * sd * mu - mu ** 2 + A * (A + B) / M           # delta u_k, k = 0..H
    dl = 2 * mu + sk * sd + (1 - 2 * A - B) / M                        # delta lambda_k
    v_bar = (A * (1 - A) + zb * B * (1 - 2 * A - B)) / M               # v_h(zbar)
    beta = B * (1 - 2 * A - B) / M
    K = len(z)
    xQ = float(((K - 2) * v_bar[1:] - s * sk * beta[1:]).sum() / (K * H))
    return {'A': A, 'B': B, 'c': c, 'p': p, 'lam': lam, 'du': du, 'dl': dl, 'zb': zb, 's': s, 'sk': sk, 'ku': ku,
            'kappa2': ku - 1 - sk ** 2, 'C_H': float((c[1:] ** 2).mean()), 'xQ': xQ, 'v_bar': v_bar}


def affine_fit(m, z):
    X = np.c_[np.ones_like(z), z]
    coef, *_ = np.linalg.lstsq(X, m, rcond=None)                         # (2, H): intercept a_h, slope b_h
    return coef[0], coef[1], m - X @ coef


def q_trace(cov_by_h, z):
    Pz = DF.affine_projector(z); Qz = np.eye(len(z)) - Pz
    return float(sum(np.trace(Qz @ C) for C in cov_by_h))


def implied_u(a, b):
    a_prev = np.r_[0.0, a[:-1]]; b_prev = np.r_[1.0, b[:-1]]
    lam = b / b_prev
    return a - lam * a_prev, lam


def jac_path(u, lam, z):
    """Jacobian of the stacked paths (len(z)*H) w.r.t. (u_0..u_{H-1}, lam_0..lam_{H-1})."""
    g = np.empty((len(z), H + 1)); g[:, 0] = z
    for h in range(H):
        g[:, h + 1] = u[h] + lam[h] * g[:, h]
    J = np.zeros((len(z), H, 2 * H))
    for k in range(H):
        du = np.ones(len(z)); dl = g[:, k].copy()
        for h in range(k, H):
            if h > k:
                du = lam[h] * du; dl = lam[h] * dl
            J[:, h, k] = du; J[:, h, H + k] = dl
    return J.reshape(len(z) * H, 2 * H)


def f_linear(th, z, sign):
    """gamma^-2 dist^2 of the first-order perturbation from the tangent cone (u_k >= 0 where p_k = 0), per d."""
    J = jac_path(th['p'][:H], th['lam'][:H], z)
    target = sign * np.r_[th['du'][:H], th['dl'][:H]]
    lo = np.full(2 * H, -np.inf); lo[:H][th['p'][:H] == 0] = 0
    res = lsq_linear(J, J @ target, bounds=(lo, np.full(2 * H, np.inf)), method='bvls', tol=1e-14)
    return float(((J @ (target - res.x)) ** 2).sum() / (len(z) * H))


def f_exact(m, z):
    fitres = DF.fit(m, z, n_random=4, seed=1, tol=1e-16)
    _, _, resid = affine_fit(m, z)
    return (fitres['loss'] - float((resid ** 2).sum())) / m.size


DESIGNS = {  # (M, initial counts)
    'D1 K3 narrow, zbar .25': (64, [8, 16, 24]),
    'D2 K4 wide, zbar .375 (B4 layout)': (64, [0, 16, 32, 48]),
    'D3 K5 very narrow, zbar .25': (64, [12, 14, 16, 18, 20]),
    'D4 K4 skewed': (64, [0, 4, 8, 32]),
    'D5 K3 wide, zbar .5': (64, [8, 32, 56]),
}


def main():
    out = {'status': 'EXACT_DETERMINISTIC_CHECK_OF_DERIVED_FORMULAS', 'H': H, 'U': U, 'R': R, 'forcing': X1.tolist(), 'designs': {}}
    lines = []
    for name, (M, ks) in DESIGNS.items():
        z = np.array(ks) / M; th = theory(X1, z, M); rec = {'M': M, 'k0': ks, 'zbar': th['zb'], 's': th['s'], 'skew': th['sk'],
                                                             'kurt': th['ku'], 'C_H': th['C_H']}
        # 1. curvature signal S
        rec['S_over_gamma2'] = {}
        for g in (1e-3, 1e-2, -1e-2):
            m = np.stack([exact(X1, M, k, g)[0] for k in ks])
            _, _, resid = affine_fit(m, z)
            rec['S_over_gamma2'][g] = float((resid ** 2).sum() / m.size / g ** 2)
        rec['S_over_gamma2_theory'] = th['s'] ** 4 * th['kappa2'] * th['C_H']
        # 2. excluded-direction variance at gamma = 0
        vs = np.stack([exact(X1, M, k, 0.)[1] for k in ks])
        rec['xQ_exact'] = q_trace([np.diag(vs[:, h]) for h in range(H)], z) / (len(z) * H)
        rec['xQ_theory'] = th['xQ']
        # 3. environmental rate shocks R in {.12, .28}
        env = {}
        for g in (0., 1e-2):
            parts = [np.stack([np.stack(exact(X1, M, k, g, R_=Rr)) for k in ks]) for Rr in (.12, .28)]   # (K, 2, H)
            means = np.stack([pp[:, 0] for pp in parts]); varis = np.stack([pp[:, 1] for pp in parts])
            env_cov = [np.cov(means[:, :, h].T, bias=True) for h in range(H)]
            env[g] = q_trace(env_cov, z) / (len(z) * H)
        th12, th28 = theory(X1, z, M, R_=.12), theory(X1, z, M, R_=.28)
        Vtil = float((((th12['c'][1:] - th28['c'][1:]) / 2) ** 2).mean())
        rec['env_Qtrace_gamma0'] = env[0.]
        rec['env_Qtrace_over_gamma2_at_1e-2'] = env[1e-2] / 1e-4
        rec['env_Qtrace_over_gamma2_theory'] = th['s'] ** 4 * th['kappa2'] * Vtil
        # 4. implied intercepts of the per-horizon affine fit (first order in gamma)
        g = 1e-4
        m = np.stack([exact(X1, M, k, g)[0] for k in ks]); a, b, _ = affine_fit(m, z)
        uP, lP = implied_u(a, b)
        rec['max_abs_err_du'] = float(np.max(np.abs((uP - th['p'][:H]) / g - th['du'][:H])))
        rec['max_abs_err_dlam'] = float(np.max(np.abs((lP - th['lam'][:H]) / g - th['dl'][:H])))
        rec['du_no_forcing_hours'] = th['du'][8:H].round(6).tolist()
        # 5. feasibility gap vs linearised cone projection
        rec['F_over_gamma2'] = {}
        for g in (1e-2, 3e-3, -1e-2, -3e-3):
            m = np.stack([exact(X1, M, k, g)[0] for k in ks])
            rec['F_over_gamma2'][g] = {'exact': f_exact(m, z) / g ** 2, 'linear': f_linear(th, z, np.sign(g))}
        out['designs'][name] = rec
        lines.append(f"== {name}: zbar {th['zb']:.3f} s {th['s']:.4f} skew {th['sk']:+.3f} kurt {th['ku']:.3f}")
        lines.append('   S/g^2 exact ' + ', '.join(f'{k:+.0e}: {v:.6e}' for k, v in rec['S_over_gamma2'].items())
                     + f"   theory {rec['S_over_gamma2_theory']:.6e}")
        lines.append(f"   x_Q exact {rec['xQ_exact']:.10e} theory {rec['xQ_theory']:.10e}")
        lines.append(f"   env Q-trace at gamma 0: {rec['env_Qtrace_gamma0']:.2e}; at 1e-2 /g^2 {rec['env_Qtrace_over_gamma2_at_1e-2']:.6e}"
                     f" theory {rec['env_Qtrace_over_gamma2_theory']:.6e}")
        lines.append(f"   implied rates: max |err| du {rec['max_abs_err_du']:.2e}, dlam {rec['max_abs_err_dlam']:.2e};"
                     f" delta u at no-forcing hours {th['du'][8]:+.4f} .. {th['du'][H - 1]:+.4f}")
        lines.append('   F/g^2 ' + '; '.join(f"{k:+.0e}: exact {v['exact']:.4e} linear {v['linear']:.4e}" for k, v in rec['F_over_gamma2'].items()))
    # 6. the closed form on the B4' design (two forcing paths, M = 16 groups per block, L = 32 blocks)
    X2 = np.r_[np.full(12, .8), np.zeros(12)]
    z = np.array([0, 4, 8, 12]) / 16
    ths = [theory(X, z, 16) for X in (X1, X2)]
    S1 = np.mean([t['s'] ** 4 * t['kappa2'] * t['C_H'] for t in ths]); xq16 = np.mean([t['xQ'] for t in ths])
    b4 = []
    for gam in (.04, .08, .12, .16, .19):
        for rho, mult in ((0, 1 / 32), (1, 1.)):
            for n in (64, 256, 1024):
                b4.append({'gamma': gam, 'rho': rho, 'n': n, 'Lambda_first_order': gam ** 2 * S1 / (mult * xq16 / n)})
    # exact S and tr(Q Sigma) of the same cells, from the features recorded by b4_analyze.py
    feats = {int(k): v for k, v in json.loads((W / 'results/b4_analysis.json').read_text())['features'].items()}
    gam_all = [0., .04, .08, .12, .16, .19]
    for r in b4:
        f = feats[gam_all.index(r['gamma'])]
        r['Lambda_exact'] = f['S'] / ((r['rho'] + (1 - r['rho']) / 32) * f['trQS0'] / r['n'] / 192)
    out['b4_design_Lambda'] = b4
    lam = np.array([r['Lambda_first_order'] for r in b4]); lex = np.array([r['Lambda_exact'] for r in b4])
    lines.append(f"== B4' design over 30 feedback cells: first-order Lambda min {lam.min():.2f}, median {np.median(lam):.1f}, max {lam.max():.0f};"
                 f" exact Lambda min {lex.min():.2f}, median {np.median(lex):.1f}, max {lex.max():.0f}; exact Lambda < 1 in {int((lex < 1).sum())} cells")
    (W / 'results/t3_check.json').write_text(json.dumps(out, indent=1, default=float) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
