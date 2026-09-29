#!/usr/bin/env python3
"""Exact check of the dilute-limit law (THEORY_GENERAL.md, Theorem G4) on two different density-dependent processes.

Each of M units is inactive or active. Per step, inactive units activate with probability a(y) and active units
deactivate with probability b(y), independently given the occupancy y of their own population.
  process A (externally forced, crowding):   a = p_k,            b = R - g*y
  process B (autocatalytic, contagion):      a = p_k + g*y,      b = R
Design: K = 4 initial states z_j = ell * zeta_j, forcing p_k = ell * 1{k < 8}, horizon 24.
Ideal index (n = 1, per entry): Lambda = S / x_Q, with S = ||Q m||^2 / d (non-affine mean response across initial
states) and x_Q = tr(Q Sigma) / d (unit noise in the excluded directions). Predicted log-slopes: 3 in ell, 1 in M,
2 in g, for both processes. The gate (contraction) feasibility gap F is reported for process B, where the linear
part expands when g > R.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import binom

HERE = Path(__file__).resolve().parent; W = HERE.parent
sys.path.insert(0, str(HERE))
import dualflow as DF  # noqa: E402

H, R = 24, .20
ZETA = np.array([0., .5, 1., 1.5])


def kernel(M, a_of_k, b_of_k):
    K = np.zeros((M + 1, M + 1))
    for k in range(M + 1):
        stay = binom.pmf(np.arange(k + 1), k, 1 - b_of_k(k))            # active units that remain active
        new = binom.pmf(np.arange(M - k + 1), M - k, a_of_k(k))          # inactive units that activate
        K[k] = np.convolve(stay, new)
    return K


def moments(process, M, ell, g):
    z = np.arange(M + 1) / M
    ks = np.rint(ZETA * ell * M).astype(int)
    kers = {}
    for forced in (True, False):
        p = ell if forced else 0.0
        if process == 'A':
            kers[forced] = kernel(M, lambda k: p, lambda k: min(max(R - g * k / M, 0), 1))
        else:
            kers[forced] = kernel(M, lambda k: min(p + g * k / M, 1), lambda k: R)
    mean = np.zeros((len(ks), H)); var = np.zeros((len(ks), H))
    for j, k0 in enumerate(ks):
        pr = np.zeros(M + 1); pr[k0] = 1
        for h in range(H):
            pr = pr @ kers[h < 8]
            mean[j, h] = pr @ z; var[j, h] = pr @ z ** 2 - mean[j, h] ** 2
    return ks / M, mean, var


def index(process, M, ell, g, with_F=False):
    zz, mean, var = moments(process, M, ell, g)
    X = np.c_[np.ones_like(zz), zz]; Pz = X @ np.linalg.solve(X.T @ X, X.T); Qz = np.eye(len(zz)) - Pz
    d = mean.size
    S = float(((Qz @ mean) ** 2).sum() / d)
    xQ = float(sum(np.trace(Qz @ np.diag(var[:, h])) for h in range(H)) / d)
    out = {'S': S, 'xQ': xQ, 'Lambda': S / xQ}
    if with_F:
        resid = mean - Pz @ mean
        out['F'] = (DF.fit(mean, zz, n_random=3, seed=1, tol=1e-16)['loss'] - float((resid ** 2).sum())) / d
    return out


def slope(xs, ys):
    return float(np.polyfit(np.log(xs), np.log(ys), 1)[0])


def main():
    res = {'status': 'EXACT_CHECK', 'R': R, 'H': H, 'zeta': ZETA.tolist()}
    lines = []
    for proc in ('A', 'B'):
        ells = [.01, .02, .04, .08]; Ms = [256, 512, 1024]; gs = [.025, .05, .1]
        L_ell = [index(proc, 1024, l, .05)['Lambda'] for l in ells]
        L_M = [index(proc, M, .04, .05)['Lambda'] for M in Ms]
        L_g = [index(proc, 1024, .04, g)['Lambda'] for g in gs]
        rec = {'Lambda_vs_ell': dict(zip(map(str, ells), L_ell)), 'Lambda_vs_M': dict(zip(map(str, Ms), L_M)),
               'Lambda_vs_g': dict(zip(map(str, gs), L_g)),
               'slopes': {'ell': slope(ells, L_ell), 'M': slope(Ms, L_M), 'g': slope(gs, L_g)}}
        res[proc] = rec
        lines.append(f"process {proc}: log-slopes of Lambda  ell {rec['slopes']['ell']:.3f} (pred 3)  M {rec['slopes']['M']:.3f} (pred 1)  "
                     f"g {rec['slopes']['g']:.3f} (pred 2)")
        lines.append('   Lambda vs ell: ' + ', '.join(f'{l}: {v:.3g}' for l, v in zip(ells, L_ell)))
    # gate feasibility for the autocatalytic process: contraction (g < R) versus expansion (g > R)
    fb = {}
    for g in (.1, .3):
        r = index('B', 256, .04, g, with_F=True); fb[str(g)] = r
        lines.append(f"process B, g {g} ({'contraction' if g < R else 'expansion'}): S {r['S']:.3e}  F {r['F']:.3e}  x_Q {r['xQ']:.3e}  F/x_Q {r['F'] / r['xQ']:.3g}")
    res['B_feasibility'] = fb
    (W / 'results/t_general_check.json').write_text(json.dumps(res, indent=1) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
