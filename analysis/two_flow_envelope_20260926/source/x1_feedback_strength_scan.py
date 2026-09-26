#!/usr/bin/env python3
"""X1 (exploratory, not pre-registered): how S and F scale with feedback strength gamma and outage level.
Exact finite-state moments (same kernel as the neural bridge) for stronger feedback and higher initial states."""
import json, math, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; W = HERE.parent
sys.path.insert(0, str(HERE)); import dualflow as DF
H, M, U, R = 24, 16, .06, .20
FORCING = np.stack([np.r_[np.ones(8), np.zeros(16)], np.r_[np.full(12, .8), np.zeros(12)]])
def binom(n, p): return np.array([math.comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(n + 1)])
def exact_mean(x, k0, g):
    zz = np.arange(M + 1) / M
    ker = [np.stack([np.convolve(binom(k, 1 - R + g * k / M), binom(M - k, U * xx)) for k in range(M + 1)]) for xx in x]
    p = np.zeros(M + 1); p[k0] = 1; mu = []
    for h in range(H):
        p = p @ ker[h]; mu.append(p @ zz)
    return np.array(mu)
rows = []
for Ks in ([0, 2, 4, 8], [0, 4, 8, 12], [4, 8, 12, 16]):
    z = np.array(Ks) / M; Pz = DF.affine_projector(z)
    for g in (0.0, .04, .08, .12, .16, .19):
        S = F = 0.0
        for f in range(2):
            T = np.stack([exact_mean(FORCING[f], k, g) for k in Ks])
            PT = Pz @ T; S += float(((T - PT) ** 2).sum())
            F += DF.fit(PT, z, n_random=6, seed=7)['loss']
        d = 8 * H
        rows.append({'initial_K': Ks, 'gamma': g, 'S': S / d, 'F': F / d, 'F_over_S': (F / S) if S > 1e-30 else None})
        print(f"K={Ks} gamma={g:.2f}  S={S/d:.3e}  F={F/d:.3e}  F/S={(F/S) if S > 1e-30 else float('nan'):.3f}", flush=True)
(W / 'results/x1_feedback_strength_scan.json').write_text(json.dumps({'status': 'EXPLORATORY_NOT_PREREGISTERED', 'rows': rows}, indent=1) + '\n')
