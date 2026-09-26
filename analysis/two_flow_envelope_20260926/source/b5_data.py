#!/usr/bin/env python3
"""Generality run (experiment plan, revision 3): law B, recovery mobilisation, restore probability R(1 + eta K/M).

Same design as B4' otherwise: H24, M16, L32, U .06, R .20, two forcing paths, training K {0,4,8,12}, interpolation
K {2,6,10,14}, rho {0,1}, 40 replications, validation n/4. The stored panels hold MAX_N = 1024 training panels, so the
runs use the prefixes n = 64 and n = 1024. Law A reuses the B4' data (gamma indices 0..3).
"""
from __future__ import annotations
import json, hashlib, math
from pathlib import Path
import numpy as np
import b4_data as A

W = Path(__file__).resolve().parent.parent
OUT_B = W / 'results/b5_data_mobilisation'
ETAS = np.array([0., .2, .5, 1.])
REPS = 40; SEED = 20260927


def restore_prob(k, eta):
    return np.minimum(1.0, A.R * (1 + eta * k / A.M))


def exact_context(x, k0, eta):
    z = np.arange(A.M + 1) / A.M
    kernels = [np.stack([np.convolve(A.binom(k, 1 - float(restore_prob(k, eta))), A.binom(A.M - k, A.U * xx)) for k in range(A.M + 1)]) for xx in x]
    probs = np.zeros((A.H + 1, A.M + 1)); probs[0, k0] = 1
    for h in range(A.H):
        probs[h + 1] = probs[h] @ kernels[h]
    mu = probs[1:] @ z; second = np.zeros((A.H, A.H))
    for h in range(A.H):
        weighted = probs[h + 1] * z
        for t in range(h, A.H):
            second[h, t] = second[t, h] = weighted @ z
            if t + 1 < A.H:
                weighted = weighted @ kernels[t + 1]
    return mu, second - np.outer(mu, mu)


def moments(kvals, eta):
    pairs = [exact_context(x, int(k), eta) for x in A.FORCING for k in kvals]
    return np.stack([p[0] for p in pairs]), np.stack([p[1] for p in pairs])


def sample_panel(rep, ei, split, panels):
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 601, rep, ei, split]))
    states = np.broadcast_to(np.tile(A.TRAIN_K, 2)[None, :, None], (panels, 8, A.L)).copy()
    uniforms = rng.random(panels)
    ys = np.empty((len(A.RHOS), panels, 8, A.H), dtype=np.float32)
    eta = ETAS[ei]; xs = np.repeat(A.FORCING, 4, axis=0)
    for h in range(A.H):
        fail = rng.binomial(A.M - states, A.U * xs[None, :, None, h])
        restore = rng.binomial(states, restore_prob(states, eta))
        states = states + fail - restore
        first = states[:, :, 0] / A.M; average = states.mean(-1) / A.M
        ys[:, :, :, h] = np.where(uniforms[None, :, None] < A.RHOS[:, None, None], first[None], average[None])
    return ys


def prepare():
    OUT_B.mkdir(parents=True, exist_ok=True)
    arrays = A.inputs()
    for ei, eta in enumerate(ETAS):
        mu, cov = moments(A.TRAIN_K, eta); mt, ct = moments(A.TEST_K, eta)
        arrays[f'mean_{ei}'] = mu; arrays[f'cov_{ei}'] = cov; arrays[f'mean_test_{ei}'] = mt; arrays[f'cov_test_{ei}'] = ct
    np.savez_compressed(OUT_B / 'design.npz', **arrays)
    for rep in range(REPS):
        path = OUT_B / f'data_rep{rep:02}.npz'
        if path.exists():
            continue
        out = {}
        for ei in range(len(ETAS)):
            out[f'train_{ei}'] = sample_panel(rep, ei, 0, A.MAX_N)
            out[f'validation_{ei}'] = sample_panel(rep, ei, 1, A.MAX_N // 4)
        np.savez_compressed(path, **out)
    (OUT_B / 'protocol.json').write_text(json.dumps({'protocol': __doc__, 'eta': ETAS.tolist(), 'reps': REPS, 'seed': SEED,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2) + '\n')
    print(json.dumps({'prepared': REPS, 'out': str(OUT_B.name)}))


if __name__ == '__main__':
    prepare()
