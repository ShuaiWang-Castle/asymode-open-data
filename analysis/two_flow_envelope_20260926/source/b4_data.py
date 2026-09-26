#!/usr/bin/env python3
"""B4' law and data: finite-population outage process with feedback gamma spanning the neural training floor.

Frozen in refine-logs/EXPERIMENT_PLAN.md (revision 1) before any B4' neural outcome.
H24, M16, L32 blocks, U .06, R .20, two forcing paths; training origins K {0,4,8,12}, interpolation K {2,6,10,14}.
gamma {0,.04,.08,.12,.16,.19}; rho {0,1} (event mixture: one block with prob rho, else the mean of L blocks);
n {64,256,1024} common prefixes; validation n/4 independent panels; 60 replications.
The event-level indicator B is shared by every context and hour of a panel; contexts evolve independently.
"""
from __future__ import annotations
import json, hashlib, math
from pathlib import Path
import numpy as np

W = Path(__file__).resolve().parent.parent
OUT = W / 'results/b4_data'
H, M, L, U, R = 24, 16, 32, .06, .20
TRAIN_K = np.array([0, 4, 8, 12]); TEST_K = np.array([2, 6, 10, 14])
FORCING = np.stack([np.r_[np.ones(8), np.zeros(16)], np.r_[np.full(12, .8), np.zeros(12)]])
GAMMAS = np.array([0., .04, .08, .12, .16, .19]); RHOS = np.array([0., 1.]); NS = np.array([64, 256, 1024])
REPS = 60; SEED = 20260926; INIT_SEEDS = [94101, 94102, 94103, 94104, 94105]
MAX_N = int(NS.max())


def binom(n, p):
    return np.array([math.comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(n + 1)])


def exact_context(x, k0, gamma):
    z = np.arange(M + 1) / M
    kernels = [np.stack([np.convolve(binom(k, 1 - R + gamma * k / M), binom(M - k, U * xx)) for k in range(M + 1)]) for xx in x]
    probs = np.zeros((H + 1, M + 1)); probs[0, k0] = 1
    for h in range(H):
        probs[h + 1] = probs[h] @ kernels[h]
    mu = probs[1:] @ z; second = np.zeros((H, H))
    for h in range(H):
        weighted = probs[h + 1] * z
        for t in range(h, H):
            second[h, t] = second[t, h] = weighted @ z
            if t + 1 < H:
                weighted = weighted @ kernels[t + 1]
    assert np.max(abs(probs.sum(1) - 1)) < 1e-12
    return mu, second - np.outer(mu, mu)


def moments(kvals, gamma):
    pairs = [exact_context(x, int(k), gamma) for x in FORCING for k in kvals]
    return np.stack([p[0] for p in pairs]), np.stack([p[1] for p in pairs])


def inputs():
    """Same input construction as the original bridge: gust slot carries forcing, identical 1/16 history."""
    raw = np.zeros((2, 48, 12)); raw[:, 24:, 2] = FORCING
    wx_mean = raw.mean((0, 1)); wx_std = raw.std((0, 1)); wx_std[wx_std == 0] = 1.
    wx = (raw - wx_mean) / wx_std
    train_y0 = np.tile(TRAIN_K / M, 2); test_y0 = np.tile(TEST_K / M, 2)
    hist = np.full((8, 24), 1 / M)
    observed = np.c_[hist, train_y0]
    scale = float(np.sqrt(np.mean(observed ** 2))); source = float(observed.mean())
    pw = np.repeat(wx[:, :24], 4, axis=0); fw = np.repeat(wx[:, 24:], 4, axis=0)
    c = np.c_[hist / scale, pw.reshape(8, -1), fw.reshape(8, -1)]
    hours = np.arange(1, 25)
    clock = np.stack([np.sin(2 * np.pi * hours / 24), np.cos(2 * np.pi * hours / 24), np.zeros(24)], axis=1)
    step = np.concatenate([fw, np.broadcast_to(clock, (8, H, 3))], axis=-1)
    return {'context': c.astype('float32'), 'step': step.astype('float32'), 'y0_train': train_y0.astype('float32'),
            'y0_test': test_y0.astype('float32'), 'scale': np.array(scale), 'source_mean': np.array(source)}


def sample_panel(rep, gi, split, panels):
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 501, rep, gi, split]))
    states = np.broadcast_to(np.tile(TRAIN_K, 2)[None, :, None], (panels, 8, L)).copy()
    uniforms = rng.random(panels)
    ys = np.empty((len(RHOS), panels, 8, H), dtype=np.float32)
    g = GAMMAS[gi]; xs = np.repeat(FORCING, 4, axis=0)
    for h in range(H):
        fail = rng.binomial(M - states, U * xs[None, :, None, h])
        restore = rng.binomial(states, R - g * states / M)
        states = states + fail - restore
        first = states[:, :, 0] / M; average = states.mean(-1) / M
        ys[:, :, :, h] = np.where(uniforms[None, :, None] < RHOS[:, None, None], first[None], average[None])
    return ys


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    arrays = inputs()
    for gi, g in enumerate(GAMMAS):
        mu, cov = moments(TRAIN_K, g); mt, ct = moments(TEST_K, g)
        arrays[f'mean_{gi}'] = mu; arrays[f'cov_{gi}'] = cov; arrays[f'mean_test_{gi}'] = mt; arrays[f'cov_test_{gi}'] = ct
    np.savez_compressed(OUT / 'design.npz', **arrays)
    for rep in range(REPS):
        path = OUT / f'data_rep{rep:02}.npz'
        if path.exists():
            continue
        out = {}
        for gi in range(len(GAMMAS)):
            out[f'train_{gi}'] = sample_panel(rep, gi, 0, MAX_N)
            out[f'validation_{gi}'] = sample_panel(rep, gi, 1, MAX_N // 4)
        np.savez_compressed(path, **out)
    protocol = {'protocol': __doc__, 'seed': SEED, 'replications': REPS, 'initialization_seeds': INIT_SEEDS,
                'gamma': GAMMAS.tolist(), 'rho': RHOS.tolist(), 'n': NS.tolist(), 'train_K': TRAIN_K.tolist(), 'test_K': TEST_K.tolist(),
                'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    print(json.dumps({'prepared': REPS, 'scale': float(arrays['scale']), 'source_mean': float(arrays['source_mean'])}))


if __name__ == '__main__':
    prepare()
