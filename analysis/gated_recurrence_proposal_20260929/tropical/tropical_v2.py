#!/usr/bin/env python3
"""Tropical v2 (PLAN_TROPICAL_V2.md): nested storm-level validation, four models, spike-cleaned targets.

  --stage inner : for every outer fold, model, learning rate and inner rotation, train on two inner storm folds and
                  record the validation curve on the third (early stop after 2,000 updates without improvement)
  --stage final : choose the learning rate and number of updates from the inner curves, refit on all outer-training
                  storms with seeds 0-4 and predict the outer test storms
Resumable; outputs are written atomically under results/v2/.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tropical_run as TR  # noqa: E402  (data loading and standardisation only)

OUT = HERE / 'results' / 'v2'
KINDS = ['NET', 'NET_B', 'ASYM_STATE', 'ASYM']
LRS = [3e-4, 1e-3, 3e-3]
WD, BATCH, MAXUP, CHECK, PATIENCE = 1e-4, 64, 6000, 200, 2000
FINAL_SEEDS = [0, 1, 2, 3, 4]
HID = {'NET': 120, 'NET_B': 120, 'ASYM': 87, 'ASYM_STATE': 87}
H = TR.H


def clean_spikes(y, m, y0):
    """Replace isolated one-hour spikes (> 1 pp above and > 5x the larger neighbour) by the neighbour mean."""
    y = y.copy()
    left = np.concatenate([y0[:, None], y[:, :-1]], 1)
    right = np.concatenate([y[:, 1:], np.zeros((len(y), 1), y.dtype)], 1)
    ml = np.concatenate([np.ones((len(y), 1), bool), m[:, :-1] > 0], 1)
    mr = np.concatenate([m[:, 1:] > 0, np.zeros((len(y), 1), bool)], 1)
    nb = np.maximum(left, right)
    spike = (m > 0) & ml & mr & (y - nb > .01) & (y > 5 * np.maximum(nb, 1e-4))
    y[spike] = 0.5 * (left[spike] + right[spike])
    return y, int(spike.sum()), int(spike.any(1).sum())


def mlp(i, h):
    return nn.Sequential(nn.Linear(i, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 1))


class Model(nn.Module):
    def __init__(self, kind, nu, nr, no, ng):
        super().__init__()
        self.kind = kind; h = HID[kind]
        if kind in ('NET', 'NET_B'):
            self.f = mlp(nu + nr + no + ng + 1, h)
            nn.init.zeros_(self.f[-1].weight); nn.init.zeros_(self.f[-1].bias)
        else:
            s = 1 if kind == 'ASYM_STATE' else 0
            self.fu = mlp(nu + no + ng + s, h); self.fr = mlp(nr + ng + s, h)
            nn.init.zeros_(self.fu[-1].weight); nn.init.constant_(self.fu[-1].bias, -10.5)
            nn.init.zeros_(self.fr[-1].weight); nn.init.constant_(self.fr[-1].bias, -3.0)

    def forward(self, xu, xr, xo, geo, y0):
        g = y0; out = []; B = y0.shape[0]
        for k in range(H):
            if self.kind in ('NET', 'NET_B'):
                s = g + self.f(torch.cat([xu[:, k], xr[:, k], xo[:, k], geo, g[:, None]], 1))[:, 0]
                if self.kind == 'NET_B':
                    c = torch.clamp(s, 0.0, 1.0); g = c + 0.01 * (s - c)            # leaky bound keeps gradients
                else:
                    g = s
            else:
                su = [xu[:, k], xo[:, k], geo]; sr = [xr[:, k], geo]
                if self.kind == 'ASYM_STATE':
                    su.append(g[:, None]); sr.append(g[:, None])
                a = torch.cat([self.fu(torch.cat(su, 1)), self.fr(torch.cat(sr, 1)), torch.zeros(B, 1)], 1)
                p = torch.softmax(a, 1)
                g = g + p[:, 0] * (1 - g) - p[:, 1] * g
            out.append(g)
        return torch.stack(out, 1)


def data(panel, splits):
    d = TR.load(panel, splits)
    d['y_raw'] = d['y'].copy()
    d['y'], n_hours, n_units = clean_spikes(d['y'], d['m'], d['y0'])
    d['cleaning'] = {'hours_replaced': n_hours, 'units_affected': n_units}
    return d


def inner_folds(d, outer):
    tr = np.flatnonzero(d['fold'] != outer)
    storms = sorted(np.unique(d['system'][tr]), key=lambda s: -float(d['w'][d['system'] == s].sum()))
    snake = [0, 1, 2, 2, 1, 0]
    return {s: snake[i % 6] for i, s in enumerate(storms)}


def train(d, kind, lr, seed, train_idx, val_idx=None, updates=MAXUP):
    X = TR.standardise(train_idx, d)
    T = {k: torch.from_numpy(v) for k, v in X.items()}
    y = torch.from_numpy(d['y']); m = torch.from_numpy(d['m']); w = torch.from_numpy(d['w']); y0 = torch.from_numpy(d['y0'])
    wm_tr = d['w'][train_idx, None] * d['m'][train_idx]
    scale = float(np.sqrt((wm_tr * d['y'][train_idx] ** 2).sum() / wm_tr.sum()))
    torch.manual_seed(seed)
    model = Model(kind, X['xu'].shape[-1], X['xr'].shape[-1], X['xo'].shape[-1], X['geo'].shape[-1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WD)
    rng = np.random.default_rng(np.random.SeedSequence([20260927, 401, seed, int(lr * 1e6), len(train_idx)]))

    def run(ix):
        ix = torch.from_numpy(ix)
        return model(T['xu'][ix], T['xr'][ix], T['xo'][ix], T['geo'][ix], y0[ix])

    def wmse(pred, ix):
        ix = torch.from_numpy(ix); wmm = w[ix, None] * m[ix]
        return ((pred - y[ix]) ** 2 * wmm).sum() / wmm.sum()

    curve, best, best_up = [], float('inf'), 0
    for up in range(updates + 1):
        if val_idx is not None and up % CHECK == 0:
            with torch.no_grad():
                vm = float(wmse(run(val_idx), val_idx))
            curve.append([up, vm])
            if vm < best:
                best, best_up = vm, up
            elif up - best_up >= PATIENCE:
                break
        if up == updates:
            break
        b = rng.choice(train_idx, size=BATCH, replace=False)
        loss = wmse(run(b), b) / scale ** 2
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if not torch.isfinite(loss):
            raise FloatingPointError(f'non-finite loss at update {up}')
    return model, run, curve, best, best_up


def write_json(path, rec):
    tmp = path.with_suffix('.tmp.json'); tmp.write_text(json.dumps(rec) + '\n'); os.replace(tmp, path)


def inner_jobs():
    return [(f, k, li, j) for f in TR.FOLDS for k in KINDS for li in range(len(LRS)) for j in range(3)]


def run_inner(d, f, kind, li, j):
    path = OUT / 'inner' / f'f{f}_{kind}_lr{li}_j{j}.json'
    if path.exists():
        return
    folds = inner_folds(d, f)
    tr_all = np.flatnonzero(d['fold'] != f)
    inner = np.array([folds[s] for s in d['system'][tr_all]])
    tr, va = tr_all[inner != j], tr_all[inner == j]
    t0 = time.perf_counter()
    _, _, curve, best, best_up = train(d, kind, LRS[li], seed=0, train_idx=tr, val_idx=va)
    write_json(path, {'fold': f, 'model': kind, 'lr': LRS[li], 'inner': j, 'val_storms': sorted(set(d['system'][va].tolist())),
                      'best_val_wmse': best, 'best_update': best_up, 'curve': curve, 'seconds': time.perf_counter() - t0})
    print(json.dumps({'inner': path.stem, 'best_update': best_up, 'best': best}), flush=True)


def choose(f, kind):
    recs = [json.loads((OUT / 'inner' / f'f{f}_{kind}_lr{li}_j{j}.json').read_text()) for li in range(len(LRS)) for j in range(3)]
    score = {li: np.mean([r['best_val_wmse'] for r in recs if r['lr'] == LRS[li]]) for li in range(len(LRS))}
    li = min(score, key=score.get)
    steps = [r['best_update'] for r in recs if r['lr'] == LRS[li]]
    return LRS[li], max(CHECK, int(np.median(steps))), {str(LRS[k]): float(v) for k, v in score.items()}


def run_final(d, f, kind, seed):
    path = OUT / 'final' / f'f{f}_{kind}_s{seed}.json'
    if path.exists():
        return
    lr, steps, scores = choose(f, kind)
    tr = np.flatnonzero(d['fold'] != f); te = np.flatnonzero(d['fold'] == f)
    t0 = time.perf_counter()
    model, run, _, _, _ = train(d, kind, lr, seed=seed, train_idx=tr, val_idx=None, updates=steps)
    with torch.no_grad():
        pred = run(te).numpy().astype(np.float32)
    npz = OUT / 'final' / f'f{f}_{kind}_s{seed}.npz'; tmp = npz.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, test_index=te, pred=pred); os.replace(tmp, npz)
    write_json(path, {'fold': f, 'model': kind, 'seed': seed, 'lr': lr, 'updates': steps, 'inner_scores': scores,
                      'parameters': sum(p.numel() for p in model.parameters()), 'seconds': time.perf_counter() - t0, 'status': 'completed'})
    print(json.dumps({'final': path.stem, 'lr': lr, 'updates': steps}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--panel', required=True); ap.add_argument('--splits', required=True)
    ap.add_argument('--stage', choices=['inner', 'final', 'smoke'], required=True)
    ap.add_argument('--worker', type=int, default=0); ap.add_argument('--workers', type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(1)
    d = data(a.panel, a.splits)
    (OUT / 'inner').mkdir(parents=True, exist_ok=True); (OUT / 'final').mkdir(parents=True, exist_ok=True)
    if a.stage == 'smoke':
        print(json.dumps(d['cleaning']))
        for f in TR.FOLDS:
            print(f, inner_folds(d, f))
        for k in KINDS:
            m = Model(k, d['xu'].shape[-1], d['xr'].shape[-1], d['xo'].shape[-1], d['geo'].shape[-1])
            print(k, sum(p.numel() for p in m.parameters()))
        return
    if a.stage == 'inner':
        jobs = inner_jobs()
        for i, j in enumerate(jobs):
            if i % a.workers == a.worker:
                run_inner(d, *j)
    else:
        jobs = [(f, k, s) for s in FINAL_SEEDS for f in TR.FOLDS for k in KINDS]
        for i, j in enumerate(jobs):
            if i % a.workers == a.worker:
                run_final(d, *j)


if __name__ == '__main__':
    main()
