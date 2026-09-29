#!/usr/bin/env python3
"""NET, ASYM and ASYM_STATE on the tropical-cyclone systems of the geo-weather development tranche D (PLAN_TROPICAL.md).

Reads the development panel read-only. Folds: the registered event folds of splits_v1D.json restricted to the 15
tropical systems. One fit per (fold, model, seed); outputs are written atomically and the run is resumable.
    python3.11 tropical_run.py --panel <features_v1D.npz> --splits <splits_v1D.json> --worker 0 --workers 3
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results' / 'fits'
ORIGIN, H = 72, 144
KINDS = ['NET', 'ASYM', 'ASYM_STATE']; SEEDS = [0, 1, 2]; FOLDS = [1, 2, 3, 4, 5]
UPDATES, CHECK, PATIENCE, BATCH, LR, WD = 3000, 100, 800, 64, 1e-3, 1e-4
HID = {'NET': 120, 'ASYM': 87, 'ASYM_STATE': 87}


def load(panel, splits):
    z = np.load(panel)
    t = z['regime'] == 'tropical'
    sysid = z['system'][t]
    groups = json.loads(Path(splits).read_text())['event_groups']
    d = {'xu': z['xu'][t][:, ORIGIN:], 'xr': z['xr'][t][:, ORIGIN:], 'xo': z['xo'][t][:, ORIGIN:], 'geo': z['geo'][t],
         'y0': z['y0'][t], 'y': z['y'][t], 'm': z['m'][t], 'w': z['w'][t], 'system': sysid,
         'fips': z['fips'][t], 'fold': np.array([groups[s] for s in sysid])}
    assert d['y'].shape[1] == H and d['xu'].shape[1] == H
    return d


def standardise(train_idx, d):
    out = {}
    for k in ('xu', 'xr', 'xo'):
        a = d[k][train_idx].reshape(-1, d[k].shape[-1])
        mu, sd = a.mean(0), a.std(0); sd[sd < 1e-8] = 1
        out[k] = np.clip((d[k] - mu) / sd, -10, 10).astype(np.float32)
    g = d['geo'][train_idx]; mu, sd = np.nanmean(g, 0), np.nanstd(g, 0); sd[~(sd > 1e-8)] = 1; mu = np.nan_to_num(mu)
    out['geo'] = np.nan_to_num(np.clip((d['geo'] - mu) / sd, -10, 10), nan=0.0).astype(np.float32)   # 15 units lack soil/forest shares
    return out


def mlp(i, h):
    return nn.Sequential(nn.Linear(i, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 1))


class Model(nn.Module):
    def __init__(self, kind, nu, nr, no, ng):
        super().__init__()
        self.kind = kind; h = HID[kind]
        if kind == 'NET':
            self.f = mlp(nu + nr + no + ng + 1, h)
            nn.init.zeros_(self.f[-1].weight); nn.init.zeros_(self.f[-1].bias)       # starts as persistence
        else:
            s = 1 if kind == 'ASYM_STATE' else 0
            self.fu = mlp(nu + no + ng + s, h); self.fr = mlp(nr + ng + s, h)
            nn.init.zeros_(self.fu[-1].weight); nn.init.constant_(self.fu[-1].bias, -10.5)  # u ~ 3e-5
            nn.init.zeros_(self.fr[-1].weight); nn.init.constant_(self.fr[-1].bias, -3.0)   # r ~ 0.05

    def forward(self, xu, xr, xo, geo, y0):
        g = y0; out = []
        B = y0.shape[0]
        for k in range(H):
            if self.kind == 'NET':
                g = g + self.f(torch.cat([xu[:, k], xr[:, k], xo[:, k], geo, g[:, None]], 1))[:, 0]
            else:
                su = [xu[:, k], xo[:, k], geo]; sr = [xr[:, k], geo]
                if self.kind == 'ASYM_STATE':
                    su.append(g[:, None]); sr.append(g[:, None])
                a = torch.cat([self.fu(torch.cat(su, 1)), self.fr(torch.cat(sr, 1)), torch.zeros(B, 1)], 1)
                p = torch.softmax(a, 1)
                g = g + p[:, 0] * (1 - g) - p[:, 1] * g
            out.append(g)
        return torch.stack(out, 1)


def fit(d, fold, kind, seed):
    key = f'f{fold}_{kind}_s{seed}'
    if (OUT / f'{key}.json').exists():
        return
    test = np.flatnonzero(d['fold'] == fold); train_all = np.flatnonzero(d['fold'] != fold)
    rng = np.random.default_rng(np.random.SeedSequence([20260926, 311, fold]))      # validation split: fixed per fold
    val = np.concatenate([rng.choice(ix, size=max(1, int(round(.2 * len(ix)))), replace=False)
                          for s in np.unique(d['system'][train_all]) for ix in [train_all[d['system'][train_all] == s]]])
    train = np.setdiff1d(train_all, val)
    X = standardise(train, d)
    T = {k: torch.from_numpy(v) for k, v in X.items()}
    y = torch.from_numpy(d['y']); m = torch.from_numpy(d['m']); w = torch.from_numpy(d['w']); y0 = torch.from_numpy(d['y0'])
    scale = float(np.sqrt((d['w'][train, None] * d['m'][train] * d['y'][train] ** 2).sum() / (d['w'][train, None] * d['m'][train]).sum()))
    torch.manual_seed(seed)
    model = Model(kind, X['xu'].shape[-1], X['xr'].shape[-1], X['xo'].shape[-1], X['geo'].shape[-1])
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    brng = np.random.default_rng(np.random.SeedSequence([20260926, 312, fold, seed]))

    def run(ix):
        ix = torch.from_numpy(ix)
        return model(T['xu'][ix], T['xr'][ix], T['xo'][ix], T['geo'][ix], y0[ix])

    def wmse(pred, ix):
        ix = torch.from_numpy(ix); wm = w[ix, None] * m[ix]
        return ((pred - y[ix]) ** 2 * wm).sum() / wm.sum()

    with torch.no_grad():
        pred_init = run(test).numpy().astype(np.float32)                               # untrained reference
    best, best_up, best_state, curve, started = float('inf'), 0, None, [], time.perf_counter()
    for up in range(UPDATES + 1):
        if up % CHECK == 0:
            with torch.no_grad():
                vm = float(wmse(run(val), val))
            curve.append([up, vm])
            if vm < best:
                best, best_up = vm, up; best_state = {k: v.clone() for k, v in model.state_dict().items()}
            elif up - best_up >= PATIENCE:
                break
        if up == UPDATES:
            break
        b = brng.choice(train, size=BATCH, replace=False)
        loss = wmse(run(b), b) / scale ** 2
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if not torch.isfinite(loss):
            raise FloatingPointError(f'{key}: non-finite loss at update {up}')
    model.load_state_dict(best_state)
    with torch.no_grad():
        pred = run(test).numpy().astype(np.float32)
    tmp = OUT / f'{key}.tmp.npz'; np.savez_compressed(tmp, test_index=test, pred=pred, pred_init=pred_init); os.replace(tmp, OUT / f'{key}.npz')
    rec = {'key': key, 'fold': fold, 'model': kind, 'seed': seed, 'parameters': sum(p.numel() for p in model.parameters()),
           'train_units': int(len(train)), 'validation_units': int(len(val)), 'test_units': int(len(test)),
           'best_update': best_up, 'best_validation_wmse': best, 'curve': curve, 'seconds': time.perf_counter() - started,
           'status': 'completed'}
    tj = OUT / f'{key}.tmp.json'; tj.write_text(json.dumps(rec) + '\n'); os.replace(tj, OUT / f'{key}.json')
    print(json.dumps({k: rec[k] for k in ('key', 'parameters', 'best_update', 'best_validation_wmse', 'seconds')}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--panel', required=True); ap.add_argument('--splits', required=True)
    ap.add_argument('--worker', type=int, default=0); ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    d = load(a.panel, a.splits)
    jobs = [(f, k, s) for s in SEEDS for f in FOLDS for k in KINDS]
    if a.smoke:
        global UPDATES
        UPDATES = 20
        for k in KINDS:
            m = Model(k, d['xu'].shape[-1], d['xr'].shape[-1], d['xo'].shape[-1], d['geo'].shape[-1])
            print(k, sum(p.numel() for p in m.parameters()))
        return
    for i, j in enumerate(jobs):
        if i % a.workers == a.worker:
            fit(d, *j)


if __name__ == '__main__':
    main()
