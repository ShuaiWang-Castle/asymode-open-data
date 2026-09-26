#!/usr/bin/env python3
"""Generality run (experiment plan, revision 3): NET, ASYM and ASYM_STATE on law A at 8,192 parameters and on law B
at 32,768 and 8,192 parameters. Training protocol identical to B4' (see b4_experiment.py); resumable atomic outputs."""
from __future__ import annotations
from pathlib import Path
import argparse, hashlib, json, os, sys, time, subprocess
import numpy as np
import torch
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b4_data as A  # noqa: E402
import b5_data as BB  # noqa: E402

PUBLIC_REV = '97e1a5b343ecdc8d4bc333fd8fc0342c01bea6c4'
UPDATES = 3000; CHECK_EVERY = 250; PATIENCE = 1000
FITS = A.OUT.parent / 'b5_fits'
COMBOS = [('A', 8192), ('B', 32768), ('B', 8192)]
FEEDBACK_IDX = [0, 1, 2, 3]          # law A: gamma {0,.04,.08,.12}; law B: eta {0,.2,.5,1}
NS = [64, 1024]; REPS = 40; KINDS = ['NET', 'ASYM', 'ASYM_STATE']


def data_dir(law):
    return A.OUT if law == 'A' else BB.OUT_B


def task_key(law, target, fi, ri, n, rep, kind):
    return f'{law}{target}_f{fi}_r{ri}_n{n}_rep{rep:02}_{kind}'


def run_one(build_model, design, data, law, target, fi, ri, n, rep, kind):
    init = A.INIT_SEEDS[rep % len(A.INIT_SEEDS)]; torch.manual_seed(init)
    model = build_model(kind, 'd2' if kind == 'NET' else 'r1', 600, 15, 24, target,
                        float(design['scale']), float(design['source_mean']))
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-4)
    c = torch.from_numpy(design['context']); step = torch.from_numpy(design['step'])
    y0 = torch.from_numpy(design['y0_train']); y0test = torch.from_numpy(design['y0_test'])
    target_y = torch.from_numpy(data[f'train_{fi}'][ri, :n].copy())
    valid = torch.from_numpy(data[f'validation_{fi}'][ri, :n // 4].copy())
    rng = np.random.default_rng(np.random.SeedSequence([20260927, 702, rep]))
    best = float('inf'); best_update = 0; policy_stop = None; best_pred = best_test = None
    curve = []; snapshots = []; started = time.perf_counter(); first_batches = hashlib.sha256(); scale = float(design['scale'])
    for update in range(UPDATES + 1):
        if update % CHECK_EVERY == 0:
            with torch.no_grad():
                pred = model(c, y0, step); test_pred = model(c, y0test, step)
                vm = float(((pred[None] - valid) ** 2).mean())
            curve.append({'update': update, 'validation_mse': vm}); snapshots.append(np.stack([pred.numpy(), test_pred.numpy()]))
            if policy_stop is None:
                if vm < best:
                    best = vm; best_update = update; best_pred = pred.numpy().copy(); best_test = test_pred.numpy().copy()
                elif update - best_update >= PATIENCE:
                    policy_stop = update
        if update == UPDATES:
            break
        ee = torch.from_numpy(rng.integers(n, size=512)); cc = torch.from_numpy(rng.integers(8, size=512))
        if update < 20:
            first_batches.update(ee.numpy().tobytes() + cc.numpy().tobytes())
        prediction = model(c, y0, step)
        loss = (((prediction[cc] - target_y[ee, cc]) / scale) ** 2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if not torch.isfinite(loss):
            raise FloatingPointError(f'nonfinite loss at {update}')
    key = task_key(law, target, fi, ri, n, rep, kind)
    tmp = FITS / f'{key}.tmp.npz'
    np.savez_compressed(tmp, best_prediction=best_pred, best_test_prediction=best_test, snapshots=np.stack(snapshots))
    os.replace(tmp, FITS / f'{key}.npz')
    params = sum(p.numel() for p in model.parameters())
    record = {'key': key, 'law': law, 'target': target, 'feedback_index': fi, 'rho': float(A.RHOS[ri]), 'n': n, 'replication': rep,
              'model': kind, 'initialization_seed': init, 'parameters': int(params), 'widths': model.widths(),
              'policy_stop': policy_stop or UPDATES, 'best_update': best_update, 'best_validation_mse': best, 'curve': curve,
              'seconds': time.perf_counter() - started, 'first20_batch_ids_sha256': first_batches.hexdigest(), 'status': 'completed'}
    tj = FITS / f'{key}.tmp.json'; tj.write_text(json.dumps(record, indent=1) + '\n'); os.replace(tj, FITS / f'{key}.json')
    print(json.dumps({k: record[k] for k in ['key', 'parameters', 'best_update', 'seconds', 'status']}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, required=True); ap.add_argument('--worker', type=int, default=0)
    ap.add_argument('--workers', type=int, default=1); ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--profile', action='store_true'); args = ap.parse_args()
    torch.set_num_threads(args.threads)
    rev = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.repo, text=True).strip()
    if rev != PUBLIC_REV:
        raise ValueError('Unexpected public code revision: ' + rev)
    sys.path.insert(0, str(args.repo / 'analysis/net_asym_affected_cv_20260914/code'))
    from models_state import build_model_ext as build_model
    FITS.mkdir(parents=True, exist_ok=True)
    designs = {law: dict(np.load(data_dir(law) / 'design.npz')) for law in ('A', 'B')}
    jobs = [(law, target, fi, ri, n, rep, kind) for rep in range(REPS) for (law, target) in COMBOS for fi in FEEDBACK_IDX
            for ri in range(len(A.RHOS)) for n in NS for kind in KINDS]
    if args.profile:
        jobs = [('B', 8192, 2, 0, 1024, 0, k) for k in KINDS]
    else:
        jobs = [j for i, j in enumerate(jobs) if i % args.workers == args.worker]
    cache = {}
    for job in jobs:
        law, target, fi, ri, n, rep, kind = job
        outj = FITS / f'{task_key(*job)}.json'
        if outj.exists() and json.loads(outj.read_text()).get('status') == 'completed':
            continue
        if (law, rep) not in cache:
            cache.clear(); cache[(law, rep)] = dict(np.load(data_dir(law) / f'data_rep{rep:02}.npz'))
        run_one(build_model, designs[law], cache[(law, rep)], *job)


if __name__ == '__main__':
    main()
