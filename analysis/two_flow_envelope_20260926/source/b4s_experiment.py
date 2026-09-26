#!/usr/bin/env python3
"""B4': train the complete public NET / ASYM architectures on the B4' law (revision 1 of the experiment plan).

The training protocol is copied from the original neural bridge (AdamW lr .001, wd 1e-4, clip 1, batch 512 drawn as
uniform event then uniform context, 3000 updates, validation every 250 updates, patience 1000, update 0 eligible).
Stored exact means are never used for fitting. Each fit writes an atomic npz + json, so the run can resume.
Jobs are ordered with the replication as the outer loop, so interim analyses see balanced cells.
"""
from __future__ import annotations
from pathlib import Path
import argparse, hashlib, json, os, sys, time, subprocess
import numpy as np
import torch
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from b4_data import OUT, NS, GAMMAS, RHOS, INIT_SEEDS, REPS  # noqa: E402
KINDS = ["ASYM_STATE"]

PUBLIC_REV = '97e1a5b343ecdc8d4bc333fd8fc0342c01bea6c4'
UPDATES = 3000; CHECK_EVERY = 250; PATIENCE = 1000
FITS = OUT.parent / 'b4_fits'


def task_key(gi, ri, n, rep, kind):
    return f'g{gi}_r{ri}_n{n}_rep{rep:02}_{kind}'


def run_one(build_model, design, data, gi, ri, n, rep, kind):
    init = INIT_SEEDS[rep % len(INIT_SEEDS)]; torch.manual_seed(init)
    model = build_model(kind, 'd2' if kind == 'NET' else 'r1', 600, 15, 24, 32768,
                        float(design['scale']), float(design['source_mean']))
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-4)
    c = torch.from_numpy(design['context']); step = torch.from_numpy(design['step'])
    y0 = torch.from_numpy(design['y0_train']); y0test = torch.from_numpy(design['y0_test'])
    target = torch.from_numpy(data[f'train_{gi}'][ri, :n].copy())
    valid = torch.from_numpy(data[f'validation_{gi}'][ri, :n // 4].copy())
    rng = np.random.default_rng(np.random.SeedSequence([20260926, 502, rep]))
    best = float('inf'); best_update = 0; policy_stop = None
    best_pred = best_test = None; curve = []; snapshots = []
    started = time.perf_counter(); first_batches = hashlib.sha256(); scale = float(design['scale'])
    for update in range(UPDATES + 1):
        if update % CHECK_EVERY == 0:
            with torch.no_grad():
                pred = model(c, y0, step); test_pred = model(c, y0test, step)
                vm = float(((pred[None] - valid) ** 2).mean()); tm = float(((pred[None] - target) ** 2).mean())
            curve.append({'update': update, 'validation_mse': vm, 'training_mse': tm})
            snapshots.append(np.stack([pred.numpy(), test_pred.numpy()]))
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
        loss = (((prediction[cc] - target[ee, cc]) / scale) ** 2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if not torch.isfinite(loss):
            raise FloatingPointError(f'nonfinite loss at {update}')
    key = task_key(gi, ri, n, rep, kind)
    outfile = FITS / f'{key}.npz'; tmp = FITS / f'{key}.tmp.npz'
    params = torch.cat([p.detach().flatten() for p in model.parameters()]).numpy()
    np.savez_compressed(tmp, best_prediction=best_pred, best_test_prediction=best_test, snapshots=np.stack(snapshots))
    os.replace(tmp, outfile)
    record = {'key': key, 'gamma': float(GAMMAS[gi]), 'rho': float(RHOS[ri]), 'n': n, 'validation_n': n // 4,
              'replication': rep, 'model': kind, 'initialization_seed': init, 'parameters': int(params.size),
              'architecture': model.structure, 'widths': model.widths(), 'updates': UPDATES, 'policy_stop': policy_stop or UPDATES,
              'best_update': best_update, 'best_validation_mse': best, 'curve': curve, 'seconds': time.perf_counter() - started,
              'first20_batch_ids_sha256': first_batches.hexdigest(), 'final_parameters_sha256': hashlib.sha256(params.tobytes()).hexdigest(),
              'status': 'completed'}
    outjson = FITS / f'{key}.json'; tmpj = FITS / f'{key}.tmp.json'
    tmpj.write_text(json.dumps(record, indent=1) + '\n'); os.replace(tmpj, outjson)
    print(json.dumps({k: record[k] for k in ['key', 'best_update', 'policy_stop', 'seconds', 'status']}), flush=True)
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, required=True); ap.add_argument('--worker', type=int, default=0)
    ap.add_argument('--workers', type=int, default=1); ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--profile', action='store_true'); args = ap.parse_args()
    torch.set_num_threads(args.threads)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.repo, text=True).strip()
    if revision != PUBLIC_REV:
        raise ValueError('Unexpected public code revision: ' + revision)
    sys.path.insert(0, str(args.repo / 'analysis/net_asym_affected_cv_20260914/code'))
    from models_state import build_model_ext as build_model
    FITS.mkdir(parents=True, exist_ok=True)
    design = dict(np.load(OUT / 'design.npz'))
    jobs = [(gi, ri, int(n), rep, kind) for rep in range(REPS) for gi in range(len(GAMMAS)) for ri in range(len(RHOS))
            for n in NS for kind in KINDS]
    if args.profile:
        jobs = [(3, 0, 256, 0, 'ASYM_STATE')]
    else:
        jobs = [job for i, job in enumerate(jobs) if i % args.workers == args.worker]
    cache = {}
    for job in jobs:
        outj = FITS / f'{task_key(*job)}.json'
        if outj.exists() and json.loads(outj.read_text()).get('status') == 'completed':
            continue
        rep = job[3]
        if rep not in cache:
            cache.clear(); cache[rep] = dict(np.load(OUT / f'data_rep{rep:02}.npz'))
        run_one(build_model, design, cache[rep], *job)


if __name__ == '__main__':
    main()
