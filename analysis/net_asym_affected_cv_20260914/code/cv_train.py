#!/usr/bin/env python3
"""Affected-county grouped-CV round: NET vs split ASYM on US EAGLE-I + ERA5.

Per outer fold k: train on the affected county-events of the three training folds, early-stop and select on
the affected county-events of fold (k+1) mod 5, and score every legal window of fold k.
Both models: parameter target 32768 (every candidate within 1%), three structural candidates x learning rates
{3e-4, 1e-3, 3e-3}, batch 512, AdamW (weight decay 1e-4), gradient clip 1, at most 3000 updates, validation
every 250 updates, stop after 1000 updates without improvement, best checkpoint kept (update 0 included).
DEV: two seeds per configuration per fold. FINAL: five seeds with the per-fold selection locked beforehand.
Phases: profile, dev, select, final.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, math, os, resource, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cv_data as CD                                   # noqa: E402
from models_v2 import KINDS, build_model, n_params     # noqa: E402

CTX_DIM, STEP_DIM, H = 600, 15, 24
TARGET = 32768
BATCH, WD, CLIP = 512, 1e-4, 1.0
VALIDATE_EVERY, MAX_UPDATES, PATIENCE = 250, 3000, 1000
LRS = (3e-4, 1e-3, 3e-3)
STRUCTURES = {'NET': ('d2', 'd3', 'd4'), 'ASYM': ('r1', 'r2', 'r3')}
DEV_SEEDS = (9101, 9102)
FINAL_SEEDS = (9201, 9202, 9203, 9204, 9205)
ENDPOINTS = {'1h': 0, '6h': 5, '24h': 23}
CODE_FILES = ('cv_train.py', 'cv_data.py', 'models_v2.py')


def sha_file(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def code_hashes(): return {f: sha_file(HERE / f) for f in CODE_FILES}


class HierSampler:
    """Uniform event, then uniform county within it, then uniform origin within that county."""

    def __init__(self, d: pd.DataFrame, seed: int, fold: int):
        kc = (d.event + '|' + d.fips).to_numpy()
        self.win_start = np.flatnonzero(np.r_[True, kc[1:] != kc[:-1]])
        self.win_n = np.diff(np.r_[self.win_start, len(d)])
        ce = d.event.to_numpy()[self.win_start]
        self.cty_start = np.flatnonzero(np.r_[True, ce[1:] != ce[:-1]])
        self.cty_n = np.diff(np.r_[self.cty_start, len(self.win_start)])
        self.n_events = len(self.cty_start)
        self.rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy=seed, spawn_key=(0xDA7A, fold))))

    def draw(self, B):
        e = self.rng.integers(0, self.n_events, B)
        ci = self.cty_start[e] + np.floor(self.rng.random(B) * self.cty_n[e]).astype(np.int64)
        return self.win_start[ci] + np.floor(self.rng.random(B) * self.win_n[ci]).astype(np.int64)


def metrics(pred, truth, w):
    se = (pred - truth) ** 2
    out = {'path_mse': float(np.sum(w * se.mean(axis=1)))}
    out.update({f'mse_{k}': float(np.sum(w * se[:, j])) for k, j in ENDPOINTS.items()})
    out['oob_fraction_weighted'] = float(np.sum(w * ((pred < 0) | (pred > 1)).mean(axis=1)))
    return out


@torch.no_grad()
def predict(model, data, split, chunk=4096):
    model.eval(); n = len(data.df[split]); out = np.empty((n, H), np.float64)
    for i in range(0, n, chunk):
        ids = torch.arange(i, min(i + chunk, n))
        c, y0, step, _ = data.batch(split, ids)
        out[i:i + len(ids)] = model(c, y0, step).double().numpy()
    model.train(); return out


def train_run(data, kind, structure, lr, seed):
    k = data.fold
    torch_seed = seed + 1000 * k
    torch.manual_seed(torch_seed)
    model = build_model(kind, structure, CTX_DIM, STEP_DIM, H, TARGET, data.scale, data.source_mean)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WD)
    sampler = HierSampler(data.df['train'], seed, k)
    tv, wv = data.truth64('validation'), data.weights['validation']
    v0 = metrics(predict(model, data, 'validation'), tv, wv)['path_mse']
    curve = [{'update': 0, 'val_path_mse': v0}]
    best_val, best_update, best_state = v0, 0, copy.deepcopy(model.state_dict())
    ids_hash, ema, t0, u = hashlib.sha256(), None, time.perf_counter(), 0
    while u < MAX_UPDATES:
        u += 1
        wi = sampler.draw(BATCH)
        if u <= 20:
            ids_hash.update(wi.astype(np.int64).tobytes())
        c, y0, step, tgt = data.batch('train', torch.from_numpy(wi))
        loss = (((model(c, y0, step) - tgt) / data.scale) ** 2).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)); opt.step()
        lv = loss.item()
        if not math.isfinite(lv):
            raise FloatingPointError(f'non-finite loss at update {u}')
        ema = lv if ema is None else .98 * ema + .02 * lv
        if u % VALIDATE_EVERY == 0:
            vm = metrics(predict(model, data, 'validation'), tv, wv)['path_mse']
            curve.append({'update': u, 'val_path_mse': vm, 'train_loss_ema': ema, 'grad_norm_pre_clip': gn})
            if vm < best_val:
                best_val, best_update, best_state = vm, u, copy.deepcopy(model.state_dict())
            elif u - best_update >= PATIENCE:
                break
    model.load_state_dict(best_state)
    rec = {'fold': k, 'model': kind, 'structure': structure, 'lr': lr, 'config_id': f'{structure}_lr{lr:g}',
           'seed': seed, 'torch_seed': torch_seed, 'parameter_target': TARGET, 'widths': model.widths(),
           'n_params': n_params(model), 'n_params_by_part': model.params_by_part(),
           'updates_run': u, 'best_update': best_update, 'best_val_path_mse': best_val, 'stopped_early': u < MAX_UPDATES,
           'curve': curve, 'scale': data.scale, 'source_mean': data.source_mean,
           'windows': {s: int(len(d)) for s, d in data.df.items()},
           'wall_seconds': round(time.perf_counter() - t0, 2), 'threads': torch.get_num_threads(),
           'first20_batch_ids_sha256': ids_hash.hexdigest(),
           'peak_rss_mb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
           'code_sha256': code_hashes(), 'folds_lock_sha256': data.lock_sha, 'torch': torch.__version__}
    return model, rec


def completed(logs: Path, pattern: str, keyf) -> set:
    done = set()
    for f in Path(logs).glob(pattern):
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get('status') == 'COMPLETED':
                    done.add(keyf(r))
    return done


def save_truth(W: Path, data):
    tdir = W / 'predictions/truth'; tdir.mkdir(parents=True, exist_ok=True)
    dt, te, y0 = data.df['test'], data.truth64('test'), data.y0_64('test')
    for e, g in dt.groupby('event'):
        f = tdir / f'{e}.npz'
        if f.exists():
            continue
        ix = g.index.to_numpy(); tmp = tdir / f'.{e}.{os.getpid()}.npz'
        np.savez_compressed(tmp, row=g.row.to_numpy(), t=g.t.to_numpy(), fips=g.fips.to_numpy().astype('U5'),
                            origin_utc=g.origin_utc.to_numpy().astype('U19'), y0=y0[ix], truth=te[ix],
                            affected=g.affected.to_numpy(), peak=g.peak.to_numpy())
        os.replace(tmp, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', required=True, choices=['profile', 'dev', 'select', 'final'])
    ap.add_argument('--root', required=True); ap.add_argument('--work', required=True)
    ap.add_argument('--worker', type=int, default=0); ap.add_argument('--n-workers', type=int, default=1)
    ap.add_argument('--threads', type=int, default=2); ap.add_argument('--profile-updates', type=int, default=100)
    a = ap.parse_args()
    W = Path(a.work); locks, logs = W / 'locks', W / 'logs'; logs.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(a.threads)

    if a.phase == 'select':
        recs = [json.loads(l) for f in sorted(logs.glob('dev_w*.jsonl')) for l in f.read_text().splitlines() if l.strip()]
        recs = [r for r in recs if r['status'] == 'COMPLETED']
        lock = {'rule': 'per outer fold and model: lowest mean over the DEV seeds of the best inner-validation path MSE '
                        '(affected county-events, event-equal); ties -> smaller lr, then structure order',
                'dev_seeds': list(DEV_SEEDS), 'folds': {}}
        for k in range(CD.K_FOLDS):
            lock['folds'][str(k)] = {}
            for kind in KINDS:
                tab = []
                for s in STRUCTURES[kind]:
                    for lr in LRS:
                        rs = sorted([r for r in recs if r['fold'] == k and r['model'] == kind and r['structure'] == s
                                     and r['lr'] == lr], key=lambda r: r['seed'])
                        if [r['seed'] for r in rs] != list(DEV_SEEDS):
                            raise RuntimeError(f'fold {k} {kind} {s} lr {lr}: DEV seeds {[r["seed"] for r in rs]}')
                        tab.append({'structure': s, 'lr': lr,
                                    'mean_best_val_path_mse': float(np.mean([r['best_val_path_mse'] for r in rs])),
                                    'best_val_path_mse_by_seed': [r['best_val_path_mse'] for r in rs],
                                    'best_updates': [r['best_update'] for r in rs], 'updates_run': [r['updates_run'] for r in rs],
                                    'widths': rs[0]['widths'], 'n_params': rs[0]['n_params']})
                tab.sort(key=lambda x: (x['mean_best_val_path_mse'], x['lr'], STRUCTURES[kind].index(x['structure'])))
                lock['folds'][str(k)][kind] = {'structure': tab[0]['structure'], 'lr': tab[0]['lr'], 'ranking': tab}
        lock['dev_records'] = len(recs)
        (locks / 'SELECTION_LOCK.json').write_text(json.dumps(lock, indent=1))
        print(json.dumps({k: {m: [v[m]['structure'], v[m]['lr']] for m in KINDS} for k, v in lock['folds'].items()}))
        return

    data = CD.CVData(Path(a.root), locks)

    if a.phase == 'profile':
        prof = {'threads': a.threads, 'updates': a.profile_updates, 'torch': torch.__version__, 'runs': []}
        data.set_fold(0)
        for kind in KINDS:
            for s in STRUCTURES[kind]:
                torch.manual_seed(1)
                m = build_model(kind, s, CTX_DIM, STEP_DIM, H, TARGET, data.scale, data.source_mean)
                opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=WD)
                smp = HierSampler(data.df['train'], 1, 0); t0 = time.perf_counter()
                for _ in range(a.profile_updates):
                    c, y0, st, tg = data.batch('train', torch.from_numpy(smp.draw(BATCH)))
                    loss = (((m(c, y0, st) - tg) / data.scale) ** 2).mean()
                    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), CLIP); opt.step()
                per = (time.perf_counter() - t0) / a.profile_updates
                t1 = time.perf_counter(); predict(m, data, 'validation'); vt = time.perf_counter() - t1
                prof['runs'].append({'model': kind, 'structure': s, 'widths': m.widths(), 'n_params': n_params(m),
                                     'n_params_by_part': m.params_by_part(), 'seconds_per_update': per,
                                     'validation_pass_seconds': vt,
                                     'projected_max_run_seconds': MAX_UPDATES * per + (MAX_UPDATES // VALIDATE_EVERY + 1) * vt})
                print(f'  {kind:4s} {s} {per * 1e3:.1f} ms/update  val pass {vt:.2f}s  params {n_params(m)} {m.params_by_part()}', flush=True)
        prof['fold0_windows'] = {s: int(len(d)) for s, d in data.df.items()}
        prof['peak_rss_mb'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
        (logs / f'PROFILE_t{a.threads}.json').write_text(json.dumps(prof, indent=1))
        return

    if a.phase == 'dev':
        jobs = [(k, kind, s, lr, seed) for k in range(CD.K_FOLDS) for seed in DEV_SEEDS for kind in KINDS
                for s in STRUCTURES[kind] for lr in LRS]
        jobs = [j for i, j in enumerate(jobs) if i % a.n_workers == a.worker]
        done = completed(logs, 'dev_w*.jsonl', lambda r: (r['fold'], r['model'], r['structure'], r['lr'], r['seed']))
        log = logs / f'dev_w{a.worker}.jsonl'
        for k, kind, s, lr, seed in jobs:
            if (k, kind, s, lr, seed) in done:
                continue
            if data.fold != k:
                data.set_fold(k)
            _, rec = train_run(data, kind, s, lr, seed)
            rec.update({'phase': 'dev', 'status': 'COMPLETED', 'exit_code': 0})
            with open(log, 'a') as f:
                f.write(json.dumps(rec) + '\n')
            print(f"  dev fold{k} {kind:4s} {s} lr{lr:g} seed{seed} best={rec['best_val_path_mse']:.6e}@{rec['best_update']} "
                  f"run={rec['updates_run']} ({rec['wall_seconds']}s)", flush=True)
        return

    if a.phase == 'final':
        lock_path = locks / 'SELECTION_LOCK.json'
        if not lock_path.exists():
            raise RuntimeError('SELECTION_LOCK.json missing: no test scoring before selection is locked')
        sel, sel_sha = json.loads(lock_path.read_text()), sha_file(lock_path)
        jobs = [(k, kind, seed) for k in range(CD.K_FOLDS) for seed in FINAL_SEEDS for kind in KINDS]
        jobs = [j for i, j in enumerate(jobs) if i % a.n_workers == a.worker]
        done = completed(logs, 'final_w*.jsonl', lambda r: (r['fold'], r['model'], r['seed']))
        log = logs / f'final_w{a.worker}.jsonl'
        for k, kind, seed in jobs:
            if (k, kind, seed) in done:
                continue
            if data.fold != k:
                data.set_fold(k)
            save_truth(W, data)
            cfg = sel['folds'][str(k)][kind]
            model, rec = train_run(data, kind, cfg['structure'], cfg['lr'], seed)
            te, we = data.truth64('test'), data.weights['test']
            t1 = time.perf_counter(); pe = predict(model, data, 'test'); inf_s = time.perf_counter() - t1
            rec.update({'phase': 'final', 'test_metrics_all': metrics(pe, te, we), 'inference_seconds_test': round(inf_s, 2),
                        'selection_lock_sha256': sel_sha, 'status': 'COMPLETED', 'exit_code': 0})
            dt = data.df['test']; se = ((pe - te) ** 2).mean(axis=1)
            pdir = W / f'predictions/fold{k}/{kind}/seed{seed}'; pdir.mkdir(parents=True, exist_ok=True)
            for e, g in dt.groupby('event'):
                ix = g.index.to_numpy()
                np.savez_compressed(pdir / f'{e}.npz', row=g.row.to_numpy(), t=g.t.to_numpy(),
                                    pred=pe[ix].astype(np.float32), path_mse=se[ix])
            cdir = W / f'checkpoints/fold{k}'; cdir.mkdir(parents=True, exist_ok=True)
            torch.save({'state_dict': model.state_dict(), 'kind': kind, 'structure': cfg['structure'], 'lr': cfg['lr'],
                        'seed': seed, 'fold': k, 'best_update': rec['best_update'], 'widths': model.widths(),
                        'n_params': n_params(model), 'ctx_dim': CTX_DIM, 'step_dim': STEP_DIM, 'horizon': H,
                        'parameter_target': TARGET, 'scale': data.scale, 'source_mean': data.source_mean,
                        'wx_mean': data.wx_mean.tolist(), 'wx_std': data.wx_std.tolist(),
                        'selection_lock_sha256': sel_sha, 'folds_lock_sha256': data.lock_sha, 'code_sha256': code_hashes(),
                        'torch': torch.__version__, 'threads': torch.get_num_threads()}, cdir / f'{kind}_seed{seed}.pt')
            with open(log, 'a') as f:
                f.write(json.dumps(rec) + '\n')
            print(f"  final fold{k} {kind:4s} seed{seed} {cfg['structure']} lr{cfg['lr']:g} best@{rec['best_update']} "
                  f"test path(all)={rec['test_metrics_all']['path_mse']:.6e} ({rec['wall_seconds']}s)", flush=True)


if __name__ == '__main__':
    main()
