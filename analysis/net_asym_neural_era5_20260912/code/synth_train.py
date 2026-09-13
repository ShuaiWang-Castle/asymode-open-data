#!/usr/bin/env python3
"""Task C: NET vs ASYM trained on synthetic conditional-response laws.

Every event in a scene has identical inputs (context x, step features, y0 = 0), so a
batch evaluates the network once per distinct scene and gathers the three paths onto
the batch. Loss and gradients are identical to evaluating each event separately; this
is verified in tests/test_synth.py. No network reads mu0, Sigma0, rho, c, gamma, M0, L,
B or any future Y.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, math, resource, sys, time
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from synth_generator import (GAMMA, H, SCENE_HOURS, SPLIT_SIZE, STAGE_CODE, calibrate,   # noqa: E402
                             exact_law, rng_for, scene_x, simulate_events)
from models import KINDS, build_model, n_params                                          # noqa: E402

CTX_DIM, STEP_DIM = 32, 2
BATCH, WD, CLIP, VALIDATE_EVERY, MAX_DEV = 512, 1e-4, 1.0, 250, 3000
N_GRID, N_DEV, LAW_DEV = (32, 128, 512), 128, 1
DEV_SEEDS, FINAL_SEEDS = (6201, 6202, 6203), (7201, 7202, 7203, 7204, 7205)
CANDIDATES = [{'config_id': f'p{p}_lr{lr:g}', 'parameter_target': p, 'lr': lr}
              for p in (8192, 32768) for lr in (3e-4, 1e-3, 3e-3)]
ENDPOINTS = {'1h': 0, '6h': 5, '24h': 23, '32h': 31}
LAWS = calibrate()['laws']
CODE_FILES = ('synth_train.py', 'synth_generator.py', 'models.py', 'core_models_base.py')


def sha_file(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def scene_inputs(dtype=torch.float32):
    X = np.stack([scene_x(h) for h in SCENE_HOURS])
    step = np.stack([X, np.broadcast_to(np.arange(H) / H, X.shape)], axis=-1)
    return torch.tensor(X, dtype=dtype), torch.zeros(len(SCENE_HOURS), dtype=dtype), torch.tensor(step, dtype=dtype)


class Env:
    """One synthetic environment: (stage, dataset seed, law, gamma). Training uses a prefix n."""
    def __init__(self, stage, seed, law, gamma):
        self.stage, self.seed, self.law, self.gamma = stage, seed, law, gamma
        rho = LAWS[law]['rho']
        if not 0.0 <= rho <= 1.0:
            raise ValueError('invalid rho')
        splits = ('train', 'validation') if stage == 'DEV' else ('train', 'validation', 'calibration', 'test')
        self.Y = {s: np.stack([simulate_events(rng_for(seed, stage, h, law, gamma, s), scene_x(h),
                                               SPLIT_SIZE[s], rho, gamma)[0] for h in SCENE_HOURS]) for s in splits}
        self.mu0 = np.stack([exact_law(scene_x(h), gamma)['mu'] for h in SCENE_HOURS]) if stage == 'FINAL' else None

    def train_view(self, n):
        tr = self.Y['train'][:, :n]
        return (torch.tensor(tr.reshape(-1, H), dtype=torch.float32), np.repeat(np.arange(len(SCENE_HOURS)), n),
                float(np.sqrt(np.mean(tr ** 2))), float(tr.mean()))

    def batch_rng(self, n):
        key = (STAGE_CODE[self.stage], self.law, 0 if self.gamma > 0 else 1, 77, n)
        return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy=self.seed, spawn_key=key)))


def scene_mse(pred3, Y):          # pred3 [3,H], Y [3,n,H] -> per scene path MSE, per-event losses
    se = (pred3[:, None, :] - Y) ** 2
    return se.mean(axis=(1, 2)), se.mean(axis=2), se


def train_run(env, kind, cfg, n, updates, keep_best):
    targets, scene_of, scale, smean = env.train_view(n)
    torch.manual_seed(env.seed)
    model = build_model(kind, CTX_DIM, STEP_DIM, H, cfg['parameter_target'], scale, smean)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=WD)
    rng, ctx, y0, step = env.batch_rng(n), *scene_inputs()
    Yv = env.Y['validation']
    def val():
        with torch.no_grad():
            return float(scene_mse(model(ctx, y0, step).double().numpy(), Yv)[0].mean())
    curve = [{'update': 0, 'val_path_mse': val()}]
    best, ids_hash, ema, t0 = (math.inf, None, None), hashlib.sha256(), None, time.perf_counter()
    for u in range(1, updates + 1):
        ids = rng.integers(0, len(targets), BATCH)
        if u <= 20:
            ids_hash.update(ids.astype(np.int64).tobytes())
        pred = model(ctx, y0, step)[torch.from_numpy(scene_of[ids])]
        loss = (((pred - targets[torch.from_numpy(ids)]) / scale) ** 2).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)); opt.step()
        lv = loss.item()
        if not math.isfinite(lv):
            raise FloatingPointError(f'non-finite loss at update {u}')
        ema = lv if ema is None else .98 * ema + .02 * lv
        if u % VALIDATE_EVERY == 0:
            vm = val()
            curve.append({'update': u, 'val_path_mse': vm, 'train_loss_ema': ema, 'grad_norm_pre_clip': gn})
            if keep_best and vm < best[0]:
                best = (vm, u, copy.deepcopy(model.state_dict()))
    rec = {'task': 'synthetic', 'model': kind, **cfg, 'stage': env.stage, 'seed': env.seed, 'law': env.law,
           'c': LAWS[env.law]['c'], 'rho': LAWS[env.law]['rho'], 'gamma': env.gamma, 'n_per_scene': n,
           'updates': updates, 'width': int(model.width), 'n_params': n_params(model), 'scale': scale,
           'source_mean': smean, 'wall_seconds': round(time.perf_counter() - t0, 2), 'threads': torch.get_num_threads(),
           'first20_batch_ids_sha256': ids_hash.hexdigest(), 'curve': curve,
           'best_val_path_mse': best[0] if keep_best else None, 'best_update': best[1] if keep_best else None,
           'final_val_path_mse': curve[-1]['val_path_mse'],
           'peak_rss_mb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
           'code_sha256': {f: sha_file(HERE / f) for f in CODE_FILES}, 'torch': torch.__version__}
    return model, best, rec


def evaluate_final(model, env):
    ctx, y0, step = scene_inputs()
    with torch.no_grad():
        p3 = model(ctx, y0, step).double().numpy()
    out = {}
    for split in ('test', 'calibration'):
        per_scene, per_event, se = scene_mse(p3, env.Y[split])
        out[split] = {'path_mse': float(per_scene.mean()), 'path_mse_by_scene': per_scene.tolist(),
                      **{f'mse_{k}': float(se[:, :, j].mean()) for k, j in ENDPOINTS.items()}}
        out[f'{split}_event_loss'] = per_event
    m = (p3 - env.mu0) ** 2
    out['vs_exact_mean'] = {'path_mse': float(m.mean()), 'path_mse_by_scene': m.mean(axis=1).tolist(),
                            **{f'mse_{k}': float(m[:, j].mean()) for k, j in ENDPOINTS.items()}}
    out['oob_fraction'] = float(((p3 < 0) | (p3 > 1)).mean())
    return p3, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', required=True, choices=['profile', 'dev', 'select', 'final'])
    ap.add_argument('--work', required=True)
    ap.add_argument('--worker', type=int, default=0); ap.add_argument('--n-workers', type=int, default=1)
    ap.add_argument('--threads', type=int, default=1)
    a = ap.parse_args()
    W = Path(a.work); locks, logs = W / 'locks', W / 'logs'; logs.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(a.threads)

    def done(pattern):
        k = set()
        for f in logs.glob(pattern):
            for line in f.read_text().splitlines():
                r = json.loads(line)
                if r.get('status') == 'COMPLETED':
                    k.add((r['phase'], r['model'], r['config_id'], r['seed'], r['law'], r['gamma'], r['n_per_scene']))
        return k

    if a.phase == 'profile':
        env = Env('DEV', DEV_SEEDS[0], LAW_DEV, GAMMA); prof = {'threads': a.threads, 'runs': []}
        for kind in KINDS:
            for cfg in (CANDIDATES[1], CANDIDATES[4]):
                _, _, rec = train_run(env, kind, cfg, N_DEV, 100, keep_best=False)
                prof['runs'].append({'model': kind, **cfg, 'width': rec['width'], 'n_params': rec['n_params'],
                                     'seconds_per_update': rec['wall_seconds'] / 100,
                                     'projected_run_seconds_3000': rec['wall_seconds'] * 30})
                print(f"  {kind:4s} {cfg['config_id']:15s} {rec['wall_seconds']*10:.2f} ms/update", flush=True)
        (logs / f'SYNTH_PROFILE_t{a.threads}.json').write_text(json.dumps(prof, indent=1))
        return

    if a.phase == 'dev':
        jobs = [(cfg, s, k) for s in DEV_SEEDS for cfg in CANDIDATES for k in KINDS]
        jobs = [j for i, j in enumerate(jobs) if i % a.n_workers == a.worker]
        dn, log, envs = done('synth_dev_w*.jsonl'), logs / f'synth_dev_w{a.worker}.jsonl', {}
        for cfg, seed, kind in jobs:
            if ('dev', kind, cfg['config_id'], seed, LAW_DEV, GAMMA, N_DEV) in dn:
                continue
            env = envs.setdefault(seed, Env('DEV', seed, LAW_DEV, GAMMA))
            _, _, rec = train_run(env, kind, cfg, N_DEV, MAX_DEV, keep_best=True)
            rec.update({'phase': 'dev', 'status': 'COMPLETED', 'exit_code': 0})
            with open(log, 'a') as f:
                f.write(json.dumps(rec) + '\n')
            print(f"  dev {kind:4s} {cfg['config_id']:15s} seed{seed} best={rec['best_val_path_mse']:.6e}@{rec['best_update']} ({rec['wall_seconds']}s)", flush=True)
        return

    if a.phase == 'select':
        recs = [json.loads(l) for f in sorted(logs.glob('synth_dev_w*.jsonl')) for l in f.read_text().splitlines()]
        recs = [r for r in recs if r['status'] == 'COMPLETED']
        lock = {'task': 'synthetic', 'environment': {'law': LAW_DEV, 'n_per_scene': N_DEV, 'gamma': GAMMA},
                'rule': 'mean over DEV seeds of best scene-equal validation path MSE against realized events; ties -> smaller target, then smaller lr',
                'final_update_rule': 'median best_update over DEV seeds, nearest 250, clamp [250, 3000]', 'models': {}}
        for kind in KINDS:
            tab = []
            for cfg in CANDIDATES:
                rs = [r for r in recs if r['model'] == kind and r['config_id'] == cfg['config_id']]
                if sorted(r['seed'] for r in rs) != list(DEV_SEEDS):
                    raise RuntimeError(f'{kind} {cfg["config_id"]}: incomplete DEV seeds')
                tab.append({**cfg, 'mean_best_val': float(np.mean([r['best_val_path_mse'] for r in rs])),
                            'best_updates': [r['best_update'] for r in sorted(rs, key=lambda r: r['seed'])]})
            tab.sort(key=lambda x: (x['mean_best_val'], x['parameter_target'], x['lr']))
            s = tab[0]
            lock['models'][kind] = {'selected_config': s['config_id'], 'parameter_target': s['parameter_target'], 'lr': s['lr'],
                                    'final_updates': int(min(MAX_DEV, max(250, round(float(np.median(s['best_updates'])) / 250) * 250))),
                                    'ranking': tab}
        (locks / 'SYNTH_SELECTION_LOCK.json').write_text(json.dumps(lock, indent=1))
        print(json.dumps({k: {x: v[x] for x in ('selected_config', 'final_updates')} for k, v in lock['models'].items()}))
        return

    if a.phase == 'final':
        lp = locks / 'SYNTH_SELECTION_LOCK.json'
        if not lp.exists():
            raise RuntimeError('SYNTH_SELECTION_LOCK.json missing')
        lock, lsha = json.loads(lp.read_text()), sha_file(lp)
        cells = [(law, GAMMA, n) for law in range(3) for n in N_GRID] + [(law, 0.0, 128) for law in range(3)]
        jobs = [(cell, s, k) for s in FINAL_SEEDS for cell in cells for k in KINDS]
        jobs = [j for i, j in enumerate(jobs) if i % a.n_workers == a.worker]
        dn, log, envs = done('synth_final_w*.jsonl'), logs / f'synth_final_w{a.worker}.jsonl', {}
        for (law, gamma, n), seed, kind in jobs:
            sel = lock['models'][kind]
            cfg = {'config_id': sel['selected_config'], 'parameter_target': sel['parameter_target'], 'lr': sel['lr']}
            if ('final', kind, cfg['config_id'], seed, law, gamma, n) in dn:
                continue
            env = envs.setdefault((seed, law, gamma), Env('FINAL', seed, law, gamma))
            tdir = W / 'predictions/synthetic/truth'; tdir.mkdir(parents=True, exist_ok=True)
            tf = tdir / f'law{law}_g{gamma:g}_seed{seed}.npz'
            if not tf.exists():
                np.savez_compressed(tf, test=env.Y['test'].astype(np.float32), calibration=env.Y['calibration'].astype(np.float32),
                                    mu0=env.mu0, scene_hours=np.array(SCENE_HOURS))
            model, _, rec = train_run(env, kind, cfg, n, sel['final_updates'], keep_best=False)
            p3, ev = evaluate_final(model, env)
            rec.update({'phase': 'final', 'evaluation': {k: v for k, v in ev.items() if not k.endswith('_event_loss')},
                        'selection_lock_sha256': lsha, 'status': 'COMPLETED', 'exit_code': 0})
            pdir = W / f'predictions/synthetic/{kind}'; pdir.mkdir(parents=True, exist_ok=True)
            tag = f'law{law}_g{gamma:g}_n{n}_seed{seed}'
            np.savez_compressed(pdir / f'{tag}.npz', pred=p3.astype(np.float32),
                                test_event_loss=ev['test_event_loss'].astype(np.float32),
                                calibration_event_loss=ev['calibration_event_loss'].astype(np.float32))
            cdir = W / f'checkpoints/synthetic/{kind}'; cdir.mkdir(parents=True, exist_ok=True)
            torch.save({'state_dict': model.state_dict(), 'kind': kind, 'config': cfg, 'seed': seed, 'law': law, 'gamma': gamma,
                        'n_per_scene': n, 'updates': sel['final_updates'], 'width': int(model.width), 'n_params': n_params(model),
                        'scale': rec['scale'], 'source_mean': rec['source_mean'], 'ctx_dim': CTX_DIM, 'step_dim': STEP_DIM,
                        'horizon': H, 'selection_lock_sha256': lsha, 'code_sha256': rec['code_sha256'], 'torch': torch.__version__},
                       cdir / f'{tag}.pt')
            with open(log, 'a') as f:
                f.write(json.dumps(rec) + '\n')
            print(f"  final {kind:4s} {tag:26s} test={ev['test']['path_mse']:.6e} vs_mu0={ev['vs_exact_mean']['path_mse']:.3e} ({rec['wall_seconds']}s)", flush=True)


if __name__ == '__main__':
    main()
