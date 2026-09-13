#!/usr/bin/env python3
"""Task A: NET vs ASYM on US EAGLE-I + ERA5.

Contract: RETROSPECTIVE_KNOWN_REANALYSIS_CONDITIONAL_RESPONSE. Both models receive the
same frozen windows, the same fit statistics and, for a paired seed, the same batch IDs.
Phases: profile, dev, select, final. 2024 predictions are produced only in `final`, and
only after locks/US_SELECTION_LOCK.json exists.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, math, resource, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from us_data import load_corpus                       # noqa: E402
from models import KINDS, build_model, n_params       # noqa: E402

CTX_DIM, STEP_DIM, H = 600, 15, 24
BATCH, WD, CLIP, VALIDATE_EVERY, MAX_DEV = 512, 1e-4, 1.0, 250, 3000
DEV_SEEDS, FINAL_SEEDS = (8101, 8102, 8103), (8201, 8202, 8203, 8204, 8205)
CANDIDATES = [{'config_id': f'p{p}_lr{lr:g}', 'parameter_target': p, 'lr': lr}
              for p in (8192, 32768) for lr in (3e-4, 1e-3, 3e-3)]
ENDPOINTS = {'1h': 0, '6h': 5, '24h': 23}
SPLITS = ('train', 'validation', 'evaluation')
CODE_FILES = ('us_train.py', 'us_data.py', 'models.py', 'core_models_base.py')


def sha_file(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def code_hashes(): return {f: sha_file(HERE / f) for f in CODE_FILES}


class USData:
    def __init__(self, root: Path, locks: Path):
        self.c = load_corpus(Path(root))
        lock = json.loads((Path(locks) / 'SPLITS.json').read_text())
        idx = pd.read_csv(Path(locks) / 'US_WINDOW_INDEX.csv.gz', dtype={'fips': str, 'event': str, 'origin_utc': str})
        self.df = {}
        for s in SPLITS:
            d = idx[idx.split == s].sort_values(['event', 'fips', 't'], kind='mergesort').reset_index(drop=True)
            keys = sorted(f'{e}|{f}|{o}' for e, f, o in zip(d.event, d.fips, d.origin_utc))
            if hashlib.sha256('\n'.join(keys).encode()).hexdigest() != lock['splits'][s]['window_id_sha256']:
                raise ValueError(f'{s}: window list differs from frozen SPLITS.json')
            if not (self.c.row_fips[d.row.to_numpy()] == d.fips.to_numpy()).all():
                raise ValueError(f'{s}: row/FIPS mismatch')
            self.df[s] = d
        self.splits_sha = sha_file(Path(locks) / 'SPLITS.json')
        self.weights = {s: self._weights(self.df[s]) for s in SPLITS}
        for s, d in self.df.items():
            if not np.isfinite(self._gather_Y(d, np.arange(-24, 25))).all():
                raise ValueError(f'{s}: illegal state inside a frozen window')
            if not self.c.weather_legal[d.row.to_numpy()].all():
                raise ValueError(f'{s}: missing-weather county inside a frozen window')
        self._fit_statistics()
        self._tensors()

    @staticmethod
    def _weights(d):
        n_origin = d.groupby(['event', 'fips']).t.transform('size').to_numpy(float)
        n_county = d.groupby('event').fips.transform('nunique').to_numpy(float)
        w = 1.0 / (float(d.event.nunique()) * n_county * n_origin)
        assert abs(w.sum() - 1.0) < 1e-9
        return w

    def _gather_Y(self, d, offsets):
        r, t = d.row.to_numpy(), d.t.to_numpy()
        return self.c.Y[r[:, None], t[:, None] + offsets[None, :]]

    def _fit_statistics(self):
        d, w = self.df['train'], self.weights['train']
        Ys = self._gather_Y(d, np.arange(-24, 1))
        self.scale = float(np.sqrt(np.sum(w * np.mean(Ys ** 2, axis=1))))
        self.source_mean = float(np.sum(w * np.mean(Ys, axis=1)))
        r, t = d.row.to_numpy(), d.t.to_numpy()
        m1, m2, off = np.zeros(12), np.zeros(12), np.arange(-23, 25)[None, :]
        for i in range(0, len(d), 8192):
            Xw = self.c.X[r[i:i + 8192, None], t[i:i + 8192, None] + off].astype(np.float64)
            m1 += np.einsum('i,ijk->k', w[i:i + 8192], Xw) / 48.0
            m2 += np.einsum('i,ijk->k', w[i:i + 8192], Xw ** 2) / 48.0
        self.wx_mean, self.wx_std = m1, np.sqrt(np.maximum(m2 - m1 ** 2, 0.0))
        if not (self.scale > 0 and (self.wx_std > 0).all()):
            raise ValueError('degenerate fit statistics')

    def _tensors(self):
        self.Y32 = torch.from_numpy(np.nan_to_num(self.c.Y, nan=0.0).astype(np.float32))
        self.Xs = torch.from_numpy(((self.c.X - self.wx_mean) / self.wx_std).astype(np.float32))
        ev = self.c.row_event.astype(np.int64); hrs = self.c.hour_utc[ev].astype(np.float64)
        self.sin = torch.from_numpy(np.sin(2 * np.pi * hrs / 24).astype(np.float32))
        self.cos = torch.from_numpy(np.cos(2 * np.pi * hrs / 24).astype(np.float32))
        self.wkd = torch.from_numpy(self.c.weekend[ev].astype(np.float32))
        self.rows = {s: torch.from_numpy(d.row.to_numpy().astype(np.int64)) for s, d in self.df.items()}
        self.ts = {s: torch.from_numpy(d.t.to_numpy().astype(np.int64)) for s, d in self.df.items()}
        self.off_hist, self.off_pw, self.off_fw = torch.arange(-24, 0), torch.arange(-23, 1), torch.arange(1, 25)
        self.c.X = None

    def batch(self, split, ids):
        r, t = self.rows[split][ids], self.ts[split][ids]
        R_, T_ = r[:, None], t[:, None]
        B, tf = len(ids), T_ + self.off_fw
        hist = self.Y32[R_, T_ + self.off_hist] / self.scale
        pw, fw = self.Xs[R_, T_ + self.off_pw], self.Xs[R_, tf]
        step = torch.cat([fw, self.sin[R_, tf][..., None], self.cos[R_, tf][..., None],
                          self.wkd[R_, tf][..., None]], dim=-1)
        c = torch.cat([hist, pw.reshape(B, -1), fw.reshape(B, -1)], dim=1)
        return c, self.Y32[r, t], step, self.Y32[R_, tf]

    def truth64(self, split): return self._gather_Y(self.df[split], np.arange(1, 25))
    def y0_64(self, split):
        d = self.df[split]; return self.c.Y[d.row.to_numpy(), d.t.to_numpy()]


class HierSampler:
    """Uniform event, then uniform county within it, then uniform origin within that county."""
    def __init__(self, d: pd.DataFrame, seed: int):
        kc = (d.event + '|' + d.fips).to_numpy()
        self.win_start = np.flatnonzero(np.r_[True, kc[1:] != kc[:-1]])
        self.win_n = np.diff(np.r_[self.win_start, len(d)])
        ce = d.event.to_numpy()[self.win_start]
        self.cty_start = np.flatnonzero(np.r_[True, ce[1:] != ce[:-1]])
        self.cty_n = np.diff(np.r_[self.cty_start, len(self.win_start)])
        self.n_events = len(self.cty_start)
        self.rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy=seed, spawn_key=(0xDA7A,))))

    def draw(self, B):
        e = self.rng.integers(0, self.n_events, B)
        ci = self.cty_start[e] + np.floor(self.rng.random(B) * self.cty_n[e]).astype(np.int64)
        return self.win_start[ci] + np.floor(self.rng.random(B) * self.win_n[ci]).astype(np.int64)


def metrics(pred, truth, w):
    se = (pred - truth) ** 2
    out = {'path_mse': float(np.sum(w * se.mean(axis=1)))}
    out.update({f'mse_{k}': float(np.sum(w * se[:, j])) for k, j in ENDPOINTS.items()})
    oob = (pred < 0) | (pred > 1)
    out['oob_fraction'] = float(oob.mean()); out['oob_weighted'] = float(np.sum(w * oob.mean(axis=1)))
    return out


@torch.no_grad()
def predict(model, data, split, chunk=4096):
    model.eval(); n = len(data.df[split]); out = np.empty((n, H), np.float64)
    for i in range(0, n, chunk):
        ids = torch.arange(i, min(i + chunk, n))
        c, y0, step, _ = data.batch(split, ids)
        out[i:i + len(ids)] = model(c, y0, step).double().numpy()
    model.train(); return out


def train_run(data, kind, cfg, seed, updates, keep_best):
    torch.manual_seed(seed)
    model = build_model(kind, CTX_DIM, STEP_DIM, H, cfg['parameter_target'], data.scale, data.source_mean)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=WD)
    sampler = HierSampler(data.df['train'], seed)
    tv, wv = data.truth64('validation'), data.weights['validation']
    curve = [{'update': 0, 'val_path_mse': metrics(predict(model, data, 'validation'), tv, wv)['path_mse']}]
    best, ids_hash, ema, t0 = (math.inf, None, None), hashlib.sha256(), None, time.perf_counter()
    for u in range(1, updates + 1):
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
            if keep_best and vm < best[0]:
                best = (vm, u, copy.deepcopy(model.state_dict()))
    rec = {'task': 'us_era5', 'model': kind, **cfg, 'seed': seed, 'updates': updates, 'width': int(model.width),
           'n_params': n_params(model), 'scale': data.scale, 'source_mean': data.source_mean,
           'wall_seconds': round(time.perf_counter() - t0, 2), 'threads': torch.get_num_threads(),
           'first20_batch_ids_sha256': ids_hash.hexdigest(), 'curve': curve,
           'best_val_path_mse': best[0] if keep_best else None, 'best_update': best[1] if keep_best else None,
           'final_val_path_mse': curve[-1]['val_path_mse'], 'peak_rss_mb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
           'code_sha256': code_hashes(), 'splits_sha256': data.splits_sha, 'torch': torch.__version__}
    return model, best, rec


def done_keys(logdir, pattern):
    keys = set()
    for f in Path(logdir).glob(pattern):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r.get('status') == 'COMPLETED':
                keys.add((r['phase'], r['model'], r['config_id'], r['seed']))
    return keys


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
        recs = [json.loads(l) for f in sorted(logs.glob('us_dev_w*.jsonl')) for l in f.read_text().splitlines()]
        recs = [r for r in recs if r['status'] == 'COMPLETED']
        lock = {'task': 'us_era5', 'rule': 'mean over DEV seeds of best event-equal 2022 path MSE; ties -> smaller target, then smaller lr',
                'final_update_rule': 'median best_update over DEV seeds, nearest 250, clamp [250, 3000]', 'models': {}}
        for kind in KINDS:
            tab = []
            for cfg in CANDIDATES:
                rs = [r for r in recs if r['model'] == kind and r['config_id'] == cfg['config_id']]
                if sorted(r['seed'] for r in rs) != list(DEV_SEEDS):
                    raise RuntimeError(f'{kind} {cfg["config_id"]}: incomplete DEV seeds {[r["seed"] for r in rs]}')
                tab.append({**cfg, 'mean_best_val': float(np.mean([r['best_val_path_mse'] for r in rs])),
                            'best_updates': [r['best_update'] for r in sorted(rs, key=lambda r: r['seed'])],
                            'per_seed': {r['seed']: r['best_val_path_mse'] for r in rs}})
            tab.sort(key=lambda x: (x['mean_best_val'], x['parameter_target'], x['lr']))
            sel = tab[0]
            upd = int(min(MAX_DEV, max(250, round(float(np.median(sel['best_updates'])) / 250) * 250)))
            lock['models'][kind] = {'selected_config': sel['config_id'], 'parameter_target': sel['parameter_target'],
                                    'lr': sel['lr'], 'final_updates': upd, 'ranking': tab}
        lock['dev_trials_used'] = len(recs)
        (locks / 'US_SELECTION_LOCK.json').write_text(json.dumps(lock, indent=1))
        print(json.dumps({k: {x: v[x] for x in ('selected_config', 'final_updates')} for k, v in lock['models'].items()}))
        return

    data = USData(Path(a.root), locks)
    print(f'[us:{a.phase}] train/val/eval windows = {[len(data.df[s]) for s in SPLITS]} scale={data.scale:.6g} '
          f'source_mean={data.source_mean:.6g} threads={a.threads}', flush=True)

    if a.phase == 'profile':
        prof = {'threads': a.threads, 'updates': a.profile_updates, 'torch': torch.__version__,
                'device': 'cpu', 'cuda_available': torch.cuda.is_available(), 'runs': []}
        for kind in KINDS:
            for cfg in (CANDIDATES[1], CANDIDATES[4]):
                torch.manual_seed(1)
                m = build_model(kind, CTX_DIM, STEP_DIM, H, cfg['parameter_target'], data.scale, data.source_mean)
                opt = torch.optim.AdamW(m.parameters(), lr=cfg['lr'], weight_decay=WD)
                smp = HierSampler(data.df['train'], 1); t0 = time.perf_counter()
                for _ in range(a.profile_updates):
                    c, y0, st, tg = data.batch('train', torch.from_numpy(smp.draw(BATCH)))
                    l = (((m(c, y0, st) - tg) / data.scale) ** 2).mean()
                    opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), CLIP); opt.step()
                per = (time.perf_counter() - t0) / a.profile_updates
                t1 = time.perf_counter(); predict(m, data, 'validation'); vt = time.perf_counter() - t1
                prof['runs'].append({'model': kind, **cfg, 'width': m.width, 'n_params': n_params(m),
                                     'seconds_per_update': per, 'updates_per_second': 1 / per,
                                     'validation_pass_seconds': vt,
                                     'projected_run_seconds_3000': 3000 * per + 12 * vt})
                print(f"  {kind:4s} {cfg['config_id']:15s} {per*1e3:.1f} ms/update  val pass {vt:.2f}s", flush=True)
        prof['peak_rss_mb'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
        (logs / f'US_PROFILE_t{a.threads}.json').write_text(json.dumps(prof, indent=1))
        return

    if a.phase == 'dev':
        jobs = [(cfg, s, k) for s in DEV_SEEDS for cfg in CANDIDATES for k in KINDS]
        jobs = [j for i, j in enumerate(jobs) if i % a.n_workers == a.worker]
        done = done_keys(logs, 'us_dev_w*.jsonl'); log = logs / f'us_dev_w{a.worker}.jsonl'
        for cfg, seed, kind in jobs:
            if ('dev', kind, cfg['config_id'], seed) in done:
                continue
            _, best, rec = train_run(data, kind, cfg, seed, MAX_DEV, keep_best=True)
            rec.update({'phase': 'dev', 'status': 'COMPLETED', 'exit_code': 0})
            with open(log, 'a') as f:
                f.write(json.dumps(rec) + '\n')
            print(f"  dev {kind:4s} {cfg['config_id']:15s} seed{seed} best={rec['best_val_path_mse']:.6e}@{rec['best_update']} "
                  f"({rec['wall_seconds']}s)", flush=True)
        return

    if a.phase == 'final':
        lock_path = locks / 'US_SELECTION_LOCK.json'
        if not lock_path.exists():
            raise RuntimeError('US_SELECTION_LOCK.json missing: no 2024 prediction before selection is locked')
        lock, lock_sha = json.loads(lock_path.read_text()), sha_file(lock_path)
        jobs = [(k, s) for s in FINAL_SEEDS for k in KINDS]
        jobs = [j for i, j in enumerate(jobs) if i % a.n_workers == a.worker]
        done = done_keys(logs, 'us_final_w*.jsonl'); log = logs / f'us_final_w{a.worker}.jsonl'
        te, we, y0e = data.truth64('evaluation'), data.weights['evaluation'], data.y0_64('evaluation')
        de = data.df['evaluation']
        tdir = W / 'predictions/us_era5/truth'; tdir.mkdir(parents=True, exist_ok=True)
        for e, g in de.groupby('event'):
            f = tdir / f'{e}.npz'
            if not f.exists():
                ix = g.index.to_numpy()
                np.savez_compressed(f, row=g.row.to_numpy(), t=g.t.to_numpy(), fips=g.fips.to_numpy().astype('U5'),
                                    origin_utc=g.origin_utc.to_numpy().astype('U19'), y0=y0e[ix], truth=te[ix], weight=we[ix])
        for kind, seed in jobs:
            sel = lock['models'][kind]
            cfg = {'config_id': sel['selected_config'], 'parameter_target': sel['parameter_target'], 'lr': sel['lr']}
            if ('final', kind, cfg['config_id'], seed) in done:
                continue
            model, _, rec = train_run(data, kind, cfg, seed, sel['final_updates'], keep_best=False)
            t1 = time.perf_counter(); pe = predict(model, data, 'evaluation'); inf_s = time.perf_counter() - t1
            rec.update({'phase': 'final', 'evaluation': metrics(pe, te, we), 'inference_seconds_evaluation': round(inf_s, 2),
                        'selection_lock_sha256': lock_sha, 'status': 'COMPLETED', 'exit_code': 0})
            pdir = W / f'predictions/us_era5/{kind}/seed{seed}'; pdir.mkdir(parents=True, exist_ok=True)
            se = ((pe - te) ** 2).mean(axis=1)
            for e, g in de.groupby('event'):
                ix = g.index.to_numpy()
                np.savez_compressed(pdir / f'{e}.npz', row=g.row.to_numpy(), t=g.t.to_numpy(),
                                    pred=pe[ix].astype(np.float32), path_mse=se[ix])
            cdir = W / 'checkpoints/us_era5'; cdir.mkdir(parents=True, exist_ok=True)
            torch.save({'state_dict': model.state_dict(), 'kind': kind, 'config': cfg, 'seed': seed,
                        'updates': sel['final_updates'], 'width': int(model.width), 'n_params': n_params(model),
                        'ctx_dim': CTX_DIM, 'step_dim': STEP_DIM, 'horizon': H, 'scale': data.scale,
                        'source_mean': data.source_mean, 'wx_mean': data.wx_mean.tolist(), 'wx_std': data.wx_std.tolist(),
                        'selection_lock_sha256': lock_sha, 'splits_sha256': data.splits_sha, 'code_sha256': code_hashes(),
                        'torch': torch.__version__, 'threads': torch.get_num_threads(),
                        'torch_rng_state_after_training': torch.get_rng_state()}, cdir / f'{kind}_seed{seed}.pt')
            with open(log, 'a') as f:
                f.write(json.dumps(rec) + '\n')
            print(f"  final {kind:4s} seed{seed} {cfg['config_id']} upd={sel['final_updates']} 2024 path={rec['evaluation']['path_mse']:.6e} "
                  f"oob={rec['evaluation']['oob_fraction']:.3e} ({rec['wall_seconds']}s)", flush=True)


if __name__ == '__main__':
    main()
