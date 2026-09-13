#!/usr/bin/env python3
"""AsymODE trajectory mainline campaign driver (protocol ASYMODE_TRAJECTORY_MAINLINE_R1_20260909).

Implements the four locked phases. Model, data, sampling and scoring cores are
imported from the read-only package; this file is orchestration only.
"""
from __future__ import annotations
import argparse, json, math, platform, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch

ARMS = ('DIRECT', 'NET', 'ASYM')
PARAM_TARGETS = (8192, 32768)
LRS = (3e-4, 1e-3, 3e-3)
BATCH, WD, CLIP = 512, 1e-4, 1.0
VALIDATE_EVERY, MAX_DEV_STEPS, MAX_FINAL_STEPS = 250, 3000, 6000
DEV_SEED1, DEV_SEEDS2 = 4101, (4102, 4103)
FINAL_SEEDS = (5101, 5102, 5103, 5104, 5105)
DEVICE = torch.device('cpu')          # no CUDA on this host; FP32, no AMP, no compile


def candidates():
    return [{'config_id': f'p{t}_lr{lr:g}', 'parameter_target': t, 'lr': lr}
            for t in PARAM_TARGETS for lr in LRS]


class Ctx:
    def __init__(self, manifest: Path, pkg_code: Path, preflight: Path):
        sys.path.insert(0, str(pkg_code))
        import data_contract as DC, core_models as CM, training_primitives as TP
        self.DC, self.CM, self.TP = DC, CM, TP
        self.ledger, self.selected = DC.load_source(manifest)[:2]
        self.selected = self.selected.sort_values('group').reset_index(drop=True)
        z = np.load(preflight / 'origin_hours_L24_H24.npz')
        self.origins = {k: z[k].astype(np.int64) for k in z.files}
        for name, year, s, e, cap in DC.PARTITIONS:            # frozen origins must be reproducible
            if not np.array_equal(self.origins[name], DC.legal_origins(year, s, e, cap)):
                raise SystemExit(f'BLOCKED: origin set {name} not reproducible from PARTITIONS')
        self.stats, self._valcache = {}, {}

    def fit_stats(self, part: str):
        if part not in self.stats:
            self.stats[part] = self.DC.fit_state_stats(self.ledger, self.selected, self.origins[part])
        return self.stats[part]

    def eval_tensors(self, year: int, part: str, ss: float):
        key = (year, part, round(ss, 12))
        if key in self._valcache:
            return self._valcache[key]
        org = self.origins[part]; G = len(self.selected)
        gi = np.repeat(np.arange(G), len(org)); hh = np.tile(org, G)
        b = self.DC.assemble_batch(self.ledger, self.selected, year, gi, hh, ss)
        t = {'c': torch.tensor(b['c'], dtype=torch.float32, device=DEVICE),
             'y0': torch.tensor(b['y0'], dtype=torch.float32, device=DEVICE),
             'known_clock': torch.tensor(b['known_clock'], dtype=torch.float32, device=DEVICE),
             'target': b['target'].reshape(G, len(org), 24),
             'states': b['true_states_for_step_ablation_only'],
             'G': G, 'N': len(org)}
        if len(self._valcache) > 3:
            self._valcache.clear()
        self._valcache[key] = t
        return t

    def predict(self, model, ev, chunk=8192):
        model.eval(); out = []
        with torch.no_grad():
            for i in range(0, ev['c'].shape[0], chunk):
                out.append(model(ev['c'][i:i+chunk], ev['y0'][i:i+chunk],
                                 ev['known_clock'][i:i+chunk]).double().cpu().numpy())
        model.train()
        return np.concatenate(out, 0).reshape(ev['G'], ev['N'], 24)

    def score(self, pred, ev):
        return self.TP.score_full_paths(pred, ev['target'], self.selected)


def build(ctx, kind, cfg, ss, sm, seed, signed_init=None):
    torch.manual_seed(seed)                       # model-init RNG, separate from data RNG
    kw = dict(parameter_target=cfg['parameter_target'], state_scale=ss, source_mean=sm)
    if kind == 'SR':
        kw['signed_init'] = signed_init
    return ctx.CM.TrajectoryModel(kind, 168, 24, 5, **kw).to(DEVICE)


def train_run(ctx, kind, cfg, fit_part, val_part, seed, steps, year=2018,
              signed_init=None, objective='path', log=None):
    st = ctx.fit_stats(fit_part); ss, sm = st['state_scale'], st['source_mean']
    model = build(ctx, kind, cfg, ss, sm, seed, signed_init)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=WD)
    sampler = ctx.TP.CompanyEqualSampler(ctx.selected, ctx.origins[fit_part], seed)
    scale = torch.tensor(ss, dtype=torch.float32, device=DEVICE)
    ev = ctx.eval_tensors(year, val_part, ss) if val_part else None
    best = {'update': 0, 'metric': math.inf, 'state': None}
    curve, t0 = [], time.perf_counter()
    for step in range(1, steps + 1):
        gi, hh = sampler.sample(BATCH)
        b = ctx.DC.assemble_batch(ctx.ledger, ctx.selected, year, gi, hh, ss)
        c = torch.tensor(b['c'], dtype=torch.float32, device=DEVICE)
        y0 = torch.tensor(b['y0'], dtype=torch.float32, device=DEVICE)
        kc = torch.tensor(b['known_clock'], dtype=torch.float32, device=DEVICE)
        opt.zero_grad(set_to_none=True)
        if objective == 'one_step':
            stt = torch.tensor(b['true_states_for_step_ablation_only'], dtype=torch.float32, device=DEVICE)
            loss = ctx.CM.one_step_ablation_loss(model, c, stt, kc)
        else:
            tg = torch.tensor(b['target'], dtype=torch.float32, device=DEVICE)
            loss = ctx.CM.path_loss(model(c, y0, kc), tg, scale)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()
        if ev is not None and (step % VALIDATE_EVERY == 0):
            _, _, res = ctx.score(ctx.predict(model, ev), ev)
            curve.append({'update': step, 'train_loss': float(loss.item()), **res})
            if res['mse_path'] < best['metric']:
                best = {'update': step, 'metric': res['mse_path'],
                        'state': {k: v.detach().clone() for k, v in model.state_dict().items()}}
    if ev is None:
        best = {'update': steps, 'metric': None,
                'state': {k: v.detach().clone() for k, v in model.state_dict().items()}}
    rec = {'model': kind, 'config_id': cfg['config_id'], 'lr': cfg['lr'],
           'parameter_target': cfg['parameter_target'],
           'parameters': int(sum(p.numel() for p in model.parameters())),
           'width': int(model.width), 'fit': fit_part, 'val': val_part, 'seed': seed,
           'objective': objective, 'signed_init': signed_init, 'steps': steps,
           'best_update': best['update'], 'best_val_mse_path': best['metric'],
           'state_scale': ss, 'source_mean': sm,
           'wall_seconds': round(time.perf_counter() - t0, 2), 'curve': curve,
           'sampler_state_final': str(sampler.state_dict())[:200]}
    if log is not None:
        log.append(rec)
    model.load_state_dict(best['state'])
    return model, rec


def persistence(ev):
    return np.repeat(ev['y0'].double().cpu().numpy().reshape(ev['G'], ev['N'], 1), 24, axis=2)


FOLDS = (('fold_A_fit', 'fold_A_val'), ('fold_B_fit', 'fold_B_val'))


def phase_main(ctx, W: Path, res: Path):
    trials = []
    # ---- stage 1: all six candidates, both folds, seed 4101
    s1 = []
    for kind in ARMS:                                   # model-balanced round robin
        for cfg in candidates():
            for fit, val in FOLDS:
                _, r = train_run(ctx, kind, cfg, fit, val, DEV_SEED1, MAX_DEV_STEPS, log=trials)
                s1.append(r)
                print(f"  s1 {kind:7s} {cfg['config_id']:14s} {fit[:6]} "
                      f"best={r['best_val_mse_path']:.6e}@{r['best_update']} ({r['wall_seconds']}s)", flush=True)
    df1 = pd.DataFrame([{k: v for k, v in r.items() if k != 'curve'} for r in s1])
    df1.to_csv(res / 'STAGE1_TRIALS.csv', index=False)

    # ---- stage 2: top two configs per arm by fold-averaged validation, seeds 4102/4103
    top = {}
    for kind in ARMS:
        m = df1[df1.model == kind].groupby('config_id').best_val_mse_path.mean().sort_values()
        top[kind] = list(m.index[:2])
    s2 = []
    for kind in ARMS:
        for cid in top[kind]:
            cfg = next(c for c in candidates() if c['config_id'] == cid)
            for seed in DEV_SEEDS2:
                for fit, val in FOLDS:
                    _, r = train_run(ctx, kind, cfg, fit, val, seed, MAX_DEV_STEPS, log=trials)
                    s2.append(r)
                    print(f"  s2 {kind:7s} {cid:14s} seed{seed} {fit[:6]} "
                          f"best={r['best_val_mse_path']:.6e}@{r['best_update']}", flush=True)
    df2 = pd.DataFrame([{k: v for k, v in r.items() if k != 'curve'} for r in s2])
    df2.to_csv(res / 'STAGE2_TRIALS.csv', index=False)

    # ---- unique configuration per arm over three development seeds and two folds
    sel, final_steps = {}, {}
    for kind in ARMS:
        # Candidate identity is (model, config_id). Filtering stage 1 by the union
        # of every arm's promoted config_ids let a configuration promoted for one
        # arm re-enter another arm with only its seed-4101 rows. Both frames are
        # now filtered on model AND config_id, and every eligible candidate must
        # carry exactly two folds x three development seeds.
        d = pd.concat([
            df1[(df1.model == kind) & df1.config_id.isin(top[kind])],
            df2[(df2.model == kind) & df2.config_id.isin(top[kind])],
        ], ignore_index=True)
        for cid in top[kind]:
            sub = d[d.config_id == cid]
            assert len(sub) == 6, (kind, cid, len(sub))
            assert set(zip(sub.fit, sub.seed)) == {
                (f, s) for f in ('fold_A_fit', 'fold_B_fit')
                for s in (DEV_SEED1, *DEV_SEEDS2)}, (kind, cid)
        agg = d.groupby('config_id').best_val_mse_path.mean().sort_values()
        sel[kind] = agg.index[0]
        chosen = d[d.config_id == sel[kind]]   # already model-filtered above
        scaled = []
        for _, r in chosen.iterrows():
            n_fit = len(ctx.origins[r.fit]); n_full = len(ctx.origins['full_2018'])
            scaled.append(r.best_update * n_full / n_fit)
        final_steps[kind] = int(min(MAX_FINAL_STEPS,
                                    max(VALIDATE_EVERY, round(float(np.median(scaled)) / VALIDATE_EVERY) * VALIDATE_EVERY)))
    lock = {'protocol_id': 'ASYMODE_TRAJECTORY_MAINLINE_R1_20260909',
            'stage1_trials': len(s1), 'stage2_trials': len(s2),
            'top2_per_arm': top, 'selected_config': sel, 'final_steps': final_steps,
            'selection_metric': 'company-equal path MSE on fold validation, exact all-origin',
            'dev_seeds': [DEV_SEED1, *DEV_SEEDS2], 'final_seeds': list(FINAL_SEEDS),
            'fold_stats': {p: ctx.fit_stats(p) for p in ('fold_A_fit', 'fold_B_fit', 'full_2018')},
            'written_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    (W / 'selection_lock.json').write_text(json.dumps(lock, indent=2))
    print('\nselection_lock.json written:', json.dumps(sel), final_steps, flush=True)

    # ---- final refit on full 2018, then the reused 2019 retrospective evaluation
    st = ctx.fit_stats('full_2018')
    ev19 = ctx.eval_tensors(2019, 'reused_2019', st['state_scale'])
    rows, preds = [], {}
    pgroup, pcomp, pres = ctx.score(persistence(ev19), ev19)
    for c_ in pcomp.itertuples():
        rows.append({'model': 'PERSISTENCE', 'seed': None, 'company': int(c_.company),
                     'mse_path': c_.mse_path, 'mse_1h': c_.mse_1h, 'mse_6h': c_.mse_6h,
                     'mse_24h': c_.mse_24h, 'oob': c_.out_of_bounds_fraction})
    comp_by = {'PERSISTENCE': {None: pcomp}}
    for kind in ARMS:
        cfg = next(c for c in candidates() if c['config_id'] == sel[kind])
        comp_by[kind] = {}
        for seed in FINAL_SEEDS:
            model, r = train_run(ctx, kind, cfg, 'full_2018', None, seed, final_steps[kind], log=trials)
            p = ctx.predict(model, ev19)
            g, cmp_, agg = ctx.score(p, ev19)
            comp_by[kind][seed] = cmp_
            preds[f'{kind}_seed{seed}'] = p.astype(np.float32)
            for c_ in cmp_.itertuples():
                rows.append({'model': kind, 'seed': seed, 'company': int(c_.company),
                             'mse_path': c_.mse_path, 'mse_1h': c_.mse_1h, 'mse_6h': c_.mse_6h,
                             'mse_24h': c_.mse_24h, 'oob': c_.out_of_bounds_fraction})
            g.to_csv(res / f'GROUP_{kind}_seed{seed}.csv', index=False)
            torch.save({'state_dict': model.state_dict(), 'record': {k: v for k, v in r.items() if k != 'curve'}},
                       W / 'main' / f'final_{kind}_seed{seed}.pt')
            print(f"  final {kind:7s} seed{seed} steps={final_steps[kind]} "
                  f"2019 path={agg['mse_path']:.6e} oob={agg['out_of_bounds_fraction']:.3e}", flush=True)
    pd.DataFrame(rows).to_csv(res / 'MAIN_COMPANY_SEED.csv', index=False)
    np.savez_compressed(res / 'PREDICTIONS_2019.npz', **preds,
                        truth=ev19['target'].astype(np.float32),
                        origin_hours=ctx.origins['reused_2019'],
                        groups=ctx.selected.group.to_numpy())
    # ---- paired summary and relative gain against DIRECT
    out = []
    for kind in list(ARMS) + ['PERSISTENCE']:
        for seed, cmp_ in comp_by[kind].items():
            e = {'model': kind, 'seed': seed}
            for c in ('mse_path', 'mse_1h', 'mse_6h', 'mse_24h', 'out_of_bounds_fraction'):
                e[c] = float(cmp_[c].mean())
            if kind != 'DIRECT':
                ref = comp_by['DIRECT'][seed] if seed in comp_by['DIRECT'] else None
                if ref is not None:
                    e.update(ctx.TP.mean_company_relative_gain(ref, cmp_))
            out.append(e)
    pd.DataFrame(out).to_csv(res / 'MAIN_REAL_RESULTS.csv', index=False)
    json.dump([{k: v for k, v in t.items() if k != 'curve'} for t in trials],
              open(res / 'TRIAL_REGISTRY.json', 'w'), indent=1)
    json.dump({t['model'] + '|' + t['config_id'] + '|' + str(t['fit']) + '|' + str(t['seed']): t['curve']
               for t in trials if t['curve']}, open(res / 'LEARNING_CURVES.json', 'w'))
    return lock


def phase_refit_from_lock(ctx, W: Path, res: Path):
    """Redo only the 15 final refits from an externally corrected lock.

    Reuses Ctx, build, train_run and score without modification. Runs no sweep,
    reads no development table, and touches no archived directory.
    """
    lock = json.loads((W / 'corrected_selection_lock_used.json').read_text())
    sel, steps = lock['selected_config'], lock['final_steps']
    st = ctx.fit_stats('full_2018')
    ev19 = ctx.eval_tensors(2019, 'reused_2019', st['state_scale'])
    (W / 'predictions').mkdir(exist_ok=True)
    np.savez_compressed(W / 'predictions' / 'truth_2019.npz',
                        truth=ev19['target'],                       # float64
                        y0=ev19['y0'].double().cpu().numpy().reshape(ev19['G'], ev19['N']),
                        groups=ctx.selected.group.to_numpy(),
                        companies=ctx.selected.company.to_numpy(),
                        origin_hours=ctx.origins['reused_2019'])
    trials, rows = [], []
    pg, pc, pr = ctx.score(persistence(ev19), ev19)
    for c_ in pc.itertuples():
        rows.append({'model': 'PERSISTENCE', 'seed': None, 'company': int(c_.company),
                     'mse_path': c_.mse_path, 'mse_1h': c_.mse_1h, 'mse_6h': c_.mse_6h,
                     'mse_24h': c_.mse_24h, 'oob': c_.out_of_bounds_fraction})
    comp_by = {'PERSISTENCE': {None: pc}}
    for kind in ARMS:
        cfg = next(c for c in candidates() if c['config_id'] == sel[kind])
        comp_by[kind] = {}
        for seed in FINAL_SEEDS:
            model, r = train_run(ctx, kind, cfg, 'full_2018', None, seed, steps[kind], log=trials)
            p = ctx.predict(model, ev19)
            g, cmp_, agg = ctx.score(p, ev19)
            comp_by[kind][seed] = cmp_
            np.savez_compressed(W / 'predictions' / f'pred_{kind}_seed{seed}.npz',
                                prediction=p.astype(np.float64),
                                model=kind, seed=seed, config_id=cfg['config_id'],
                                steps=steps[kind])
            g.to_csv(res / f'GROUP_{kind}_seed{seed}.csv', index=False)
            torch.save({'state_dict': model.state_dict(),
                        'record': {k: v for k, v in r.items() if k != 'curve'},
                        'state_scale': r['state_scale'], 'source_mean': r['source_mean'],
                        'config_id': cfg['config_id'], 'parameter_target': cfg['parameter_target'],
                        'lr': cfg['lr'], 'kind': kind, 'seed': seed, 'steps': steps[kind]},
                       W / 'main' / f'final_{kind}_seed{seed}.pt')
            for c_ in cmp_.itertuples():
                rows.append({'model': kind, 'seed': seed, 'company': int(c_.company),
                             'mse_path': c_.mse_path, 'mse_1h': c_.mse_1h,
                             'mse_6h': c_.mse_6h, 'mse_24h': c_.mse_24h,
                             'oob': c_.out_of_bounds_fraction})
            print(f"  refit {kind:7s} seed{seed} cfg={cfg['config_id']:15s} steps={steps[kind]} "
                  f"2019 path={agg['mse_path']:.6e} oob={agg['out_of_bounds_fraction']:.3e} "
                  f"({r['wall_seconds']}s)", flush=True)
    pd.DataFrame(rows).to_csv(res / 'MAIN_COMPANY_SEED.csv', index=False)
    out = []
    for kind in list(ARMS) + ['PERSISTENCE']:
        for seed, cmp_ in comp_by[kind].items():
            e = {'model': kind, 'seed': seed,
                 'config_id': sel.get(kind), 'steps': steps.get(kind)}
            for c in ('mse_path', 'mse_1h', 'mse_6h', 'mse_24h', 'out_of_bounds_fraction'):
                e[c] = float(cmp_[c].mean())
            if kind != 'DIRECT' and seed in comp_by['DIRECT']:
                e.update(ctx.TP.mean_company_relative_gain(comp_by['DIRECT'][seed], cmp_))
            out.append(e)
    pd.DataFrame(out).to_csv(res / 'MAIN_REAL_RESULTS.csv', index=False)
    json.dump([{k: v for k, v in t.items() if k != 'curve'} for t in trials],
              open(res / 'REFIT_TRIAL_REGISTRY.json', 'w'), indent=1)
    json.dump({f"{t['model']}|{t['seed']}": t['curve'] for t in trials},
              open(res / 'REFIT_CURVES.json', 'w'))


def phase_ablation(ctx, W: Path, res: Path):
    lock = json.loads((W / 'corrected_selection_lock_used.json').read_text())
    cfg = next(c for c in candidates() if c['config_id'] == lock['selected_config']['ASYM'])
    steps = lock['final_steps']['ASYM']
    trials, rows = [], []
    # A1: SR, matched ASYM hyperparameters, signed init chosen on 2018 only
    dev = []
    for si in (-.05, +.05):
        for seed in (DEV_SEED1, *DEV_SEEDS2):
            for fit, val in FOLDS:
                _, r = train_run(ctx, 'SR', cfg, fit, val, seed, MAX_DEV_STEPS,
                                 signed_init=si, log=trials)
                dev.append(r)
                print(f"  A1 SR init{si:+.2f} seed{seed} {fit[:6]} "
                      f"best={r['best_val_mse_path']:.6e}", flush=True)
    d = pd.DataFrame([{k: v for k, v in r.items() if k != 'curve'} for r in dev])
    d.to_csv(res / 'ABLATION_SR_DEV.csv', index=False)
    best_si = float(d.groupby('signed_init').best_val_mse_path.mean().sort_values().index[0])
    st = ctx.fit_stats('full_2018')
    ev19 = ctx.eval_tensors(2019, 'reused_2019', st['state_scale'])
    for name, kind, obj, si in (('SR', 'SR', 'path', best_si),
                                ('ASYM_STEP_ONLY', 'ASYM', 'one_step', None)):
        for seed in FINAL_SEEDS:
            model, r = train_run(ctx, kind, cfg, 'full_2018', None, seed, steps,
                                 signed_init=si, objective=obj, log=trials)
            g, cmp_, agg = ctx.score(ctx.predict(model, ev19), ev19)   # always 24-step recursive
            for c_ in cmp_.itertuples():
                rows.append({'model': name, 'seed': seed, 'company': int(c_.company),
                             'mse_path': c_.mse_path, 'mse_1h': c_.mse_1h,
                             'mse_6h': c_.mse_6h, 'mse_24h': c_.mse_24h,
                             'oob': c_.out_of_bounds_fraction})
            print(f"  A {name:15s} seed{seed} 2019 path={agg['mse_path']:.6e}", flush=True)
    pd.DataFrame(rows).to_csv(res / 'ABLATION_COMPANY_SEED.csv', index=False)
    summ = (pd.DataFrame(rows).groupby(['model', 'seed'])
            [['mse_path', 'mse_1h', 'mse_6h', 'mse_24h']].mean().reset_index())
    summ['signed_init_selected'] = best_si
    summ['matched_config'] = cfg['config_id']
    summ['note'] = 'conditional structural ablation at matched ASYM hyperparameters; not an independently tuned forecaster'
    summ.to_csv(res / 'ABLATION_RESULTS.csv', index=False)
    json.dump([{k: v for k, v in t.items() if k != 'curve'} for t in trials],
              open(res / 'ABLATION_TRIALS.json', 'w'), indent=1)


def phase_controlled(ctx, W: Path, res: Path):
    sys.path.insert(0, str(W / '_package/ASYMODE_CC_STANDALONE_20260909/code'))
    import controlled_data as CD
    CM, TP = ctx.CM, ctx.TP

    def tt(d, keys):
        return {k: torch.tensor(np.asarray(d[k]), dtype=torch.float32, device=DEVICE) for k in keys}

    def run(kind, cfg, data, seed, steps):
        st = data['fit_statistics']
        torch.manual_seed(seed)
        m = CM.TrajectoryModel(kind, 1, 24, 1, cfg['parameter_target'],
                               st['state_scale'], st['source_mean']).to(DEVICE)
        opt = torch.optim.AdamW(m.parameters(), lr=cfg['lr'], weight_decay=WD)
        scale = torch.tensor(st['state_scale'], dtype=torch.float32, device=DEVICE)
        fit = tt(data['fit'], ('c', 'y0', 'known_clock', 'target'))
        val = tt(data['validation'], ('c', 'y0', 'known_clock'))
        vt = np.asarray(data['validation']['clean_truth'])
        rng = np.random.default_rng(seed + 900000)
        best = {'metric': math.inf, 'update': 0, 'state': None}
        n = fit['c'].shape[0]
        for step in range(1, steps + 1):
            ix = torch.tensor(rng.integers(0, n, BATCH), device=DEVICE)
            opt.zero_grad(set_to_none=True)
            CM.path_loss(m(fit['c'][ix], fit['y0'][ix], fit['known_clock'][ix]),
                         fit['target'][ix], scale).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), CLIP); opt.step()
            if step % VALIDATE_EVERY == 0:
                with torch.no_grad():
                    p = m(val['c'], val['y0'], val['known_clock']).double().cpu().numpy()
                mse = float(((p - vt) ** 2).mean())
                if mse < best['metric']:
                    best = {'metric': mse, 'update': step,
                            'state': {k: v.clone() for k, v in m.state_dict().items()}}
        m.load_state_dict(best['state'])
        return m, best

    d0 = CD.make_controlled_splits(DEV_SEED1)
    picks, rows = {}, []
    for kind in ARMS:
        best = None
        for cfg in candidates():
            _, b = run(kind, cfg, d0, DEV_SEED1, MAX_DEV_STEPS)
            print(f"  E2 sel {kind:7s} {cfg['config_id']:14s} val={b['metric']:.6e}", flush=True)
            if best is None or b['metric'] < best[1]['metric']:
                best = (cfg, b)
        picks[kind] = best[0]
    for seed in FINAL_SEEDS:
        d = CD.make_controlled_splits(seed)
        for kind in ARMS:
            m, _ = run(kind, picks[kind], d, seed, MAX_DEV_STEPS)
            for split in ('source_eval', 'high_eval'):
                s = d[split]
                with torch.no_grad():
                    p = m(*[torch.tensor(np.asarray(s[k]), dtype=torch.float32, device=DEVICE)
                            for k in ('c', 'y0', 'known_clock')]).double().cpu().numpy()
                t = np.asarray(s['clean_truth'])
                rows.append({'model': kind, 'seed': seed, 'split': split,
                             'config_id': picks[kind]['config_id'],
                             'mse_path': float(((p - t) ** 2).mean()),
                             'mse_1h': float(((p[:, 0] - t[:, 0]) ** 2).mean()),
                             'mse_6h': float(((p[:, 5] - t[:, 5]) ** 2).mean()),
                             'mse_24h': float(((p[:, 23] - t[:, 23]) ** 2).mean())})
            print(f"  E2 {kind:7s} seed{seed} "
                  f"src={rows[-2]['mse_path']:.4e} high={rows[-1]['mse_path']:.4e}", flush=True)
    pd.DataFrame(rows).to_csv(res / 'CONTROLLED_RESULTS.csv', index=False)
    json.dump({k: v['config_id'] for k, v in picks.items()},
              open(res / 'CONTROLLED_SELECTED_CONFIGS.json', 'w'), indent=1)


def phase_package(ctx, W: Path, res: Path):
    env = {'python': sys.version.split()[0], 'torch': torch.__version__,
           'numpy': np.__version__, 'pandas': pd.__version__,
           'platform': platform.platform(), 'machine': platform.machine(),
           'device': str(DEVICE), 'cuda_available': torch.cuda.is_available(),
           'mps_available': torch.backends.mps.is_available(),
           'fp32': True, 'amp': False, 'torch_compile': False}
    (res / 'ENVIRONMENT.json').write_text(json.dumps(env, indent=2))
    lines = []
    for p in sorted(res.rglob('*')):
        if p.is_file() and p.name != 'SHA256SUMS.txt':
            import hashlib
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            lines.append(f'{h}  {p.relative_to(res)}')
    (res / 'SHA256SUMS.txt').write_text('\n'.join(lines) + '\n')
    print(f'packaged {len(lines)} result files')


def main():
    ap = argparse.ArgumentParser(description='AsymODE trajectory mainline campaign driver')
    ap.add_argument('--data-manifest', required=True)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--work', required=True)
    ap.add_argument('--phase', required=True, choices=['main', 'refit_from_lock', 'ablation', 'controlled', 'package'])
    ap.add_argument('--package-code', required=True)
    ap.add_argument('--preflight', required=True)
    a = ap.parse_args()
    W = Path(a.work); res = W / 'results'; res.mkdir(parents=True, exist_ok=True)
    (W / 'main').mkdir(exist_ok=True)
    torch.set_num_threads(8)
    ctx = Ctx(Path(a.data_manifest), Path(a.package_code), Path(a.preflight))
    print(f'[{a.phase}] sets={len(ctx.selected)} companies={ctx.selected.company.nunique()} device={DEVICE}', flush=True)
    t0 = time.perf_counter()
    {'main': phase_main, 'refit_from_lock': phase_refit_from_lock, 'ablation': phase_ablation,
     'controlled': phase_controlled, 'package': phase_package}[a.phase](ctx, W, res)
    print(f'[{a.phase}] done in {time.perf_counter()-t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
