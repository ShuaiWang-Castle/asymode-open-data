#!/usr/bin/env python3
"""Task B: ANEEL targeted-repair evaluation of NET and ASYM (no DIRECT, SR or ablation).

Order enforced: locks/ANEEL_SELECTION_LOCK.json must be committed and unchanged before any
2019 number is computed. Qualified local repair checkpoints are copied bit for bit,
verified against protocol 5.3, replayed by inference only, and one seed per model is
retrained as a determinism spot check. Scoring reuses the package scorer; the paired
direction is Delta = MSE_NET - MSE_ASYM.
"""
from __future__ import annotations
import argparse, hashlib, json, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch

HERE = Path(__file__).resolve().parent
KINDS, SEEDS = ('NET', 'ASYM'), (5101, 5102, 5103, 5104, 5105)
SPEC = {'NET': (32768, 3e-4, 3500), 'ASYM': (8192, 1e-3, 1000)}
BLOCK_HOURS, N_BOOT, BOOT_SEED = 168, 2000, 20260912
METRICS = ('mse_path', 'mse_1h', 'mse_6h', 'mse_24h', 'out_of_bounds_fraction')


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    for x in ('root', 'work', 'repair-dir', 'scratch'):
        ap.add_argument(f'--{x}', required=True)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--skip-retrain', action='store_true')
    a = ap.parse_args()
    G, A, REP = Path(a.root), Path(a.work), Path(a.repair_dir)
    R1 = G / 'analysis/asymode_trajectory_r1_20260909'; PKG = R1 / '_package/ASYMODE_CC_STANDALONE_20260909'
    lock_rel = 'analysis/net_asym_neural_era5_20260912/locks/ANEEL_SELECTION_LOCK.json'
    subprocess.run(['git', '-C', str(G), 'ls-files', '--error-unmatch', lock_rel], check=True, capture_output=True)
    if subprocess.run(['git', '-C', str(G), 'diff', '--quiet', 'HEAD', '--', lock_rel]).returncode != 0:
        raise SystemExit('BLOCKED: ANEEL selection lock changed after commit')
    lock = json.loads((G / lock_rel).read_text())
    if lock['status'] != 'PASS':
        raise SystemExit('BLOCKED: selection lock did not pass')
    out = A / 'results/aneel'; out.mkdir(parents=True, exist_ok=True)
    rep = {'selection_lock_sha256': sha(G / lock_rel), 'lock_commit': subprocess.run(
        ['git', '-C', str(G), 'log', '-1', '--format=%H', '--', lock_rel], capture_output=True, text=True).stdout.strip()}

    man = json.loads((R1 / 'data_resolution/DATA_MANIFEST.json').read_text())
    man['ledger_path'] = str(R1 / 'data_resolution/rebuilt_selected_ledgers.npz')
    man['selection_path'] = str(PKG / 'metadata/unit_selection.csv')
    rt = Path(a.scratch) / 'aneel_runtime_manifest.json'; rt.write_text(json.dumps(man))
    sys.path.insert(0, str(HERE))
    import aneel_driver_repaired as RD
    torch.set_num_threads(a.threads)
    ctx = RD.Ctx(rt, PKG / 'code', R1 / 'preflight')
    rep['origin_counts'] = {k: int(len(v)) for k, v in ctx.origins.items()}
    rep['origin_counts_match_protocol'] = rep['origin_counts'] == {'fold_A_fit': 3745, 'fold_A_val': 1920, 'fold_B_fit': 5665,
                                                                    'fold_B_val': 1920, 'full_2018': 7608, 'reused_2019': 7608}
    rep['collections'], rep['companies'] = int(len(ctx.selected)), int(ctx.selected.company.nunique())
    st = ctx.fit_stats('full_2018')
    ev = ctx.eval_tensors(2019, 'reused_2019', st['state_scale'])
    truth, org = ev['target'], ctx.origins['reused_2019']
    groups, comp = ctx.selected.group.to_numpy(), ctx.selected.company.to_numpy()
    y0 = np.stack([np.asarray(ctx.ledger[f'g{g}_y2019_y'], dtype=np.float64)[org] for g in groups])
    if not np.array_equal(y0.astype(np.float32).ravel(), ev['y0'].numpy()):
        raise SystemExit('BLOCKED: float64 y0 does not match the scored float32 input')
    pdir = A / 'predictions/aneel'; pdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pdir / 'truth_2019.npz', truth=truth, y0=y0, groups=groups, companies=comp, origin_hours=org)

    cdir = A / 'checkpoints/aneel'; cdir.mkdir(parents=True, exist_ok=True)
    preds, ver = {}, []
    for kind in KINDS:
        for seed in SEEDS:
            src, dst = REP / 'main' / f'final_{kind}_seed{seed}.pt', cdir / f'{kind}_seed{seed}.pt'
            shutil.copy2(src, dst)
            if sha(src) != sha(dst):
                raise SystemExit('BLOCKED: checkpoint copy differs')
            ck = torch.load(dst, map_location='cpu', weights_only=False)
            meta_ok = ((int(ck['parameter_target']), float(ck['lr']), int(ck['steps'])) == SPEC[kind]
                       and ck['kind'] == kind and int(ck['seed']) == seed and ck['record']['fit'] == 'full_2018')
            stats_ok = ck['state_scale'] == st['state_scale'] and ck['source_mean'] == st['source_mean']
            m = ctx.CM.TrajectoryModel(kind, 168, 24, 5, ck['parameter_target'], ck['state_scale'], ck['source_mean'])
            m.load_state_dict(ck['state_dict'])
            t0 = time.perf_counter(); p = ctx.predict(m, ev); inf = time.perf_counter() - t0
            ref = np.load(REP / 'predictions' / f'pred_{kind}_seed{seed}.npz')['prediction']
            f32_lossless = bool(np.array_equal(p.astype(np.float32).astype(np.float64), p))
            (pdir / kind).mkdir(exist_ok=True)
            np.savez_compressed(pdir / kind / f'seed{seed}.npz', pred=p.astype(np.float32))
            preds[(kind, seed)] = p
            ver.append({'model': kind, 'seed': seed, 'checkpoint_sha256': sha(dst), 'metadata_matches_protocol_5_3': meta_ok,
                        'fit_statistics_match_full_2018': stats_ok, 'width': int(m.width),
                        'n_params': int(sum(q.numel() for q in m.parameters())),
                        'replay_max_abs_diff_vs_repair_predictions': float(np.abs(p - ref).max()),
                        'prediction_float32_lossless': f32_lossless, 'inference_seconds': round(inf, 2)})
            if not (meta_ok and stats_ok and f32_lossless):
                raise SystemExit(f'BLOCKED: {kind} seed{seed} failed qualification')
            print(f"  replay {kind:4s} seed{seed} diff_vs_repair={ver[-1]['replay_max_abs_diff_vs_repair_predictions']:.1e}", flush=True)
    rep['checkpoints'] = ver

    spot = []
    if not a.skip_retrain:
        for kind in KINDS:
            cfg = next(c for c in RD.candidates() if c['config_id'] == lock['selection'][kind]['selected_config'])
            t0 = time.perf_counter()
            model, r = RD.train_run(ctx, kind, cfg, 'full_2018', None, SEEDS[0], lock['selection'][kind]['final_updates'])
            p = ctx.predict(model, ev)
            spot.append({'model': kind, 'seed': SEEDS[0], 'config_id': cfg['config_id'], 'updates': lock['selection'][kind]['final_updates'],
                         'threads': torch.get_num_threads(), 'wall_seconds': round(time.perf_counter() - t0, 1),
                         'max_abs_diff_retrained_vs_checkpoint': float(np.abs(p - preds[(kind, SEEDS[0])]).max())})
            print(f"  retrain spot check {kind} seed{SEEDS[0]}: max diff {spot[-1]['max_abs_diff_retrained_vs_checkpoint']:.1e}", flush=True)
    rep['retrain_spot_check'] = spot

    pers = np.repeat(y0[:, :, None], 24, axis=2)
    seed_rows, comp_rows, grp_rows = [], [], []
    comp_tables = {}
    for variant, f in (('raw', lambda z: z), ('output_clip_01', lambda z: np.clip(z, 0.0, 1.0))):
        for kind in KINDS + ('PERSISTENCE',):
            for seed in (SEEDS if kind != 'PERSISTENCE' else (None,)):
                p = f(preds[(kind, seed)] if kind != 'PERSISTENCE' else pers)
                g, c, r = ctx.score(p, ev)
                seed_rows.append({'variant': variant, 'model': kind, 'seed': seed, **r})
                comp_tables[(variant, kind, seed)] = c
                for x in g.itertuples():
                    grp_rows.append({'variant': variant, 'model': kind, 'seed': seed, 'group': x.group, 'company': x.company,
                                     **{k: getattr(x, k) for k in METRICS}})
                for x in c.itertuples():
                    comp_rows.append({'variant': variant, 'model': kind, 'seed': seed, 'company': x.company,
                                      **{k: getattr(x, k) for k in METRICS}})
    sd, cd, gd = pd.DataFrame(seed_rows), pd.DataFrame(comp_rows), pd.DataFrame(grp_rows)
    sd.to_csv(out / 'ANEEL_SEED_SUMMARY.csv', index=False)
    cd.to_csv(out / 'ANEEL_COMPANY_SEED.csv', index=False); gd.to_csv(out / 'ANEEL_COLLECTION_SEED.csv', index=False)

    def paired(df, keys):
        n = df[df.model == 'NET'].set_index(keys)[list(METRICS)]
        s = df[df.model == 'ASYM'].set_index(keys)[list(METRICS)]
        d = (n - s).add_prefix('delta_'); return pd.concat([n.add_prefix('NET_'), s.add_prefix('ASYM_'), d], axis=1).reset_index()
    pc, pg = paired(cd, ['variant', 'seed', 'company']), paired(gd, ['variant', 'seed', 'group', 'company'])
    pc.to_csv(out / 'ANEEL_COMPANY_PAIRED_DELTA.csv', index=False); pg.to_csv(out / 'ANEEL_COLLECTION_PAIRED_DELTA.csv', index=False)
    pcm = pc.groupby(['variant', 'company'])[[c for c in pc.columns if c.startswith(('NET_', 'ASYM_', 'delta_'))]].mean().reset_index()
    pcm.to_csv(out / 'ANEEL_COMPANY_PAIRED_DELTA_SEED_MEAN.csv', index=False)

    blocks = org // BLOCK_HOURS; ub = np.unique(blocks); bi = np.searchsorted(ub, blocks); nb = len(ub)
    cnt = np.bincount(bi, minlength=nb).astype(float) * 24
    companies = np.unique(comp)
    summary = {'delta_convention': 'MSE_NET - MSE_ASYM; positive supports ASYM', 'block_hours': BLOCK_HOURS, 'n_blocks': int(nb),
               'n_resamples': N_BOOT, 'status': 'DESCRIPTIVE_DEPENDENCE_SENSITIVITY_ON_REUSED_2019', 'variants': {}}
    for variant in ('raw', 'output_clip_01'):
        f = (lambda z: z) if variant == 'raw' else (lambda z: np.clip(z, 0.0, 1.0))
        S = {(k, s): np.stack([np.bincount(bi, weights=((f(preds[(k, s)])[g] - truth[g]) ** 2).sum(axis=1), minlength=nb)
                               for g in range(len(groups))]) for k in KINDS for s in SEEDS}
        def overall(k, s, mult):
            gm = (S[(k, s)] @ mult) / (cnt @ mult)
            return float(pd.Series(gm).groupby(comp).mean().mean())
        ones = np.ones(nb)
        point = float(np.mean([overall('NET', s, ones) - overall('ASYM', s, ones) for s in SEEDS]))
        rng = np.random.default_rng(BOOT_SEED); draws = []
        for _ in range(N_BOOT):
            mult = np.bincount(rng.integers(0, nb, nb), minlength=nb).astype(float)
            draws.append(np.mean([overall('NET', s, mult) - overall('ASYM', s, mult) for s in SEEDS]))
        per_seed = []
        for s in SEEDS:
            g = RD.__dict__  # placeholder to keep namespace explicit
            rel = ctx.TP.mean_company_relative_gain(comp_tables[(variant, 'NET', s)], comp_tables[(variant, 'ASYM', s)])
            per_seed.append({'seed': s, 'delta_path_company_equal': overall('NET', s, ones) - overall('ASYM', s, ones),
                             'asym_vs_net_mean_company_relative_gain': rel['mean_company_relative_gain'],
                             'asym_vs_net_ratio_of_means_gain': rel['ratio_of_means_gain'],
                             'zero_net_denominator_companies': rel['zero_direct_denominator_companies']})
        dcomp = pcm[pcm.variant == variant].delta_mse_path
        summary['variants'][variant] = {
            'delta_path_point_seed_mean': point,
            'delta_path_block_p2.5': float(np.percentile(draws, 2.5)), 'delta_path_block_p50': float(np.percentile(draws, 50)),
            'delta_path_block_p97.5': float(np.percentile(draws, 97.5)), 'per_seed': per_seed,
            'mean_company_relative_gain_seed_mean': float(np.mean([x['asym_vs_net_mean_company_relative_gain'] for x in per_seed])),
            'ratio_of_means_gain_seed_mean': float(np.mean([x['asym_vs_net_ratio_of_means_gain'] for x in per_seed])),
            'seeds_with_positive_delta': int(sum(x['delta_path_company_equal'] > 0 for x in per_seed)),
            'companies_with_positive_seed_mean_delta': int((dcomp > 0).sum()), 'companies': int(len(dcomp)),
            'endpoint_delta_seed_mean': {k: float(sd[(sd.variant == variant) & (sd.model == 'NET')][k].mean()
                                                  - sd[(sd.variant == variant) & (sd.model == 'ASYM')][k].mean())
                                         for k in ('mse_1h', 'mse_6h', 'mse_24h')}}
    ind = []
    for kind in KINDS:
        p = preds[(kind, SEEDS[0])]
        e = (p - truth) ** 2
        long = pd.DataFrame({'company': np.repeat(comp, e.shape[1]), 'group': np.repeat(groups, e.shape[1]),
                             'se': e.mean(axis=2).ravel()})
        mine = long.groupby(['company', 'group']).se.mean().groupby('company').mean().mean()
        ref = sd[(sd.variant == 'raw') & (sd.model == kind) & (sd.seed == SEEDS[0])].mse_path.iloc[0]
        ind.append({'model': kind, 'seed': SEEDS[0], 'independent_company_equal_path_mse': float(mine),
                    'package_scorer_path_mse': float(ref), 'abs_diff': float(abs(mine - ref))})
    summary['independent_scorer_check'] = ind
    rep['summary'] = summary
    (out / 'ANEEL_PAIRED_SUMMARY.json').write_text(json.dumps(summary, indent=1, default=float))
    (out / 'ANEEL_CHECKPOINT_VERIFICATION.json').write_text(json.dumps({k: v for k, v in rep.items() if k != 'summary'}, indent=1, default=float))
    for v, s in summary['variants'].items():
        print(f"  {v:15s} Delta(NET-ASYM) path = {s['delta_path_point_seed_mean']:+.4e}  block 95% [{s['delta_path_block_p2.5']:+.3e}, "
              f"{s['delta_path_block_p97.5']:+.3e}]  seeds+ {s['seeds_with_positive_delta']}/5  companies+ "
              f"{s['companies_with_positive_seed_mean_delta']}/{s['companies']}  relgain {s['mean_company_relative_gain_seed_mean']:+.4%}", flush=True)
    print('  independent scorer max diff:', max(x['abs_diff'] for x in ind))


if __name__ == '__main__':
    main()
