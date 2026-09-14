"""Post-hoc descriptive references, written after the evaluation results had been seen.

Nothing here trains, tunes, selects a configuration, or chooses a county, event or
cell. It re-reads saved truth and prediction shards, result tables and the trial
registry, and writes results/POSTHOC_DESCRIPTIVES.json. The pre-registered estimands
stay in analyze.py; these numbers only help read them.

- US: a zero-training persistence path (y_hat[k] = y0) scored with the main
  event -> county -> origin -> lead weights; the untrained (update-0) NET and ASYM,
  rebuilt exactly as train_run builds them and checked against the update-0 loss in
  the trial registry, scored on 2022 and 2024; how far NET's raw outputs leave [0, 1];
  the event-equal Delta with each overlap component removed; 2022 validation curves
  recorded during development (locked configurations) and during the final fits
  (recorded, never used for stopping).
- Synthetic: test risk of the exact conditional mean against realized test paths
  (the irreducible part of new-event risk); validation curves of the final fits.
- Compute: per-run wall-clock totals from the trial registry.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scoring import event_level_mse  # noqa: E402

KINDS = ('NET', 'ASYM')
US_SEEDS = (8201, 8202, 8203, 8204, 8205)
US_ENDS = {'1h': 0, '6h': 5, '24h': 23}
SYN_SEEDS = (7201, 7202, 7203, 7204, 7205)
OOB_MASKS = {'below_0': lambda p: p < 0, 'above_1': lambda p: p > 1,
             'below_minus_1e-3': lambda p: p < -1e-3, 'below_minus_1e-2': lambda p: p < -1e-2}


def registry(A: Path) -> list:
    return [json.loads(l) for l in (A / 'logs/TRIAL_REGISTRY.jsonl').read_text().splitlines() if l.strip()]


def curve_summary(curve: list) -> dict:
    cv = pd.DataFrame(curve).sort_values('update')
    trained = cv[cv['update'] > 0]
    i = int(trained.val_path_mse.to_numpy().argmin())
    return {'val_update0': float(cv.val_path_mse.iloc[0]) if cv['update'].iloc[0] == 0 else None,
            'val_min_after_training': float(trained.val_path_mse.iloc[i]), 'update_at_min': int(trained['update'].iloc[i]),
            'val_last': float(cv.val_path_mse.iloc[-1]), 'last_update': int(cv['update'].iloc[-1])}


def us(A: Path) -> dict:
    comps = [list(c) for c in json.loads((A / 'locks/SPLITS.json').read_text())['splits']['evaluation']['overlap_components']]
    events = [e for c in comps for e in c]
    metrics = ['path_mse'] + [f'mse_{k}' for k in US_ENDS]
    rows, negative = [], []
    for e in events:
        t = np.load(A / f'predictions/us_era5/truth/{e}.npz')
        truth, fips = t['truth'].astype(np.float64), t['fips']
        ev = np.full(len(fips), e)

        def agg(v):
            return float(event_level_mse(v, ev, fips).iloc[0])

        pers = (t['y0'].astype(np.float64)[:, None] - truth) ** 2
        rec = {'event': e, 'n_windows': int(len(fips)), 'PERSISTENCE_path_mse': agg(pers.mean(1)),
               **{f'PERSISTENCE_mse_{k}': agg(pers[:, j]) for k, j in US_ENDS.items()}}
        for kind in KINDS:
            acc = {m: [] for m in metrics}
            for s in US_SEEDS:
                z = np.load(A / f'predictions/us_era5/{kind}/seed{s}/{e}.npz')
                if not (np.array_equal(z['row'], t['row']) and np.array_equal(z['t'], t['t'])):
                    raise SystemExit(f'row alignment failed: {kind} seed{s} {e}')
                p = z['pred'].astype(np.float64); se = (p - truth) ** 2
                acc['path_mse'].append(agg(se.mean(1)))
                for k, j in US_ENDS.items():
                    acc[f'mse_{k}'].append(agg(se[:, j]))
                if kind == 'NET':
                    for name, f in OOB_MASKS.items():
                        rec.setdefault(f'NET_frac_{name}_seeds', []).append(agg(f(p).mean(1)))
                    negative.append(p[p < 0])
            for m, v in acc.items():
                rec[f'{kind}_{m}_seed_mean'] = float(np.mean(v))
        rows.append(rec)
    E = pd.DataFrame(rows)
    summ = json.loads((A / 'results/us_era5/US_SUMMARY.json').read_text())['variants']['raw']
    overall = {'PERSISTENCE_path_mse': float(E.PERSISTENCE_path_mse.mean()),
               **{f'PERSISTENCE_mse_{k}': float(E[f'PERSISTENCE_mse_{k}'].mean()) for k in US_ENDS}}
    check = {}
    for kind in KINDS:
        for m in metrics:
            v = float(E[f'{kind}_{m}_seed_mean'].mean())
            overall[f'{kind}_{m}'] = v
            check[f'{kind}_{m}_abs_diff_vs_US_SUMMARY'] = abs(v - summ[f'{kind}_seed_mean'][m])
        overall[f'{kind}_path_relative_gain_vs_persistence'] = 1 - overall[f'{kind}_path_mse'] / overall['PERSISTENCE_path_mse']
    for name in OOB_MASKS:
        overall[f'NET_event_equal_fraction_{name}'] = float(np.mean([np.mean(x) for x in E[f'NET_frac_{name}_seeds']]))
    neg = np.concatenate(negative)
    overall['NET_negative_outputs_unweighted_quantiles'] = {f'q{int(round(q * 100)):02d}': float(np.quantile(neg, q)) for q in (.01, .10, .50, .90)}
    per_event = E[['event', 'n_windows', 'PERSISTENCE_path_mse', 'NET_path_mse_seed_mean', 'ASYM_path_mse_seed_mean']].to_dict('records')

    ES = pd.read_csv(A / 'results/us_era5/US_EVENT_SEED.csv'); ES = ES[ES.variant == 'raw']
    d = (ES[ES.model == 'NET'].pivot(index='seed', columns='event', values='path_mse')
         - ES[ES.model == 'ASYM'].pivot(index='seed', columns='event', values='path_mse'))
    loco = []
    for i, c in enumerate(comps):
        per_seed = d[[x for x in events if x not in c]].mean(axis=1)
        loco.append({'removed_component': i, 'removed_events': c, 'delta_path_seed_mean': float(per_seed.mean()),
                     'seeds_positive': int((per_seed > 0).sum())})

    lock = json.loads((A / 'locks/US_SELECTION_LOCK.json').read_text())['models']
    curves = []
    for r in registry(A):
        if r.get('task') != 'us_era5' or not r.get('curve'):
            continue
        if r.get('phase') == 'dev' and r.get('config_id') != lock[r['model']]['selected_config']:
            continue
        curves.append({'phase': r['phase'], 'model': r['model'], 'seed': r['seed'], 'config_id': r['config_id'],
                       **curve_summary(r['curve'])})
    return {'persistence_reference': {'definition': 'y_hat[k] = y0 for every lead; zero training; not a learned model',
                                      'overall': overall, 'per_event': per_event, 'consistency_check': check},
            'leave_one_component_out': loco, 'validation_curves_2022_locked_configs': curves}


def us_init_reference(G: Path, A: Path, threads: int) -> dict:
    """Update-0 models rebuilt exactly as us_train.train_run builds them (manual_seed, then build_model).

    The rebuilt 2022 validation loss is compared with the update-0 value in the trial registry, so this
    is the untrained model the locked training started from. No optimizer is built, nothing is trained."""
    import torch
    import us_train as UT
    from models import build_model
    torch.set_num_threads(threads)
    data = UT.USData(G, A / 'locks')
    lock = json.loads((A / 'locks/US_SELECTION_LOCK.json').read_text())['models']
    reg = {(r['model'], r['seed']): r for r in registry(A)
           if r.get('task') == 'us_era5' and r.get('phase') == 'final' and r.get('status') == 'COMPLETED'}
    tv, wv = data.truth64('validation'), data.weights['validation']
    te, we = data.truth64('evaluation'), data.weights['evaluation']
    de = data.df['evaluation']; ev, fips = de.event.to_numpy(), de.fips.to_numpy()
    keep = ('path_mse', 'mse_1h', 'mse_6h', 'mse_24h')
    persist = {}
    for split, t_, w_ in (('validation', tv, wv), ('evaluation', te, we)):
        m_ = UT.metrics(np.repeat(data.y0_64(split)[:, None], UT.H, axis=1), t_, w_)
        persist[split] = {k: m_[k] for k in keep}
    rows = []
    for kind in KINDS:
        sel = lock[kind]
        for s in US_SEEDS:
            torch.manual_seed(s)
            m = build_model(kind, UT.CTX_DIM, UT.STEP_DIM, UT.H, sel['parameter_target'], data.scale, data.source_mean)
            v0 = UT.metrics(UT.predict(m, data, 'validation'), tv, wv)['path_mse']
            pe = UT.predict(m, data, 'evaluation'); me = UT.metrics(pe, te, we)
            rows.append({'model': kind, 'seed': s, 'val2022_path_mse_rebuilt': v0,
                         'val2022_abs_diff_vs_registry_update0': abs(v0 - reg[(kind, s)]['curve'][0]['val_path_mse']),
                         **{f'eval2024_{k}': me[k] for k in keep + ('oob_fraction',)},
                         'eval2024_event_path_mse': event_level_mse(((pe - te) ** 2).mean(1), ev, fips).to_dict()})
    R = pd.DataFrame(rows)
    OS = pd.read_csv(A / 'results/us_era5/US_OVERALL_SEED.csv'); OS = OS[OS.variant == 'raw']
    ES = pd.read_csv(A / 'results/us_era5/US_EVENT_SEED.csv'); ES = ES[ES.variant == 'raw']
    summary = {}
    for kind in KINDS:
        q, o = R[R.model == kind], OS[OS.model == kind]
        summary[kind] = {'update0': {k: float(q['eval2024_' + k].mean()) for k in keep + ('oob_fraction',)},
                         'trained': {k: float(o[k].mean()) for k in keep + ('oob_fraction',)},
                         'val2022_update0': float(q.val2022_path_mse_rebuilt.mean())}
    d0 = R[R.model == 'NET'].set_index('seed').eval2024_path_mse - R[R.model == 'ASYM'].set_index('seed').eval2024_path_mse
    per_event = []
    for e in sorted(ES.event.unique()):
        rec = {'event': e}
        for kind in KINDS:
            rec[f'{kind}_update0'] = float(np.mean([x[e] for x in R[R.model == kind].eval2024_event_path_mse]))
            rec[f'{kind}_trained'] = float(ES[(ES.model == kind) & (ES.event == e)].path_mse.mean())
        per_event.append(rec)
    return {'definition': 'untrained update-0 models: torch.manual_seed(seed) then build_model with the locked configuration, '
                          'exactly as train_run; ASYM starts at u=.05*source_mean, r=.05*(1-source_mean), stay=.95 '
                          '(relaxation towards the train source mean); NET starts with a near-zero increment',
            'persistence_weight_vector': persist, 'per_seed': rows, 'summary': summary,
            'delta_update0_seed_mean': float(d0.mean()), 'delta_update0_by_seed': {int(k): float(v) for k, v in d0.items()},
            'delta_trained_seed_mean': summary['NET']['trained']['path_mse'] - summary['ASYM']['trained']['path_mse'],
            'max_val2022_abs_diff_vs_registry_update0': float(R.val2022_abs_diff_vs_registry_update0.max()),
            'per_event_seed_mean': per_event}


def synthetic(A: Path) -> dict:
    D = pd.read_csv(A / 'results/synthetic/SYNTH_FINAL_SEED_MODEL.csv')
    orc = []
    for (g, law), _ in D.groupby(['gamma', 'law']):
        for s in SYN_SEEDS:
            t = np.load(A / f'predictions/synthetic/truth/law{law}_g{g:g}_seed{s}.npz')
            Y, mu = t['test'].astype(np.float64), t['mu0']
            orc.append({'gamma': g, 'law': law, 'seed': s, 'oracle_test_path_mse': float(((Y - mu[:, None, :]) ** 2).mean())})
    M = D.merge(pd.DataFrame(orc), on=['gamma', 'law', 'seed'])
    M['excess_test_over_oracle'] = M.test_path_mse - M.oracle_test_path_mse
    cells = (M.groupby(['gamma', 'law', 'n_per_scene', 'model'])
             .agg(test_path_mse=('test_path_mse', 'mean'), oracle_test_path_mse=('oracle_test_path_mse', 'mean'),
                  vs_mu0_path_mse=('vs_mu0_path_mse', 'mean'), excess_test_over_oracle=('excess_test_over_oracle', 'mean'))
             .reset_index())
    LC = pd.read_csv(A / 'results/synthetic/SYNTH_LEARNING_CURVES.csv')
    keys = ['gamma', 'law', 'n_per_scene', 'seed', 'model']
    F = pd.DataFrame([{**dict(zip(keys, k)), **curve_summary(q.to_dict('records'))} for k, q in LC.groupby(keys)])
    F['min_at_last_update'] = F.update_at_min == F.last_update
    F['rel_excess_last_over_min'] = F.val_last / F.val_min_after_training - 1
    agg = dict(fits=('seed', 'size'), min_at_last_update=('min_at_last_update', 'sum'),
               median_rel_excess_last_over_min=('rel_excess_last_over_min', 'median'),
               max_rel_excess_last_over_min=('rel_excess_last_over_min', 'max'))
    return {'oracle_reference': {'definition': 'test risk of the exact conditional mean mu0 against realized test paths; '
                                               'new-event risk minus this is the estimation part',
                                 'cells': cells.to_dict('records')},
            'final_fit_validation_curves': {'by_model': F.groupby('model').agg(**agg).reset_index().to_dict('records'),
                                            'by_model_law': F.groupby(['model', 'law']).agg(**agg).reset_index().to_dict('records')}}


def compute(A: Path) -> dict:
    R = pd.DataFrame(registry(A))
    R['stage_or_phase'] = R['stage'].fillna(R['phase']).str.lower()
    T = (R.groupby(['task', 'stage_or_phase', 'model'])
         .agg(runs=('wall_seconds', 'size'), wall_hours=('wall_seconds', lambda s: s.sum() / 3600),
              mean_wall_seconds=('wall_seconds', 'mean'), threads=('threads', 'first')).reset_index())
    return {'note': 'sum of per-run wall-clock seconds in the trial registry; CPU only (no CUDA device); '
                    'excludes profiling, replay, scoring and analysis',
            'by_task_stage_model': T.to_dict('records'), 'total_wall_hours': float(R.wall_seconds.sum() / 3600)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument('--root', default=str(Path(__file__).resolve().parents[3]))
    ap.add_argument('--threads', type=int, default=2)
    a = ap.parse_args(); A, G = Path(a.work), Path(a.root)
    out = {'status': 'POST_HOC_DESCRIPTIVE', 'written_after_evaluation_results_were_seen': True,
           'delta_convention': 'MSE_NET - MSE_ASYM; positive favours ASYM',
           'us_era5': {**us(A), 'initialization_reference': us_init_reference(G, A, a.threads)},
           'synthetic': synthetic(A), 'compute': compute(A)}
    (A / 'results/POSTHOC_DESCRIPTIVES.json').write_text(
        json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)) + '\n')
    print('wrote results/POSTHOC_DESCRIPTIVES.json')


if __name__ == '__main__':
    main()
