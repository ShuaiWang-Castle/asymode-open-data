#!/usr/bin/env python3
"""Pre-registered analysis for the affected-county CV round. Written before any DEV or final result exists.

Reads the truth and prediction shards, the fold and selection locks and the run logs; writes results/ and
figures/. Delta = MSE_NET - MSE_ASYM; positive favours ASYM.

Scoring: squared loss per seed first, then averaged over seeds; within an event, counties equal and origins
equal within a county, leads equal. Populations: all legal test windows, affected county-events (peak >= 1%,
conditional on realized impact) and unaffected county-events. Groupings: all 26 events, the three type
groups and the five catalogue types. Independent units are overlap components: the sign-flip test runs over
component deltas (exact up to 16 components, Monte Carlo above), the interval resamples components.
Decision per grouping and population: 'supports ASYM' if p < 0.05 and both the event-equal and the
component-mean Delta are positive; 'supports NET' if p < 0.05 and both are negative; otherwise 'undecided'.
References that are not models and select nothing: persistence, the untrained (update-0) models, mean signed
errors, and a separability descriptor (share of windows whose observed path both rises and falls).
Secondary metric (PI decision of 2026-09-14): event RMSE = sqrt of the event path MSE, per seed, reported beside
MSE with its own component sign-flip p. The decision above uses MSE only.
"""
from __future__ import annotations
import argparse, itertools, json, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
W = HERE.parent
sys.path.insert(0, str(HERE))
import cv_data as CD                      # noqa: E402

KINDS = ('NET', 'ASYM')
FINAL_SEEDS = (9201, 9202, 9203, 9204, 9205)
POPS = ('all', 'affected', 'unaffected')
ENDS = {'1h': 0, '6h': 5, '24h': 23}
MC_DRAWS, BOOT_DRAWS, EXACT_MAX, RNG_SEED = 200_000, 4000, 16, 20260914
CHANGE_TOL = 1e-3


def county_equal(values, fips):
    """Per-window values (n,) or (n, m) -> mean over counties of the mean over each county's windows."""
    codes, _ = pd.factorize(np.asarray(fips))
    n = np.bincount(codes).astype(float)
    v = np.asarray(values, float)
    if v.ndim == 1:
        return float((np.bincount(codes, weights=v) / n).mean())
    return np.array([(np.bincount(codes, weights=v[:, j]) / n).mean() for j in range(v.shape[1])])


def read_logs(pattern):
    recs = []
    for f in sorted((W / 'logs').glob(pattern)):
        recs += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    return [r for r in recs if r.get('status') == 'COMPLETED']


def event_tables(lock):
    fold_of = {e: f['fold'] for f in lock['folds'] for e in f['events']}
    rows, lead_rows, sep_rows = [], [], []
    for e, typ in sorted(lock['event_types'].items()):
        k = fold_of[e]; T = np.load(W / f'predictions/truth/{e}.npz')
        truth, y0, fips, aff = T['truth'].astype(np.float64), T['y0'].astype(np.float64), T['fips'], T['affected'].astype(bool)
        steps = np.diff(np.concatenate([y0[:, None], truth], axis=1), axis=1)
        rises, falls = (steps >= CHANGE_TOL).any(axis=1), (steps <= -CHANGE_TOL).any(axis=1)
        preds = {}
        for kind in KINDS:
            for s in FINAL_SEEDS:
                z = np.load(W / f'predictions/fold{k}/{kind}/seed{s}/{e}.npz')
                if not (np.array_equal(z['row'], T['row']) and np.array_equal(z['t'], T['t'])):
                    raise SystemExit(f'window keys differ: {e} {kind} seed{s}')
                preds[(kind, s)] = z['pred'].astype(np.float64)
        preds[('PERSISTENCE', 0)] = np.repeat(y0[:, None], 24, axis=1)
        for pop in POPS:
            m = np.ones(len(fips), bool) if pop == 'all' else (aff if pop == 'affected' else ~aff)
            if not m.any():
                continue
            base = {'event': e, 'fold': k, 'type': typ, 'type_group': lock['type_group_map'][typ], 'population': pop,
                    'n_windows': int(m.sum()), 'n_counties': int(len(set(fips[m].tolist())))}
            fm = fips[m]
            sep_rows.append({**base, 'any_change': county_equal((rises | falls)[m], fm),
                             'rise_and_fall': county_equal((rises & falls)[m], fm),
                             'rise_only': county_equal((rises & ~falls)[m], fm), 'fall_only': county_equal((falls & ~rises)[m], fm)})
            for (kind, s), P in preds.items():
                err = P[m] - truth[m]; se = err ** 2
                rows.append({**base, 'model': kind, 'seed': s, 'path_mse': county_equal(se.mean(axis=1), fm),
                             **{f'mse_{h}': county_equal(se[:, j], fm) for h, j in ENDS.items()},
                             'mean_signed_error': county_equal(err.mean(axis=1), fm),
                             'oob_fraction': county_equal(((P[m] < 0) | (P[m] > 1)).mean(axis=1), fm)})
                lead = county_equal(se, fm)
                lead_rows += [{'event': e, 'population': pop, 'model': kind, 'seed': s, 'lead_h': j + 1, 'mse': float(lead[j])}
                              for j in range(24)]
    return pd.DataFrame(rows), pd.DataFrame(lead_rows), pd.DataFrame(sep_rows)


def paired(M):
    out = []
    learned = M[M.model != 'PERSISTENCE']
    for (e, pop), g in learned.groupby(['event', 'population']):
        n, a = g[g.model == 'NET'].set_index('seed'), g[g.model == 'ASYM'].set_index('seed')
        d = n.path_mse - a.path_mse
        p = M[(M.event == e) & (M.population == pop) & (M.model == 'PERSISTENCE')].iloc[0]
        r0 = g.iloc[0]
        out.append({'event': e, 'fold': int(r0.fold), 'type': r0.type, 'type_group': r0.type_group, 'population': pop,
                    'n_windows': int(r0.n_windows), 'n_counties': int(r0.n_counties),
                    'NET_path_mse': float(n.path_mse.mean()), 'ASYM_path_mse': float(a.path_mse.mean()),
                    'PERSISTENCE_path_mse': float(p.path_mse),
                    'NET_path_rmse': float(np.sqrt(n.path_mse).mean()), 'ASYM_path_rmse': float(np.sqrt(a.path_mse).mean()),
                    'PERSISTENCE_path_rmse': float(np.sqrt(p.path_mse)),
                    'delta_rmse_seed_mean': float((np.sqrt(n.path_mse) - np.sqrt(a.path_mse)).mean()),
                    'seeds_delta_rmse_positive': int(((np.sqrt(n.path_mse) - np.sqrt(a.path_mse)) > 0).sum()),
                    **{f'delta_rmse_{h}': float((np.sqrt(n[f'mse_{h}']) - np.sqrt(a[f'mse_{h}'])).mean()) for h in ENDS},
                    'delta_seed_mean': float(d.mean()), 'delta_seed_sd': float(d.std(ddof=1)), 'seeds_delta_positive': int((d > 0).sum()),
                    **{f'delta_mse_{h}': float((n[f'mse_{h}'] - a[f'mse_{h}']).mean()) for h in ENDS},
                    'NET_mean_signed_error': float(n.mean_signed_error.mean()), 'ASYM_mean_signed_error': float(a.mean_signed_error.mean()),
                    'PERSISTENCE_mean_signed_error': float(p.mean_signed_error), 'NET_oob_fraction': float(n.oob_fraction.mean())})
    return pd.DataFrame(out)


def signflip_p(x, rng):
    x = np.asarray(x, float); n = len(x); obs = abs(x.mean())
    if n <= EXACT_MAX:
        signs = np.array(list(itertools.product((-1.0, 1.0), repeat=n)))
        return float((np.abs((signs * x).mean(axis=1)) >= obs - 1e-15).mean()), 'exact'
    hits = 0
    for i in range(0, MC_DRAWS, 50_000):
        s = rng.choice((-1.0, 1.0), size=(min(50_000, MC_DRAWS - i), n))
        hits += int((np.abs((s * x).mean(axis=1)) >= obs - 1e-15).sum())
    return hits / MC_DRAWS, 'monte_carlo'


def group_summary(P, M, lock, pop, events, label, rng):
    q = P[(P.population == pop) & P.event.isin(events)]
    if q.empty:
        return None
    comp_of = {e: i for i, c in enumerate(lock['components']) for e in c['events']}
    comps = {}
    for e in sorted(q.event):
        comps.setdefault(comp_of[e], []).append(e)
    keys = list(comps.values())
    ev = q.set_index('event').delta_seed_mean
    cdelta = np.array([ev[c].mean() for c in keys])
    p, method = signflip_p(cdelta, rng)
    boot = [ev[[e for i in rng.integers(0, len(keys), len(keys)) for e in keys[i]]].mean() for _ in range(BOOT_DRAWS)]
    point, cmean = float(ev.mean()), float(cdelta.mean())
    if p < 0.05 and point > 0 and cmean > 0:
        decision = 'supports ASYM'
    elif p < 0.05 and point < 0 and cmean < 0:
        decision = 'supports NET'
    else:
        decision = 'undecided'
    evr = q.set_index('event').delta_rmse_seed_mean
    cdelta_r = np.array([evr[c].mean() for c in keys])
    p_r, _ = signflip_p(cdelta_r, np.random.default_rng(RNG_SEED + 1))
    s = M[(M.population == pop) & M.event.isin(events) & (M.model != 'PERSISTENCE')]
    piv = s.pivot_table(index=['event', 'seed'], columns='model', values='path_mse')
    seed_delta = (piv['NET'] - piv['ASYM']).unstack('seed').mean(axis=0)
    return {'grouping': label, 'population': pop, 'events': int(len(q)), 'components': int(len(keys)),
            'delta_event_equal': point, 'delta_component_mean': cmean, 'components_positive': int((cdelta > 0).sum()),
            'events_positive': int((q.delta_seed_mean > 0).sum()),
            'signflip_p_components': p, 'signflip_method': method,
            'min_attainable_p': (2.0 / 2 ** len(keys)) if len(keys) <= EXACT_MAX else None,
            'component_bootstrap_p2.5': float(np.percentile(boot, 2.5)), 'component_bootstrap_p97.5': float(np.percentile(boot, 97.5)),
            'seeds_event_equal_delta_positive': int((seed_delta > 0).sum()),
            'NET_path_mse_event_equal': float(q.NET_path_mse.mean()), 'ASYM_path_mse_event_equal': float(q.ASYM_path_mse.mean()),
            'PERSISTENCE_path_mse_event_equal': float(q.PERSISTENCE_path_mse.mean()),
            'mean_event_relative_gain_asym_vs_net': float(((q.NET_path_mse - q.ASYM_path_mse) / q.NET_path_mse).mean()),
            'ratio_of_event_means_gain': float((q.NET_path_mse.mean() - q.ASYM_path_mse.mean()) / q.NET_path_mse.mean()),
            **{f'delta_mse_{h}_event_equal': float(q[f'delta_mse_{h}'].mean()) for h in ENDS},
            'decision': decision,
            'delta_rmse_event_equal': float(evr.mean()), 'delta_rmse_component_mean': float(cdelta_r.mean()),
            'components_positive_rmse': int((cdelta_r > 0).sum()), 'signflip_p_components_rmse_secondary': p_r,
            'NET_path_rmse_event_equal': float(q.NET_path_rmse.mean()), 'ASYM_path_rmse_event_equal': float(q.ASYM_path_rmse.mean()),
            'PERSISTENCE_path_rmse_event_equal': float(q.PERSISTENCE_path_rmse.mean()),
            'rmse_direction_agrees_with_mse': bool(np.sign(evr.mean()) == np.sign(point))}


def untrained_reference(G, threads, sel, finals):
    import torch
    import cv_train as CT
    from models_v2 import build_model
    torch.set_num_threads(threads)
    data = CD.CVData(G, W / 'locks')
    logs = {(r['fold'], r['model'], r['seed']): r for r in finals}
    rows, worst = [], 0.0
    for k in range(CD.K_FOLDS):
        data.set_fold(k)
        tv, wv, te, dt = data.truth64('validation'), data.weights['validation'], data.truth64('test'), data.df['test']
        for kind in KINDS:
            cfg = sel['folds'][str(k)][kind]
            for s in FINAL_SEEDS:
                torch.manual_seed(s + 1000 * k)
                m = build_model(kind, cfg['structure'], CT.CTX_DIM, CT.STEP_DIM, CT.H, CT.TARGET, data.scale, data.source_mean)
                v0 = CT.metrics(CT.predict(m, data, 'validation'), tv, wv)['path_mse']
                worst = max(worst, abs(v0 - logs[(k, kind, s)]['curve'][0]['val_path_mse']))
                pe = CT.predict(m, data, 'test')
                for e, g in dt.groupby('event'):
                    ix = g.index.to_numpy(); aff = g.affected.to_numpy(); fips = g.fips.to_numpy()
                    se = ((pe[ix] - te[ix]) ** 2).mean(axis=1)
                    for pop in POPS:
                        mm = np.ones(len(ix), bool) if pop == 'all' else (aff if pop == 'affected' else ~aff)
                        if mm.any():
                            rows.append({'event': e, 'population': pop, 'model': kind, 'seed': s,
                                         'untrained_path_mse': county_equal(se[mm], fips[mm])})
    return pd.DataFrame(rows), worst


def figures(P, M, L):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    F = W / 'figures'; F.mkdir(parents=True, exist_ok=True)
    col = {'winter': '#1f77b4', 'wind_tropical': '#d62728', 'convective_flood': '#2ca02c'}
    order = P[P.population == 'all'].sort_values(['type_group', 'event']).event.tolist()
    jr = np.random.default_rng(1)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8.5), sharex=True)
    for ax, pop in zip(axes, ('all', 'affected')):
        q = P[P.population == pop].set_index('event').reindex(order)
        ax.bar(range(len(order)), q.delta_seed_mean.to_numpy(), color=[col.get(g, '0.5') for g in q.type_group])
        s = M[(M.population == pop) & (M.model != 'PERSISTENCE')].pivot_table(index=['event', 'seed'], columns='model', values='path_mse')
        d = (s['NET'] - s['ASYM']).unstack('seed').reindex(order)
        for i, e in enumerate(order):
            v = d.loc[e].to_numpy(); ax.scatter(i + jr.uniform(-.2, .2, len(v)), v, s=9, color='k', zorder=3)
        ax.axhline(0, color='k', lw=.8); ax.set_yscale('symlog', linthresh=1e-5)
        ax.set_ylabel(f'Delta = MSE_NET - MSE_ASYM ({pop})')
    axes[0].set_title('Per-event paired Delta (bar = seed mean, dots = seeds; colour = type group; symlog axis)')
    axes[1].set_xticks(range(len(order))); axes[1].set_xticklabels(order, rotation=60, fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in col.values()]
    axes[0].legend(handles, list(col), fontsize=8)
    fig.tight_layout(); fig.savefig(F / 'cv_event_delta_by_type.png', dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    ta = M[M.population == 'affected'][['event', 'type_group']].drop_duplicates()
    for ax, grp in zip(axes, col):
        evs = ta[ta.type_group == grp].event
        q = L[(L.population == 'affected') & L.event.isin(evs)]
        for kind, c in (('NET', '#1f77b4'), ('ASYM', '#d62728'), ('PERSISTENCE', '#555555')):
            z = q[q.model == kind].groupby(['event', 'lead_h']).mse.mean().groupby('lead_h').mean()
            ax.plot(z.index, z.to_numpy(), color=c, ls='--' if kind == 'PERSISTENCE' else '-', label=kind)
        ax.set_title(f'{grp} (affected): event-equal MSE by lead'); ax.set_xlabel('lead (h)')
    axes[0].set_ylabel('MSE (shared scale)'); axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(F / 'cv_lead_mse_by_group.png', dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, grp in zip(axes, col):
        evs = ta[ta.type_group == grp].event
        q = L[(L.population == 'affected') & L.event.isin(evs) & (L.model != 'PERSISTENCE')]
        piv = q.pivot_table(index=['event', 'seed', 'lead_h'], columns='model', values='mse')
        d = (piv['NET'] - piv['ASYM']).groupby(['event', 'lead_h']).mean().groupby('lead_h').mean()
        ax.plot(d.index, d.to_numpy(), color='k'); ax.axhline(0, color='k', lw=.6)
        ax.set_title(f'{grp} (affected): Delta by lead'); ax.set_xlabel('lead (h)')
    axes[0].set_ylabel('Delta (signed)')
    fig.tight_layout(); fig.savefig(F / 'cv_lead_delta_by_group.png', dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=str(W.parents[1])); ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--skip-untrained', action='store_true')
    a = ap.parse_args()
    rng = np.random.default_rng(RNG_SEED)
    lock = json.loads((W / 'locks/FOLDS.json').read_text())
    sel = json.loads((W / 'locks/SELECTION_LOCK.json').read_text())
    finals = read_logs('final_w*.jsonl')
    got = {(r['fold'], r['model'], r['seed']) for r in finals}
    missing = [(k, m, s) for k in range(CD.K_FOLDS) for m in KINDS for s in FINAL_SEEDS if (k, m, s) not in got]
    if missing:
        raise SystemExit(f'missing final runs: {missing}')
    by = {(r['fold'], r['model'], r['seed']): r for r in finals}
    checks = {'final_runs': len(finals),
              'paired_first20_batch_ids_identical': all(by[(k, 'NET', s)]['first20_batch_ids_sha256'] == by[(k, 'ASYM', s)]['first20_batch_ids_sha256']
                                                        for k in range(CD.K_FOLDS) for s in FINAL_SEEDS),
              'parameter_counts_within_1pct': all(abs(r['n_params'] / r['parameter_target'] - 1) <= 0.01 for r in finals),
              'selection_lock_sha256_single': len({r['selection_lock_sha256'] for r in finals}) == 1,
              'folds_lock_sha256_single': len({r['folds_lock_sha256'] for r in finals}) == 1}
    M, L, SEP = event_tables(lock)
    P = paired(M)
    R = W / 'results'; R.mkdir(parents=True, exist_ok=True)
    M.to_csv(R / 'CV_EVENT_MODEL_SEED.csv', index=False); L.to_csv(R / 'CV_LEAD_EVENT_MODEL_SEED.csv', index=False)
    P.to_csv(R / 'CV_EVENT_PAIRED.csv', index=False); SEP.to_csv(R / 'CV_SEPARABILITY.csv', index=False)
    groupings = [('all_events', sorted(lock['event_types']))]
    groupings += [(f'type_group:{g}', sorted(e for e, t in lock['event_types'].items() if lock['type_group_map'][t] == g))
                  for g in CD.GROUP_ORDER]
    groupings += [(f'type:{t}', sorted(e for e, tt in lock['event_types'].items() if tt == t))
                  for t in sorted(set(lock['event_types'].values()))]
    summ = [s for pop in POPS for label, evs in groupings if (s := group_summary(P, M, lock, pop, evs, label, rng))]
    S = pd.DataFrame(summ); S.to_csv(R / 'CV_GROUP_SUMMARY.csv', index=False)
    sel_rows = []
    for k, v in sel['folds'].items():
        for kind in KINDS:
            rk = v[kind]['ranking']
            sel_rows.append({'fold': int(k), 'model': kind, 'structure': v[kind]['structure'], 'lr': v[kind]['lr'],
                             'dev_mean_best_val': rk[0]['mean_best_val_path_mse'],
                             'dev_margin_to_runner_up': (rk[1]['mean_best_val_path_mse'] - rk[0]['mean_best_val_path_mse'])
                                                        / rk[0]['mean_best_val_path_mse'],
                             'dev_best_updates': json.dumps(rk[0]['best_updates'])})
    FR = pd.DataFrame([{'fold': r['fold'], 'model': r['model'], 'seed': r['seed'], 'structure': r['structure'], 'lr': r['lr'],
                        'widths': json.dumps(r['widths']), 'n_params': r['n_params'], 'best_update': r['best_update'],
                        'updates_run': r['updates_run'], 'wall_seconds': r['wall_seconds']} for r in finals])
    FR.to_csv(R / 'CV_FINAL_RUNS.csv', index=False); pd.DataFrame(sel_rows).to_csv(R / 'CV_SELECTION.csv', index=False)
    out = {'status': 'PRE_REGISTERED_ANALYSIS', 'delta_convention': 'MSE_NET - MSE_ASYM; positive favours ASYM',
           'checks': checks, 'decisions': S[['grouping', 'population', 'events', 'components', 'delta_event_equal',
                                              'components_positive', 'signflip_p_components', 'decision', 'delta_rmse_event_equal', 'signflip_p_components_rmse_secondary']].to_dict('records'),
           'best_update_zero_counts': FR.assign(z=FR.best_update == 0).groupby('model').z.sum().to_dict(),
           'separability_by_group_population': SEP.groupby(['type_group', 'population'])[['any_change', 'rise_and_fall', 'rise_only', 'fall_only']]
                                                  .mean().reset_index().to_dict('records')}
    if not a.skip_untrained:
        U, worst = untrained_reference(Path(a.root), a.threads, sel, finals)
        U.to_csv(R / 'CV_UNTRAINED_EVENT_SEED.csv', index=False)
        piv = U.pivot_table(index=['event', 'population', 'seed'], columns='model', values='untrained_path_mse')
        du = (piv['NET'] - piv['ASYM']).groupby(['event', 'population']).mean().rename('delta_untrained_seed_mean').reset_index()
        P2 = P.merge(du, on=['event', 'population'], how='left'); P2.to_csv(R / 'CV_EVENT_PAIRED.csv', index=False)
        out['checks']['untrained_rebuild_max_abs_diff_vs_logged_update0_val'] = worst
        out['untrained_event_equal_delta_by_population'] = P2.groupby('population').delta_untrained_seed_mean.mean().to_dict()
    (R / 'CV_SUMMARY.json').write_text(json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)) + '\n')
    figures(P, M, L)
    print(json.dumps(out['checks'], indent=1))
    print(S[['grouping', 'population', 'events', 'components', 'delta_event_equal', 'components_positive',
             'signflip_p_components', 'decision', 'delta_rmse_event_equal', 'signflip_p_components_rmse_secondary']].to_string(index=False))


if __name__ == '__main__':
    main()
