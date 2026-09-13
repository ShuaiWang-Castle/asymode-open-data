#!/usr/bin/env python3
"""Build every result table and figure from saved predictions, truth and trial registries.

Written before any formal result existed. Squared errors are formed per seed first and
nothing averages predictions across seeds. Delta = MSE_NET - MSE_ASYM (positive favours
ASYM). Intervals are descriptive dependence sensitivity on retrospective data.
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from scoring import cluster_bootstrap, exact_signflip_p   # noqa: E402

KINDS = ('NET', 'ASYM')
US_SEEDS, SY_SEEDS, N_GRID = (8201, 8202, 8203, 8204, 8205), (7201, 7202, 7203, 7204, 7205), (32, 128, 512)
US_ENDS, SY_ENDS = {'1h': 0, '6h': 5, '24h': 23}, {'1h': 0, '6h': 5, '24h': 23, '32h': 31}
VARIANTS = ('raw', 'output_clip_01')


def jsonl(paths):
    return [json.loads(l) for p in paths for l in Path(p).read_text().splitlines() if l.strip()]


def clipv(p, v): return p if v == 'raw' else np.clip(p, 0.0, 1.0)


# ----------------------------------------------------------------------------- Task A
def us(G, A, out, figs, plt):
    spl = json.loads((A / 'locks/SPLITS.json').read_text())['splits']['evaluation']
    events, comps = spl['events'], spl['overlap_components']
    comp_of = {e: i for i, c in enumerate(comps) for e in c}
    T = {e: np.load(A / f'predictions/us_era5/truth/{e}.npz') for e in events}
    ev_rows, cty_rows, lead_rows = [], [], []
    for v in VARIANTS:
        for k in KINDS:
            for s in US_SEEDS:
                for e in events:
                    t, z = T[e], np.load(A / f'predictions/us_era5/{k}/seed{s}/{e}.npz')
                    if not (np.array_equal(z['row'], t['row']) and np.array_equal(z['t'], t['t'])):
                        raise SystemExit(f'US {k} seed{s} {e}: window keys differ from truth')
                    p = clipv(z['pred'].astype(np.float64), v); se = (p - t['truth']) ** 2
                    df = pd.DataFrame({'fips': t['fips'], 'path': se.mean(1), 'oob': ((p < 0) | (p > 1)).mean(1),
                                       **{kk: se[:, j] for kk, j in US_ENDS.items()}})
                    cty = df.groupby('fips').mean()
                    em = cty.mean()
                    ev_rows.append({'variant': v, 'model': k, 'seed': s, 'event': e, 'component': comp_of[e],
                                    'path_mse': em.path, **{f'mse_{kk}': em[kk] for kk in US_ENDS}, 'oob_fraction': em.oob,
                                    'n_windows': len(df), 'n_counties': len(cty)})
                    for f, r in cty.iterrows():
                        cty_rows.append({'variant': v, 'model': k, 'seed': s, 'event': e, 'fips': f, 'path_mse': r.path,
                                         'n_origins': int((t['fips'] == f).sum())})
                    lead = pd.DataFrame(se).groupby(t['fips']).mean().mean(0).to_numpy()
                    lead_rows += [{'variant': v, 'model': k, 'seed': s, 'event': e, 'lead_h': j + 1, 'mse': float(lead[j])} for j in range(24)]
    E, C, Ld = pd.DataFrame(ev_rows), pd.DataFrame(cty_rows), pd.DataFrame(lead_rows)
    E.to_csv(out / 'US_EVENT_SEED.csv', index=False); C.to_csv(out / 'US_COUNTY_SEED.csv', index=False)
    Ld.to_csv(out / 'US_LEAD_CURVE_EVENT_SEED.csv', index=False)
    mcols = ['path_mse', 'mse_1h', 'mse_6h', 'mse_24h', 'oob_fraction']
    O = E.groupby(['variant', 'model', 'seed'])[mcols].mean().reset_index()   # event-equal overall per seed
    O.to_csv(out / 'US_OVERALL_SEED.csv', index=False)
    summ = {'delta_convention': 'MSE_NET - MSE_ASYM', 'evaluation_events': events, 'overlap_components': comps, 'variants': {}}
    for v in VARIANTS:
        e = E[E.variant == v]
        piv = {m: e.pivot_table(index='seed', columns='event', values=m) for m in ('path_mse', 'mse_1h', 'mse_6h', 'mse_24h')}
        net = {m: piv[m].loc[:, :] for m in piv}
        dse = {m: (e[e.model == 'NET'].pivot_table(index='seed', columns='event', values=m)
                   - e[e.model == 'ASYM'].pivot_table(index='seed', columns='event', values=m)) for m in piv}
        cb = cluster_bootstrap(dse['path_mse'], comps)
        o = O[O.variant == v]
        by_seed = (o[o.model == 'NET'].set_index('seed')[mcols] - o[o.model == 'ASYM'].set_index('seed')[mcols])
        nm, am = o[o.model == 'NET'][mcols].mean(), o[o.model == 'ASYM'][mcols].mean()
        ev_net = e[e.model == 'NET'].groupby('event').path_mse.mean(); ev_asym = e[e.model == 'ASYM'].groupby('event').path_mse.mean()
        summ['variants'][v] = {
            'NET_seed_mean': nm.to_dict(), 'ASYM_seed_mean': am.to_dict(),
            'delta_seed_mean': (nm - am).to_dict(), 'delta_by_seed_path': by_seed.path_mse.to_dict(),
            'seeds_with_positive_delta_path': int((by_seed.path_mse > 0).sum()),
            'events_with_positive_seed_mean_delta_path': int((dse['path_mse'].mean(0) > 0).sum()), 'n_events': len(events),
            'mean_event_relative_gain_asym_vs_net': float(((ev_net - ev_asym) / ev_net).mean()),
            'ratio_of_event_equal_means_gain': float((ev_net.mean() - ev_asym.mean()) / ev_net.mean()),
            'component_bootstrap_path': cb}
    pe = []
    for v in VARIANTS:
        e = E[E.variant == v]
        g = e.groupby(['event', 'model'])[mcols + ['n_windows', 'n_counties']].mean().unstack('model')
        d = (e[e.model == 'NET'].set_index(['seed', 'event']).path_mse - e[e.model == 'ASYM'].set_index(['seed', 'event']).path_mse)
        for ev in events:
            pe.append({'variant': v, 'event': ev, 'component': comp_of[ev], 'n_windows': int(g.loc[ev, ('n_windows', 'NET')]),
                       'n_counties': int(g.loc[ev, ('n_counties', 'NET')]),
                       **{f'NET_{m}': g.loc[ev, (m, 'NET')] for m in mcols}, **{f'ASYM_{m}': g.loc[ev, (m, 'ASYM')] for m in mcols},
                       'delta_path_seed_mean': float(d.xs(ev, level='event').mean()),
                       'delta_path_seed_sd': float(d.xs(ev, level='event').std(ddof=1)),
                       'seeds_delta_positive': int((d.xs(ev, level='event') > 0).sum())})
    PE = pd.DataFrame(pe); PE.to_csv(out / 'US_EVENT_PAIRED.csv', index=False)
    cc = C[C.variant == 'raw']
    cd = (cc[cc.model == 'NET'].groupby(['event', 'fips']).path_mse.mean() - cc[cc.model == 'ASYM'].groupby(['event', 'fips']).path_mse.mean()).rename('delta_path_seed_mean').reset_index()
    cd = cd.merge(cc[cc.model == 'NET'].groupby(['event', 'fips']).n_origins.first().reset_index(), on=['event', 'fips'])
    cd.to_csv(out / 'US_COUNTY_PAIRED.csv', index=False)
    adverse = {'net_better_largest': cd.nsmallest(10, 'delta_path_seed_mean').to_dict('records'),
               'asym_better_largest': cd.nlargest(10, 'delta_path_seed_mean').to_dict('records'),
               'counties_delta_positive': int((cd.delta_path_seed_mean > 0).sum()), 'counties': int(len(cd))}
    summ['county_level'] = adverse
    reg = jsonl(sorted((A / 'logs').glob('us_final_w*.jsonl')))
    ind = []
    for k in KINDS:
        r = next(x for x in reg if x['model'] == k and x['seed'] == US_SEEDS[0])
        mine = float(O[(O.variant == 'raw') & (O.model == k) & (O.seed == US_SEEDS[0])].path_mse.iloc[0])
        ind.append({'model': k, 'seed': US_SEEDS[0], 'groupby_scorer': mine, 'training_driver_weight_vector_scorer': r['evaluation']['path_mse'],
                    'abs_diff': abs(mine - r['evaluation']['path_mse'])})
    summ['independent_scorer_check'] = ind
    summ['conditional_cells'] = us_conditional(G, A, out, E)
    (out / 'US_SUMMARY.json').write_text(json.dumps(summ, indent=1, default=float))
    if plt:
        lc = Ld[Ld.variant == 'raw'].groupby(['model', 'seed', 'lead_h']).mse.mean().reset_index()
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for k, col in zip(KINDS, ('#1f77b4', '#d62728')):
            q = lc[lc.model == k]
            for s in US_SEEDS:
                ax[0].plot(q[q.seed == s].lead_h, q[q.seed == s].mse, color=col, alpha=.25, lw=.8)
            ax[0].plot(q.groupby('lead_h').mse.mean().index, q.groupby('lead_h').mse.mean(), color=col, lw=2, label=k)
        ax[0].set(xlabel='lead (h)', ylabel='event-equal MSE', title='US 2024: path error by lead (seeds thin)'); ax[0].legend()
        pr = PE[PE.variant == 'raw']
        d = (E[(E.variant == 'raw') & (E.model == 'NET')].set_index(['seed', 'event']).path_mse
             - E[(E.variant == 'raw') & (E.model == 'ASYM')].set_index(['seed', 'event']).path_mse).reset_index()
        for i, ev in enumerate(events):
            ax[1].scatter([i] * len(US_SEEDS), d[d.event == ev].path_mse, color='k', s=12, alpha=.6)
        ax[1].bar(range(len(events)), pr.delta_path_seed_mean, color=['#d62728' if x > 0 else '#1f77b4' for x in pr.delta_path_seed_mean], alpha=.5)
        ax[1].axhline(0, color='k', lw=.8); ax[1].set_xticks(range(len(events)), events, rotation=40, ha='right', fontsize=8)
        ax[1].set(ylabel='Delta = MSE_NET - MSE_ASYM', title='US 2024: per-event paired Delta (dots = seeds)')
        fig.tight_layout(); fig.savefig(figs / 'us_era5_lead_curve_and_event_delta.png', dpi=140); plt.close(fig)
    return summ


def us_conditional(G, A, out, E):
    """Train-defined cells on inputs only; evaluation descriptives. Not a Lambda estimate."""
    import us_data as UD
    c = UD.load_corpus(G)
    idx = pd.read_csv(A / 'locks/US_WINDOW_INDEX.csv.gz', dtype={'fips': str})
    gi = UD.RAW_CHANNELS.index('gust')
    def feats(d):
        r, t = d.row.to_numpy(), d.t.to_numpy()
        y0 = c.Y[r, t]; gmax = c.X[r[:, None], t[:, None] + np.arange(1, 25)[None, :], gi].max(1)
        return y0, gmax
    tr = idx[idx.split == 'train']; ytr, gtr = feats(tr)
    th = {'y0_q50': float(np.quantile(ytr, .5)), 'y0_q90': float(np.quantile(ytr, .9)),
          'gust_future_max_q33': float(np.quantile(gtr, 1 / 3)), 'gust_future_max_q67': float(np.quantile(gtr, 2 / 3))}
    ev = idx[idx.split == 'evaluation'].sort_values(['event', 'fips', 't'], kind='mergesort').reset_index(drop=True)
    y0, gm = feats(ev)
    yb = np.digitize(y0, [th['y0_q50'], th['y0_q90']], right=True); gb = np.digitize(gm, [th['gust_future_max_q33'], th['gust_future_max_q67']], right=True)
    rows = []
    spl = json.loads((A / 'locks/SPLITS.json').read_text())['splits']['evaluation']
    comp_of = {e: i for i, cc in enumerate(spl['overlap_components']) for e in cc}
    for yi in range(3):
        for gj in range(3):
            m = (yb == yi) & (gb == gj); sub = ev[m]
            rec = {'y0_bin': yi, 'gust_bin': gj, 'n_windows': int(m.sum()), 'n_county_events': int(sub.groupby(['event', 'fips']).ngroups) if m.any() else 0,
                   'n_events': int(sub.event.nunique()), 'n_components': int(len({comp_of[e] for e in sub.event.unique()}))}
            if m.sum() < 200 or rec['n_events'] < 2:
                rec.update({'status': 'NA_INSUFFICIENT_SUPPORT'}); rows.append(rec); continue
            truths, preds = [], {k: {s: [] for s in US_SEEDS} for k in KINDS}
            for e, g in sub.groupby('event'):
                t = np.load(A / f'predictions/us_era5/truth/{e}.npz')
                key = pd.Series(np.arange(len(t['row'])), index=pd.MultiIndex.from_arrays([t['row'], t['t']]))
                ix = key.loc[list(zip(g.row, g.t))].to_numpy()
                truths.append(t['truth'][ix])
                for k in KINDS:
                    for s in US_SEEDS:
                        preds[k][s].append(np.load(A / f'predictions/us_era5/{k}/seed{s}/{e}.npz')['pred'][ix].astype(np.float64))
            Tt = np.concatenate(truths)
            risk = {k: float(np.mean([((np.concatenate(preds[k][s]) - Tt) ** 2).mean() for s in US_SEEDS])) for k in KINDS}
            dseed = [((np.concatenate(preds['NET'][s]) - Tt) ** 2).mean() - ((np.concatenate(preds['ASYM'][s]) - Tt) ** 2).mean() for s in US_SEEDS]
            rec.update({'status': 'DESCRIPTIVE', 'within_cell_target_sd': float(np.sqrt(Tt.var(axis=0).mean())),
                        'within_cell_y0_sd': float(y0[m].std()), 'within_cell_gust_future_max_sd': float(gm[m].std()),
                        'NET_window_mean_path_mse': risk['NET'], 'ASYM_window_mean_path_mse': risk['ASYM'],
                        'delta_window_mean': risk['NET'] - risk['ASYM'], 'seeds_delta_positive': int(sum(x > 0 for x in dseed))})
            rows.append(rec)
    pd.DataFrame(rows).to_csv(out / 'US_CONDITIONAL_CELLS.csv', index=False)
    return {'thresholds_from_train': th, 'rule': 'y0 bins by train q50/q90; given future 24h max gust bins by train terciles; inputs only',
            'weighting_note': 'cell risks are window means within the cell, not the event-equal main estimand',
            'cells': rows}


# ----------------------------------------------------------------------------- Task C
def synthetic(A, out, figs, plt):
    from synth_generator import SCENE_HOURS, H, exact_law, scene_x
    cal = json.loads((A / 'results/synthetic/EXACT_LAW_CALIBRATION.json').read_text())
    laws = cal['calibration']['laws']
    reg = [r for r in jsonl(sorted((A / 'logs').glob('synth_final_w*.jsonl'))) if r['status'] == 'COMPLETED']
    rows = []
    for r in reg:
        ev = r['evaluation']
        rows.append({'law': r['law'], 'gamma': r['gamma'], 'n_per_scene': r['n_per_scene'], 'seed': r['seed'], 'model': r['model'],
                     'config_id': r['config_id'], 'updates': r['updates'], 'width': r['width'], 'n_params': r['n_params'],
                     'wall_seconds': r['wall_seconds'], 'test_path_mse': ev['test']['path_mse'],
                     **{f'test_path_mse_scene{h}h': x for h, x in zip(SCENE_HOURS, ev['test']['path_mse_by_scene'])},
                     **{f'test_mse_{k}': ev['test'][f'mse_{k}'] for k in SY_ENDS},
                     'vs_mu0_path_mse': ev['vs_exact_mean']['path_mse'],
                     **{f'vs_mu0_path_mse_scene{h}h': x for h, x in zip(SCENE_HOURS, ev['vs_exact_mean']['path_mse_by_scene'])},
                     **{f'vs_mu0_mse_{k}': ev['vs_exact_mean'][f'mse_{k}'] for k in SY_ENDS},
                     'calibration_path_mse': ev['calibration']['path_mse'], 'oob_fraction': ev['oob_fraction']})
    D = pd.DataFrame(rows).sort_values(['gamma', 'law', 'n_per_scene', 'seed', 'model'])
    D.to_csv(out / 'SYNTH_FINAL_SEED_MODEL.csv', index=False)
    key = ['gamma', 'law', 'n_per_scene', 'seed']
    num = [c for c in D.columns if c.startswith(('test_', 'vs_mu0_', 'calibration_'))]
    P = (D[D.model == 'NET'].set_index(key)[num] - D[D.model == 'ASYM'].set_index(key)[num]).add_prefix('delta_').reset_index()
    P.to_csv(out / 'SYNTH_PAIRED_DELTA_SEED.csv', index=False)
    trace = {}
    for g in (0.04, 0.0):
        for h in SCENE_HOURS:
            trace[(g, h)] = float(np.trace(exact_law(scene_x(h), g)['Sigma']) / H)
    summ = []
    for (g, law, n), q in P.groupby(['gamma', 'law', 'n_per_scene']):
        c = laws[law]['c']; noise = c * np.mean([trace[(g, h)] for h in SCENE_HOURS])
        dm = D[(D.gamma == g) & (D.law == law) & (D.n_per_scene == n)]
        summ.append({'gamma': g, 'law': law, 'c': c, 'rho': laws[law]['rho'], 'n_per_scene': n,
                     'actual_path_sd_mean_over_scenes': float(np.sqrt(noise)),
                     'noise_variance_per_training_event_over_n': noise / n,
                     'NET_test_path_mse': float(dm[dm.model == 'NET'].test_path_mse.mean()), 'ASYM_test_path_mse': float(dm[dm.model == 'ASYM'].test_path_mse.mean()),
                     'delta_test_mean': float(q.delta_test_path_mse.mean()), 'delta_test_sd_over_seeds': float(q.delta_test_path_mse.std(ddof=1)),
                     'seeds_delta_test_positive': int((q.delta_test_path_mse > 0).sum()),
                     'NET_vs_mu0_path_mse': float(dm[dm.model == 'NET'].vs_mu0_path_mse.mean()), 'ASYM_vs_mu0_path_mse': float(dm[dm.model == 'ASYM'].vs_mu0_path_mse.mean()),
                     'delta_vs_mu0_mean': float(q.delta_vs_mu0_path_mse.mean()), 'delta_vs_mu0_sd_over_seeds': float(q.delta_vs_mu0_path_mse.std(ddof=1)),
                     'seeds_delta_vs_mu0_positive': int((q.delta_vs_mu0_path_mse > 0).sum()),
                     'exact_signflip_p_vs_mu0_over_seeds': exact_signflip_p(q.delta_vs_mu0_path_mse.to_numpy()),
                     **{f'delta_vs_mu0_scene{h}h_mean': float(q[f'delta_vs_mu0_path_mse_scene{h}h'].mean()) for h in SCENE_HOURS}})
    S = pd.DataFrame(summ); S.to_csv(out / 'SYNTH_CELL_SUMMARY.csv', index=False)
    sel = []
    for (g, law, n, seed), _ in P.groupby(key):
        tag = f'law{law}_g{g:g}_n{n}_seed{seed}'
        zc = {k: np.load(A / f'predictions/synthetic/{k}/{tag}.npz') for k in KINDS}
        dcal = (zc['NET']['calibration_event_loss'].astype(np.float64) - zc['ASYM']['calibration_event_loss'].astype(np.float64)).ravel()
        choose = 'ASYM' if dcal.mean() > 0 else 'NET'
        tt = {k: float(D[(D.gamma == g) & (D.law == law) & (D.n_per_scene == n) & (D.seed == seed) & (D.model == k)].test_path_mse.iloc[0]) for k in KINDS}
        se = dcal.std(ddof=1) / math.sqrt(len(dcal))
        sel.append({'gamma': g, 'law': law, 'n_per_scene': n, 'seed': seed, 'calibration_delta_mean': float(dcal.mean()),
                    'calibration_delta_se': float(se), 'near_tie_abs_delta_below_2se': bool(abs(dcal.mean()) < 2 * se),
                    'selected': choose, 'test_selected': tt[choose], 'test_always_NET': tt['NET'], 'test_always_ASYM': tt['ASYM'],
                    'test_better_of_two': min(tt.values()), 'selected_equals_test_better': tt[choose] == min(tt.values())})
    SEL = pd.DataFrame(sel); SEL.to_csv(out / 'SYNTH_CALIBRATION_SELECTION.csv', index=False)
    SELs = SEL.groupby(['gamma', 'law', 'n_per_scene']).agg(
        times_ASYM_selected=('selected', lambda s: int((s == 'ASYM').sum())), near_ties=('near_tie_abs_delta_below_2se', 'sum'),
        test_selected=('test_selected', 'mean'), test_always_NET=('test_always_NET', 'mean'), test_always_ASYM=('test_always_ASYM', 'mean'),
        selection_matched_test_better=('selected_equals_test_better', 'sum')).reset_index()
    SELs.to_csv(out / 'SYNTH_CALIBRATION_SELECTION_SUMMARY.csv', index=False)
    mu = {h: exact_law(scene_x(h))['mu'] for h in SCENE_HOURS}; rep = {}
    for h, m in mu.items():
        prev = np.r_[0.0, m[:-1]]; d = m - prev
        u = np.where(d > 0, d / (1 - prev), 0.0); r = np.where(d < 0, -d / np.where(prev > 0, prev, 1.0), 0.0)
        y, path = 0.0, []
        for k in range(H):
            y = y + u[k] * (1 - y) - r[k] * y; path.append(y)
        rep[f'{h}h'] = float(np.abs(np.array(path) - m).max())
    main = S[S.gamma > 0]
    theory = {'exact_representation_of_mu0_by_asym_rate_schedule_max_abs_error': rep,
              'net_can_represent_mu0': 'yes: an increment that depends only on the step index reproduces any path',
              'reading': 'both model classes contain the exact conditional mean, so risk differences are estimation effects of the two recursions under noise',
              'spearman_noise_over_n_vs_delta_vs_mu0_main_grid': float(pd.Series(main.noise_variance_per_training_event_over_n).rank().corr(pd.Series(main.delta_vs_mu0_mean).rank())),
              'note': 'descriptive association across 9 cells; not a threshold the networks must satisfy'}
    curves = []
    for r in reg:
        for c in r['curve']:
            curves.append({'gamma': r['gamma'], 'law': r['law'], 'n_per_scene': r['n_per_scene'], 'seed': r['seed'], 'model': r['model'],
                           'update': c['update'], 'val_path_mse': c['val_path_mse'], 'train_loss_ema': c.get('train_loss_ema')})
    pd.DataFrame(curves).to_csv(out / 'SYNTH_LEARNING_CURVES.csv', index=False)
    json.dump({'theory_diagnostics': theory, 'laws': laws, 'scene_law_diagnostics': cal['scenes']},
              open(out / 'SYNTH_SUMMARY.json', 'w'), indent=1, default=float)
    if plt:
        for metric, lab in (('test_path_mse', 'new-event path MSE'), ('vs_mu0_path_mse', 'MSE vs exact conditional mean')):
            fig, ax = plt.subplots(2, 3, figsize=(13, 7), sharey='row')
            md = D[D.gamma > 0]; top = [md[metric].min() * .95, md[metric].max() * 1.05]
            for j, law in enumerate(range(3)):
                q = md[md.law == law]
                for k, col in zip(KINDS, ('#1f77b4', '#d62728')):
                    qq = q[q.model == k]
                    ax[0, j].scatter(np.log2(qq.n_per_scene) + (.08 if k == 'ASYM' else -.08), qq[metric], color=col, s=10, alpha=.5)
                    ax[0, j].plot(np.log2(N_GRID), qq.groupby('n_per_scene')[metric].mean().to_numpy(), color=col, lw=2, label=k)
                ax[0, j].set(title=f"law {law}: 8h SD target {laws[law]['target_sd_8h_gamma004']}", xticks=np.log2(N_GRID), xticklabels=N_GRID, ylim=top)
                dcol = f'delta_{metric}'; pq = P[(P.gamma > 0) & (P.law == law)]
                ax[1, j].scatter(np.log2(pq.n_per_scene), pq[dcol], color='k', s=12, alpha=.6)
                ax[1, j].plot(np.log2(N_GRID), pq.groupby('n_per_scene')[dcol].mean().to_numpy(), color='k', lw=1.5)
                ax[1, j].axhline(0, color='grey', lw=.8); ax[1, j].set(xticks=np.log2(N_GRID), xticklabels=N_GRID, xlabel='training events per scene')
            ax[0, 0].set_ylabel(lab + ' (shared scale)'); ax[1, 0].set_ylabel('Delta = NET - ASYM (signed)'); ax[0, 0].legend()
            fig.tight_layout(); fig.savefig(figs / f'synthetic_{metric}_by_law_and_n.png', dpi=140); plt.close(fig)
    return {'cells': S.to_dict('records'), 'selection_summary': SELs.to_dict('records'), 'theory': theory}


# ----------------------------------------------------------------------------- Task B figure and cost/registry
def aneel_fig(A, out, figs, plt):
    pc = pd.read_csv(A / 'results/aneel/ANEEL_COMPANY_PAIRED_DELTA.csv')
    if plt:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
        for i, v in enumerate(VARIANTS):
            q = pc[pc.variant == v]; comps = sorted(q.company.unique())
            m = q.groupby('company').delta_mse_path.mean().reindex(comps)
            ax[i].bar(range(len(comps)), m, color=['#d62728' if x > 0 else '#1f77b4' for x in m], alpha=.5)
            for j, cpy in enumerate(comps):
                ax[i].scatter([j] * q[q.company == cpy].shape[0], q[q.company == cpy].delta_mse_path, color='k', s=8, alpha=.6)
            ax[i].axhline(0, color='k', lw=.8); ax[i].set(title=f'ANEEL 2019 ({v}): per-company Delta, dots = seeds', xlabel='company', ylabel='MSE_NET - MSE_ASYM')
            ax[i].set_xticks(range(len(comps)), comps, fontsize=7)
        fig.tight_layout(); fig.savefig(figs / 'aneel_company_delta.png', dpi=140); plt.close(fig)


def cost_and_registry(A, out):
    recs = []
    for pat, task, phase in (('us_dev_w*.jsonl', 'us_era5', 'dev'), ('us_final_w*.jsonl', 'us_era5', 'final'),
                             ('synth_dev_w*.jsonl', 'synthetic', 'dev'), ('synth_final_w*.jsonl', 'synthetic', 'final')):
        recs += jsonl(sorted((A / 'logs').glob(pat)))
    ver = json.loads((A / 'results/aneel/ANEEL_CHECKPOINT_VERIFICATION.json').read_text())
    for r in ver.get('retrain_spot_check', []):
        recs.append({'task': 'aneel', 'phase': 'spot_retrain', 'status': 'COMPLETED', 'exit_code': 0, **r})
    with open(A / 'logs/TRIAL_REGISTRY.jsonl', 'w') as f:
        for r in recs:
            f.write(json.dumps(r, default=float) + '\n')
    rows = []
    for r in recs:
        if r.get('phase') in ('final',):
            rows.append({'task': r['task'], 'model': r['model'], 'config_id': r['config_id'], 'width': r['width'], 'n_params': r['n_params'],
                         'updates': r['updates'], 'train_wall_seconds': r['wall_seconds'], 'threads': r.get('threads'),
                         'inference_seconds': r.get('inference_seconds_evaluation')})
    cost = pd.DataFrame(rows).groupby(['task', 'model', 'config_id', 'width', 'n_params', 'updates', 'threads'], dropna=False).agg(
        runs=('train_wall_seconds', 'size'), train_wall_seconds_mean=('train_wall_seconds', 'mean'),
        inference_seconds_mean=('inference_seconds', 'mean')).reset_index()
    for c in ver.get('checkpoints', []):
        pass
    an = pd.DataFrame(ver.get('checkpoints', []))
    if len(an):
        cost = pd.concat([cost, an.groupby('model').agg(width=('width', 'first'), n_params=('n_params', 'first'),
                                                         runs=('seed', 'size'), inference_seconds_mean=('inference_seconds', 'mean')).reset_index().assign(task='aneel', config_id='repair lock')], ignore_index=True)
    cost.to_csv(out / 'COST_TABLE.csv', index=False)
    return len(recs)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', required=True); ap.add_argument('--work', required=True)
    ap.add_argument('--tasks', default='us,synthetic,aneel')
    a = ap.parse_args(); G, A = Path(a.root), Path(a.work)
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    except Exception:
        plt = None
    figs = A / 'figures'; figs.mkdir(exist_ok=True)
    tasks = a.tasks.split(',')
    if 'us' in tasks:
        o = A / 'results/us_era5'; o.mkdir(parents=True, exist_ok=True); us(G, A, o, figs, plt); print('US tables written')
    if 'synthetic' in tasks:
        o = A / 'results/synthetic'; o.mkdir(parents=True, exist_ok=True); synthetic(A, o, figs, plt); print('synthetic tables written')
    if 'aneel' in tasks:
        aneel_fig(A, A / 'results/aneel', figs, plt); print('ANEEL figure written')
    if set(tasks) >= {'us', 'synthetic', 'aneel'}:
        n = cost_and_registry(A, A / 'results'); print(f'TRIAL_REGISTRY.jsonl: {n} records; COST_TABLE.csv written')


if __name__ == '__main__':
    main()
