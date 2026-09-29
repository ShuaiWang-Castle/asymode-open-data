#!/usr/bin/env python3
"""Analysis of the tropical fits (PLAN_TROPICAL.md; written before the fits finished).

Per test unit (county-event): MSE over observed forecast hours, averaged over the three seeds, for NET, ASYM,
ASYM_STATE, their untrained initialisations, persistence (y0 held) and the all-zero forecast.
  * pooled design-weighted and unweighted MSE; per system; cluster bootstrap over systems (2,000 draws)
  * P1: rd = (MSE_NET - MSE_ASYM)/(MSE_NET + MSE_ASYM) against the path level (observed window mean, ex post; mean of
        the two models' predicted paths, ex ante): Spearman, system-level Spearman, level bins
  * P2: pooled NET vs ASYM; low-level units
  * P3: per-system slope of the hourly relative recovery rate on the outage level in calm post-peak hours, with
        6-hour time-since-peak bins as fixed effects; against the system mean rd (descriptive, 15 systems)
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tropical_run as TR  # noqa: E402

FITS = HERE / 'results' / 'fits'
B = 2000


def unit_mse(pred, y, m):
    return (m * (pred - y) ** 2).sum(-1) / np.maximum(m.sum(-1), 1)


def main(panel, splits):
    d = TR.load(panel, splits); n = len(d['y'])
    recs = [json.loads(p.read_text()) for p in sorted(FITS.glob('*.json'))]
    expected = len(TR.FOLDS) * len(TR.KINDS) * len(TR.SEEDS)
    if len(recs) != expected or any(r['status'] != 'completed' for r in recs):
        raise SystemExit(f'incomplete: {len(recs)}/{expected} fits')
    mse = {k: np.full((len(TR.SEEDS), n), np.nan) for k in TR.KINDS}
    mse_init = {k: np.full((len(TR.SEEDS), n), np.nan) for k in TR.KINDS}
    path = {k: np.full((len(TR.SEEDS), n, TR.H), np.nan, dtype=np.float32) for k in TR.KINDS}
    for f in TR.FOLDS:
        for k in TR.KINDS:
            for si, s in enumerate(TR.SEEDS):
                with np.load(FITS / f'f{f}_{k}_s{s}.npz') as a:
                    ix, pr, p0 = a['test_index'], a['pred'], a['pred_init']
                if not np.isfinite(pr).all():
                    raise SystemExit(f'non-finite predictions in f{f}_{k}_s{s}')
                mse[k][si, ix] = unit_mse(pr, d['y'][ix], d['m'][ix]); mse_init[k][si, ix] = unit_mse(p0, d['y'][ix], d['m'][ix])
                path[k][si, ix] = pr
    for k in TR.KINDS:
        assert np.isfinite(mse[k]).all(), f'{k}: some units not covered'
    E = {k: mse[k].mean(0) for k in TR.KINDS}
    E.update({f'{k}_init': mse_init[k].mean(0) for k in TR.KINDS})
    E['persistence'] = unit_mse(np.repeat(d['y0'][:, None], TR.H, 1), d['y'], d['m'])
    E['zero'] = unit_mse(np.zeros_like(d['y']), d['y'], d['m'])
    w = d['w'].astype(float); sysv = d['system']; systems = np.unique(sysv)
    level = (d['m'] * d['y']).sum(1) / np.maximum(d['m'].sum(1), 1)
    level_hat = 0.5 * (path['NET'].mean(0).mean(1) + path['ASYM'].mean(0).mean(1))
    df = pd.DataFrame({'system': sysv, 'fips': d['fips'], 'fold': d['fold'], 'w': w, 'y0': d['y0'], 'level': level,
                       'level_hat': level_hat, **{f'mse_{k}': v for k, v in E.items()}})
    rng = np.random.default_rng(20260926)
    idx_by_sys = {s: np.flatnonzero(sysv == s) for s in systems}

    def wmean(col, ix, weights=True):
        ww = w[ix] if weights else np.ones(len(ix))
        return float((df[col].values[ix] * ww).sum() / ww.sum())

    def boot(fn):
        vals = []
        for _ in range(B):
            ix = np.concatenate([idx_by_sys[s] for s in rng.choice(systems, size=len(systems), replace=True)])
            vals.append(fn(ix))
        return [float(np.quantile(vals, .025)), float(np.quantile(vals, .975))]

    allix = np.arange(n); out = {'status': 'ANALYSIS_AS_PLANNED (PLAN_TROPICAL.md)', 'units': n, 'systems': len(systems)}
    models = TR.KINDS + [f'{k}_init' for k in TR.KINDS] + ['persistence', 'zero']
    out['pooled'] = {wt: {k: wmean(f'mse_{k}', allix, wt == 'weighted') for k in models} for wt in ('weighted', 'unweighted')}
    pairs = [('NET', 'ASYM'), ('ASYM_STATE', 'ASYM'), ('NET', 'ASYM_STATE'), ('NET', 'persistence'), ('ASYM', 'persistence'),
             ('ASYM_STATE', 'persistence')]
    out['differences'] = {}
    for a_, b_ in pairs:
        for wt in ('weighted', 'unweighted'):
            f = lambda ix, a_=a_, b_=b_, wt=wt: wmean(f'mse_{a_}', ix, wt == 'weighted') - wmean(f'mse_{b_}', ix, wt == 'weighted')
            out['differences'][f'{a_}-{b_} ({wt})'] = {'point': f(allix), 'system_bootstrap_95': boot(f)}
    per_sys = []
    for s in systems:
        ix = idx_by_sys[s]
        per_sys.append({'system': s, 'units': int(len(ix)), 'mean_level': float(level[ix].mean()), 'max_level': float(level[ix].max()),
                        **{k: wmean(f'mse_{k}', ix) for k in TR.KINDS + ['persistence', 'zero']},
                        'net_better_units': int((E['NET'][ix] < E['ASYM'][ix]).sum())})
    out['per_system'] = per_sys
    # P1: level against the relative difference
    tot = E['NET'] + E['ASYM']; ok = tot > 1e-12
    rd = np.where(ok, (E['NET'] - E['ASYM']) / np.where(ok, tot, 1), np.nan); df['rd'] = rd
    p1 = {}
    for name, lv in (('level_observed (ex post)', level), ('level_predicted (ex ante)', level_hat)):
        r, p = spearmanr(lv[ok], rd[ok])
        sys_rd = np.array([np.nanmean(rd[idx_by_sys[s]]) for s in systems]); sys_lv = np.array([lv[idx_by_sys[s]].mean() for s in systems])
        rs, ps = spearmanr(sys_lv, sys_rd)
        q = pd.qcut(pd.Series(lv[ok]).rank(method='first'), 10, labels=False)
        bins = []
        for b_ in range(10):
            sel = np.flatnonzero(ok)[q.values == b_]
            bins.append({'decile': b_ + 1, 'level_range': [float(lv[sel].min()), float(lv[sel].max())], 'units': int(len(sel)),
                         'share_NET_better': float((E['NET'][sel] < E['ASYM'][sel]).mean()),
                         'weighted_delta_NET_minus_ASYM': float(((E['NET'][sel] - E['ASYM'][sel]) * w[sel]).sum() / w[sel].sum()),
                         'median_rd': float(np.median(rd[sel]))})
        p1[name] = {'spearman_units': [float(r), float(p)], 'spearman_systems': [float(rs), float(ps)], 'deciles': bins}
    p1['units_with_both_mse_zero_excluded'] = int((~ok).sum())
    out['P1'] = p1
    # P3: recovery-rate slope on level in calm post-peak hours, per system
    z = np.load(panel); t = z['regime'] == 'tropical'
    gi = list(z['damage_features']).index('gust'); g_raw = z['xu'][t][:, TR.ORIGIN:, gi]
    p3 = []
    for s in systems:
        rows = []
        for u in idx_by_sys[s]:
            yy, mm = d['y'][u], d['m'][u]
            if yy.max() < .005:
                continue
            pk = int(np.argmax(yy)); calm = g_raw[u] <= np.median(g_raw[u])
            for k in range(pk, TR.H - 1):
                if mm[k] and mm[k + 1] and yy[k] >= .005 and calm[k + 1]:
                    rows.append((yy[k], (yy[k] - yy[k + 1]) / yy[k], (k - pk) // 6))
        if len(rows) < 30:
            p3.append({'system': s, 'hours': len(rows), 'slope': None}); continue
        r_ = pd.DataFrame(rows, columns=['y', 'rate', 'tsp'])
        X = pd.get_dummies(r_['tsp'], prefix='t', drop_first=False).astype(float); X['y'] = r_['y']
        coef = np.linalg.lstsq(X.values, r_['rate'].values, rcond=None)[0]
        p3.append({'system': s, 'hours': len(rows), 'slope': float(coef[-1]), 'mean_rd': float(np.nanmean(rd[idx_by_sys[s]]))})
    have = [x for x in p3 if x['slope'] is not None]
    rs = spearmanr([x['slope'] for x in have], [x['mean_rd'] for x in have]) if len(have) >= 4 else (np.nan, np.nan)
    out['P3'] = {'per_system': p3, 'spearman_slope_vs_mean_rd': [float(rs[0]), float(rs[1])],
                 'crowding_systems (slope<0)': sum(x['slope'] < 0 for x in have), 'mobilisation_systems (slope>0)': sum(x['slope'] > 0 for x in have)}
    df.to_csv(HERE / 'results' / 'tropical_units.csv', index=False)
    (HERE / 'results' / 'tropical_analysis.json').write_text(json.dumps(out, indent=1, default=float) + '\n')
    L = [f"units {n}, systems {len(systems)}"]
    for wt in ('weighted', 'unweighted'):
        L.append(f'pooled MSE x1e4 ({wt}): ' + ', '.join(f'{k} {v * 1e4:.3f}' for k, v in out['pooled'][wt].items()))
    for k, v in out['differences'].items():
        L.append(f"  {k}: {v['point'] * 1e4:+.4f} [{v['system_bootstrap_95'][0] * 1e4:+.4f}, {v['system_bootstrap_95'][1] * 1e4:+.4f}] x1e-4")
    L.append(pd.DataFrame(per_sys).assign(**{c: lambda x, c=c: x[c] * 1e4 for c in TR.KINDS + ['persistence', 'zero']})
             .to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    for name in ('level_observed (ex post)', 'level_predicted (ex ante)'):
        v = p1[name]
        L.append(f"P1 {name}: Spearman units {v['spearman_units'][0]:+.3f} (p {v['spearman_units'][1]:.2g}), systems {v['spearman_systems'][0]:+.3f} (p {v['spearman_systems'][1]:.2g})")
        L.append(pd.DataFrame(v['deciles']).to_string(index=False, float_format=lambda x: f'{x:.5g}'))
    L.append(f"P3: slopes {[(x['system'], None if x['slope'] is None else round(x['slope'], 3)) for x in p3]}; "
             f"Spearman(slope, mean rd) {out['P3']['spearman_slope_vs_mean_rd'][0]:+.3f} (p {out['P3']['spearman_slope_vs_mean_rd'][1]:.2g})")
    print('\n'.join(L))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
