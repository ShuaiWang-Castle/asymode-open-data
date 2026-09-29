#!/usr/bin/env python3
"""Analysis of tropical v2 (PLAN_TROPICAL_V2.md; written before the v2 fits finished).

Per unit and model: MSE over observed forecast hours against the spike-cleaned target (main) and the raw target
(sensitivity), as the mean over five seeds of per-seed MSE and as the MSE of the seed-averaged prediction.
Contrasts that isolate one factor each (R1 = rates do not read the state; gate structure; boundedness):
  ASYM - ASYM_STATE (R1), ASYM_STATE - NET_B (gate structure), NET_B - NET (bound), NET - ASYM (the paper's pair).
System is the inference unit: per-system paired effects, cluster bootstrap over systems, leave-one-system-out.
Q1: Spearman over systems between level and ASYM - ASYM_STATE (relative). Q2: NET_B vs NET in low-level systems.
Q3: Spearman over systems between level and the NET-vs-ASYM relative difference, with leave-one-system-out range.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tropical_run as TR  # noqa: E402
import tropical_v2 as V  # noqa: E402

FINAL = V.OUT / 'final'; B = 2000


def unit_mse(p, y, m):
    return (m * (p - y) ** 2).sum(-1) / np.maximum(m.sum(-1), 1)


def main(panel, splits):
    d = V.data(panel, splits); n = len(d['y']); w = d['w'].astype(float); sysv = d['system']; systems = np.unique(sysv)
    recs = [json.loads(p.read_text()) for p in FINAL.glob('*.json')]
    exp = len(TR.FOLDS) * len(V.KINDS) * len(V.FINAL_SEEDS)
    if len(recs) != exp:
        raise SystemExit(f'incomplete: {len(recs)}/{exp}')
    P = {k: np.full((len(V.FINAL_SEEDS), n, V.H), np.nan, np.float32) for k in V.KINDS}
    for r in recs:
        with np.load(FINAL / f"f{r['fold']}_{r['model']}_s{r['seed']}.npz") as a:
            P[r['model']][V.FINAL_SEEDS.index(r['seed']), a['test_index']] = a['pred']
    for k in V.KINDS:
        assert np.isfinite(P[k]).all(), k
    evalp = {k: (np.clip(P[k], 0, 1) if k == 'NET_B' else P[k]) for k in V.KINDS}   # NET_B is evaluated in range
    out = {'status': 'ANALYSIS_AS_PLANNED (PLAN_TROPICAL_V2.md)', 'cleaning': d['cleaning'],
           'selected': {f"f{r['fold']}_{r['model']}": {'lr': r['lr'], 'updates': r['updates']} for r in recs if r['seed'] == 0}}
    level = (d['m'] * d['y']).sum(1) / np.maximum(d['m'].sum(1), 1)
    idx = {s: np.flatnonzero(sysv == s) for s in systems}; rng = np.random.default_rng(20260927)
    wmean = lambda v, ix: float((v[ix] * w[ix]).sum() / w[ix].sum())
    for target in ('cleaned', 'raw'):
        y = d['y'] if target == 'cleaned' else d['y_raw']
        E = {}
        for k in V.KINDS:
            E[k] = np.mean([unit_mse(evalp[k][s], y, d['m']) for s in range(len(V.FINAL_SEEDS))], 0)
            E[f'{k}_ensemble'] = unit_mse(evalp[k].mean(0), y, d['m'])
        E['NET_clipped'] = np.mean([unit_mse(np.clip(P['NET'][s], 0, 1), y, d['m']) for s in range(len(V.FINAL_SEEDS))], 0)
        E['persistence'] = unit_mse(np.repeat(d['y0'][:, None], V.H, 1), y, d['m']); E['zero'] = unit_mse(np.zeros_like(y), y, d['m'])
        allix = np.arange(n)
        res = {'pooled_weighted_x1e4': {k: wmean(v, allix) * 1e4 for k, v in E.items()}}
        contrasts = {'NET-ASYM': ('NET', 'ASYM'), 'ASYM-ASYM_STATE (R1)': ('ASYM', 'ASYM_STATE'),
                     'ASYM_STATE-NET_B (gate structure)': ('ASYM_STATE', 'NET_B'), 'NET_B-NET (bound)': ('NET_B', 'NET'),
                     'NET-ASYM (ensembles)': ('NET_ensemble', 'ASYM_ensemble')}
        res['contrasts_x1e4'] = {}
        for name, (a_, b_) in contrasts.items():
            diff = E[a_] - E[b_]
            boot = [wmean(diff, np.concatenate([idx[s] for s in rng.choice(systems, len(systems))])) for _ in range(B)]
            loso = [wmean(diff, np.flatnonzero(sysv != s)) for s in systems]
            per = {s: wmean(diff, idx[s]) * 1e4 for s in systems}
            res['contrasts_x1e4'][name] = {'point': wmean(diff, allix) * 1e4, 'system_bootstrap_95': list(np.quantile(boot, [.025, .975]) * 1e4),
                                           'leave_one_system_out': [min(loso) * 1e4, max(loso) * 1e4], 'per_system': per,
                                           'systems_favouring_first': int(sum(v < 0 for v in per.values()))}
        sl = np.array([level[idx[s]].mean() for s in systems])
        def rel(a_, b_):
            t = E[a_] + E[b_]; ok = t > 1e-12
            return np.where(ok, (E[a_] - E[b_]) / np.where(ok, t, 1), np.nan)
        q = {}
        for name, (a_, b_) in (('Q3 NET vs ASYM', ('NET', 'ASYM')), ('Q1 ASYM vs ASYM_STATE', ('ASYM', 'ASYM_STATE'))):
            r_ = rel(a_, b_); sr = np.array([np.nanmean(r_[idx[s]]) for s in systems])
            rho = spearmanr(sl, sr)[0]; rl = [spearmanr(np.delete(sl, i), np.delete(sr, i))[0] for i in range(len(systems))]
            q[name] = {'spearman_systems': float(rho), 'leave_one_out_range': [float(min(rl)), float(max(rl))]}
        low = np.array([level[idx[s]].mean() < .002 for s in systems])
        lowix = np.concatenate([idx[s] for s, lo in zip(systems, low) if lo])
        q['Q2 low-level systems x1e4'] = {k: wmean(E[k], lowix) * 1e4 for k in ('NET', 'NET_clipped', 'NET_B', 'ASYM_STATE', 'ASYM', 'persistence')}
        res['questions'] = q
        out[target] = res
    (V.OUT / 'v2_analysis.json').write_text(json.dumps(out, indent=1, default=float) + '\n')
    c = out['cleaned']
    print('cleaning:', d['cleaning'])
    print('pooled weighted MSE x1e4 (cleaned target):', {k: round(v, 3) for k, v in c['pooled_weighted_x1e4'].items()})
    for name, v in c['contrasts_x1e4'].items():
        print(f"  {name}: {v['point']:+.3f} [{v['system_bootstrap_95'][0]:+.3f}, {v['system_bootstrap_95'][1]:+.3f}]; "
              f"LOSO {v['leave_one_system_out'][0]:+.3f}..{v['leave_one_system_out'][1]:+.3f}; systems favouring first {v['systems_favouring_first']}/15")
    for name, v in c['questions'].items():
        print(f'  {name}: {v}')
    print('raw target, pooled:', {k: round(v, 3) for k, v in out['raw']['pooled_weighted_x1e4'].items() if not k.endswith('ensemble')})


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
