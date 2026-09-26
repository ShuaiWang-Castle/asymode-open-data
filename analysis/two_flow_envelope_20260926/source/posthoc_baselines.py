#!/usr/bin/env python3
"""POST-HOC (not pre-registered), added after the independent review of 2026-09-26.

For every feedback cell of B4' (all 30, not only resolved ones) and both stopping rules:
  * winners and resolved counts; always-NET / always-ASYM sign baselines;
  * risk-difference predictors and their MAE: envelope, classical x_Q - S, -S, -(S+F), and the gamma=0 mean offset
    plus -S or -(S+F) (one constant from the no-feedback cells of the same rule);
  * per-cell e_N^Q / S (the Q-direction risk relative to the zero-Q baseline S), not a regression slope;
  * stratified bootstrap (replications resampled within initialisation strata, 2,000 draws) of the no-feedback
    intercepts alpha^P and their differences for NET, ASYM and ASYM_STATE, and of NET's Q-direction slope beta^Q.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent; W = HERE.parent
sys.path.insert(0, str(HERE))
from b4_data import OUT, GAMMAS, RHOS, NS, REPS, TRAIN_K, M, INIT_SEEDS  # noqa: E402
from b4_analyze import projector_stack, stratified_halfwidth, c_rho, D  # noqa: E402

FITS = OUT.parent / 'b4_fits'
KINDS = ('NET', 'ASYM', 'ASYM_STATE')
B = 2000


def per_rep(design, rule):
    _, P = projector_stack(TRAIN_K / M)
    out = {}
    for gi in range(len(GAMMAS)):
        m = design[f'mean_{gi}']; Pm = P(m); Qm = m - Pm
        for ri in range(len(RHOS)):
            for n in NS:
                arr = {k: np.full((REPS, 3), np.nan) for k in KINDS}   # risk, eP, eQ
                for rep in range(REPS):
                    for k in KINDS:
                        f = FITS / f'g{gi}_r{ri}_n{n}_rep{rep:02}_{k}.npz'
                        with np.load(f) as a:
                            g = a['best_prediction'] if rule == 'original_early_stop' else a['snapshots'][-1, 0]
                        Pg = P(g)
                        arr[k][rep] = [((g - m) ** 2).sum() / D, ((Pg - Pm) ** 2).sum() / D, (((g - Pg) - Qm) ** 2).sum() / D]
                out[(gi, ri, int(n))] = arr
    return out


def fit_line(x, y):
    X = np.c_[np.ones(len(x)), x]
    return np.linalg.lstsq(X, y, rcond=None)[0]


def main():
    design = dict(np.load(OUT / 'design.npz'))
    feats = {int(k): v for k, v in json.loads((W / 'results/b4_analysis.json').read_text())['features'].items()}
    inits = np.array([INIT_SEEDS[r % len(INIT_SEEDS)] for r in range(REPS)])
    strata = [np.flatnonzero(inits == s) for s in np.unique(inits)]
    rng = np.random.default_rng(20260926)
    report = {'status': 'POST_HOC_AFTER_INDEPENDENT_REVIEW'}
    lines = []
    for rule in ('original_early_stop', 'fixed_3000_ablation'):
        R = per_rep(design, rule)
        cells = []
        for (gi, ri, n), arr in R.items():
            f = feats[gi]; x_P = c_rho(RHOS[ri]) * f['trPS0'] / n / D; x_Q = c_rho(RHOS[ri]) * f['trQS0'] / n / D
            d = arr['NET'][:, 0] - arr['ASYM'][:, 0]
            cells.append({'gi': gi, 'gamma': float(GAMMAS[gi]), 'rho': float(RHOS[ri]), 'n': n, 'S': f['S'], 'F': f['F'], 'xP': x_P, 'xQ': x_Q,
                          'delta': d.mean(), 'hw': stratified_halfwidth(d, inits),
                          **{f'{q}_{k}': arr[k][:, j].mean() for k in KINDS for j, q in enumerate(('risk', 'eP', 'eQ'))}})
        c = pd.DataFrame(cells)
        cal = c[c.gamma == 0]
        env = {k: fit_line(cal.xP.values, cal[f'eP_{k}'].values) for k in ('NET', 'ASYM')}
        envQ = fit_line(cal.xQ.values, cal['eQ_NET'].values)
        off = cal.delta.mean()
        fb = c[c.gamma > 0].copy()
        preds = {
            'envelope': env['NET'][0] + env['NET'][1] * fb.xP - env['ASYM'][0] - env['ASYM'][1] * fb.xP - fb.F + envQ[0] + envQ[1] * fb.xQ - fb.S,
            'classical x_Q - S': fb.xQ - fb.S, '-S': -fb.S, '-(S+F)': -(fb.S + fb.F),
            'gamma0 offset - S': off - fb.S, 'gamma0 offset - (S+F)': off - (fb.S + fb.F)}
        res = fb[fb.delta.abs() > fb.hw]
        tab = []
        for name, p in preds.items():
            tab.append({'predictor': name, 'MAE_all30_1e-6': float((p - fb.delta).abs().mean() * 1e6),
                        'resolved_correct': int((np.sign(p[res.index]) == np.sign(res.delta)).sum()), 'resolved': int(len(res))})
        tab.append({'predictor': 'always NET (sign only)', 'MAE_all30_1e-6': None, 'resolved_correct': int((res.delta < 0).sum()), 'resolved': int(len(res))})
        tab.append({'predictor': 'always ASYM (sign only)', 'MAE_all30_1e-6': None, 'resolved_correct': int((res.delta > 0).sum()), 'resolved': int(len(res))})
        ratio = (fb.eQ_NET / fb.S)
        # stratified bootstrap of intercepts and NET's Q slope
        boots = []
        for _ in range(B):
            idx = np.concatenate([rng.choice(s, size=len(s), replace=True) for s in strata])
            calb = []
            for (gi, ri, n), arr in R.items():
                if gi != 0:
                    continue
                f = feats[gi]
                calb.append({'xP': c_rho(RHOS[ri]) * f['trPS0'] / n / D, 'xQ': c_rho(RHOS[ri]) * f['trQS0'] / n / D,
                             **{f'eP_{k}': arr[k][idx, 1].mean() for k in KINDS}, 'eQ_NET': arr['NET'][idx, 2].mean()})
            cb = pd.DataFrame(calb)
            a = {k: fit_line(cb.xP.values, cb[f'eP_{k}'].values)[0] for k in KINDS}
            boots.append({'aN': a['NET'], 'aA': a['ASYM'], 'aS': a['ASYM_STATE'], 'aN_minus_aA': a['NET'] - a['ASYM'],
                          'aS_minus_aA': a['ASYM_STATE'] - a['ASYM'], 'aS_minus_aN': a['ASYM_STATE'] - a['NET'],
                          'betaQ_NET': fit_line(cb.xQ.values, cb.eQ_NET.values)[1]})
        bt = pd.DataFrame(boots)
        ci = {k: [float(bt[k].quantile(.025)), float(bt[k].quantile(.975))] for k in bt.columns}
        point = {'aN': env['NET'][0], 'aA': env['ASYM'][0], 'betaQ_NET': envQ[1]}
        report[rule] = {'table': tab, 'winners_feedback_cells': {'NET': int((fb.delta < 0).sum()), 'ASYM': int((fb.delta > 0).sum())},
                        'resolved_winners': {'NET': int((res.delta < 0).sum()), 'ASYM': int((res.delta > 0).sum())},
                        'eQ_NET_over_S_range': [float(ratio.min()), float(ratio.max())], 'bootstrap_95': ci, 'point': point,
                        'cells': fb.assign(eQ_NET_over_S=ratio).round(12).to_dict(orient='records')}
        lines.append(f'== {rule}: feedback cells NET-better {int((fb.delta < 0).sum())}/30; resolved {len(res)} '
                     f'(NET {int((res.delta < 0).sum())}, ASYM {int((res.delta > 0).sum())})')
        lines.append(pd.DataFrame(tab).to_string(index=False, float_format=lambda v: f'{v:.3f}'))
        lines.append(f'   per-cell e_N^Q / S range: {ratio.min():.4f} .. {ratio.max():.4f}')
        for k in ('aN_minus_aA', 'aS_minus_aA', 'aS_minus_aN', 'betaQ_NET'):
            scale = 1 if k == 'betaQ_NET' else 1e6
            lines.append(f'   {k}: 95% bootstrap [{ci[k][0]*scale:+.3f}, {ci[k][1]*scale:+.3f}]' + ('' if k == 'betaQ_NET' else 'e-6'))
    (W / 'results/posthoc_baselines.json').write_text(json.dumps(report, indent=1, default=float) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
