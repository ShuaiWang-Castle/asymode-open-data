#!/usr/bin/env python3
"""X2 (exploratory, not pre-registered): do architecture-specific error envelopes explain the neural ranking?
For each architecture and component (P = shared affine part, Q = excluded part), regress the realized component error
(squared bias + variance over the 30 datasets) on the ideal estimator's variance in that component, x = c(rho) tr(.)/n:
    e = alpha + beta * x   (alpha: floor independent of data noise; beta: statistical slope after implicit regularisation).
Predicted Delta = (e_N^P - e_A^P) + (e_N^Q - S). Leave-one-cell-out sign accuracy versus the ideal-estimator criterion."""
import json
from pathlib import Path
import numpy as np, pandas as pd
W = Path(__file__).resolve().parent.parent
NB = W / 'inputs/neural_bridge'  # public layout; the as-run script read the same files from a private folder
s = pd.read_csv(NB / 'neural_summary.csv')
b1 = json.load(open(W / 'results/b1_exact_anatomy.json'))
out = {'status': 'EXPLORATORY_NOT_PREREGISTERED'}
for design in ('training_support',):
    for rule in ('original_early_stop', 'fixed_3000_ablation'):
        t = s[(s.design == design) & (s.rule == rule)].copy()
        recs = []
        for _, r in t.iterrows():
            g = b1[f'gamma_{r.gamma}']; cell = next(c for c in g['cells'] if c['rho'] == r.rho and c['n'] == r.n)
            S = g['S']; xP = cell['risk_affine'] - S; xQ = cell['risk_saturated'] - xP
            recs.append({'gamma': r.gamma, 'rho': r.rho, 'n': int(r.n), 'S': S, 'xP': xP, 'xQ': xQ,
                         'eNP': r.NET_common_bias_squared + r.NET_common_variance,
                         'eAP': r.ASYM_common_bias_squared + r.ASYM_common_variance,
                         'eNQ': r.NET_excluded_bias_squared + r.NET_excluded_variance,
                         'delta': r.delta, 'halfwidth': r.MC_halfwidth,
                         'ideal_delta': cell['delta_sat_minus_affine']})
        d = pd.DataFrame(recs)
        def fitlin(x, y):
            X = np.c_[np.ones_like(x), x]; coef, *_ = np.linalg.lstsq(X, y, rcond=None); return coef
        fits = {k: fitlin(d.xP.values if k != 'eNQ' else d.xQ.values, d[k].values).tolist() for k in ('eNP', 'eAP', 'eNQ')}
        # leave-one-cell-out prediction of the sign of delta
        pred = []
        for i in range(len(d)):
            m = np.arange(len(d)) != i
            c = {k: fitlin((d.xP if k != 'eNQ' else d.xQ).values[m], d[k].values[m]) for k in ('eNP', 'eAP', 'eNQ')}
            r = d.iloc[i]
            e = {k: c[k][0] + c[k][1] * (r.xP if k != 'eNQ' else r.xQ) for k in c}
            pred.append((e['eNP'] - e['eAP']) + (e['eNQ'] - r.S))
        d['envelope_pred_delta'] = pred
        resolved = d[np.abs(d.delta) > d.halfwidth]
        acc_env = float(np.mean(np.sign(resolved.envelope_pred_delta) == np.sign(resolved.delta))) if len(resolved) else None
        acc_ideal = float(np.mean(np.sign(-resolved.ideal_delta) == np.sign(resolved.delta))) if len(resolved) else None
        # ideal criterion: saturated (flexible) minus affine (restricted) risk; Delta_NN = risk_N - risk_A has the same orientation
        acc_ideal = float(np.mean(np.sign(resolved.ideal_delta) == np.sign(resolved.delta))) if len(resolved) else None
        out[f'{design}|{rule}'] = {'fits_alpha_beta': fits, 'n_cells': len(d), 'n_resolved_cells': int(len(resolved)),
                                   'loo_sign_accuracy_envelope_resolved': acc_env, 'sign_accuracy_ideal_resolved': acc_ideal,
                                   'cells': d.round(10).to_dict(orient='records')}
        print(f'== {design} | {rule}')
        for k, v in fits.items():
            print(f'   {k}: floor alpha = {v[0]*1e6:+.3f}e-6   slope beta = {v[1]:.3f}')
        print(f'   resolved cells (|delta| > MC half-width): {len(resolved)} of {len(d)}')
        print(f'   sign accuracy on resolved cells: envelope (LOO) = {acc_env}   ideal-estimator criterion = {acc_ideal}')
        print(d[['gamma', 'rho', 'n', 'delta', 'halfwidth', 'envelope_pred_delta', 'ideal_delta']].assign(
            delta=lambda x: x.delta * 1e6, halfwidth=lambda x: x.halfwidth * 1e6,
            envelope_pred_delta=lambda x: x.envelope_pred_delta * 1e6, ideal_delta=lambda x: x.ideal_delta * 1e6
        ).to_string(index=False, float_format=lambda v: f'{v:+.3f}'))
(W / 'results/x2_training_envelopes.json').write_text(json.dumps(out, indent=1, default=float) + '\n')
