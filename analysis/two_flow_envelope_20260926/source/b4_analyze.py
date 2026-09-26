#!/usr/bin/env python3
"""Pre-registered B4' analysis (experiment plan, revision 1).

1. Exact S = ||Qm||^2/d and F = dist(Pm, A)^2/d for every gamma on the B4' design.
2. Realised component errors of the trained predictors against the exact conditional mean, per cell:
   e_N^P = E||P(g_N - m)||^2/d, e_N^Q = E||Q g_N - Qm||^2/d, e_A^P = E||P g_A - Pm||^2/d; Delta = risk_N - risk_A.
3. Envelopes e = alpha + beta * x calibrated on the gamma = 0 cells only (x = ideal-estimator variance of the component).
4. Envelope prediction for gamma > 0: Delta_pred = (e_N^P - e_A^P - F) + e_N^Q - S; classical ideal criterion x_Q - S.
5. Sign accuracy on resolved cells (|Delta| > paired 95% half-width, stratified by initialisation); kill checks K4-K6.
Positive Delta favours the two-flow (ASYM) predictor.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent; W = HERE.parent
sys.path.insert(0, str(HERE))
import dualflow as DF  # noqa: E402
from b4_data import OUT, GAMMAS, RHOS, NS, REPS, TRAIN_K, TEST_K, M, H, L, INIT_SEEDS  # noqa: E402

FITS = OUT.parent / 'b4_fits'
D = 8 * H
MIN_REPS = 10


def c_rho(rho):
    return rho + (1 - rho) / L


def projector_stack(z):
    Pz = DF.affine_projector(z)
    def P(a):   # a: (..., 8, H) -> projection over the 4 initial states within each forcing, per hour
        out = np.empty_like(a)
        for f in range(2):
            out[..., 4 * f:4 * f + 4, :] = np.einsum('ij,...jh->...ih', Pz, a[..., 4 * f:4 * f + 4, :])
        return out
    return Pz, P


def exact_features(design):
    z = TRAIN_K / M; Pz, P = projector_stack(z)
    rows = {}
    for gi, g in enumerate(GAMMAS):
        m = design[f'mean_{gi}']; cov = design[f'cov_{gi}']
        Pm = P(m); Qm = m - Pm
        S = float((Qm ** 2).sum()) / D
        F = 0.0
        for f in range(2):
            F += DF.fit(Pm[4 * f:4 * f + 4], z, n_random=6, seed=31 + f)['loss']
        diag = np.stack([np.diag(cov[c]) for c in range(8)])
        trS = float(diag.sum()); trPS = float(sum(Pz[i, i] * diag[4 * f + i].sum() for f in range(2) for i in range(4)))
        rows[gi] = {'gamma': float(g), 'S': S, 'F': F / D, 'trS0': trS, 'trPS0': trPS, 'trQS0': trS - trPS}
    return rows


def stratified_halfwidth(x, inits):
    x = np.asarray(x, float); inits = np.asarray(inits)
    groups = [x[inits == s] for s in np.unique(inits)]
    N = len(x); var = sum((len(g) / N) ** 2 * g.var(ddof=1) / len(g) for g in groups if len(g) > 1)
    dof = max(N - len(groups), 1)
    return float(stats.t.ppf(.975, dof) * np.sqrt(var))


def load_cells(design, rule):
    z = TRAIN_K / M; _, P = projector_stack(z)
    recs = []
    for gi in range(len(GAMMAS)):
        m = design[f'mean_{gi}']; Pm = P(m); Qm = m - Pm
        for ri, rho in enumerate(RHOS):
            for n in NS:
                dN, eNP, eNQ, eAP, qA, inits = [], [], [], [], [], []
                for rep in range(REPS):
                    fN = FITS / f'g{gi}_r{ri}_n{n}_rep{rep:02}_NET.npz'; fA = FITS / f'g{gi}_r{ri}_n{n}_rep{rep:02}_ASYM.npz'
                    if not (fN.exists() and fA.exists()):
                        continue
                    with np.load(fN) as a, np.load(fA) as b:
                        if rule == 'original_early_stop':
                            gN, gA = a['best_prediction'], b['best_prediction']
                        else:
                            gN, gA = a['snapshots'][-1, 0], b['snapshots'][-1, 0]
                    PgN = P(gN); PgA = P(gA)
                    eNP.append(((PgN - Pm) ** 2).sum() / D); eNQ.append((((gN - PgN) - Qm) ** 2).sum() / D)
                    eAP.append(((PgA - Pm) ** 2).sum() / D); qA.append(((gA - PgA) ** 2).sum() / D)
                    dN.append((((gN - m) ** 2).sum() - ((gA - m) ** 2).sum()) / D)
                    inits.append(INIT_SEEDS[rep % len(INIT_SEEDS)])
                if len(dN) < MIN_REPS:
                    continue
                recs.append({'gi': gi, 'gamma': float(GAMMAS[gi]), 'rho': float(rho), 'n': int(n), 'reps': len(dN),
                             'delta': float(np.mean(dN)), 'halfwidth': stratified_halfwidth(dN, inits),
                             'eNP': float(np.mean(eNP)), 'eNQ': float(np.mean(eNQ)), 'eAP': float(np.mean(eAP)),
                             'ASYM_max_Q_residual': float(np.max(qA))})
    return pd.DataFrame(recs)


def main():
    design = dict(np.load(OUT / 'design.npz'))
    feats = exact_features(design)
    out = {'status': 'PREREGISTERED_ANALYSIS (revision 1)', 'features': feats}
    lines = []
    for rule in ('original_early_stop', 'fixed_3000_ablation'):
        cells = load_cells(design, rule)
        if cells.empty:
            continue
        cells['S'] = cells.gi.map(lambda g: feats[g]['S']); cells['F'] = cells.gi.map(lambda g: feats[g]['F'])
        cells['xP'] = [c_rho(r.rho) * feats[r.gi]['trPS0'] / r.n / D for r in cells.itertuples()]
        cells['xQ'] = [c_rho(r.rho) * feats[r.gi]['trQS0'] / r.n / D for r in cells.itertuples()]
        cal = cells[cells.gamma == 0]
        fits = {}
        for k, x in (('eNP', 'xP'), ('eAP', 'xP'), ('eNQ', 'xQ')):
            X = np.c_[np.ones(len(cal)), cal[x].values]
            fits[k] = np.linalg.lstsq(X, cal[k].values, rcond=None)[0].tolist() if len(cal) >= 2 else [np.nan, np.nan]
        env = lambda k, x: fits[k][0] + fits[k][1] * x
        cells['envelope_pred'] = [env('eNP', r.xP) - env('eAP', r.xP) - r.F + env('eNQ', r.xQ) - r.S for r in cells.itertuples()]
        cells['ideal_pred'] = cells.xQ - cells.S
        cells['resolved'] = cells.delta.abs() > cells.halfwidth
        held = cells[(cells.gamma > 0) & cells.resolved]
        acc = lambda col: float(np.mean(np.sign(held[col]) == np.sign(held.delta))) if len(held) else None
        # K6: does NET's excluded-component error grow with S at fixed (rho, n)?
        k6 = []
        for (rho, n), g in cells.groupby(['rho', 'n']):
            if g.gamma.nunique() >= 3:
                sl = stats.linregress(g.S, g.eNQ); k6.append({'rho': rho, 'n': n, 'slope_eNQ_on_S': sl.slope, 'p': sl.pvalue})
        res = {'fits_alpha_beta_on_gamma0': fits, 'cells': cells.round(12).to_dict(orient='records'),
               'held_out_resolved': int(len(held)), 'held_out_total': int((cells.gamma > 0).sum()),
               'sign_accuracy_envelope': acc('envelope_pred'), 'sign_accuracy_ideal': acc('ideal_pred'),
               'K4_power_ok': bool(len(held) >= 12), 'K6_slopes': k6}
        out[rule] = res
        lines.append(f'== {rule}: cells={len(cells)} (min reps {cells.reps.min()}), held-out resolved {len(held)}/{int((cells.gamma > 0).sum())}')
        for k, v in fits.items():
            lines.append(f'   {k}: alpha={v[0]*1e6:+.3f}e-6  beta={v[1]:.3f}')
        lines.append(f'   sign accuracy on resolved held-out cells: envelope={res["sign_accuracy_envelope"]}  ideal={res["sign_accuracy_ideal"]}')
        show = cells[['gamma', 'rho', 'n', 'reps', 'S', 'F', 'delta', 'halfwidth', 'envelope_pred', 'ideal_pred']].copy()
        for col in ('S', 'F', 'delta', 'halfwidth', 'envelope_pred', 'ideal_pred'):
            show[col] = show[col] * 1e6
        lines.append(show.to_string(index=False, float_format=lambda v: f'{v:+.3f}'))
    (W / 'results/b4_analysis.json').write_text(json.dumps(out, indent=1, default=float) + '\n')
    print('\n'.join(lines))
    for gi, f in feats.items():
        print(f"gamma={f['gamma']:.2f}: S={f['S']:.3e}  F={f['F']:.3e}  F/S={(f['F']/f['S']) if f['S'] > 1e-20 else float('nan'):.3f}")


if __name__ == '__main__':
    main()
