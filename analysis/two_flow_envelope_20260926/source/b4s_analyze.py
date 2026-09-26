#!/usr/bin/env python3
"""Pre-registered phase-2 analysis (experiment plan, revision 2): does state access or the two-flow parameterisation set
the shared-component training floor?

For ASYM (state-independent rates), ASYM_STATE (state-reading rates) and NET, per cell: shared (P) and excluded (Q)
component errors against the exact conditional mean. Envelopes e^P = alpha + beta * x_P are calibrated on gamma = 0.
Decision 1: alpha_S^P <= (alpha_A^P + alpha_N^P)/2 -> the floor advantage mainly comes from the two-flow
parameterisation; otherwise from restricting state access. Decision 2: at gamma >= .08, rho = 0, ASYM_STATE's
excluded-component error <= 25% of S -> reading the state lets the two-flow model use the excluded signal.
Pairwise paired differences use stratified 95% half-widths over the 60 replications.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent; W = HERE.parent
sys.path.insert(0, str(HERE))
from b4_data import OUT, GAMMAS, RHOS, NS, REPS, TRAIN_K, M, H, L, INIT_SEEDS  # noqa: E402
from b4_analyze import projector_stack, stratified_halfwidth, c_rho, D  # noqa: E402

FITS = OUT.parent / 'b4_fits'
KINDS = ('ASYM', 'ASYM_STATE', 'NET')
PAIRS = (('ASYM_STATE', 'ASYM'), ('NET', 'ASYM_STATE'), ('NET', 'ASYM'))   # difference = risk(first) - risk(second)


def main():
    design = dict(np.load(OUT / 'design.npz'))
    feats = json.loads((W / 'results/b4_analysis.json').read_text())['features']
    _, P = projector_stack(TRAIN_K / M)
    recs = []
    for gi in range(len(GAMMAS)):
        m = design[f'mean_{gi}']; Pm = P(m); Qm = m - Pm
        for ri, rho in enumerate(RHOS):
            for n in NS:
                risk = {k: [] for k in KINDS}; eP = {k: [] for k in KINDS}; eQ = {k: [] for k in KINDS}; inits = []
                for rep in range(REPS):
                    files = {k: FITS / f'g{gi}_r{ri}_n{n}_rep{rep:02}_{k}.npz' for k in KINDS}
                    if not all(f.exists() for f in files.values()):
                        continue
                    for k, f in files.items():
                        with np.load(f) as a:
                            g = a['best_prediction']
                        Pg = P(g)
                        eP[k].append(((Pg - Pm) ** 2).sum() / D); eQ[k].append((((g - Pg) - Qm) ** 2).sum() / D)
                        risk[k].append(((g - m) ** 2).sum() / D)
                    inits.append(INIT_SEEDS[rep % len(INIT_SEEDS)])
                if len(inits) < 10:
                    continue
                rec = {'gi': gi, 'gamma': float(GAMMAS[gi]), 'rho': float(rho), 'n': int(n), 'reps': len(inits),
                       'S': feats[str(gi)]['S'], 'xP': c_rho(rho) * feats[str(gi)]['trPS0'] / n / D}
                for k in KINDS:
                    rec[f'eP_{k}'] = float(np.mean(eP[k])); rec[f'eQ_{k}'] = float(np.mean(eQ[k])); rec[f'risk_{k}'] = float(np.mean(risk[k]))
                for a, b in PAIRS:
                    d = np.array(risk[a]) - np.array(risk[b])
                    rec[f'd_{a}_minus_{b}'] = float(d.mean()); rec[f'hw_{a}_minus_{b}'] = stratified_halfwidth(d, inits)
                recs.append(rec)
    cells = pd.DataFrame(recs)
    out = {'status': 'PREREGISTERED_ANALYSIS (revision 2)', 'cells': cells.round(12).to_dict(orient='records')}
    if cells.empty:
        print('no complete cells yet'); return
    cal = cells[cells.gamma == 0]
    alpha = {}
    for k in KINDS:
        X = np.c_[np.ones(len(cal)), cal.xP.values]
        alpha[k] = np.linalg.lstsq(X, cal[f'eP_{k}'].values, rcond=None)[0].tolist()
    mid = (alpha['ASYM'][0] + alpha['NET'][0]) / 2
    decision1 = 'two-flow parameterisation' if alpha['ASYM_STATE'][0] <= mid else 'state-access restriction'
    strong = cells[(cells.gamma >= .08) & (cells.rho == 0)]
    q_frac = (strong['eQ_ASYM_STATE'] / strong['S']).tolist()
    decision2 = bool(len(strong) and max(q_frac) <= .25)
    out.update({'envelope_P_alpha_beta_gamma0': alpha, 'floor_midpoint': mid, 'decision1_floor_source': decision1,
                'ASYM_STATE_excluded_error_over_S_strong_cells': q_frac, 'decision2_state_reading_uses_signal': decision2})
    (W / 'results/b4s_analysis.json').write_text(json.dumps(out, indent=1, default=float) + '\n')
    print(f"cells={len(cells)} (min reps {cells.reps.min()})")
    for k in KINDS:
        print(f"  shared-component envelope {k:10s}: alpha={alpha[k][0]*1e6:+.3f}e-6  beta={alpha[k][1]:.3f}")
    print(f"  decision 1 (floor source): {decision1}  [alpha_S={alpha['ASYM_STATE'][0]*1e6:.3f}e-6 vs midpoint {mid*1e6:.3f}e-6]")
    print(f"  decision 2 (state-reading two-flow uses excluded signal): {decision2}; eQ_S/S at strong cells: {[round(v, 3) for v in q_frac]}")
    show = cells[['gamma', 'rho', 'n', 'S'] + [f'd_{a}_minus_{b}' for a, b in PAIRS] + [f'hw_{a}_minus_{b}' for a, b in PAIRS]].copy()
    for col in show.columns[3:]:
        show[col] = show[col] * 1e6
    print(show.to_string(index=False, float_format=lambda v: f'{v:+.2f}'))


if __name__ == '__main__':
    main()
