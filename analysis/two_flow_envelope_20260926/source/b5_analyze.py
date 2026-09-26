#!/usr/bin/env python3
"""Pre-registered generality analysis (experiment plan, revision 3), separately for each (law, size) combination.

H1: shared-component floors alpha_S < alpha_N and alpha_A < alpha_N (envelopes calibrated on the no-feedback cells).
H2: beta_N < beta_S (shared component).
H3: at S > 0, rho = 0 cells ASYM_STATE is not worse than ASYM beyond the half-width; at no-feedback cells the difference is
    within +-half-width or favours ASYM_STATE.
H4: the NET-versus-ASYM envelope criterion (calibrated on no-feedback cells) predicts the sign of Delta on resolved
    feedback cells with accuracy >= 80% and >= the classical ideal-estimator criterion x_Q - S.
K7: H1 fails in any new combination. K8: H4 accuracy < 80% in any new combination.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent; W = HERE.parent
sys.path.insert(0, str(HERE))
import b4_data as A  # noqa: E402
import b5_data as BB  # noqa: E402
import dualflow as DF  # noqa: E402
from b4_analyze import projector_stack, stratified_halfwidth, c_rho, D  # noqa: E402
from b5_experiment import FITS, COMBOS, FEEDBACK_IDX, NS, REPS, KINDS, data_dir  # noqa: E402


def features(law):
    design = dict(np.load(data_dir(law) / 'design.npz'))
    z = A.TRAIN_K / A.M; Pz, P = projector_stack(z)
    out = {}
    for fi in FEEDBACK_IDX:
        m = design[f'mean_{fi}']; cov = design[f'cov_{fi}']; Pm = P(m)
        S = float(((m - Pm) ** 2).sum()) / D
        F = sum(DF.fit(Pm[4 * f:4 * f + 4], z, n_random=4, seed=41 + f)['loss'] for f in range(2)) / D
        diag = np.stack([np.diag(cov[c]) for c in range(8)])
        trS = float(diag.sum()); trPS = float(sum(Pz[i, i] * diag[4 * f + i].sum() for f in range(2) for i in range(4)))
        out[fi] = {'S': S, 'F': F, 'trPS0': trPS, 'trQS0': trS - trPS, 'mean': m}
    return out


def cells_for(law, target, feats):
    _, P = projector_stack(A.TRAIN_K / A.M)
    recs = []
    for fi in FEEDBACK_IDX:
        m = feats[fi]['mean']; Pm = P(m); Qm = m - Pm
        for ri, rho in enumerate(A.RHOS):
            for n in NS:
                acc = {k: {'risk': [], 'eP': [], 'eQ': []} for k in KINDS}; inits = []
                for rep in range(REPS):
                    files = {k: FITS / f'{law}{target}_f{fi}_r{ri}_n{n}_rep{rep:02}_{k}.npz' for k in KINDS}
                    if not all(f.exists() for f in files.values()):
                        continue
                    for k, f in files.items():
                        with np.load(f) as a:
                            g = a['best_prediction']
                        Pg = P(g)
                        acc[k]['risk'].append(((g - m) ** 2).sum() / D); acc[k]['eP'].append(((Pg - Pm) ** 2).sum() / D)
                        acc[k]['eQ'].append((((g - Pg) - Qm) ** 2).sum() / D)
                    inits.append(A.INIT_SEEDS[rep % len(A.INIT_SEEDS)])
                if len(inits) < 10:
                    continue
                rec = {'law': law, 'target': target, 'fi': fi, 'rho': float(rho), 'n': int(n), 'reps': len(inits),
                       'S': feats[fi]['S'], 'F': feats[fi]['F'],
                       'xP': c_rho(rho) * feats[fi]['trPS0'] / n / D, 'xQ': c_rho(rho) * feats[fi]['trQS0'] / n / D}
                for k in KINDS:
                    for q in ('risk', 'eP', 'eQ'):
                        rec[f'{q}_{k}'] = float(np.mean(acc[k][q]))
                for a, b in (('NET', 'ASYM'), ('ASYM_STATE', 'ASYM'), ('NET', 'ASYM_STATE')):
                    d = np.array(acc[a]['risk']) - np.array(acc[b]['risk'])
                    rec[f'd_{a}_{b}'] = float(d.mean()); rec[f'hw_{a}_{b}'] = stratified_halfwidth(d, inits)
                recs.append(rec)
    return pd.DataFrame(recs)


def linfit(x, y):
    X = np.c_[np.ones(len(x)), x]
    return np.linalg.lstsq(X, y, rcond=None)[0].tolist()


def main():
    feats = {law: features(law) for law in ('A', 'B')}
    report = {'status': 'PREREGISTERED_ANALYSIS (revision 3)'}
    lines = []
    for law, target in COMBOS:
        c = cells_for(law, target, feats[law])
        if c.empty:
            lines.append(f'{law}{target}: no complete cells yet'); continue
        cal = c[c.fi == 0]
        env = {f'eP_{k}': linfit(cal.xP.values, cal[f'eP_{k}'].values) for k in KINDS}
        env.update({f'eQ_{k}': linfit(cal.xQ.values, cal[f'eQ_{k}'].values) for k in ('NET', 'ASYM_STATE')})
        H1 = env['eP_ASYM_STATE'][0] < env['eP_NET'][0] and env['eP_ASYM'][0] < env['eP_NET'][0]
        H2 = env['eP_NET'][1] < env['eP_ASYM_STATE'][1]
        low = c[(c.fi > 0) & (c.rho == 0)]
        H3a = bool((low['d_ASYM_STATE_ASYM'] <= low['hw_ASYM_STATE_ASYM']).all()) if len(low) else None
        z0 = c[c.fi == 0]
        H3b = bool(((z0['d_ASYM_STATE_ASYM'].abs() <= z0['hw_ASYM_STATE_ASYM']) | (z0['d_ASYM_STATE_ASYM'] < 0)).all())
        e = lambda k, x: env[k][0] + env[k][1] * x
        c['envelope_pred'] = [e('eP_NET', r.xP) - e('eP_ASYM', r.xP) - r.F + e('eQ_NET', r.xQ) - r.S for r in c.itertuples()]
        c['ideal_pred'] = c.xQ - c.S
        held = c[(c.fi > 0) & (c['d_NET_ASYM'].abs() > c['hw_NET_ASYM'])]
        acc_env = float(np.mean(np.sign(held.envelope_pred) == np.sign(held.d_NET_ASYM))) if len(held) else None
        acc_id = float(np.mean(np.sign(held.ideal_pred) == np.sign(held.d_NET_ASYM))) if len(held) else None
        H4 = (acc_env is not None and acc_env >= .8 and acc_env >= acc_id)
        key = f'{law}{target}'
        report[key] = {'envelopes': env, 'H1': bool(H1), 'H2': bool(H2), 'H3_low_noise_signal': H3a, 'H3_no_feedback': H3b,
                       'H4': bool(H4), 'held_out_resolved': int(len(held)), 'held_out_total': int((c.fi > 0).sum()),
                       'accuracy_envelope': acc_env, 'accuracy_ideal': acc_id, 'K7_triggered': not H1, 'K8_triggered': not H4,
                       'cells': c.round(12).to_dict(orient='records')}
        lines.append(f'== {key}: cells={len(c)} (min reps {c.reps.min()})')
        for k in KINDS:
            lines.append(f"   shared floor/slope {k:10s}: alpha={env['eP_' + k][0]*1e6:+.3f}e-6  beta={env['eP_' + k][1]:.3f}")
        lines.append(f"   excluded-direction slope NET beta_Q={env['eQ_NET'][1]:.3f}  ASYM_STATE beta_Q={env['eQ_ASYM_STATE'][1]:.3f}")
        lines.append(f'   H1={H1}  H2={H2}  H3(low-noise, S>0)={H3a}  H3(no feedback)={H3b}  H4={H4}  '
                     f'[resolved {len(held)}/{int((c.fi > 0).sum())}: envelope {acc_env}, ideal {acc_id}]')
        show = c[['fi', 'rho', 'n', 'S', 'd_NET_ASYM', 'hw_NET_ASYM', 'envelope_pred', 'ideal_pred', 'd_ASYM_STATE_ASYM', 'hw_ASYM_STATE_ASYM', 'd_NET_ASYM_STATE', 'hw_NET_ASYM_STATE']].copy()
        for col in show.columns[3:]:
            show[col] = show[col] * 1e6
        lines.append(show.to_string(index=False, float_format=lambda v: f'{v:+.2f}'))
    (W / 'results/b5_analysis.json').write_text(json.dumps(report, indent=1, default=float) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
