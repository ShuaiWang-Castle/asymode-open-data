#!/usr/bin/env python3
"""POST-HOC additions to the registered generality analysis (b5_analyze.py itself is unchanged).

Run after b5_analyze.py has written results/b5_analysis.json.
  1. Completeness: every one of the 5,760 fits (3 combinations x 4 feedback levels x 2 rho x 2 n x 40 replications
     x 3 models) has a completed record and finite predictions, and every analysed cell has all 40 paired triplets.
  2. The reason K8 triggered or passed: envelope accuracy below 80%, below the classical criterion, or no resolved cells.
  3. Baselines on the resolved feedback cells (always NET, always ASYM; the bias-only predictors -S and -(S+F) always
     predict NET because S > 0) and the MAE over all feedback cells of the envelope, the classical criterion, -S,
     -(S+F), and the no-feedback mean offset plus -S or -(S+F).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent; W = HERE.parent
sys.path.insert(0, str(HERE))
import b4_data as A  # noqa: E402
from b5_experiment import FITS, COMBOS, FEEDBACK_IDX, NS, REPS, KINDS, task_key  # noqa: E402


def completeness():
    missing, bad = [], []
    for law, target in COMBOS:
        for fi in FEEDBACK_IDX:
            for ri in range(len(A.RHOS)):
                for n in NS:
                    for rep in range(REPS):
                        for k in KINDS:
                            key = task_key(law, target, fi, ri, n, rep, k)
                            j, z = FITS / f'{key}.json', FITS / f'{key}.npz'
                            if not (j.exists() and z.exists()):
                                missing.append(key); continue
                            if json.loads(j.read_text()).get('status') != 'completed':
                                bad.append(key); continue
                            with np.load(z) as a:
                                if not all(np.isfinite(a[x]).all() for x in ('best_prediction', 'best_test_prediction', 'snapshots')):
                                    bad.append(key)
    expected = len(COMBOS) * len(FEEDBACK_IDX) * len(A.RHOS) * len(NS) * REPS * len(KINDS)
    return {'expected': expected, 'missing': len(missing), 'incomplete_or_nonfinite': len(bad), 'examples': (missing + bad)[:10]}


def main():
    comp = completeness()
    if comp['missing'] or comp['incomplete_or_nonfinite']:
        print(json.dumps(comp, indent=1)); raise SystemExit('generality run incomplete: analysis not valid')
    reg = json.loads((W / 'results/b5_analysis.json').read_text())
    out = {'status': 'POST_HOC_AFTER_INDEPENDENT_REVIEW', 'completeness': comp}
    lines = [f"completeness: {comp['expected']} fits expected, 0 missing, 0 incomplete or non-finite"]
    for law, target in COMBOS:
        key = f'{law}{target}'; r = reg[key]; c = pd.DataFrame(r['cells'])
        if int(c.reps.min()) != REPS or len(c) != len(FEEDBACK_IDX) * len(A.RHOS) * len(NS):
            raise SystemExit(f'{key}: cell table incomplete (min reps {int(c.reps.min())}, cells {len(c)})')
        fb = c[c.fi > 0].copy(); held = fb[fb.d_NET_ASYM.abs() > fb.hw_NET_ASYM]
        off = float(c[c.fi == 0].d_NET_ASYM.mean())
        preds = {'envelope (registered)': fb.envelope_pred, 'classical x_Q - S (registered)': fb.ideal_pred,
                 '-S': -fb.S, '-(S+F)': -(fb.S + fb.F), 'no-feedback offset - S': off - fb.S, 'no-feedback offset - (S+F)': off - (fb.S + fb.F)}
        tab = [{'predictor': k, 'MAE_all_feedback_cells_1e-6': float((p - fb.d_NET_ASYM).abs().mean() * 1e6),
                'resolved_correct': int((np.sign(p[held.index]) == np.sign(held.d_NET_ASYM)).sum()), 'resolved': int(len(held))}
               for k, p in preds.items()]
        tab.append({'predictor': 'always NET (sign only)', 'MAE_all_feedback_cells_1e-6': None,
                    'resolved_correct': int((held.d_NET_ASYM < 0).sum()), 'resolved': int(len(held))})
        tab.append({'predictor': 'always ASYM (sign only)', 'MAE_all_feedback_cells_1e-6': None,
                    'resolved_correct': int((held.d_NET_ASYM > 0).sum()), 'resolved': int(len(held))})
        acc_env, acc_id = r['accuracy_envelope'], r['accuracy_ideal']
        if not len(held):
            reason = 'no resolved feedback cells'
        else:
            reason = '; '.join(x for x in (f'envelope accuracy {acc_env:.3f} < 0.80' if acc_env < .8 else '',
                                           f'envelope accuracy {acc_env:.3f} < classical {acc_id:.3f}' if acc_env < acc_id else '') if x) or 'passed'
        out[key] = {'K8_triggered': r['K8_triggered'], 'K8_reason': reason, 'K7_triggered': r['K7_triggered'],
                    'winners_feedback_cells': {'NET': int((fb.d_NET_ASYM < 0).sum()), 'ASYM': int((fb.d_NET_ASYM > 0).sum())},
                    'resolved_winners': {'NET': int((held.d_NET_ASYM < 0).sum()), 'ASYM': int((held.d_NET_ASYM > 0).sum())},
                    'table': tab}
        lines.append(f"== {key}: K7 {r['K7_triggered']}; K8 {r['K8_triggered']} ({reason}); feedback cells NET-better "
                     f"{int((fb.d_NET_ASYM < 0).sum())}/{len(fb)}; resolved {len(held)} (NET {int((held.d_NET_ASYM < 0).sum())}, "
                     f"ASYM {int((held.d_NET_ASYM > 0).sum())})")
        lines.append(pd.DataFrame(tab).to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    (W / 'results/b5_posthoc.json').write_text(json.dumps(out, indent=1, default=float) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
