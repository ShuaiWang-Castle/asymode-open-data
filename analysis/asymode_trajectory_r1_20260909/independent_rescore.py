#!/usr/bin/env python3
"""Independent recomputation of the main table from stored predictions.

Deliberately does NOT import training_primitives.score_full_paths; it reimplements
the company-equal aggregation from the stored arrays so a scorer bug shows up as
a disagreement rather than cancelling out.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd


def main(work: Path):
    res = work / 'results'
    z = np.load(res / 'PREDICTIONS_2019.npz')
    truth = z['truth'].astype(np.float64)                 # [G, N, 24]
    groups = z['groups']
    sel = pd.read_csv(work / '_package/ASYMODE_CC_STANDALONE_20260909/metadata/unit_selection.csv')
    sel = sel[sel.evaluation_status == 'included'].sort_values('group').reset_index(drop=True)
    assert list(sel.group) == list(groups), 'group order mismatch'
    comp_of = dict(zip(sel.group.astype(int), sel.company.astype(int)))

    rows = []
    for key in z.files:
        if key in ('truth', 'origin_hours', 'groups'):
            continue
        p = z[key].astype(np.float64)
        e = (p - truth) ** 2
        per_group = pd.DataFrame({
            'group': groups,
            'company': [comp_of[int(g)] for g in groups],
            'mse_path': e.mean(axis=(1, 2)),
            'mse_1h': e[:, :, 0].mean(axis=1),
            'mse_6h': e[:, :, 5].mean(axis=1),
            'mse_24h': e[:, :, 23].mean(axis=1)})
        per_comp = per_group.groupby('company')[['mse_path', 'mse_1h', 'mse_6h', 'mse_24h']].mean()
        model, seed = key.rsplit('_seed', 1)
        rows.append({'model': model, 'seed': int(seed),
                     **{c: float(per_comp[c].mean()) for c in per_comp.columns}})
    mine = pd.DataFrame(rows).sort_values(['model', 'seed']).reset_index(drop=True)
    mine.to_csv(res / 'INDEPENDENT_RESCORE.csv', index=False)

    ref = pd.read_csv(res / 'MAIN_REAL_RESULTS.csv')
    ref = ref[ref.seed.notna()].copy(); ref['seed'] = ref.seed.astype(int)
    j = mine.merge(ref, on=['model', 'seed'], suffixes=('_mine', '_ref'))
    diffs = {}
    for c in ('mse_path', 'mse_1h', 'mse_6h', 'mse_24h'):
        diffs[c] = float((j[f'{c}_mine'] - j[f'{c}_ref']).abs().max())
    out = {'rows_compared': len(j), 'max_abs_difference': diffs,
           'tolerance': 1e-12,
           'pass': all(v <= 1e-12 for v in diffs.values())}
    (res / 'INDEPENDENT_RESCORE_DIFF.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0 if out['pass'] else 1


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--work', required=True)
    raise SystemExit(main(Path(ap.parse_args().work)))
