#!/usr/bin/env python3
"""Paired seed statistics and a descriptive 168h block dependence sensitivity.

Five paired seeds are NOT five independent field samples; they are reported as
paired replicates of the same retrospective evaluation. The block interval is a
descriptive dependence sensitivity, never a confirmatory p-value.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

SEEDS = (5101, 5102, 5103, 5104, 5105)


def main(work: Path):
    res = work / 'results'
    z = np.load(res / 'PREDICTIONS_2019.npz')
    truth = z['truth'].astype(np.float64); groups = z['groups']; hours = z['origin_hours']
    sel = pd.read_csv(work / '_package/ASYMODE_CC_STANDALONE_20260909/metadata/unit_selection.csv')
    sel = sel[sel.evaluation_status == 'included'].sort_values('group').reset_index(drop=True)
    comp = np.array([sel.set_index('group').company[int(g)] for g in groups])

    def company_mse(pred, mask=None):
        e = (pred - truth) ** 2
        e = e[:, mask] if mask is not None else e
        per_g = e.mean(axis=(1, 2))
        return pd.Series(per_g).groupby(comp).mean()

    out, rows = {}, []
    for seed in SEEDS:
        d = company_mse(z[f'DIRECT_seed{seed}'].astype(np.float64))
        for arm in ('NET', 'ASYM'):
            a = company_mse(z[f'{arm}_seed{seed}'].astype(np.float64))
            rel = ((d - a) / d)
            rows.append({'arm': arm, 'seed': seed,
                         'mean_company_relative_gain': float(rel.mean()),
                         'median_company_relative_gain': float(rel.median()),
                         'companies_favouring_arm': int((rel > 0).sum()),
                         'n_companies': int(len(rel)),
                         'ratio_of_means_gain': float((d.mean() - a.mean()) / d.mean())})
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(res / 'PAIRED_PER_SEED.csv', index=False)

    summ = []
    for arm in ('NET', 'ASYM'):
        s = per_seed[per_seed.arm == arm]
        summ.append({'arm': arm,
                     'paired_seeds': len(s),
                     'mean_of_seed_gains': float(s.mean_company_relative_gain.mean()),
                     'min_seed_gain': float(s.mean_company_relative_gain.min()),
                     'max_seed_gain': float(s.mean_company_relative_gain.max()),
                     'seeds_with_positive_gain': int((s.mean_company_relative_gain > 0).sum()),
                     'mean_companies_favouring_arm': float(s.companies_favouring_arm.mean())})
    pd.DataFrame(summ).to_csv(res / 'PAIRED_SUMMARY.csv', index=False)

    # descriptive 168h common-calendar block resampling, seed-averaged predictions
    rng = np.random.default_rng(20260909)
    block = 168
    edges = np.arange(hours.min(), hours.max() + block, block)
    bid = np.digitize(hours, edges) - 1
    blocks = np.unique(bid)
    avg = {arm: np.mean([z[f'{arm}_seed{s}'].astype(np.float64) for s in SEEDS], axis=0)
           for arm in ('DIRECT', 'NET', 'ASYM')}
    boot = {a: [] for a in ('NET', 'ASYM')}
    for _ in range(2000):
        pick = rng.choice(blocks, size=len(blocks), replace=True)
        mask = np.concatenate([np.flatnonzero(bid == b) for b in pick])
        d = company_mse(avg['DIRECT'], mask)
        for arm in ('NET', 'ASYM'):
            a = company_mse(avg[arm], mask)
            boot[arm].append(float(((d - a) / d).mean()))
    ci = {arm: {'p2.5': float(np.quantile(v, .025)), 'p50': float(np.quantile(v, .5)),
                'p97.5': float(np.quantile(v, .975))} for arm, v in boot.items()}
    out = {'block_hours': block, 'n_blocks': int(len(blocks)), 'resamples': 2000,
           'basis': 'seed-averaged predictions; blocks shared across companies to retain common shocks',
           'interpretation': 'descriptive dependence sensitivity only; not a confirmatory test; '
                             'five paired seeds are not five independent field samples',
           'mean_company_relative_gain_vs_DIRECT': ci}
    (res / 'BLOCK_SENSITIVITY.json').write_text(json.dumps(out, indent=2))
    print(per_seed.to_string(index=False, float_format=lambda x: f'{x:.6g}'))
    print()
    print(pd.DataFrame(summ).to_string(index=False, float_format=lambda x: f'{x:.6g}'))
    print()
    print(json.dumps(ci, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--work', required=True)
    main(Path(ap.parse_args().work))
