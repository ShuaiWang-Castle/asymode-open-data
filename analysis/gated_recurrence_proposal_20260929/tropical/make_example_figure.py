#!/usr/bin/env python3
"""Retrospective illustration (not a prediction-time selection): Hurricane Isaias (2020), two counties chosen after the
analysis by the rule in results/running_example.json. One figure per training budget.
Colours fixed across the paper: NET blue, ASYM (two flow) red, ASYM_STATE red dashed, observed black, persistence grey.
Thin lines: the three seeds; thick lines: their mean. Both MSE definitions are printed:
  'per-seed' = mean over seeds of each seed's MSE; 'mean curve' = MSE of the seed-averaged prediction."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tropical_run as TR  # noqa: E402

BLUE, RED, INK, GREY = '#2a78d6', '#e34948', '#0b0b0b', '#8a8987'
PANEL, SPLITS = (sys.argv[1], sys.argv[2]) if len(sys.argv) == 3 else (None, None)   # features_v1D.npz, splits_v1D.json


def mse(p, y, m):
    return float((m * (p - y) ** 2).sum() / max(m.sum(), 1))


def main():
    if PANEL is None:
        raise SystemExit('usage: make_example_figure.py <features_v1D.npz> <splits_v1D.json>')
    d = TR.load(PANEL, SPLITS); ex = json.loads((HERE / 'results/running_example.json').read_text())
    units = [int(np.flatnonzero((d['fips'] == ex[q]['fips']) & (d['system'] == ex['system']))[0]) for q in ('p10', 'p90')]
    names = ['Orange County, NC (37135): 10th percentile of path level', 'Washington County, RI (44009): 90th percentile of path level']
    h = np.arange(1, TR.H + 1); table = {}
    for budget, fitdir in (('3000', 'results/fits'), ('6000', 'results/fits_long')):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
        table[budget] = {}
        for ax, u, name in zip(axes, units, names):
            fold = int(d['fold'][u]); y, m = d['y'][u], d['m'][u]
            pr = {}
            for k in TR.KINDS:
                pr[k] = []
                for s in TR.SEEDS:
                    a = np.load(HERE / fitdir / f'f{fold}_{k}_s{s}.npz'); i = int(np.flatnonzero(a['test_index'] == u)[0])
                    pr[k].append(a['pred'][i])
                pr[k] = np.array(pr[k])
            pers = np.full(TR.H, d['y0'][u])
            ax.axhline(0, color=GREY, lw=.6)
            for s in range(3):
                ax.plot(h, pr['NET'][s] * 100, color=BLUE, lw=.8, alpha=.35)
                ax.plot(h, pr['ASYM'][s] * 100, color=RED, lw=.8, alpha=.35)
            ax.plot(h, pr['NET'].mean(0) * 100, color=BLUE, lw=2, label='NET (net flow), seed mean')
            ax.plot(h, pr['ASYM'].mean(0) * 100, color=RED, lw=2, label='ASYM (two flow), seed mean')
            ax.plot(h, pr['ASYM_STATE'].mean(0) * 100, color=RED, lw=1.4, ls='--', label='ASYM_STATE, seed mean')
            ax.plot(h, pers * 100, color=GREY, lw=1.6, ls=':', label='persistence')
            ax.plot(h, np.where(m > 0, y, np.nan) * 100, color=INK, lw=2.2, label='observed')
            row = {k: {'per_seed': float(np.mean([mse(p, y, m) for p in pr[k]]) * 1e4), 'mean_curve': mse(pr[k].mean(0), y, m) * 1e4}
                   for k in TR.KINDS}
            row['persistence'] = mse(pers, y, m) * 1e4; table[budget][ex['p10']['fips'] if u == units[0] else ex['p90']['fips']] = row
            txt = ('MSE x1e-4    per-seed  mean curve\n'
                   f"NET         {row['NET']['per_seed']:9.4f} {row['NET']['mean_curve']:9.4f}\n"
                   f"ASYM        {row['ASYM']['per_seed']:9.4f} {row['ASYM']['mean_curve']:9.4f}\n"
                   f"persistence {row['persistence']:9.4f} {row['persistence']:9.4f}")
            top = max(np.nanmax(y * 100), np.max(pr['NET'] * 100), np.max(pr['ASYM'] * 100))
            bot = min(0, np.min(pr['NET'] * 100), np.min(pr['ASYM'] * 100))
            ax.set_ylim(bot - .05 * (top - bot), top + .55 * (top - bot))
            left = u == units[1]
            ax.text(.01 if left else .99, .98, txt, transform=ax.transAxes, ha='left' if left else 'right', va='top', fontsize=7.5,
                    family='monospace', color=INK, bbox=dict(boxstyle='round,pad=.3', fc='white', ec='#d8d7d2', lw=.6))
            ax.set_title(name, fontsize=9, color=INK); ax.set_xlabel('hours after forecast origin', color=INK)
            ax.set_ylabel('customers out (%)', color=INK)
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
        hd, lb = axes[0].get_legend_handles_labels()
        fig.legend(hd, lb, loc='outside lower center', ncol=5, fontsize=8, frameon=False)
        label = 'registered budget (3,000 updates)' if budget == '3000' else 'post-hoc longer budget (6,000 updates)'
        fig.suptitle(f'Hurricane Isaias (2020), retrospective illustration, {label}; thin lines: three seeds', fontsize=10, color=INK)
        fig.savefig(HERE / f'figures/running_example_isaias_{budget}.png', dpi=170)
        plt.close(fig)
    (HERE / 'results/running_example_mse.json').write_text(json.dumps(table, indent=1) + '\n')
    print(json.dumps(table, indent=1))


if __name__ == '__main__':
    main()
