"""Compact D01 comparison and figure from completed scores only; no fitting."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE/'results/v1'
SPECS = ['past_only', 'future_adjusted', 'aligned_exposure']
NAMES = ['Past weather', 'Past + future severity control', 'Same-window weather (retrospective)']
CONTRASTS = ['B_vs_A', 'C_vs_B', 'D_vs_C', 'E_vs_D', 'F_vs_E', 'F_vs_D']
LABELS = ['Individual weather x geography', 'Weather compounds', 'Directional weather order',
          'Compound x geography', 'Order x geography', 'Joint geographic modulation (primary)']


def main():
    scores = {s: json.loads((OUT/f'data_first_d01_{s}.json').read_text()) for s in SPECS}
    for s in SPECS:
        assert scores[s]['meta']['audit'] == scores[SPECS[0]]['meta']['audit']
        assert scores[s]['meta']['source_sha256'] == scores[SPECS[0]]['meta']['source_sha256']
        assert scores[s]['meta']['protocol_sha256'] == scores[SPECS[0]]['meta']['protocol_sha256']
    summary = dict(audit=scores[SPECS[0]]['meta']['audit'], exploratory=True,
                   lower_is_better=True, confidence='95% conditional merged-event-cluster bootstrap',
                   primary='F/D tropical-winter balanced MSE of next-24h burden', specifications={})
    for s, data in scores.items():
        primary = data['contrasts']['F_vs_D']['burden']
        summary['specifications'][s] = dict(
            contrasts={c: data['contrasts'][c]['burden'] for c in CONTRASTS},
            secondary_risk_joint=data['contrasts']['F_vs_D']['rise_risk'],
            nulls={f'F_vs_F_null{i}':data['contrasts'][f'F_vs_F_null{i}']['burden'] for i in range(1,4)},
            primary_checks=dict(point_improves=primary['headline']<0,
                                event_interval_below_zero=primary['headline_ci95'][1]<0,
                                improved_folds=sum(x<0 for x in primary['fold_headline']),
                                removal_improves=primary['headline_without_biggest']<0,
                                beats_all_nulls=all(data['contrasts'][f'F_vs_F_null{i}']['burden']['headline']<0
                                                    for i in range(1,4))))
    (OUT/'data_first_d01_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    colors = ['#466f9b', '#c17837', '#3c8c76']
    fig, ax = plt.subplots(figsize=(11.6,6.8))
    y = np.arange(len(CONTRASTS))[::-1]
    for j,s in enumerate(SPECS):
        p = np.array([scores[s]['contrasts'][c]['burden']['headline'] for c in CONTRASTS])*100
        ci = np.array([scores[s]['contrasts'][c]['burden']['headline_ci95'] for c in CONTRASTS])*100
        # Draw endpoints directly: a percentile interval need not contain its point estimate.
        yy = y+(j-1)*.20
        ax.hlines(yy,ci[:,0],ci[:,1],color=colors[j],linewidth=1.8,alpha=.85)
        ax.scatter(p,yy,color=colors[j],s=35,label=NAMES[j],zorder=3)
    ax.axvline(0,color='#444444',linewidth=1,linestyle='--')
    ax.set_yticks(y,LABELS)
    ax.set_xlabel('Change in tropical/winter balanced burden MSE (%) — lower is better')
    ax.set_title('D01: does extra weather–geography structure generalize to held-out events?',loc='left',pad=20)
    ax.grid(axis='x',alpha=.15)
    ax.spines[['top','right','left']].set_visible(False)
    ax.tick_params(axis='y',length=0)
    ax.legend(loc='upper center',bbox_to_anchor=(.48,-.14),frameon=False,ncol=1,fontsize=9)
    fig.text(.02,.012,'Public D: 81 systems, 68 merged event groups; 5 event folds. Lines: 95% cluster intervals. Exploratory associations, not causal effects.',fontsize=8,color='#555555')
    fig.tight_layout(rect=(0,.07,1,1))
    fig.savefig(OUT/'fig_data_first_d01.png',dpi=180,bbox_inches='tight')
    plt.close(fig)
    print(json.dumps({s:summary['specifications'][s]['primary_checks'] for s in SPECS},indent=2))


if __name__ == '__main__':
    main()
