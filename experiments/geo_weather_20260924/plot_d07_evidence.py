"""Scientific figure from D07's saved, independently checked outputs."""
from __future__ import annotations
import os
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
import gzip
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evaluate_cr_tail import HERE, RUNS, load_outcomes


def main():
    if os.getpriority(os.PRIO_PROCESS,0)<15:os.nice(15-os.getpriority(os.PRIO_PROCESS,0))
    folder=HERE/'results/v1'
    evidence=json.loads((folder/'d07_evidence.json').read_text())
    a=json.loads(gzip.decompress((folder/'d07_attribution.json.gz').read_bytes()))
    data=load_outcomes()
    with np.load(RUNS/'d07_selectivity_20260929/attribution_unit_rows.npz',allow_pickle=False) as z:
        gains=data['meta']['w'][z['cohort_S']]*z['oof_independent_host__gain'][z['cohort_S']]
    fig,axes=plt.subplots(2,2,figsize=(12.8,8.4),layout='constrained')
    plt.rcParams.update({'font.size':10})
    blue,red='#2563a5','#be5046'
    ax=axes[0,0];positive=np.sort(np.maximum(gains,0))[::-1]
    curve=100*np.r_[0,np.cumsum(positive)]/positive.sum()
    ax.plot(np.arange(len(curve)),curve,color=blue,lw=2)
    ax.axvline(73,color='#667085',ls='--',lw=1)
    ax.scatter([73],[curve[73]],color=blue,zorder=3)
    ax.text(105,77,f'73 / 726 S units\n{curve[73]:.2f}% of positive gain\n267 improve; 459 worsen',fontsize=10)
    ax.set(title='A. OOF gains concentrate in a small tail',xlabel='S units, ranked by positive weighted SSE gain',ylabel='Cumulative share of positive gain (%)',ylim=(0,104))
    s=a['summaries']['oof_independent_host']['supports']['common_observed']['S_outcome_phenotypes']
    names=['severe_hours/1','severe_hours/2..6','severe_hours/>=7','true_peak/[.1,.2)','true_peak/[.2,1]']
    labels=['1 severe\nhour','2–6 severe\nhours','≥7 severe\nhours','10–20%\ntrue peak','≥20%\ntrue peak']
    values=np.array([s[n]['totals']['gain'] for n in names]);ax=axes[0,1]
    bars=ax.bar(np.arange(5),values,color=np.where(values>=0,blue,red),width=.65)
    ax.bar_label(bars,labels=[f'{v:+.1f}' for v in values],padding=3,fontsize=9)
    ax.axhline(0,color='#667085',lw=1);ax.set_xticks(np.arange(5),labels,fontsize=9)
    ax.set(title='B. Different outage phenotypes offset each other',ylabel='Net design-weighted SSE improvement',ylim=(-210,315))
    ax.text(.02,.95,'Duration and peak partitions overlap.\nOutcome-defined descriptive groups.',transform=ax.transAxes,va='top',fontsize=9)
    ax=axes[1,0];xs=np.arange(2);width=.25
    for j,(cohort,color) in enumerate([('all','#777f89'),('S',blue),('nonS',red)]):
        vals=[100*r['splits']['OUTER'][cohort]['RMSE_change_fraction'] for r in evidence['write_interventions']]
        bars=ax.bar(xs+(j-1)*width,vals,width,label=cohort,color=color)
        ax.bar_label(bars,labels=[f'{v:+.3f}' for v in vals],padding=3,fontsize=8)
    ax.axhline(0,color='#667085',lw=1);ax.set_xticks(xs,['Write weather ×0.95','Write weather ×1.05'])
    ax.set(title='C. Write strength mainly changes non-S error',ylabel='OUTER RMSE change (%) — positive is worse',ylim=(-2.6,2.4));ax.legend(frameon=False,ncol=3,loc='upper left')
    ax.text(.03,.77,'OUTER severe false peaks: 49 → 47 / 50',transform=ax.transAxes,fontsize=9)
    ax=axes[1,1];small=[r for r in evidence['parameter_interventions'] if r['relative_epsilon']==.001]
    xs=np.arange(len(small));width=.34
    for j,(cohort,color) in enumerate([('S',blue),('nonS',red)]):
        vals=[100*r['splits']['OUTER'][cohort]['RMSE_change_fraction'] for r in small]
        bars=ax.bar(xs+(j-.5)*width,vals,width,label=cohort,color=color)
        ax.bar_label(bars,labels=[f'{v:+.3f}' for v in vals],padding=3,fontsize=8)
    ax.axhline(0,color='#667085',lw=1);ax.set_xticks(xs,['Weather control','Raw geography','Geo basis'],fontsize=9)
    ax.set(title='D. FIT descent directions carry non-S costs',ylabel='OUTER RMSE change (%) — positive is worse',ylim=(-.9,6.6));ax.legend(frameon=False,ncol=2,loc='upper right')
    ax.text(.03,.86,'Separate 0.1% relative parameter steps;\nweather false peaks: 49 → 56',transform=ax.transAxes,fontsize=9)
    for ax in axes.flat:
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('D07: frozen public-D response allocation and finite sensitivity experiments',fontsize=15)
    for extension in ('png','pdf'):
        path=folder/f'fig_d07_response_selectivity.{extension}'
        if path.exists():raise FileExistsError(path)
        fig.savefig(path,dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
