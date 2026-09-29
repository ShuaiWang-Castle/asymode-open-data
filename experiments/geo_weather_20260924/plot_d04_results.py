"""Static D04 score and geographic-response figures from compact validated results."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/d04_matplotlib')
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm,TwoSlopeNorm

HERE=Path(__file__).resolve().parent
OUT=HERE/'results/v1'
REGIMES=['tropical','winter','synoptic_wind','convective','heavy_rain']
ARMS=['A_shared_main','B_shared_pairs','C_geo_main','D_geo_pairs']


def save(fig,stem):
    for suffix in ('png','pdf'):
        path=OUT/(stem+'.'+suffix)
        if path.exists():raise FileExistsError('Preserve existing D04 figure')
        fig.savefig(path,dpi=180,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def main():
    scores=json.loads((OUT/'d04_response_scores.json').read_text())['design_weights']
    ref=np.asarray(scores['arms']['zero_change']['per_regime_mse'])
    matrix=np.array([100*(np.asarray(scores['arms'][a]['per_regime_mse'])/ref-1) for a in ARMS])
    bound=max(1,float(np.max(abs(matrix))))
    fig,ax=plt.subplots(figsize=(11,4.6))
    im=ax.imshow(matrix,cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-bound,vcenter=0,vmax=bound),aspect='auto')
    for i in range(4):
        comp=scores['comparisons'][ARMS[i]+'_vs_zero_change']['merged_event']['per_regime_mse']
        interval=np.array(comp['conditional_score_ci95_percent'])
        for j in range(5):
            ax.text(j,i,f'{matrix[i,j]:+.1f}%\n[{interval[0,j]:+.1f}, {interval[1,j]:+.1f}]',
                    ha='center',va='center',fontsize=9,
                    color='white' if abs(matrix[i,j])>.65*bound else '#192638')
    ax.set_xticks(range(5),['Tropical','Winter','Synoptic wind','Convective','Heavy rain'])
    ax.set_yticks(range(4),['Shared weather','Shared + pairs','Geo x weather','Geo x weather/pairs'])
    ax.set_title('D04: one-hour net-change error on held-out events',loc='left',pad=15,fontweight='bold')
    fig.colorbar(im,ax=ax,label='MSE change vs zero change (%)',shrink=.85)
    fig.text(.02,.01,'Lower is better. All valid held-out hours; observed previous stock is a control.\nIntervals resample fixed OOF scores by merged event; they omit fitting/selection uncertainty. This is not the neural-model screen.\nNumerically provisional: the fits did not meet the registered gradient stopping criterion.',fontsize=9)
    fig.tight_layout(rect=(0,.16,1,1));save(fig,'fig_d04_oof_errors')
    surfaces=json.loads((OUT/'d04_response_surfaces.json').read_text())
    names=surfaces['meta']['weather_names']
    response=surfaces['responses']['D_geo_component']
    for regime in REGIMES:
        mats=[]
        for domain in ('all_observed','within_proxy'):
            columns=[]
            for t in range(6):
                d=response[regime][domain]['county_types'][str(t)]
                columns.append(np.asarray(d['mean']).T*100 if d['rows'] else np.full((12,3),np.nan))
            mats.append(np.concatenate(columns,axis=1))
        finite=np.concatenate([m[np.isfinite(m)] for m in mats]);bound=max(1e-8,float(np.max(abs(finite))))
        threshold=max(bound/200,1e-8)
        fig,axes=plt.subplots(2,1,figsize=(13.5,9.5),sharex=True,layout='constrained')
        for ax,m,title in zip(axes,mats,['All observed paths','Within approximate joint-history support']):
            im=ax.imshow(m,cmap='RdBu_r',norm=SymLogNorm(linthresh=threshold,vmin=-bound,vmax=bound),aspect='auto')
            ax.set_yticks(range(12),names,fontsize=9)
            ax.set_title(title,loc='left',fontsize=11)
            for edge in [2.5,5.5,8.5,11.5,14.5]:ax.axvline(edge,color='white',lw=1.5)
        axes[1].set_xticks(range(18),[f'T{t}\n{lag} h' for t in range(6) for lag in ['1–6','7–24','25–48']],fontsize=8)
        fig.colorbar(im,ax=axes,label='Net-change sensitivity (percentage points / common weather SD)',shrink=.8)
        fig.suptitle(f'D04 geographic modulation of local weather response: {regime.replace("_"," ")}',fontsize=14,fontweight='bold')
        fig.supxlabel('Types are descriptive navigation groups. Frozen-model geographic component; finite lag bands; no causal or physical-unit interpretation.\nColors use a symmetric logarithmic scale; no cells are selected by significance. Support is an approximation, not exchangeability.\nNumerically provisional: fits did not meet the registered gradient criterion; fitted-surface uncertainty is not estimated.',fontsize=9)
        save(fig,'fig_d04_geo_response_'+regime)


if __name__=='__main__':main()
