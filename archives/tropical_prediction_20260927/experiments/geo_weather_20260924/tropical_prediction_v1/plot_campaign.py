"""Publication figure exports from verified aggregate tables only."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
RESULTS=HERE/'results'
OUT=RESULTS/'figures'
LABELS={'zero':'Zero forecast','persistence':'Persistence','damped':'Damped persistence',
        'host':'Matched AsymODE host','fusion':'Joint GCRK-rate model',
        'tree_era5_l2':'ERA5 residual tree','tree_dual_l2':'Dual-weather residual tree',
        'tree_dual_l1':'Dual-weather MAE tree','blend':'Inner-selected ensemble'}
COLORS={'host':'#7298AC','fusion':'#C47870','blend':'#343F54'}

def style(ax):
    ax.spines[['top','right','left']].set_visible(False)
    ax.tick_params(axis='y',length=0)
    ax.grid(axis='x',color='#E5E7EB',linewidth=.7)
    ax.set_axisbelow(True)

def save(fig,name):
    fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(OUT/f'{name}.png',dpi=220,bbox_inches='tight')
    plt.close(fig)

def main():
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.labelsize':13,
        'axes.titlesize':15,'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})
    df=pd.read_csv(RESULTS/'overall_metrics.csv').query("weighting=='design'").set_index('model')
    names=list(LABELS);yy=np.arange(len(names))
    fig,axes=plt.subplots(1,2,figsize=(12.2,5.8),sharey=True)
    for ax,metric,title in zip(axes,['rmse','mae'],['a  Pooled RMSE','b  Pooled MAE']):
        for j,name in enumerate(names):
            x=df.loc[name,metric];c=COLORS.get(name,'#A6ACB4')
            ax.plot([0,x],[j,j],color=c,alpha=.30,lw=2)
            ax.scatter(x,j,s=65,color=c,zorder=3)
            ax.annotate(f'{x:.5f}',(x,j),xytext=(8,0),textcoords='offset points',va='center',fontsize=11)
        ax.set_title(title,loc='left',pad=18);ax.set_xlabel('Outage fraction (lower is better)')
        ax.set_xlim(0,df[metric].max()*1.30);style(ax)
    axes[0].set_yticks(yy,[LABELS[n] for n in names]);axes[0].invert_yaxis()
    fig.subplots_adjust(left=.255,right=.98,bottom=.18,top=.87,wspace=.25)
    fig.text(.255,.04,'15 development systems | 1,633 county-events | 144-hour conditional hindcasts',fontsize=11,color='#555555')
    save(fig,'forecast_metrics')

    df=pd.read_csv(RESULTS/'per_system_metrics.csv')
    storms=pd.read_csv(HERE.parent/'data_v1/outcomes/development_systems.csv').query("regime=='tropical'").sort_values('origin')
    ids=list(storms.system);labels=[]
    for r in storms.itertuples():
        name=str(r.storm).split(' (')[0]
        if name.startswith('('):name='Unnamed '+r.system
        labels.append(name.title()+' '+str(r.origin)[:4])
    fig,axes=plt.subplots(1,2,figsize=(12.5,8.2),sharey=True)
    for ax,metric,title in zip(axes,['rmse','mae'],['a  Per-system RMSE','b  Per-system MAE']):
        for j,event in enumerate(ids):
            z=df[df.system==event].set_index('model')
            vals=[z.loc[n,metric] for n in COLORS]
            ax.plot([min(vals),max(vals)],[j,j],color='#C9CDD2',lw=1.2,zorder=1)
            for name,marker in zip(COLORS,['o','s','D']):
                ax.scatter(z.loc[name,metric],j,s=35,c=COLORS[name],marker=marker,zorder=3)
        ax.set_xscale('log');ax.set_xlabel('Outage fraction (log scale)');ax.set_title(title,loc='left',pad=17);style(ax)
    axes[0].set_yticks(range(len(ids)),labels);axes[0].invert_yaxis()
    handles=[plt.Line2D([],[],color=COLORS[n],marker=marker,ls='',markersize=7,label=LABELS[n]) for n,marker in zip(COLORS,['o','s','D'])]
    fig.legend(handles=handles,loc='lower center',ncol=3,bbox_to_anchor=(.55,.035),frameon=False,fontsize=11)
    fig.subplots_adjust(left=.235,right=.98,bottom=.14,top=.92,wspace=.25)
    save(fig,'storm_transfer')
    print('Saved two PDF/PNG figures from verified result tables')

if __name__=='__main__':main()
