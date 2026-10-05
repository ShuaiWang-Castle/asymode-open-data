"""Report all T05 results plus weather/outage profiles; no cherry-picked winning events."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from report_t04 import table
HERE=Path(__file__).resolve().parent;OUT=HERE/'results/t05';PRIVATE=HERE.parents[1].parent/'t05_artifacts'
LABEL={'HOST':'AsymODE HOST','BASE_GCRK':'HOST + 原GCRK','DOSE':'HOST + 显式天气记忆','DOSE_GCRK':'天气记忆 + 原GCRK',
       'SHARED':'共享简单记忆','GEO_GAIN':'简单地理幅度核','GEO_MIX':'简单地理快慢核'}

def plot_profiles():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from t05_screen import tailmean
    d=dict(np.load(PRIVATE/'data.npz'));meta=json.loads((OUT/'build_audit.json').read_text())
    screen=pd.read_csv(OUT/'weather_window_screen.csv')
    chosen=screen[screen.selected].sort_values(['cross_cutoff_count','system'],ascending=[False,True]).head(4)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(4,2,figsize=(14,11),layout='constrained')
    for row,(_,r) in enumerate(chosen.iterrows()):
        sel=d['system']==r.system;orig=np.load(HERE/'data_v1/outcomes/panel_v1D_outcomes.npz')
        y=orig['y'][d['original_index'][sel]].astype(float);obs=orig['observed'][d['original_index'][sel]]
        curve=np.nanmean(np.where(obs,y,np.nan),axis=0)*100
        hs=[tailmean(d['base'][sel,:,meta['base_names'].index('era5:'+c)])/v for c,v in [('g_mean',18),('rain_mean',2),('snow_mean',1),('frz_mean',.5)]]
        intensity=np.nanmax(np.stack(hs),axis=0);x=pd.date_range(r.start,periods=216,freq='h')
        q=np.nanquantile(intensity,[.25,.5,.75],axis=0)
        a,b=axes[row];a.fill_between(x,q[0],q[2],color='#5b91b0',alpha=.2)
        a.plot(x,q[1],color='#3b789b',lw=1.8);a.axhline(1,color='#aaaaaa',ls=':',lw=1)
        b.plot(x,curve,color='#b45058',lw=1.8)
        a.set_ylabel('Weather index');b.set_ylabel('Customers out (%)')
        a.set_title(f'{r.system} | {r.start[:10]} | {int(r.n)} counties',loc='left',fontsize=12)
        b.set_title(f'{int(r.cross_cutoff_count)} counties: peaks straddle cutoff',loc='left',fontsize=12)
        for ax in (a,b):
            ax.axvline(x[71],color='#555555',ls='--',lw=1.2);ax.set_ylim(bottom=0)
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=2));ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'));ax.grid(axis='y',alpha=.15)
    fig.suptitle('Repeated weather exposure and observed outage trajectories',fontsize=18)
    fig.supxlabel('UTC | Dashed line: last observed hour 71. Weather: county median/IQR; outages: county mean.',fontsize=12)
    fig.savefig(OUT/'repeated_weather_profiles.png',dpi=160);plt.close(fig)

def report():
    m=pd.read_csv(OUT/'metrics.csv');c=pd.read_csv(OUT/'comparisons.csv');s=pd.read_csv(OUT/'per_seed.csv');e=pd.read_csv(OUT/'per_event.csv')
    v=json.loads((OUT/'verification.json').read_text());screen=pd.read_csv(OUT/'weather_window_screen.csv');selected=screen[screen.selected].sort_values('start')
    desc=pd.read_csv(OUT/'selected_outage_description.csv');t=json.loads((OUT/'training_audit.json').read_text())
    full=m.query("group=='all' and weighting=='design' and scope=='full144'").copy();full['模型']=full.model.map(LABEL)
    comp=c.query("group=='all'").copy();comp['模型']=comp.model.map(LABEL);comp['对照']=comp.reference.map(LABEL)
    ep=e.pivot(index='system',columns='model',values='rmse');sp=s.query("group=='all'").pivot(index='seed',columns='model',values='rmse')
    pairs=[('BASE_GCRK','HOST'),('DOSE_GCRK','DOSE'),('GEO_GAIN','SHARED'),('GEO_MIX','SHARED')]
    lines=[]
    for arm,base in pairs:
        r=comp[(comp.model==arm)&(comp.reference==base)].iloc[0]
        sign='改善' if r.rmse_gain_pct>0 else '恶化';wins=int((ep[arm]<ep[base]).sum());seedwins=int((sp[arm]<sp[base]).sum())
        lines.append(f'- **{LABEL[arm]} 对 {LABEL[base]}：{sign}{abs(r.rmse_gain_pct):.2f}%**，县簇95%区间[{r.ci025_pct:.2f}%, {r.ci975_pct:.2f}%]；窗口胜出{wins}/8、单seed胜出{seedwins}/3。')
    evt=ep.copy()
    for arm,base in pairs:evt[arm+'_gain_pct']=100*(1-ep[arm]/ep[base])
    evt=evt[[a+'_gain_pct' for a,b in pairs]].reset_index().merge(selected[['system','start']],on='system')
    seedtable=sp.copy()
    for arm,base in pairs:seedtable[arm+'_gain_pct']=100*(1-sp[arm]/sp[base])
    seedtable=seedtable[[a+'_gain_pct' for a,b in pairs]].reset_index()
    cross=c[(c.group=='cross_cutoff') & c.model.isin([a for a,b in pairs])]
    duplicate_design=pd.read_csv(OUT/'design_multiplicity_comparisons.csv')
    text=f'''# T05：重复天气冲击公开数据上的地理核检验

2026-10-04。用户要求沿更像比赛的多次风暴方向寻找数据并训练。完成{len(t)}/105个拟合（7臂×5折×3 seeds），所有模型在新筛选范围从头训练，未复用T03/T04权重。

## 结果

{chr(10).join(lines)}

主结果是完整144小时设计加权pooled指标，不是折RMSE平均；神经模型先平均三个seed预测。无LightGBM或跨模型BLEND。

{table(full[['模型','mae','rmse']])}

### 地理的配对独立贡献

{table(comp[['模型','对照','rmse_gain_pct','ci025_pct','ci975_pct']])}

这次同时检验了原核在有/无显式天气记忆时的效果。DOSE仍为原四类风雨/湿风通道的48维摘要，雪/冰天气及其累积载荷仍在基础输入；没有为新范围另调输入或超参数。BASE_GCRK/DOSE_GCRK使用40维地理的原核；简单核使用4维地理。不同核的信息量和容量并不匹配，不能单凭它们的排序认定某一结构本质更优。

T04热带范围中，原核在DOSE基础上恶化1.32%，两个简单核相对SHARED分别恶化0.12%/0.36%。即使T05出现较好结果，也不能仅凭跨数据集差异认定“多轮冲击导致核收益”：两轮试验的季节、县、事件、训练分布都变化了。此处是机制相关数据范围的探索，重复/非重复冲击分层用于进一步检查，尚非因果归因。

## 找到的范围和与比赛的关系

对已有81个开发窗口作天气/覆盖筛选，共{int(screen.eligible.sum())}个合格，按预定重复冲击占比取前8。所选范围包含1024县窗口、856个不同县；不是8次独立风暴，也不是原15个热带窗口的重评分。

天气指标为3小时尾随均值的max(阵风/18m/s、雨/2mm/h、雪/1mm/h、冻雨/0.5mm/h)。两个峰至少24小时、至多120小时分隔，峰高>=1，prominence>=0.25，中间有至少25%的回落；第二峰在预测期。窗口需至少50县、>=20县且>=20%县双峰、天气覆盖各>=95%、停电覆盖>=99%。没有用未来停电幅度或核的收益筛选。

{table(selected[['system','regime','start','n','repeated_count','cross_cutoff_count']])}

所有所选窗口的县都用于训练/评分，不仅限于有双峰的县。跨cutoff统计是所选最强峰对中的首峰在历史段、次峰在预测段；很多其他峰对的两峰都在未来，不能都称为“历史一轮、未来一轮”。

选定后计算的停电描述（不参与筛选）：第二峰前6小时平均停电>=0.1%为仍有停电；第二峰后24小时内峰值较前6小时均值上涨>=1个百分点为再次上涨。它们可能有其他原因，不能直接当作天气因果标签。

{table(desc)}

天气及停电剖面见repeated_weather_profiles.png：按跨cutoff县数最多的4个窗口展示，依据天气选图，不依据模型胜负。

### 最贴近比赛的具名真实案例

1. **2018年3月2日/7日，新泽西Riley/Quinn。** NJBPU报告第二轮来临时首轮仍约29,000客户未恢复，风和湿雪造成树木/电线损伤。这是很合适的新数据构建目标，但早于现有HRRR-v3框架的2018-07起点；本轮没有训练该案例。来源：https://www.nj.gov/bpu/pdf/reports/20180725-6A-winter-storm-report.pdf 。
2. **2024年1月9日/12日的连续冬季风暴。** NWS确认两轮强低压及中间弱降雪；现有S01263覆盖1月6–15日、样本遍布美国东部，不能等同于威斯康星专项数据。现有S01262只到1月12日，不能声称覆盖完整的第二轮。来源：https://www.weather.gov/mkx/EarlyJanuary2024_Recap 。
3. **数据源。** EAGLE-I有2014–2022年县级15分钟公开历史记录，可用于重建2018案例；仍须核对当年县覆盖、客户分母和逐小时天气，不将可获得性当作已完成下载。https://impact.ornl.gov/en/datasets/eagle-i-power-outage-data-2014-2022/ 。

## 稳定性与更严格的相似场景

逐seed配对RMSE改善百分比，正值更好：

{table(seedtable)}

逐窗口配对改善，全部正负结果保留：

{table(evt)}

只在“首峰已观测、次峰在预测段”的{v['groups']['cross_cutoff']['county_events']}个县窗口评分：

{table(cross[['model','reference','rmse_gain_pct','ci025_pct','ci975_pct']])}

重复天气县/非重复县、无权重以及+1/+6/+24/+48准确预测时点MAE/RMSE均在metrics.csv。本任务目标是停电比例，不能与比赛OSI直接比较；四个时点也不是比赛后缀汇总。

## 评估边界和复现

- 所有county全局分折，同县跨窗口始终同一角色；无跨事件留出要求。FIT标准化、INNER选checkpoint，OUTER不选超参数。保留原抽样设计权重仅作这一开发范围的加权评分，不声称筛选后仍代表全国风暴总体。
- 有效县窗口小时{v['valid_event_county_hours']}，不同县小时{v['unique_county_hours']}，额外重复{v['duplicate_extra_county_hours']}。unique_county_hour_sensitivity.csv按每个县小时的重复次数除权，并平均各起点的误差；不平均预测，不把后起点信息送入前起点模型。
- 上述unique评分还将原设计权改为县小时等权，不能仅把其变化归因于重复除权。另补充design_multiplicity_sensitivity.csv：保留原设计权，再除同县同小时的出现次数，单独检验重复评分因素。其配对改善和同县簇区间如下。

{table(duplicate_design)}

- 描述性区间为2000次县簇重采样，同县所有窗口整体抽取，条件于已拟合模型/划分，未涵盖重训练、数据选择和多重比较。一个窗口或seed的最好结果不作整体结论。
- 保存checkpoint回放7/7完全一致；pooled重构误差{v['pooled_reconstruction_max_error']:.2e}；配对初始预测最大误差{v['paired_initial_max_error']:.2e}。
- 仍为已查看开发数据上的给定天气回溯实验，ERA5/未来HRRR不代表起点可获得的六天天气预报。多峰阈值是探索性代理，需要与官方风暴记录核对；不能将其等同于独立气象系统计数。

复现：依次运行t05_screen.py、t05_repeated_weather.py build、run_t05.py、score_t05.py、report_t05.py。私有缓存、逐县预测、权重放在仓库旁t05_artifacts；只上传代码、协议、图和汇总结果。输入SHA、选择规则、完整训练曲线和核验见selection.json、build_audit.json、training_audit.json、verification.json。
'''
    (HERE/'notes/T05_REPEATED_WEATHER_RESULTS_20261004_ZH.md').write_text(text)
    print('\n'.join(lines))

if __name__=='__main__':
    import sys
    if '--profiles-only' in sys.argv:plot_profiles()
    else:plot_profiles();report()
