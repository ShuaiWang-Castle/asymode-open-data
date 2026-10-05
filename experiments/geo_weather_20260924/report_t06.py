"""Complete T06 report; every registered event, seed and geographic comparison."""
from t06_featured_subset import *
from report_t04 import table

LABEL={'DOSE':'无核AsymODE + 天气记忆','GCRK':'天气记忆 + 原GCRK','CRK':'天气记忆 + 完整CRK',
       'CRK_SHUFFLE':'完整CRK + 地理对应置换','GEO_MLP':'天气记忆 + 直接geo40输入'}
NAMES={'S00023':'ISAIAS','S00034':'MICHAEL','S00046':'DELTA','S00047':'ZETA'}

def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d,meta=load_data();p=dict(np.load(PRIVATE/'oof_ensembles.npz',allow_pickle=False))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    colors={'DOSE':'#89939d','GCRK':'#537fa9','CRK':'#b6464b','CRK_SHUFFLE':'#dca09c','GEO_MLP':'#3c927c'}
    for ax,event in zip(axs.flat,meta['events']):
        k=d['system']==event
        # Common observation support; county mean only, not a hand-picked county.
        ax.plot(np.arange(1,145),np.nanmean(np.where(d['m'][k],d['y'][k],np.nan),axis=0)*100,color='#222222',lw=2,label='Observed')
        for a in ARMS:
            ax.plot(np.arange(1,145),np.nanmean(np.where(d['m'][k],p[a][k],np.nan),axis=0)*100,
                    color=colors[a],lw=1.4,ls='--' if a=='CRK_SHUFFLE' else '-',label=a)
        ax.set_title(f'{NAMES[event]} | {int(k.sum())} county-windows',loc='left');ax.set_xlabel('Forecast hour');ax.set_ylabel('Customers out (%)')
        ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.15)
    handles,labels=axs.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=6)
    fig.suptitle('T06: all four selected windows | county means on observed support',fontsize=15)
    fig.savefig(OUT/'all_window_trajectories.png',dpi=160);plt.close(fig)

def report():
    meta=json.loads((OUT/'build_audit.json').read_text());v=json.loads((OUT/'verification.json').read_text())
    audit=json.loads((OUT/'independent_audit.json').read_text())
    m=pd.read_csv(OUT/'metrics.csv');c=pd.read_csv(OUT/'comparisons.csv');s=pd.read_csv(OUT/'per_seed.csv');e=pd.read_csv(OUT/'per_event.csv')
    tails=pd.read_csv(OUT/'tail_diagnostics.csv');sens=pd.read_csv(OUT/'unique_county_hour_sensitivity.csv');leave=pd.read_csv(OUT/'leave_one_window.csv')
    training=json.loads((OUT/'training_audit.json').read_text())
    full=m.query("group=='all' and weighting=='design' and scope=='full144'").copy();full['模型']=full.model.map(LABEL)
    comp=c[c.group=='all'].copy();comp['模型']=comp.model.map(LABEL);comp['对照']=comp.reference.map(LABEL)
    ep=e.pivot(index='system',columns='model',values='rmse');sp=s[s.group=='all'].pivot(index='seed',columns='model',values='rmse')
    lines=[]
    for _,r in comp.iterrows():
        improvement='改善' if r.rmse_gain_pct>0 else '恶化'
        interval='区间支持改善' if r.ci025_pct>0 else ('区间支持恶化' if r.ci975_pct<0 else '区间跨零')
        lines.append(f'- **{LABEL[r.model]} 相对 {LABEL[r.reference]}：{improvement}{abs(r.rmse_gain_pct):.2f}%**，县簇95%区间[{r.ci025_pct:.2f}%, {r.ci975_pct:.2f}%]，{interval}；逐seed胜出{int((sp[r.model]<sp[r.reference]).sum())}/3，逐窗口胜出{int((ep[r.model]<ep[r.reference]).sum())}/4。')
    from score_t06 import PAIRS
    sg=sp.copy();eg=ep.copy()
    for a,b in PAIRS:
        sg[a+'_vs_'+b]=100*(1-sp[a]/sp[b]);eg[a+'_vs_'+b]=100*(1-ep[a]/ep[b])
    cols=[a+'_vs_'+b for a,b in PAIRS];sg=sg[cols].reset_index();eg=eg[cols].reset_index();eg['storm']=eg.system.map(NAMES)
    registration=pd.read_csv(OUT/'dataset_selection.csv').query('selected_t06')
    checks=json.loads((OUT/'implementation_checks.json').read_text())
    text=f'''# T06：强风与地理异质性子集上的完整复杂核验证

2026-10-04 America/New_York。T05完成后按用户要求执行第二条线。完成{v['neural_fits']}/75个拟合：5臂×5个全局县折×seeds0/1/2，每次600步、INNER县选择checkpoint。全部在选定子集从头训练，未复用旧权重，未简化地理核。

## 主结果

{chr(10).join(lines)}

{table(full[['模型','mae','rmse']])}

主指标为全部所选县、完整144小时、原设计权的pooled MAE/RMSE；先平均三seed预测，另报单seed。误差单位为停电比例，相对改善单位为%；不平均fold RMSE，不将其与比赛OSI/后缀评分混比。

{table(comp[['模型','对照','rmse_gain_pct','ci025_pct','ci975_pct']])}

## 数据范围与保留的复杂结构

使用已有15个热带开发窗口的T04审查表，仅按数值覆盖、强风与地形异质性筛选：原质量门槛通过，ERA5县阵风峰值90分位>=25m/s、地形起伏县际IQR>=80m。选择规则在T06训练前登记；四个窗口、{meta['n']}县窗口、{meta['counties']}不同县，保留其全部县。阈值为探索性数据范围定义，不声称外部物理损伤标准；之前看过此开发数据，不是独立确认。

{table(registration[['system','storm','county_events','era5_g_mean_peak_p90','relief_p95_p5_iqr','observed_fraction','hrrr_available']])}

CRK是已有I20实现的完整受控响应核，在本次子集重新训练：geo40直接通路、32地标全子集阶数核基、4×8记忆状态、每模式4个Cayley转换平面、天气/地理写入保留转换读出，新增15209参数；它位于唯一损伤MLP内部，恢复网络/stock更新保留。此次改变数据范围和训练任务，不声称新发明，也不把该结果与I20事件留出900步成绩当同协议比较。

CRK_SHUFFLE保留完全相同架构，在每折FIT/INNER/OUTER集合分别按州内固定置换地理；同县跨窗口映射固定，单县州组不变。置换可能产生不自然天气/地理对应，不能单独证明因果。不同模型信息和容量不匹配，CRK超过DOSE本身不够证明地理净贡献；须同时看置换、直接接入和原核。

| 模型 | 总参数量 |
|---|---:|
{chr(10).join('| '+a+' | '+str(checks[a]['parameter_count'])+' |' for a in ARMS)}

## 所有种子与窗口

各配对完整144h RMSE改善%，正值更好：

{table(sg)}

{table(eg)}

排除单个窗口后的敏感性（仅诊断，不按结果删窗口）：

{table(leave)}

所有四个窗口的共同观测支持县均值轨迹见all_window_trajectories.png；县均值不能替代县级峰值评价。

## 严重停电、漏峰与假峰

severe=预测窗观测峰>=10%；jump=相邻共同观测最大增量>=1pp。子组由结果定义，仅用于诊断，不参与数据选择或预测时模型选择。zero=观测支持全零，nonsevere=观测峰<10%；部分缺测不能解释为全窗确定无停电。

{table(tails[['model','group','county_events','peak_mae','true_peak_time_mae','full_window_peak_ratio_median','false_peak_count','false_peak_weighted_rate']].astype(object).where(pd.notna(tails[['model','group','county_events','peak_mae','true_peak_time_mae','full_window_peak_ratio_median','false_peak_count','false_peak_weighted_rate']]),'—'))}

峰值幅度比为无权中位数；假峰率为县窗口设计加权率。全零子组阈值>0.1%，非严重子组阈值>=10%；它们不是同一种报警。severe/jump整体误差、固定+1/+6/+24/+48准确时点和无权结果完整保存在metrics.csv；支持数见verification.json。

## 重复县小时与验证

有效县窗口小时{v['valid_event_county_hours']}；不同县小时{v['unique_county_hours']}；额外重复{v['duplicate_extra_county_hours']}。以下将每个起点的误差权重除以同县同UTC小时的重复次数；unique_equal为独立县小时等权，design_divide_multiplicity保留原设计权再除重复次数。不平均预测，不将后起点信息送入前起点模型。

{table(sens)}

- 五个fold0/seed0 checkpoint回放最大误差均为0；OOF县全局分折且每县窗口恰好覆盖一次，FIT标准化、INNER选checkpoint。
- 配对初始预测最大误差{v['paired_initial_max_error']:.2e}；各折SSE/SAE重构主指标最大误差{v['pooled_reconstruction_max_error']:.2e}。
- 独立算术审查{audit['checks']}项通过，主指标最大误差{audit['headline_arithmetic_max_error']:.2e}；独立tuple-key分组核对重复县小时。
- 当前环境下已有复杂核/精确伴随/核代数测试32项通过；T06接入预检验证五臂配对和有限梯度。环境与源码/缓存hash在build_audit.json，全部训练曲线/参数变化在training_audit.json。
- 2000次县簇区间条件于已拟合模型及固定四窗口/划分，不涵盖初始化重训、风暴抽样、范围选择或多重比较。三seed只能提供有限稳定性证据。
- 未来ERA5/HRRR为给定天气回溯输入，不能称起点可获得的实时六天预报；未接触封存C、main或论文。

本轮无自动扩seed、范围重选或候选专属延长训练。若后续继续，须依据完整正负证据另记具体假设与预算。
'''
    (HERE/'notes/T06_FEATURED_SUBSET_RESULTS_20261004_ZH.md').write_text(text)
    print('\n'.join(lines),flush=True)

if __name__=='__main__':plot();report()
