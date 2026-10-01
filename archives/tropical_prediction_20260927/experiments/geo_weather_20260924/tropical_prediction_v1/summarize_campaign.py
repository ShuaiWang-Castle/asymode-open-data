"""Verify, score and export a completed campaign; never choose on outer scores."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from run_campaign import ROOT,HERE,OUT,DATA,MODELS,CODE_HASH

REPORT=HERE/'results'
LABELS={'zero':'Zero forecast','persistence':'Persistence','damped':'Damped persistence',
        'host':'Matched AsymODE host','fusion':'Joint GCRK-rate model',
        'tree_era5_l2':'ERA5 residual tree','tree_dual_l2':'Dual-weather residual tree',
        'tree_dual_l1':'Dual-weather MAE tree','blend':'Inner-selected ensemble'}

def metric(p,y,m,w):
    wm=m.astype(float)*w[:,None];den=wm.sum();e=p.astype(float)-y
    mse=float(np.sum(wm*e*e)/den)
    return dict(n=int(m.sum()),mse=mse,rmse=float(np.sqrt(mse)),mae=float(np.sum(wm*np.abs(e))/den))

def mdtable(df,columns):
    out=['| Model | '+' | '.join(columns)+' |','|---|'+'---:|'*len(columns)]
    for name in LABELS:
        row=df[df.model==name].iloc[0]
        out.append('| '+LABELS[name]+' | '+' | '.join(f'{row[c]:.5f}' for c in columns)+' |')
    return '\n'.join(out)

def main():
    REPORT.mkdir(exist_ok=True)
    a=dict(np.load(DATA));y=np.nan_to_num(a['target']).astype(float);m=a['target_observed'].astype(bool)
    n=len(y);names=MODELS+['blend'];pred={name:np.full_like(y,np.nan) for name in names}
    seeds={(name,s):np.full_like(y,np.nan) for name in ['host','fusion'] for s in range(3)}
    seen=np.zeros(n,int);weights=[];training=[]
    data_hash=hashlib.sha256(DATA.read_bytes()).hexdigest()
    for fold in range(1,6):
        p=OUT/f'fold{fold}';done=json.loads((p/'DONE.json').read_text())
        assert done['code']==CODE_HASH and done['data_sha256']==data_hash
        z=np.load(p/'outer_predictions.npz');idx=z['idx'];seen[idx]+=1
        assert np.all(a['outer_fold'][idx]==fold)
        sel=json.loads((p/'selection.json').read_text());ww=sel['weights']
        assert min(ww.values())>=0 and np.isclose(sum(ww.values()),1)
        for name in names:
            assert z[name].shape==(len(idx),144) and np.isfinite(z[name]).all()
            assert (z[name]>=0).all() and (z[name]<=1).all()
            pred[name][idx]=z[name]
        assert np.allclose(z['blend'],sum(ww[name]*z[name] for name in MODELS))
        for name in ['host','fusion']:
            for s in range(3):seeds[name,s][idx]=np.load(OUT/f'cache/f{fold}_outer_{name}_s{s}.npz')['P']
            assert np.allclose(z[name],np.mean([seeds[name,s][idx] for s in range(3)],0),atol=1e-7)
        for name,v in ww.items():weights.append(dict(fold=fold,model=name,weight=v))
        iz=np.load(p/'inner_predictions.npz');ii=iz['idx'];assert set(ii).isdisjoint(idx)
        ip=sum(ww[name]*iz[name] for name in MODELS)
        assert np.isclose(metric(ip,y[ii],m[ii],a['w'][ii])['mse'],sel['inner_mse'],rtol=1e-5)
    assert (seen==1).all() and all(np.isfinite(v).all() for v in pred.values())
    for file in (OUT/'cache').glob('*.json'):
        r=json.loads(file.read_text());assert r['identity']['code']==CODE_HASH
        training.append(dict(tag=file.stem,kind=r['identity']['kind'],seconds=r['seconds'],steps=r['identity'].get('steps')))
    training=pd.DataFrame(training)
    assert len(training)==145
    assert training['steps'].notna().sum()==70
    rows=[];point=[];events=[];strata=[]
    for wn,w in [('design',a['w']),('untrimmed',a['w_raw']),('unweighted',np.ones(n))]:
        for name,p in pred.items():rows.append(dict(model=name,weighting=wn,**metric(p,y,m,w)))
    for name,p in pred.items():
        for h in [1,6,24,48]:point.append(dict(model=name,horizon=h,**metric(p[:,h-1:h],y[:,h-1:h],m[:,h-1:h],a['w'])))
        for event in np.unique(a['system']):
            ii=a['system']==event;events.append(dict(model=name,system=event,**metric(p[ii],y[ii],m[ii],a['w'][ii])))
        for stratum in np.unique(a['stratum']):
            ii=a['stratum']==stratum;strata.append(dict(model=name,stratum=stratum,**metric(p[ii],y[ii],m[ii],a['w'][ii])))
    overall=pd.DataFrame(rows);points=pd.DataFrame(point);per_event=pd.DataFrame(events)
    pd.DataFrame(weights).to_csv(REPORT/'ensemble_weights.csv',index=False)
    training.to_csv(REPORT/'training_manifest.csv',index=False)
    overall.to_csv(REPORT/'overall_metrics.csv',index=False)
    points.to_csv(REPORT/'horizon_metrics.csv',index=False)
    per_event.to_csv(REPORT/'per_system_metrics.csv',index=False)
    pd.DataFrame(strata).to_csv(REPORT/'stratum_metrics.csv',index=False)
    seedrows=[]
    for (name,s),p in seeds.items():seedrows.append(dict(model=name,seed=s,**metric(p,y,m,a['w'])))
    pd.DataFrame(seedrows).to_csv(REPORT/'seed_metrics.csv',index=False)
    # Paired family resampling of fixed OOF predictions; not a refit bootstrap.
    fam=np.unique(a['family']);rng=np.random.default_rng(20260927)
    counts=rng.multinomial(len(fam),np.ones(len(fam))/len(fam),size=2000)
    sse={name:np.array([np.sum(a['w'][a['family']==f,None]*m[a['family']==f]*(p[a['family']==f]-y[a['family']==f])**2) for f in fam]) for name,p in pred.items()}
    sae={name:np.array([np.sum(a['w'][a['family']==f,None]*m[a['family']==f]*np.abs(p[a['family']==f]-y[a['family']==f])) for f in fam]) for name,p in pred.items()}
    comparisons=[]
    for name in ['fusion','tree_era5_l2','tree_dual_l2','tree_dual_l1','blend']:
        for base in ['host','zero','tree_dual_l2']:
            if name==base:continue
            gains=100*(1-np.sqrt((counts@sse[name])/(counts@sse[base])))
            mae_gains=100*(1-(counts@sae[name])/(counts@sae[base]))
            comparisons.append(dict(model=name,baseline=base,rmse_gain_pct=float(100*(1-np.sqrt(sse[name].sum()/sse[base].sum()))),
                lower95=float(np.quantile(gains,.025)),upper95=float(np.quantile(gains,.975)),
                mae_gain_pct=float(100*(1-sae[name].sum()/sae[base].sum())),
                mae_lower95=float(np.quantile(mae_gains,.025)),mae_upper95=float(np.quantile(mae_gains,.975)),
                systems_improved=int((sse[name]<sse[base]).sum()),systems=len(fam)))
    comp=pd.DataFrame(comparisons);comp.to_csv(REPORT/'paired_family_bootstrap.csv',index=False)
    audit=dict(county_events=n,systems=len(np.unique(a['system'])),families=len(fam),observed_cells=int(m.sum()),
        neural_fits=70,tree_fits=75,source_code_hash=CODE_HASH,data_sha256=data_hash,all_prediction_ranges_verified=True,
        every_outer_row_once=True,sealed_confirmation_read=False,bootstrap_draws=2000,bootstrap_seed=20260927)
    (REPORT/'evaluation_audit.json').write_text(json.dumps(audit,indent=2))
    design=overall[overall.weighting=='design'];by=design.set_index('model')
    gain=lambda name,base:100*(1-by.loc[name,'rmse']/by.loc[base,'rmse'])
    bc=comp[(comp.model=='blend')&(comp.baseline=='host')].iloc[0]
    fc=comp[(comp.model=='fusion')&(comp.baseline=='host')].iloc[0]
    report=['# 纯热带预测：15 系统五折完整实验','',
        '本次使用 data_v1 的全部热带开发系统，完成 70 次神经网络拟合、75 次树模型拟合。所有模型从公共数据重新拟合，外层每个神经模型均平均 seeds 0/1/2 的预测。',
        '', '## 主结果：完整 144 小时，设计加权 pooled 指标','',mdtable(design,['mae','rmse','mse']),
        '',f'内层选择的融合相对匹配 HOST 的 RMSE 变化：改善 {gain("blend","host"):.2f}%；相对全零：改善 {gain("blend","zero"):.2f}%。负数表示恶化。',
        f'融合对 HOST 的配对家族重采样 95% 区间为 [{bc.lower95:.2f}%, {bc.upper95:.2f}%]，{int(bc.systems_improved)}/15 个系统改善。',
        f'融合对 HOST 的 MAE 改善为 {100*(1-by.loc["blend","mae"]/by.loc["host","mae"]):.2f}%。',
        '',f'联合 GCRK＋速率模型相对 HOST：改善 {gain("fusion","host"):.2f}%。该对比同时改变地理核和适配器，不能单独归因于地理信息。',
        f'联合模型对 HOST 的 95% 区间为 [{fc.lower95:.2f}%, {fc.upper95:.2f}%]，{int(fc.systems_improved)}/15 个系统改善。区间跨零时不声称稳定正收益。',
        '', '## 点时效结果','']
    for h in [1,6,24,48]:report += [f'### +{h} h','',mdtable(points[points.horizon==h],['mae','rmse']),'']
    report+=['## 训练与模型设计','',
        '- HOST 复用仓库 AsymODE：32 单元损伤网络、16 单元恢复网络，恢复率逐小时更新。',
        '- 融合臂在相同初始化 HOST 上加入仓库的 GCRK，以及两个 16 单元零输出初始化速率头；全部参数联合优化。',
        '- 两个神经臂共享 ERA5/HRRR 危险度、缺失指示、时钟、历史停电摘要和六个县级背景变量；地理核另读取真实物理地理描述。',
        '- 原 HOST 完整 features_v1D.npz 未随分支提交，因此这是危险度输入上的新匹配对照，不是旧 11.5% MSE 数字的复现。',
        '- 每次神经拟合固定 900 次 Adam 更新、batch 128、损伤学习率 0.003、恢复 0.0003；每次重新拟合变换和校准缓冲。',
        '- 树模型共享全时效预测器，使用固定 300 棵树、15 叶；残差版本使用平方损失，直接版本使用绝对损失。',
        '- 外层训练集合内部四个完整事件折产生样本外预测；只据其设计加权 MSE 选择非负且和为 1 的融合权重。',
        '- 权重在整条 144 小时路径上固定，不按预测时效切换模型。外层预测和标签未参与权重拟合。',
        '', '## 验证范围与材料','',
        '15 个系统、1,633 个县—事件、234,874 个有效未来县—小时。事件外推而非县外推；同一县可以出现在其他风暴训练数据中。',
        '三种子的逐种子结果在 seed_metrics.csv；每场风暴及层级结果、未截断抽样权重敏感性、无权重结果均保留。',
        'paired_family_bootstrap.csv 给出 2,000 次配对家族重采样区间，针对固定外层预测，不包含重新训练和开发选择不确定性。',
        'ERA5 和滚动 HRRR 用作已知天气输入，属于条件回报预测。Milton 缺失的 59 个 HRRR 小时保留缺失指示，没有删除相应目标。',
        '客户分母及部分地理和背景字段使用仓库的既有公共快照，并非每场历史风暴起报时可获得的版本；SAIDI 2023 / SAIFI 2023 已排除。',
        '方向选择已看过开发集已有热带结果；本次仍是开发证据。封存 C 未读取，尚不能将这些结果称为独立确认或业务部署精度。',
        '内层神经模型用单种子、最终外层用三种子均值，两者方差不同，是本次计算预算下的明确近似。',
        '', '复现：先运行 prepare.py，再依次运行 run_campaign.py --fold 1 至 --fold 5，最后运行 summarize_campaign.py。建议并行不超过两折（8 GB 内存）。']
    report+=['','## 当前可支持的论文结论','',
        '这轮实验中，内层选择的集成取得最低的完整路径 RMSE；双天气源绝对损失树取得最低的 MAE。两者存在误差取舍，不能写成两项指标都由融合获胜。',
        '联合 GCRK＋速率头的 RMSE 和 MAE 均差于匹配 HOST，三个配对种子也都没有改善。因此本轮不能以“地理核稳定提升精度”为主张。',
        '融合对 HOST 的 RMSE 改善区间跨零，对双天气源残差树也跨零；相对强基线的优势尚未得到明确确认。',
        '短时效与完整路径的优胜顺序不同：近端持续性预测很强。本次没有按外层结果挑选时效或拼接一条新预测路径。',
        '这份成果适合支撑“气象条件下的热带停电预测”开发研究。封存确认和使用起报时实际可获得的气象预报进行评估，仍是后续独立环节。']
    (REPORT/'REPORT_zh.md').write_text('\n'.join(report)+'\n')
    # Paper-ready table source: nine columns, vertically centered Model.
    lines=[r'\begin{table*}[t]',r'\centering\small',
        r'\caption{Development event-held-out accuracy on 15 tropical systems. Design-weighted pooled MAE and RMSE of the public outage fraction; lower is better. Weather inputs are conditional hindcast information.}',
        r'\label{tab:tropical_forecasts}',r'\begin{tabular}{lcccccccc}',r'\toprule',
        r'\multirow{2}{*}{Model} & \multicolumn{4}{c}{MAE} & \multicolumn{4}{c}{RMSE} \\',
        r'\cmidrule(lr){2-5}\cmidrule(lr){6-9}',
        r'& $+1$ h & $+6$ h & $+24$ h & $+48$ h & $+1$ h & $+6$ h & $+24$ h & $+48$ h \\',r'\midrule']
    for name in names:
        r=points[points.model==name].set_index('horizon')
        lines.append(LABELS[name]+' & '+' & '.join(f'{r.loc[h,metric]:.5f}' for metric in ['mae','rmse'] for h in [1,6,24,48])+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table*}']
    (REPORT/'results_table.tex').write_text('\n'.join(lines)+'\n')
    print(design.to_string(index=False));print(comp[comp.model=='blend'].to_string(index=False))
    print('VERIFIED ALL 145 FITS AND FIVE OUTER FOLDS')

if __name__=='__main__':main()
