"""Produce the T04 factual report from verified aggregate results."""
from pathlib import Path
import json
import pandas as pd

HERE=Path(__file__).resolve().parent
OUT=HERE/'results/t04'
LABEL={'HOST':'原AsymODE HOST','DOSE':'AsymODE + 原48维天气记忆','GCRK':'天气记忆 + 原复杂GCRK',
       'GEO_MLP':'天气记忆 + 直接地理输入','SHARED':'简单记忆：全县共享',
       'GEO_GAIN':'简单记忆：地理调幅度','GEO_MIX':'简单记忆：地理调快慢比例'}

def table(df):
    def fmt(x):
        return f'{x:.5f}' if isinstance(x,float) else str(x).replace('|','/').replace('\n',' ')
    lines=['| '+' | '.join(map(str,df.columns))+' |','| '+' | '.join(['---']*len(df.columns))+' |']
    lines.extend('| '+' | '.join(fmt(x) for x in row)+' |' for row in df.itertuples(index=False,name=None))
    return '\n'.join(lines)

def main():
    m=pd.read_csv(OUT/'metrics.csv');c=pd.read_csv(OUT/'comparisons.csv')
    s=pd.read_csv(OUT/'per_seed.csv');e=pd.read_csv(OUT/'per_event.csv')
    audit=pd.read_csv(OUT/'dataset_scope_audit.csv');v=json.loads((OUT/'verification.json').read_text())
    training=json.loads((OUT/'training_audit.json').read_text())
    main=m.query("group=='all' and weighting=='design' and scope=='full144'").copy()
    main['模型']=main.model.map(LABEL)
    primary=c.query("group=='all' and reference=='SHARED'").copy();primary['模型']=primary.model.map(LABEL)
    seed=s.query("group=='all'").pivot(index='seed',columns='model',values='rmse')
    seed_table=pd.DataFrame({'seed':seed.index,'地理幅度相对共享改善%':100*(1-seed.GEO_GAIN/seed.SHARED),
                             '地理快慢比例相对共享改善%':100*(1-seed.GEO_MIX/seed.SHARED)})
    subgroup=m.query("weighting=='design' and scope=='full144'").pivot(index='group',columns='model',values='rmse')
    event=e.pivot(index='system',columns='model',values='rmse')
    wins={arm:int((event[arm]<event.SHARED).sum()) for arm in ('GEO_GAIN','GEO_MIX')}
    h=m.query("group=='all' and weighting=='design' and scope!='full144'").copy()
    h['模型']=h.model.map(LABEL)
    verdict=[]
    for _,r in primary.iterrows():
        direction='改善' if r.rmse_gain_pct>0 else '恶化'
        sig='区间跨0，未建立稳定增益' if r.ci025_pct<=0<=r.ci975_pct else ('区间支持改善' if r.ci025_pct>0 else '区间支持退化')
        verdict.append(f"- **{r['模型']}**相对共享记忆{direction} **{abs(r.rmse_gain_pct):.2f}%**；描述性县簇95%区间[{r.ci025_pct:.2f}%, {r.ci975_pct:.2f}%]，{sig}；15个窗口中{wins[r.model]}个改善。")
    subset=subgroup[['SHARED','GEO_GAIN','GEO_MIX']].reset_index()
    profile=audit[['storm','county_events','observed_fraction','hrrr_available','quality_candidate','exclusion_reason']].fillna('')
    text=f'''# T04：简单地理天气记忆核首轮结果

2026-10-04 UTC。分支research/tropical-evidence-20260926。本轮按照先推送的方案完成{len(training)}/45个拟合：三个模型×seeds 0/1/2×原全局县级五折，各600步，INNER选择checkpoint。全部15个热带开发窗口保留；不是跨事件泛化评估。

## 结论

{chr(10).join(verdict)}

以下为三seed预测先平均、再计算完整144小时设计加权pooled指标。原模型结果在同一缓存上重新组装，浮点累计可能与T03最后几位有差异。

{table(main[['模型','mae','rmse']])}

这次没有LightGBM，也没有多架构BLEND；只比较AsymODE内部的天气记忆设计。新增地理贡献必须比较GEO_GAIN/GEO_MIX与SHARED，不能仅凭超过HOST就归功于地理。

## 具体改动

- 保留原AsymODE损伤形成网络、逐小时恢复网络、损伤logit平滑器、发生门和停电状态递推；所有模型参数共同训练。
- 删除原GCRK的32维隐藏响应递推、方向交互、地理编码和动态校准。
- 用固定0.9/0.98两个归一化EWMA的凸组合表示快慢天气记忆，四类通道各自一个混合比例alpha、一个正幅度gain；两套天气源共享参数。
- SHARED只有8个新增参数；GEO_GAIN/GEO_MIX各24个。地理只用树冠覆盖、地形起伏、排水不良比例、开发用地比例；FIT标准化后tanh压缩。alpha在(0,1)，gain在(0.5,2)，地理系数固定L2惩罚1e-4。
- 组合在原始天气单位进行，然后用FIT的原DOSE标准化常数处理。两个原EWMA位置输入相同组合值，其他滚动均值/最大值保留。这是对显式天气记忆的简化，SHARED与原DOSE并不等价。
- 同一seed的初始预测和训练批次配对；主网络/记忆学习率0.003，恢复网络0.0003；没有OUTER调参。原T03直接地理MLP使用40维地理，与新核4维地理不是容量/信息完全匹配的对照；本轮先回答小地理机制相对共享机制的问题。

## 逐seed稳定性

正值表示地理改善；未挑选种子。

{table(seed_table)}

完整比较和县簇区间见comparisons.csv。区间用2000次县簇重采样，同县跨窗口整体保留，仅条件于本次模型和划分；未包含重训练、数据选择和多重比较的不确定性。

## 数据范围仍是后续重点

评分前已固定质量门槛：县数至少50、全窗停电观测覆盖至少99%、两套天气各有效至少95%、四项地理有效至少99%。12/15窗口通过，仅表示数值覆盖合格。ALBERTO与一个无最佳路径窗口的事件身份/时间范围仍需核验，不能因此称为12个已确认飓风。

{table(profile)}

敏感性分析使用同一批在全范围训练的模型，没有在子集重训，不能等同于“限定该数据集训练后”的结果：

{table(subset)}

all=全15窗口；quality_candidate=12个覆盖候选；complete_observation=全216小时标签完整的县事件；quality_complete=两者交集。每组具体样本量见verification.json。

基于天气/地理特征而非本轮模型胜负，ISAIAS、MICHAEL、DELTA、ZETA值得优先核对事件时间窗和源数据：均140县、天气完整、风暴暴露较强且县际地理变化明显。MILTON应先补齐/核实59个缺失HRRR小时（当前有效72.7%）；ALEX和无最佳路径的小窗口县数不足50。没有依据OUTER胜负把它们从主结果中删掉。

## 四个准确预测时点

本表是固定起点后的准确+1/+6/+24/+48小时，不是比赛的后缀汇总；目标为停电比例，不是比赛OSI。

{table(h[['模型','scope','mae','rmse']])}

## 核验与复现

OOF县恰好覆盖一次；全局县级划分避免同县跨事件落入不同角色。训练前配对最大误差{v['paired_initial_max_error']:.1e}；fold0/seed0三个保存checkpoint回放最大误差均为0；由各折SSE/SAE重构pooled指标的最大误差{v['pooled_reconstruction_max_error']:.2e}。

训练曲线、参数变化幅度、外层alpha/gain范围和少量记忆系数记录在training_audit.json；无原始数据、逐县预测或完整模型权重上传。详细文件：metrics.csv、per_seed.csv、per_event.csv、comparisons.csv、dataset_scope_audit.csv、verification.json。

仍属于已查看开发数据、给定未来天气条件的回溯评估，不能称为独立确认或实时六天预报。复杂度导致原核表现欠佳是待验证解释；本轮结果不应被扩大为对所有地理信息的否定或肯定。
'''
    (HERE/'notes/T04_SIMPLE_GEOGRAPHY_RESULTS_20261004_ZH.md').write_text(text)
    (OUT/'REPRODUCE.md').write_text('''# T04 reproduction

Use the same environment and authorized local data/cache as T03. No downloads or training data are bundled here.

```bash
python experiments/geo_weather_20260924/t04_simple_geography.py audit
python experiments/geo_weather_20260924/t04_simple_geography.py check
python experiments/geo_weather_20260924/run_t04.py
python experiments/geo_weather_20260924/score_t04.py
python experiments/geo_weather_20260924/report_t04.py
```

Audit/scope rules were committed before fitting. Defaults read the sibling t03_artifacts/data.npz; T03_ARTIFACTS can override that input path. Private T04 checkpoints/predictions/logs are written to sibling t04_artifacts. Complete jobs are resumed. Reproducing the original run requires the cache SHA in verification.json and the same T03 source/dependencies. All trained arms, seeds and predeclared scopes are retained.
''')
    print('\n'.join(verdict))

if __name__=='__main__':main()
