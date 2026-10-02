"""Deterministic aggregate-only report; no new fits or selection of favorable splits."""
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE/'results/t02'


def main():
    audit = json.loads((OUT/'audit.json').read_text())
    c = pd.read_csv(OUT/'comparisons.csv')
    f = pd.read_csv(OUT/'fold_diagnostics.csv')
    m = pd.read_csv(OUT/'metrics.csv')
    memberships = json.loads((OUT/'county_splits.json').read_text())
    assert len(c)==81 and len(m)==405 and len(f)==405 and audit['fits']==20250
    assert audit['source_sha256']==hashlib.sha256((HERE/'t02_split_stability.py').read_bytes()).hexdigest()
    assert audit['protocol_sha256']==hashlib.sha256((HERE/'notes/T02_SPLIT_STABILITY_PROTOCOL_20261002.md').read_bytes()).hexdigest()
    reconstruction_errors=[]
    for key,group in f.groupby(['system','sample','method','split']):
        selected=m[(m.system==key[0])&(m['sample']==key[1])&(m.method==key[2])&(m.split==key[3])]
        assert len(group)==5 and set(group.fold)==set(range(5))
        assert int(group.n_test.sum())==int(selected.n.iloc[0])
        for arm in ('B','G'):
            r=selected[selected.arm==arm].iloc[0]
            reconstruction_errors.extend([abs(np.sqrt(np.average(group[arm.lower()+'_rmse']**2,weights=group.test_weight))-r.rmse),
                abs(np.average(group[arm.lower()+'_mae'],weights=group.test_weight)-r.mae)])
    assert max(reconstruction_errors)<1e-12
    verification=dict(source_and_protocol_hashes_match=True,all_configuration_counts_match=True,
        pooled_reconstruction_max_absolute_error=float(max(reconstruction_errors)),
        replayed_t01_metrics=audit['replayed_t01_metrics'],t01_replay_max_absolute_error=audit['replay_max_absolute_error'])
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    rows = []
    for (system,storm,sample,method), group in c.groupby(['system','storm','sample','method'],sort=False):
        signatures = set()
        for split in group.split:
            labels = memberships[f'{system}/{sample}/{method}/{split}']
            signatures.add(tuple(sorted(tuple(sorted(k for k,v in labels.items() if v==fold)) for fold in set(labels.values()))))
        row = dict(system=system,storm=storm,sample=sample,method=method,n=int(group.n.iloc[0]),
            configurations=len(group),distinct_outer_partitions=len(signatures),
            median_gain_vs_b_pct=group.gain_vs_b_pct.median(),min_gain_vs_b_pct=group.gain_vs_b_pct.min(),max_gain_vs_b_pct=group.gain_vs_b_pct.max(),
            median_gain_vs_p_pct=group.gain_vs_p_pct.median(),wins_vs_b=int((group.gain_vs_b_pct>0).sum()),
            wins_vs_p=int((group.gain_vs_p_pct>0).sum()),joint_wins=int(((group.gain_vs_b_pct>0)&(group.gain_vs_p_pct>0)).sum()))
        required = 4 if method=='random' else 3
        row['scheduling_rule_pass'] = bool(row['joint_wins']>=required and row['median_gain_vs_b_pct']>0 and row['median_gain_vs_p_pct']>0)
        rows.append(row)
    s = pd.DataFrame(rows)
    s.to_csv(OUT/'stability_summary.csv',index=False)
    decisions = []
    for (system,storm), group in s.groupby(['system','storm'],sort=False):
        decisions.append(dict(system=system,storm=storm,passes_all_required_screens=bool(group.scheduling_rule_pass.all())))
    (OUT/'scheduling_decisions.json').write_text(json.dumps(decisions,indent=2)+'\n')
    lines = [
        '# T02：同事件县留出的划分稳定性验证',
        '', '2026-10-02。承接T01；六个候选事件固定，不按本轮结果追加划分。跨事件泛化不是必选目标。',
        '', '## 本轮做了什么',
        '', f'- 完成{audit["configurations"]}个事件/样本/划分组合，{audit["fits"]:,}次Ridge拟合；每个组合5个外折、3个内折、3个alpha、5个输入臂。',
        '- 六事件各有五套随机县划分、四套空间分块。MICHAEL、DELTA、ZETA另做完整144小时观测县的全套敏感性检查。',
        '- B：天气+前72小时停电摘要；G：B+geo40；P0/P1/P2：B+三套打乱地理。先逐折产生OOF预测，再按设计权pooled计算RMSE/MAE，不平均折RMSE。',
        '- 预测目标仍是未来144小时的已观测峰值，且输入包含未来ERA5。这是开发集条件hindcast探针，不是业务预报，也不是完整轨迹AsymODE/GCRK实验。',
        '', '## 主样本结果',
        '', '改善为正，恶化为负；中位数及范围来自划分配置，不是置信区间。P列为各配置三套置换对照RMSE的中位数，再计算G相对它的收益；没有做置换预测集成。',
        '', '|事件|划分|相对B改善中位数|最小～最大|相对P改善中位数|同时胜B/P|不同外层分区数|',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in s[s['sample']=='primary'].itertuples():
        lines.append(f'|{r.storm}|{"随机" if r.method=="random" else "空间"}|{r.median_gain_vs_b_pct:+.2f}%|{r.min_gain_vs_b_pct:+.2f}%～{r.max_gain_vs_b_pct:+.2f}%|{r.median_gain_vs_p_pct:+.2f}%|{r.joint_wins}/{r.configurations}|{r.distinct_outer_partitions}|')
    lines += ['', '## 完整观测县敏感性', '', '这里只保留未来144个小时均有观测的县，重新生成划分、训练和验证。因此变化同时包含样本删除及重划分影响，不能把它全部解释成缺失观测的影响。',
        '', '|事件|县数|划分|相对B改善中位数|最小～最大|相对P改善中位数|同时胜B/P|',
        '|---|---:|---|---:|---:|---:|---:|']
    for r in s[s['sample']=='complete_only'].itertuples():
        lines.append(f'|{r.storm}|{r.n}|{"随机" if r.method=="random" else "空间"}|{r.median_gain_vs_b_pct:+.2f}%|{r.min_gain_vs_b_pct:+.2f}%～{r.max_gain_vs_b_pct:+.2f}%|{r.median_gain_vs_p_pct:+.2f}%|{r.joint_wins}/{r.configurations}|')
    lines += ['', '## 原高提升是否集中', '']
    for system in ('S00034','S00037'):
        r=c[(c.system==system)&(c['sample']=='primary')&(c.method=='random')&(c.split==20261001)].iloc[0]
        lines.append(f'- {r.storm}原划分：G相对B改善{r.gain_vs_b_pct:.2f}%，{int(r.fold_wins)}/5折胜出；收益最大一折贡献净SSE改善的{100*r.best_fold_share_net_gain:.1f}%；去掉该折后，其余四折pooled RMSE改善为{r.rmse_gain_without_best_fold_pct:+.2f}%。贡献为正的县中，前五县占全部正SSE贡献的{100*r.top5_positive_sse_share:.1f}%。')
    lines += ['', '以上删折只是事后误差集中诊断，不能当作新主结果。贡献超过100%表示其他折合计抵消了部分收益。',
        '', '## 调度结论与下一步', '']
    passed=[d['storm'] for d in decisions if d['passes_all_required_screens']]
    lines.append('通过预先固定调度规则的事件：'+('、'.join(passed) if passed else '**无**')+'。规则要求随机至少4/5、空间至少3/4配置同时胜过B和P，且两类中位数均为正；有缺失观测时，完整观测敏感性也须满足。它是探索性工作排序规则，不是显著性检验。')
    lines += [
        '', '不能把T01中MICHAEL的23.68%或MILTON的5.12%写成稳定地理增益。本轮也不能证明非线性地理核无效：这里只测试了低容量线性峰值模型。',
        '', '后续继续找数据集和合适窗口：',
        '1. 对六候选做逐事件输入/标签审计：固定事件锚点、时区、停电分母变化、观测缺口、预测窗覆盖。不得按测试县真实峰移动窗口。当前T02没有修改原窗口。',
        '2. 优先核验MILTON的天气预报时效及空间分辨率；它在当前清单中观测完整，适合排查数据链路，但不能由此称为地理核收益最佳事件。同时保留其他候选和负结果。',
        '3. 把ERA5回溯条件与真正可用的HRRR预报分开；先检查预报起报时间和有效时效，不能将事后分析天气包装为预报。根据缺口决定是否扩充县覆盖、事件窗口或其他公开停电数据。',
        '4. 明确数据与信息边界后，再另立有限预算的完整轨迹HOST/GCRK/直接地理拼接比较，输出+1/+6/+24/+48的pooled MAE/RMSE。地理先验+历史更新保留为待检验设计，不凭线性探针替换原核。',
        '', '## 核验与限制',
        '', f'- T01的18项事件/输入臂结果复现，RMSE/MAE最大绝对差{audit["replay_max_absolute_error"]:.3g}；全部原输入哈希通过。',
        f'- 用每折加权误差和权重重建B/G全县pooled指标，最大绝对差{max(reconstruction_errors):.3g}；不是折RMSE算术平均，记录见`verification.json`。',
        f'- 完成{audit["oof_coverage_count"]:,}个县在不同配置下的OOF覆盖检查；训练/测试县不重叠，每县每配置只评分一次。预处理仅训练折拟合，内层选择alpha。',
        '- 空间划分没有缓冲带，临界县仍可能相邻；旋转0/90或45/135可能生成相同外层分区，故单列不同分区数。内层分区也可能不同。四配置不能当四份独立证据，不能把重复配置计数理解为独立成功次数。',
        '- 范围/胜出数是描述性稳定性指标，没有计算推断性置信区间。全部样本均是已查看开发D，保留原previously_used标记，不把它重命名为独立确认。',
        '- 首次启动在used标记断言处停止，尚未拟合；修正为保留已使用开发事件的标记后运行。本轮未读取封存确认集，未上传逐县预测。',
        '', '## 文件与复现',
        '', '- `results/t02/stability_summary.csv`：稳定性汇总；`metrics.csv`：全部配置/输入臂MAE和RMSE。',
        '- `comparisons.csv`、`fold_diagnostics.csv`：收益、集中性及空间距离；`observation_audit.csv`：观测覆盖。',
        '- `county_splits.json`、`inner_choices.json`、`audit.json`：划分、参数选择、来源哈希和核验。',
        '- `t02_split_stability.py`为运行入口，依赖同目录T01；`summarize_t02.py`生成本报告及汇总。运行前范围见`notes/T02_SPLIT_STABILITY_PROTOCOL_20261002.md`。',
        '', '保留原结果，复算可指定新输出目录（从仓库根目录运行）：',
        '```bash', "python3 - <<'PY'", 'import sys', 'from pathlib import Path', "sys.path.insert(0, 'experiments/geo_weather_20260924')", 'import t02_split_stability as t', "t.OUT = Path('/tmp/t02_reproduction')", 't.main()', 'PY', '```', '']
    (HERE/'notes/T02_SPLIT_STABILITY_RESULTS_20261002_ZH.md').write_text('\n'.join(lines))
    print(s.to_string(index=False)); print('Scheduling decisions:', decisions)


if __name__=='__main__': main()
