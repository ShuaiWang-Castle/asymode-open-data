"""Render the archived result tables without changing any recorded metric."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE / 'results/geography_results_20261008'
ARMS = ['ZERO', 'TIMESFM', 'DOSE', 'GCRK', 'COMPACT']
LABELS = ['全0预测', 'TimesFM 3.0', 'AsymODE', '＋原地理核', '＋新地理核']
EVENTS = ['michael', 'delta', 'winter', 'ice']
NAMES = ['Michael', 'Delta', '1月24日冬季风暴', '1月底冰冻风暴']


def read(path):
    return json.loads(path.read_text())


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '|' + '|'.join(['---'] + ['---:'] * (len(headers)-1)) + '|'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def metric_table(groups, names, key='rmse'):
    rows = []
    for arm, label in zip(ARMS, LABELS):
        cells = []
        for group in groups:
            m = group['models'][arm]
            cell = f"{m[key]:.6f}"
            if key == 'rmse' and arm != 'DOSE':
                change = 100 * (m[key] / group['models']['DOSE'][key] - 1)
                cell += f"（{'↑' if change > 0 else '↓'}{abs(change):.2f}%）"
            if m[key] == min(v[key] for v in group['models'].values()):
                cell = '**' + cell + '**'
            cells.append(cell)
        rows.append([label] + cells)
    return table(['模型'] + names, rows)


def main():
    original = read(ROOT / 'original_results.json')['cohorts']
    expanded = read(ROOT / 'expanded_counties/complete_results.json')['cohorts']
    storms = read(ROOT / 'more_storms/complete_results.json')['cohorts']
    joint = read(ROOT / 'more_storms/joint_vs_separate.json')
    same = read(ROOT / 'expanded_counties/same_county_comparison.json')
    # User-approved first table, checked against unrounded archived values.
    expected = [
        ['0.136812（↑49.66%）', '0.145082（↑25.28%）', '0.063833（↑37.53%）', '0.070845（↑6.74%）'],
        ['0.136793（↑49.64%）', '0.144946（↑25.17%）', '0.063816（↑37.49%）', '0.070809（↑6.69%）'],
        ['0.091417', '0.115802', '0.046414', '0.066372'],
        ['0.089976（↓1.58%）', '0.115497（↓0.26%）', '0.047171（↑1.63%）', '0.065455（↓1.38%）'],
        ['0.085404（↓6.58%）', '0.113360（↓2.11%）', '0.047218（↑1.73%）', '0.065400（↓1.46%）'],
    ]
    first = metric_table([original[c] for c in EVENTS], NAMES)
    for row in expected:
        for cell in row:
            assert cell in first.replace('**', ''), cell
    scalar_checks = 0
    for group in list(original.values()) + [x for g in expanded.values() for x in g.values()] + [x for g in storms.values() for x in g.values()]:
        for arm, m in group['models'].items():
            expected_gain = 100 * (1 - m['rmse'] / group['models']['DOSE']['rmse'])
            assert abs(expected_gain - m['rmse_reduction_vs_asymode_pct']) < 1e-10
            if 'per_seed' in m:
                for key in ['rmse', 'mae']:
                    assert abs(sum(x[key] for x in m['per_seed']) / 3 - m[key]) < 1e-12
            scalar_checks += 1
    parts = [
        '# 地理核实验结果报告',
        '结果日期：2026年10月7日；整理与复核日期：2026年10月8日。先展示原始四场，再展示数据范围拓展。原始结果未修改。',
        '## 1 原始四场风暴',
        '指标为停电比例RMSE，越低越好。括号相对同列AsymODE计算：↓表示误差下降，↑表示误差上升；加粗为同列最小值。先合并一个种子的五折测试预测计分，再平均种子0、1、2的指标，不直接平均折RMSE，也不先集成预测。',
        first,
        '县数分别为147、113、113、115。新核相对AsymODE的RMSE降幅为6.58%、2.11%、−1.73%、1.46%；1月24日冬季风暴中AsymODE最好。',
        '来源：[原始精确指标](../results/geography_results_20261008/original_results.json)。证据等级[C]：开发期验证，未作为未接触数据的确认实验。',
        '## 2 此前扩大县域的结果',
        '以下保留已经完成的县域拓展试验，县数变为249、138、138、140。本轮后续方向已改为固定147县增加风暴，未继续新增县。',
        metric_table([expanded[c]['all'] for c in EVENTS], NAMES),
        '样本组成已经改变，不能将本表RMSE与上一表直接相减，作为扩大数据范围的净提升。下面仅在原有县集合上比较两套训练结果，正数表示扩大训练范围后误差上升。',
        table(['模型'] + NAMES, [[label] + [f"{same[c][arm]['rmse_increase_pct']:+.2f}%" for c in EVENTS] for arm, label in zip(ARMS[2:], LABELS[2:])]),
        '来源：[扩大县域指标](../results/geography_results_20261008/expanded_counties/complete_results.json)、[原县集合对比](../results/geography_results_20261008/expanded_counties/same_county_comparison.json)。证据等级[C]。',
        '## 3 固定147县增加风暴',
        '保留Michael的147个县级单元（北卡89个、弗吉尼亚58个），新增Isaias 2020、Zeta 2020、Ophelia 2023。前三列分别训练；最后一列联合Michael共四场训练，合计588条县—风暴样本，仍是147个独立县。',
        metric_table([storms[c]['all'] for c in ['isaias','zeta','ophelia','pooled4']], ['Isaias','Zeta','Ophelia','四场合并']),
        '合并训练中新核相对AsymODE降低RMSE 4.60%，相对原核降低6.13%；三个新增事件单独训练中，两场改善、一场退步。不能写成所有风暴均稳定提升。',
        '### 3.1 在相同样本上比较分别训练与合并训练',
        table(['模型','分别训练RMSE','合并训练RMSE','合并后误差升幅'], [[label,f"{joint['joint_vs_separate'][arm]['separate_rmse']:.6f}",f"{joint['joint_vs_separate'][arm]['joint_rmse']:.6f}",f"{-joint['joint_vs_separate'][arm]['joint_vs_separate_rmse_reduction_pct']:.2f}%"] for arm,label in zip(ARMS[2:],LABELS[2:])]),
        '同样588条测试记录上，合并训练均未优于分别训练；新核误差升高6.87%。分别训练为每场600步，联合训练为总共600步，训练预算的影响尚未排除。新核比合并AsymODE好，不等于增加风暴本身改善了预测。',
        '### 3.2 合并模型按事件拆分',
        '下表三种训练模型均来自四场联合拟合，再按事件计分；与单场训练结果区分。',
        metric_table([storms['pooled4'][s] for s in ['S00034','S00023','S00047','S00040']], ['Michael','Isaias','Zeta','Ophelia']),
        '来源：[多风暴精确指标](../results/geography_results_20261008/more_storms/complete_results.json)、[联合与单场训练对比](../results/geography_results_20261008/more_storms/joint_vs_separate.json)。证据等级[C]。',
        '## 4 随机种子表现',
        '下表为新核相对同种子AsymODE的RMSE降幅，正数为改善。汇总按平均RMSE计算，不是种子百分比的简单平均。',
    ]
    seed_rows=[]
    for c,name in zip(['isaias','zeta','ophelia','pooled4'],['Isaias','Zeta','Ophelia','四场合并']):
        g=storms[c]['all']['models'];gains=[100*(1-x['rmse']/y['rmse']) for x,y in zip(g['COMPACT']['per_seed'],g['DOSE']['per_seed'])]
        seed_rows.append([name]+[(f'{x:+.4f}%' if 0<abs(x)<.005 else f'{x:+.2f}%') for x in gains]+[f"{g['COMPACT']['rmse_reduction_vs_asymode_pct']:+.2f}%"])
    parts += [table(['场景','种子0','种子1','种子2','汇总'],seed_rows),
        '四场合并三个种子均改善；Zeta的平均提升主要来自种子0，Ophelia三个种子均未改善。三个种子的均值不构成统计显著性证明。',
        '## 5 MAE补充', 'MAE同样越低越好。以下按与RMSE相同的样本及种子口径计算。',
        '### 5.1 原始四场',metric_table([original[c] for c in EVENTS],NAMES,'mae'),
        '### 5.2 扩大县域',metric_table([expanded[c]['all'] for c in EVENTS],NAMES,'mae'),
        '### 5.3 固定县域增加风暴',metric_table([storms[c]['all'] for c in ['isaias','zeta','ophelia','pooled4']],['Isaias','Zeta','Ophelia','四场合并'],'mae'),
        'Zeta和Ophelia中全0预测的MAE低于三种训练模型；四场合并中新核相对AsymODE的MAE仅降低约0.32%。不能把RMSE收益写成所有误差指标均明显改善。',
        '## 6 模型与评估设置',
        '新地理核保留有界递归结构，采用六个变量：relief_p95_p5、canopy_mean、forest_near_developed、poorly_drained_share、root_limiting_share、developed_frac。减少冗余地理变量，并将逐县向量长度缩放改为训练县确定的统一均方根尺度。仍有逐特征标准化、有界变换和中心化，不能称为完全不归一化，也未证明已经消除共线性。变量筛选与尺度改动尚未拆开消融。',
        '固定县域多风暴试验使用前72小时停电历史、已知未来天气，预测之后144小时。每次拟合600步，损伤侧学习率0.003、恢复侧0.0003；每100步用调参MSE选权重，允许选初始化状态。未来停电标签不作为输入；特征统计仅由拟合县确定。',
        '同一县在所有风暴中使用相同县组划分，每折测试一个县组、调参下一个县组，其余三个组拟合。测试县的其他风暴记录不能进入拟合或调参。这是已知风暴下的县级留出，不是未见风暴迁移测试。',
        '停电数据沿用稀疏EAGLE-I记录口径：已采集时点没有县记录则记0，小时值按实际采集次数平均；再除以既有2024年客户数并裁剪到0至1。所有县在归档时间范围内出现，不等于每个时点均有独立监测覆盖。',
        'TimesFM 3.0使用停电客户数输入和18个天气/时间通道，将预测除以相同客户数分母，以中位数预测计分。当前结果只对应归档配置，不能外推为所有TimesFM用法无效。',
        '## 7 数值核验与保存',
        '原始四场表的20个单元格及百分比已与用户指定版本逐项比对。原始基线归档620个文件、县域拓展归档1,085个文件、多风暴归档1,098个文件均通过其SHA-256清单校验。复制到本分支的源JSON和CSV保持原字节，不使用中途partial_summary替换最终结果。',
        '2026年10月8日从多风暴保存预测独立核算333个数值，最大差异1.39×10⁻¹⁷；另检查历史180次权重复算记录及60项隔离检查。此次整理没有重训或重新执行所有权重推理。核算记录见[prediction_metric_audit.json](../results/geography_results_20261008/prediction_metric_audit.json)。',
        '训练期间每50步保存恢复状态，包括优化器和随机状态；完成标记最后写入，重启跳过已完成任务。原始与拓展试验分开保存，保留负结果。',
        '来源归档：DMDA_Baseline_Comparison_20261007.zip、DMDA_Expanded_Scopes_20261007.zip、DMDA_More_Storms_Fixed_Counties_20261007.zip。训练阶段记录提交23caa8dc103a7de5247bf17f7b7d20d1fe3df4cd；本分支此次恢复的是结果、配置、核验及报告，不声称该训练提交或全部训练源码已恢复到远端。原始输入与权重仍保存在已有归档。',
        '结果表均标为[C]开发期证据：Michael误差曾用于修改模型，其他事件也曾被查看；不作为未接触数据的确认性结论。下一步需分别排除联合训练预算不足，并在未参与设计的事件上验证。',
    ]
    target=HERE/'notes/GEOGRAPHY_RESULTS_20261008_ZH.md'
    target.write_text('\n\n'.join(parts)+'\n')
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name!='publication_manifest.json'}
    (ROOT/'publication_manifest.json').write_text(json.dumps(dict(files=hashes,verified_model_groups=scalar_checks,original_table_cells_checked=20,metric_values_changed=False),indent=2)+'\n')
    print(json.dumps(dict(report=str(target),files=len(hashes),verified_model_groups=scalar_checks)))


if __name__=='__main__':main()
