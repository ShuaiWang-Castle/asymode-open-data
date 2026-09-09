# AsymODE / DMDA — 当前目录数据的独立执行包

版本：2026-09-09。**只需要本包和CC所在项目目录已有的数据，不需要原 handoff ZIP、旧目录结构、旧稿件或历史checkpoint。**
本次改变的是交付/数据入口，不是再次更改已收束的科学问题。science protocol_id仍为`ASYMODE_TRAJECTORY_MAINLINE_R1_20260909`。

## 当前文章问题

在匹配历史、可得驱动和完整trajectory监督下，源池damage/recovery结构是否比普通完整轨迹学习带来可靠预测收益？
核心比较固定为 **DIRECT / NET / ASYM**。普通single-flow由真正直接输出整条trajectory的DIRECT代表，不能用删去一个方向的SR代替。
SR是消融；投影不是本轮新实验主线，旧投影优于旧双率的事实保留。2019是反复使用的回顾性评价，不能称fresh holdout。没有已核验的新GPU收益。

## 阅读与执行

1. 校验本包`SHA256SUMS.txt`；读取`CLAUDE_CODE_EXECUTE_ZH.md`和`CONFIG_LOCK.json`。
2. 读取`paper/PAPER_CORE_AND_THEORY_ZH.md`：动机、理论职责、主实验/消融/附录的位置。
3. 读取`DATA_CONTRACT_ZH.md`、`CODE_STATUS.md`、`code/`和`evidence/HISTORICAL_EVIDENCE_ZH.md`。
4. 保留启动时`pwd`为D，运行数据发现、身份核验与preflight。新工作目录为D/runs/asymode_trajectory_r1_20260909。
5. 使用本包模型/数据/采样/评分核心实现正式driver；进行GPU正确性与预算profile，然后实际运行真实三臂主比较。不是只生成计划或最终稿。

```bash
# 先在项目工作目录保存此值；不要先 cd 到解压目录后丢失数据根目录。
D="$(pwd)"
# 解压本ZIP后，将P设置为实际存在的独立包目录（不假设 /mnt/data）。
P="/actual/path/ASYMODE_CC_STANDALONE_20260909"
W="$D/runs/asymode_trajectory_r1_20260909"
python "$P/code/verify_package.py" --root "$P"
python "$P/code/prepare_local_data.py" --data-root "$D" --output "$W/data_resolution"
python "$P/code/preflight.py" --data-manifest "$W/data_resolution/DATA_MANIFEST.json" --output "$W/preflight"
```

`/actual/path`是待CC根据附件真实解压位置解析的占位符，不是让用户重建旧路径。
数据脚本按身份识别NPZ，不按旧路径认数据。无现成NPZ时可重建本地选中记录/旧版CSV。具体fallback与失败边界见数据合同；不得因为原ZIP删除就停止。

## 包内/包外

包含：研究思想、完整执行约束、模型/数据/评分/采样核心、固定cohort、台账数组指纹、历史反证摘要与CSV、运行准备工具、CPU预检。
不包含：大型原始数据、旧模型权重、陈旧完整稿件、已完成GPU训练器或新实验结果。正式campaign driver由CC围绕所给核心实现并运行；此边界在执行prompt中明确。

读取历史结果仅为约束主张。它们不得进入新trajectory主表或成为额外特征/监督。不得重新展开干预识别、证书、报告压缩、投影竞赛或天气组件。
