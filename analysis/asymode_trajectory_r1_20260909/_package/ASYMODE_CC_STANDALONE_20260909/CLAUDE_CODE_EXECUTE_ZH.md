# Claude Code：实际执行 AsymODE 完整轨迹主线实验

**版本：ASYMODE_TRAJECTORY_MAINLINE_R1_20260909。你是执行器，不是另起研究方向的研究代理。**
任务：使用当前目录已有数据和本包固定 ANEEL cohort，实现并真实运行下面锁定的实验，返回可重放的结果包。不要只给计划、论文改稿或建议。

## 0. 最高优先级与范围

当前用户修正：普通单流基线必须是直接预测完整 trajectory 的模型，不能由双流删除 damage/recovery 定义。
核心比较只有 **DIRECT / NET / ASYM**；SR 只是结构删减消融。
本轮不运行新的 DN_READ / DN_PROJ，也不研究“结构训练前还是后处理”。历史投影胜过 TR 的反证完整保留在附录，不得删除或宣称已被新实验推翻。
不做新的方向标签训练、方向日志校正、AFF 模型竞赛、Gamma 神经阈值、低状态重训练 campaign、天气组件、干预识别、证书或报告压缩。
不要自行添加 Transformer、GNN、attention、memory kernels、外部数据、额外损失项或“提升版”模型。

### 文件与优先级
1. 本 prompt 和 CONFIG_LOCK.json。
2. 本包 paper/PAPER_CORE_AND_THEORY_ZH.md、code/core_models.py、code/data_contract.py。
3. 本包 DATA_CONTRACT_ZH.md、metadata/ 固定 cohort 与指纹、evidence/HISTORICAL_EVIDENCE_ZH.md。
4. 当前目录中的旧稿和旧代码仅是可选历史参照；不允许成为缺失即停工的依赖。
本地旧 SR-first / PROJ-first 工作顺序被本文件替代；不要求恢复旧文件。

## 1. 独立包与当前目录数据：不需要原 handoff ZIP

**用户已删除原始 handoff；数据仍在 CC 当前文件夹。只使用这个新 ZIP、其内代码/元数据/文章思想，以及本地既有数据。禁止要求旧 ZIP、PATH_ALIASES.json、H/00_START_HERE.md 或旧 checkpoint。**
本次只是去除交付依赖，科学 protocol_id 仍为 ASYMODE_TRAJECTORY_MAINLINE_R1_20260909；有相同新协议的已完成 run 先核验，不能因换包名重复训练。

- D = CC 启动时的项目工作目录，必须在切换目录前记录 `pwd`。
- P = 本包实际解压后的 ASYMODE_CC_STANDALONE_20260909 根目录。
- W = D/runs/asymode_trajectory_r1_20260909；新代码、缓存、logs/checkpoints全部写这里。
- 本包只读，D里的源数据只读。不得把打包环境的 /mnt/data 路径当服务器路径。

先检查当前 GPU 任务与同协议已有 campaign：不杀进程、不重复运行；旧 one-step checkpoint 不是新 path-trained模型。
从 shell 中保存 D，再解压本 ZIP、确定 P；不要硬编码附件挂载目录。下面三条是随包**已实现**的准备命令：

```bash
# D=$(pwd) 必须在开始切换目录前执行；P 使用实际解压目录。
W="$D/runs/asymode_trajectory_r1_20260909"
mkdir -p "$W"
python "$P/code/verify_package.py" --root "$P"
python "$P/code/prepare_local_data.py" --data-root "$D" --output "$W/data_resolution"
python "$P/code/preflight.py" --data-manifest "$W/data_resolution/DATA_MANIFEST.json" --output "$W/preflight"
```

`prepare_local_data.py` 会依次：搜索任意文件名的本地库存 NPZ并检查canonical数组；或由完整 `pilot_records_g*.npz` 重建；或由本地旧版18列2018/2019 ANEEL CSV/ZIP重建。不联网下载、不重新筛选cohort、不重新估计分母。
- 文件重新压缩导致ZIP级hash不同，但canonical数组逐项hash相同，可以接受并记录。
- 只有48个canonical库存数组的NPZ也可运行净监督主实验；没有fp/fm时标注gross审计缺失，不伪造方向数据。
- 完整48组库存/192数组指纹与固定24集合/16公司的映射均在本包metadata中，不依赖旧目录。
- 不自动穿越目录符号链接；从项目配置发现其他本地数据路径时，重复传 `--data-root`；已找到精确NPZ可追加 `--ledger /actual/path.npz`。
- 找不到时读 `DATA_DISCOVERY_FAILURE.json` 并检索D内目录、README、配置。可新增**只做路径/schema适配**的adapter，不改cohort、分母、小时边界、标签或划分；适配后必须通过canonical库存指纹。只有数据身份确实无法确认才报告BLOCKED_DATA和具体缺什么，不能要求用户恢复已删除ZIP，也不能仅因旧文件名不存在而停工。

所有已实现的命令先 `--help` 再调用。包提供经过CPU核查的**模型/数据/评分/采样核心和本地数据入口**，并非已完成正式campaign driver；你必须实现第6节的训练、选择、评价、打包驱动并实际执行。不能把driver尚未编写当成只返回计划的理由。
不升级全局CUDA、不付费开新资源、不推送仓库。

## 2. 数据固定与禁止泄漏

读取 `$W/data_resolution/DATA_MANIFEST.json` 获得实际ledger路径；selection一律为本包 `metadata/unit_selection.csv`，不依赖本地同名旧清单。
主 ledger SHA256：03392a7f94bf9fbef6cf59c127afbd0ca829db27bcde0843fcaad666d5184e06。
24 个 included set、16 个 company。按 selection 中 evaluation_status=='included' 后按 group 排序；group 编号不连续。
每个 g/year 的 y 长度8761；完整ledger的fp/fm/area长度8760。原约定不改小时边界；有方向数组时核对 diff(y)=fp-fm，缺失时只标审计不可用，不影响已核验库存的净路径训练。

**训练 allowlist**：过去24小时库存、当前库存、set、已知 clock，目标仅 future stocks。
fp/fm 仅 ledger 守恒审计可读；不作 loss、初始化或选参。area、未来结束信息、事故剩余时长、事后 cause、2019 projection列绝不进入训练。
禁止把 frozen_nn_rates.csv 整表输入新 pipeline。
固定 cohort 与分母本身含历史回顾性选择；fold统计重新计算并不把它变成 prospective study。

### 窗口
L=24：历史为 y[t-24:t]，不含 y0=y[t]。H=24：标签 y[t+1:t+25]。
整个 [t-24,t+24] 的49个状态都须在原 month-interior mask 与对应 partition 内合法；不拼接压缩后的有效小时。
未来 clock 使用 t..t+23 的原六小时 one-hot4 + weekend1，共24×5；它们预测时已知。
1/6/24h终点必须从同一24h路径预测、同一起点集合抽取。不能另开1h样本集后拼成曲线。
本轮新增 history，因此不能复用上一窗口 history=0 的 origin 索引。

主 L24/H24 的预检计数（按附包实现）应为：

| partition | 每集合 | 24集合总计 |
|---|---:|---:|
| fold_A_fit | 3745 | 89880 |
| fold_A_val | 1920 | 46080 |
| fold_B_fit | 5665 | 135960 |
| fold_B_val | 1920 | 46080 |
| full_2018 | 7608 | 182592 |
| reused_2019 | 7608 | 182592 |

不符先定位 mask/边界错误，不能删样本“对齐”结果。保存所有 origin IDs 和哈希。

### 2018 调参与2019评价
A: 2018上半年fit -> 7–9月validation；B: 前9月fit -> 10–12月validation。
fit最后目标 <= validation nominal start -48h；validation仍使用名义季度。history和target整个窗口合法。
所有 normalization、source_mean、分层阈值逐fit重算；不能用全2018统计预处理早期fold。
2019模型评价必须等 selection_lock.json 写好以后。2019已反复使用，输出名称为 REUSED_2019_RETROSPECTIVE_EVALUATION，不得称 fresh/unseen/confirmatory holdout。

## 3. 核心模型实现必须与下列定义一致

### 3.1 同一可得信息，不同预测函数约束

```python
# hist: [B,24], y0: [B], clocks: [B,24,5], group_onehot: [B,24]
# s is a FIT-ONLY, company-equal RMS of available history/current stocks.
c = cat([group_onehot, hist / s, clocks.flatten(1)], dim=1)  # [B,168]
# c explicitly EXCLUDES y0. Do not add y0 to the rate encoder through a side channel.
```

三类模型都可用当前 y0；ASYM 只通过源池因子使用它，这就是要检验的结构限制。
context 在整个预测期冻结，包含同样的历史与已知未来 calendar。不要让某个方法额外读未来真实状态/未来天气，也不要只给 DIRECT 历史。

### 3.2 DIRECT：真实直接轨迹基线

```python
v = cat([c, y0[:, None] / s], dim=1)
# ResidualMLP includes a learned linear skip and a nonlinear 2-hidden-layer MLP.
yhat = y0[:, None] + s * direct_net(v)  # output [B,24] in ONE network call
```

禁止将它实现为 signed-rate；禁止24个单独训练的一步模型；禁止只看当前状态不给相同历史；禁止从 one-step loss 的 baseline checkpoint 拿 rollout 冒充。
保留 learned linear skip 和 persistence skip，不故意去掉强基线的常见预测通道。本模型称 matched residual MLP direct forecaster；不是完整 TiDE 复现，不挂 TiDE/SOTA 名称。

### 3.3 NET：一般净动力学控制

```python
y = y0
for k in range(24):
    delta = s * net(cat([c, clocks[:, k], y[:, None] / s], dim=1)).squeeze(-1)
    y = y + delta
    predictions.append(y)
```

递归读自己的预测 y。一个 scalar 输出可以表达完整净变化，不能删除任何方向。
它控制“递归结构 vs 直接多时域输出”的区别。不是原 signed single-rate。

### 3.4 ASYM：源池结构预测器

```python
# rates conditioned on origin history and known driver; NOT on recursive y.
logits = rate_net(cat([c_at_each_step, clocks], dim=-1))   # [B,24,2]
p = softmax(cat([logits, zeros_like(logits[..., :1])], dim=-1), dim=-1)
u, r = p[..., 0], p[..., 1]
y = y0
for k in range(24):
    y = y + u[:, k] * (1 - y) - r[:, k] * y
    predictions.append(y)
```

使用原离散三角形率类 U,R>=0,U+R<=1。不改成无约束率、不换 exponential integrator、不冒称已经学习连续物理 hazard。
rate schedules可整批计算，但state递推与梯度必须保留。
初始化为 net-state-only 的内点：a0=.95，pi0=clip(mean_fit_Y,1e-4,1-1e-4)，U0=.05*pi0,R0=.05*(1-pi0)。只控制初始bias，不是硬rate下限。
不要重新引入R≈1e-6的恢复边界初始化，也不要用ending标签“帮忙”初始化。

### 3.5 相同完整轨迹损失

```python
pred = model(c, y0, clocks)          # interface DOES NOT accept target
loss = (((pred - target) / s) ** 2).mean()
loss.backward()                    # backprop through all 24 steps
```

不detach中间预测、不truncate BPTT、不在main训练中teacher-force，不加rate、derivative、projection、conservation或directional loss。
主要评分在原始状态尺度。s只是一整个fold共有的正标量，不是逐公司/逐时域归一化以改变主目标。
所有主模型未裁剪raw预测用于主评分；输出越界率也报。冻结后额外报告一次统一[0,1]裁剪的附表（NET的clip只对输出评分，不回灌递归）；不得拿各自更有利的版本组成主表。

**重要：带history之后context不再只有192种取值。禁止复用旧192-cell moment loss代替本轮样本路径损失。** 可缓存输入、batch向量化，但不能合并不相同上下文。

## 4. 训练目标权重、模型容量与调参固定

公司等权 -> 同公司集合等权 -> 集合内合法origin等权 -> 24个horizon等权。
训练 batch 实现：均匀抽company，再均匀抽其set，再均匀抽该set的origin；用独立数据RNG保证三个主模型同seed看到同样的采样序列。验证/评价精确计算全origin加权，不用随机小batch估计选参指标。
按公司聚合不是声称公司统计独立。

每个主模型相同六候选：
- 总可训练参数目标 {8192,32768}，由附包match_width纯计数确定width，误差<5%。不得按validation表现改width匹配规则。
- learning rate {3e-4,1e-3,3e-3}。
- AdamW, weight_decay=1e-4, batch=512, clip_grad_norm=1, dropout=0, 无scheduler。

阶段1：6候选×2fold×seed4101；每run最多3000更新，每250更新完整validation并保存best checkpoint，训练到预算结束。可记录step0但step0不作为“已训练最好结果”。
阶段2：每个model取source平均validation最好的2候选，补seed4102/4103、两个fold同预算；汇总三个开发seed与两fold选唯一配置。不得让ASYM享有更多搜索机会。
两fold平均用各fold原始company-equal path MSE等权，不混入ending/gross/2019。
最终 refit：五个paired seeds 5101..5105；步数按选定候选各fold最佳update×N_full/N_fit的中位数取整，最多6000（若全局对称缩短预算则最多2T）。每model在评价前锁定步数，不用2019早停。
保存 selection_lock.json：所有候选、种子、best_update、fold统计、选择结果、最终步数、代码/数据哈希。

参数机会与梯度样本预算对齐，并不等于 FLOPs / walltime 完全相等。三种实现差异必须报告实际时间、显存、参数数与推理cost，不写“相同计算量”掩盖递归开销。

## 5. 资源与工程规则

先做CPU/小GPU correctness及吞吐profile，使用现有空闲资源，不超过24 aggregate GPU-hour上限或当前已有更小配额，取较小者（沿用既定预算，不授权新付费资源）。并行2GPU则2倍计GPU-hours。不得自行扩额。
用300步/小规模profile估计主线完整总预算。若预计超过可用额度的80%，在任何正式trial前将所有主模型development steps统一降为
T=floor_to_250(3000 * (0.8*available_gpu_seconds/projected_primary_gpu_seconds))，上限3000，下限1000；final上限2T，记录resource_plan。
需要T<1000时保留核查/profile，标资源阻塞，不跑不对称残缺“比较”冒充完成。
搜索与确认按model-balanced round-robin推进；达到额度保存resume state与已完成完整轮次，不优先跑满ASYM而放弃DIRECT。
额外消融/controlled仅在主要真实比较所需预算已预留后运行，不能反过来拿toy占满预算。未完成项明确NOT_RUN/INCOMPLETE。
不要把失败trial删掉后补一个新配置；确定性数据/代码bug修复后受影响的配对arms必须同步重跑并留痕。

正式训练前使用 `data_contract.preload_stocks` 一次性读取2018库存，只读缓存；避免每batch反复解压NPZ。2019仅在selection_lock后进入模型评价。可以预构造/缓存输入或device-side采样，但必须逐批对齐reference IDs、信息和损失，不能改company权重。

默认FP32、AMP关闭、torch.compile关闭。工程提速仅在与reference前向/梯度对齐后启用且记入配置。不要把历史CPU小模型的速度推断为新GPU结果。

## 6. 正确性闸门

运行附包preflight，随后GPU上做相同模型小批次的前向/反向与CPU FP64参考比较。建议前向 atol1e-6/rtol1e-4，归一化gradient max relative error<=1e-3；不符需定位，不能静默调宽容差。
必查：forward不接收target；打乱/改变未来标签不改变同一输入的预测；H=24输出维度；梯度流过完整路径；ASYM/SR无额外clip即保持[0,1]；DIRECT确实一次前向输出24个未来值；NET依赖自己的递归state。
保存loss/gradient检查，不将12/20/300步smoke/profile曲线当科学结果。
为处理训练代码尚不存在，请在W实现以下接口并先--help与小范围smoke，再实际运行：

```text
run_campaign.py --data-manifest W/data_resolution/DATA_MANIFEST.json --protocol P/CONFIG_LOCK.json --work W --phase main
run_campaign.py --data-manifest W/data_resolution/DATA_MANIFEST.json --protocol P/CONFIG_LOCK.json --work W --phase ablation
run_campaign.py --data-manifest W/data_resolution/DATA_MANIFEST.json --protocol P/CONFIG_LOCK.json --work W --phase controlled
run_campaign.py --data-manifest W/data_resolution/DATA_MANIFEST.json --protocol P/CONFIG_LOCK.json --work W --phase package
```

这些是你须实现的driver接口，不是声称随包已有的命令。核心数据/模型函数须优先直接import本包只读源码（或先复制并记录hash的W源码副本），避免手抄改变函数类。
`training_primitives.py` 已给出公司等权采样器、完整路径独立评分器；`controlled_data.py` 已给出锁定生成机制。不要另写不同权重、不同toy或用平衡后的随机验证batch选参。
实际运行、记录exit code、stdout/stderr与完整命令。不得只创建这些命令的字符串或README就宣布完成。

## 7. 两项核心实验

### E1：真实trajectory主比较，最高优先级
完整执行第2–6节。三主模型DIRECT/NET/ASYM，另计算无训练PERSISTENCE。
主表列：company-equal path MSE、1h/6h/24h endpoint MSE、训练/推理cost；另提供相对DIRECT的mean-company-relative gain。

    gain(model)=mean_company[(MSE_DIRECT,company - MSE_model,company) / MSE_DIRECT,company].

先对每个seed分别算，再汇总paired seeds。若某公司DIRECT分母为零，该公司relative与全cohort mean-company-relative均标N/A并报告数量，不通过删除零分母公司定义一个不同总体；绝不加入任意epsilon制造收益；raw MSE仍全保留。
不要把ratio-of-means代替上述均值；二者可在不同列并列。
main selection只用path MSE，不能用更有利的endpoint/stratum替换。
固定一个可选描述切片：initial Y_t > 对应set的fit q90；threshold source-only、不重训、不称新支持外推。这只是次表，不改变主判定。

### E2：单个可解完整trajectory controlled case
本任务不是新的“field evidence”，不复用旧一步checkpoint。
x~Uniform[-1,1]，U=.005+.035*(x+1)/2，R=.20；H24内x固定且三模型均已知。
source Y0~Uniform[.25,.45]；high Y0~Uniform[.55,.75]，机制不变；所有输入与真实标签重新生成，不能真实数据改Y而照抄未来。
每dataset seed：2048 fit，1024 source validation，4096 source evaluation，4096 high evaluation，split独立。noise为未来每步iid Normal(0,.005^2)，不污染输入Y0、不clip noisy targets；保留无噪真trajectory用作MSE评价。
三个主模型均无history、context=x[:,None]、known_clock=x重复24步且clock_dim=1；这是与真实任务分开的controlled信息合同。代码核心支持这些维度。
同一六候选预算/lr、最多3000更新；第一dataset seed4101按source validation分别选配置；随后五独立dataset/fit seeds5101..5105锁定复算。所有主模型获得同样预算。真实与controlled不共用根据target挑出的配置。
仍使用full-path loss并记录source/high path和相同1/6/24 endpoints。native U/R只能作为ASYM附加诊断，不把DIRECT/NET的R记为预测失败。
不加多个regime、不扫Gamma、不临时挑更容易让ASYM赢的初态区间。允许真实或controlled任何方向的负结果。

## 8. 消融不是主基线

### A1：SR 单向源池删减
通过 core_models 的 SR 类，s=tanh(net(c,clock))，U=relu(s),R=relu(-s)，仍完整path训练。
固定使用选定ASYM的参数目标、lr、优化设置；仅预先规定signed_init in {-.05,+.05}，用2018两fold/三个开发seed选初始化，不看2019。最终五seed按同一规则refit。
这是一项条件于匹配超参的结构消融，不是完整独立HPO过的普通预测器；表中明确标注。不得仅凭它弱于ASYM宣称主任务成功。
SR的误差理论只适用于定义明确的限制类。完整轨迹训练SR不自动具有旧单步r*的精确式，旧公式仅在单步解析诊断中核对。

### A2：ASYM_STEP_ONLY 训练目标消融
使用相同ASYM配置、相同origin与目标状态集合，训练时改成每个k用真实y_{t+k}的一步MSE；core_models.one_step_ablation_loss已给出。
测试始终24步纯递归。严禁将其teacher forcing性能放入主路径表冒充预测。它只隔离训练目标，不代表新的best-tuned方法族。

### 历史投影与其他诊断
复制本包 `evidence/` 中历史CSV与说明进入 appendix_evidence，注明旧单步任务、受控冻结、非新训练；本轮无需找到原latest-audit目录或旧checkpoint。
不重做source-projection真实模型。保存“DN+source triangle R RMSE .004903 < oldTR .007545；5/5seed”的事实。
旧Gamma、log audit等不新增campaign，也不要求从已删除ZIP恢复。历史CSV的用途由本包evidence说明限定。

## 9. 结果解释强约束

- ASYM稳定优于DIRECT与NET：仅支持该cohort、信息合同、预算下的trajectory结构收益，不是普遍two-flow最优。
- ASYM胜DIRECT但不胜NET：不能把递归方式的收益归于damage/recovery分离。
- ASYM只胜SR：主线收益未闭环。
- DIRECT/NET更好：保留结果，不换metric/subset、不新增loss救场、不删除baseline。
- native R不是物理hazard真值；净监督不识别任意gross mechanism；不能因DIRECT没有原生R让其在rate表落败。
- 原三角离散更新和具体源池结构的稳定性质可以说明偏置，但不是“DT必然趋零”或一般误差下界。
- 2019是复用回顾性评价，seed不是独立数据；不写现场收益、fresh test、因果机制已经验证。

若做区间：同步抽取跨公司共同日历时间块，保留跨公司共同冲击；主用168h块，报告作为描述性依赖敏感性。不得按窗口iid bootstrap，不能把5seed当5份独立真实样本。统计实现若未验证，先提供完整company/seed/hour原始误差，不伪造区间。

## 10. 返回内容与执行完成标准

必须有：
- START_HERE.md：实际完成/失败/未运行；只陈述已经执行的内容。
- DATA_AND_INFORMATION_CONTRACT.json、SPLIT_MANIFEST.json、origin IDs及hash。
- CONFIG_LOCK.json、resource_plan.json、selection_lock.json、trial registry（含失败trial）。
- 实际新driver与全部源码、每trial config、learning curves、checkpoint/optimizer/resume state。
- MAIN_REAL_RESULTS.csv、company/set/seed细表、预测数组或可完全重建的输出。
- ABLATION_RESULTS.csv、CONTROLLED_RESULTS.csv；未运行时状态清楚，不能填旧数据。
- appendix_evidence/：旧投影反证与negative controls来源，不混进新主表。
- NEGATIVE_AND_FAILURES.md、ENVIRONMENT.json、完整运行命令与exit codes、REPRODUCE.sh。
- 独立评分器重算结果差异日志、SHA256SUMS.txt、最终ZIP与短报告。

主表原始结果至少提供：protocol_id, dataset, split, model, config_id, seed, company, group, origin_hour, horizon, y0, truth, prediction，或等价分列NPZ+有schema的索引表。
checkpoint保留所需列：model definition/width/parameter count、state_scale/source_mean、selected updates、data/code hashes、RNG states。

**开始后先给简短接收与资源状态，再实际执行。不要再输出研究方向备选菜单，不要重新写文章最终稿，不要等待我选模型；上述模型、范围、超参规则、停点已经明确。**

## 11. 本次交付状态与启动顺序

这是独立代码/思想交付包，不含大数据集或历史checkpoint。打包侧只执行本地准备和CPU小批次测试；没有新GPU tuning、没有新真实收益、没有新2019模型评价。
先读 `00_START_HERE.md` -> 本执行任务 -> `CONFIG_LOCK.json` -> `paper/PAPER_CORE_AND_THEORY_ZH.md` -> `DATA_CONTRACT_ZH.md` -> `code/` -> `evidence/HISTORICAL_EVIDENCE_ZH.md`。
进入main真实任务，优先保证DIRECT/NET/ASYM完整公平比较；追加工作顺序与预算遵守前文。不再扩充文章、不等待用户选择科学路线。
