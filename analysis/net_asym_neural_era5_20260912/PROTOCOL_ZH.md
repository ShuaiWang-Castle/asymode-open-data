# Claude Code 执行任务：NET / Asym 神经实验，全部通过 GitHub 交付

请实际完成数据获取、核验、代码实现、训练、评价和 GitHub 交付，不要只返回计划。

仓库：
https://github.com/ShuaiWang-Castle/asymode-open-data

本任务不需要任何附件、交接 ZIP 或 ChatGPT 沙箱文件。
所有必要指令已写在下面；仓库没有的新增实现，由你根据本指令编写并提交到本轮研究分支。

美国 EAGLE-I 停电数据与匹配 ERA5 可以直接用于正式实验。
不要求使用巴西天气，不将巴西地理／天气匹配作为任何任务的前置条件。

---

## 1. 唯一研究问题与模型边界

唯一问题：

在相同可用信息、相同完整轨迹监督下，什么样的条件响应分布使单个净流 NET 更好，什么样的分布使显式双流 Asym 更好？

唯一学习模型：NET、ASYM。

NET：
    y_hat[k+1] = y_hat[k] + f_theta(c, x[k], y_hat[k])

ASYM：
    y_hat[k+1] = y_hat[k]
                 + U_theta(c, x[k]) * (1-y_hat[k])
                 - R_theta(c, x[k]) * y_hat[k]

c 是预测起点固定的历史／上下文；x[k] 是本任务声明允许使用的外部信息。
ASYM 的率函数不任意读取递归中的当前预测状态；该状态通过源池因子进入。
NET 可以表达正负净变化，也可以表达双流所生成的净响应。

严禁把 NET 实现成 signed single-rate：
    relu(s)*(1-y) - relu(-s)*y
这个对象是旧的单向源池限制，不是本轮 NET。

两模型都递推完整路径，按完整路径平方损失训练。
禁止未来真实状态 teacher forcing、方向标签辅助训练、事故结束信息泄漏。
不新增 DIRECT、SR、source-rate projection、地理 kernel、邻县模型、干预识别或证书任务。
允许报告统一输出裁剪敏感性，但它不是新增学习模型。

本轮三个实验任务：
A. 美国 EAGLE-I + ERA5：真实条件响应比较。
B. ANEEL：固定历史研究样本上的定向修复与配对评价。
C. Synthetic：实际神经模型在同 x、零初态、不同条件响应分布下的比较。

三个任务可以独立推进。不能因为 ANEEL 缺巴西天气，就不跑美国 ERA5；
也不能因为某个数据源暂时阻塞，就停止其余任务。

---

## 2. GitHub 工作方式与读取顺序

### 2.1 保护当前工作区

先记录：
    git status --short
    git remote -v
    git ls-remote --heads origin

有未提交文件时不要覆盖。使用独立 clone 或 git worktree。
禁止 reset --hard、force push、删除旧实验目录、覆盖旧预测或修改历史校验值。

本轮使用独立分支：
    research/net-asym-neural-era5-20260912

如果同名分支已有其他工作，读取后使用唯一后缀创建新分支，不能覆盖。

已核查的历史数据／代码基准：
    branch: aneel-open-data
    commit: de406e16e0de4601e221600a2cd5379eb36246c2

旧实验目录：
    analysis/asymode_trajectory_r1_20260909

其中旧包目录只是 GitHub 中已经存在的普通目录：
    analysis/asymode_trajectory_r1_20260909/_package/ASYMODE_CC_STANDALONE_20260909

可以直接读取，不需要取得任何 ZIP。

先检查远端是否已有更新的修复结果、新实验代码或论文。
记录其 commit、文件差异和适用协议。可复用通过核验的成果，但不得静默改变上述历史数据身份。
仓库旧 prompt 中的 DIRECT/SR/旧 synthetic 安排不能覆盖本指令。

### 2.2 必须读取的文件

先读数据：
    data/README.md
    data/SHA256SUMS.txt
    configs/panel_manifest_g3-all-26.json
    scripts/verify_open_data.py
    scripts/build_panel.py
    scripts/build_drivers.py
    scripts/fetch_era5.py
    data/aneel/README.md
    data/aneel/MANIFEST.json
    scripts/fetch_aneel.py

再读旧实验：
    analysis/asymode_trajectory_r1_20260909/results/START_HERE.md
    analysis/asymode_trajectory_r1_20260909/results/NEGATIVE_AND_FAILURES.md
    analysis/asymode_trajectory_r1_20260909/results/STAGE1_TRIALS.csv
    analysis/asymode_trajectory_r1_20260909/results/STAGE2_TRIALS.csv
    analysis/asymode_trajectory_r1_20260909/selection_lock.json
    analysis/asymode_trajectory_r1_20260909/run_campaign.py
    analysis/asymode_trajectory_r1_20260909/data_resolution/DATA_MANIFEST.json

再读上述 _package 目录中的：
    DATA_CONTRACT_ZH.md
    metadata/unit_selection.csv
    metadata/LEDGER_REFERENCE.json
    metadata/ACTUAL_COLUMNS.json
    code/core_models.py
    code/data_contract.py
    code/prepare_local_data.py
    code/preflight.py
    code/training_primitives.py
    SHA256SUMS.txt

不要直接执行旧 REPRODUCE.sh：
它会从头训练旧实验，不是 checkpoint-only 重放。

### 2.3 本轮新增目录

所有新增研究代码和记录写入：
    analysis/net_asym_neural_era5_20260912/

第一步将本指令保存为：
    PROTOCOL_ZH.md

同时生成机器可读：
    CONFIG_LOCK.json
    SOURCE_COMMITS.json

以后任何必要修订必须记录原因、时间、影响范围和是否已接触评价结果。

---

## 3. 数据盘点：不能只读 README 就宣布数据齐全

创建 DATA_INVENTORY.csv，记录：
    数据名、仓库相对路径、源URL、字节数、SHA256、字段、
    年份、时间粒度、许可证、核验状态、用途、缺口。

实际打开二进制文件，检查 shape、字段、时间、ID 和数值范围。
旧 manifest 的 PASS 不是本轮自动 PASS。
验证脚本中的 skip 不算通过。

优先读取仓库已有派生数据，不必重新下载全部原始气象网格。
缺文件时：
1. 检查其他已记录分支、Git LFS、仓库说明和明确的数据目录。
2. 检查当前项目及配置指向的本地数据，不无差别遍历整台机器。
3. 尝试仓库指定的官方来源。
4. 仍缺失则在本轮分支写 MISSING_DATA_REQUEST.md，明确缺哪个文件或字段、
   尝试过什么来源、阻塞哪个子任务；继续其他可执行任务。

不要索要已删除的交接 ZIP，也不要把“补充包里应该有”当作数据来源。
官方原始文件本身采用 ZIP 格式可以读取；禁止的是用 ZIP 作为本轮交接和交付方式。

---

## 4. A：美国 EAGLE-I + ERA5，必须纳入正式运行

### 4.1 数据来源与语义

停电目标：
    data/interim/panel_<YYYY-MM-DD>.npz

天气输入：
    data/interim/drivers_<YYYY-MM-DD>.npz

事件清单：
    configs/panel_manifest_g3-all-26.json

基准应包含 26 对文件，必须逐对核验，不凭清单代替实际检查。
ERA5 是输入 x；EAGLE-I 是目标 y。不能只下载 ERA5 就称完成停电实验。

检查：
- panel 的 y、observed、denominator、fips、ts。
- driver 的 X、channels、fips 和实际时间字段。
- y 与 observed 的形状；mask 的布尔语义。
- 15 分钟停电时间与小时天气时间。
- county FIPS 是否一致，必要时按 key 重排，而不是按行数默认对齐。
- 分母是否有限且为正；已观测 y 是否有限、是否越界。
- 天气单位、通道顺序、非有限值及构建代码中的插值规则。

存储天气预计为12通道：
    cape, cloud, gust, precip, pressure, rh,
    snowfall, soil_moisture, t2m_c, u10, v10, wind_speed

config 中额外的 clock_sin/cos 是生成特征，不能把不存在的第13/14列当原始天气列。
若字段与预期不同，核查构建代码并记录 schema 适配，不能猜测时间起点或通道含义。

observed=False 不能补零后进入损失。
部分 observed=True 的零来自仓库的密集化规则，不等于每格均有原始零报告。
保留现有分母，不用评价期 y 反推分母。记录历史年份使用固定客户分母的限制。
原始目标异常不能通过 clipping 偷偷“修好”。

### 4.2 天气信息合同

本轮正式使用：
    RETROSPECTIVE_KNOWN_REANALYSIS_CONDITIONAL_RESPONSE

即两个模型均获得未来24小时的 ERA5 再分析路径，研究给定外部条件后的响应拟合。
必须明确：这是回顾性、给定再分析天气的条件响应实验，
不是预测时已获得真实未来天气的部署实验。

不需要巴西天气。
不需要为了本任务重新申请 CDS 账号；已有派生数组齐全时直接使用。
发现构建代码使用双向插值，应记录，不能把这些输入改称历史可得天气预报。

### 4.3 样本构造与划分

L=24小时，H=24小时，预测起点每6小时取一次。
从15分钟库存取精确 UTC 整点快照，不用四个季度小时均值替代库存状态。

输入：
- 过去24小时 y，不含当前 y0；y0 单独传入。
- 过去24小时12维天气。
- 本任务明示给定的未来24小时12维天气。
- 每步外部特征：当小时12维天气、UTC时刻sin/cos、weekend。

origin context 固定，rollout 中不重新读取真实历史。
不添加邻县、地理 kernel、县ID embedding 或额外静态变量。

全部49个所需状态及输入天气必须合法。
双方使用完全相同的窗口清单。不要分别按模型筛选或补值。
不要根据未来峰值、是否发生第二波、是否“受灾明显”来选择县。

冻结划分：
- train：2018–2021。
- validation：2022。
- evaluation：2024。
按当前26事件清单，预计分别为14、6、6个事件；以文件审计核验数量。
不额外寻找2023来改变划分。
这些数据属于既有研究接触过的回顾性资料，不叫 fresh holdout。

同一 (FIPS, origin_UTC) 重复窗口仅保留一次，按较早 event_day 归属，保存去重映射。
检查不同 split 是否共享同一县的原始历史／目标时间，禁止交叉泄漏。
同一 split 内重叠事件窗可保留其非重复起点，但不确定性分析按事件窗重叠连通分量聚类。
不把每个县或每个滚动窗口当作独立风暴。

### 4.4 主评分

采用 event→county→origin→lead 逐级等权的 path MSE；
训练采样对应为事件等概率、县等概率、起点等概率。
同时报告1h、6h、24h端点，均来自同一批24小时预测路径。

保存每个事件、县、起点、模型、seed 的预测及损失。
主表之外必须给逐事件结果、配对风险差和不利案例。
不允许只交总体两行均值。

美国任务不因缺少巴西天气而降级成“只做数据检查”。

---

## 5. B：ANEEL 固定台账与历史选参修复

### 5.1 优先用 GitHub 已存在的小台账

先查并核验：
    analysis/asymode_trajectory_r1_20260909/data_resolution/rebuilt_selected_ledgers.npz

基准文件预期：
    bytes: 3508338
    SHA256: bc51d2f326ef3f18b13f0da60523385c9ffbe7b085e661061ef03ed3794e39dc

更关键的是 _package/metadata/LEDGER_REFERENCE.json 中192个数组的指纹：
转为 contiguous little-endian float64 后逐项核对。
容器重压缩导致文件hash改变时，不能仅凭文件名接受；须核对数组身份。

使用 included 的24个collection、16家公司；group编号不连续。
台账通过即可训练，不必等待全国年度原始文件。

缺台账时，检查已记录本地目录，再按 data/aneel/MANIFEST.json 获取2018/2019原始文件，
复用原 prepare_local_data.py 重建。
不要默认下载全部年份。Parquet 与历史 ZIP 可能不是同一快照，
格式转换或重新下载后仍必须核验最终数组身份。

### 5.2 不能改变的观测定义

- 一行是中断记录，不是一场独立风暴，不是唯一客户ID。
- 公司与consumer-set联合key；ID保留前导零。
- 使用冻结2018分母，不改为2019中位分母。
- 库存为加权活动记录之和，不自动等于去重客户停电比例。
- 活动区间 [start,end)，小时流量区间 (t,t+1]。
- 保留原naive时间语义，不猜测巴西时区或DST。
- 按完整记录去重，不能合并不同馈线恰好相同的起止时间。
- 不新增原因／中断类型过滤；非计划中断不自动等于天气中断。
- ends、starts、area 可用于审计，不作为预测输入或方向监督。
- 不把记录ending数量直接解释为客户恢复hazard。

L24/H24，保持 month-interior mask 和48h purge。
六个分区每collection的起点数应为：
    fold_A_fit: 3745
    fold_A_val: 1920
    fold_B_fit: 5665
    fold_B_val: 1920
    full_2018: 7608
    reused_2019: 7608

全年度共182592个collection-origin，但不是182592个独立事件。
ANEEL本轮只用历史状态、set与calendar，不声称已控制同样天气。
2019始终标为 reused retrospective evaluation。

### 5.3 定向修复，不重新扫参

旧 run_campaign.py 的错误是：
    df1.config_id.isin(sum(top.values(), []))
它把其他模型晋级的同名配置带入了本模型最终选择。

只用已有2018表，按 (model, config_id) 筛选本模型晋级候选，
并验证每个候选具有2 folds×3 development seeds的完整记录。

已核查的修复配置：
    NET:  parameter_target=32768, lr=3e-4, updates=3500
    ASYM: parameter_target=8192,  lr=1e-3, updates=1000

最终seeds：5101–5105。
batch512，AdamW weight_decay=1e-4，gradient clipping=1。

先寻找仓库／本地是否已有合格修复checkpoint；有则核验后直接重放。
没有则按上述配置运行。不得用旧 ASYM lr=3e-4、2750步的结果冒充修复结果。
不补跑DIRECT、SR或一步训练消融。

复用原168维context、5维逐步calendar、标准化、cohort和评分权重。
提交选择重算表和锁定配置后，才运行2019模型评分。
公司等权为主，逐company／collection结果全部保留。

---

## 6. C：实际神经 Synthetic，不依赖任何未上传的生成器

不要使用旧 controlled_data.py 的非零初态宽度实验。
下面是新生成机制的完整规格。仓库已有等价实现时先核验再复用；
没有则自行实现并提交源码和测试，不请求补充包。

### 6.1 相同外部路径、相同零初态

H=32。
三个场景：x[k]在前4、8、12小时分别为1，之后为0。
固定 u=0.06，r=0.20，gamma=0.04。

每个局部服务块有 M0=16 个响应组，K0=0：
    damage[k]  ~ Binomial(M0-K[k], u*x[k])
    restore[k] ~ Binomial(K[k], r-gamma*K[k]/M0)
    K[k+1] = K[k]+damage[k]-restore[k]
    Z[k] = K[k]/M0

两个二项变量在给定当前状态下独立，从当前两个互不重叠源池中抽样。
不能先更新damage后再用改变后的库存抽restore。
生成器保证所有路径从零开始并位于[0,1]。

固定 L=32 个等大内部服务块、总人口12288。
每个事件抽取一次 B~Bernoulli(rho)，整条事件保持同一个B：
    B=1：所有块共享同一条随机路径；
    B=0：各块独立生成路径，取人口平均。

实现上可生成 L 条独立 Z，再取：
    Y = Z[0]                   if B=1
    Y = mean(Z, axis=blocks)   if B=0

由此：
    E[Y|x] = mu0
    Cov(Y|x) = c*Sigma0
    c = rho+(1-rho)/L

只改变rho，保持局部过程和条件均值不变。
不是改变总人口、初态或天气，也不是给均值曲线叠加白噪声。

### 6.2 三种条件分布

用8小时场景、gamma=.04、M0=16的有限状态转移矩阵计算 mu0、Sigma0。
令 sd0=sqrt(trace(Sigma0)/32)。
三个协方差乘子：
    c = (target_sd/sd0)^2
    target_sd in {0.011, 0.020, 0.060}
    rho=(c-1/L)/(1-1/L)

必须验证rho在[0,1]，不能静默截断。
对另外两个场景复用同一组c，并重新报告实际条件路径SD，
不能把其他场景仍标为1.1/2.0/6.0个百分点。

有限状态 evaluator 的实现：
- 枚举K=0..16，通过两个二项分布的卷积建立每小时转移矩阵。
- 从K0=0前向传播，得到每小时概率分布、均值和二阶矩。
- 通过跨时点转移矩阵乘积计算完整 E[Y_s Y_t]，得到Sigma0。
- 混合模型的均值保持mu0，协方差为c*Sigma0。
- 用独立Monte Carlo检查精确均值、协方差和零初态。
evaluator只用于生成器校准与独立评价，不供神经训练读取真实均值或真实率。

### 6.3 真正的神经学习

每个分布环境训练一个网络，同时学习三个外部场景。
不是给每一条事件拟合两个／三个参数，也不是用常数Parameter冒充神经网络。

输入：
    origin context = 完整x[0:H]
    step features = (x[k], k/H)
    y0=0

双方都可以读取时间及完整确定的外部路径。
不提供M0、L、rho、c、gamma、B、未来真实Y、随机转移数或精确条件分布。

n∈{32,128,512}，表示每场景训练事件数；三个场景总计3n。
每场景另设独立：
    validation: 512 events
    calibration: 512 events
    test: 2048 events

DEV dataset seeds：6201–6203。
FINAL dataset seeds：7201–7205。
通过SeedSequence明确派生不同stage/scene/law/split的独立随机流。
同一个配对seed下，两模型使用同一数据和同一minibatch事件ID。
不同n可用同一最大训练池前缀，但不得与validation/calibration/test重叠。

主网格：3分布×3样本量×5seeds×2模型，共90次最终神经拟合。
另做gamma=0、n=128、3分布×5seeds×2模型，共30次对照。
对照使用同一组c，不为制造结果重新选择分布。

gamma=0时的网络优劣、是否出现风险交叉，都必须实际测量。
禁止把低维理论阈值当成神经模型必须满足的结论。

---

## 7. 模型核心、公共调参和工程检查

### 7.1 复用同一个模型核心

以基准 _package/code/core_models.py 为实现起点，
只暴露 NET/ASYM，保留其残差MLP、线性skip、初始化及参数计数逻辑。
允许根据任务改变context_dim、clock_dim、horizon，不改变科学比较。

接口：
    prediction = model(context, y0, step_features)    # [batch,H]

内部状态使用原始比例；仅输入/损失按fit scale标准化。

NET：
    y = y + scale * net(concat(context, step_x, y/scale))

ASYM：
    u,r,stay = softmax(concat(rate_logits(context,step_x), zero_logit))
    y = y + u*(1-y) - r*y

第三个softmax槽是stay，不是第三个物理流。
预测期不能重新编码包含真实未来观测的context。

主损失：
    mean(((prediction-target)/fit_scale)**2)

两模型采用同一信息、样本、路径权重和标准化数据范围。
原始输出进入主评分；统一[0,1]输出裁剪单列敏感性，不回灌递推。

### 7.2 新任务调参

美国任务：
- 6候选：参数目标8192/32768 × lr=3e-4/1e-3/3e-3。
- DEV seeds8101–8103，仅用2018–2021训练、2022验证。
- 每候选最多3000更新，每250更新验证。
- 按三DEV seed平均event-equal path MSE选配置。
- final更新数为获选配置best_update的中位数，取最近250，范围250–3000。
- FINAL seeds8201–8205，仍仅用2018–2021拟合。
- 2022不静默并入训练，2024只用于冻结后的评价。

Synthetic：
- 同样6候选，使用DEV seeds6201–6203。
- 只在中间c、每场景n=128、三个场景混合的DEV环境选配置。
- 验证指标为三个场景等权的新事件path MSE，不使用精确均值。
- 锁定配置和final更新数后，应用到全部分布、样本量和零反馈对照。

两个新任务均使用batch512、AdamW、weight_decay=1e-4、clip_grad_norm=1。
并列时优先较小参数目标，再较小lr。
最终模型容量可以因验证选择不同；不要宣称最终参数完全相等。

ANEEL严格使用第5节修复配置，不套用新的调参过程。

### 7.3 先过正确性检查，再正式运行

必须检查：
- CPU FP64与实际设备FP32的forward、loss和关键梯度。
- NET能表示一般仿射净响应，不是signed source-pool。
- ASYM从合法起点递推保持[0,1]。
- 改变target不改变forward。
- 两模型的训练／评价sample IDs一致。
- 24步与32步的shape、时刻对应和端点索引正确。
- 从checkpoint-only重放能复算预测和评分。

先做100更新profile，记录CPU/GPU、Torch/CUDA、峰值显存和吞吐。
预加载数据；禁止每个batch重新解压NPZ或读CSV。
初始使用FP32，不通过降低精度掩盖不稳定。

使用当前已有计算资源，不新开付费服务器。
本轮默认最多24个累计GPU小时，预留15%用于评分和重放。
预算不足时，在final评价前锁定对称缩减方案：
先将新任务final seeds从5减至3，再将DEV更新从3000减至1500。
美国与Synthetic都应保留正式配对运行，不把美国任务直接取消。
ANEEL已锁定修复不随意改步数或seed数。
任何仍无法完成的范围明确标为未运行；不按赢家选择删减。

---

## 8. 结果必须连接分布，不只交两行总体均值

所有比较方向固定：
    Delta = MSE_NET - MSE_ASYM
正数支持双流，负数支持单流。

### Synthetic

报告：
- 3场景×3分布×3样本量的逐seed结果。
- 新事件path MSE与对精确条件均值的MSE，二者分开。
- 实际条件均值、协方差、路径SD、跨时相关结构。
- 理论诊断与实际神经风险的对应及不一致。
- 1/6/24/32h端点、越界、参数量、耗时和学习曲线。
- gamma=0对照。

对冻结的两网络，用独立calibration events选择较低风险者，
再在未用于校准的test上评价，报告与always-NET、always-ASYM相比的风险。
这是两模型之间的选择规则，不是新增第三种动力学模型。
接近平局时报告不确定性，不强迫每个格子判赢家。

### 美国与ANEEL

同时给overall及逐event/company/collection的配对差。
不把mean-company-relative gain与ratio-of-means gain互换。
美国按事件重叠分量配对重采样；ANEEL保留共同calendar的168h块敏感性。
先逐seed算平方损失，再聚合；不能先平均预测后把ensemble风险当seed平均风险。
县、小时、滚动起点和训练seed都不能冒充独立事件样本。

真实数据只能做有支持的条件分布诊断：
- 先用训练期定义条件分组／相似性规则。
- 记录组内天气与历史仍有多少差异，以及有多少独立事件／时间块。
- 样本不足或无法估计时填NA，不造出精确Lambda。
- 不用评价期标签决定匹配规则或所谓高／低噪声分组。
- ANEEL calendar cell不叫同天气；美国相似天气也不叫完全相同x。
- 不用人为的单个n_eff替代整个协方差结构。

至少产出：
1. 美国逐事件主表与完整路径误差曲线。
2. ANEEL修复后的公司／collection配对差。
3. 神经Synthetic的分布×样本量结果表和风险曲线。
4. 校准选择的独立test表现。
5. 参数／运行成本表与全部负结果。

图从保存预测与结果CSV生成。
绝对风险图使用共享尺度，差值图单独使用有符号尺度。
保留真实数据点和不确定性，不平滑出未经观测的选择边界。

---

## 9. 全部通过 GitHub 交付

本轮分支内至少包含：

    analysis/net_asym_neural_era5_20260912/
        PROTOCOL_ZH.md
        CONFIG_LOCK.json
        SOURCE_COMMITS.json
        README.md
        intake/
            DATA_INVENTORY.csv
            DATA_AUDIT.json
            SCHEMA_DRIFT.md
            MISSING_DATA_REQUEST.md
        locks/
            SPLITS.json
            DATA_HASHES.json
            RESOURCE_PLAN.json
            SELECTION_LOCKS.json
        code/
        tests/
        checkpoints/
        predictions/
        results/
            us_era5/
            aneel/
            synthetic/
        figures/
        logs/
            TRIAL_REGISTRY.jsonl
            ENVIRONMENT.json
            NEGATIVE_AND_FAILURES.md
        REPLAY_ONLY.sh
        REPRODUCE_ALL.sh
        RESULTS_ZH.md
        SHA256SUMS.txt

REPLAY_ONLY.sh必须只加载checkpoint并评分，不能重新训练。
REPRODUCE_ALL.sh才允许完整重跑。

阶段性commit：
1. 数据盘点、协议、split/resource锁定。
2. 实现与测试。
3. 实际结果、预测、图和解释。

提交并push到本轮独立研究分支；不修改main或已有实验分支，不自动合并。
只stage明确列出的本轮文件，不使用git add .把私人文件或其他项目带入提交。
保留各数据源归属和许可证，不把ANEEL派生数据与美国数据混成一个无来源文件。

单个预测文件控制在适合GitHub的大小，按任务/模型/seed/event分片。
大checkpoint用仓库可用的Git LFS或同仓库Release独立文件；
必须有hash、读取方式和验证记录，不生成交接ZIP，不上传凭据。
原始大数据可以由官方URL+hash重建，不必重复上传全国原始档案。

至少对每个任务每模型的一个seed做独立评分函数核查，
最终主表所有checkpoint执行重放。
推送后从新目录fetch本轮commit，验证关键文件可读、hash一致，
并至少完成一个小规模checkpoint-only重放。

若GitHub写权限失败：保留本地commit，明确报告失败，不声称已经上传。
缺数据清单也要进入本轮分支；不要把整个交付改成附件。

---

## 10. 执行与最终回复

先用不超过12行报告实际找到的数据、固定commit、校验状态、
是否已有合格ANEEL修复、CPU/GPU及预计工时。

然后继续执行，不在“计划完成”处停下。

最终回复只围绕实际完成情况：
- 分支名、最终commit与GitHub结果入口。
- 美国ERA5、ANEEL、Synthetic各自实际运行了什么。
- NET／Asym的正结果、负结果、未决结果。
- 数据缺口与明确未运行范围。
- checkpoint-only重放命令。

不要要求用户提供旧ZIP，不等待巴西天气，
不要把理论曲线、已有报告或未执行的训练写成本轮结果。
