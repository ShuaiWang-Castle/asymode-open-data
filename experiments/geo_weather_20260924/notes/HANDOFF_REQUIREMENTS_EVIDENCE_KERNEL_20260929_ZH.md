# 天气—地理复杂调制—冲击—停电：研究交接报告

**供 GPT Pro / Claude 独立审查、诊断和下一版设计使用。日期：2026-09-29。**

本报告汇总 PI（研究负责人）本批工作的要求、公共开发数据中的证据、既有设计的理由及失败结果。它不预设必须保留当前 CRK，也不授权接手者自动运行下一轮训练。正文区分用户要求、实际观测、结构假设和下一步建议。来源均指向结果冻结版本 `54310b30ba4407e0d0ab0c9c9ac5d1ff85726ec0`，避免分支变化造成口径漂移。

## 0. 两分钟了解当前状态

研究对象是美国县级停电比例轨迹。PI 希望找到并建模：**天气过程 → 地理的复杂调制 → 潜在冲击形成、叠加与累积 → 停电**。地理应该改变县如何响应一整段天气历史，不能只作为固定倍率或地理到停电的独立回归器。

本批已完成：原 GCRK/时空核诊断；I18 地理归一化小改；D01–D05 从总体统计到县级结构、高维信息、条件响应和大停电取证；文献阅读；I20 全子集地理条件受控响应核 CRK 的实现与完整五折训练。

**最终结果没有达到预期。** I20 在预登记的严重受灾县集合 S 上，完整144小时设计加权 RMSE 仅下降 **0.365%**，95%改善区间 **[−1.024%, +1.299%]**，没有达到10%目标。全D RMSE上升1.050%，五类平衡MSE上升4.203%；非S县被预测成峰值≥10%的原始计数从4增至133。所有训练已结束，I18/I20监控均暂停，没有新实验排队。

当前最有价值的共同问题不是“如何再加几个模块”，而是：**数据中可见的天气信号为什么没有被转为正确县、正确时段、正确幅度的冲击？真实地理对应究竟能解释多少？** 目前没有充分证据把剩余误差归于单一原因，也没有证明复杂地理关系不存在。

## 1. PI 的要求：接手者需要保留的研究方向

### 1.1 核心叙事与研究对象

以下为用户已明确表达的要求摘要，而非本报告新加的研究假设：

- 核心是“天气—地理复杂影响—冲击—停电”，后续方法和稿件都要围绕这一链条；潜在冲击不是由停电库存唯一反演出的真实损伤。
- **从数据出发建立强 motivation。** 先寻找天气先后、叠加、持续暴露、前期条件和县结构共同出现的症状，再设计核；不能先堆一个模型再补故事。
- **不要被 overall 统计收紧研究空间。** 相反方向、不同峰位、不同阈值、不同记忆时长及县群构成可能互相稀释。应保留连续 county 结构、局部方向、弱信号和失败格，不要求每个局部现象先改善总体MSE。
- 对大幅停电要做过程分析：停电之前及之后的天气、地理条件、触发、累积、恢复/观测混合。回顾分析可使用事后信息，但预测可用信息必须另列，不能混用。
- 要让多种地理联合参与，**不预先指定“坡度×植被”等一两个物理配方**；不能因主效应不显著而禁止联合项。用户希望设计能覆盖所有可能组合；实现必须诚实说明有限状态、地标及样本支持带来的限制。

### 1.2 架构与创新要求

- 地理通过 **kernel 架构进入单一 damage MLP 的隐藏层**。不另开手工 hazard 通道，不把人工“风雨×林冠/土壤”特征直接加进损伤率，也不复制两个同类damage网络取平均。
- **不必完全参照 GCRK。** GCRK是设计深度的参考，不是必须小修的模板。下一方案至少需要更清楚、更有实质内容的结构创新，明确新增能力和代价。
- 不接受只说“动态”“高维”“高阶”或写一个未定义的 H 函数。需要明确输入、状态、写入、保留、转换、读出、核公式、初始化、稳定性、训练和可检验对照。
- 创新目标应与数据中可重复的过程差异连接；组合既有模块、增加参数或画出复杂结构，不自动成为方法创新。
- 期望相对无核 AsymODE 在受灾县取得**约10%或以上的显著改善**。这是要检验的目标，不是可以保证的结果；不能从预期倒推出数据已支持某机制。

### 1.3 实验和操作边界

- 只使用公共数据及本线公共开发面板D。**sealed C 不构建、不读取、不评分；不读取/编辑/导入 `paper_v1/`；不引入受限数据或其衍生结果。** 本报告是研究交接，不是论文写作。
- 模型与统计诊断使用各自明确登记的任务，不能把滚动24小时、一小时差分和固定起点144小时结果直接排成模型优劣表。
- 先完整单seed筛选；值得继续的设计才另行登记多seed、配对NULL及消融。真实地理的净信息最终需**同结构重新训练的对照**；冻结模型把地理换均值不等价。
- 新设计先登记目标、预算、信息口径和理由。当前I20授权已经完成，不自动开启新arm、追加seed、改变损失或只延长候选训练。
- 当前机器8核/16GB；最后获批上限为三折并行、每折两数值线程、nice≥15。并行是计算授权，不是无限扩大实验预算。不得触碰其他会话进程或覆盖已有训练目录。
- 只推送两个research分支，绝不main；显式提交本线文件并做隐私检查。现阶段接手者可先开展只读分析和设计，具体新运行需另定范围。

## 2. 数据究竟是什么

| 项目 | 当前D口径 |
|---|---|
| 覆盖 | 8,457县事件，2,410个县，81个天气system，80个family，68个合并事件组 |
| 天气类别 | 热带、冬季、大尺度风、对流、强降雨 |
| 时间 | 每个县事件216小时；前72小时为历史，起点库存为第71小时，预测窗为72…215共144小时 |
| 停电观测 | EAGLE-I可用15分钟客户停电比例的小时均值；分母为已有2024 modeled customer数；不是人口比例、小时末库存或故障次数 |
| 天气 | 既有ERA5再分析及宿主累积/极值特征；有历史HRRR研究，但I20没有换天气源 |
| 12个基础天气通道 | CAPE、cloud、gust、precip、pressure、RH、snowfall、soil moisture、2m temperature、u10、v10、wind speed |
| 地理 | geo40：地形/坡向分布、林冠/土地覆盖、土壤/排水等，含既有共位与空间平滑字段；县级描述不能恢复县内联合空间分布 |
| 6个背景 | log客户数、RUCC、log人口密度、合作社份额、log1p供电单位数、log1p历史SAIDI；背景不等于纯物理地理 |
| 抽样权重 | 原始逆入选概率 `w_raw=1/(pi_system*pi_county)`；主权重w按天气类中位权重的10倍截顶，报告原始权重敏感性 |
| 留出 | 原event五折，相关family/近日期共享县系统合并；每条县事件恰好一次留出。县可跨事件重复，因此不等于新县/新地区验证 |
| 观测支持 | 预测窗缺失3,592/1,217,808小时（0.295%）；缺测不能当0，不丢零停电或 `used=False` 单位 |

D已被多轮探索，是开发证据，不能把事后形成的子组/解释改称独立确认。50,302个分析窗口及上百万小时也不是相同数量的独立风暴。

**时间语义是关键限制。** p[T]平均的是T、T+15、T+30、T+45分钟；ERA5区间量（降水/降雪/小时阵风）索引T对应(T−1,T]，瞬时量对应T。回顾对齐中的weather[T+1]不能偷换为较早时刻可用的预测输入。当前没有重建标签，也没有量化修正时间口径能带来多少收益。小时缓存未保留每小时有效季度小时数、原始计数和裁剪标记；尖峰不能凭形状一概判为错误。

数据与分割来源：[DATASET_DESIGN](https://github.com/ShuaiWang-Castle/asymode-open-data/blob/54310b30ba4407e0d0ab0c9c9ac5d1ff85726ec0/experiments/geo_weather_20260924/DATASET_DESIGN.md)；[features_v1.py](https://github.com/ShuaiWang-Castle/asymode-open-data/blob/54310b30ba4407e0d0ab0c9c9ac5d1ff85726ec0/experiments/geo_weather_20260924/panel_v1/features_v1.py)；[D读取/变量字典](https://github.com/ShuaiWang-Castle/asymode-open-data/blob/54310b30ba4407e0d0ab0c9c9ac5d1ff85726ec0/experiments/geo_weather_20260924/d04_data.py)；[D05大停电取证](https://github.com/ShuaiWang-Castle/asymode-open-data/blob/54310b30ba4407e0d0ab0c9c9ac5d1ff85726ec0/experiments/geo_weather_20260924/notes/D05_LARGE_OUTAGE_FORENSICS_RESULTS_20260928.md)。

## 3. 数据中已经发现什么：证据、动机与未决问题

### 3.1 D01：显式叠加/顺序交互并未带来稳定预测增量

50,302个有效滚动24小时窗口，控制天气强度、历史、县背景和起始库存，再逐步加入叠加、顺序及地理调制。联合地理调制在三种规格下，热带/冬季负担MSE分别变化 **+3.292%、+3.227%、+5.249%**，五折均未改善。普通顺序项在控制后续天气强度后约+0.079%，区间跨0；简单持续值参考也显示回归探针本身没有建立强基线。

含义：不能以这些统计探针宣称“复杂天气×地理已经证实”；也不能用其失败排除别的表示、局部机制、支持不足或数值问题。D01使用少数地理调制摘要的范围已被用户明确要求拓宽，不能沿用它作为后续搜索空间上限。来源：[D01结果](https://github.com/ShuaiWang-Castle/asymode-open-data/blob/54310b30ba4407e0d0ab0c9c9ac5d1ff85726ec0/experiments/geo_weather_20260924/notes/D01_DATA_FIRST_RESULTS_20260928.md)。

### 3.2 D02：县结构连续，时序与方向确有异质症状，但混合因素很多

- 六个导航县组来自40地理+6背景，平均silhouette仅 **0.173**；前两PC解释46.70%。它们是粗导航，不是六种天然机制或六个应直接硬编码的专家。
- 在完整、内部峰子集中，六组阵风峰到停电峰的加权中位时差为 **6、17、7、16.1、9、1小时**。但T1分天气类后为 **6、−2、4、22.2、2.2小时**；事件构成已能明显改变摘要，不能直接赋予县类型固定时常。
- 同方向不同幅度、原始异号在控制后消失、顺序关联在共同天气范围后变弱等现象同时存在。它们值得追踪，但均不等于净地理作用。
- 坡向与风向投影的可靠支持仅 **14,507/50,302窗口（28.8%）**。无支持格保留缺失，不能填零后解释为没有效应。

含义：保留完整条件曲线、联合天气支持和事件集中度，不能仅看overall，也不能只挑显著格。来源：[D02县级图谱](https://github.com/ShuaiWang-Castle/asymode-open-data/blob/54310b30ba4407e0d0ab0c9c9ac5d1ff85726ec0/experiments/geo_weather_20260924/notes/D02_COUNTY_COMPLEXITY_RESULTS_20260928.md)。

### 3.3 D03：高维信息确实被压缩，但输入方差不是响应机制

| 描述审计 | 发现 | 对设计的合理动机 | 尚不能推出 |
|---|---|---|---|
| geo40线性谱 | 90%描述方差需要11个方向；entropy rank9.59 | 保留连续、多方向地理通路，审查过度压缩 | 真实响应秩为11，或旧4维非线性码一定不足 |
| 坡向分布 | 580县一阶合成度<0.1但二阶≥0.1 | 对向坡面抵消平均方向，仍可能保留轴向结构 | 已确认风向×坡向停电机制；已有八方向字段也不是新数据 |
| 24小时天气路径 | 均值仅保留68.96%设计加权路径变异；前8个DCT系数保留97.66% | 单一均值会丢过程形状，值得保留路径 | 丢失的变异必然解释停电，或必须用DCT |
| 15个顺序摘要 | 对前39个强度/趋势/同步摘要线性投影后，剩余方差中位83.02% | 顺序统计不只是这些线性摘要的重述 | 顺序净效应、非加性、因果或可泛化预测增益 |

许多方向的事件支持高度集中，因此“变量多”不代表可识别的信息多。来源：[D03高维审计](https://github.com/ShuaiWang-Castle/asymode-open-data/blob/54310b30ba4407e0d0ab0c9c9ac5d1ff85726ec0/experiments/geo_weather_20260924/notes/D03_HIGH_DIM_STRUCTURE_RESULTS_20260928.md)。

### 3.4 D04：条件响应探针更能拟合回落，漏掉上升，还会制造假峰

使用严格过去1–48小时天气、观测p(t−1)、连续geo40和背景分析一小时净变化。它是一步条件诊断，不能当作自由运行AsymODE。

- 对8,300个观测增量≥1pp的小时，真实设计加权平均 **+3.459pp**，四个模型平均预测却约 **−0.071至−0.087pp**。
- 地理低秩调制C/A使五类差分MSE **+0.935%**；包含更多天气组合的D/B为 **+3.065%**。负变化子集可改善而正变化子集恶化，总体平均掩盖了分工差异。
- 扩大预测增量会伴随假峰；D在270个无观测正增量的县事件中，预测>0.1pp正峰的无权比例达85.19%。不能以最大预测值变大等同捕捉冲击。
- **100次主拟合加16次历史敏感性拟合，没有一次达到预登记梯度阈值。** 有限值和训练目标下降不等于收敛，曲面符号及低秩分量不能作为已识别机制。

含义：触发、幅度、回落和假峰必须分别检查；求解质量限制必须跟随结论。来源：[D04条件响应](https://github.com/ShuaiWang-Castle/asymode-open-data/blob/54310b30ba4407e0d0ab0c9c9ac5d1ff85726ec0/experiments/geo_weather_20260924/notes/D04_CONDITIONAL_IMPACT_RESPONSE_RESULTS_20260928.md)。

### 3.5 D05：大停电漏峰的最直接证据

固定定义：S=预测窗观测峰值≥10%；J=最大有效相邻小时净增量≥1pp。S共726，J共2,963，交集723。**高库存与大跳升高度重叠，但不是同一任务。**

**（a）幅度缺口不是挪动峰时就能补上。** S中全窗预测最大值/真实最大值的中位数，宿主为 **8.032%无权 / 1.598%设计加权**；I18为 **8.936% / 1.225%**。这是比值的中位数，不是误差改善。即使全窗口找峰，预测仍非常小；时点也并非已准确。

**（b）大跳升附近有天气信号。** 同县同事件、其他固定时钟作对照时，J锚前1小时的热带阵风差为 **+1.745个参考SD [1.216,2.085]**，降水为 **+2.511 [1.866,2.977]**。对流相应为 **+0.221 [0.042,0.440]**、**+0.414 [0.307,0.589]**。对照不要求无停电；区间为逐坐标条件区间，未经多重探索校正。

**（c）历史和近时过程都值得保留。** 冬季S峰前24–7h及6–1h降雪分别高出 **+1.507z、+2.189z**。热带S的风雨反对称坐标与“较早降水、较晚阵风增强”相符，但J对应坐标区间跨0，且S有效事件支持很少。不能据此确认天气顺序造成停电。

**（d）不能把所有大停电都想成平滑慢累积。** J最大跳升后，下一小时净回落≥该跳升80%的设计权比例为 **22.59% [18.74,26.53]%**；持续6小时保持半跳升以上为 **7.69% [4.68,11.76]%**。最大差分选择、报告过程和回归均值可能贡献回落，不能把这些比率当错误率或修复速度。

**（e）地理关联有线索，但严重病例的可比性很差。** 同系统同小时匹配天气/背景时，S设计权覆盖97.79%；再限制前一小时库存差≤2pp，仅剩 **121/726病例、28.89%权重覆盖**。无法用剩余小子集代表全部高峰。固定风险时钟中树冠的条件边际关联在部分天气类为正，但没有排除其他相关地理、基础设施、完整联合天气路径和观测机制。

含义：数据支持认真研究天气→冲击的条件映射与历史状态；**未证明静态地理倍率是唯一瓶颈，未识别任意具体地理联合机制。** 全部1,200地理对比及3,408天气坐标保留，不能只引用上面示例。来源：[D05大停电取证](https://github.com/ShuaiWang-Castle/asymode-open-data/blob/54310b30ba4407e0d0ab0c9c9ac5d1ff85726ec0/experiments/geo_weather_20260924/notes/D05_LARGE_OUTAGE_FORENSICS_RESULTS_20260928.md)。

## 4. 宿主、原GCRK和I18：不要重复哪些误判

无核宿主 `W+Cin` 有唯一damage MLP（32→32隐藏宽度）和recovery网络（宽16），Cin为已有六维县背景。预报由observed p71起步，之后不读未来真实库存：

$$
p_t=\operatorname{clip}\{p_{t-1}+u_t(1-p_{t-1})-r_t p_{t-1},0,1\}.
$$

损伤隐藏变化还需经过W2、后续读出、logit平滑、发生门等，再成为u及stock，因此核状态变大不等于预测峰有效变大。宿主输入已有天气累积/极值，不是仅看当前天气。

原GCRK已经具有：geo40联合非线性4维码，逐坐标记忆，天气输入相关rank-2 skew，非交换时间递推及有界读出。它**不是**静态倍率，也不是完全没有顺序。地理固定参数λ/a/Ω在事件内不变，但转移仍随天气变。新增约610参数。

原核一个可推导的限制是共同写入步长ν=min λ。无skew、恒定驱动下，稳态 `e*_j=(ν/λ_j)d_j`，最慢坐标可压低其他坐标的稳态幅度。这是代数性质，尚未通过实际归因证明是漏峰根因。冻结诊断也反对“状态普遍撞界”“门全关死”的简单故事。

I18仅把逐县地理径向归一改为fit侧公共RMS，未新增学习参数。完整五折相对宿主：热带/冬季平衡MSE **−2.358% [−5.081,+1.218]%**；全五类 **−0.366% [−4.208,+5.544]%**；历史pooled+1 RMSE **−0.484% [−1.530,+0.639]%**。大尺度风MSE **+2.0745%**越过登记+2%线，因此筛选失败，不扩seed。早期三折−1.384%已不是最终结论。

I14时空GCRK试过邻县扩散/上风向混合，未形成稳健宿主优势；旧邻接来自抽样县，近地面风也不等同风暴移动，不应把空间扩散直接当作真实电网/抢修联系。恢复侧、节点暴露等仍是候选方向，历史提议不等于当前已批准实现。

来源：[GCRK结构审计](https://github.com/ShuaiWang-Castle/asymode-open-data/blob/54310b30ba4407e0d0ab0c9c9ac5d1ff85726ec0/experiments/geo_weather_20260924/notes/KERNEL_BASELINE_DESIGN_AUDIT_20260928.md)；[I18完整结果](https://github.com/ShuaiWang-Castle/asymode-open-data/blob/54310b30ba4407e0d0ab0c9c9ac5d1ff85726ec0/experiments/geo_weather_20260924/notes/I18_GEO_RMS_RESULTS_20260928.md)；[早期核诊断综述](https://github.com/ShuaiWang-Castle/asymode-open-data/blob/54310b30ba4407e0d0ab0c9c9ac5d1ff85726ec0/experiments/geo_weather_20260924/notes/KERNEL_REVIEW_20260928.md)。

## 5. I20 CRK 为什么这样设计，具体核是什么

以下为**已经实现并训练失败的候选**，不是本报告建议接手者无条件保留的最终方案。设计意图和效果必须分开。

### 5.1 从证据/限制到组件的映射

| 动机或旧结构限制 | I20选择 | 所检验的假设及代价 |
|---|---|---|
| 用户要求所有地理组合；不预选坡度×植被 | 全40维直接通路+全子集阶数地理核 | 避免人工排除组合；有限地标仍压缩函数空间 |
| 旧核共享幅度门；D05幅度严重不足 | 隐藏水平和偏离并行、逐坐标尺度、地理条件写入 | 是否改善有条件触发；tanh饱和仍可能压幅 |
| 近时天气与较长历史并存 | 四个8维状态，初始记忆3/8/24/72h | 允许多尺度；不是四种命名物理机制 |
| 旧ν=min λ使不同尺度写入耦合 | 每模式用自己的1−ρ写入 | Q=I时稳态不受其他慢模式压制；长τ单脉冲仍被衰减 |
| 天气可能改变既有状态的后续作用 | 天气+地理控制保留、转换和读出 | 超出固定县参数；并未证明需要这些额外自由度 |
| 需要顺序和状态组成变化，同时可审计稳定性 | 多平面Cayley正交转换+耗散 | 状态范数有界；转换不主动放大，表达受固定平面限制 |
| 不能换成新黑箱停电预测器 | 原damage第一隐藏层到W2之间加残差 | 保留即时宿主及原stock/恢复映射；下游仍可能成为瓶颈 |

### 5.2 地理核：所有子集可进入，但不是任意组合均可识别

fit-only预处理后的g∈R⁴⁰，每坐标一维RBF核κ_j，定义：

$$
k_G(g,g')=\sum_{s=1}^{40}\pi_s\binom{40}{s}^{-1}
\sum_{|A|=s}\prod_{j\in A}\exp\!\left[-\frac{(g_j-g'_j)^2}{2\ell_j^2}\right],
\qquad \pi_s>0,\ \sum_s\pi_s=1.
$$

这是PSD地理相似度核。用非负基本对称多项式递推O(40²)求全阶，不枚举2⁴⁰子集；学习40长度尺度和40阶质量。32个地标只由训练侧唯一县的地理选择，得到 `[g; k_G(g,c_1)/√32; …]` 共72维条件。地标核基未白化，不是完整GP或带误差保证的Nyström。

**重要限制：同一阶的子集经平均共享阶质量，没有给每一个子集独立系数；32地标只张成有限函数空间。** 完整g通路补充直接访问，但不保证任意高阶关系都能学会。π也不能解释为物理交互阶数。

### 5.3 天气读入及四类控制头

令h_t为原damage第一隐藏表示，b_t为严格过去的prefix参考（前72h累积、之后固定）。取逐坐标fit RMS尺度：

$$
u_t=[\operatorname{asinh}(h_t/c_h);\operatorname{asinh}((h_t-b_t)/c_v)]\in\mathbb R^{64}.
$$

这里u_t是本节的隐藏天气输入，**不同于宿主的标量损伤率u**。保留水平以避免只看变化丢掉持续高暴露，保留偏离以表示相对先前背景的改变。共享融合及带完整地理/天气直通的头输出：写入d、记忆τ、转换θ、读出ω。隐藏坐标不是天然风/雨物理变量。

### 5.4 状态更新和实际响应核

四个模式m=1…4，各8维，零初态：

$$
x_{m,t}=\rho_{m,t}Q_{m,t}x_{m,t-1}+(1-\rho_{m,t})d_{m,t},
\quad \rho=\exp(-1/\tau),\quad 2\le\tau\le96\ \text{小时}.
$$

- d用两个tanh读入之差作零表示锚定，并除2√8，保证范数≤1；隐藏输入为0不等于物理平静天气。
- Q由每模式四个共享学习平面的rank-2反对称矩阵作Cayley变换后相乘；角度由地理及当前天气控制。Q正交，只转换历史组成，不凭空增加其范数。
- 因为ρ∈(0,1)，每模式状态保持在单位球。各头不依赖前态，所以相同外部输入下两初态轨迹收缩；这不是对不同天气、参数梯度或真实物理的全局保证。

写入b_{m,s}=(1−ρ_{m,s})d_{m,s}，A_{m,t}=ρ_{m,t}Q_{m,t}，则展开为：

$$
x_{m,t}=\sum_{s\le t}\left(\prod_{k=t}^{s+1}A_{m,k}\right)b_{m,s},
\qquad K_m(t,s\mid g,u_{0:t})=C_{m,t}\prod_{k=t}^{s+1}A_{m,k}.
$$

K是**给定系数路径上从历史注入到当前读出的有序响应算子**，不是PSD相似度核，也不是对原天气的完整导数。后续天气可以改变此前注入留下的作用；矩阵非交换和写入/遗忘都能产生顺序差异，因此Q=I也不等于没有顺序。解释原始天气交换时必须重新生成其全部历史输入，不能直接互换隐藏控制系数冒充真实天气干预。

### 5.5 回到唯一damage网络

读出为模式加权拼接后除√4、乘谱范数≤1的P，各ω∈(1/2,2)，因此读出范数≤2：

$$
\widetilde h_t=h_t+\min(1,s/200)\tanh(\alpha)\,c\,r_t.
$$

α初值0，起点精确等于**无核宿主**；分支保留0.2 drop-path。这不是严格嵌套原GCRK。新增15,209参数，远大于旧610；任何收益都不能自动归于地理信息。

实现保留自动微分参考，用一阶精确伴随训练；县分块512只累计梯度，900次全fit更新中的每一步都覆盖整fit集，不是增加随机小批量更新次数。该伴随不支持二阶求导。

### 5.6 本设计主动放弃/尚未解决的能力

- 给定天气后的状态方程是仿射的。相同当前天气但不同前态储量，不能令τ或写入阈值本身不同；一般状态依赖反馈尚未加入。
- 四模式无直接状态交换；由旧32个坐标时间常数改成4个标量τ，存在真实自由度取舍。
- 含全阶地理相似度，不等于恢复了县内天气—植被—地形共位、电网结构或线路暴露。
- 长τ仍通过(1−ρ)压低单次脉冲，有限读出/饱和及原下游门控仍可能压幅；目前未完成I20训练后完整逐层归因。
- 地标退化、高阶局部化、模式/平面塌缩、τ贴边、尾部梯度饱和等**是需要检查的问题，不是已经测得的I20失败原因**。

完整公式、证明、初值、测试与取舍：[CRK完整数学设计](https://github.com/ShuaiWang-Castle/asymode-open-data/blob/54310b30ba4407e0d0ab0c9c9ac5d1ff85726ec0/experiments/geo_weather_20260924/notes/KERNEL_CONTROLLED_RELAXATION_DESIGN_20260928.md)；实现：[controlled_relaxation.py](https://github.com/ShuaiWang-Castle/asymode-open-data/blob/54310b30ba4407e0d0ab0c9c9ac5d1ff85726ec0/src/asymode/controlled_relaxation.py)、[controlled_relaxation_scan.py](https://github.com/ShuaiWang-Castle/asymode-open-data/blob/54310b30ba4407e0d0ab0c9c9ac5d1ff85726ec0/src/asymode/controlled_relaxation_scan.py)、[gcrk_train.py](https://github.com/ShuaiWang-Castle/asymode-open-data/blob/54310b30ba4407e0d0ab0c9c9ac5d1ff85726ec0/src/asymode/gcrk_train.py)。

## 6. 文献依据和创新边界

| 原始文献 | 本批借鉴 | 没有继承或证明的内容 |
|---|---|---|
| [Duvenaud et al., Additive Gaussian Processes, 2011](https://papers.nips.cc/paper_files/paper/2011/file/4c5bde74a8f110656874902f07378009-Paper.pdf) | 全阶子集核可高效求值 | 不是新发明全阶核，不继承GP后验，也不识别因果交互 |
| [HiPPO, 2020](https://arxiv.org/abs/2008.07669) | 历史压缩必须说明对象与尺度 | CRK不是其最优多项式投影，不继承压缩误差保证 |
| [S4, 2022](https://arxiv.org/abs/2111.00396) | 状态递推与响应核的联系 | 本方案系数随天气变化，不等于固定卷积实现 |
| [Mamba, 2023](https://arxiv.org/abs/2312.00752) | 条件化写入、保留和读出 | 连续天气不保证受益；未搬用完整backbone或GPU速度保证 |
| [Neural CDE, 2020](https://arxiv.org/abs/2005.08926) | 路径驱动潜在动态 | CRK不是该模型的直接实现，路径依赖不等于因果识别 |
| [Helfrich et al., scoRNN, 2018](https://proceedings.mlr.press/v80/helfrich18a.html) | Cayley正交参数化 | 有限平面/角度不覆盖任意动态，不保证参数梯度全局稳定 |

领域阅读另包括天气与植被暴露、复合天气、跨地区共享、失效/恢复过程；各篇阅读深度和局限见[天气地理过程文献](https://github.com/ShuaiWang-Castle/asymode-open-data/blob/54310b30ba4407e0d0ab0c9c9ac5d1ff85726ec0/experiments/geo_weather_20260924/notes/KERNEL_GEO_PROCESS_LITERATURE_20260928.md)。这些研究的数据粒度、故障/库存目标及设施信息与本项目不同，不能直接移植为县级机制结论。

**诚实的创新定位：** CRK尝试把“联合地理控制整条天气历史响应算子”具体化为可区分的写入、保留、转换、读出，并以大峰和假峰反驳它。这目前是一个可检验的组合设计假设；没有预测收益，没有独立证明全新理论，也没有证明优于GCRK的科学创新已经成立。接手者应判断还缺什么新的问题刻画、可识别结构或方法贡献，而不是替这次失败包装故事。

## 7. I20冻结比较及结果：不得更换口径

固定 `CRK+Cin / v1_crk_s0`，对照 `W+Cin / v1_host_s0`；相同公共D、原event五折、seed0、每折900全fit Adam更新、原loss/权重。无核宿主为已训练的冻结同预算对照，未为新标签重复训练。训练损失没有针对S重加权；S仅用于登记的评价主集合。

训练仍是掩膜stock平方损失：每折在fit侧计算各天气类 `Z_r=Σ_i w_i Σ_t m_it y_it²`，行权重为 `w_i/Z_r`，再整体缩放；Z_r为0的类别训练权重为0。它不是专为尾部加权的损失，也不等于报告中的pooled设计加权MSE或D04的类别原始MSE平均。损失、训练精度与尾部幅度的关系仍需检查，不能把这一点隐去后只讨论核。

主指标：S中共同观测144小时的设计加权RMSE，目标下降≥10%。下表统一以 `(CRK/host−1)×100%` 表示误差变化，负值好：

| 指标 | 相对变化 | 95%区间 |
|---|---:|---:|
| S RMSE（726县事件） | −0.365% | [−1.299%, +1.024%] |
| 任意正停电RMSE（8,190） | +1.071% | [−0.199%, +2.989%] |
| J RMSE（2,963） | +0.776% | [−0.434%, +2.932%] |
| 全D RMSE（8,457） | +1.050% | [−0.241%, +2.978%] |
| 热带/冬季平衡MSE | +2.494% | [−1.399%, +9.427%] |
| 五类平衡MSE | +4.203% | [+0.709%, +8.900%] |

前四项是1999次合并事件组区间（seed20260928）；末两项是旧2000次类内family区间（seed20260924）。S绝对RMSE从0.062433226到0.062205225；family敏感性改善区间[−0.815%,+1.232%]也跨0。仅2/5折S点估计改善；原始权重和无权均未改变主结论。

S虽有726县事件，但仅55个支持合并组；组加权小时Kish有效数14.57，最大单组占17.89%。所有区间条件于已拟合seed0和观察结果定义的集合，不包含重训/初始化不确定性。

假峰两种方向必须同时报告：

| 集合及报警阈值 | 原始计数（宿主→CRK） | 设计加权率（宿主→CRK） | 率差95%区间（百分点） |
|---|---:|---:|---:|
| 267个观测全零县事件，预测峰>0.1% | 243→187 | 93.071%→35.516% | [−79.406,−16.140] |
| 7,731个非S县事件，预测峰≥10% | 4→133 | 0.027956%→0.303121% | [+0.129847,+0.549851] |

完整观察144小时的敏感性方向一致。全D MAE下降18.44%，但不能以此替换RMSE主目标。I20完整窗口分数不能与I18历史“pooled +1”直接混比：后者实际为73…215后缀；I20另有严格固定起点+h快照（列h−1），也是不同统计量。

71项实现/集成检查通过（1个旧外部参考跳过），并行交接另有24项检查；最终审计9,594项断言通过。双方8,457条OOF恰好一次覆盖，冻结源码/数据/输出hash一致，检查点及导出有限。最终alpha约−1.13到−1.25，没有停在0；这只排除“分支始终未打开”，不证明学到了地理净信息。900步是匹配预算，不是优化收敛证书。

完整结果：[I20中文结果](https://github.com/ShuaiWang-Castle/asymode-open-data/blob/54310b30ba4407e0d0ab0c9c9ac5d1ff85726ec0/experiments/geo_weather_20260924/notes/I20_CONTROLLED_RESPONSE_RESULTS_20260929.md)；[I20全尾部报告](https://github.com/ShuaiWang-Castle/asymode-open-data/blob/54310b30ba4407e0d0ab0c9c9ac5d1ff85726ec0/experiments/geo_weather_20260924/results/v1/i20_cr_tail_s0.json)；[I20最终审计](https://github.com/ShuaiWang-Castle/asymode-open-data/blob/54310b30ba4407e0d0ab0c9c9ac5d1ff85726ec0/experiments/geo_weather_20260924/results/v1/i20_final_audit_s0.json)；执行登记：[I20执行登记](https://github.com/ShuaiWang-Castle/asymode-open-data/blob/54310b30ba4407e0d0ab0c9c9ac5d1ff85726ec0/experiments/geo_weather_20260924/notes/I20_CONTROLLED_RESPONSE_SCREEN_20260928.md)。

## 8. 接手后值得优先回答的问题（建议，不是已确定答案）

1. **误报与漏报能否在同一事件、相近天气过程中被区分？** 对I20新出现的133个非S大假峰、真S仍漏峰案例和普通正确案例作并列过程诊断；同时保留结果无关的固定风险集，避免只看病例。检查地理、先前状态、天气历史、支持和事件集中度，而非只列相关性排名。
2. **幅度到底在哪一层被压掉或错误放大？** 从原始天气/宿主历史特征→第一隐藏层→核写入→状态→读出→W2/logit→发生门/平滑→u/r→p逐层审计。比较短脉冲、持续平台和先后组合；核读出关闭后的模型仍是共同训练后的宿主，不能冒充独立无核模型。
3. **有限表示到底能支持多少“所有组合”？** 检查32地标相似度、有效秩和高阶局部化，全部geo40的梯度/条件作用；不能只展示π阶质量。若提出新表示，说明如何在约68个事件组下共享信息，而非列出指数级候选后逐一挑最好。
4. **动态过程需要什么额外结构？** 可能是状态依赖触发、跨模式作用、多峰过程、非线性先局部后聚合、恢复联系或观测过程；这些都是候选，不要一次全加。每个方向给出现有结构无法表达的具体例子、数据支持与反驳实验。
5. **地理净信息如何和模型容量、地区指纹分开？** 先检查整个设计相对宿主能否兑现，再为幸存者前瞻登记同结构重训的常量/安全置换等对照；固定同县跨事件对应和fit-only处理。容量匹配与信息NULL回答不同问题。
6. **怎样证明顺序而非严重度/持续性/阶段？** 同时保留边际天气与真实路径支持。任意小时shuffle会破坏气象物理和自相关，不能把性能下降全部归于顺序。原始天气变换需重建历史特征，且模型灵敏度、观测关联和因果效应分开表述。
7. **数值、损失和输入限制各占多少？** D04未达梯度准则；I20没有收敛证书。不能只延长候选或删尖峰来兑现目标。若改loss、时间标签或天气源，必须另立同预算宿主和信息口径，分离新数据与核结构收益。

更好的设计不必沿用I20；但应明确究竟解决哪个已验证缺口，以及哪些假设一旦不成立就应放弃。对“至少10%”给可验证的筛选方案和失败条件，而不是承诺。

## 9. 希望 GPT Pro / Claude 各自交付什么

请独立给出以下内容；无需默认认可本报告中对失败的解释：

1. **证据审查表：** 哪些motivation成立、哪些只是线索、哪些推论不成立；注明文件/公式/结果位置。
2. **失败机制的优先级：** 2–4个相互可区分的假设，分别列支持、反证、缺失证据及最低成本诊断。
3. **一个主方案和必要备选：** 数据到模块的对应、完整状态/核方程、所有地理组合的结构资格与有限容量、时间因果性、稳定性及边界、宿主精确退化、参数/计算预算。不要只给结构名词。
4. **创新核查：** 对照最相关一手文献和GCRK/I20，写清已有组件、真正新增的结构/问题、尚未证明的理论或实证主张。
5. **固定比较计划：** 明确S/全D/各天气类、峰幅/对时/假峰、权重和事件支持、区间、同预算宿主、幸存者信息NULL及停止条件；不要看结果后换目标。
6. **可执行下一步：** 区分只读审计、必要新数据和需另批准的训练。若缺大数组/检查点，列确切所需文件，不宣称已完成复现。

可以直接使用这一简短任务说明：

> 请先阅读本报告与第10节的固定版本材料。我们的核心目标是从公共县级数据证明并建模“天气过程→地理复杂调制→潜在冲击→停电”，地理通过唯一damage MLP内的kernel进入，不预选少数地理配方，不以overall无收益排除局部现象。I20完整五折失败，严禁把复杂模型本身当成创新或机制证据。请先独立审查数据动机和失败原因，再给出一个有实质创新、公式明确、可识别、可计算且可反驳的主设计；以相对无核AsymODE的S县事件144小时加权RMSE下降约10%为待检验目标，同时控制严重假峰。不要重写历史指标、宣称因果或自动启动训练。请按第9节交付，明确哪些结论已验证、哪些还需要数据。

## 10. GitHub位置、固定版本与最短阅读路径

公开仓库：[ShuaiWang-Castle/asymode-open-data](https://github.com/ShuaiWang-Castle/asymode-open-data)。**研究内容在research分支，不要只读main。**

- [主研究分支](https://github.com/ShuaiWang-Castle/asymode-open-data/tree/research/geo-weather-process-20260924)
- [同步研究分支](https://github.com/ShuaiWang-Castle/asymode-open-data/tree/research/tropical-evidence-20260926)
- [本报告依据的完整结果快照54310b3](https://github.com/ShuaiWang-Castle/asymode-open-data/tree/54310b30ba4407e0d0ab0c9c9ac5d1ff85726ec0)
- I20模型/训练源登记：`6b6b9647bd96e9452910a1c886e6b18c03674db0`；计算调度登记：`d737ae7b4b27baf9c0ff6fbf758587ff714a4bfa`；结果提交：`54310b30ba4407e0d0ab0c9c9ac5d1ff85726ec0`。

### 第一轮必读（按顺序）

| 次序 | 材料 | 用途 |
|---|---|---|
| 1 | [I20中文结果](https://github.com/ShuaiWang-Castle/asymode-open-data/blob/54310b30ba4407e0d0ab0c9c9ac5d1ff85726ec0/experiments/geo_weather_20260924/notes/I20_CONTROLLED_RESPONSE_RESULTS_20260929.md) | 先明确当前效果和失败，不被早期结果误导 |
| 2 | [D05大停电取证](https://github.com/ShuaiWang-Castle/asymode-open-data/blob/54310b30ba4407e0d0ab0c9c9ac5d1ff85726ec0/experiments/geo_weather_20260924/notes/D05_LARGE_OUTAGE_FORENSICS_RESULTS_20260928.md) | 大峰、天气前史、条件地理、时间/观测边界 |
| 3 | [D03高维审计](https://github.com/ShuaiWang-Castle/asymode-open-data/blob/54310b30ba4407e0d0ab0c9c9ac5d1ff85726ec0/experiments/geo_weather_20260924/notes/D03_HIGH_DIM_STRUCTURE_RESULTS_20260928.md) 与 [D02县级图谱](https://github.com/ShuaiWang-Castle/asymode-open-data/blob/54310b30ba4407e0d0ab0c9c9ac5d1ff85726ec0/experiments/geo_weather_20260924/notes/D02_COUNTY_COMPLEXITY_RESULTS_20260928.md) | 高维信息、县结构和局部异质性 |
| 4 | [D04条件响应](https://github.com/ShuaiWang-Castle/asymode-open-data/blob/54310b30ba4407e0d0ab0c9c9ac5d1ff85726ec0/experiments/geo_weather_20260924/notes/D04_CONDITIONAL_IMPACT_RESPONSE_RESULTS_20260928.md) 与 [D01结果](https://github.com/ShuaiWang-Castle/asymode-open-data/blob/54310b30ba4407e0d0ab0c9c9ac5d1ff85726ec0/experiments/geo_weather_20260924/notes/D01_DATA_FIRST_RESULTS_20260928.md) | 已尝试的统计探针及负结果/数值限制 |
| 5 | [GCRK结构审计](https://github.com/ShuaiWang-Castle/asymode-open-data/blob/54310b30ba4407e0d0ab0c9c9ac5d1ff85726ec0/experiments/geo_weather_20260924/notes/KERNEL_BASELINE_DESIGN_AUDIT_20260928.md) | 原GCRK已经会什么，哪些不足只是推测 |
| 6 | [CRK完整数学设计](https://github.com/ShuaiWang-Castle/asymode-open-data/blob/54310b30ba4407e0d0ab0c9c9ac5d1ff85726ec0/experiments/geo_weather_20260924/notes/KERNEL_CONTROLLED_RELAXATION_DESIGN_20260928.md) 与 [I20执行登记](https://github.com/ShuaiWang-Castle/asymode-open-data/blob/54310b30ba4407e0d0ab0c9c9ac5d1ff85726ec0/experiments/geo_weather_20260924/notes/I20_CONTROLLED_RESPONSE_SCREEN_20260928.md) | 当前CRK公式、理由、固定预算和口径 |

### 代码与数据证据入口

| 内容 | 文件 |
|---|---|
| 宿主、原核、新核 | [asym_host.py](https://github.com/ShuaiWang-Castle/asymode-open-data/blob/54310b30ba4407e0d0ab0c9c9ac5d1ff85726ec0/src/asymode/asym_host.py)；[gcrk.py](https://github.com/ShuaiWang-Castle/asymode-open-data/blob/54310b30ba4407e0d0ab0c9c9ac5d1ff85726ec0/src/asymode/gcrk.py)；[controlled_relaxation.py](https://github.com/ShuaiWang-Castle/asymode-open-data/blob/54310b30ba4407e0d0ab0c9c9ac5d1ff85726ec0/src/asymode/controlled_relaxation.py)；[controlled_relaxation_scan.py](https://github.com/ShuaiWang-Castle/asymode-open-data/blob/54310b30ba4407e0d0ab0c9c9ac5d1ff85726ec0/src/asymode/controlled_relaxation_scan.py) |
| 训练与arm接入 | [gcrk_train.py](https://github.com/ShuaiWang-Castle/asymode-open-data/blob/54310b30ba4407e0d0ab0c9c9ac5d1ff85726ec0/src/asymode/gcrk_train.py)；[screen.py](https://github.com/ShuaiWang-Castle/asymode-open-data/blob/54310b30ba4407e0d0ab0c9c9ac5d1ff85726ec0/experiments/geo_weather_20260924/screen.py) |
| 新尾部评价与独立算术审计 | [evaluate_cr_tail.py](https://github.com/ShuaiWang-Castle/asymode-open-data/blob/54310b30ba4407e0d0ab0c9c9ac5d1ff85726ec0/experiments/geo_weather_20260924/evaluate_cr_tail.py)；[audit_i20_results.py](https://github.com/ShuaiWang-Castle/asymode-open-data/blob/54310b30ba4407e0d0ab0c9c9ac5d1ff85726ec0/experiments/geo_weather_20260924/audit_i20_results.py) |
| 旧平衡MSE定义 | [evaluate_v1.py](https://github.com/ShuaiWang-Castle/asymode-open-data/blob/54310b30ba4407e0d0ab0c9c9ac5d1ff85726ec0/experiments/geo_weather_20260924/evaluate_v1.py) |
| 数据设计与实际特征构建 | [DATASET_DESIGN](https://github.com/ShuaiWang-Castle/asymode-open-data/blob/54310b30ba4407e0d0ab0c9c9ac5d1ff85726ec0/experiments/geo_weather_20260924/DATASET_DESIGN.md)；[features_v1.py](https://github.com/ShuaiWang-Castle/asymode-open-data/blob/54310b30ba4407e0d0ab0c9c9ac5d1ff85726ec0/experiments/geo_weather_20260924/panel_v1/features_v1.py)；[splits_v1D.json](https://github.com/ShuaiWang-Castle/asymode-open-data/blob/54310b30ba4407e0d0ab0c9c9ac5d1ff85726ec0/experiments/geo_weather_20260924/splits_v1D.json) |
| D01–D03结果 | [D01汇总](https://github.com/ShuaiWang-Castle/asymode-open-data/blob/54310b30ba4407e0d0ab0c9c9ac5d1ff85726ec0/experiments/geo_weather_20260924/results/v1/data_first_d01_summary.json)；[县级动态](https://github.com/ShuaiWang-Castle/asymode-open-data/blob/54310b30ba4407e0d0ab0c9c9ac5d1ff85726ec0/experiments/geo_weather_20260924/results/v1/county_dynamics_d02.json)；[高维结构](https://github.com/ShuaiWang-Castle/asymode-open-data/blob/54310b30ba4407e0d0ab0c9c9ac5d1ff85726ec0/experiments/geo_weather_20260924/results/v1/highdim_structure_d03.json) |
| D04评分/响应结构 | [D04评分](https://github.com/ShuaiWang-Castle/asymode-open-data/blob/54310b30ba4407e0d0ab0c9c9ac5d1ff85726ec0/experiments/geo_weather_20260924/results/v1/d04_response_scores.json)；[D04响应图谱](https://github.com/ShuaiWang-Castle/asymode-open-data/blob/54310b30ba4407e0d0ab0c9c9ac5d1ff85726ec0/experiments/geo_weather_20260924/results/v1/d04_response_surfaces.json) |
| D05观测、天气及地理证据 | [观测审计](https://github.com/ShuaiWang-Castle/asymode-open-data/blob/54310b30ba4407e0d0ab0c9c9ac5d1ff85726ec0/experiments/geo_weather_20260924/results/v1/d05_observation_audit.json)；[同县天气对照](https://github.com/ShuaiWang-Castle/asymode-open-data/blob/54310b30ba4407e0d0ab0c9c9ac5d1ff85726ec0/experiments/geo_weather_20260924/results/v1/d05_within_county.json)；[地理匹配](https://github.com/ShuaiWang-Castle/asymode-open-data/blob/54310b30ba4407e0d0ab0c9c9ac5d1ff85726ec0/experiments/geo_weather_20260924/results/v1/d05_geography_matches.json)；[旧模型峰值](https://github.com/ShuaiWang-Castle/asymode-open-data/blob/54310b30ba4407e0d0ab0c9c9ac5d1ff85726ec0/experiments/geo_weather_20260924/results/v1/d05_frozen_models.json) |
| I20完整评分与来源 | [I20全尾部报告](https://github.com/ShuaiWang-Castle/asymode-open-data/blob/54310b30ba4407e0d0ab0c9c9ac5d1ff85726ec0/experiments/geo_weather_20260924/results/v1/i20_cr_tail_s0.json)；[I20判定](https://github.com/ShuaiWang-Castle/asymode-open-data/blob/54310b30ba4407e0d0ab0c9c9ac5d1ff85726ec0/experiments/geo_weather_20260924/results/v1/screen_i20_s0_verdict.json)；[I20最终审计](https://github.com/ShuaiWang-Castle/asymode-open-data/blob/54310b30ba4407e0d0ab0c9c9ac5d1ff85726ec0/experiments/geo_weather_20260924/results/v1/i20_final_audit_s0.json) |
| 历史决策及最新状态 | [RESEARCH_LOG](https://github.com/ShuaiWang-Castle/asymode-open-data/blob/54310b30ba4407e0d0ab0c9c9ac5d1ff85726ec0/experiments/geo_weather_20260924/RESEARCH_LOG.md)；[IDEAS](https://github.com/ShuaiWang-Castle/asymode-open-data/blob/54310b30ba4407e0d0ab0c9c9ac5d1ff85726ec0/experiments/geo_weather_20260924/aris/IDEAS.md)；[CYCLE](https://github.com/ShuaiWang-Castle/asymode-open-data/blob/54310b30ba4407e0d0ab0c9c9ac5d1ff85726ec0/experiments/geo_weather_20260924/aris/CYCLE.md) |
| 领域与动力学文献细读 | [天气地理过程文献](https://github.com/ShuaiWang-Castle/asymode-open-data/blob/54310b30ba4407e0d0ab0c9c9ac5d1ff85726ec0/experiments/geo_weather_20260924/notes/KERNEL_GEO_PROCESS_LITERATURE_20260928.md)；[受控动力学文献](https://github.com/ShuaiWang-Castle/asymode-open-data/blob/54310b30ba4407e0d0ab0c9c9ac5d1ff85726ec0/experiments/geo_weather_20260924/notes/KERNEL_CONTROLLED_DYNAMICS_REVIEW_20260928.md)；[记忆与路径文献](https://github.com/ShuaiWang-Castle/asymode-open-data/blob/54310b30ba4407e0d0ab0c9c9ac5d1ff85726ec0/experiments/geo_weather_20260924/notes/KERNEL_MEMORY_PATH_REVIEW_20260928.md) |

### GitHub提供什么，暂未提供什么

已公开且可直接审查：源码、固定split、设计/执行登记、紧凑JSON、分析图、文献笔记及研究日志。仓库公开状态和固定结果文件的匿名访问已经核对。

大型 `data/interim/panel_v1/features_v1D.npz`、原始天气/停电缓存，以及 `runs/geo_weather_20260924/<label>/fold*/{outer.npz,final.pt,DONE.json}` **未提交GitHub**。公共来源不等于本地加工面板与模型文件已经随仓库发布。只有GitHub时可以审查方法和现有统计，不能声称重新跑过训练或县级预测分析。重现需另取得本线D文件，保持hash和原split；不要为补文件顺手运行含C的构建流程。

已登记D特征SHA256：`f043bb39e8abd48183670e2acc3c0cecebd7107ea9be0e28771912cfee358c48`；实际训练/评价源码、宿主和候选输出摘要见审计JSON。不要调用会导入稿件模块的旧比较入口；本轮安全评价入口为 `evaluate_cr_tail.py` 和已审计脚本。

交接状态：只编写本报告，不新增实验、不修改冻结设计，不重启监控，也不直接向其他模型/人员发送项目内容。PI自行转交本报告及GitHub链接。
