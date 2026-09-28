# Kernel 文献阅读笔记（2026-09-28）

本笔记只讨论公开文献与本仓库的公开数据研究设计；未训练新模型。配套方案见
[KERNEL_PROPOSALS_20260928.md](KERNEL_PROPOSALS_20260928.md)，状态均为 **proposed，待 PI 讨论**。

阅读深度：**M** = 阅读主文的方法、公式或稳定性段落（不是逐页读完全部附录）；**A** = 核对原始论文/会议页面的摘要与书目信息，未核验全部方法细节。每条的“对本项目”是我们的设计推论，不是该论文在停电数据上的结论。读取日期均为 2026-09-28。部分 arXiv 主文通过 Hugging Face paper markdown 读取；缺失时回到作者的 arXiv PDF 或正式会议 PDF。链接统一指向原始来源。

## 1. 时空点过程与非平稳核

| 文献与主来源 | 深度 | 对本项目的用途与边界 |
|---|---|---|
| Hawkes (1971), *Spectra of some self-exciting and mutually exciting point processes*, Biometrika 58(1):83–90. [DOI](https://doi.org/10.1093/biomet/58.1.83) | A；公式结合下条综述核对 | 历史触发的影响核提供“过去的冲击改变未来响应”的语言，但停电比例是存量，不能把每小时比例当作新的点事件。原文也提醒不同机制可能具有相同的二阶统计结构，相关性或漂亮的核图不自动识别物理机制。 |
| Reinhart (2018), *A Review of Self-Exciting Spatio-Temporal Point Processes and Their Applications*, Statistical Science. [主文](https://arxiv.org/pdf/1708.02647), [DOI](https://doi.org/10.1214/17-STS629) | M：§1–2、§3.6 与协变量讨论 | 背景强度与历史触发应分开；核的积分量、边界、缺失事件及观测机制影响解释，不能把共享天气造成的空间聚集都归因于传播。恢复核应保存冲击历史与存量的区别，并把时间记忆、空间联系和地理信息分别作为待检验因素。 |
| Chen, Amos, Nickel (2021), *Neural Spatio-Temporal Point Processes*, ICLR. [主文](https://arxiv.org/abs/2011.04583) | M：§3，式 (7)–(12) | 用连续隐藏状态与事件跳变编码历史，同时区分时间强度和条件空间密度；这支持把历史状态放在率网络内部，而不是只添加县级静态截距。县级小时存量没有逐设备失效时间，不能直接套其点过程似然或把潜在跳变解释为真实事故。 |
| Dong, Cheng, Xie (2023), *Spatio-temporal point processes with deep non-stationary kernels*, ICLR. [主文](https://arxiv.org/abs/2211.11179) | M：§3.1，式 (4) | 把“历史位置/时刻”与“时空位移”分别参数化，再作低秩组合，说明低维参数化不必意味着空间与时间各自孤立。可借鉴为少数地理条件响应模式，但该文的强度非负约束和近似定理不是我们的 stock 闭环稳定性证明。 |
| Zhu, Li, Peng, Xie, *Imitation Learning of Neural Spatio-Temporal Point Processes*, TKDE，2021 online / 2022 卷期 34(11):5391–5402. [主文](https://arxiv.org/abs/1906.05467), [作者书目](https://www.contrib.andrew.cmu.edu/~shixianz/pub.html) | M：§3.1，式 (1)–(2)，异质 Gaussian diffusion | 地点条件化的混合核学习不同的空间影响范围与方向，启发用公开服务区属性条件化恢复联系，而不是只用距离乘一个标量。其 imitation-learning 目标针对离散事件；不应为使用一个空间核而一并替换当前设计加权轨迹损失。 |
| Okawa et al. (2019), *Deep Mixture Point Processes: Spatio-temporal Event Prediction with Rich Contextual Information*, KDD. [原文摘要](https://arxiv.org/abs/1906.08952) | A | 上下文控制核混合权重，提供“少数响应模式 + 输入选择”的简洁思路，可用于不同记忆尺度或县内暴露类型。它不证明人口加权地理会提高停电预测，也不提供当前反馈系统的界。 |
| Zhu, Bukharin, Xie et al. (2021 workshop; 2022 IEEE JSTSP), *Early Detection of COVID-19 Hotspots Using Spatio-Temporal Data*. [主文](https://arxiv.org/abs/2106.00072), [作者书目](https://www.contrib.andrew.cmu.edu/~shixianz/pub.html) | M：§III，式 (1)、(6) | 该文是潜变量 Gaussian process：时间 Gaussian 协方差核与非平稳空间核相乘，适合借鉴地点相关的影响几何。**不是 Hawkes 历史触发核，也不是 GCRK 有序递推的稳定性依据**；其协方差相关不能直接解读为抢修资源流动。 |

## 2. 图上的传播与平滑

| 文献与主来源 | 深度 | 对本项目的用途与边界 |
|---|---|---|
| Li, Yu, Shahabi, Liu (2018), *Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting*, ICLR. [主文](https://arxiv.org/abs/1707.01926) | M：§2.2，式 (1)–(3) | 双向随机游走把有向图上的上游与下游关系放进递推；对恢复核，服务区联系比县间近邻更有明确含义。论文的可学习扩散卷积系数并不天然满足我们要求的非负、行和受限，所以不能直接引用它来证明单位界。 |
| Yu, Yin, Zhu (2018), *Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting*, IJCAI. [会议页](https://www.ijcai.org/proceedings/2018/505), [主文](https://arxiv.org/abs/1709.04875) | M：§3.2 | 局部图滤波与门控时间卷积是高效组合，但图滤波的作用取决于图所表示的关系。它适合作为“时空结构应一起处理”的参照，不能说明抽样县图已代表电网或资源共享。 |
| Wu, Pan, Long, Jiang, Zhang (2019), *Graph WaveNet for Deep Spatial-Temporal Graph Modeling*, IJCAI. [会议页](https://www.ijcai.org/proceedings/2019/264) | A | 自适应邻接矩阵针对固定图漏边问题，提示当前抽样图的缺失是结构问题。为避免县 ID 记忆及不可解释的稠密联系，本线优先用公共协变量参数化稀疏权重，并单独检验真实联系是否优于匹配随机联系。 |
| Poli et al. (2019), *Graph Neural Ordinary Differential Equations*. [原文](https://arxiv.org/abs/1911.07532) | A | 把图层写成连续隐藏动力学，为同一递推内结合时间与空间提供统一记法。连续深度本身没有单位界、因果识别或预测收益保证，仍须约束生成算子及离散格式。 |
| Chamberlain et al. (2021), *GRAND: Graph Neural Diffusion*, ICML. [主文](https://proceedings.mlr.press/v139/chamberlain21a/chamberlain21a.pdf) | M：§3，右随机矩阵与离散稳定性 | 非负行归一化联系给出最大值原理，是使用 Markov 混合保存县状态最大范数的直接参照。连续扩散与小时离散传播必须分别检查；输入/状态依赖注意力的有界性也不等于两条不同轨迹的收缩。 |
| Eliasof, Haber, Treister (2021), *PDE-GCN: Novel Architectures for Graph Neural Networks Motivated by Partial Differential Equations*, NeurIPS. [主文](https://proceedings.neurips.cc/paper_files/paper/2021/file/1f9f9d8ff75205aa73ec83e543d8b571-Paper.pdf) | M：§3.1，Theorems 1–2 | 原文区分耗散扩散与保留能量的二阶双曲动力学，解释为什么纯平滑可能抹掉邻县差别。这里的 hyperbolic wave 不能直接等同于风暴平流；若今后引入非耗散空间分支，要有机制和单独离散稳定性论证。 |
| Rusch, Chamberlain, Rowbottom, Mishra, Bronstein (2022), *Graph-Coupled Oscillator Networks*, ICML. [会议页](https://proceedings.mlr.press/v162/rusch22a.html) | A | 受控阻尼振子把过平滑与平衡点稳定性联系起来，提醒“更强扩散”不是唯一空间核。停电恢复一般没有周期振荡证据，故本轮不直接加振子，只保留为局部差异持续被平滑时的后续参照。 |

## 3. 稳定递推与输入依赖状态空间

| 文献与主来源 | 深度 | 对本项目的用途与边界 |
|---|---|---|
| Haber, Ruthotto (2017), *Stable Architectures for Deep Neural Networks*, Inverse Problems. [原文](https://arxiv.org/abs/1705.03341), [DOI](https://doi.org/10.1088/1361-6420/aa9a90) | A | 把网络看成离散动力系统，强调前向扰动与反向梯度的稳定性应进入架构设计。可支持先证明再筛选的路线，但不能代替本项目新递推的离散界。 |
| Chang, Chen, Haber, Chi (2019), *AntisymmetricRNN: A Dynamical System View on Recurrent Neural Networks*, ICLR. [主文](https://arxiv.org/abs/1902.09689) | M：§3–4，式 (8)–(13) | 反对称部分控制方向而不直接增加线性能量，阻尼与步长共同影响离散稳定性；GCRK 的隐式 resolvent 是不同的离散实现。尤其不能从“连续系统反对称”直接推断任意显式步长稳定。 |
| Erichson et al. (2021), *Lipschitz Recurrent Neural Networks*, ICLR. [主文](https://arxiv.org/abs/2006.12070) | M：§3–4，Theorem 1 与式 (6) | 线性耗散与 Lipschitz 非线性分开，使反馈增益可分析。恢复核若读入预测存量，必须控制整条 p→state→r→p 回路；只有隐藏状态有界仍可能对扰动很敏感。 |
| Chen, Rubanova, Bettencourt, Duvenaud (2018), *Neural Ordinary Differential Equations*, NeurIPS. [主文](https://arxiv.org/abs/1806.07366) | M：§2，伴随公式 | 伴随思想有助于低内存反传，但本项目是固定小时离散系统，应求所实现递推的离散精确伴随。不能直接把连续伴随代入并假定梯度等价；有 stock 反馈时原有两个独立伴随也不能不加修改地沿用。 |
| Rubanova, Chen, Duvenaud (2019), *Latent ODEs for Irregularly-Sampled Time Series*, NeurIPS. [原文](https://arxiv.org/abs/1907.03907) | A | 观测历史推断初始隐藏状态、之后连续演化的区分，支持让恢复状态从前缀历史开始。当前规则小时面板与遮罩不需要为了“有记忆”就换完整变分 Latent ODE。 |
| Kidger, Morrill, Foster, Lyons (2020), *Neural Controlled Differential Equations for Irregular Time Series*, NeurIPS. [原文](https://arxiv.org/abs/2005.08926) | A | 控制路径把不断到来的外生信息送入隐藏动力学，适合解释天气驱动与初始条件的区别。模型输入应是给定天气、前缀及自身预测，不能在开环窗口吸收后验观测停电。 |
| Gu, Goel, Ré (2022), *Efficiently Modeling Long Sequences with Structured State Spaces* (S4), ICLR. [主文](https://arxiv.org/abs/2111.00396) | M：§2–3 | 状态空间递推与卷积核是同一线性时不变系统的两种表示，结构化矩阵降低长期记忆的成本。可借多尺度记忆和归一化思想，但本线只有 216 小时，未证明瓶颈在序列长度，不宜直接换大 S4 骨干。 |
| Orvieto et al. (2023), *Resurrecting Recurrent Neural Networks for Long Sequences* (LRU), ICML. [主文](https://arxiv.org/abs/2303.06349) | M：§3.2–3.4 | 稳定指数参数化将记忆长度和旋转频率分开，并强调初始化、输入归一化；适合解释为何应检查有效记忆和读出尺度，而不是只看是否有核。单个稳定特征值条件不应被扩大成任意输入依赖、非正规矩阵乘积的统一收缩结论。 |
| Gu, Dao (2023 preprint; 后续版本 2024), *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. [主文](https://arxiv.org/abs/2312.00752) | M：§2–3.2，Algorithms 1–2 | 输入依赖的时间步长、输入及读出映射启发天气选择的恢复记忆，例如相同沉积在不同后续天气下消退不同。Mamba 的扫描加速不自动适用于读入预测 stock 的非线性闭环，也不自动给出 GCRK 单位界。 |

## 4. 直接影响方案的判断

1. **先区分数据对象。** 点过程文献提供的是历史影响结构，目标仍是县级比例存量；不能凭 stock 轨迹分离真实失效和恢复率，也不能以 Hawkes 正反馈替代人口守恒。
2. **先解释图，再增加联系。** DCRNN、GRAND、PDE-GCN 指向三个不同问题：联系方向、平滑稳定性、是否应保留局部差异。它们没有证实 10 m 风就是损伤传播方向，更没有证实抽样县图是抢修网络。
3. **记忆、幅度与读出要一并检查。** 稳定状态能记住输入，不会产生输入中不存在的极端天气。更慢的恢复可能用错误的率分解补偿过小损伤，必须以诊断和配对 null 辨别。
4. **本轮不堆模型家族。** 保留一个损伤 MLP，先讨论恢复隐藏核，再讨论该核内的服务区联系；县内节点核是针对暴露聚合的另一条备选，不与所有新组件同时开跑。

## 5. 引文修订记录

- Hawkes 的本条是 Biometrika 58(1):83–90；不要误链到同年另一篇 JRSS B *Point Spectra of Some Mutually Exciting Point Processes*。
- Zhu–Li–Peng–Xie 的 2021 是 online 年份，正式 TKDE 卷期是 2022；同时写清两者，避免把它当两篇论文。
- COVID hotspot 工作保留 2021 workshop 年份，并注明 2022 期刊版；核类型改正为 GP 协方差核。
- PDE-GCN 的对应机制写为 diffusion / hyperbolic wave；方向性风暴 advection 是本项目另待论证的物理假设。
- 本轮只有摘要层核对的论文已标 A；不能把这份笔记描述成 23 篇均已全文精读。
