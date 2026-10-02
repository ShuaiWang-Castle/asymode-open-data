# 文献核查：区域停电负担与恢复速率（2026-10-02）

只用网页检索，阅读摘要页、arXiv/PMC 的 HTML 页和书目元数据。每条文献都标了核实状态。

被核查的三项内容：

1. **发现 1**：县停电达到峰值后，150 km 内其他县同时停电越多，本县其后 24 小时恢复越慢；邻县安静时，自身停电再大也恢复得快。加入区域负担后，峰后 24 小时剩余比例的事件外 R² 从 0.08 升到 0.53；天气类型标签和静态地理此后没有增量。
2. **发现 2**：恢复率不读区域状态的人口平衡模型，在热带风暴中恢复快了约 4–5 倍。
3. **方法**：在神经 ODE / 人口平衡停电模型的恢复侧加空间核，县 i 的恢复率取决于邻县的停电状态，经存量与时间耦合，在起报点位于风暴开始之后的滚动设定下评估。

## 1. 结论

1. **A（发现 1 是否已有）：部分已有。** "事件越大恢复越慢"和"恢复时长有空间依赖"都已发表。公用事业内部，恢复时长随事件停电数增长，已有定量公式（Ahmad & Dobson 2026；Carrington et al. 2021）；单个停电的恢复时长取决于邻近停电和整场事件的峰值停电数，与自身影响的用户数无关，也不随致灾原因变化（Wu et al. 2022）。县级上，飓风停电时长有显著的空间自相关（Ganz et al. 2023，空间自回归系数约 0.38；Duan & Ji 2025，Moran's I 约 0.75）。
2. **A 中未见的部分**：用一个预测时可观测的量（一定半径内其他县的同期停电比例），在全国县-小时公开面板上、跨五类天气、按事件留出地量化它对峰后恢复的解释力；"邻域安静则无论自身规模都恢复快"这一交互形式；"已知区域负担后天气类型和静态地理无增量"的对照。已有的县级工作仍报告自身峰值规模是时长的显著或最强预测量（Ganz et al. 2023；Willems et al. 2024）。Ganz 的空间杜宾模型含邻县时长和邻县协变量的空间滞后，但只作控制项，没有解释邻县停电规模的作用。
3. **B（恢复率空间耦合的预测模型）：未见同构模型，概念上部分已有。** 点名的模型都不是：Zhu et al. 2026 的空间耦合在停电存量的加性激发项上，恢复率是各单元常数；Chen et al. 2026 的神经 ODE 明确假设各单元独立，恢复率只读本地协变量；OutageDiT 逐县建模，没有显式恢复率；Wei/Ji 的非平稳随机过程是无穷服务台队列；Carrington–Dobson 一系没有空间成分。
4. **B 中最接近的先例**：Danziger & Barabási 2022 的"恢复耦合"模型里，节点修复率是邻域功能水平的函数（跨基础设施层，理论与模拟，未做留出预测）；Wu et al. 2022 的簇破碎模型里，恢复指数随事件强度变化；Duan & Ji 2025 用图注意力网络聚合邻县峰值停电来预测时长类别（黑箱分类，事件内交叉验证）；Zhu et al. 的交叉激发项在数值上也会使"邻域停电多时本单元存量降得慢"。
5. **C（滚动起报 / ETR）：问题定义与基线已有；"实时区域停电状态作预测量"部分已有。** 定义有三类：事件或辖区级 ETR（恢复到 95%–99.5% 的时间）、工单级 ETR（随现场信息更新）、县或网格存量轨迹的滚动预测。用到实时系统状态的有 Jaech et al. 2019（全系统过去 3、8 小时停电数，非风暴日）、Zhu et al. 和 Jiang et al.（邻单元历史停电，进入发生侧）、Duan & Ji（邻县峰值）。未见把邻域同期停电比例作为恢复率调制量的滚动预测。
6. **按本次检索，新的是**：把邻域停电状态以距离核的形式放在恢复流量上，经存量递推与时间耦合，并在风暴开始之后起报的设定下评估——这个组合未见发表。发现 2 这类定量诊断也未见。
7. **不新、不能说首创的是**：大事件恢复慢；恢复有空间依赖；恢复率依赖同时停电数；自身影响规模不决定恢复时长（工单级已有）。机制归因也不能定：Danziger & Barabási 认为大事件的非线性不是抢修资源约束造成的。

## 2. 最相关的 20 篇文献

| # | 作者（年份），刊物 | 题目 | 标识 | 它说明了什么 | 与我们的关系 | 核实 |
|---|---|---|---|---|---|---|
| 1 | Wu, Meng, Danziger, Cornelius, Tian, Barabási (2022), Nature Communications 13:7372 | Fragmentation of outage clusters during the recovery of power distribution grids | DOI 10.1038/s41467-022-35104-9 | 三家美国公用事业约 68 万条停电记录。单个停电的恢复时长与最近邻停电的停电时长正相关，并随整场事件的峰值停电数上升；与受影响用户数无关；规律不随致灾原因变化。簇破碎模型给出未恢复比例按 exp(−2φ(N)t) 衰减。 | **定性上抢先。** 工单尺度、公用事业内部已得到"邻近状态和事件强度决定恢复，自身规模和灾种不决定"。差别：不是县级跨公用事业面板，没有留出事件的预测评估，没有按距离定义的区域负担。 | 已核实 |
| 2 | Danziger, Barabási (2022), Nature Communications 13:955 | Recovery coupling in multilayer networks | DOI 10.1038/s41467-022-28379-5；arXiv:2011.04623 | 2019 年记录 500 多万条停电。平时修复量与损伤量近似线性，大事件落到同一条非线性曲线上。模型中节点修复率是其邻域内支撑网络功能水平的线性函数。作者用 Imelda 洪水的自然实验论证偏离不是资源约束造成的。 | **概念上抢先。** "修复率依赖邻域状态"的模型形式已有，但耦合对象是其他基础设施层，是理论与模拟模型，没有预测评估。对机制归因是直接的不同意见。 | 已核实 |
| 3 | Ahmad, Dobson (2026), PMAPS 2026 | Typical models of the distribution system restoration process | arXiv:2603.16841 | 四家配电公用事业。恢复总时长为以事件规模为条件的对数正态，中位数随停电数按幂律增长，一次项指数 1.35–1.92（作者称超线性），二次项为负。没有空间分析。 | **支持，部分抢先。** 公用事业内部"事件越大恢复越慢"的定量规律。差别：规模是本公司事件的停电数，不是邻域状态。 | 已核实 |
| 4 | Carrington, Dobson, Wang (2021), IEEE Trans. Power Systems 36(6):5814–5823 | Extracting resilience metrics from distribution utility data using outage and restore process statistics | DOI 10.1109/TPWRS.2021.3074898；arXiv:2011.00693 | 把韧性曲线分解为停电过程和恢复过程，给出平均恢复时长、恢复率关于事件停电数的公式。系统级，无空间。 | **背景。** 恢复过程和恢复率的标准定义及其规模依赖。 | 已核实 |
| 5 | Wang, Maharjan, Zheng, Liu, Wang (2026), Scientific Reports 16:6334 | Data-driven quantification and visualization of resilience metrics of power distribution systems | DOI 10.1038/s41598-026-37040-w；arXiv:2508.12408 | 指出已有恢复曲线多不考虑停电量对恢复动态的影响；恢复时间随停电量非线性上升，归因于抢修人员和设备饱和。 | **支持。** 自身规模饱和项的直接依据，也说明"恢复不读停电量"是已被指出的缺口。没有邻域耦合。 | 已核实 |
| 6 | Afsharinejad, Ji, Wilcox (2021), Joule 5(9):2504–2520 | Large-scale data analytics for resilient recovery services from power failures | DOI 10.1016/j.joule.2021.07.006；arXiv:2012.15420 | 纽约、马萨诸塞两个配电网约 170 场事件。恢复按"先修影响大的故障"进行，少数小故障的长尾占大部分中断时长；从中等到极端事件恢复明显恶化。 | **支持。** 严重事件恢复更慢，且由长尾决定。不涉及邻域状态。 | 已核实 |
| 7 | Ganz, Duan, Ji (2023), PNAS Nexus 2(10):pgad295 | Socioeconomic vulnerability and differential impact of severe weather-induced power outages | DOI 10.1093/pnasnexus/pgad295 | 8 场飓风、588 个县；时长定义为峰值到降至 5% 以下。空间杜宾模型含邻县时长和邻县协变量的空间滞后；邻县时长的空间自回归系数约 0.38，自身峰值规模系数为正；社会经济地位每降一个十分位，时长约增 6%。 | **部分抢先。** 县级恢复时长的空间依赖已有，目标定义也相近，按模型设定邻县的峰值停电也已进入回归。差别：空间项只作控制，邻县变量的系数放在补充表里，正文没有解释邻县停电规模的作用（补充表没有读到，应补读）；用的是整场事件的峰值和事后时长，不是起报时可观测的状态；只有飓风；关于静态属性的结论与我们不同。 | 已核实 |
| 8 | Duan, Ji (2025), arXiv 预印本 | Graph Attention Network for Predicting Duration of Large-Scale Power Outages Induced by Natural Disasters | arXiv:2511.10898 | 4 场飓风、501 个县。节点特征含本县峰值停电户数，图注意力在相邻县之间聚合，预测时长三分类；时长的 Moran's I 约 0.75；预测在峰值之后做。 | **最接近的机器学习先例。** 恢复时长预测里已用到邻县峰值停电。差别：黑箱分类，不是恢复率或动力学；事件内五折交叉验证，不是留事件；没有机制解释。 | 已核实 |
| 9 | Willems, Kar, Levinson, Turner, Brewer, Prica (2024), IEEE Access 12:184431–184441 | Probabilistic Restoration Modeling of Wide-Area Power Outage | DOI 10.1109/ACCESS.2024.3509263 | 5 场飓风，县-服务区分辨率，预测恢复到 95% 的时间。七种模型里对数线性回归不输复杂模型（整体调整 R² 0.67）。停电规模（初始停电户数及其空间分布）是最强预测量。 | **部分抢先，也是基线。** 公开数据上的县级恢复时间预测；"空间分布"已被提到。差别：只有飓风；摘要没有说明是否用了邻域同期停电。 | 已核实（摘要层面） |
| 10 | Jamal, Hasan (2023), Int. J. Disaster Risk Science 14(6):995–1010 | A Generalized Accelerated Failure Time Model to Predict Restoration Time from Power Outages | DOI 10.1007/s13753-023-00529-3；arXiv:2302.12157 | Irma、佛罗里达各县。考虑空间依赖的广义加速失效时间模型，拟合改善 12%；投资者所有公用事业占比高、收入低的县停电更久。 | **支持。** 恢复时间有空间依赖。差别：单一事件的横截面模型；静态协变量显著。 | 已核实（摘要层面） |
| 11 | Zhu, Yao, Xie, Qiu, Qiu, Wu (2026), INFORMS J. on Data Science 5(2):102–118 | Quantifying Grid Resilience Against Extreme Weather Using Large-Scale Customer Power Outage Data | DOI 10.1287/ijds.2023.0017；arXiv:2109.09711 | 停电存量的条件强度 = 累积天气项 + 邻单元历史停电的加权指数衰减和。交叉项解释为停电向相邻单元传播；衰减率 β_j 解释为单元 j 的恢复率，是常数。三个服务区，单步滚动预测。 | **不同。** 空间耦合在存量的加性激发上，恢复率不随邻域变化。但交叉项在数值上同样使"邻域停电多时本单元存量降得慢"，是必须对照的替代形式。 | 已核实 |
| 12 | Chen, Fioretto, Qiu, Zhu (2026), IEEE Trans. Smart Grid 17(3):2506–2516 | Global-Decision-Focused Neural ODEs for Proactive Grid Resilience Management | DOI 10.1109/TSG.2025.3642407；arXiv:2502.18321 | SIR 式神经 ODE（未受影响、停电、已恢复）。失效率和恢复率都是本地协变量的神经网络；原文明确假设各单元在本地条件下独立演化。事前预测。 | **不同，最直接的对比对象。** 同为神经 ODE 人口平衡模型，没有空间耦合，恢复率不读区域状态。 | 已核实 |
| 13 | Zhu, Qiu, Xie (2026), arXiv 预印本 | OutageDiT: A Generative Foundation Model for Power Outage Forecasting and Scenario Simulation | arXiv:2609.01896 | 全美县级记录。14 天历史加未来天气协变量，生成 7 天、15 分钟分辨率的停电轨迹。基线有 DeepAR、TFT、CSDI、TimesFM、Chronos。 | **不同。** 逐县建模，文中没有提到邻县或空间耦合，也没有显式恢复率。是滚动起报设定下的强基线。 | 已核实 |
| 14 | Jiang, Xie, Qiu (2024), arXiv 预印本 | Spatio-Temporal Conformal Prediction for Power Outage Data | arXiv:2411.17099 | 沿用第 11 行的模型（邻单元历史停电进入发生率），加图上的保形预测区间，滑动窗口滚动评估。 | **不同。** 说明"邻单元实时停电作预测量"在发生侧已有。 | 已核实 |
| 15 | Wei, Ji, Galvan, Couvillon, Orellana, Momoh (2016), Applied Mathematics 7(3):233–249 | Non-Stationary Random Process for Large-Scale Failure and Recovery of Power Distribution | DOI 10.4236/am.2016.73022；arXiv:1202.4720 | 故障与恢复的非平稳随机过程，聚合后是 M_t/G_t/∞ 队列；恢复时长分布只依赖故障发生时刻（Ike 数据）。 | **不同。** 无穷服务台，没有并发数或邻域依赖。刊物是 Applied Mathematics，不是 Applied Energy。 | 已核实 |
| 16 | Ji, Wei, Mei 等 12 人 (2016), Nature Energy 1:16052 | Large-scale data analysis of power grid resilience across multiple US service regions | DOI 10.1038/nenergy.2016.52 | 纽约上州四个服务区（Sandy 与日常运行）。前 20% 的故障影响 84% 的用户；占 89% 的小故障造成 56% 的用户中断时长。 | **背景。** 恢复优先次序与长尾；恢复不与邻域状态耦合。 | 已核实（摘要层面） |
| 17 | Duffey (2019), Int. J. Disaster Risk Science 10(1):134–148 | Power Restoration Prediction Following Extreme Events and Disasters | DOI 10.1007/s13753-018-0189-2 | 飓风、野火、暴雪等事件的未恢复概率都呈简单指数衰减；野火和飓风趋势相同；结果按损伤和社会扰动程度分组。 | **支持。** "灾种不决定恢复，严重程度决定"。差别：事件级曲线，没有空间成分。 | 已核实（摘要层面） |
| 18 | Jaech, Zhang, Ostendorf, Kirschen (2019), IEEE Trans. Power Systems 34(1):773–781 | Real-Time Prediction of the Duration of Distribution System Outages | DOI 10.1109/TPWRS.2018.2860904；arXiv:1804.01189 | 工单级时长预测：停电发生时给初值，随现场文字报告更新。19 个特征里包含过去 3 小时、8 小时的全系统停电数。数据排除了大风暴。 | **C 的先例。** 实时系统停电状态已被用作 ETR 预测量，但是在非风暴日，用全系统计数，没有距离结构。 | 已核实 |
| 19 | Wanik, Anagnostou, Hartman, Layton (2018), J. Homeland Security & Emergency Management 15(1) | Estimated Time of Restoration (ETR) Guidance for Electric Distribution Networks | DOI 10.1515/jhsem-2016-0063 | 事件级 ETR 定义为 99.5% 用户恢复所需时间。输入为预测停电数、峰值受影响用户、每队每日修复量、每日抢修队数。康涅狄格三场风暴。 | **C 的标准定义与基线。** 恢复时间由停电量与抢修能力之比决定。没有邻域耦合。 | 已核实（摘要层面） |
| 20 | Johnson, Jackson, Baroud, Staid (2024), Environmental Research Letters 19(4):044048 | Can socio-economic indicators of vulnerability help predict spatial variations in the duration and severity of power outages due to tropical cyclones? | DOI 10.1088/1748-9326/ad3568 | 20 场热带气旋的县级时长与停电比例。控制天气和环境因素后，社会经济变量的作用基本可以忽略。 | **支持。** 县级静态社会经济属性对时长无增量。与第 7、10 行相反，说明这一点在文献中有分歧。 | 已核实（摘要层面） |

## 3. 补充背景

除另注外，均已打开页面核实题目、作者和刊物；内容只到摘要层面。

- **工程可靠性**：IEEE Std 1366-2022, *IEEE Guide for Electric Power Distribution Reliability Indices*（[标准目录页](https://www.sis.se/en/produkter/electrical-engineering/power-transmission-and-distribution-networks/general/ieee-1366-2022/)）。重大事件日按 2.5β 法从日 SAIDI 判定并单独统计，行业把大事件当作另一种运行状态，但不对区域耦合建模。Larsen, LaCommare, Eto, Sweeney (2016), Energy 117:29–46, DOI 10.1016/j.energy.2016.10.063：可靠性年度趋势。
- **飓风恢复时间的事前统计模型**：Liu, Davidson, Apanasovich (2007), IEEE Trans. Power Systems 22(4):2270–2279, DOI 10.1109/TPWRS.2007.907587；Nateghi, Guikema, Quiring (2011), Risk Analysis 31(12):1897–1906, DOI 10.1111/j.1539-6924.2011.01618.x；同三位作者 (2014), Natural Hazards 74(3):1795–1811, DOI 10.1007/s11069-014-1270-9。从摘要和检索结果看，预测量是风、降水、土地覆盖等，不含区域同期停电状态。Han、Guikema、Quiring 一系的飓风停电模型针对停电发生量，本次没有逐篇核实。
- **恢复规模律与有限抢修能力**：Duffey, Ha (2013), IEEE Trans. Power Systems 28(1):3–9, DOI 10.1109/TPWRS.2012.2203832。Hines, Apt, Talukdar (2009), Energy Policy 37(12):5249–5259, DOI 10.1016/j.enpol.2009.07.049（检索摘要称大停电规模与持续时间正相关，强度未核实）。Zapata 等 (2008), IEEE/PES T&D Latin America, DOI 10.1109/TDC-LA.2008.4641852：多抢修队排队模型，恢复率隐含地依赖辖区内未修复停电总数。Walsh, Layton, Wanik, Mellor (2018), Infrastructures 3(3):33, DOI 10.3390/infrastructures3030033：智能体模型，ETR 对停电数和抢修队数敏感。Dobson, Ekisheva (2023), IEEE Trans. Power Systems, DOI 10.1109/TPWRS.2023.3292328，arXiv:2208.06985：输电事件时长。
- **县级恢复的空间回归**：Mitsova, Esnard, Sapat, Lai (2018), Natural Hazards 94(2):689–709, DOI 10.1007/s11069-018-3413-x（Irma，空间滞后模型，农村合作社辖区恢复更久）。Best 等 (2023), Natural Hazards 117(1):851–873, DOI 10.1007/s11069-023-05886-2（Isaac，多阶段恢复的空间回归）。
- **EAGLE-I 相关**：Brelsford 等 (2024), Scientific Data 11:271, DOI 10.1038/s41597-024-03095-5（数据说明）。Kar 等 (2022), ORNL 报告 *RePOWERD: Restoration of Power Outage from Wide-Area Severe Weather Disruptions*（[机构页面](https://www.ornl.gov/publication/repowerd-restoration-power-outage-wide-area-severe-weather-disruptions)；指数衰减、多元回归、智能体三类恢复模型）。Ahmad, Dobson (2025), arXiv:2511.12685（按时间和位置重叠提取事件；指出 EAGLE-I 采样的是存量曲线，不记录单个停电的起止）。Li, Ma, Li, Mostafavi (2025), arXiv:2509.02653（四场飓风的恢复时长和相对恢复率）。Jacobs, Piburn, Myers (2026), arXiv:2608.12560（恢复轨迹的层级贝叶斯模型）。Fatehi, Biswas, Nazari (2025), arXiv:2512.22699（图注意力网络预测停电数，不如 LSTM，不建模恢复）。
- **ETR 与评估方式**：Bogireddy, Muthukaruppan, Carls (2025), arXiv:2505.00225（工单级 ETR，随每次修订更新，对比公用事业自己发布的 ETR）。Yang 等 (2026), Risk Analysis 46:e70275, arXiv:2512.06644（STO-CAST，6 小时临近预报并同化实时观测，留一风暴交叉验证）。Essus, Vatsavai, Rachunok (2026), arXiv:2608.24665（随机划分的成绩被时空自相关抬高；留州、留事件评估下模型常常不优于零基线）。Liu 等 (2026), Nature Reviews Electrical Engineering 3(7):436–449, DOI 10.1038/s44287-026-00298-3（综述，只核实了书目）。
- **资源受限恢复的一般模型**：Böttcher 等 (2015), Scientific Reports 5:16571, DOI 10.1038/srep16571（恢复依赖健康人群产生的资源）。Zhang, Liu (2008), J. Math. Anal. Appl. 348(1):433–443, DOI 10.1016/j.jmaa.2008.07.042（饱和治疗函数，只核实了书目）。
- **冲击的空间分布影响恢复**：Rachunok, Nateghi, *The Sensitivity of Electric Power Infrastructure Resilience to the Spatial Distribution of Disaster Impacts*, arXiv:1902.02879（arXiv 页给出的期刊版 DOI 为 10.1016/j.ress.2019.106658）。飓风案例的仿真，说明冲击的空间结构会改变电网恢复过程。
- **未能核实**：美国能源部对 Irene 与 Sandy 的恢复对比报告。检索结果显示 Sandy 恢复到 95% 用了约 10 天，Irene 约 5 天。原文是 PDF，没有读，没有采用。

## 4. 对我们方法的含义

**应引用**

- 现象的先例：Wu et al. 2022；Danziger & Barabási 2022；Ahmad & Dobson 2026；Carrington et al. 2021；Wang et al. 2026；Duffey 2019。
- 县级空间依赖与恢复时间预测：Ganz et al. 2023；Duan & Ji 2025；Willems et al. 2024；Jamal & Hasan 2023；Johnson et al. 2024。
- 模型对比对象：Chen et al. 2026（无耦合的神经 ODE）；Zhu et al. 2026 与 Jiang et al. 2024（发生侧空间耦合）；OutageDiT（逐县基础模型）。
- 滚动起报与 ETR：Wanik et al. 2018；Jaech et al. 2019；Essus et al. 2026（评估方式）；Brelsford et al. 2024（数据）。

**可以主张**

- 在公开的县-小时面板上，跨公用事业、跨天气类型，用一个预测时可观测的邻域停电状态量化恢复放慢，并给出事件外的解释力和配对 null。按本次检索未见同类量化。
- 在已知区域负担的条件下，天气类型标签和静态地理对恢复没有事件外预测增量。措辞限定在"我们的面板、留事件评估、以区域负担为条件"。
- 恢复侧的空间核：邻域停电状态调制本县恢复流量，经存量递推与时间耦合。与 Chen et al.（各单元独立）和 Zhu et al.（存量上的加性激发）区别清楚。
- 不读区域状态的人口平衡模型在热带风暴中恢复过快的定量诊断。

**不要主张**

- 不要说首次发现"大事件恢复慢""恢复有空间依赖""恢复率依赖同时停电数"。这些已有。
- 不要把"自身规模不决定恢复"当作新发现而不提 Wu et al.。还要解释与 Ganz、Willems 的差别：他们在县级发现自身峰值规模显著。Ganz 的模型虽含邻县变量的空间滞后，自身峰值系数仍为正；Willems 的摘要没有说明是否控制了邻域。目标也不同：他们预测到 5% 或 95% 的时长，我们看的是 24 小时剩余比例。
- 不要把机制写成抢修队饱和或互助资源竞争。我们没有人员数据。Danziger & Barabási 的论证是：资源约束应使单位时间修复量趋于饱和，而观测到的是随损伤增加而下降，所以他们归因于对道路等支撑系统的恢复耦合。区域负担也可能只是本地损伤深度的代理。用中性名称，如"区域停电负担"或"空间恢复上下文"。
- 不要说"地理无关"或"社会经济无关"。Ganz、Mitsova、Jamal & Hasan 报告了社会经济、城乡和公用事业类型的效应，Willems 的预测量里有土地利用和用户密度。
- 不要说首次把图结构或实时系统状态用于恢复时间预测。Duan & Ji 已用图注意力网络；Jaech 已用全系统近几小时的停电数。
- 不要把县级存量的下降速度称为维修率。EAGLE-I 记录的是净存量。

**审稿人大概率会要求的对照**

- 与发生侧空间耦合的对照：把同样的邻域停电状态加在损伤侧或存量的加性项上（Zhu et al. 的形式），与恢复侧的乘性调制比较。两者的可区分处在于，自身存量为零的县在加性形式下会因邻县停电而新增停电，在恢复侧形式下不会。
- 与"邻域状态只作普通输入特征"的对照（Duan & Ji 式的聚合）。
- 区域负担是否只是本地损伤强度的代理：控制自身峰值、本地天气暴露后的增量；打乱邻域的配对 null。
- 留事件评估，并与持续性基线、零基线、指数衰减或对数线性回归基线比较（Willems et al. 2024；Essus et al. 2026）。

## 5. 核查范围与局限

- 只读了摘要页、arXiv 与 PMC 的 HTML 页，以及 CrossRef、Semantic Scholar、Europe PMC 的书目元数据。Springer、Nature、IEEE Xplore、INFORMS 的正文页打不开，相关条目标为"摘要层面"。
- 有一次抓取（能源部的一个报告页）实际返回了 PDF。该来源没有采用，列在"未能核实"里。
- 没检索到不等于不存在。2026 年的预印本、会议论文、行业和监管报告覆盖不全。
- 最需要补读的两处：Willems et al. 2024 的正文（"停电规模及其空间分布"具体指哪些变量）；Ganz et al. 2023 的补充表 S1（邻县峰值停电的空间滞后系数的符号和大小）。这两处决定发现 1 在县级被抢先到什么程度。
- 表中数值（空间自回归系数、Moran's I、幂律指数范围、特征个数等）取自页面摘录，引用前应对照原文。

## 6. 复核记录（Claude Code，2026-10-02）

报告由独立助手检索撰写。下列条目由我通过 Crossref、arXiv 和 Europe PMC 的书目接口再次核对了题目、作者、刊物和编号，全部一致：

- 表中第 1 行（Wu et al. 2022，Nature Communications 13:7372）、第 2 行（Danziger & Barabási 2022，Nature Communications 13:955；arXiv:2011.04623）、第 3 行（Ahmad & Dobson，arXiv:2603.16841；会议名未由我核对）、第 7 行（Ganz, Duan, Ji 2023，PNAS Nexus 2(10):pgad295）、第 8 行（Duan & Ji，arXiv:2511.10898）、第 9 行（Willems et al. 2024，IEEE Access 12:184431–184441）、第 11 行（Zhu et al.，arXiv:2109.09711）、第 12 行（Chen et al.，arXiv:2502.18321）、第 13 行（OutageDiT，arXiv:2609.01896）。

内容上我只核对到摘要：

- Wu et al. 2022 的摘要：单个停电的恢复时长与邻近停电的停电时长和停电强度有关，与受影响用户数无关；数据来自三家美国电力公司。"不随致灾原因变化"和记录条数不在摘要里，取自助手读到的正文页，我没有核对。
- Ganz et al. 2023 的摘要：8 场飓风、588 个县；社会经济脆弱性每变一个十分位，预期停电时长变 6.1%。空间杜宾模型和邻县系数的说法来自助手读到的正文页，我没有核对。

其余条目未由我复核，引用前须对照原文。
