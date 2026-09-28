# 停电、恢复与数据设计文献笔记（2026-09-28）

结论：文献支持给恢复侧加入历史与条件依赖，也支持用受影响资产附近的环境表征暴露；但县级 customers-out 存量不能单独识别故障流、维修流或人员资源竞争。恢复核应先作为可检验的预测结构，不能直接解释成已观测到的真实维修机理。

范围：交接清单的前 18 篇，以及天气物理、基线与抽样部分的 7 项，共 25 项。本文仅做公开文献核读与方案依据整理；未读入评估数据，未训练，未修改稿件。点过程、图算子和状态空间文献另见 `LITERATURE_KERNELS_20260928.md`。

阅读状态说明：**正文核读**表示已打开原文并核读注明的相关章节，不表示逐字通读全文；**摘要/节选**表示本次只能取得出版页摘要或可公开读取的正文节选；标准和书籍另标范围，不能冒充全文阅读。每项题名后保留一至两句用途与局限；文末综合判断中的推导是本项目的分析，不是原文结论。

## 1. 天气、环境与观测对象

**01. Han et al. (2009), “Estimating the spatial distribution of power outages during hurricanes in the Gulf coast region.”** RESS 94(2):199–210；[出版页与 DOI](https://www.sciencedirect.com/science/article/pii/S0951832008000665)，10.1016/j.ress.2008.02.018。**阅读：摘要及出版页正文节选。**

负二项回归与主成分表征用于处理停电计数的过离散及天气/环境变量相关性，并以可观测的风暴特征替代难以用于新风暴的名称指标，支持本项目避免将地理编码变成事件身份捷径。研究预测事件尺度的空间停电分布，不能作为小时恢复核有效、或地理参数具有因果解释的证据。

**02. Nateghi, Guikema & Quiring (2014), “Power Outage Estimation for Tropical Cyclones: Improved Accuracy with Simpler Models.”** Risk Analysis 34(6):1069–1078；[出版摘要](https://onlinelibrary.wiley.com/doi/10.1111/risa.12131)，10.1111/risa.12131（online 2013，卷期 2014）。**阅读：摘要。**

作者发现变量较少的随机森林仍可获得有竞争力的结果，而只用风速不足以刻画停电，支持先检查天气、环境和暴露信息是否真正进入预测。该摘要不支持把本项目低估直接归因于 ERA5，也不能用来证明复杂时空核优于较简单的条件模型。

**03. Guikema et al. (2014), “Predicting Hurricane Power Outages to Support Storm Response Planning.”** IEEE Access 2:1364–1373；[IEEE 出版页](https://ieeexplore.ieee.org/abstract/document/6949604/)，10.1109/ACCESS.2014.2365716。**阅读：摘要。**

用公开数据构造可迁移的飓风停电估计，并在 Sandy 与历史情景中展示用途，支持公开暴露代理和跨地域验证的研究路线。该工作主要服务事件影响范围与响应准备，并不验证 county stock 的 144 h 开环恢复预测；个别存档页的年份与 IEEE 卷期不一致，应以 2014 出版记录为准。

**04. Wanik et al. (2015), “Storm outage modeling for an electric distribution network in Northeastern USA.”** Natural Hazards 79:1359–1384；[作者稿](https://hartman.byu.edu/docs/files/WanikAnagnostouHartmanFredianiAstitha_StormDamage.pdf)，10.1007/s11069-015-1908-2。**阅读：正文，数据、§2.6、模型比较。**

对架空线路周围 60 m 缓冲区提取土地覆盖、并按 2 km 天气网格聚合，比县域面积平均更直接对应受损资产暴露；文中由多个 town 组成的 Area Work Center 也给出恢复组织单元跨行政边界的具体例子。其目标是需人工处置的 trouble spots，且有线路/OMS 数据，因此人口加权节点只是可公开实现的代理，不能声称与线路暴露等价，也不能从工作区设置推定任意邻县间存在人员流动。

**05. Cerrai et al. (2019), “Predicting Storm Outages Through New Representations of Weather and Vegetation.”** IEEE Access 7:29639–29654；[机构存档全文](https://www.nlr.gov/docs/fy19osti/73928.pdf)，10.1109/ACCESS.2019.2902558。**阅读：正文，天气/植被表示、模型与验证设置。**

动态 LAI、风持续时间和天气表示的改进说明同一静态植被环境对天气的响应会随条件变化，给地理条件下的隐藏状态更新提供依据。作者使用事件级故障目标、组合模型和较细天气/资产信息，不能据此为本项目增加灾种分支或宣称核已有收益；其业务预报与 hindcast 的区分也提醒我们明确未来再分析天气条件下的任务边界。

**06. Tervo et al. (2021), “Predicting power outages caused by extratropical storms.”** NHESS 21:607–627；[全文](https://nhess.copernicus.org/articles/21/607/2021/)，10.5194/nhess-21-607-2021。**阅读：正文，风暴对象识别、数据与预测设置。**

从 ERA5 风/气压场识别并跟踪移动风暴对象，再结合森林资料预测影响等级，为时空核考虑移动天气结构提供具体参照。芬兰风暴对象的严重度分类不等于美国县级 customers-out 轨迹，而且风暴平移方向应由对象/场的演变估计，不能直接把 10 m 风向当作传播速度。

**07. Alpay et al. (2020), “Dynamic Modeling of Power Outages Caused by Thunderstorms.”** Forecasting 2(2):151–162；[全文](https://mdpi-res.com/d_attachment/forecasting/forecasting-02-00008/article_deploy/forecasting-02-00008.pdf)，10.3390/forecast2020008。**阅读：正文，数据与动态建模设置。**

以 HRRR 和小时故障报告建模，指出天气与报告存在时滞，以及修复上游故障后才显露其他故障的观测过程，提示学习到的延迟可能同时包含物理与报告延迟。目标是小时新故障/工单计数而非未恢复客户存量，且排除一次严重龙卷风后的长恢复区间，不能将其结果直接当作长恢复尾部的验证。

**08. Kabir, Guikema & Kane (2018), “Statistical modeling of tree failures during storms.”** RESS 177:68–79；[出版摘要及节选](https://www.sciencedirect.com/science/article/abs/pii/S095183201730707X)，10.1016/j.ress.2018.04.026。**阅读：摘要及正文节选。**

树高、胸径、明显缺陷、修剪及邻近变化等与倒树风险相关，提示 canopy fraction 只覆盖地理脆弱性的一部分。数据来自单次风暴且作者明确限制跨地点/风暴应用，不能用其变量重要性证明本项目某种地理编码应占优，也不能把倒树风险直接等同停电存量。

**09. Mukherjee, Nateghi & Hastak (2018), “A multi-hazard approach to assess severe weather-induced major power outage risks in the U.S.”** RESS 175:283–305；[NSF 存档作者稿](https://par.nsf.gov/servlets/purl/10065103)，10.1016/j.ress.2018.03.015。**阅读：正文，摘要、结论与局限。**

将重大停电风险与持续时间放在多灾种、基础设施和社会环境背景下分析，支持保留五类天气并同时报告量级和持续时间。其州级重大事件记录和代理变量无法解析县小时失效/维修过程，缺少基础设施信息会影响解释，也不能从地区关联推出邻县共享维修资源。

**10. Brelsford et al. (2024), “A dataset of recorded electricity outages by United States county 2014–2022.”** Scientific Data 11:271；[全文](https://www.nature.com/articles/s41597-024-03095-5)，10.1038/s41597-024-03095-5。**阅读：正文，观测、Data Records、覆盖与质量。**

EAGLE-I 提供的是每 15 min 报告的未供电客户存量，客户不是人口且覆盖随地区和时间变化；原论文所述发布文件省略零值行，因而缺行可能是零或采集缺口。由此应保留实际版本的覆盖/缺测协议、核实归一化分母，不能将相邻差分直接当成故障和维修事件；旧发布版的省零规则也不能未经核对直接套到当前 harmonized panel。

## 2. 故障、恢复与空间结构

**11. Wei et al. (2014), “Learning Geotemporal Nonstationary Failure and Recovery of Power Distribution.”** IEEE TNNLS 25(1):229–240；[作者稿](https://arxiv.org/pdf/1304.7710)，10.1109/TNNLS.2013.2271853（预印本 2013，卷期 2014）。**阅读：正文，§III–IV、VI–VII。**

将故障到达与依赖故障发生时点的恢复时长分开建模，得到恢复流为故障历史与持续时间分布的卷积，直接支持恢复侧记忆；但 §VI 明确说明仅有聚合存量时，正/负净变化只给出故障/恢复流的下界，不能识别时变恢复时长分布。其无穷服务台队列不是有限人员容量模型，城市间相似性也未构成已识别的资源共享网络。

**12. Ji et al. (2016), “Large-scale data analysis of power grid resilience across multiple US service regions.”** Nature Energy 1:16052；[全文](https://www.nature.com/articles/nenergy201652)，10.1038/nenergy.2016.52。**阅读：正文，基础设施脆弱性、恢复与 Discussion。**

区分设备故障和下游客户服务损失，并显示恢复优先次序与大量小故障形成的长尾对客户受影响时长很重要，支持将损伤历史/剩余负荷放入恢复侧并独立检查 customer-hours。其细粒度公用事业资料包含县 stock 不具备的信息，设备拓扑的非局地影响、道路可达性和维修组织不能仅由邻县距离恢复出来，径向配电网的这些影响也不应统称级联传播。

**13. Liu, Davidson & Apanasovich (2008), “Spatial generalized linear mixed models of electric power outages due to hurricanes and ice storms.”** RESS 93(6):897–912；[出版页](https://www.sciencedirect.com/science/article/pii/S0951832007001305)，10.1016/j.ress.2007.03.038。**阅读：摘要及出版页正文节选。**

在飓风和冰暴计数模型中引入空间随机效应，说明天气/资产协变量之外仍可能有空间残差结构，支持地理核必须与空间 null 做配对比较。相关性可以来自共同天气、未观测设施或报告方式，不足以证明定向故障传播或跨县人员竞争，且事件计数目标没有验证 stock 动力学。

**14. Liu et al. (2005), “Negative Binomial Regression of Electric Power Outages in Hurricanes.”** Journal of Infrastructure Systems 11(4):258–267；[DOI](https://doi.org/10.1061/(ASCE)1076-0342(2005)11:4(258))、[作者上传全文](https://www.researchgate.net/publication/239387945_Negative_Binomial_Regression_of_Electric_Power_Outages_in_Hurricanes)。**阅读：正文，数据、模型解释、局限及结论。**

用变压器数量代理线路暴露、负二项分布处理过离散，并比较不同空间聚合尺度，提示不能把县的资产数量、面积及位置误差混作同一种地理效应。作者明确指出公司/风暴指标限制外推，且这组资料中细尺度土地覆盖未改善预测，因此“加入更多地理细节必有益”不是已有结论。

**15. Nateghi, Guikema & Quiring (2011), “Comparison and Validation of Statistical Methods for Predicting Power Outage Durations in the Event of Hurricanes.”** Risk Analysis 31(12):1897–1906；[原始摘要索引](https://pubmed.ncbi.nlm.nih.gov/21488925/)，10.1111/j.1539-6924.2011.01618.x。**阅读：摘要。**

以 Ivan 训练并用其他飓风验证多种持续时间模型，支持把恢复持续时间作为单独评估对象并做跨系统验证。这里监督的是 outage duration，摘要不足以断言县级相对峰值半衰时间等于维修时长，更不能把当前模型中的瞬时 r 参数当作已识别的实测维修率。

**16. Dobson (2023), “Models, Metrics, and Their Formulas for Typical Electric Power System Resilience Events.”** IEEE Transactions on Power Systems；[作者稿](https://arxiv.org/pdf/2303.07930)，10.1109/TPWRS.2023.3300125。**阅读：正文，过程定义与指标公式。**

把故障与恢复的点过程区分，再从两者导出性能曲线与不同韧性指标，支持同时检查峰值、受影响面积积分和时长，而不是把峰后 SSE 全归于恢复机理。这里的过程参数需要对应的事件时刻信息，平均韧性曲线及其公式不保证单个县事件的 stock 能唯一分解两条流。

**17. Zhu, Yao, Xie, Qiu, Qiu & Wu, “Quantifying grid resilience against extreme weather using large-scale customer power outage data.”** arXiv:2109.09711；[v3 全文](https://arxiv.org/html/2109.09711v3)（首次提交 2021，本文核读的 v3 为 2025-08-05；不要把预印本起始年和修订年混写）。**阅读：正文，方法、预测评估与局限。**

把累积天气影响及邻近地区历史写入 stock 的条件强度，给天气记忆和空间历史依赖提供直接先例，同时作者承认同时失效/恢复使物理分解困难。核读到的预测实验为一步/数小时预测，不是本项目 144 h 开环验证；指数衰减系数和定向历史关联也不能单凭观测数据确认为真实维修速度或因果传播。

**18. Duffey (2019), “Power Restoration Prediction Following Extreme Events and Disasters.”** International Journal of Disaster Risk Science 10:134–148；[全文](https://link.springer.com/article/10.1007/s13753-018-0189-2)，10.1007/s13753-018-0189-2（online 2018，卷期 2019）。**阅读：正文，§2、事件比较及§7。**

用峰后未恢复比例的指数衰减及残余项概括多类事件，并将损伤复杂度、可达性和持续恶劣天气与恢复时间尺度联系，支持条件化、多时间尺度恢复状态。其“人类学习”解释依赖近固定初始待恢复总数等假设，曲线拟合不能证明该解释唯一，更不适合直接搬到仍有新故障持续进入的整段预测窗口。

## 3. 天气物理、基线与抽样

**19. Klawa & Ulbrich (2003), “A model for the estimation of storm losses and the identification of severe winter storms in Germany.”** NHESS 3:725–732；[全文](https://nhess.copernicus.org/articles/3/725/2003/nhess-3-725-2003.pdf)，10.5194/nhess-3-725-2003。**阅读：正文，§2–4。**

以阵风超过当地高分位阈值的相对幅度、非线性响应和人口暴露解释保险风暴损失，提示绝对风速相同不代表相同局地异常与暴露。该经验模型针对德国大范围冬季风暴损失，三次幂不是普适停电物理律，也不能据此给本项目增设手工灾害通道或复制到全部天气类别。

**20. Jones (1998), “A simple model for freezing rain ice loads.”** Atmospheric Research 46(1–2):87–97；[NOAA 托管原文](https://training.weather.gov/wdtd/courses/woc/winter/fcst-hzds/zr-impacts/story_content/external_files/Jones_AtmosRes_1998.pdf)，10.1016/S0169-8095(97)00053-7。**阅读：正文，§2 模型及讨论。**

由降水与风驱动的水通量随时间累积估计径向冰厚，支持“天气历史改变后续损伤/恢复条件”的记忆结构。模型假定撞击水均冻结且不含完整融冰过程，所以天气依赖的遗忘/融化应标为本项目待检验假设，不能称为该文已给出的精确恢复机理。

**21. ISO 12494:2017, “Atmospheric icing of structures.”** 第 2 版；[ISO 官方范围说明](https://www.iso.org/standard/72443.html)。**阅读：官方 Scope 与书目信息；未取得、未阅读全文标准。**

官方范围指出冰载及结冰造成的迎风面积/阻力变化是结构荷载问题，支持天气、地理与持续暴露共同作用的物理动机。范围同时明确电力架空线路结冰由 IEC 标准覆盖，ISO 可用于相关塔/桅结构，因此不能把 ISO 12494 泛称为配电导线故障或县级停电预测标准。

**22. Das, Kong, Sen & Zhou (2024), “A decoder-only foundation model for time-series forecasting” (TimesFM).** ICML，PMLR 235:10148–10167；[会议版本](https://proceedings.mlr.press/v235/das24c.html)、[作者全文](https://arxiv.org/pdf/2310.10688)。**阅读：正文，§3–5、推断与评估设置。**

基于 patch 的 decoder-only 预训练支持跨数据集零样本预测，适合作为只看历史轨迹时的强基线，并要求明确 context、输出长度及滚动方式。2024 原文定义的是不含数据集专属协变量的原始模型，不能拿它替代当前 TimesFM 3.0 的版本/协变量实现说明，也不能用已获得的未来再分析天气让不同信息集的比较变成同条件比较。

**23. Ansari et al. (2024), “Chronos: Learning the Language of Time Series.”** TMLR；[作者全文](https://arxiv.org/html/2403.07815v2)、[终稿](https://www.stat.berkeley.edu/~mmahoney/pubs/2619_Chronos_Learning_the_Lang.pdf)。**阅读：正文，分词/概率预测及§5.5、§6.1局限。**

用缩放、量化和语言模型预测 token 概率；原文特别讨论稀疏序列的平均绝对值缩放与固定取值范围可能截掉大峰值，给峰值/近零区间分别诊断提供具体动机。原始版本的单变量和量化限制不能自动归因到 TimesFM 或后续 Chronos 版本，也不是本项目条件均值收缩机制的直接证据。

**24. Horvitz & Thompson (1952), “A Generalization of Sampling Without Replacement from a Finite Universe.”** JASA 47(260):663–685；[原文扫描件](https://stat.cmu.edu/~brian/905-2008/papers/Horvitz-Thompson-1952-jasa.pdf)，10.1080/01621459.1952.10483446。**阅读：正文扫描页 664–669，尤其式 (4)–(6)。**

原文区分每次抽取概率和最终 inclusion probability，并以后者倒数构造总体总量的无偏估计，直接支持记录真实抽样概率再评价面板总体。这里的设计无偏性质不能直接移植给裁剪权重、随机分母的归一化比率或训练后同样本性能，更不能把大量相关县小时当作独立重复系统。

**25. Lohr (2021), “Sampling: Design and Analysis,” 3rd ed.** Chapman & Hall/CRC；[作者书页](https://www.sharonlohr.com/sampling-design-and-analysis-3e)、[作者开放前言](https://www.sharonlohr.com/s/SDA-Preface.pdf)，10.1201/9780429298899。**阅读：作者书页、目录与前言；未通读整书。**

第三版强调设计先于分析，覆盖分层、多阶段、复杂调查方差与非概率样本等问题，提醒我们先声明系统/县/小时的抽样层级和目标总体，再解释设计权重、损失权重与有效样本量。这里不能引用未核读的章节作为当前裁剪方案或 family bootstrap 的证明；2021 年出版记录与某些目录的版权年份差异应与具体版本信息一起保留。

## 4. 对恢复核的综合约束

### 4.1 可观测的是净状态变化，不是两条独立流

以宿主离散更新 `p_next − p = u(1−p) − r p` 为例，在 `0 < p < 1` 时，一个净增量只提供一个约束，许多满足边界的 `(u,r)` 都可给出同样变化。记忆状态能帮助预测，不能凭拟合优度声称真实失效和修复流已被识别；尤其不能用观测下降量直接监督“真实恢复率”。Wei 的聚合数据讨论提供了直接文献依据；上述代数是本项目的结构分析。[Wei et al.](https://arxiv.org/pdf/1304.7710)

在理想、同分母且无报告误差的存量观测下，正净变化才给故障流下界，负净变化才给恢复流下界；实际县数据还混有覆盖/报告变化。即使在峰后，新增故障、重接后再失效和客户层面的异质性也使 stock 的相对半衰时间不同于单次维修时长。[Brelsford et al.](https://www.nature.com/articles/s41597-024-03095-5)，[Wei et al.](https://arxiv.org/pdf/1304.7710)

可从公开 stock 比较的是预测曲线、净变化、观测窗内峰值/面积积分、阈值超越和带删失标记的消退时长。若使用“恢复核”名称，应解释为影响宿主恢复分支的潜在历史状态；物理率、维修年龄分布和资源容量要更强观测或额外假设才能解释。144 h 内未观察到消退不等于永不恢复，短窗外尾部也不能赋一个固定完成时间。

### 4.2 为什么恢复历史合理，为什么不能承诺修好全部峰值

故障发生时点、损伤严重程度、优先级、持续天气和通达性均可改变后续恢复；这些证据支持稳定的有界历史状态、条件化时间尺度，以及允许加快或减慢的残差读出。[Wei et al.](https://arxiv.org/pdf/1304.7710)，[Ji et al.](https://www.nature.com/articles/nenergy201652)，[Duffey](https://link.springer.com/article/10.1007/s13753-018-0189-2)

这是建模动机，不是本项目已证明的误差原因：宿主中恢复项为 `−r p`，在 `p≈0` 时直接作用很小；固定 `u` 时降低恢复只能保留已产生的存量，无法生成足够的新损伤。因而恢复侧方案仍应报告量级和起始漏报，并与固定损伤、关闭恢复的可行上界比较；联合重训改变 `u` 后则不能再沿用该冻结因果解释。

### 4.3 资源共享：哪些有证据，哪些仍是假设

有证据：Wanik 的 Area Work Center 跨多个 town 组织工作，Ji 讨论维修优先级、设备层级、人员和可达性，Duffey 将持续恶劣环境与恢复困难联系。它们支持“恢复不是孤立县常数”这一动机，尚不足以给出我们每一条县际边的实际人员流量或有限服务容量。[Wanik et al.](https://hartman.byu.edu/docs/files/WanikAnagnostouHartmanFredianiAstitha_StormDamage.pdf)，[Ji et al.](https://www.nature.com/articles/nenergy201652)

仍是假设：以距离、州界或近邻图代替公用事业服务区/人员调度；把行归一化状态混合解释成维修队运输；把正耦合解释成救援、负耦合解释成竞争。单纯扩散具有平滑与信息共享作用，并不自动实现有限资源守恒；在没有公开服务区/容量证据时，应称“空间恢复上下文”，并用无空间、置乱地理/图结构及配对宿主控制区分预测价值与空间平滑。

### 4.4 对架构与试验的直接落实

1. 地理只调制隐藏核的状态更新、时间尺度或耦合，保留单一损伤 MLP；上述文献没有要求灾种门控、多专家或手工风险通道。
2. 人口/客户加权的小区节点可检验县面积平均是否错配暴露，但公开人口不是线路、客户不是人口；首先保留数据来源与变量含义，不能把分辨率提升当成收益证明。
3. 恢复侧可优先比较“局部负荷/天气历史”与“再加入空间恢复上下文”，二者都需要零开口配对、稳定性及缺测/前缀初始化协议；文献关联不能代替 null。
4. 设计权重回答目标总体，尾部强调损失改变优化重点，两者须分开；以五类天气和独立系统为评估层级，报告整体与条件表现，防止只解释少量重灾县或大量近零小时。

以上均为待 PI 讨论的设计约束；本笔记不构成训练批准，也不声称任何候选核已有效。
