# D11：原始路径、隐藏路径与地理增量的有限诊断登记

PI授权：2026-09-30「沿着GPT的建议去做」。先执行下一设计稿第6节，暂不实施第7节神经候选。
主线仍为天气→复杂地理调制→冲击形成、组合与累积→停电；本阶段检查可迁移信息，不识别物理冲击。

## 固定比较与数据边界

只读公共D的原OUTER1 FIT 6350个单位内的已登记数据，沿用D08固定输入面板及D09事件留出2/3。
每套配置沿用同一冻结D09 new900 carrier（eb91994），以其closed预测为残差基准。
分别在871/883面板FIT拟合，对原留出折全部2100/1769单位评价。两套carrier不同，不能称独立seed。
6条件：R_shared/R_real/R_permuted/H_shared/H_real/H_permuted，共12终点。
R为原weather12，H为冻结carrier的h1=32。公共辅助为当前xu42+context6+y0（49列），不含geo40。
context6和carrier已含县背景；所谓地理增量是相对已有背景的geo40增量，不是完全无地理对照。
D08面板是历史输入转导选择；D反复用于开发，不是新的独立确认。
绝不读取/构建/评估sealed C、年度原始标签、竞赛数据或paper_v1。oracle不进入标签、特征或选择。

## 路径与核预算（输入拟合均限FIT）

完整216小时路径；forecast为72..215，y0为71时观测。R的cape/precip/snowfall采用log1p(max(x,0))。
路径通道统计在FIT216估计；路径坐标统计在FITforecast144估计。输入几何用每merged-group等质量、组内原w比例。
q包括当前/初始/当前增量、绝对hour/72，及6/24/72小时窗口中(time/72,全部通道)的一级和全部二级signature。
因此R/H维数583/3463。路径在t只使用<=t的输入；不接触y/m。四块RBF的加权和：level-triplet+clock权1/2，
三个signature块各1/6，各自距离按FIT标准化后的坐标RMS归一化，length固定1；坐标sd floor1e-4。
geo40在unique FIT counties上中位填补/标准化（floor1e-6，不活跃坐标scale1），保留全部40维。
联合地理核为40个一维RBF的所有非空subset产品的按阶归一化平均，40个阶数质量均1/40。
shared k_geo=1；real是真实geo；permuted按unique FIT counties整条40维profile固定derangement，seed20260930。
同县跨时/事件共用donor；未见query县按县ID哈希映射FIT donor，不读取其真实geo生成打乱特征。
此扩展不是held完整双射，报告seen/unseen病例/县/设计权与donor碰撞，单置换不提供permutation p值或因果证据。
每个CV/fullFIT各自固定64个input-only(unitID,hour)地标：哈希排序group、组内unit，轮转取token，时钟
72/96/120/144/168/192/215按token哈希轮转；同一split六条件用相同token，保持唯一。
仅当这七时钟不足64个独特token时，按其余forecast时钟按单位ID固定哈希次序补齐；实际D11 FIT规模不触发该小数组fallback。
Nyström Gram eigenfloor1e-6，保留64列；禁止按结果扩地标、换长度、阶数或置换seed。
预测直接使用psi，不使用锚定差分。参考q只用于覆盖检查：在同一时钟将路径通道固定在hour71。
报告当前/参考核最大相似度、原地标相似度集中、diag(k)-||psi||²近似残差、Gram spectrum及有效秩。
小覆盖不是失败后重采地标的理由，也不是证明信息不存在；联合支持与容量有限必须保留。

## 同容量回归和选择

49公共列+64核列=113，另加不惩罚截距；144小时共享一个scalar head，非144独立头。
目标是 sum FIT w*m*(y-Pclosed-Xbeta-bias)^2 / sum FIT w*m + lambda||beta||²（beta为标准化predictor坐标系数）。
用原w、不用HT、不用D09类别Z_r归一化。它与神经训练目标不同，是信息探针，不能称同优化目标架构对照。
预测前所有减法/统计提升float64；predictor标准化限每CV-train/FIT有效格点，sd floor1e-4，常数列精确置零仍保留。
lambda固定[1e-4,1e-2,1]；CV为各innerFIT的另三个原event folds，每次完整重估输入几何/地标/回归统计。
按原w*m验证SSE之和选lambda，精确tie取较大值，不按S或held标签选。108个CV小解+12个终点解=最多120解。
冻结carrier历史训练包含ridge CV validation组，因此CV仅是条件于冻结carrier的grouped正则选择，
不是整条神经流程crossfit；最终inner-held2/3仍未用于特征/正则选择。
不clip修正；报告预测出[0,1]的数量/质量。探针不遵守stock可达约束，不能部署或作为正式停电成绩。

## 固定评价、漂移与决策

完整144小时原w/m，分别报告all/S（观测峰>=10%）/nonS/J/全部类别/各折、严重小时数与连续时长、假峰。
严格记录2rDelta（对齐）、Delta²（幅度代价）及净收益，含病例和事件集中度。FIT与held分开。
13固定比较：6条件vsclosed、3个Rvs对应H、R/H真实geo各vs shared和permuted。
沿用D09固定1999次分层paired merged-group bootstrap及family敏感性（seed20260929）；同一计划用于全部比较。
共同535病例另比两套配置的修正、峰幅和收益换位；小Kish/高权集中明确报告。
另精确拆解pre-ReLU第一层 a=Wz+Acontext+b 的两配置差：
preprocess=((W2+W3)/2)(z3-z2)+((A2+A3)/2)(c3-c2)，
learned=((W3-W2)/2)(z2+z3)+((A3-A2)/2)(c2+c3)+(b3-b2)。
两项相加恰等于a3-a2；报告RMS及交叉项，ReLU后/最终预测不据此作因果归因。
共同病例另报原始同输入的标准化漂移、D09 Z_r和原w/Z_r相对权漂移、隐藏校准/地标变化；
不可将不同Nyström旋转基下的系数差当作函数漂移。

探针支持原路径的探索性门：R_shared vs H_shared的S pooled及两折均正、pooled区间下界>0；
同时保留all/nonS成本及严重假峰，不因overall差而删除局地症状。
地理信息支持门：R_real vs R_shared和R_permuted的S pooled及两折均正、pooled区间下界>0。
门是信息证据，不是原正式10%目标已实现。任何结果都不自动训练神经候选、加seed/NULL/fullD或扩预算。
失败或两路径无迁移信号时，优先审查联合支持、局地暴露与观测接口，不能靠提高signature阶数掩盖失败。

## 运行、审计与理论限定

登记源：d11_probe.py/d11_features.py/d11_ridge.py及本文件；先合成检查、review、commit并push两个research分支再读实际数组。
最多两自有进程，每个两数值线程nice>=15；exclusive新目录，所有partial/FAILED保留，不碰其他会话。
全六D09 DONE/source/data/split/roster/output hash守卫先于数组读取。frozen model state前后hash相同。
非有限值、身份/组泄漏、显著负Gram特征值、方程残差或scope漂移失败即保留证据，不临时改设计。
低coverage/秩不足/边界lambda选择是科学结果。完成后独立复算coverage/分数/对齐恒等式/CV选择/资源及hash。
只显式提交source、笔记、紧凑JSON/gzip；先扫描私人路径、邮箱、竞赛标记；push两个research分支，绝不main。

固定模型的输入Lipschitz界不保证换训练事件后学习稳定；参见
[Bousquet–Elisseeff](https://jmlr.org/papers/volume2/bousquet02a/bousquet02a.pdf)。有限Nyström质量取决于采样和有效维数，
[Rudi等](https://proceedings.neurips.cc/paper/2015/file/03e0704b5690a2dee1861dc3ad3316c9-Paper.pdf)不能作为64地标足够的证明。
后续若实施球投影须检查球外radial derivative=0、工作区和真实天气变化保留；联合锚点差分在无支持当前输入
可能遗留-Cpsi(ref)，不能默认自然回零。原始路径是新计算接口，唯一damage MLP不等于唯一天气通路。
