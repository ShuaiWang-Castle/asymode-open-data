# 本地数据合同：路径可变，科学身份不可变

## 1. 所需数据和固定身份

主任务为ANEEL旧版2018/2019固定cohort：24个included集合、16家公司。唯一授权的分组映射、历史选择记录、固定训练年分母是本包`metadata/unit_selection.csv`，不是当前目录随意找到的另一张selection表。
筛选`evaluation_status == included`，再按`group`升序；group编号不连续。不要重新挑选cohort，也不要重新用2019筛选。

`metadata/LEDGER_REFERENCE.json`包含原192个数组的shape与SHA256：统一转为连续little-endian float64后哈希。它只有身份元数据，没有时间序列数值。
原容器SHA256为`03392a7f94bf9fbef6cf59c127afbd0ca829db27bcde0843fcaad666d5184e06`。压缩参数/成员顺序可变；数组identity不可变。

- 每group/year的`g{group}_y{year}_y`是8761个小时边界状态；共48条库存序列。
- 完整ledger还含`fp/fm/area`，各8760，总计192数组。
- y是冻结分母下的加权活动记录，不自动等于去重客户数。
- 同时有fp/fm时核对`diff(y)=fp-fm`；仅有canonical y也足够执行净监督trajectory主实验。方向审计缺失记NOT_AVAILABLE。
- hash2019数据做身份核验，不等于提前用2019模型表现选参；后者仍禁止。

## 2. 数据发现顺序

`code/prepare_local_data.py --data-root D --output W/data_resolution`只读地搜索指定目录：

1. NPZ：按数组key+逐项指纹识别，名称和旧父目录不重要。优先尝试名称含ledger的候选，再尝试其他NPZ。
2. 完整`pilot_records_g*.npz`：按本包固定分母重建。重复字节内容只使用一份，不叠加记录；冲突副本拒绝。
3. 本地旧版2018/2019 ANEEL CSV或含单一CSV的ZIP：必须为`metadata/ACTUAL_COLUMNS.json`列出的18列，UTF8、分号。文件名含`interrup`或`aneel`及年份；只有符合身份的重建结果才接受。

可重复`--data-root`搜索配置里找到的其他真实本地目录；支持`--ledger`指定已定位的NPZ。不自动遍历目录符号链接，请显式把其目标传作根目录。
此“年度原始CSV/ZIP”是数据源，不是已删除的项目handoff ZIP。不得让CC把两者混淆。

若本地只有其他布局（例如clean_2018.npy/clean_2019.npy，或改名的年度CSV），先查配置/README确定数据路径，允许仅修改路径识别或增加格式适配：输出同一组canonical库存并通过指纹。不得用重新计算分母、换时区、插值、删样本来制造“可运行”数据。
适配无法确定身份时保存BLOCKED_DATA及缺失清单；不要要求已删除的handoff，亦不要把身份不明的数据用于正式结果。

## 3. 原始记录重建语义

小时flux使用右闭区间：`(t,t+1]`；库存使用`start <= boundary < end`。源时间是旧文件naive时间字符串，不做UTC/DST换算。
过滤与既有版本相同：`start>0, end>start, n>0, denom>0, n<=denom`；分母使用selection固定值，不使用当年中位数替换。
旧CSV用NumCPFCNPJ与IdeConjuntoUnidadeConsumidora组成key，映射到已锁定group。完整原始行的双哈希去重，不按(start,end,n)压缩不同记录；两个年度分别去重。保存原始文件hash和最终canonical数组核查。
end/n/denom用于重建观测数据，不是允许在预测输入中使用事故未来结束信息。
重建`fp/fm/area`是审计操作，主训练只读y，不能把净标签任务升级为方向监督。

## 4. 训练输入与窗口

L=24，H=24，历史`y[t-24:t]`，当前`y[t]`，目标`y[t+1:t+25]`。
已知calendar为t..t+23的6小时分段one-hot4+weekend1。49个状态必须全部在原month-interior mask及partition内合法。
context维数168：24个group one-hot + 24历史/s + 24×5未来已知clock。context不含当前y0；DIRECT/NET额外显式读取y0，ASYM只经源池因子读取当前状态。所有方法使用相同可得信息，函数结构不同。
不使用未来天气、方向日志、area、事故剩余时长、结束信息、事后cause等特征。禁止读取frozen_nn_rates.csv。

| partition | 每集合origin | 24集合origin |
|---|---:|---:|
| fold_A_fit | 3745 | 89880 |
| fold_A_val | 1920 | 46080 |
| fold_B_fit | 5665 | 135960 |
| fold_B_val | 1920 | 46080 |
| full_2018 | 7608 | 182592 |
| reused_2019 | 7608 | 182592 |

A:上半年fit→三季度validation；B:前9月fit→四季度validation。fit最后target最多到名义validation开始前48h。计数不符应修复边界而非随意删样本。
每fold仅用fit可得历史/当前库存做公司等权RMS与mean；不得用full2018统计处理早期fold。以源端验证path MSE选参，写selection_lock后才评价2019。
2019曾被反复使用且cohort本身含历史回顾性排除；新pipeline不能把它变成prospective/fresh/confirmatory holdout。
