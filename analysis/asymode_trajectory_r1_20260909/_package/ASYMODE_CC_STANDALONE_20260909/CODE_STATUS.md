# 代码边界与执行状态

## 已随包实现

- `verify_package.py`：stdlib校验自身SHA256清单，无旧包依赖。
- `prepare_local_data.py`：当前目录递归搜索，固定cohort，容器/数组身份核验，选中记录/旧版CSV台账重建，生成实际路径manifest。只写output。
- `data_contract.py`：通过本地manifest读取库存；原始小时mask、forward folds、窗口、fit-only统计、样本拼装。
- `core_models.py`：DIRECT/NET/ASYM/SR前向、参数预算匹配、full path loss及命名的一步消融。与前一已锁定版本保持一致。
- `training_primitives.py`：独立RNG、公司→集合→origin等权采样；FP64原始路径评分器；严格区分均值相对收益与均值之比。
- `controlled_data.py`：锁定的单一E2生成机制与独立splits，无训练器。
- `preflight.py`：数据/窗口/参数数目/12更新forward-backward数值预检，不是科学性能试验。
- `test_primitives.py`：无需真实数据的采样/评分、controlled、模型接口、旧CSV转换/完整行去重等小测试。

## 必须由CC实现并实际运行

正式`run_campaign.py`、GPU正确性检查与profile、对称预算规划、六候选/forward-fold选择、重复seed确认、最终refit、推理/预测存储、独立评分交叉复核、断点续跑/失败记录和结果打包。
所有科学规则已经在`CLAUDE_CODE_EXECUTE_ZH.md`与`CONFIG_LOCK.json`确定。实现driver是工程任务，不授权CC重新研究模型定义。
`--phase main/ablation/controlled/package`是待实现的正式driver接口，不是声称包内已有完整campaign。

## 打包侧核查含义

local_checks中的PASS仅表示本机准备流程与CPU小批次检查通过，不代表CC服务器已验证CUDA、更不代表新真实收益。
现成台账路径和仅选中记录重建路径均需验证；全国CSV路径只做小fixture验证，完整全国CSV扫描未在本次包重建中运行。
没有执行真实新tuning、正式controlled训练或新2019模型评价。
