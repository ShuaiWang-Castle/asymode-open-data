# 历史证据：保留反证，不当成新trajectory结果

本目录全部是已保存的旧协议结果副本；不需要原目录、旧checkpoint或私有仓库才能读取。它们不代表本次训练或新服务器回包。

## 受控冻结投影审计

旧任务是单步净增量监督，不是新H=24完整路径训练。
`projection_summary.csv`、`projection_by_seed.csv`、`prediction_summary.csv`保留原表：
SR恢复RMSE约0.044772；显式TR约0.007545；DN+正确source三角投影约0.004903，5/5种子优于旧TR。
三角投影将冻结净函数在source参考状态上的响应拟合为`u(1-y)-r*y`，约束u,r≥0且u+r≤1。系数提取与替换预测不同；既有反证不应被删去。

这限制“显式两个头/训练时分离独占优势”的表述；不证明直接trajectory方法差，也不替代新真实数据比较。
本轮不新增投影campaign；若以后要声称训练时结构优于所有后处理，则必须重新纳入匹配比较，本轮不作此主张。

## 旧真实checkpoint重放

`OLD_REAL_REPLAYED_COMPARISONS.csv`保存接收阶段的旧checkpoint主口径重放摘要：全源TR相对SR净MSE平均公司相对收益约+1.412%，ending收益约+0.311%且seed跨0；低源高状态主结果约-7.893%，内点初始化约-1.169%。
这些是旧一步/限制模型协议，不能混入新DIRECT/NET/ASYM全路径主表。2019已重复开发使用。

## 当前结论边界

保留“净loss可能使recovery补偿遗漏damage”的受限模型insight，不能把它升级成一般DIRECT的缺陷。
当前核心任务是匹配完整trajectory学习；真实结构收益尚未闭环，尚无已核验的该新协议GPU调参回包。
不生成或拼接不存在的旧Gamma/方向校准/全国训练证据。正文与理论职责以paper目录文件为准；旧数字只限制主张，不指导按2019选参。
