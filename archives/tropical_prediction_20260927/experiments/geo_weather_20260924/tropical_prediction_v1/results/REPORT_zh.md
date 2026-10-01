# 纯热带预测：15 系统五折完整实验

本次使用 data_v1 的全部热带开发系统，完成 70 次神经网络拟合、75 次树模型拟合。所有模型从公共数据重新拟合，外层每个神经模型均平均 seeds 0/1/2 的预测。

## 主结果：完整 144 小时，设计加权 pooled 指标

| Model | mae | rmse | mse |
|---|---:|---:|---:|
| Zero forecast | 0.00978 | 0.05805 | 0.00337 |
| Persistence | 0.00973 | 0.05728 | 0.00328 |
| Damped persistence | 0.00971 | 0.05729 | 0.00328 |
| Matched AsymODE host | 0.01078 | 0.04858 | 0.00236 |
| Joint GCRK-rate model | 0.01171 | 0.05073 | 0.00257 |
| ERA5 residual tree | 0.01439 | 0.05345 | 0.00286 |
| Dual-weather residual tree | 0.01197 | 0.04813 | 0.00232 |
| Dual-weather MAE tree | 0.00954 | 0.04921 | 0.00242 |
| Inner-selected ensemble | 0.01108 | 0.04584 | 0.00210 |

内层选择的融合相对匹配 HOST 的 RMSE 变化：改善 5.65%；相对全零：改善 21.03%。负数表示恶化。
融合对 HOST 的配对家族重采样 95% 区间为 [-4.24%, 17.02%]，10/15 个系统改善。
融合对 HOST 的 MAE 改善为 -2.81%。

联合 GCRK＋速率模型相对 HOST：改善 -4.41%。该对比同时改变地理核和适配器，不能单独归因于地理信息。
联合模型对 HOST 的 95% 区间为 [-9.46%, -1.18%]，6/15 个系统改善。区间跨零时不声称稳定正收益。

## 点时效结果

### +1 h

| Model | mae | rmse |
|---|---:|---:|
| Zero forecast | 0.00088 | 0.01588 |
| Persistence | 0.00016 | 0.00114 |
| Damped persistence | 0.00016 | 0.00114 |
| Matched AsymODE host | 0.00053 | 0.00512 |
| Joint GCRK-rate model | 0.00051 | 0.00415 |
| ERA5 residual tree | 0.00145 | 0.00686 |
| Dual-weather residual tree | 0.00099 | 0.00436 |
| Dual-weather MAE tree | 0.00165 | 0.01779 |
| Inner-selected ensemble | 0.00099 | 0.00573 |

### +6 h

| Model | mae | rmse |
|---|---:|---:|
| Zero forecast | 0.00093 | 0.01641 |
| Persistence | 0.00029 | 0.00187 |
| Damped persistence | 0.00029 | 0.00187 |
| Matched AsymODE host | 0.00146 | 0.01447 |
| Joint GCRK-rate model | 0.00161 | 0.01355 |
| ERA5 residual tree | 0.00156 | 0.00691 |
| Dual-weather residual tree | 0.00105 | 0.00455 |
| Dual-weather MAE tree | 0.00169 | 0.01816 |
| Inner-selected ensemble | 0.00136 | 0.01019 |

### +24 h

| Model | mae | rmse |
|---|---:|---:|
| Zero forecast | 0.00993 | 0.05660 |
| Persistence | 0.00949 | 0.05460 |
| Damped persistence | 0.00948 | 0.05460 |
| Matched AsymODE host | 0.00921 | 0.04026 |
| Joint GCRK-rate model | 0.00969 | 0.04216 |
| ERA5 residual tree | 0.01025 | 0.04490 |
| Dual-weather residual tree | 0.00936 | 0.04182 |
| Dual-weather MAE tree | 0.00802 | 0.04190 |
| Inner-selected ensemble | 0.00860 | 0.03744 |

### +48 h

| Model | mae | rmse |
|---|---:|---:|
| Zero forecast | 0.01524 | 0.07640 |
| Persistence | 0.01498 | 0.07539 |
| Damped persistence | 0.01497 | 0.07540 |
| Matched AsymODE host | 0.01559 | 0.05885 |
| Joint GCRK-rate model | 0.01704 | 0.06407 |
| ERA5 residual tree | 0.02443 | 0.07486 |
| Dual-weather residual tree | 0.01924 | 0.06625 |
| Dual-weather MAE tree | 0.01514 | 0.06249 |
| Inner-selected ensemble | 0.01761 | 0.06135 |

## 训练与模型设计

- HOST 复用仓库 AsymODE：32 单元损伤网络、16 单元恢复网络，恢复率逐小时更新。
- 融合臂在相同初始化 HOST 上加入仓库的 GCRK，以及两个 16 单元零输出初始化速率头；全部参数联合优化。
- 两个神经臂共享 ERA5/HRRR 危险度、缺失指示、时钟、历史停电摘要和六个县级背景变量；地理核另读取真实物理地理描述。
- 原 HOST 完整 features_v1D.npz 未随分支提交，因此这是危险度输入上的新匹配对照，不是旧 11.5% MSE 数字的复现。
- 每次神经拟合固定 900 次 Adam 更新、batch 128、损伤学习率 0.003、恢复 0.0003；每次重新拟合变换和校准缓冲。
- 树模型共享全时效预测器，使用固定 300 棵树、15 叶；残差版本使用平方损失，直接版本使用绝对损失。
- 外层训练集合内部四个完整事件折产生样本外预测；只据其设计加权 MSE 选择非负且和为 1 的融合权重。
- 权重在整条 144 小时路径上固定，不按预测时效切换模型。外层预测和标签未参与权重拟合。

## 验证范围与材料

15 个系统、1,633 个县—事件、234,874 个有效未来县—小时。事件外推而非县外推；同一县可以出现在其他风暴训练数据中。
三种子的逐种子结果在 seed_metrics.csv；每场风暴及层级结果、未截断抽样权重敏感性、无权重结果均保留。
paired_family_bootstrap.csv 给出 2,000 次配对家族重采样区间，针对固定外层预测，不包含重新训练和开发选择不确定性。
ERA5 和滚动 HRRR 用作已知天气输入，属于条件回报预测。Milton 缺失的 59 个 HRRR 小时保留缺失指示，没有删除相应目标。
客户分母及部分地理和背景字段使用仓库的既有公共快照，并非每场历史风暴起报时可获得的版本；SAIDI 2023 / SAIFI 2023 已排除。
方向选择已看过开发集已有热带结果；本次仍是开发证据。封存 C 未读取，尚不能将这些结果称为独立确认或业务部署精度。
内层神经模型用单种子、最终外层用三种子均值，两者方差不同，是本次计算预算下的明确近似。

复现：先运行 prepare.py，再依次运行 run_campaign.py --fold 1 至 --fold 5，最后运行 summarize_campaign.py。建议并行不超过两折（8 GB 内存）。

## 当前可支持的论文结论

这轮实验中，内层选择的集成取得最低的完整路径 RMSE；双天气源绝对损失树取得最低的 MAE。两者存在误差取舍，不能写成两项指标都由融合获胜。
联合 GCRK＋速率头的 RMSE 和 MAE 均差于匹配 HOST，三个配对种子也都没有改善。因此本轮不能以“地理核稳定提升精度”为主张。
融合对 HOST 的 RMSE 改善区间跨零，对双天气源残差树也跨零；相对强基线的优势尚未得到明确确认。
短时效与完整路径的优胜顺序不同：近端持续性预测很强。本次没有按外层结果挑选时效或拼接一条新预测路径。
这份成果适合支撑“气象条件下的热带停电预测”开发研究。封存确认和使用起报时实际可获得的气象预报进行评估，仍是后续独立环节。
