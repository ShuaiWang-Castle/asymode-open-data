# 稿件替换清单（公开数据版结果）

对应稿件：`writing/manuscript_gcrk_20260918/main.tex` 与 `gcrk_technical_companion.tex`。
下面只给位置（节、标签、段落顺序）和英文替换文本，不引用原句。所有数字的出处见
`RESULTS.md`。稿件正文我没有改；这里的英文段落是草稿，供 PI 取用。

**先说结论（决定怎么写结果部分）**：在公开数据上，同一宿主加上 GCRK 以后，主指标全程
RMSE 5 个 seed 全部略差（+0.53%），25–48 h 差 3.3%（5/5），MAE 差 6.5%（5/5）；峰值大小
与峰值时间误差 4/5 个 seed 略好。两个学习模型都比全零、持续、TimesFM 好得多（RMSE
0.0263 对 0.0305）。所以原稿"GCRK 相对宿主在各时距都改进"的表述在公开数据上不成立，
下面 C 节的替换文本是按真实结果写的。

---

## A. 数据部分（原数据描述整体替换）

A1. `sec:data` 之前、引言第一段的最后一句（介绍原数据来源的那一句）替换为：

```latex
We study this setting on five multi-state wind events in public county outage records, where each forecast starts from the last observed hour and must carry the existing outage stock through one or more later storm waves.
```

A2. `sec:data` 第一小节正文（事件、日期、县数、训练与测试县的描述）替换为：

```latex
We assemble 216-hour windows around five multi-state wind events of 2019--2024 chosen by a fixed rule on NOAA Storm Events metadata: each is driven by strong gusts with convective lines or gust fronts, spans several days, begins after a quiet 72-hour prefix, and brings a second wind, precipitation, or cold-sector wave. An event's county set is its wind-report footprint after data-quality gates, giving 2,660 county-events in 1,756 counties across 45 states. Hourly outage fractions come from EAGLE-I county records, divided by the publisher's modelled 2024 county customer counts. A missing record is read as zero while the county's feed is in service (a record within seven days); the remaining 0.06\% of county-hours are excluded from every loss and metric.
```

A3. Figure 1 换成 `figures/fig1_event_county_impacts_open.pdf`（仍是三联：第一波、第二波、合并；
代表事件为 2021 年 12 月窗口），图注见 `CAPTIONS.md`。子图标题改为 "Wave A"、"Wave B"、"Both waves"。

A4. `sec:data` 第一小节中定义 $p_{i,t}$ 之后、说明观测截断和天气来源的两句替换为：

```latex
Outage observations are used through hour 71. ERA5 reanalysis weather is supplied for all 216 hours, so every compared forecast is conditioned on the realised weather, a perfect-forecast setting.
```

A5. `sec:data` 第二小节"再增长"一段的计数替换为：

```latex
Renewed growth is common in the panel. In 1,202 of the 2,660 county-events the outage fraction rises by at least two percentage points within two hours during the forecast window; 124 of these had already reached 2\% earlier in the window. Of the 1,843 county-events with no customers out at hour 71, 882 reach at least 2\% during the forecast window.
```

A6. 同一小节县对例子一段替换为（县对的选取规则见 `results/weather_twin_pair.json`）：

```latex
County differences also motivate a conditional weather response. During the second wave of the December 2021 window, Harvey and Saline counties in Kansas have nearly identical reanalysis weather (maximum gusts of 30.3 and 28.6~m\,s$^{-1}$, 0.7~mm of precipitation), yet their outage fractions peak at 1.55\% and 59.85\%.
```
原段后半关于"同一县内先降后升"的例子删去；如需要，可换成 Figure 4 中 Newaygo 或 Mason（Michigan）两波之间的回落与再起。

A7. 描述地理描述符数量与内容的那一句替换为：

```latex
We use 31 public geographic descriptors: terrain from the USGS 3D Elevation Program (mean elevation, relief, slope, steep-slope share, ruggedness, and eight aspect shares), tree canopy and land cover (USFS NLCD Tree Canopy Cover and Annual NLCD, 2021), SSURGO soils (poorly drained, hydric, shallow, and high-water-table shares and their union), a windthrow-susceptibility index (forest share times root-limiting soil share), county land area, and 50-km smoothed relief, canopy, windthrow susceptibility, and poor drainage.
```

A8. `tab:inputs` 两行：
* Natural geography → `31 public descriptors control the GCRK response and its damage-side readout.`
* County context → `Customer scale (EAGLE-I), rural--urban code (USDA ERS), population density (Census), utility mix and reliability (EIA-861), and five-nearest-county weather inform recovery.`

A9. `gcrk_technical_companion.tex` 第 39 行第一句（地理输入的维数与预处理）替换为（后面的 "Center and bound it as" 不变）：

```latex
Let $\bm g_i\in\R^{31}$ be the geographic input, standardized with the fitting counties' means and standard deviations and clipped to $[-5,5]$; a missing soil descriptor is set to the fitting mean.
```

---

## B. 目标、损失与指标只基于停电比例后必须改的方法句子

B1. `sec:data` 第三小节（原窗口指数的定义小节，含一个显示公式及其后说明）整节替换为：

```latex
\subsection{One trajectory, scored along its whole path}
The prediction target is the outage fraction $p_{i,t}$ itself. Outage observations are used through hour 71; from the observed $p_{i,71}$ the model produces one open-loop path over hours 72--215, and every metric is computed on that path. We report the RMSE and MAE of $p$ over all forecast hours, the RMSE by lead time $\ell=t-71$ in four segments (1--6, 7--24, 25--48, and 49--144 hours), and, for every county-event whose observed forecast-window peak reaches 1\%, the errors in the peak's magnitude and hour. Because the forecast is one path, the lead-time segments describe the same trajectory rather than separate predictions of the same hour.
```
同时删除该公式用到的宏定义，以及全文其他对该指数的引用（摘要、引言、Figure 2 占位框、Joint estimation 段、技术附录）。

B2. `sec:results` 之前的 "Joint estimation" 段，前两句替换为：

```latex
We minimize the mean squared error of $p$ over the observed county-hours 72--215 of the complete hourly rollout; the same criterion, evaluated on held-out inner counties, selects the training length. Host, GCRK, and recovery parameters receive the same trajectory-loss gradient.
```
其后关于两组学习率、每十步 FIT 校准、200 步开启、0.2 drop-path 的句子不变。

B3. 引言最后一段中“用评分与县轨迹评估完整模型”那句，把评分改为 "trajectory errors by lead time"。

B4. Figure 2 占位框最后一行（从一条路径到多个评分时距）改为 "... one path over hours 72–215"。

B5. 技术附录最后一节第一句改为：

```latex
One trajectory objective, the mean squared error of $p$ over the observed county-hours 72--215, updates the host, kernel, and recovery networks.
```
该节最后一句改为：`Empirical statements in the report refer to the five-seed, county-held-out evaluation on public data described there.`
附录第一段关于实现来源提交号的句子由 PI 决定是否保留。

---

## C. 结果部分

C1. 摘要最后两句（数据来源一句与结果数字一句）替换为下面这句；若要写入 LOEO，可在其后加
"When a whole event is held out, neither model beats the all-zero forecast."：

```latex
On 2,660 county-events from five multi-state wind events in public outage records, the trajectory model lowers the RMSE of an all-zero forecast from 0.0305 to 0.0263, whereas adding GCRK to the same host does not lower it further (0.0264; a change of +0.5\%, with a 95\% county-resampling interval of $-1.3$\% to $+2.1$\%); GCRK slightly reduces peak-timing error and increases false activity in quiet hours.
```

C2. `tab:results` 换成 `results/table_main_main.tex`（行：All zero、Persistence、TimesFM、Weather host W、
AsymODE + GCRK；列：全程 RMSE、四个提前量分段 RMSE、MAE；W 与 GCRK 为 5 个 seed 的均值 ± 标准差）。
原表的 MAE/RMSE 双组八列结构改为一组 RMSE 五列加 MAE。表注见 `CAPTIONS.md`。

C3. `sec:results` 第一小节第一段（数据与评估设计描述）替换为：

```latex
We evaluate with five county-grouped outer folds: all events of a county are held out together, so every forecast concerns a county the models have not seen. Within each outer fold, each model selects its training length on three county-grouped inner folds, evaluated every ten steps with at least 400 and at most 1,600 steps and patience 200, and is then refit from scratch on all development counties. Five initialization seeds are run for every fold; W and GCRK share the host initialization of a seed. All forecasts start from the observed $p_{i,71}$ and proceed open-loop. A leave-one-event-out design, in which each event is held out in turn, checks transfer to an unseen event.
```
小节标题建议改为 "County-held-out evaluation"。

C4. 同小节第二段（分时距格点数与表的说明）替换为：

```latex
Errors are pooled over the 382,736 observed held-out county-hours: 15,959, 47,850, 63,797, and 255,130 at leads of 1--6, 7--24, 25--48, and 49--144 hours. Table~\ref{tab:results} compares an all-zero forecast, persistence of $p_{i,71}$, zero-shot TimesFM with ERA5 covariates, the weather host W, and AsymODE with GCRK.
```

C5. 同小节第三段（与宿主的配对比较数字）替换为：

```latex
Both trained models reduce the all-zero RMSE of 0.0305 to 0.0263 (W) and 0.0264 (GCRK); zero-shot TimesFM does not (0.0309), and persistence is worse than zero (0.0321). Against its own host, GCRK does not lower the error, and the difference is within sampling uncertainty: resampling counties gives a 95\% interval of $-1.3$\% to $+2.1$\% for the change in RMSE, and one event carries 78\% of the squared outage signal. Its full-rollout RMSE is higher in all five seeds (by 0.5\% on average), its 25--48-hour RMSE is higher by 3.3\% in all five seeds, and the other lead-time segments show no consistent difference. The same trained GCRK network with its kernel exit closed reaches 0.0273: with the exit closed, the network trained with the kernel is worse than W, and the kernel's output recovers most, but not all, of the gap. On MAE both trained models are worse than the all-zero forecast (0.0056 and 0.0060 against 0.0045), because the target is zero in most county-hours.
```
接在上段之后加入留一事件（LOEO）的一段（数字见 `RESULTS.md` 第 6 节）：

```latex
Holding out each event in turn is harder. Neither W nor GCRK then beats the all-zero forecast (RMSE 0.0314 and 0.0306 against 0.0305). GCRK is better than W in four of five seeds, by 2.5\% overall (95\% interval over event-by-state blocks $-6.7$\% to $+0.9$\%) and by 6.5\% and 8.4\% at 7--24 and 25--48 hours, and the same GCRK network with its kernel exit closed is lowest (0.0300, below the all-zero forecast in four of five seeds). All three forecasts overshoot in amplitude under transfer (the RMSE-minimising scale factors are 0.40, 0.54, and 0.68), and one common rescaling removes most of their differences, so the ordering mainly reflects the size of the overshoot. The kernel's output adds error on average: for one held-out event and seed, it lifts ten counties in the interior West and on the Maine coast to 37--70\% outages where at most 21\% were observed.
```

C6. `sec:results` 第二小节（个例与阈值诊断两段）替换为：

```latex
Figure~\ref{fig:cases} shows eight held-out counties chosen by a fixed rule. All start the forecast with no customers out, so every peak shown is an onset. With seed 0, GCRK places its peak within five hours of the observed peak in six of the eight counties, but the magnitudes are far off: it underestimates the six largest peaks by factors of 5 to 29, including the 31.9\% peak of the county in Figure~\ref{fig:kernel}, and overshoots two smaller peaks in Michigan. Zero-shot TimesFM stays near zero in all eight. Across the 1,609 county-events whose observed peak reaches 1\%, GCRK lowers the mean absolute peak-magnitude error by 0.5\% and the mean peak-time error from 19.3 to 18.3 hours, in four of five seeds.

Threshold diagnostics show the cost of that sensitivity. The share of observed-zero hours forecast above 0.001 rises from 30.9\% for W to 42.9\% for GCRK in all five seeds, while the share of active hours forecast below half the observed level changes from 60.5\% to 60.0\%.
```

C7. `sec:results` 第三小节（地理响应的解释）替换为：

```latex
GCRK's inspectable quantities make its role measurable. In the county of Figure~\ref{fig:kernel}, the kernel lowers the second-wave damage logit, mainly through snow and freezing inputs, at hours when the damage rate is already near zero, so the forecast barely changes. Where the kernel changes a forecast most (supplementary Figure~S3), its push is carried almost entirely by the weather inputs. Replacing a county's geography with those of 200 random donor counties inside each trained model shows that its own geography is better than a random one in every seed (one-sided Wilcoxon $p\le3\times10^{-4}$), but only slightly: the median own-geography percentile is 0.44--0.48 and the RMSE difference is at most $5\times10^{-4}$. The county-level effects of GCRK are two-sided and concentrated. In each seed, 1\% of county-events carry about half of all gains and half of all losses, one event carries about 80\% of both, and the per-county differences correlate weakly across seeds (mean Spearman 0.23), so single-seed county comparisons mostly reflect initialization.

Aggregate outage records leave customer locations, equipment condition, and repair operations unresolved, so the learned response should be interpreted as a predictive representation rather than a recovered physical network.
```

C8. 结论段中报告结果数字的那一句替换为：

```latex
On five public wind events, the population-balance trajectory model clearly outperforms zero-shot and persistence baselines, but adding GCRK to the same host does not improve pooled accuracy: it slightly improves peak timing, raises false activity in quiet hours, and its county-level effects are two-sided and seed-dependent.
```
结论的最后一句（贡献表述）建议 PI 按新结果决定是否保留"let geography shape the evolving response"的定位。

---

## D. 图与表

| 稿件位置 | 新文件 | 说明 |
|---|---|---|
| `fig:data`（Figure 1） | `figures/fig1_event_county_impacts_open.pdf` | 三联地图，只用观测数据 |
| `fig:framework`（Figure 2） | 不变（示意图） | 占位框文字按 B4 改 |
| `fig:kernel`（Figure 3） | `figures/fig3_gcrk_interpretation_open.pdf` | 2×2：kernel 贡献热图、推力驱动因素、净推力细条加预报、分提前量误差变化；按 PREREG 规则选县；从单栏 `figure` 改为 `figure*` |
| 补充 Figure S3 | `figures/fig3s_gcrk_largest_kernel_effect_posthoc_open.pdf` | 事后按模型输出选县，必须标明 |
| `fig:cases`（Figure 4） | `figures/fig4_county_trajectories_open.pdf` | 2×4：观测、GCRK、TimesFM；原来的放大插图取消 |
| `tab:results` | `results/table_main_main.tex` | 见 C2 |
| 可选：按事件表 | `results/tables_main.md` 的 "By event" 节 | 正文篇幅不够时放补充材料 |

## E. 参考文献

删除原数据来源条目；新增以下来源（按期刊格式补全）：
* EAGLE-I Power Outage Data 2014–2022 (ORNL, DOI 10.13139/ORNLNCCS/1975202) and 2024 (DOI 10.13139/OLCF/2500278);
* Hersbach et al. (2020), The ERA5 global reanalysis, QJRMS 146:1999–2049;
* NOAA NCEI Storm Events Database;
* USGS 3D Elevation Program; USFS NLCD Tree Canopy Cover (v2025-6); USGS Annual NLCD Collection 1;
* USDA NRCS Soil Survey Geographic Database (SSURGO), via Soil Data Access;
* U.S. Census Bureau 2023 Gazetteer and cartographic boundaries; USDA ERS Rural-Urban Continuum Codes 2023; EIA Form 861 (2023);
* TimesFM（稿件已引用 `das2024timesfm`，现用 3.0 版权重，建议注明版本）。

## F. 需要 PI 决定的写作问题

1. 结果与原稿叙事相反。按县留出（同一批风暴）时 GCRK 与宿主在抽样误差内不可区分（按县重抽
   95% 区间 −1.3% 到 +2.1%）；按事件留出时 GCRK 在 4/5 个 seed 好于宿主，但两个模型都不优于
   全零预测，差别主要是幅度过冲大小不同（最优缩放后差距缩小 83%）。是否仍把 GCRK 作为最终方法、
   贡献怎么表述，由 PI 决定。
4. 两份外部审阅的核对结果见 `RESULTS.md` 第 11、12 节；本清单里的 C1、C5 和 LOEO 段已按其收紧
   （不确定性区间、幅度过冲、分解只作算术拆分）。
2. Figure 3 按规则选到的县里 kernel 对预测几乎没有影响；机制展示可以放补充图 S3，但它是事后按
   模型输出选的，正文若引用必须说明。
3. 评估是"完美天气预报"设定（ERA5 再分析覆盖全窗口），需要在数据节写明。
