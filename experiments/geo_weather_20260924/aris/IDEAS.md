# Ideas ledger (ARIS style)

## D09 incremental geography-conditioned write (2026-09-29; finite paired training active)

PI要求把诊断落实为新设计和实际重训。保留原CRK的全部geo40/任意阶组合、多时标状态、
旋转和读出，只把单侧饱和锚点差写入改为直接天气方向与联合地理调制的天气增量。
精确谱尺度与径向限幅给出deposit对标定隐藏天气输入、联合地理特征的分量敏感度界；
不冒称整个原始天气到停电的敏感度界，也不保证响应方向完整或预测提升。新增参数0。
候选的联合写入容量也改变，不能把比较称为只改变数值条件或等函数重参数化。

固定D08面板、原事件fold2/3分别inner留出，W/旧CRK/新写入各seed0、900次全innerFIT更新，
六个有限任务；主要风险与S评分覆盖留出事件的全部原县。名单设计具有输入传导性，
不宣称独立未知事件验证。点推进门与CI分开，原全D约10%目标不变；通过也不自动开全D。
74项合成检查通过，独立评分已审查，源码和设计登记后先做临时开口的旧/新核三步FIT资源预检；
此登记时尚未跑真实预检或训练。详见 `notes/D09_INCREMENTAL_WRITE_DESIGN_20260929_ZH.md`。
登记提交eb91994已推送两research分支。真实旧/新核883单位三步开口预检均通过，
稳定更新约1.1/1.2秒，峰RSS1.038/0.884GiB，临时模型丢弃；21:24 ET fresh六任务队列
实际启动，coordinator21795，初始W/CRK fold2为21815/21816，两任务并行/各两线程/nice15。
结果待900步完整导出；不按中途分数改设计。新增纯汇总精度登记：float32合成数据暴露严格SSE恒等式
误报，另用d09_precision_score.py提升计算视图为float64，原scorer/模型/输出均不改；
7合成回归通过，边界舍入比较语义明确记录。独立终点审核脚本24项合成断言通过，
全部six-DONE守卫后重算主要点估计/1999次group及family区间/假峰/推进门。I18/I20监控暂停；原冻结源码/数据/产物保持不变，
C和稿件不触及。

## I20 controlled response kernel (2026-09-29; screen failed, parked)

中文结论：完整五折、每折900步、seed0已完成；S高停电县事件完整144小时设计加权RMSE仅下降
**0.365%**，merged-group 95%改善区间 **[-1.024%, +1.299%]**，未达到10%目标，区间也不支持
稳定改善。Family敏感性同样跨0。全D RMSE上升1.050%，热带/冬季平衡MSE上升2.494%，
五类平衡MSE上升4.203% [0.709%, 8.900%]。全零县低阈值假峰减少，但非S大假峰计数4→133，
设计加权率0.027956%→0.303121%。独立复核通过9,594项断言，源码/数据/输出hash与冻结登记一致，
双方8,457条OOF预测恰好一次覆盖，数值有限。复核结果并非外部科学审稿或因果证明。

状态：**parked after failed single-seed screen**。没有新训练队列，不自动加seed、NULL或预算。
后续若获授权，先诊断非S误放大与真S漏检对应的天气历史和地理结构；不事后改阈值或用MAE改善
代替原RMSE目标。核心天气→地理复杂调制→冲击→停电链条保留为待验证假设。I20交付后暂停监控，
I18继续暂停。详见 `notes/I20_CONTROLLED_RESPONSE_RESULTS_20260929.md`。

以下保留原授权和执行历史。

The PI requests a deeper redesign than GCRK and a complete comparison to no-kernel AsymODE, aiming for
about 10% or greater improvement in affected county-events. I20 uses all-subset geographic kernel features
plus full geo40 access, explicit write/retain/transform/read heads, and four bounded controlled relaxation
states within the existing damage MLP. It is non-nested with GCRK and retains the host objective and inputs.
The complete mathematical design and literature attribution are in
`notes/KERNEL_CONTROLLED_RELAXATION_DESIGN_20260928.md`; execution rules are in
`notes/I20_CONTROLLED_RESPONSE_SCREEN_20260928.md`. Arm `CRK+Cin`, label `v1_crk_s0`, five event folds,
seed0, 900 full-fit updates matched to host. **Training launched 2026-09-29 00:05 ET**, registered source
commit `6b6b964`, after 71 checks passed (one external-reference skip) and a disposable fit-only resource
preflight passed: 6,350 fit units, 10.143 s cold full update, peak RSS 1.82 GiB. One worker, two threads,
nice>=15, county chunks512; full-fit Adam semantics retained. Primary S-cohort target is >=10% full-window
design-weighted RMSE reduction. At launch no outer I20 result existed. This authorization supersedes older proposal-only language below, without authorizing extra
seeds, NULLs, a changed loss, C access or manuscript work. I18 monitoring remains paused.

2026-09-29 compute amendment: PI freed CPU and explicitly authorized multiple processes/threads.
Increase to at most three folds in parallel, retaining two threads per fold and the same model/seed/budget.
Use the separately registered parallel coordinator, preserve the live first fold and original manifest;
24 scheduling/evaluation checks passed. See `notes/I20_COMPUTE_AMENDMENT_20260929.md`.
At the parallel handoff, folds1/2/3 were live; the first-fold process was retained; folds4/5 were pending.

Status: **active**, **parked**, **abandoned** (with the reason and, where applicable, prior art), **absorbed** (merged
into another idea).

| id | idea | status | note |
|---|---|---|---|
| I01 | GCRK: geography-conditioned response kernel on the damage hidden state | parked | fails on public data (open_gcrk RESULTS 18-21); PI suggests testing it without the bound on its opening alpha (I10) |
| I02 | v0 local mechanisms with learned physical scalars | abandoned | zero gradient at step 0, clamping, weak identification (DESIGN section 7) |
| I03 | Exposure-integrated hazard (EIH): customer-weighted integral of local, weather-gated, fading-memory hazards; competing-hazard entry | abandoned (PI 2026-09-27) | geography enters only through GCRK or a kernel architecture in the hidden layer, never through hand-built hazard features; DESIGN v1; audits F0/F1 built in |
| I04 | Sub-county geography via ERA5 elevation bands | abandoned on these data | C01, C02, C06 |
| I05 | HRRR as the weather source of the system | active | C06, C07; host inputs not yet on HRRR |
| I06 | Load x trigger slots (chain reactions) | active, untested at scale | W1 event-grouped single seed: -0.28% vs base |
| I07 | Within-county normalised susceptibility | parked | W1 single seed -1.42%; no fingerprinting by construction |
| I08 | Target cleaning of EAGLE-I artefacts | abandoned as a gain | C03 |
| I09 | Designed panel of parent weather systems for the whole system | active (design registered) | DATASET_DESIGN v1: five regimes, sealed confirmation tranche, pre-window gates, near-miss controls, regime-balanced estimand |
| I10 | Remove the bound on GCRK's opening (beta = alpha instead of tanh(alpha)); and the non-negativity clip of the EIH coefficients (signed hazard) | active (PI request 2026-09-26) | to test on the new panel with three seeds |
| I11 | Host peak magnitude (predicted peaks 0.18x observed above 20%) | active, **next by DATASET_DESIGN 9.3** (Stage 0: two headline regimes) | the largest error budget (notes/DATA_PATTERNS.md); the host fails zero in synoptic wind and heavy rain on the designed panel |
| I12 | Hazard dictionary for every regime (local gust exceedance, phase-resolved precipitation, temperature-gated loads, node-level compound products, antecedent wetness, convective organisation) | active (design) | from the physical review of the panel design (contrib/REVIEW_dataset_physics.md section 5) |
| I13 | Diagnose the heavy-rain loss of the hazard pathway (+11% vs host at seed 0 with the full or the mechanical dictionary) before any new arm: which active terms fire in heavy-rain systems (candidate: poorly drained x wetness x gust exceedance, where wet soils meet thunderstorm gusts without outages), by stratum and hazard class | active (diagnostic, development only) | 2026-09-27 cycle; coefficients in runs/.../v1_H*_s0/fold0k/DONE.json |
| I14 | Spatio-temporal GCRK: the response states of neighbouring counties are coupled inside the kernel's recurrence, e_t = M_t^-1 (P_t e_{t-1} + f_t), P_t a Markov mixing (non-negative shares, row sums <= 1, so the unit bound holds): a static share kappa_s over the neighbours (exp(-d / 50 km) x exp(-gamma |code_i - code_j|^2), the geography code of the kernel), and a directional share kappa_a from upwind neighbours (ERA5 10 m wind at the pair's midpoint, scaled by speed / 10 m/s). Neighbours: the sampled county-events of the same system within 150 km, at most 8 (panel_v1/space_v1.py). All three scalars start at 0, so the arm equals GCRK at step 0 (paired init) | screened 2026-09-27, seed 0: ties the host (pooled -0.06%, regime-balanced +0.3%), beats temporal GCRK (-0.50% pooled, -0.4% balanced; tropical -3.2, winter -1.5, synoptic +2.5); coupling used in 2 of 5 folds, gamma ~ 0; not queued for more seeds | reason: GCRK is only a temporal kernel; each county's state sees its own weather alone. Single seed first: STGCRK+Cin seed 0 against GCRK+Cin and the host W+Cin seed 0, five event folds. Later (PI): a recovery-side kernel; a spatial null (neighbours replaced at matched distance) only for a survivor |

## Proposed after the frozen kernel review (2026-09-28; no training authorization)

| id | idea | status | note |
|---|---|---|---|
| I15 | Recovery hidden-state response kernel driven by the existing damage hidden departure and predicted damage rate; minimal version has no predicted-stock feedback | proposed, awaiting PI discussion | Kernel states and codes are active but forecast contributions remain small; recovery lacks dedicated impact history. Fixed-damage zero-recovery ceiling still misses most large peaks, so this tests history, not a claim that slower restoration cures magnitude. Formulas, boundedness, paired initialization and screening in notes/KERNEL_PROPOSALS_20260928.md. |
| I16 | Public service-county links inside the same recovery recurrence, with fixed context counties and bounded Markov mixing | proposed, after I15 and public-graph audit; awaiting PI discussion | The sampled spatial graph is not a service network; strong learned mixing has negligible frozen output sensitivity. Public EIA links are a proxy for shared service context, not observed crew flows; context coverage and sealing must be audited before implementation. |
| I17 | Population/geography-type response states within each county, combined after local nonlinear processing into the one shared damage hidden layer | proposed alternative, awaiting PI discussion | Targets exposure aggregation and magnitude. Existing geography already includes 50 km summaries and co-location fields; the change is preserving local weather-geography correspondence before aggregation, not adding another county-level feature list. |

No proposal is queued. Frozen constant-geography substitutions are dependence diagnostics, not trained NULLs.
The registered primary estimand after Stage 0 is tropical/winter balanced MSE; all-five balance and pooled RMSE
are reported separately. The prospective single-seed screen must be explicit and frozen before a PI-approved run.

## I18 geographic-kernel screen (2026-09-28; authorized before training)

| id | idea | status | note |
|---|---|---|---|
| I18 | Replace the county-specific geography norm in GCRK with one frozen training-fold RMS norm, retaining radial geography information; all rate/kernel parameters and recurrence otherwise unchanged | completed five folds; screen failed, no seed or NULL expansion | Headline MSE vs host −2.358% [95% −5.081,+1.218], vs GCRK −3.445% [−7.744,−0.071]; all-five vs host −0.366%; pooled +1 RMSE vs host −0.484%. Only failed gate: synoptic wind +2.0745% vs host exceeds frozen +2% limit. Original registration unchanged; full results and export validation in notes/I18_GEO_RMS_RESULTS_20260928.md. |

## I19 geographic read-in (2026-09-28; design requested, no training queued)

| id | idea | status | note |
|---|---|---|---|
| I19 | On the fixed I18 base, let geography set a bounded diagonal metric on hidden weather departure before both deposition normalization and its gate; 128 added parameters, identity initialization, same recurrence and one damage MLP | parked by latest PI direction: data analysis first; implementation and training not started | notes/I19_GEO_READIN_PROPOSAL_20260928.md remains the historical proposal based on three folds. Full I18 results supersede its interim performance narrative. Existing lambda/a/Omega already condition the kernel; the input-selection hypothesis is not an identified bottleneck. |

## Current priority after I18 (2026-09-28)

The PI requested data-first analysis of weather order, overlap/compound exposure, and geography before choosing
the next kernel architecture. Use D only, existing family-held-out structure and explicit controls for severity,
duration, initial outage and repeated counties; distinguish sequence, cross-weather alignment and geography
correspondence NULLs. I19 is paused, and no proposal or historical queue authorizes another training run.
I18's primary gain does not eliminate the large-peak gap: median predicted/observed peak is 8.94% on the
726 observed-large cases, versus 9.03% for original GCRK. This is descriptive, not a selection rule.

D01 fixes the first data-analysis scope before fitting: additive controls, individual weather/geography,
compound exposure, directional order, compound/geography, then order/geography; event-held-out ridge probes
with future-severity and same-window sensitivities. See `notes/D01_DATA_FIRST_PROTOCOL_20260928.md`.

D01 completed all three specifications on five event folds. Joint geographic modulation increases headline
burden MSE by +3.292%, +3.227%, +5.249% (all merged-event intervals above zero; 0/5 folds improve in each).
No probe satisfies its exploratory candidate rule. The probes also lack a stable advantage over persistence,
so the result cannot rule out physical weather/geography mechanisms or within-county co-location. Next evidence
should distinguish inadequate summaries from inadequate spatial support, with narrowly specified comparisons;
it does not justify selecting an architecture merely by adding more county-level cross-products.
See `notes/D01_DATA_FIRST_RESULTS_20260928.md`. I19 remains parked; no neural expansion is queued.

PI correction: D01's overall result must not narrow the investigation prematurely. D02 retains the broad
weather/geography space and asks whether county structure reveals opposing directions, shifted timing,
different thresholds or event-composition effects. Current-event-outcome-blind structural types and full
support/uncertainty maps come before another architecture choice; overall MSE is not the atlas's gate.
See `notes/D02_COUNTY_COMPLEXITY_SCOPE_20260928.md`.

D02 completed a full county-response atlas rather than ranking one kernel: 2,410 counties, continuous
46D structure plus six coarse partitions; six weather-anchor trajectories; 56 drivers across all five
regimes/six types, three responses and all fixed-effect/history sensitivities. Timing differs within and
between regimes. Full-range mean-gust associations in tropical events differ in amplitude, while restricting
weather support can change signs. Apparent raw rain-sign cancellation in heavy-rain events disappears after
joint county/system-phase/history adjustment; that adjustment cannot identify which component explains it.
Tropical gust/rain order associations remain hypotheses, with broad common-support intervals. No single
example is the next-design gate. Preserve continuous within-type variation, nonlinear support/density,
multi-peak and longer-history possibilities; do not turn six clusters into six asserted physical mechanisms.
See `notes/D02_COUNTY_COMPLEXITY_RESULTS_20260928.md`; all weak, null and undefined cells remain in the outputs.

## High-dimensional impact-kernel direction (2026-09-28; design research only)

PI's durable core: **weather -> complex geographic modulation -> impact -> outage**. Geography transforms
weather into impact formation, interactions and accumulation; the single damage head and existing host
dynamics connect this latent representation to observed stock. Outage-response surfaces diagnose the chain
but do not identify physical impacts or separate failure/recovery from stock alone. Carry this structure
into future manuscripts, separating observations, assumptions and mechanisms.

D03 completed the outcome-value-blind audit: geo40 needs 11 linear directions for 90% descriptor variance
(geo40+context6: 14), weather means retain 69.0%/73.1% of standardized 24-hour path variation, eight fixed
time coefficients retain 97.7%/98.3%, and 580 counties have weak first but stronger second aspect harmonic
at threshold 0.1. These motivate retaining continuous geography, time shape and direction distributions;
none proves a useful response rank or physical interaction. See `notes/D03_HIGH_DIM_STRUCTURE_RESULTS_20260928.md`.

Recommended statistical direction: jointly estimated nonlinear weather-lag response surfaces with continuous
geographic modification, low-rank/smooth sharing, cross-weather and repeated same-weather lag-pair terms,
joint-path support and event/county dependence. Preserve higher-order and longer-history possibilities.
Candidate hidden implementation: bounded geography-conditioned read-in, multiple memory scales, symmetric
and directed second-order states, and bounded readout into the existing damage layer. GCRK already expresses
order; explicit conditional impact response is the proposal, not first-time memory capability.

This is a research direction, not a new I-number screen or training queue. I19 remains parked. Synthesis:
`notes/KERNEL_HIGH_DIM_DATA_MOTIVATION_20260928.md`; primary references:
`notes/KERNEL_HIGH_DIM_LITERATURE_20260928.md`, `notes/KERNEL_GEO_PROCESS_LITERATURE_20260928.md`.

## D04 response evidence and implementation implications (2026-09-28)

The finite high-dimensional statistical analysis is complete. It keeps the core chain weather -> geographic
modulation -> latent impact formation/combination/accumulation -> outage, all counties and all five regimes.
C/A all-five MSE is +0.935%, D/B +3.065%; these are conditional one-hour net-change diagnostics, not a new
neural screen. They do not validate a next kernel and are not a gate for deleting local phenomena.

Retained evidence: strong rise/decline asymmetry, underpredicted positive-increment peaks, false peaks in
no-positive events, and heterogeneous local lag sensitivities. The more complex model actively uses synchronous,
symmetric and ordered weather blocks, but this correlated decomposition is not unique mechanism evidence.
County-type differences also exist without explicit geography×weather terms; exposure composition and event
concentration remain material. Removing outage-history controls barely changes the total geo-component RMS.

All 116 primary/sensitivity fits are finite but miss the gradient criterion, and only the chosen operator
was saved. Next priority before architecture selection is an explicitly scoped solver precision/stability
and observation-time audit, preserving this run and all unfavorable findings. Keep the candidate continuous
geo-conditioned read-in, multiple memory scales and symmetric/directed hidden states as a hypothesis;
do not hard-code provisional county-type signs or copy selected rank2 into the neural architecture.
No extra fit or neural experiment has started. Full evidence and limitations:
`notes/D04_CONDITIONAL_IMPACT_RESPONSE_RESULTS_20260928.md`.

## D05 authorized data forensics (2026-09-28)

The PI asks why large outages are missed and whether weather before/after their occurrence and county
structure explain the failures. D05 is now specified in `notes/D05_LARGE_OUTAGE_FORENSICS_SCOPE_20260928.md`.
Separate rapid onset from high stock; test timing versus amplitude underprediction, inspect persistence
and observation support, retain all weather combinations, and use both outcome-selected anchors and an
outcome-independent fixed-clock risk set for county comparisons. No new neural training is authorized by
these descriptive results alone. The scientific chain remains weather -> geography-modulated latent impact
-> outage; neither stock nor matched associations identifies physical damage. Record findings after the
finite one-process analysis, with unsupported cells and negative findings retained.

## D05 completed: missed peaks and kernel motivation (2026-09-28)

The bounded audit is complete. Large net jumps (2,963) and high stock (726) overlap in 723 county-events.
All three frozen neural trajectories strongly attenuate severe peaks; Georms full-window/common-support
peak ratio median is 8.936% unweighted / 1.225% design-weighted. Even arbitrary timing within that window
does not repair amplitude. At a J maximum, 22.59% design-weighted cases reverse >=80% next hour, with
observational support explicit; this is neither a reporting-error rate nor physical restoration speed.

Same-county contrasts show preceding/near-time weather signals and anchor-dependent wind/rain order
coordinates, while winter high stock follows elevated prior snowfall. Centered products do not establish
physical co-occurrence or non-additivity. Same-event/time full-risk matching retains conditional continuous
geography associations; severe-stock state matching collapses from 671 to 121 cases, so state/accumulation
and common support cannot be replaced by one static geographic multiplier. The 40-coordinate results and
all weak/reversed findings remain public; no post-hoc coordinate becomes a confirmed mechanism.

Candidate design constraints: geography-conditioned short-shock and slower-memory transformations inside
the single damage MLP, explicit onset/amplitude/false-peak checks alongside unchanged overall evaluation,
and conditional order tests preserving exposure intensity and support. Impact remains latent. Do not
hard-code forest/terrain signs, infer physical rates from stock, or start a new neural arm from these
descriptions. Full evidence: `notes/D05_LARGE_OUTAGE_FORENSICS_RESULTS_20260928.md`.

## Joint geography and dynamic conditioning clarification (2026-09-28)

The PI explicitly asks about combinations such as soil/vegetation/terrain and their changing influence
through weather history. Static geography can parameterize a dynamic process: the same county may have
different forcing sensitivity, thresholds and persistence as latent state evolves. D05 marginal slopes
do not rule out this joint response, and the severe-stock matching support loss does not prove a static
model cannot work. Existing GCRK already mixes geo40 nonlinearly and has geography-conditioned recurrence;
I18 did not newly introduce dynamics. Avoid presenting the candidate as the first geo interaction/memory.

Candidate extension: retain an exactly recoverable GCRK baseline, expand joint geography conditioning,
then separately test weather-dependent and prior-latent-state-dependent read-in, threshold and dissipative/
skew parameters. Positive dissipation plus skew structure can retain a bounded state, but state-dependent
gradients must be derived and checked. The existing custom backward is not automatically valid.

Distinguish county attribute products from actual spatial co-location. Existing steep-forest and forest-soil
descriptors provide partial joint information, but no sediment-material inventory, channel connectivity or
line-level exposure is observed in geo40. A more expressive county kernel cannot reconstruct lost joint
spatial structure. Mudslide/soil-failure mechanisms are physical analogies, not observed outage labels.
No new fit, neural arm, NULL or data expansion was launched. See
`notes/KERNEL_JOINT_GEOGRAPHY_DYNAMIC_STATE_20260928.md` for primary sources, feature semantics and nested tests.

## PI correction: no predetermined geographic combinations (2026-09-28)

Physical examples must not define a hand-picked interaction list. The default candidate is a dense joint
map of the full geo40 vector, with full-vector access alongside the old code and a learned residual
representation. No chosen variable pairs, semantic block masks, required main effects, predetermined
signs or fixed second-order ceiling. Conditional residual heads may depend on available weather and
prior latent state, with the existing numerical bounds retained. Semantic groups are for source checks
and post-fit reporting, not restrictions on which coordinates may interact.

This permits arbitrary attribute subsets to participate structurally; finite capacity, optimization and
joint data support still limit representation and identification. Original GCRK already permits implicit
high-order mixing, so the new hypothesis is broader joint conditional capacity beyond the sole four-code
bottleneck, followed by separately testable weather/state conditioning. Distinguish nested added-capacity
comparisons from budget-matched non-nested controls. The prior low-rank cross-block recipe is an optional
representation comparison, not the default design. Updated durable instructions are in `program.md` and
the joint-geography note. This clarification does not start a new fit or experiment queue.

## D06：外部审查后的宿主响应链诊断（2026-09-29）

PI 要求分析 GPT 审查并继续。先按 `notes/D06_HOST_RESPONSE_INTERFACE_SCOPE_20260929.md` 验证公共 D 上的宿主可达包络、完整 S 轨迹 oracle、FIT 目标权重和冻结 fold1 隐层工作点。外部合成例不当作 D 结果；scattering 暂为未训练候选。本轮不改冻结模型、不启动新 arm；监控保持暂停。

## D06 完成：可达性不是充分解释，响应选择性成为下一诊断重点（2026-09-29）

外部 15 项探针复现通过；真实 D 下界违反设计权小时比例 8.820%，但全部 726 个 S 的合法速率 oracle SSE 只占宿主 SSE 的 0.8190%–0.8294%（数值证书区间，非 CI）。因此恢复 cap 存在失配，不能充分解释严重漏峰。

冻结 fold1 回放与导出最大差 1.49e-8。CRK 相对宿主 FIT S RMSE 降 44.03%，OUTER S 反而升 0.455%；OUTER 总体升 4.52%。核 effect/h1 在 OUTER S 峰时中位数 70.49%，写入接近单位范数且大量 tanh 坐标饱和，不能说整核太弱。真实组合基函数控制项仅约原地理项 0.5%–1.5%，但中心化有效秩 13.87–17.43，没有一维坍塌证据。S 仍占 89.10%–93.42% 的 FIT 归一化零预测损失。五折 non-S 假峰交集为共同 3、仅宿主 1、仅 CRK 130。

优先进一步区分局部导数/时序选择性和跨事件泛化；维持全地理组合空间。scattering 能量恒等式成立但点态界不同，本轮不集成、不训练、不自动新增 arm/seed/NULL。I18/I20 监控保持暂停。完整报告：`notes/D06_GPT_AUDIT_REAL_DATA_RESULTS_20260929_ZH.md`。

## D06 独立复核后的修正（2026-09-29）

原 JSON 22 个引用数值核对一致，外部三项合成检查复现通过。FIT S RMSE −44.03% 不等于严重峰值普遍学会：峰幅比设计权中位数 1.231%→0.800%，90% 分位数 16.13%→90.61%；OUTER non-S RMSE +25.31%。下一问题应同时包含训练内部响应分配与跨事件稳定性。

饱和锚点限制单坐标一侧写入的推导正确；补充结构界显示，零锚点每模式写入范数至多 0.5，饱和锚点的坐标盒上界可趋向 1，幅度预算与方向偏置耦合。此为结构性质，不是已测优化归因。先做配对误差方向分解、真实/预测峰时联合工作点及统一 FIT 分母下的事件梯度设计；保持全地理组合空间。函数保持的坐标平衡不保持 Adam 轨迹，稳定状态不等于控制映射敏感度有界。本轮仅核对汇总与代数，没有新前向、梯度或训练。见 `notes/D06_INDEPENDENT_REVIEW_RESPONSE_20260929_ZH.md`。

## D07 真实数据与有限干预登记（2026-09-29）

PI 要求深入当前 D 并做实验提供证据。按 `notes/D07_RESPONSE_SELECTIVITY_SCOPE_20260929.md` 执行逐单位收益分解、联合写入方向与 ±5% 内部写入敏感性、原 FIT 目标下事件梯度与六个固定小参数扰动。只用冻结公共 D/seed0，保留全部不利结果，不启动新全训练候选；旧监控暂停。

## D07 完成：从总体泛化问题收束到响应分配与控制敏感边界（2026-09-29）

全五折配对显示267/726严重县事件轨迹获益，覆盖19.98%设计权，前73单位贡献95.81%正收益；FIT409/559获益，不能说只拟合少数县，但其设计权仅46.87%、前56单位占83.27%正收益。训练峰幅与轨迹收益不同：52个FIT轨迹获益单位峰幅误差反升。OOF短时严重峰受损（仅1小时RMSE恶化12.75%），持续≥7小时仅改善1.06%，两者抵消；完整144观测S改善只0.0096%。

方向锁定是广泛工作区症状，实际相邻写入仍有小变化；假峰预测峰时控制Jacobians更大、锁定较低，弱化“假峰因为更饱和”解释。两个固定写入实验中，×0.95仅减轻non-S（OUTER RMSE−2.05%、假峰49→47），S几乎不变。统一减强度不能恢复严重峰。

61原系统×S/non-S的112梯度分区恢复全FIT梯度，相对L2差3.90e-6。S目标占82.66%，全S/non-S梯度余弦−0.476，天气/组合地理/出口块存在冲突，原地理块不统一冲突。0.1%天气控制梯度扰动使OUTER S RMSE−0.348%，但non-S+5.10%、假峰49→56；1%扰动已失去FIT下降并明显增加假峰。六个OUTER全目标均恶化，五个S目标改善，不把全集合结论扩大到所有亚组。

数据要求下一版保留短时起涨和持续累积、联合地理的任意组合、控制输入尺度及写入选择性。未来标签/真峰只能用于诊断，不能用于模型路由；状态有界仍不足以控制输入敏感边界。FIT-only函数保持的地理条件处理是可登记候选，不是已验证机制。没有新增训练/倍率/seed/NULL/scattering；I18/I20监控保持暂停。详见 `notes/D07_REAL_DATA_SELECTIVITY_RESULTS_20260929_ZH.md`。

PI补充全局覆盖目标并允许固定小面板先测。小面板应基于FIT的完整天气路径×联合geo40覆盖和事件分组，不按OUTER获益择县；用于机制/优化定位，最终回到全D原五折，并另行登记地理环境转移检验。规模、输入覆盖与真实泛化须区分；小面板不限制地理组合空间，也不使用未来真峰作为路由。本轮没有追加小面板训练。

## D07独立复核与固定输入小面板D08（2026-09-29，登记）

现有CRK二选一整轨迹oracle的S RMSE改善上限1.95994%，只约束两条冻结预测，说明减少错误响应之外还需新增有效修正。达到10%所需SSE收益约现有正收益4.895倍，不是出口幅度倍率。冻结天气non-S梯度只有S的5.59%，总梯度几乎沿S；冲突余弦不证明严重病例得不到学习。严重小时数不等于连续过程，补连续段与观测支持审计。

按PI已授权的小数据测试方向登记D08：原fold1FIT全部53合并组、cap24、16确定性联合输入核心加8均匀随机尾部，保留216h天气12与geo40；不按S/峰时/CRK收益选县。冻结名单、尺度、pi与哈希后查看真实响应和既有冻结FIT预测。实际输入对照不足也保留，不重抽至有利。原fold2…5标签不冒充冻结模型的事件留出。首个写入结构候选与地理等函数重参数化分开，未启动新训练/seed/NULL或恢复监控。登记见notes/D08_INPUT_PANEL_SCOPE_20260929.md，审查回应见notes/D07_INDEPENDENT_REVIEW_RESPONSE_20260929_ZH.md。

## D08完成：真实条件差异与部分代表的小面板（2026-09-29）

输入先冻结的1219县事件覆盖819县、61系统、全部53合并组，保留216h天气12与geo40。284对照记录为260唯一无序对子：天气近/地理远的绝对真峰差中位1.286个百分点，21/157记录≥10个百分点；地理近/天气远中位2.325个百分点，29/127记录≥10个百分点。均值遮住普通与极端差异并存，但剩余天气、基础设施、背景及测量未平衡，不能据此归因地理或特定组合。40合成检查、22真实输入检查、420评分行独立复算通过；冻结源码/训练产物/缓存保持不变。

面板复现收益集中、近满幅缓变写入及已有小天气参数扰动的S收益/non-S代价；没有完整复现原设计权峰幅比中位数下降，也漏掉全部4个原FIT CRK基线假峰。原权Kish支持187.57，w/pi仅89.41。固定名单保留，不按结果重抽；可用于机制/数值测试，不能成为选择新核的唯一风险门。原折2…5标签不把已训练FIT缓存变成留出预测，49.19%的S下降不是新模型成绩。严重小时总数与最长连续段分开：全FIT308个≥7小时单位中13个不满足连续≥7，其中10个完整观测，不能把总数等同于持续物理机制。

下一版须同时生成遗漏的有效响应与抑制错误响应；只对现有整轨迹开关的真值oracle改善仅1.96%。首个结构对照单独改变写入控制工作区与条件敏感度，保留宿主/Q/rho/出口/目标；函数保持的地理重参数化单独检验。需要另行固定事件内层重训协议及全FIT/OUTER假峰检查，再回到全D原五折与10% S目标，不用小面板结果自动扩大核、训练预算、seed或NULL。本轮无新前向、梯度、干预或训练；I18/I20暂停，无待运行队列。详见notes/D08_FIXED_INPUT_PANEL_RESULTS_20260929_ZH.md及results/v1/d08_evidence.json。
