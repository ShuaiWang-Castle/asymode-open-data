# Ideas ledger (ARIS style)

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
