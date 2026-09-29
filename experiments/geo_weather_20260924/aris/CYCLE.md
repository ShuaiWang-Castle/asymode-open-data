# ARIS cycle — what every scheduled wake-up does (2026-09-27)

**D06 active, 2026-09-29:** PI requests verification and continuation of the external GPT audit.
Execute only `notes/D06_HOST_RESPONSE_INTERFACE_SCOPE_20260929.md`: public-D reachability/oracle,
FIT objective and frozen fold1 diagnostics. No new neural training; I18/I20 monitors stay paused.

**I20 final state, 2026-09-29: I20 complete; the registered screen failed.** All five folds completed
900 updates and automatic evaluation. Independent final audit passed 9,594 assertions: exact held-out
coverage for both models, source/data/artifact hashes, finite exports/checkpoints, all tail point scores
and intervals independently recalculated. S full144h design-weighted RMSE improved only **0.365%**,
merged-group 95% improvement interval **[-1.024%, +1.299%]**; neither positive improvement nor the 10%
target is supported. Family sensitivity also crosses zero. All-D RMSE worsened **1.050%**, headline
balanced MSE **2.494%**, all-five balanced MSE **4.203% [0.709%, 8.900%]**. Low-threshold alarms in
observed-zero units fell, but non-S predicted peaks>=10% rose from **4 to 133** (weighted rate
0.027956% to 0.303121%). No training/coordinator process remains alive.

中文结论与完整口径：`notes/I20_CONTROLLED_RESPONSE_RESULTS_20260929.md`；复核回执：
`results/v1/i20_final_audit_s0.json`。本轮结束后暂停 I20 监控，I18 保持暂停，不归档聊天。
当前无待运行队列；无自动加 seed、NULL、新 arm 或候选专属延长预算。任何旧队列指令均为历史。
核心天气→地理复杂调制→冲击→停电假设仍需数据证据；本次负结果不识别机制或地理净信息。

### Historical I20 authorization and execution record

**Latest PI authorization, 2026-09-28: I20 implementation and full five-fold training.** The PI requests a
substantially redesigned kernel and comparison to no-kernel AsymODE, targeting about 10% or greater error
reduction in affected county-events. Current design/registration:
`notes/KERNEL_CONTROLLED_RELAXATION_DESIGN_20260928.md` and
`notes/I20_CONTROLLED_RESPONSE_SCREEN_20260928.md`. Arm `CRK+Cin`, label `v1_crk_s0`, seed0, five existing
event folds, 900 full-fit updates matched to `v1_host_s0` / `W+Cin`; fixed original data/loss. Launch only
after implementation checks and source registration commit. **I20 execution phase: launched 2026-09-29
00:05 ET**, registered commit `6b6b964`; initial runner PID 42136, fold1 PID 42149. This later authorization supersedes the older “no new neural arm authorized” text;
it does not authorize extra seeds, NULLs, changed loss, C access or paper work. I18's monitor stays paused.

**Compute amendment, 2026-09-29:** the PI freed CPU and authorized parallel training. Use at most
three training processes, two numerical threads each, nice>=15; see
`notes/I20_COMPUTE_AMENDMENT_20260929.md`. Preserve the already-running first fold while handing off the
serial coordinator to `run_cr_parallel.py`. Resource plan uses exact full-fit
gradient accumulation over county chunks (one Adam update and one shared drop-path coin per full step),
not stochastic minibatch training. Preserve partial run folders and never duplicate a live queue. Both
the affected-county target and all-D/false-peak results must be reported, not substituted for each other.

Handoff completed at about 00:20 ET: coordinator PID46146, adopted fold1 PID42149 unchanged,
fold2 PID46192, fold3 PID46196. Old coordinator PID42136 exited. Folds4/5 remain pending. After startup,
all three training PIDs were active at nice15, combined RSS about3.15 GiB, memory-pressure free42%.
The source/evaluation hash manifest remains unchanged; compute amendment registration is `d737ae7`.

### I20 execution phase (completed; retained for provenance)

The 71 targeted checks passed (one external-reference skip); the disposable public-D fit-only preflight
passed on 6,350 fit units, one full update 10.143 s including cold JIT/calibration, peak RSS 1,952,989,184 bytes.
County chunk512, exact first-order adjoint, geographic rematerialization only. Preflight model discarded;
formal folds start from seed0. Resource record: `results/v1/i20_resource_preflight.json`.

1. Read `runs/geo_weather_20260924/v1_crk_s0/{SCREEN_RUN,RUN_STATUS,HOST_REFERENCE}.json`,
   `fold*/{DONE,RUNNER_RECEIPT}.json`, `logs/runner_i20.log` and `logs/screen_v1_crk_s0_f*.log`.
   Cross-check actual owned PIDs, CPU/RSS and system memory; do not infer a stall from the 20-update log interval.
2. Follow the current parallel coordinator in RUN_STATUS; `COMPUTE_AMENDMENT.json` records handoff and
   `PARALLEL_WORKERS` controls future concurrency (1–3). Keep at most three children, two threads each,
   nice>=15. The original SCREEN_RUN workers=1 is preserved historical registration, superseded only for
   scheduling by this compute amendment. Do not duplicate the queue, change
   frozen source files, overwrite any existing/empty fold directory, or signal unrelated processes.
3. The queue automatically validates all five exports and scores `screen_i20_s0_headline_vs_host.json`,
   `screen_i20_s0_all5_vs_host.json`, `i20_cr_tail_s0.json`, `screen_i20_s0_verdict.json`.
   A completed/scored receipt must match frozen source/data and artifact hashes. Partial outputs are preserved.
4. Primary target is S (observed forecast peak>=10%), full144h design-weighted RMSE reduction>=10% versus
   frozen no-kernel `v1_host_s0`/`W+Cin`. Report point target, merged-group confidence interval, family sensitivity,
   all-positive/J/all-D, per-regime/fold and false-peak results; fixed-origin +h snapshots and legacy pooled
   suffix +h must stay distinct. Single seed cannot establish net geographic information or causality.
5. Normal unchanged progress stays quiet; notify new completed folds, failure, resource anomaly, all results
   or a needed user decision. If training is complete but aggregation fails, first check that no evaluator
   is alive, then repair only aggregation using the registered code; do not retrain or overwrite evidence.
6. Record final conclusion in RESEARCH_LOG/IDEAS, update current state, explicitly stage only I20 compact
   results/notes, scan private paths/emails/restricted markers, push both research branches, never main.
   Pause the I20 monitor after delivering results, or after a failure requiring user intervention; do not archive
   the chat. Do not resume I18, add seeds/NULLs/new arms, use C, or edit/import the manuscript directory.

**Historical state, 2026-09-28 after I18:** all five I18 folds are complete and scored; the frozen screen failed
only the synoptic-wind +2% guardrail. No seed/NULL expansion or next-arm training is queued. The PI now asks
for data-first analysis of weather order, overlap and geography; I19 is parked. This state supersedes all
older automatic queue/next-design directions below. C and paper work remain prohibited.

Long-running work on the designed panel follows the ARIS split (Auto-claude-code-research-in-sleep,
`skills/shared-references/external-cadence.md`). The scheduler only decides **when** to look. It never decides a result:
* keep and discard follow program.md rule 3b and DATASET_DESIGN amendment 2 S10;
* a claim needs an independent reviewer receipt (`aris/CLAIMS.md`).

Every wake-up runs the steps below in order and stops early if nothing changed.

## 1. Look

* Folds done per label (`runs/geo_weather_20260924/<label>/fold0k/DONE.json`), the running job, and the queue logs
  (`logs/queue_v1_*.log`).
* Load average and memory.
* **One training process at a time** (the PI uses the machine). Everything runs at the lowest priority (nice ≥ 15).
  Never touch processes this session did not start.

## 2. Evaluate what finished

For every label with five folds done and no result file, run `evaluate_v1.py` against its registered comparators and
write `results/v1/screen_s<seed>_<arm>_vs_<ref>.json`.

| arm (label) | comparators |
|---|---|
| v1_gcrkopen_s0 (GCRK+Cin, opening unbounded) | host; v1_gcrk_s0 (bounded twin) |
| v1_gcrk_s0 (GCRK+Cin, opening bounded) | host |
| v1_Hmech_era5_s0 (mechanical-load subset, ERA5) | host; v1_Hera5_s0 (full ERA5 dictionary) |
| v1_Hmech_hrrr_s0 (the same subset, HRRR) | host; v1_Hmech_era5_s0 (source twin) |
| v1_Hhrrr_s0 (full HRRR dictionary) | host; v1_Hera5_s0 (source twin) |

## 3. Decide by rule

* **Single-seed screen (seed 0).** Discard an arm if its regime-balanced relative change against the host is ≥ 0.
  Otherwise it is a candidate: queue seeds 1 and 2 for the arm, for its twin, and for the host.
* **Three-seed keep** (seed-averaged predictions). All of the following must hold:
  * the regime-balanced improvement against the host exceeds 1%, with a 95% family-cluster interval that excludes 0;
  * the improvement against the twin is > 0, also with an interval that excludes 0;
  * no non-headline regime gets worse by more than 2%.
* **Stage 0 (DATASET_DESIGN 9.3).** Host seeds 1-2 are needed anyway: they decide the headline regimes. Seeds 3-5 are
  the A/A run, which decides the seed count.

## 3b. After Stage 0 (2026-09-27)

* The headline regimes are tropical and winter. The keep test's regime-balanced gain averages over these two.
  Synoptic wind, convective and heavy rain may not get worse by more than 2% (seed-averaged point estimate).
* Fewer than three regimes qualify, so the next new work is the host itself (idea I11). It needs a written design
  before it runs. The only pathway work that continues is the running candidate test: the HRRR mechanical subset,
  seed 2, against its ERA5 twin and the host.

## 4. Queue the next work, in this order (skip what is done)

1. The seed-0 queue `jobs_v1_stageB_s0b.txt`: GCRK open, mechanical ERA5, GCRK bounded, mechanical HRRR, full HRRR.
2. Host seeds 1-2, five folds each (Stage 0).
3. Seeds 1-2 of every candidate and its twin (step 3). After Stage 0, only the HRRR mechanical subset (see 3b).
4. The A/A host seeds 3-5 (`jobs_v1_aa_s345.txt`, queued after the candidate).
5. Then the host work of I11, once its design is written.

A new design (a new arm or feature set) is not started by a wake-up. It needs a line in `aris/IDEAS.md` and a reason
written before it runs. The sealed tranche C is never built, read or evaluated.

## 5. Record and push

* One line per decision in `RESEARCH_LOG.md`. `aris/CLAIMS.md` changes only with a reviewer receipt.
* Before every commit, a scan for private paths and the PI's email. The PI lifted the internal firewall on 2026-09-28
  (FIREWALL.md removed). The GitHub repository is public, so non-public data and numbers measured on it still stay out
  of every commit unless the PI confirms they may be published.
* Commit with the attribution line, then push `research/geo-weather-process-20260924` and fast-forward
  `research/tropical-evidence-20260926`. Never push main.

## 6. Report

Two to five lines to the PI, in Chinese:
* what finished;
* each verdict, with its number and interval;
* what runs next, and when it should finish.

If nothing finished and nothing failed, say nothing.

## Paper phase (from 2026-09-27 14:20; revised 15:10)

The PI asked for an open-data manuscript with four forecasters on the development tranche: all zero, TimesFM
(zero-shot, weather covariates), AsymODE (host W+Cin) and AsymODE + GCRK. The geography x weather pathway (hazard
features as a competing hazard) was removed by the PI on 2026-09-27: geography enters only through GCRK or a kernel
architecture in the hidden layer, never through hand-built hazard features. Learned models use five initializations,
averaged in forecast space, but only for a design with a single-seed gain over the host. Every wake-up in this phase:

1. **Look.** Check `runs/geo_weather_20260924/timesfm_v1D/` and the running job, if any. The paper queue
   `jobs_v1_paper.txt` is stopped. No GCRK seeds 1-4 run until a GCRK design beats the host at seed 0 (the bounded
   GCRK: +0.7%). The next design is a kernel extension (spatial coupling of the response state), under discussion
   with the PI. It runs only once its design is written in `aris/IDEAS.md` and agreed.
2. **Evaluate.** Run `paper_v1/evaluate_paper.py` whenever a model's set of seeds grows or TimesFM finishes. It
   writes `results/v1/paper_tables.json`: pooled MAE and RMSE at +1, +6, +24 and +48 hours, design-weighted and
   unweighted, by regime, and RMSE against AsymODE with family-cluster intervals.
3. **Figures and text.** Whenever a model's set of seeds grows, rerun, in this order and at nice 15:
   `paper_v1/counterfactual_v1.py` (GCRK counterfactuals, then `summary`), `paper_v1/figures_v1.py`
   (Figures 1-4), `paper_v1/evaluate_paper.py`. They write `paper_v1/generated/*.tex`, and `main.tex` quotes numbers
   only through those macros, so nothing is copied by hand. `describe_v1.py` and `twins_v1.py` depend on the data
   only and are rerun only if the panel changes. Descriptive analyses use the development tranche only. Text that
   states a direction (abstract, Sections 5.2-5.4, conclusion) is reread against the new numbers each time.
4. **Record, push and report**, as above.

## Kernel comparison (from 2026-09-27 15:45; PI: kernels first, the paper waits)

The PI asked to compare plain AsymODE, AsymODE + GCRK (temporal kernel) and AsymODE + spatio-temporal GCRK (I14)
before any more writing, and only then to move on (a recovery-side kernel is the next idea). Every wake-up:

1. **Look.** `jobs_v1_stgcrk_s0.txt` (STGCRK+Cin, seed 0, five event folds). The PI approved a speed-up on
   2026-09-27 19:05: fold 1 continues at nice 20 (it cannot be reniced upwards), folds 2-5 run two at a time at nice 10
   (`jobs_v1_stgcrk_s0_f2345.txt`, `logs/queue_v1_stgcrk_f2345.log`), so at most three training processes. If the
   machine keeps no idle CPU for the PI's own work, go back to two. TimesFM is done.
2. **Evaluate.** When the five folds of `v1_stgcrk_s0` are done, run `compare_kernels_v1.py --seed 0` (paired seed 0:
   pooled MAE and RMSE at +1/+6/+24/+48 h, relative RMSE with family-cluster intervals overall, by regime, by initial
   state and without the three worst systems, large-outage peaks, the learned coupling per fold) and
   `evaluate_v1.py --arm v1_stgcrk_s0 --host v1_host_s0` and `--host v1_gcrk_s0` for the regime-balanced numbers.
3. **Decide by rule** (single seed first). The spatio-temporal kernel is a candidate if it beats the host and the
   temporal GCRK at seed 0 (regime-balanced and pooled +1 h). Only then: seeds 1-2 of it, of GCRK and of the host, and
   the spatial null (neighbours replaced at matched distance). Otherwise report and discuss the next kernel with the PI.
4. **Record, push and report**, as in sections 5 and 6. When TimesFM finishes, rerun `paper_v1/evaluate_paper.py`
   so its tables are current, but do not edit the manuscript.

## Frozen diagnostic phase (2026-09-28; historical, superseded below)

No training job is running or queued by this review. I15-I17 are proposed only; a scheduler must not treat the
proposal document as authorization. The next dependent step is PI discussion of
`notes/KERNEL_PROPOSALS_20260928.md`, preferably I15's minimal recovery-history kernel without stock feedback.
Do not regenerate or edit `paper_v1/`; do not build, read or evaluate C.

Completed: seed-0 comparison reproduced; ten kernel checkpoints rebuilt without graph edges lost; trajectory
onset/magnitude/recovery diagnostics, latent-code and readout diagnostics, primary literature notes, and candidate
stability/initialization/screening plans. See `notes/KERNEL_REVIEW_20260928.md` for the decision-facing summary.

Metric clarification: the post-Stage-0 registered primary is the tropical/winter balanced MSE, not the all-five
mean. Re-reading ST on that primary gives -1.28% [-3.27,+1.72] vs host and -2.38% [-4.84,-0.29] vs GCRK;
all-five vs host remains +0.30%, heavy rain +2.60%. Preserve the earlier no-more-seeds decision; do not silently
change the historical screen or reopen its queue. Any prospective screen clarification is discussed and recorded
before training. Frozen geography constants (standardized g=0, exact z=0, fitted mean z) are not matched trained NULLs.

Once a design is approved: implement only that design, adapt optional kernel metadata handling in
`compare_kernels_v1.py` before passing it a recovery-only arm, preserve score definitions, and run correctness
checks for the new recurrence. Seed 0 and five existing event folds come first; more seeds and information NULLs
only for survivors. Respect the PI's existing maximum of three low-priority training processes and reduce load
when necessary; never touch another session's processes.

## I18 execution phase (2026-09-28; completed, retained as the execution record)

The PI authorized this chat to run experiments and asked to prioritize the geographic kernel.
I18 is the first screen: fixed fit-only geography RMS normalization in the existing damage-side GCRK;
I15-I17 remain deferred. The exact design and gates are frozen in
`notes/I18_GEO_RMS_SCREEN_20260928.md`. Run label `v1_gcrk_georms_s0`, seed 0, five event folds, 900 steps.

`run_i18_screen.py --workers 1 --threads 2` is a finite queue: starts at nice >=15, owns only its own
children, validates completed metadata/exports, and evaluates host and original GCRK after all folds.
Its ignored run-folder `WORKERS` file may be atomically changed to 2 only after memory/CPU headroom
is verified; reducing it to 1 stops future overlap, not another process. Source hashes and a runner
lock prevent mixed implementations and duplicate queues. No automatic seed or NULL expansion.

Follow the current run's status and own logs; report only completed folds, failures, resource issues or
final results. Never relaunch a live runner; inspect an interrupted partial fold without overwriting it.
When all folds finish, verify the four `screen_i18_s0_{headline,all5}_vs_{host,gcrk}.json` reports,
`kernels_georms_s0.json`, and `screen_i18_s0_verdict.json`; record, scan, commit and push both research
branches. A candidate is provisional only; a failed screen gets no extra seeds. C and the paper stay untouched.

Launched at 14:55 ET on 2026-09-28 from registered code/design commit 395350d, one worker and two threads.
Correctness checks: 27 passed, 1 skipped (the external live-reference source is unavailable).
This chat has a 15-minute follow-up monitor; it follows only this label and pauses after final reporting
or a failure requiring PI input. Ordinary progress without a new completed fold is quiet.

PI update, 15:36 ET: parallel execution explicitly authorized. The live queue now uses
two workers, each with two threads and nice 15. This supersedes the initial one-worker
resource preference; the registered model, steps, folds and source hashes stay fixed.
Monitor resource pressure; reduce future overlap if a material resource issue appears.

PI subsequently requested the next geographic-kernel design while I18 runs. I19 is
written in `notes/I19_GEO_READIN_PROPOSAL_20260928.md` as a proposal only; no new arm
is implemented or queued. Finish and report I18, then pause its monitor as specified;
the proposal must not cause automatic next-version training.

## I18 closed; data-first analysis (2026-09-28; current operating state)

The finite I18 queue completed all five 900-step folds and scored both comparators. Independent single-thread
`validate_exports()` and source-manifest/coverage checks passed: 8,457 D county-events, 81 systems, 80 families,
each held-out unit once, the same indices as host/GCRK, finite exports and bounded predictions. See
`results/v1/i18_s0_export_validation.json` and `notes/I18_GEO_RMS_RESULTS_20260928.md`.

Frozen verdict: `screen_failed_no_seed_expansion`. Tropical/winter balanced MSE is −2.358% [95% −5.081,+1.218]
vs host and −3.445% [−7.744,−0.071] vs original GCRK; all-five is −0.366% [−4.208,+5.544] vs host.
Pooled +1 RMSE is −0.484% [−1.530,+0.639] vs host. The sole failed gate is synoptic wind +2.0745% vs host,
above the prospectively frozen +2% maximum; do not relax it after seeing this result. No extra I18 seed or NULL.
The I18 follow-up monitor is paused after result delivery and independent verification; the empty, evaluated queue must not be relaunched.

The PI's latest priority is to examine weather sequence and overlap interacting with geography in the data
before implementing the next architecture. I19 is parked; I15-I17 remain deferred. Use D only and preserve
family grouping, repeated-county identity, outcome-blind weather definitions and separate severity/order/
alignment controls. Zero-outage cases remain in scope. This exploratory analysis does not authorize new
neural training or paper edits; any later model design, data-source change and screening plan must be explicit
before its first run. Earlier three-fold I18 numbers remain historical observations only.

D01 is the bounded statistical analysis in `notes/D01_DATA_FIRST_PROTOCOL_20260928.md` and
`analyze_data_relationships_v1.py`: six nested ridge probes, three fixed historical specifications,
existing event folds and same-state geographic correspondence controls. Fitting these diagnostic regressions
is within the PI's data-analysis request; it does not restart a neural experiment or change its evaluation.

D01 is now complete, including all three specifications and independent output/arithmetic verification.
The primary joint geographic modulation fails its exploratory candidate criteria in every specification;
see `notes/D01_DATA_FIRST_RESULTS_20260928.md`. Poor/uncertain performance against the descriptive persistence
reference limits interpretation: do not promote this negative probe into a no-physical-interaction claim.
The finite diagnostic process has exited. No next-version neural run, extra I18 seed or trained neural NULL
is authorized by these results; I18's monitor remains paused and this chat remains open.

## D02 county-structure exploration (PI correction; current priority)

The PI explicitly asked to broaden the data investigation across county structure and direction instead of
narrowing on the basis of an overall statistic. Follow `notes/D02_COUNTY_COMPLEXITY_SCOPE_20260928.md`:
current-event-outcome-blind county types, weather-anchored dynamics, and within-county/within-system/two-way
association decompositions. Global MSE is not an exclusion gate for this atlas. Preserve negative, weak and
unsupported cells, and distinguish sign cancellation, lag/threshold mixing and event composition.
This authorizes the described statistical exploration, not a new neural arm. D only; C and paper unchanged.

D02 is complete: the outcome-blind county structure/continuous map, six weather-anchor dynamics,
and all 56-driver association decompositions are saved with support and uncertainty. See
`notes/D02_COUNTY_COMPLEXITY_RESULTS_20260928.md`. Six types are coarse navigation partitions
(mean silhouette 0.173), not six natural mechanisms. The atlas shows timing/composition differences,
support-dependent directions and order associations; it does not establish causal geographic mechanisms,
robust cancellation or a model-selection winner. In particular, an overall statistic remains no exclusion
gate. Sparse numerical projections and absorbed controls passed independent weighted-dummy checks;
final source hashes match and finite statistics/explicit missing cells are retained. Finite statistical
processes have exited; no neural queue is pending, I18 monitoring remains paused, chat stays open.

## D03 input structure and impact-kernel design (2026-09-28; current operating state)

D03 is complete: public-D weather/geography audit on the same 50,302 supported windows, no outage-value
reads, one nice-15/two-thread process, no neural training. Pure geo40 and geo40+context6 spectra are separate.
Weather means retain 69.0%/73.1% of standardized 24-hour path variation under design/group-equal weights;
eight fixed time coefficients retain 97.7%/98.3%. These are representation facts, not response evidence or
selected kernel ranks. See `notes/D03_HIGH_DIM_STRUCTURE_RESULTS_20260928.md`.

The PI explicitly fixed the durable research chain for future analysis, design and manuscripts:
**weather -> complex geographic modulation -> impact formation/interaction/accumulation -> outage.**
The kernel models weather-to-latent-impact transformation inside the single damage MLP; the existing
host and outage dynamics, including recovery and context, provide the observation link. Impact is not
measured physical damage. Do not replace this motivation with generic county-feature regression or claim
that stock identifies the latent process. This clarification is also recorded at the top of `program.md`.

`notes/KERNEL_HIGH_DIM_DATA_MOTIVATION_20260928.md` synthesizes completed evidence and primary literature.
Next statistical work should jointly estimate nonlinear weather-lag-geography response surfaces, geographic
modulation of simultaneous and ordered cross-/same-weather histories, full support and clustered uncertainty.
Continuous response geometry and a bounded multiscale hidden impact kernel are proposed, not fitted or
registered neural arms. Retain broad exploration and weak/reversed cells; no overall-MSE exclusion gate,
fixed six-expert interpretation, automatic seed expansion, C access or paper edits. I18 monitoring stays
paused and no new training queue is pending.

## D04 execution (PI authorized 2026-09-28)

The PI directed this chat to execute the agreed analysis. Follow the frozen statistical scope in
`notes/D04_CONDITIONAL_IMPACT_RESPONSE_SCOPE_20260928.md`: strict-past 48-hour weather, continuous geography,
joint low-rank main/composite modulation, original event folds and all-hour held-out evaluation. This is
conditional net-stock-change analysis, not a new neural outage arm or identification of physical impact.
The finite `run_d04_response.py` queue uses one process, nice 15 and two threads; preserve partial folds and
verify its lock/process and source manifest before any restart. Selection stays inside training events.
Complete scores, numerical/support/response diagnostics and the registered history-control sensitivity;
retain local weak/negative findings without an overall-MSE exclusion gate. I18 monitoring remains paused.

## D04 completed (2026-09-28; current operating state)

The 100-fit primary queue and registered 16-fit history sensitivity are complete; their processes exited.
All five immutable fold exports, original source hashes, masks, heldout coverage and finite predictions were
verified. Keep all partial/completed directories and source registration unchanged; no restart or extra fits.
See `notes/D04_CONDITIONAL_IMPACT_RESPONSE_RESULTS_20260928.md` and the five compact D04 result JSONs.

Geographic modulation does not improve this statistical probe: C/A all-five MSE +0.935% [95% +0.419,+1.638],
D/B +3.065% [+1.846,+5.313]. Local positive and negative response diagnostics, all weather/county types,
component decomposition, history sensitivity and support concentration remain reported; global MSE is not
an exclusion gate. A larger predicted increment maximum also raises false-peak rates and does not establish
correct timing. Proxy support is not geographic exchangeability or a shared weather-reference distribution.

Numerical limitation: all 116 fits are finite, but none met the registered gradient stopping criterion.
Do not interpret the fitted operator, selected rank or local signs as stable physical structure, nor the
negative probe as evidence that complex weather/geography interactions do not exist. Before architecture
inference, prioritize a separately specified numerical/observation-time audit, including initialization
stability and the distinction between impact onset and net-stock decline. No new neural arm, extra seed,
NULL or numerical refit is queued. I18 monitoring remains paused; C and paper stay untouched, chat stays open.

## D05 large-outage forensics (2026-09-28; current operating state)

The PI resumed computation and explicitly requested data-level explanations for missed large outages.
Follow `notes/D05_LARGE_OUTAGE_FORENSICS_SCOPE_20260928.md`: separate rapid net-stock jumps from high
stock, audit observation/weather timing, retain full weather order/overlap coordinates and continuous
county geography, and compare same-event/time counties without outcome-selected quiet controls.
This is a bounded exploratory description on examined D, not causal verification or a new neural arm.
One nice-15 process/two threads; preserve partial outputs. The finite runner is `run_d05_forensics.py`.
I18 monitoring stays paused, frozen D04 remains unchanged, and no sealed C or manuscript access is allowed.

## D05 completed (2026-09-28; current operating state)

The finite audit registered at 665218d is complete; its computation and report processes exited.
All D-only identities, observation timestamps/masks/denominators and original three-model five-fold exports
passed validation. Source/input/result hashes match; all seven compact analysis JSONs and three PNG/PDF
figure sets are retained. See `notes/D05_LARGE_OUTAGE_FORENSICS_RESULTS_20260928.md`.

J >=1pp maximum net increase has 2,963 county-events; S >=10% stock has 726, overlap 723. Georms S peak
amplitude ratios remain 8.936% unweighted / 1.225% design-weighted even using each prediction's full-window
maximum on common observed support. Weather timing alone cannot account for this amplitude deficit.
Weather trajectories and prior accumulation differ around the two anchors; matching reveals conditional
geography associations, but joint/order weather balance, observation processes and causal mechanisms remain
unidentified. State-adjusted S matching covers only 121 cases / 28.89% design weight; no representative
severe-case mechanism claim is allowed. Retain all 3,408 weather coordinates and 1,200 geographic comparisons.

Next kernel hypotheses must separate short shock formation from accumulated latent state, preserve all
continuous geography, and test ordered/compound information beyond main weather effects. No new neural
arm, seed, NULL, data reconstruction or refit is running. Overall MSE remains no exclusion gate. I18 stays
paused; sealed C and manuscripts remain untouched; preserve this chat and all artifacts.
