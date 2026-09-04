# Locked CC plan V2: competition-informed adjudication without claim reversal

**Authority:** this file supersedes `04_LOCKED_RESCUE_EXPERIMENT.md` and `07_CC_NEXT_STEP.md` for all new code and experiments on `open-audit-20260904`.

**Immediate scope:** construct the clean paper path and run the implementation pilot only. Do not launch the full five-fold campaign until Shuai reviews the pilot report.

## 1. Single scientific question

The experiment asks:

> After preserving the empirically successful process-specific asymmetry, correcting the one-flow optimization path, training both classes adequately, and evaluating a broader storm cohort, does retaining two simultaneous nonnegative conditional-mean components improve held-out transitions and open-loop storm forecasts relative to collapsing the same process proposals to one signed flow?

This is an adjudication of the current evidence, not a search for a favorable subset or a leaderboard model.

## 2. Claims held fixed before the run

The following hypotheses are retained rather than withdrawn:

- **H-dynamics:** the state-scaled two-process update is a useful inductive bias for outage trajectories;
- **H-separation:** retaining two nonnegative components can outperform a one-flow collapse when state dispersion and process coactivity make the richer class valuable;
- **H-representation:** interruption and restoration require different representations, with a richer and more temporally structured interruption side and a simpler recovery side;
- **H-rollout:** a transition-level gain can survive a 24-hour open-loop rollout when the local errors do not cancel or amplify adversely.

The run may support, qualify, or fail to support these hypotheses. Existing positive results remain archived as reproduced legacy evidence until this run is complete.

## 3. Data cohort and event definition

### 3.1 Development cohort

Use every panel in:

```text
configs/panel_manifest_g3-all-26.json
```

The `g2-convective-11` cohort remains a historical comparison subset. Do not tune the new method on `g2` and then describe `g2` as untouched confirmation.

The target population must be stated narrowly:

> large-footprint US storm-day panels with adequate EAGLE-I coverage, a county denominator, and matched hourly weather.

### 3.2 Event table before fitting

Before any neural fit, create:

```text
analysis/gpt_rescue_20260904/cc_v2/event_design_table.csv
analysis/gpt_rescue_20260904/cc_v2/event_design_table.md
```

One row per panel, with no model result columns:

```text
event id
family
year
number of counties
observation coverage
NOAA interval start/end
number of valid hourly origins
zero-origin share
0<y0<=0.01 share
y0>0.01 share
future-onset share (diagnostic only)
median / p90 outage fraction
weather-footprint summaries
customer-denominator summaries
```

State summaries describe the task; they may not be used to delete events. Fold construction may use family, year, county footprint, coverage, and weather/geographic summaries, but not outage severity, old model residuals, or prior gains.

### 3.3 Fixed event-centered origins

The current full-window origin grid overweights quiet pre-storm and late-recovery periods. Replace it in the new main task with an outcome-blind, equal-count event-centered rule.

For each panel, construct three anchor origins on the existing hourly grid:

1. six hours before the first NOAA event time represented in the panel;
2. the midpoint of the NOAA event interval;
3. six hours after the last NOAA event time.

Round each anchor to the nearest valid origin having at least 24 hours of past context and 24 hours of future targets. Remove duplicate anchors but do not replace them using outage outcomes. Each event-origin block receives equal weight. Preserve the legacy full-window grid as a secondary evaluation computed from the selected models, not as a second tuning task.

Do not delete `y0=0` rows. The full sample and the state strata are both required.

### 3.4 Outer and inner splits

Create one fixed five-fold event assignment balanced greedily on exogenous descriptors only. Each panel appears in exactly one test fold. Save the map and digest before training.

Within every source event, hash counties into fixed 80% fit and 20% validation groups. A county's rows and origins stay on one side. Checkpoint selection uses the equal-event mean validation objective over **all** source events; no single validation storm may control stopping.

## 4. Clean paper-only model path

Create a new module; do not extend the historical variant zoo:

```text
src/asymode_paper/
    features.py
    asymmetric_flows.py
    initialization.py
    trainer.py
```

The old competition and open-data modules remain untouched for reproduction.

### 4.1 Preserve semantic asymmetry, not challenge-specific dimensions

The exact `59/17/43` dimensions belong to the competition dataset. On the open data, reproduce the **roles** and document the exact available channels in:

```text
analysis/gpt_rescue_20260904/cc_v2/FEATURE_MAP.md
```

The blocks are:

- `x_u`: rich interruption-magnitude context—current/forecast weather, causal accumulated-hazard features, fixed county covariates, and storm-footprint summaries;
- `x_occ`: a narrower, genuinely different occurrence block—instantaneous hazard plus a small static block; it must not be the hidden representation of the magnitude network;
- `x_r`: recovery context—weather, county covariates, pre-origin outage-history summaries, and legal neighbor summaries; no current simulated state and no clock channel.

Every feature must have provenance, availability time, normalization source, and missingness treatment. Do not silently fabricate an unavailable competition feature.

### 4.2 Interruption proposal

Use the pared-down competition structure and no additional mechanism:

\[
z^U_t=\tfrac12\{f_{U,1}(x^U_t)+f_{U,2}(x^U_t)\},
\]

where both `f_U` networks are two-layer width-32 MLPs.

Use the learned first-order hold already present in the competition genealogy:

\[
\ell^U_t=q_t\ell^U_{t-1}+(1-q_t)z^U_t,
\qquad q_t=\operatorname{sigmoid}(g_U(x^U_t)).
\]

Use a distinct occurrence gate and independent background path:

\[
g_t=\operatorname{sigmoid}(g_{\mathrm{occ}}(x^{\mathrm{occ}}_t)),
\]

\[
\widetilde U_t
=g_t C_U\operatorname{sigmoid}(\ell^U_t)
+C_{\mathrm{bkg}}\operatorname{sigmoid}(g_{\mathrm{bkg}}(x^U_t)).
\]

The gate and background path are separate modules. They may not be folded into the magnitude trunk. Use the current paper's hourly rate scale after checking units; do not transplant a challenge cap merely because its numeric value appeared in the competition code. Choose caps once from training-transition quantiles and then freeze them for every fold and arm. Require `C_U+C_bkg+C_R<=1` so the primary state update preserves `[0,1]` without relying on a clamp.

### 4.3 Restoration proposal

Use one recovery GLM:

\[
\widetilde R_t=C_R\operatorname{sigmoid}(w_R^\top x^R_t+b_R).
\]

Recompute it every eight forecast hours and hold it fixed between recomputations. This schedule is fixed, not tuned. The recovery network does not read the current simulated state; state dependence enters only through `-R_tY_t`.

### 4.4 Main structural arms: output-level nesting

Both arms compute the same proposals `U_tilde` and `R_tilde` using the same process-specific architecture.

Define

\[
s_t=\widetilde U_t-\widetilde R_t,
\qquad c_t=\min\{\widetilde U_t,\widetilde R_t\}.
\]

The two reported arms are:

\[
\begin{aligned}
\texttt{asym_two_flow}:\quad
&U_t=\widetilde U_t,\quad R_t=\widetilde R_t,\\
\texttt{asym_one_flow}:\quad
&U_t=[s_t]_+,\quad R_t=[-s_t]_+.
\end{aligned}
\]

Thus the one-flow arm removes only `c_t`. It retains the same interruption ensemble, occurrence gate, background path, recovery GLM, process-specific inputs, and parameter budget. This is the primary comparison.

Do not use the earlier shared-GRU `MODEL_V3_REFERENCE.py` as the final model. It remains a mathematical prototype only.

## 5. Class-correct initialization and one-flow branch safeguard

Build unique adjacent hourly transitions from the fit counties; overlapping forecast windows must not duplicate the same transition during Stage A.

Fit the exact bounded constant two-flow class:

\[
(\widehat U_0,\widehat R_0)
=\arg\min_{0\le U,R\le C}
\sum_i w_i\{\Delta Y_i-U(1-Y_i)+RY_i\}^2.
\]

For the one-flow class, fit both bounded rays:

\[
\widehat a_0=\arg\min_{0\le a\le C}
\sum_iw_i\{\Delta Y_i-a(1-Y_i)\}^2,
\]

\[
\widehat b_0=\arg\min_{0\le b\le C}
\sum_iw_i\{\Delta Y_i+bY_i\}^2.
\]

A reported one-flow fit is trained from **both** fixed initial branches—interruption start and restoration start—and the final start is selected using source-event validation only. This is an optimization safeguard, not a third scientific arm. Test data may not choose the branch.

The old `u0-r0` initialization is prohibited. The calibrated update-0 model is always a checkpoint candidate.

## 6. Training schedule

The competition evidence makes inadequate optimization an unacceptable explanation. Define budget in gradient updates, not epochs.

### Stage A: theory-aligned transition pretraining

```text
1,600 optimizer updates
validation every 200 updates
```

Use teacher forcing on unique observed transitions. Sample source events uniformly, then sample a minibatch within the event. Within a minibatch, sample half from `Y_t<=0.01` and half from `Y_t>0.01` when both pools exist; fall back transparently when an event lacks one pool. Both arms receive the identical sampler.

### Stage B: open-loop rollout fine-tuning

```text
3,200 optimizer updates
validation every 200 updates
minimum 1,600 updates before stopping
patience 6 validation checks after the minimum
```

Fine-tune on 24-hour recursive MSE over the three event-centered origins. Events and event-origin blocks are equally weighted. Use the same optimizer, learning rate, batch size, gradient clipping, normalization, and update budget for both arms.

Checkpoint candidates are update 0 and every validation checkpoint. Record gradient updates, examples processed, validation curves, gradient norms by module, wall time, and selected branch/start.

Do not perform a learning-rate, width, cap, loss-weight, gate, or history-length sweep. A pilot may reveal an implementation failure, but not trigger model search.

## 7. Baselines

The pilot and main table contain:

1. `asym_two_flow`;
2. `asym_one_flow`;
3. exact constant two-flow transition/rollout;
4. exact constant one-flow transition/rollout;
5. damped persistence;
6. all-zero;
7. same-information histogram gradient boosting at h+1 and h+24 in the full main run.

The constant two-flow baseline is the load-bearing check that the learned context contributes beyond the state equation alone.

Keep the historical `net_scaled` result in the ledger, but do not use it as the sole comparator in the new table.

## 8. Endpoints and what they prove

### 8.1 Transition endpoint

The theorem-aligned primary endpoint is equal-event teacher-forced one-step MSE:

\[
d_e^{\mathrm{step}}
=\operatorname{MSE}^{\mathrm{TF}}_{\mathrm{one},e}
-\operatorname{MSE}^{\mathrm{TF}}_{\mathrm{two},e}.
\]

Report it on:

- all observed transitions;
- `Y_t=0`;
- `0<Y_t<=0.01`;
- `Y_t>0.01`.

A positive result in `Y_t>0.01` is the minimum evidence for an interior concurrent-component interpretation. A gain confined to the boundary remains meaningful evidence for interruption/recovery directional decoupling, but is not labeled interior concurrency.

### 8.2 Forecast endpoint

The operational primary endpoint is equal-event 24-hour path MSE and h+24 MSE from the three event-centered origins:

\[
d_e^{24}
=\operatorname{MSE}^{24}_{\mathrm{one},e}
-\operatorname{MSE}^{24}_{\mathrm{two},e}.
\]

Report the same origin-state strata. h+48 and the legacy full-origin grid are secondary boundaries, computed without retuning.

### 8.3 Absolute-usefulness gate

Report each neural arm against its own update-0 constant model and against the fitted constant two-flow baseline. If the trained model does not improve on the constant model, do not interpret feature-specific rates even if the two-flow/one-flow difference is positive.

### 8.4 Statistical unit

Average optimization starts and neural seeds within event first. The event is the inferential unit. For the full run report every event effect, equal-event mean and median, event bootstrap interval, paired sign-flip test, positive-event count, leave-one-event influence, and between-seed spread.

Do not treat origins, counties, folds, starts, or seeds as independent storm replications.

## 9. Pilot only

After the event table is frozen, select three pilot events without using outage severity or prior model performance: one exogenous medoid from a convective group, one from winter, and one from tropical/wind, using standardized weather-footprint, coverage, and county-count descriptors.

Run:

```text
3 events x 1 model seed x
    [1 two-flow start + 2 fixed one-flow branch starts]
= 9 optimization jobs
```

These produce six reported event-model fits. Deterministic baselines add no neural fits.

The pilot passes implementation review only if:

- exact constant class fits are reproduced numerically;
- update 0 is scored and eligible;
- repeated runs with the same seed reproduce numerically;
- every module receives a finite gradient when its input regime is present;
- neither arm leaves `[0,1]` before any optional assertion clamp;
- both trained arms are compared with their own update-0 model;
- the report includes full, boundary, near-boundary, and interior strata;
- no event or metric is dropped because of its sign.

The pilot is not paper evidence and does not authorize a claim revision. Stop after producing the pilot report.

## 10. Main run, only after approval

The approved campaign is:

```text
5 outer folds x 3 model seeds x
    [1 two-flow start + 2 one-flow branch starts]
= 45 optimization jobs,
```

reported as 30 neural fold-seed model estimates, plus 10 HGB fits. No architecture sweep is authorized.

After the main report is frozen, at most one appendix ablation may be run: replace the process-specific capacity/input allocation by a symmetric host while retaining the same output-level collapse. This isolates the competition-derived representation asymmetry from the flow-separation result. It is not run before the main comparison.

## 11. Interpretation table

| New result | Defensible conclusion |
|---|---|
| Two-flow wins one-step and 24 h, including the interior stratum | the second conditional-mean component adds value beyond onset-only directional decoupling and survives rollout |
| Two-flow wins overall but not in the interior | separating onset and recovery directions helps this task; an interior concurrency claim remains open |
| One-step win, no rollout win | local representation value is lost through recursive propagation or forecasting misspecification |
| Rollout win, no one-step win | a forecast gain exists, but it is not attributed to the oracle-gap mechanism |
| Difference weak after repair | the legacy effect was not robust to the adjudicating design; report both without retroactively erasing the old result |
| Neither neural model beats the constant two-flow baseline | the process-specific features/training do not transfer; do not interpret learned rates |
| HGB wins absolute accuracy while two-flow beats one-flow | retain a structural model-selection contribution, not a state-of-the-art forecast claim |

## 12. Forbidden actions

Before the pilot review, do not:

- edit manuscript result macros or conclusions;
- relabel the earlier confirmatory result as refuted or withdrawn;
- delete zero rows;
- select events or origins using outage outcomes, residuals, or prior gains;
- add a GRU, Transformer, recovery burden, secondary-damage state, new gate, semi-parametric term, or family-specific model;
- change more than one architecture component after seeing pilot accuracy;
- average the legacy confirmatory result with the undertrained unified-v2 result;
- run the 45-job main campaign.

## 13. Required pilot deliverables

Write everything under:

```text
analysis/gpt_rescue_20260904/cc_v2/
```

Required files:

```text
FEATURE_MAP.md
event_design_table.csv
event_design_table.md
event_folds_v2.json
origin_rule_audit.md
MODEL_IMPLEMENTATION_AUDIT.md
pilot_results.json
pilot_event_effects.csv
pilot_training_diagnostics.csv
PILOT_REPORT.md
REPRODUCTION_COMMANDS.md
```

`PILOT_REPORT.md` must distinguish: preserved legacy evidence, newly confirmed implementation facts, pilot-only observations, and still-open scientific claims.
