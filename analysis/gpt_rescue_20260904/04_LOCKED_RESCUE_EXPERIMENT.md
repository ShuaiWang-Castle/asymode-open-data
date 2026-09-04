# Locked rescue experiment

## 1. One scientific question

The next experiment asks only:

> After correcting optimization and using an architecture in which the one-flow model is an exact submodel, does a learned concurrent-flow component improve held-out conditional transitions and 24-hour forecasts across heterogeneous storm events?

This is not a search for a leaderboard model. It is a direct test of the value of the second conditional-mean component.

## 2. Data cohort and folds

### Cohort

Use all 26 panels in `configs/panel_manifest_g3-all-26.json`. The eleven-event `g2` cohort is retained only for historical reproduction.

### Outer folds

Create five fixed event-level folds. The assignment may use only:

- event family;
- calendar year;
- county footprint;
- observation coverage;
- weather summaries and geographic footprint.

Do not use outage severity, model residuals, or prior gains. Each event appears in one test fold. The fold map is written once before training and carried by digest in every result file.

### Source-event validation

Within every source event, hash counties into fixed 80% training and 20% validation sets. All rows and forecast origins for one county remain on one side. The checkpoint criterion is the equal-event mean validation loss across all source events.

### Forecast origins

For the first rescue run, preserve the existing origin construction so that algorithmic changes are not mixed with a new task definition. Weight events equally during training and inference. In the same output, report origin-state strata and the future-onset diagnostic.

Before the final paper run, construct an outcome-blind storm-conditioned origin mask using NOAA event intervals and spatial footprints. If this mask materially changes the share of quiet origins, treat it as a data-definition revision and rerun only the selected one-flow/two-flow pair plus deterministic baselines. Do not tune the model on both origin definitions.

## 3. Model: one nested neural family

### Shared context encoder

Both arms use the same context:

- 24 hours of past hourly weather;
- the current and forecast-hour weather;
- UTC clock;
- a minimal fixed set of pre-event county covariates that are available for all eligible counties: log customer count, customer density, land area, RUCC, and EIA reliability/service-territory summaries with missingness indicators.

No county identity embedding is allowed. The current outage state is not fed to the context encoder; it enters only through the known multipliers in the transition equation.

Encode weather history with one small GRU:

```text
hidden size: 16
layers: 1
```

Concatenate the GRU state, current/future weather, clock, and county statics. Pass the result through a shared two-layer MLP of width 32.

### Nested flow heads

From the shared representation, compute a signed direction `s_t` and a nonnegative concurrency component `c_t`:

\[
s_t=C\tanh z_{s,t},
\qquad
c_t=(C-|s_t|)\sigma(z_{c,t}),
\]

with `C=0.25`. Set

\[
U_t=[s_t]_+ + c_t,
\qquad
R_t=[-s_t]_+ + c_t.
\]

The transition is

\[
Y_{t+1}=Y_t+U_t(1-Y_t)-R_tY_t.
\]

Because `U_t,R_t≤C` and `U_t+R_t≤2C=0.5`, the map preserves `[0,1]`; no numerical state clamp is used in the primary implementation.

### Arms

1. `nested_two_flow`: learns both `s_t` and `c_t`;
2. `nested_one_flow`: uses the identical encoder and signed head but fixes `c_t=0`.

The second arm is an exact submodel of the first. The difference is the availability of the concurrent component, not independent feature extractors, different network widths, or a different training loop.

A parameter-budget control is allowed only in the appendix: give the one-flow model a second signed residual head with the same scalar-head parameter count. It must not replace the exact nested comparison as the primary test.

## 4. Initialization

Fit each class's exact constant model on training transitions.

For two flows, solve

\[
\min_{U,R\ge0}\sum_i
\{\Delta Y_i-U(1-Y_i)+RY_i\}^2.
\]

For one flow, solve the two rays

\[
\min_{a\ge0}\sum_i\{\Delta Y_i-a(1-Y_i)\}^2,
\]

and

\[
\min_{b\ge0}\sum_i\{\Delta Y_i+bY_i\}^2,
\]

and select the lower-loss branch. Initialize the output biases to these class-specific constant optima and initialize output weights at zero. The calibrated initialization is a checkpoint candidate at update 0.

The old `u0-r0` initialization is prohibited.

## 5. Training objective and budget

### Stage A: transition pretraining

Build unique observed hourly transitions from panel trajectories; do not duplicate the same transition through overlapping forecast-origin windows. Train with equal-event sampling under teacher forcing:

\[
L_{\mathrm{step}}
=\frac1{|\mathcal E|}\sum_e
\frac{\sum_{i\in e}M_i
(\widehat Y_{i,t+1}^{\mathrm{TF}}-Y_{i,t+1})^2}
{\sum_{i\in e}M_i}.
\]

Fixed budget:

```text
2,000 optimizer updates
validation every 250 updates
```

### Stage B: rollout fine-tuning

Fine-tune the same checkpoint on 24-hour recursive rollout MSE with equal-event minibatch sampling:

\[
L_{24}
=\frac1{|\mathcal E|}\sum_e
\frac{\sum_{i\in e,t\le24}M_{it}
(\widehat Y_{it}-Y_{it})^2}
{\sum_{i\in e,t\le24}M_{it}}.
\]

Fixed budget:

```text
3,000 optimizer updates
validation every 250 updates
minimum 1,000 fine-tuning updates before early stopping
patience 8 validation checks
```

All arms use the same optimizer, learning rate, batching, update count, gradient clipping, and checkpoint rule. Report optimizer updates and examples processed; “epoch” is not an admissible budget unit.

## 6. Baselines

One main table contains:

1. `nested_two_flow`;
2. `nested_one_flow`;
3. a constrained no-weather affine transition model;
4. histogram gradient boosting with the same available information, fitted directly for h+1 and h+24;
5. damped persistence;
6. all-zero.

The no-weather affine model is load-bearing. If neither neural model beats it on held-out one-step MSE, the paper cannot interpret the learned weather-dependent components.

## 7. Endpoints

### Primary theorem-aligned endpoint

Equal-event teacher-forced one-step MSE on held-out events:

\[
d_e^{\mathrm{step}}
=\operatorname{MSE}_{1,e}^{\mathrm{TF}}
-\operatorname{MSE}_{2,e}^{\mathrm{TF}}.
\]

Positive means the second component helps.

### Primary forecasting endpoint

Equal-event 24-hour path MSE and h+24 MSE from the same fitted models:

\[
d_e^{24}
=\operatorname{MSE}_{1,e}^{24}
-\operatorname{MSE}_{2,e}^{24}.
\]

h+48 is a secondary boundary diagnostic and does not trigger model selection.

### Prespecified strata from the same predictions

Report, without refitting:

```text
y0 = 0
0 < y0 <= 0.01
y0 > 0.01
future-onset: y0 <= 0.01 and max(Y_1:24) > 0.02
```

The `y0>0.01` result is the minimum evidence needed for an interior concurrent-flow interpretation. Future-onset is a diagnostic because it uses the future target to define the subset.

## 8. Statistical inference

Neural seeds are averaged within each event. The event is the inferential unit.

For each primary difference, report:

- all 26 event effects;
- equal-event mean and median;
- event bootstrap 95% interval;
- exact or Monte Carlo paired sign-flip test over events;
- positive-event count;
- leave-one-event influence;
- between-seed optimization spread.

Do not treat rows, counties, folds, or seeds as independent storm replications.

## 9. Fit count

### Implementation pilot, not paper evidence

Choose one outcome-blind event from each of three broad families after the event table is built. Run one seed for the two neural arms:

```text
3 events × 2 arms × 1 seed = 6 pilot fits
```

The pilot passes only if:

- the class-optimal initialization is correctly reproduced;
- epoch/update 0 is scored;
- neither arm is boundary-degenerate solely because of initialization;
- each arm improves its own update-0 validation score;
- two repeated runs with the same seed are bitwise or numerically reproducible.

### Main run

```text
5 outer folds × 3 seeds × 2 neural arms = 30 neural fits
5 folds × 2 horizons = 10 HGB fits
```

The deterministic affine, persistence, and zero baselines are negligible. No architecture or loss-weight sweep is authorized.

## 10. Decision table

| Outcome | Interpretation |
|---|---|
| Two-flow wins one-step and 24h, including `y0>0.01` | the second conditional-mean component adds predictive value and survives rollout |
| Two-flow wins only at zero/near-zero states | onset/recovery directional decoupling helps; concurrent interior-flow claim is unsupported |
| One-step win, no 24h win | local representation gain is lost through recursive propagation or model misspecification |
| 24h win, no one-step win | a forecast benefit exists, but it cannot be attributed to the oracle-gap theorem |
| Neither neural arm beats the affine baseline | the weather encoder/data are not informative enough; do not interpret learned rates |
| HGB wins absolute accuracy while two-flow beats one-flow | retain a structural model-selection contribution, not a state-of-the-art forecasting claim |

## 11. Only one post-main ablation

After the main report is frozen, remove the 24-hour weather-history GRU while keeping the nested heads and all other settings fixed. This tests the single algorithmic diagnosis that onset timing requires accumulated weather context.

No additional gate, memory state, secondary-damage head, family-specific network, semi-parametric component, or hyperparameter sweep is authorized in this rescue stage.
