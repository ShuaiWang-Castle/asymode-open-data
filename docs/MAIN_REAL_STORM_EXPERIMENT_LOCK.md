# Main real-storm experiment lock

## One empirical question

> Does allowing simultaneous nonnegative damage and restoration flows reduce held-out one-step transition error, and does that advantage survive recursive prediction to 24 hours on a held-out storm event?

This is the only real-data experiment authorized before the main table is frozen.

## What the experiment can and cannot establish

The experiment tests the predictive relevance of the one-flow/two-flow theory. The theorem itself is mathematical and does not need empirical proof. The experiment does **not** identify causal physical damage or crew-restoration rates, and the teacher-forced contrast is not a plug-in estimate of the population oracle gap `G(x)`.

A forecasting advantage is attributed to flow separation only when:

1. the two-flow NN improves held-out teacher-forced one-step MSE over the parameter-matched one-flow NN; and
2. the same two-flow NN improves held-out h+24 rollout MSE.

The first condition tests whether the structural distinction is visible at the conditional-transition level. The second tests whether it survives neural estimation and the exact rollout-error recursion.

## Dataset and forecast task

- Cohort: every event in `configs/panel_manifest_g2-convective-11.json`.
- No event may be removed, reweighted, or relabeled after seeing performance.
- This cohort informed earlier exploratory work; the run is a locked re-analysis, not a pristine external confirmation.
- Forecast task: the first 24 hours after every admissible rolling origin.
- Primary forecast endpoint: h+24 MSE.
- Structural endpoint: teacher-forced one-step MSE over transitions 0->1 through 23->24.
- Secondary diagnostics: h+6 MSE and complete 24-hour path MSE.
- Event is the inferential unit. Seed measures optimization variability only.
- `2021-06-21` may be used only as an implementation smoke test; it never replaces the full eleven-event evaluation.

## Outer and source-validation design

For each outer fold:

1. hold out one complete event as test;
2. use the remaining ten events as source environments;
3. within each source event, split counties 80/20 by the deterministic SHA256 rule in `configs/source_validation_g2.json`;
4. keep all origins from one county on the same side;
5. estimate normalization and initialization from source-training rows only;
6. use the ten source-validation subsets only for checkpoint selection;
7. average training and validation losses equally over source events.

The test event is never used for preprocessing, initialization, training, validation, or model selection.

## Neural models

Exactly two neural classes are fitted.

### Two-flow NN

```text
D_theta(x)(1-y) - R_theta(x)y
```

Separate width-32, two-hidden-layer, nonnegative bounded rate networks with caps 0.25/0.25.

### One-flow NN

Parameter-matched `net_scaled` with one width-48 signed network. Positive output acts on `(1-y)` and negative output on `y`. It can switch direction across `x`, but cannot keep damage and restoration simultaneously positive at the same `x`.

Inputs, state scaling, normalization, optimizer, batch budget, stopping rule, seeds, and scored rows are identical. Each arm is initialized by the same flow-matching principle expressed in that arm's own parameterization; the signed arm is not initialized by subtracting rates calibrated for the susceptible-flow model.

## Fixed objective

Both neural models use event-balanced 24-hour rollout MSE only:

```text
L_train = mean_over_source_events L_rollout,event(steps 1:24)
```

A source event is sampled uniformly before drawing a minibatch within it. Validation is the equal-event mean of the same rollout loss.

Teacher-forced one-step loss is an evaluation endpoint, not a training term. No mixed loss, loss-weight sweep, or per-horizon neural refit is allowed in the main experiment.

Shared budget:

```text
Adam, lr=3e-3
batch size=512
epoch cap=60
patience=12
seeds=0,1,2
fixed calibrated initialization rule
```

## Context baselines

- `hgb_same_information`: direct histogram gradient boosting at h+6 and h+24 using the origin state and the same weather/clock path available by that horizon.
- `damped_persistence`: one source-fitted decay coefficient at h+6 and h+24.

HGB and persistence establish practical forecasting context. Only two-flow versus `net_scaled` tests the structural theory.

## Fit budget

- Neural: 2 models x 11 held-out events x 3 seeds = **66 fits**.
- HGB: 11 held-out events x 2 horizons = **22 fits**.
- Damped persistence: negligible.
- Total substantive fits: **88**.

No other fit begins before the main report is archived.

## Held-out endpoints

For each fitted neural model and test event:

- recursively roll from each origin to 24 hours;
- separately compute teacher-forced one-step predictions using the observed current state at each step;
- score a transition only when both the current and next state are observed;
- average the three neural seeds within event before inference.

Define

```text
d_step[e] = MSE_TF_one_flow[e] - MSE_TF_two_flow[e]
d_24[e]   = MSE_h24_one_flow[e] - MSE_h24_two_flow[e]
```

Positive values favor two flows.

## Hypotheses

### H1: transition-level structural signal

```text
mean_event d_step[e] > 0
```

This is the closest held-out predictive counterpart to the oracle gap. It is not an estimator of `G(x)` because the current state is noisy, the panel design is dynamic, and the neural classes may not attain their population oracles.

### H2: 24-hour forecast usefulness

```text
mean_event d_24[e] > 0
```

This asks whether the transition-level benefit survives finite-sample estimation and rollout error propagation.

The joint flow-separation claim is an intersection-union claim: H1 and H2 must both pass at level 0.05.

## Inference

For H1, H2, h+6, and path-24, report:

- every one of the eleven event differences;
- equal-event mean and median;
- exact two-sided sign-flip randomization p-value over all `2^11` assignments;
- 50,000-resample event bootstrap 95% interval;
- positive-event count;
- leave-one-event influence means;
- within-event seed spread as an optimization diagnostic.

The inferential gate for H1 or H2 is:

```text
positive equal-event mean AND exact sign-flip p < 0.05
```

The bootstrap interval, sign count, and leave-one-event influence are reported diagnostics, not additional redundant vetoes.

## Claim logic

- **H1 and H2 pass:** evidence that flow separation improves held-out transition prediction and that the benefit survives 24-hour rollout.
- **H1 passes, H2 fails:** the transition-level structural signal is lost through estimation or rollout.
- **H1 fails, H2 passes:** report the forecast gain, but do not attribute it to the oracle-gap theorem.
- **Both fail:** the current real task lies in a one-flow-sufficient or estimation/misspecification-dominated regime.

HGB may outperform both flow models. That limits practical competitiveness but does not alter the mathematical oracle-gap result.

## Explicitly deferred

Until the main table is frozen, do not run:

- mixed one-step/rollout objectives;
- family or phase-separation studies;
- event-shift mechanism tests;
- D-6 or a real-data `Gamma` plug-in;
- information-gated models;
- recovery memory or secondary-damage modules;
- semiparametric models;
- width/depth searches;
- 48-hour campaigns.

The supplied balanced two-state solvable case is the only synthetic experiment in the paper.
