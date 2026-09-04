# Training-objective and temporal-context addendum

These findings are source-level and supplement `01_ROOT_CAUSE_ANALYSIS.md`.

## 1. Stage-A checkpoints are selected by rollout loss, not transition loss

`train_two_stage` receives one `val_fn`. The pilot defines that function as the equal-event **24-hour rollout** validation loss. The same function is used to select the best checkpoint during teacher-forced Stage A and during rollout Stage B.

Consequently, the checkpoint called `stage_a_best` is not the model with the best held-out one-step conditional-transition risk. It is the Stage-A iterate with the best rollout risk. This weakens the intended theory-to-estimator bridge even in the absence of the dead-gradient defects.

The minimal correction is one of:

1. use a one-step validation function for Stage A and rollout validation for Stage B; or
2. treat Stage A as a fixed pretraining schedule with no checkpoint selection, then let Stage B perform rollout selection.

Do not use the same metric while describing the stages as optimizing two different statistical targets.

## 2. Transition deduplication is incompatible with origin-specific history features

Stage A deduplicates a transition by `(county, next_hour)` and keeps the first anchor-window occurrence. Yet `x_r` contains pre-origin outage-history summaries, and `x_u` contains cumulative-since-origin weather features. The same physical transition can therefore have different context values under different forecast origins.

Keeping the first occurrence makes the context assignment depend on anchor ordering. It is neither:

- a unique physical transition dataset with context defined at the transition time; nor
- an origin-conditioned rollout dataset retaining every origin-transition pair.

A clean choice is required:

- for theory-aligned transition pretraining, construct each transition once with context defined at its own physical time from legal exogenous/past information; or
- for origin-conditioned pretraining, retain origin-transition pairs and use explicit weights so repeated physical transitions do not dominate.

The current hybrid should not be retained.

## 3. Eight-step recovery blocks have arbitrary rolling-origin phase

Recovery is recomputed when `step % 8 == 0`. Under a single fixed competition cutoff this defines one common block calendar. Under rolling origins, the same physical hour can fall at a different block phase depending on the forecast origin.

This is not data leakage, but it is an unstable representation of recovery time. Before retaining the mechanism, compare the intended semantics:

- wall-clock aligned blocks;
- time-since-exogenous-storm-peak blocks; or
- hourly recovery evaluation.

This is a semantic adjudication, not a broad sweep. It should occur only after the P0 repair gates, because the current pilot cannot isolate it.

## 4. Output-level collapse supplies a function-class comparison but not a clean optimization comparison

For the collapsed arm, the loss depends on `s=U_tilde-R_tilde` and is invariant to common shifts of the two proposals. The proposal modules therefore contain a nuisance direction with zero population effect. The two-flow arm does not have the same flat direction because its common component changes the drift by `c(1-2y)`.

The present parameter matching is conservative in one sense—the one-flow arm receives all proposal capacity—but ill-conditioned in another. If the repaired pilot still exhibits optimizer cancellation after the P0 fixes, the single authorized parameterization comparison is explicit direction/concurrency coordinates:

\[
U=[s]_+ + c,\qquad R=[-s]_+ + c,
\]

with identical process-specific encoders in both arms and `c=0` in the one-flow arm.

## 5. Required stage-specific diagnostics

The next trainer must save, separately for Stage A and Stage B:

```text
training objective
validation objective
selected checkpoint criterion
transition rows or origin-transition pairs used
number of distinct physical transitions
number of repeated physical transitions and total weight
hold-state reset policy
recovery-block phase policy
```

Without these fields, a one-step theorem, transition pretraining, and rollout evaluation can again be conflated even if the numerical result looks favorable.
