# Minimal repair gate before any new accuracy experiment

This is a repair protocol, not a model search. It authorizes only the changes needed to make the existing pilot interpretable. The manuscript, legacy result status, event cohort, and main scientific hypothesis remain unchanged until the repaired pilot is reviewed.

## Phase R0 — data-path audit in the original mounted environment

Run:

```bash
PYTHONPATH=src:experiments \
python analysis/post_pilot_root_cause_20260904/transition_data_audit.py
```

The script requires the `data/interim/panel_*.npz` files used by the pilot. It must report:

- the share of Stage-A rows whose intermediate current state is unobserved;
- anchor-window transition composition versus all adjacent observed pairs;
- positive, negative, and quiet transition shares;
- state variance and active-state share;
- row-pooled versus equal-event constant fits.

No retraining is permitted before this audit is complete.

## Phase R1 — deterministic code repairs

### R1.1 Correct the adjacent transition mask

A Stage-A row at physical time `t` is legal only when

```text
observed[t] AND observed[t+1]
```

holds. Never use forecast-origin observability as a substitute for current-state observability. Do not zero-fill an unobserved teacher-forced current state.

### R1.2 Make both interruption MLPs trainable while preserving update zero

For each interruption head:

1. initialize hidden-layer weights independently with a standard live initialization;
2. initialize hidden biases conventionally;
3. set only the final output **weight** to zero;
4. set the final output bias to the calibrated constant logit.

At update zero the emitted flow remains exactly constant, but hidden representations differ. Required gradient test:

- after the first backward pass, each final output weight has nonzero gradient and the two heads' gradients differ;
- after one optimizer update, a second backward pass gives nonzero gradient to at least one hidden-layer weight in each head;
- after ten updates, the two heads are not numerically identical.

A module-level norm is insufficient; save parameter-level reachability.

### R1.3 Train the hold on sequences

Stage A must process short ordered sequences, not independent rows with `held_prev=None` each time. It may use teacher-forced observed states while carrying the held interruption logit through consecutive hours. The recovery recomputation state and its phase must also be explicit.

A deterministic test must show:

- changing the hold gate changes the sequence prediction when raw logits vary over time;
- the hold parameters receive nonzero gradient;
- the recurrence exactly matches a direct reference calculation.

### R1.4 Replace the dead zero start by a live branch start

Keep the exact zero ray as the update-0 **baseline**, but do not require gradient descent to escape a ReLU kink and saturated sigmoid floor.

For each one-flow direction, create a fixed live initialization with a prespecified small flow scale derived from source data only. The recommended rule is the event-balanced median of positive per-exposure transition rates for the interruption start and of negative per-exposure rates for the restoration start, clipped to the fixed caps. This is an optimization start, not a new scientific arm. The exact ray optimum remains separately reported.

Both starts must pass:

- nonzero gradient to the active process pathway;
- finite validation loss;
- measurable movement away from update zero under a deterministic smoke sequence.

### R1.5 Match initialization to the training estimand

If the primary training objective is equal-event risk, compute neural calibration with equal-event weights. Continue to report row-pooled constant baselines if operationally useful, but do not silently use a row-pooled optimum as the equal-event neural initialization.

### R1.6 Fix deterministic baseline and preprocessing defects

- select the constant one-flow ray by comparing the two ray SSEs, not coefficient magnitudes;
- compute static imputation values from source fit counties only and apply them unchanged to validation/test counties;
- make clock normalization match the documented rule, or update the documentation before execution;
- save every normalization and imputation digest.

## Phase R2 — replace the degenerate pilot sampling

### R2.1 Use three distinct held-out folds

The implementation pilot must select exactly one test event from each of three distinct outer folds. Selection remains outcome-blind for paper inference. The report must list the training-event set and verify that all three estimators are distinct.

For a separate implementation stress test, it is permissible to choose one real event with adequate interior-state support or use a synthetic deterministic sequence. Such a stress test is explicitly not paper evidence.

### R2.2 Define storm time by footprint intensity, not the union minimum/maximum

The earliest and latest NOAA rows across all counties span almost the full panel. Replace them with an outcome-blind hourly footprint process, for example the number/share of panel counties carrying a compatible NOAA event or exceeding a fixed standardized weather-footprint threshold.

Freeze three anchors relative to that footprint:

```text
pre-peak: 12 h before footprint peak
peak: footprint peak
post-peak: 12 h after footprint peak
```

or an equivalent fixed rule based on threshold crossings of the same exogenous footprint. An anchor may not be clipped into validity; the panel must contain the required context, or that anchor is marked unavailable. Report the resulting distances to the panel boundaries.

### R2.3 Stage A uses all legal unique adjacent transitions in the storm-conditioned interval

Do not restrict transition pretraining to the three rollout-origin windows. Define an outcome-blind storm-conditioned interval around the exogenous footprint and use every unique adjacent observed pair inside it. The three anchors remain for rollout evaluation/fine-tuning only.

## Phase R3 — mandatory treatment and training diagnostics

Every fitted model must save per event, origin, and forecast step:

```text
U_tilde
R_tilde
s = U_tilde - R_tilde
c = min(U_tilde, R_tilde)
U and R after collapse
g_occ
q_hold
raw and held interruption logits
recovery block phase
```

Report:

- mean, median, p90, p99 and zero share of `c`;
- mean absolute delivered transition treatment `|c(1-2Y)|`;
- fraction of cells for which the two arms' transition functions differ by more than `1e-6`;
- gradient norms by **parameter group**, including hidden weights;
- update-0 and selected-checkpoint performance;
- whether every learned module improves on its initialized constant behavior.

A one-flow/two-flow comparison is not interpretable when the delivered treatment is numerically negligible.

## Phase R4 — repaired pilot only

After R0–R3 pass, rerun:

```text
3 distinct folds
× 1 seed
× [1 two-flow job + 2 one-flow live starts]
```

Use the same fixed update budgets. Do not add or remove features, change widths, tune rate caps, or introduce another model family.

The repaired pilot is reviewed on four gates:

### Gate A: estimator reachability

- interruption hidden weights receive gradient and change;
- two heads diverge;
- hold has nonzero temporal effect;
- both one-flow starts are trainable;
- no state clamp activation.

### Gate B: data validity

- zero illegal teacher-forced rows;
- three distinct training problems;
- anchors not boundary-clipped;
- transition composition and state support reported.

### Gate C: treatment strength

- final `c` and `c(1-2Y)` are saved;
- the structural arms differ on a non-negligible set of cells;
- any full/interior comparison is accompanied by its treatment-dose distribution.

### Gate D: absolute usefulness

- each trained model is compared with its update-0 model;
- both are compared with exact constant one-flow/two-flow baselines;
- a no-weather affine transition baseline is included for one-step risk.

The signs of the three-event results are not a promotion criterion. Passing the four implementation/data gates is. Once they pass, the pilot may inform whether a 26-event main run is worth the compute; it cannot itself establish the paper claim.

## Deferred parameterization question

Output-level collapse is functionally nested but carries a flat common mode. After the repaired pilot, and only if optimization diagnostics still show cancellation or saturation, replace it with an explicit direction–concurrency parameterization:

\[
U=[s]_+ + c,\qquad R=[-s]_+ + c,
\]

where the process-specific asymmetric encoders are retained and the one-flow arm fixes `c=0`. This is one prespecified parameterization ablation, not an architecture sweep.

## Forbidden before repaired-pilot review

- full five-fold, three-seed campaign;
- changes to paper conclusions or legacy evidence labels;
- selection of events by previous model gains;
- new gates, memory states, recovery architectures, transformers, or semi-parametric components;
- width, learning-rate, cap, objective-weight, or origin-grid sweeps;
- interpreting the current negative pilot as evidence against the class-level theorem.
