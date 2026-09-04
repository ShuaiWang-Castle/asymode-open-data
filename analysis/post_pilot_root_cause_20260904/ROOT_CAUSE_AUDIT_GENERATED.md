# Generated post-pilot root-cause audit

This report is generated from committed code and committed pilot outputs; it performs no model retraining.

## Load-bearing findings

### 1. The interruption MLPs are not merely identical; their feature-learning weights are dead

`apply_modular_init` zeros every linear layer in both ReLU MLPs. At that point all hidden activations are exactly zero and every downstream weight is zero. A backward pass therefore reaches only the final scalar bias; it cannot reach the final output weight or either hidden layer. The audit probe measured:

- two-flow `head_a` non-final-bias gradient maximum: `0.000e+00`;
- two-flow `head_a` final-bias gradient: `4.268e-05`;
- the corresponding `head_b` values: `0.000e+00` and `4.268e-05`.

Thus the nominal pair of width-32 interruption networks learns only a time-invariant scalar bias in this pilot. The rich 32-channel interruption input cannot enter either magnitude head. Module-level nonzero-gradient checks missed this because they aggregate the live final bias with all dead weights.

### 2. Stage A cannot train the first-order hold by construction

`step_from_state` resets `held_prev=None`, `r_prev=None`, and `step=0` on every transition: `True`. Consequently the hold gate is outside the one-step computation graph in Stage A. The committed checkpoint diagnostics are consistent with this: the hold-gradient mean is effectively zero.

### 3. The interruption-ray safeguard is a dead start

With the zero/zero one-flow start, the total gradient in the independent probe is `9.327e-12`. The collapse is evaluated at the ReLU kink `s=U-R≈0`, while the main interruption MLP is already dead. The committed interruption-start Stage-A validation is flat to printed precision for long stretches. Therefore the nominal two-start safeguard effectively supplies only one trainable start—the restoration start—which is selected in all three reported events.

### 4. Three reported pilot events represent only two distinct optimisation problems

Pilot event-to-fold map: `{'2024-05-08': 2, '2022-03-12': 1, '2018-10-11': 2}`. The number of distinct held-out folds is `2`, not three. Events in the same fold use the same source events, county split, normalization, initialization, seed, and deterministic training path. The committed training-side records for the duplicate fold are identical for every arm/start: `True`.

### 5. The selected origin rule does not isolate storm dynamics

The committed origin audit shows that 62% of anchors are clipped to legal boundaries, so almost every panel uses an early boundary origin, one midpoint, and a late boundary origin. Stage A then draws unique transitions only from the 24-hour windows following those anchors—not from all unique panel transitions. This sampling scheme can omit the actual interruption pulse and heavily represent quiet/recovery transitions, which is consistent with the fitted constant interruption ray being zero.

### 6. The pilot cannot measure the intended structural contrast at update zero

The fitted source constants have `U0` equal to zero or about `2.7e-5`, versus `R0` of roughly `1.3e-2` to `2.7e-2`. The one-flow collapse deletes `min(U,R)`, which is therefore initially negligible. This does not establish that the learned final concurrency is negligible—the required final rate/concurrency traces were not saved—but it does establish that the pilot begins in a nearly unseparated regime and lacks the diagnostics needed to show that it ever leaves it.

### 7. A challenge-specific temporal representation was transferred without its original time semantics

In the challenge there was one fixed cutoff, so cumulative hazard, the first-order hold, and the eight-step recovery schedule had a common temporal origin. Here these states reset independently at every forecast origin. `path_*_since_origin` and the eight-step recovery phase therefore depend on an arbitrary rolling-origin coordinate. In Stage A, temporal state is reset altogether. The imported mechanisms are not operating under the semantics under which they were successful.

### 8. The tropical failure is a transfer failure of the fitted context model, not evidence about one versus two flows

The same fold-2 trained models perform acceptably on `2024-05-08` but both are much worse than their update-0 constant model on `2018-10-11`. Because the learned model is identical before test evaluation, this contrast isolates target-event shift. It does not identify the flow-collapse effect. The likely failing object is the learned context-to-rate map, especially the recovery/statics path, while the intended nonlinear interruption magnitude path was dead.

## Additional implementation defects before any main run

- `constant_one_flow` chooses its ray by comparing coefficient magnitudes (`a_ray >= b_ray`) rather than the two ray SSEs: `True`. This did not change this pilot because `a_ray=0`, but it is incorrect in general.
- Static missing values are imputed using each event's own county median before the source/test split, contrary to the documented fit-source-only preprocessing rule.
- The feature map says clock channels are not normalized, but the pilot standardizes every `x_u` column.
- Final proposal/rate/concurrency/gate/hold summaries were requested but are absent, preventing attribution of the small two-flow/one-flow differences.

## Pilot task composition

| event      |   fold | family     |   zero_origin_share |   near_zero_share |   interior_share |   future_onset_share |   median_outage |   p90_outage |
|:-----------|-------:|:-----------|--------------------:|------------------:|-----------------:|---------------------:|----------------:|-------------:|
| 2018-10-11 |      2 | tropical   |            0.503639 |          0.327511 |       0.16885    |           0.00567376 |     1.04381e-05 |  0.0906734   |
| 2022-03-12 |      1 | winter     |            0.665818 |          0.330363 |       0.00381922 |           0.00435323 |     0           |  0.000366535 |
| 2024-05-08 |      2 | convective |            0.562147 |          0.387006 |       0.0508475  |           0.00423729 |     0           |  0.00158167  |

The winter medoid has only about 0.38% interior origins, and two of the three events have very small future-onset shares. Family diversity alone did not create an informative pilot for the second-flow question.

## Performance and update-zero movement

| test_event   |   rel_tf_mse_full |   rel_path_mse_full |   rel_h24_mse_full |   two_vs_own_update0_path_full |   one_vs_own_update0_path_full |   two_vs_constant_two_flow_path_full |   one_vs_constant_two_flow_path_full |
|:-------------|------------------:|--------------------:|-------------------:|-------------------------------:|-------------------------------:|-------------------------------------:|-------------------------------------:|
| 2024-05-08   |         0.0652811 |         0.58164     |        0.550869    |                    7.92723e-06 |                    7.46818e-06 |                          7.92723e-06 |                          7.68224e-06 |
| 2022-03-12   |         0         |         3.57294e-05 |        7.38522e-05 |                    1.24268e-06 |                    1.24268e-06 |                          1.24269e-06 |                          1.24268e-06 |
| 2018-10-11   |        -0.0514037 |         0.151546    |        0.503359    |                   -0.000223179 |                   -0.000224541 |                         -0.000223179 |                         -0.000223989 |

## Root-cause hierarchy

1. **P0 implementation failure:** all interruption MLP feature weights are dead; Stage A cannot train the hold; one of the two one-flow starts is effectively dead.
2. **P0 experiment-design failure:** three events reduce to two training problems; anchors are boundary-clipped and Stage A samples only their windows.
3. **P1 model-transfer mismatch:** challenge temporal states are reset under a rolling-origin task, and the learned context map fails sharply on the tropical target.
4. **P1 measurement failure:** final `U`, `R`, common component, gate, and hold trajectories were not saved, so the pilot cannot verify whether the structural treatment was nontrivial.
5. **Scientific question remains open:** the pilot's null is not a clean estimate of the value of one versus two conditional-mean flows.

## Immediate implication

Do not launch the 45-job main campaign and do not revise the manuscript conclusion from this pilot. First repair only the P0 items, run a deterministic gradient/trajectory smoke test, and then repeat a three-distinct-fold pilot. No new model family or hyperparameter sweep is warranted before those controls pass.
