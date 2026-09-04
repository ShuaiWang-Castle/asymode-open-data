# Root-cause analysis of the negative V2 pilot

## 1. Executive reading

The observed pilot differences are near zero, but the null cannot be attributed to the scientific one-flow/two-flow contrast. The nominal interruption model was reduced by initialization to a scalar bias model, the temporal hold was disconnected from Stage A and inert in Stage B, one one-flow start was effectively non-trainable, and the data design supplied only two distinct trained models. The experiment therefore compared two largely recovery-driven models whose initial common component was almost zero.

This diagnosis preserves two distinctions:

- **mathematical result:** the one-flow representation gap and identification geometry remain statements about model classes under their assumptions;
- **empirical adjudication:** the current pilot did not realize a trainable member of the intended rich two-flow class and did not measure a nontrivial concurrency treatment.

## 2. P0 code failure: the interruption feature learner is permanently dead

The interruption magnitude module is a depth-three linear/ReLU network:

```text
Linear -> ReLU -> Linear -> ReLU -> Linear
```

`apply_modular_init` sets the weights and biases of **every** linear layer to zero and then changes only the final bias. Consequently, for every input `x_u`,

```text
hidden_1 = 0
hidden_2 = 0
raw_logit = final_bias
```

Backpropagation cannot repair this state:

- the final output weight receives `hidden_2`, which is zero;
- the second hidden layer receives the final output weight, which is zero;
- the first hidden layer receives the second-layer weight, which is zero.

Only the final scalar bias has a nonzero gradient. An independent backward probe on committed code measured a maximum gradient of exactly `0.000e+00` for every non-final-bias parameter in both interruption MLPs, while each final bias had gradient `4.268e-05`.

Therefore:

- the 32 interruption features are never used by the magnitude pathway;
- the two width-32 heads remain identical forever;
- their average is one scalar function, not an ensemble;
- sufficient optimization updates cannot repair the architecture because the relevant Jacobian is zero.

The pilot's module-level gradient check reported a nonzero `gn_head_a` because it pooled the live final bias with all dead feature weights. The correct gate is parameter-level reachability, not module-level norm.

## 3. P0 temporal failure: the hold cannot learn

The teacher-forced interface calls:

```python
proposals(x_u_t, x_occ_t, x_r_t, held_prev=None, r_prev=None, step=0)
```

for every transition. Thus Stage A never forms the recurrence

\[
\ell_t=q_t\ell_{t-1}+(1-q_t)z_t,
\]

and the hold gate is absent from the Stage-A computational graph.

In Stage B, the magnitude logit is constant over counties and time because the MLP feature weights are dead. If `z_t=z` at every step and `\ell_0=z`, then `\ell_t=z` for every value of `q_t`; hence the hold remains unidentifiable there as well. The committed diagnostics show a hold-gradient mean on the order of `10^{-15}`.

This matters because the competition result attributed a substantial pulse-width effect to the first-order hold. The pilot contained a module with that name, but not a trainable realization of that mechanism.

## 4. P0 comparator failure: one of the two one-flow starts is not a usable optimization start

The exact constant interruption-ray optimum equals zero on all three pilot training problems. Mapping `(U_0,R_0)=(0,0)` into the modular network sends the bounded sigmoid pathways to their numerical floor. The subsequent output collapse is evaluated at the ReLU kink `s=U-R\approx0`.

The independent probe measured total gradient norm `9.327e-12` from this start. The committed Stage-A validation trace is flat to printed precision over many checks. Hence the nominal two-start safeguard does not provide two viable attraction basins; in practice, only the restoration-ray start trains, and it is selected for every reported event.

The appropriate distinction is:

- the exact zero ray is a valid **constant baseline**;
- it is not automatically a useful **neural optimization initialization** for a sparse, context-dependent interruption function.

The exact global constant solution should remain in the table, but neural initialization must be live while preserving the same update-0 conditional mean to numerical tolerance.

## 5. P0 data-path failure: the teacher-forced mask is not the adjacent-observation mask

For an anchor origin `k`, the pilot stores

```python
m[:,t] = observed[:,k] & observed[:,k+1+t]
```

and then uses this same mask for the transition from time `k+t` to `k+t+1`. For `t>0`, the correct transition mask is

```python
observed[:,k+t] & observed[:,k+t+1].
```

The hourly state array is converted with `nan_to_num` before packing. Therefore, when the intermediate current state is unobserved but the origin and next target are observed, the teacher-forced row enters with a fabricated current state of zero.

This is especially consequential here because the scientific contrast is state scaled and the interruption signal is strongest at zero. The exact contaminated-row share still needs to be computed in the mounted pilot environment; the GitHub Actions checkout does not contain the `panel_*.npz` bytes. The source-level defect, however, is unambiguous.

## 6. P0 design failure: three events are only two training problems

The selected events are:

```text
2024-05-08  fold 2
2022-03-12  fold 1
2018-10-11  fold 2
```

Because the pilot trains on every event outside the held-out fold, the two fold-2 tests use the same source events, county hashes, normalization, initialization, seed, and deterministic optimization trajectory. Their selected stage, selected update, validation score, Stage-A best, and Stage-B best are identical for every arm/start.

Thus the pilot evaluates three target events but only two trained estimators. This is useful for detecting target-event transfer failure, but it is not a three-replication implementation pilot.

## 7. P0 origin failure: the nominal event-centered task is nearly a fixed boundary grid

The union of NOAA rows over a large multi-state panel spans almost the entire seven-day panel. After enforcing 24 hours of history and 24 hours of future targets:

- 48 of 78 anchors are clipped to the legal boundary;
- 24 of 26 pre anchors are index 24;
- 25 of 26 post anchors are index 143;
- the median NOAA interval occupies 162 of 168 panel hours.

The resulting origin set is approximately `[24, midpoint, 143]`, not a storm-centered set. Stage A then takes unique transitions only from the 24-hour windows following those anchors. It does not use all unique adjacent observed transitions, and it can underrepresent the actual interruption pulse while retaining quiet and late-recovery periods.

The fitted global constants are consistent with such a task: `U_0` is zero or approximately `2.7e-5`, while `R_0` lies between `1.3e-2` and `2.7e-2`.

## 8. The structural treatment was initially negligible and finally unmeasured

Both arms first produce `U_tilde` and `R_tilde`; the one-flow arm removes

\[
c_t=\min\{\widetilde U_t,\widetilde R_t\}.
\]

At update zero, the source constants imply

```text
c_0 / R_0 = 0% to 0.1%.
```

Thus the two arms begin nearly identical. A null would be unsurprising unless training develops a material common component. However, the pilot does not save final trajectories or summaries of

```text
U_tilde, R_tilde, s_t, c_t, occurrence gate, hold gate.
```

Accordingly, the experiment cannot answer whether the structural intervention remained negligible or whether it became active but failed to improve prediction. This is a measurement failure, not merely a missing visualization.

The actual pointwise difference between the two-flow and collapsed one-flow drifts is

\[
m_2(y)-m_1(y)=c_t(1-2y).
\]

A future report must record both `c_t` and `c_t(1-2Y_t)`; the latter is the delivered treatment in transition space.

## 9. The exact constant initializer is correct but mismatched to the optimization problem

The constant solvers are numerically correct. The problem is their use as the only neural starting point.

First, the constants are fitted by concatenating all source rows with unit weights, whereas training and validation claim to target an equal-event risk. The initialization is therefore optimal for a different estimand.

Second, a global constant optimum near zero is expected when interruption is rare and context dependent. Passing that value through bounded sigmoids produces very negative biases and very small derivatives. This is a weak starting point for discovering rare conditional interruption even when the overall constant solution is correct.

Third, the deterministic `constant_one_flow` baseline is selected in code by comparing `a_ray >= b_ray`, i.e. coefficient magnitude, rather than comparing the two ray SSEs. This happened not to affect the pilot because `a_ray=0`, but it is incorrect generally.

## 10. Competition-to-open-data temporal semantics do not match

The final challenge model had one fixed cutoff. Its cumulative hazard, held interruption logit, and eight-hour restoration schedule all shared that physical time origin.

The open-data pilot has many rolling origins. Each origin resets:

- accumulated-since-origin hazard features;
- the held interruption state;
- the phase of the eight-step restoration schedule.

A forecast issued one hour later can therefore assign a different internal history and a different recovery-block phase to the same physical clock time. Stage A resets those states on every individual transition. The architecture preserves the names of the competition mechanisms but not their original semantics.

This mismatch is a plausible source of model transfer failure and should be treated separately from the flow-separation question.

## 11. The tropical event diagnoses context transfer, not flow collapse

`2024-05-08` and `2018-10-11` use the exact same fold-2 trained estimators. On the former, both trained arms improve over their update-0 constant models. On the latter, both are much worse than those constants:

```text
2018-10-11 constant two-flow path MSE: 3.107e-4
2018-10-11 trained two-flow path MSE:  5.339e-4
2018-10-11 trained one-flow path MSE:  5.347e-4
```

Because the two arms fail together and differ only slightly, the load-bearing fact is not the sign of the two-versus-one difference. It is that the learned context-to-rate map transfers poorly to that tropical target. The recovery/statics path is a leading candidate because it was the only strongly trainable process module, but final coefficient and rate traces were not saved, so this attribution remains a hypothesis.

## 12. Output-level nesting is functionally valid but optimization-redundant

For the one-flow arm, the output depends only on

\[
s=\widetilde U-\widetilde R.
\]

Any common change `(U_tilde+a,R_tilde+a)` leaves the prediction unchanged. The restricted model therefore carries an unidentifiable common mode even though its function class is correct. This gives the one-flow optimizer a flat nuisance direction and can induce cancellation or saturation between process-specific proposal modules.

The clean long-run parameterization is explicit direction plus concurrency:

\[
U=[s]_+ + c,\qquad R=[-s]_+ + c,
\]

with the one-flow model fixing `c=0`. Process-specific asymmetric encoders may still feed `s` and `c`; architectural symmetry is not required. This change should be considered only after the P0 data and gradient gates pass, because it changes parameterization rather than merely repairing a bug.

## 13. Data informativeness of the three pilot events

The events were selected without model outcomes, but family diversity did not ensure support for the second-flow question:

| event | family | zero share | near-zero share | interior share | future-onset share |
|---|---|---:|---:|---:|---:|
| 2024-05-08 | convective | 56.2% | 38.7% | 5.1% | 0.42% |
| 2022-03-12 | winter | 66.6% | 33.0% | 0.38% | 0.44% |
| 2018-10-11 | tropical | 50.4% | 32.8% | 16.9% | 0.57% |

The winter medoid is nearly devoid of interior origins. It can test numerical stability in a quiet regime, but it provides little power to test the value of an interior common component. This does not justify selecting favorable events for the main study. It means an **implementation stress test** and a **paper evaluation** should be distinguished: the former must exercise every pathway; the latter must include the full frozen cohort.

## 14. Relation to the theory

The exact oracle gap and identification results remain conditional statements about the class

\[
m(x,y)=U(x)(1-y)-R(x)y.
\]

The negative pilot does not contradict them, because:

1. the nonlinear interruption function was never trainable;
2. the delivered common component was initially negligible and finally unmeasured;
3. the one-step data path contains a mask error and does not use the intended transition population;
4. the pilot target distribution is dominated by boundary states;
5. recursive rollout and cross-event transfer add estimation and misspecification terms outside the oracle theorem.

The correct status is therefore:

> The class-level theory remains intact; the current estimator-level pilot is inconclusive because the intended estimator and empirical treatment were not realized.

## 15. Root-cause ranking

### Proven P0 causes

1. dead interruption feature weights;
2. disconnected/inert first-order hold;
3. effectively dead interruption-ray start;
4. adjacent-state observation-mask error;
5. only two distinct trained estimators;
6. boundary-degenerate origin rule and anchor-only transition sample;
7. missing final treatment-dose diagnostics.

### Strong P1 explanations

1. global constant initialization saturates a rare interruption process;
2. challenge temporal semantics do not transfer to rolling origins;
3. one-flow output collapse has a redundant common mode;
4. recovery/context mapping transfers poorly to the tropical target;
5. row-pooled initialization and equal-event training optimize different risks.

### Still open scientific questions

1. whether a live asymmetric two-process NN beats its exact one-flow submodel;
2. whether any gain occurs only at onset or also in the interior;
3. whether one-step gain survives a 24-hour rollout;
4. whether broader storm-family data support the same conclusion as the legacy cohort.
