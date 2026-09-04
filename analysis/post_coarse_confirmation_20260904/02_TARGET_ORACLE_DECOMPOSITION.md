# Post-hoc target-oracle decomposition

**Evidence status:** explanatory diagnostic only. It uses the already revealed 2022/2024 target outcomes to fit target-event oracle coefficients inside the frozen source-defined clusters. It is not a new confirmation experiment and cannot be used for model selection on these events.

## 1. Exact sample decomposition

Let `P` denote the 2018--2021 source pool and let `Q=e` be one confirmation event. Keep the source-fitted scaler and the eight source-fitted K-means cells fixed. Let

- `f_{j,P}` be the source-fitted predictor in class `j`, where `j=1` is the union-of-rays one-flow class and `j=2` is the two-flow box;
- `f_{j,Q}^*` be the exact target-event empirical-risk minimizer in the same class and the same fixed cells;
- `R_Q` be target active-48 one-step MSE.

Then the observed paired difference has the exact algebraic decomposition

\[
\begin{aligned}
d_Q
&=R_Q(f_{1,P})-R_Q(f_{2,P})\\
&=\underbrace{R_Q(f_{1,Q}^*)-R_Q(f_{2,Q}^*)}_{G_Q^{\mathrm{oracle}}}
 +\underbrace{\{R_Q(f_{1,P})-R_Q(f_{1,Q}^*)\}}_{T_{1,Q}}
 -\underbrace{\{R_Q(f_{2,P})-R_Q(f_{2,Q}^*)\}}_{T_{2,Q}}.
\end{aligned}
\]

`G_Q^oracle` measures the target-event representation advantage of retaining two rates at the frozen resolution. `T_{j,Q}` measures source-to-target coefficient/conditional-law drift within class `j`.

## 2. Aggregate diagnosis

Across the eleven events with an available active-48 endpoint:

| quantity | equal-event mean | median | positive events |
|---|---:|---:|---:|
| observed `d_Q` | `-3.5138e-07` | `+3.6674e-07` | 6/11 |
| target-oracle gap `G_Q^oracle` | `+9.0285e-07` | `+6.0597e-07` | 10/11 |
| one-flow transport penalty `T_1` | `+5.4389e-06` | `+5.2110e-06` | 11/11 |
| two-flow transport penalty `T_2` | `+6.6932e-06` | `+5.3091e-06` | 11/11 |
| `T_1-T_2` | `-1.2542e-06` | `-3.3947e-07` | 0/11 |

The decomposition reconciles exactly to floating-point precision. A conservative one-extra-degree-of-freedom correction to the in-sample target-oracle gap leaves the corrected gap positive in 10/11 events and an equal-event mean of approximately `8.23e-07`.

The key conclusion is therefore not “the target events are one-flow.” It is:

> At the fixed K=8 resolution, the second rate usually reduces target-event oracle transition risk, but source-fitted two-flow coefficients incur more cross-event transport error than the one-flow restriction. The one-flow constraint acts as regularization under conditional-law shift.

This is the missing term in the source-cell `Gamma` argument. `Gamma` prices an extra degree of freedom under a fixed design/distribution; it does not by itself price the stability of the additional coefficient across storm environments.

## 3. Cluster-level failure

Target-weighted contributions identify two source cells as decisive:

- **Cluster 6** contributes a positive target one-step difference of about `+2.74e-06` per assigned row and drives much of the benefit on `2022-04-13`, `2022-06-17`, `2022-07-23`, `2024-05-08`, and `2024-05-26`.
- **Cluster 1** contributes about `-4.34e-06` per assigned target row and dominates the aggregate harm. Its source rates are `U=0.001537`, `R=0.008572`, while its one-flow source branch is interruption-only.

For `2024-09-27`, cluster 1 contains 39.25% of active transitions. The target-event oracle in this cell is essentially interruption-only (`U=0.006797`, `R≈0`), so the transported source restoration component is harmful. Cluster 1 contributes `-8.718e-06` of the event's total `-9.245e-06` one-step difference, or 94.3%.

Several other cells exhibit branch instability. Source cluster 3 is interruption-only in both frozen arms, yet the target-event best one-flow branch is restoration in 7 of 11 events in which the cell appears. The source two-flow estimator therefore cannot realize a target representation gap in that cell because its fitted restoration coefficient has already collapsed to zero.

## 4. Relation to rollout

Post-hoc event-level rank correlations are high:

\[
\rho_S(d^{\mathrm{step}},d^{\mathrm{path24}})=0.945,
\qquad
\rho_S(d^{\mathrm{step}},d^{h24})=0.964.
\]

The target-oracle gap also ranks the observed one-step and path differences strongly (Spearman `0.945` and `0.927`, respectively). These results should not be treated as a registered predictor test, but they indicate that the representation signal is meaningful and that the principal obstacle is transport of the conditional rates, not the rollout recursion itself.

## 5. Theory consequence

The paper's flow-selection criterion needs an out-of-environment term. At a minimum, the empirical organization should distinguish

\[
\text{target representation gain}
\quad\text{from}\quad
\text{additional two-flow transport cost}.
\]

A richer formal result can express the target risk of a source-fitted two-rate estimator through the target information matrix and source estimation covariance, together with coefficient drift. The current confirmation shows that the drift term is first-order: it is larger than the target oracle gap in equal-event mean.

The full event and cluster tables are stored beside this note. No model is promoted from this post-hoc analysis.