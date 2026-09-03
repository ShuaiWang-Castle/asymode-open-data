# Unified AISTATS v2 study protocol

**Status:** locked before the v2 pilot is run. This document replaces the previous collection of parallel model ideas. The study now has one core theorem, one proposed neural model, one primary experiment, and two ablations.

## 1. Scientific question

The paper asks one question:

> When does separating a signed state change into nonnegative interruption and restoration rates improve transfer to an unseen event?

Power outages are the application, not a second independent contribution. The statistical object is a bounded state driven by two opposing conditional flows.

## 2. Core theory

Let

\[
m_x(y)=U(x)(1-y)-R(x)y,
\qquad U(x),R(x)\ge 0.
\]

The main theorem is the **environment-dependent projection theorem**. If a one-rate interruption branch is fitted under source-event distribution \(P\),

\[
a_P^*(x)=U(x)-R(x)\frac{C_P(x)}{A_P(x)},
\]

where

\[
A_e(x)=\mathbb E_e[(1-Y)^2\mid X=x],
\qquad
C_e(x)=\mathbb E_e[Y(1-Y)\mid X=x].
\]

Its target-event risk under \(Q\) decomposes exactly as

\[
\mathbb E_Q[(m_x(Y)-a_P^*(x)(1-Y))^2\mid X=x]
=
R(x)^2\frac{v_Q(x)}{A_Q(x)}
+
R(x)^2A_Q(x)
\left(
\frac{C_P(x)}{A_P(x)}-\frac{C_Q(x)}{A_Q(x)}
\right)^2.
\]

The first term is the target-event oracle approximation gap; the second is a nonnegative event-projection-shift penalty. The restoration-only branch has the symmetric formula.

The identification geometry is retained as a supporting lemma, not a separate empirical mechanism:

\[
Q(x)=\mathbb E[\phi(Y)\phi(Y)^\top\mid X=x],
\quad
\phi(Y)=(1-Y,-Y)^\top,
\]

\[
\det Q(x)=\operatorname{Var}(Y\mid X=x),
\qquad
v(x)\le \lambda_{\min}(Q(x))\le 2v(x).
\]

It establishes when the two rates are recoverable. It does **not** guarantee that the two-rate predictor beats a collapsed predictor. The predictive comparison also contains finite-sample estimation error and model misspecification.

The paper therefore makes no claim that local identifiability explains event-family ordering, and no claim that event holdout must always enlarge the empirical two-rate gain.

## 3. Data and statistical unit

The primary cohort is exactly the eleven events in `configs/panel_manifest_g2-convective-11.json`:

- 2021-05-04
- 2021-06-21
- 2021-08-11
- 2021-12-11
- 2022-04-13
- 2022-06-08
- 2022-06-17
- 2022-07-23
- 2024-05-08
- 2024-05-26
- 2024-06-26

No event is removed because its result is unfavorable. The event is the inferential unit; neural seeds are averaged inside each event.

`2021-06-21`, the physical-feature medoid of the cohort audit, may be used only for smoke tests. It is not a substitute for the final eleven-event evaluation.

## 4. Forecast stage and primary targets

The paper studies the **first 24 forecast hours** of the storm-response trajectory. This is the interval in which opposing interruption and restoration flows can both affect the state while recursive long-tail recovery has not yet become the sole scientific target.

Primary endpoints:

\[
h=6\text{ hours},\qquad h=24\text{ hours}.
\]

The complete 24-hour path is retained in training and reported as a secondary metric. One-hour and 48-hour results from earlier protocols remain in the evidence ledger as boundary diagnostics, but they are not v2 headline claims.

## 5. Proposed model

The proposed model remains the existing neural two-rate dynamics with no semiparametric component and no new memory state:

\[
Y_{t+1}=Y_t+U_\theta(X_t)(1-Y_t)-R_\theta(X_t)Y_t.
\]

`U_theta` and `R_theta` are two nonnegative bounded neural networks. The state equation, input channels, capacities, and parameterization remain unchanged.

The theory-matched structural comparator is `net_scaled`, a parameter-matched single signed neural rate that must choose one direction at a time while retaining the same state-dependent scaling.

No gated damage head, recovery-burden state, distributed-lag module, recurrent network, transformer, or semiparametric term is part of v2.

## 6. Training objective

Training is event balanced. For training event \(e\), define the masked rollout loss over the 24-hour path

\[
L_{\mathrm{roll},e}
=
\frac{\sum_{i,t\le24}M_{eit}(\widehat Y_{eit}-Y_{eit})^2}
{\sum_{i,t\le24}M_{eit}}.
\]

Define a teacher-forced one-step loss using the observed current state in the rate equation:

\[
L_{\mathrm{step},e}
=
\frac{\sum_{i,t\le24}\widetilde M_{eit}
(\widehat Y^{\mathrm{TF}}_{ei,t+1}-Y_{ei,t+1})^2}
{\sum_{i,t\le24}\widetilde M_{eit}}.
\]

The fixed training objective is

\[
\boxed{
L_{\mathrm{train}}
=
\frac1{|\mathcal E_{\mathrm{train}}|}
\sum_{e\in\mathcal E_{\mathrm{train}}}
\left[
\frac12L_{\mathrm{roll},e}
+
\frac12L_{\mathrm{step},e}
\right].
}
\]

The one-step term is not an auxiliary story: it is the empirical conditional-transition risk that appears in the identification and projection theorems. Its coefficient is fixed at 1/2 and is not tuned.

## 7. Validation and optimization

Each outer fold holds out one entire test event. Two validation events are selected without outcome or model-result information using physical weather, footprint, and coverage features. The fixed map is `configs/event_split_map_g2_two_validation.json`. The remaining eight events train the model.

For validation event \(e\), define

\[
L_{\mathrm{val},e}
=
\frac12L_{\mathrm{path24},e}
+
\frac14L_{6,e}
+
\frac14L_{24,e}.
\]

The checkpoint score is the equal-event mean over the two validation events. This replaces early stopping on a single chronological neighbor, which selected the previous two-rate model at a median of only three epochs.

Shared optimization settings for all neural arms:

- Adam, learning rate `3e-3`
- epoch cap `60`
- patience `12`
- batch size `512`
- identical initialization calibration
- identical masks, rows, normalization, and checkpoint rule

All learned preprocessing and normalization use training events only.

## 8. One primary experiment

Run the eleven leave-one-event-out folds with three neural seeds. Compare exactly four methods:

1. `two_rate_v2` — proposed neural two-rate model and objective;
2. `net_scaled_v2` — parameter-matched structural comparator trained with the same objective;
3. `hgb_same_information` — strong direct-regression baseline using the same available information;
4. `damped_persistence` — simple operational baseline.

The main table reports equal-event RMSE at h+6 and h+24, the complete event-level paired differences, event bootstrap intervals, exact paired randomization, sign counts, and leave-one-event influence.

No method is required to win every event. The primary structural claim is supported when the equal-event two-rate advantage over `net_scaled` is positive with a bootstrap interval above zero and a randomization p-value below 0.05. Sign counts remain visible but are not used as an additional arbitrary veto once the inferential criterion is met.

## 9. Ablations

Only two ablations are authorized.

### A1. Structural collapse

`two_rate_v2` versus `net_scaled_v2`. This is also the theorem-matched baseline and measures the cost of collapsing concurrent opposing flows into one signed projection.

### A2. Remove the theorem-aligned one-step term

Train the identical two-rate architecture under event-balanced rollout-only loss:

\[
L_{\mathrm{train}}^{\mathrm{roll-only}}
=
|\mathcal E_{\mathrm{train}}|^{-1}
\sum_e L_{\mathrm{roll},e}.
\]

This determines whether directly estimating conditional transitions improves transfer and optimization. No coefficient sweep is allowed.

The previous one-validation, pooled-row protocol is retained only as historical evidence and is not rerun as a third ablation.

## 10. Pilot and promotion rule

Before the full three-seed experiment, run all eleven outer folds with seed 0 for `two_rate_v2`, `net_scaled_v2`, and `two_rate_rollout_only`.

The protocol is promoted unchanged when:

- all integrity and leakage tests pass;
- median selected epoch is not concentrated at epoch 1;
- the proposed model has finite, nondegenerate predictions on every event;
- the h+6/h+24 equal-event results can be computed and reconciled.

Promotion does not depend on obtaining a positive score. A negative pilot changes the scientific conclusion, not the frozen protocol.

## 11. Paper contribution hierarchy

The paper will present:

1. **Core theorem:** a collapsed one-rate model is an event-distribution-dependent projection, with an exact target oracle gap and event-shift penalty; a minimax corollary gives positive worst-event projection regret when event projection ratios differ.
2. **Supporting identification lemma:** conditional state variance exactly controls local separation of the two rate functions.
3. **Empirical study:** one event-held-out 24-hour forecasting experiment tests whether the structural advantage survives estimation and misspecification; the rollout-only and one-rate variants are the only ablations.

There is no third independent model mechanism. The neural two-rate architecture is deliberately simple so the experiment tests the theorem rather than an accumulation of engineering modules.
