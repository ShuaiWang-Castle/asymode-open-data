# Pre-registration: does the paper's own index explain the weak empirical result?

Written before `Γ` has been computed on any real panel. Nothing below is a finding.

## Why this exists

The manuscript asserts that its selection index organises the empirical outcome:

> "This organization accommodates the real-data result without weakening the
> theorem: a positive oracle gap can coexist with weak or heterogeneous
> predictive gains." (`sections/05_discussion.tex`)

That is an *explanatory claim about the data*, and the repository does not
currently contain the measurement behind it. `results/d6_information_geometry_*`
reports `v`, `A`, `B`, `lambda_min` and `N*v`, but never `kappa`, never `G_n`,
and never `Gamma_n`. The paper therefore explains a null with a quantity it
never evaluated. This file fixes what will be computed and how it will be read,
so that the answer cannot be chosen after seeing it.

## The estimator

For each driver-space neighbourhood cell built exactly as in
`experiments/d6_information_geometry.py` (training rows of the pinned outer
split, k-nearest neighbours in a training-fitted 5-component PCA space), let
`Y` be the origin state and `Z` the observed one-step change at the origin.
Within the cell, fit the two-flow conditional mean by nonnegative least squares

    Z ~ U (1 - Y) + R (-Y),      U >= 0, R >= 0,

giving `U_hat`, `R_hat` and residual variance `sigma2_hat` on `n - 2` degrees of
freedom. With `A = E[(1-Y)^2]`, `B = E[Y^2]`, `v = Var(Y)` from the same cell,

    kappa_hat = min( R_hat^2 / A , U_hat^2 / B )
    G_hat     = v * kappa_hat
    Gamma_hat = n * G_hat / sigma2_hat

This is the plug-in of Theorem 1 and Proposition 2. It inherits their scope: it
is a local fixed-design benchmark, not a neural generalisation theorem.

## Hypotheses

**H-G1 (level).** The panels sit in the regime the theory calls one-flow
favourable: the median `Gamma_hat` over cells is below 1.

**H-G2 (ordering).** The index has explanatory content across events: the
event-level median `Gamma_hat` is positively rank-correlated with the event-level
two-flow gain already archived in
`results/event_transfer_confirmatory_20260903/04_CORE_EVENT_RESULTS.csv`.

## Kill conditions, fixed now

* **H-G1 fails if the median `Gamma_hat` exceeds 1.** Then the theory predicts
  two flows *should* pay on these panels, the observed gain is weak anyway, and
  **the paper may not claim that its index explains the weak result.** The
  discussion must instead say that the index does not account for it and name
  the untested alternatives (pooling misspecification, rollout propagation,
  estimation noise).
* **H-G2 fails if the Spearman correlation is <= 0**, or if its sign is not
  stable across the k in {50, 200, 800} neighbourhood rules. A near-zero
  correlation is the more damaging outcome: it means the index separates
  regimes in theory but carries no information about where two flows actually
  helped, and the explanatory sentence in the discussion must be withdrawn
  rather than softened.
* Cells flagged by the existing rule (`n < 30` or `v < 1e-8`) are excluded, as
  in D-6. The exclusion is inherited, not introduced here.
* Both hypotheses are reported for every neighbourhood rule. Reporting only the
  rule with the strongest correlation is forbidden.

## The limitation that must be stated with the result

The eleven event-level gains were known to the analyst before this file was
written; they are already in the manuscript's appendix. `Gamma_hat` is a
property of the observed states and drivers alone -- it involves no fit of the
neural arms and no test-set outcome -- so the *computation* is blind to them,
but the *choice to run this test* was not. The correlation in H-G2 is therefore
an out-of-sample claim about the index only in the sense that the index was
never tuned to the gains; it is not a pre-data prediction, and the paper must
describe it that way.

`n = 11` events. This test has low power and cannot establish absence of a
relationship; it can only detect a strong one.

---

## Addendum, written before computing: H-G3, the threshold test

H-G2 treats `Gamma` as a continuous predictor across eleven events. That is the
weaker reading of the theory. The theory's actual statement is a **threshold**:
two flows are favoured exactly when `Gamma_n > 1`. The corresponding test is a
regime split, and it can be run at the level of individual forecast cells rather
than eleven event aggregates.

**Construction.** For each outer fold, the neighbourhood geometry is fitted on
training rows only, exactly as in D-6/D-8. `Gamma_hat` is then evaluated **at the
driver location of each held-out test row**, using the `U_hat`, `R_hat`,
`sigma2_hat`, `v`, `A`, `B` of its nearest training neighbours. This is the
cross-fitted selector the discussion defers ("A cross-fitted empirical
`Gamma_hat` could eventually be tested as a model selector"). It uses only the
archived predictions in
`results/event_transfer_confirmatory_20260903/predictions/` and the public
panels; nothing is retrained.

**H-G3.** Among test cells with `Gamma_hat > 1`, the two-flow model has lower
squared error than the parameter-matched one-flow model; among cells with
`Gamma_hat < 1`, it does not.

**Kill conditions, fixed now.**

* H-G3 fails if the two-flow win rate in the `Gamma_hat > 1` group is not higher
  than in the `Gamma_hat < 1` group, at both primary horizons.
* It also fails if the difference between the two groups is not stable in sign
  across the k in {50, 200, 800}.
* A failure here is more consequential than H-G2's. H-G2 asks whether the index
  ranks events; H-G3 asks whether the index's own threshold identifies the cells
  where the extra flow pays. **If H-G3 fails, the paper may not present
  `Gamma_n` as a practical selection rule, and the sentence deferring a
  cross-fitted selector must say that the obvious version of it was tried here
  and did not separate the regimes.**
* Cells are weighted equally. The `Gamma_hat > 1` group is expected to be a
  minority; its size is reported, and if it falls below 200 cells at any rule the
  result is reported as underpowered rather than as a verdict.

---

## Addendum, written before computing: H-G4, the affected-subset prediction

The target is exactly zero in roughly half of all scored cells. Theorem 1 makes a
directional prediction about what happens when those cells are removed:
`G = v * kappa`, and both factors should rise on affected data -- `v` because the
state stops being pinned at zero, `kappa` because a county that never loses power
has `U = 0` in that neighbourhood and therefore `G = 0` exactly.

**This is a restriction defined on the observed state alone.** It uses no model
output, no error, and no fitted quantity. It is applied identically to both arms.

**H-G4.** Restricting evaluation to affected cells increases the two-flow
advantage over the parameter-matched one-flow model, and the increase is monotone
in how affected the stratum is.

Two stratifications, both fixed now:

* **Cell level** -- by the observed target at the scored horizon:
  `y = 0`, `0 < y <= 0.01`, `0.01 < y <= 0.05`, `0.05 < y <= 0.15`, `y > 0.15`.
* **County-event level** -- by the peak observed `y` of that county over the
  forecast window: `peak = 0`, `0 < peak <= 0.02`, `0.02 < peak <= 0.10`,
  `peak > 0.10`. This is the "only predict the affected ones" reading.

**Kill conditions, fixed now.**

* H-G4 fails if the two-flow advantage does not increase from the zero/near-zero
  strata to the most affected stratum, at both primary horizons.
* It also fails if the ordering is not monotone across the intermediate strata,
  since the theory's prediction is monotone in `v`, not merely different at the
  extremes.
* Strata with fewer than 300 scored cells are reported but excluded from the
  monotonicity verdict as underpowered.
* **Restricting evaluation changes the estimand.** Whatever the outcome, any
  number computed on a subset is reported as conditional on that subset and never
  substituted for the full-cohort result in the main table. The full-cohort
  number stays primary.
* The complementary risk is stated with the result: conditioning on a large
  observed outage selects cells where the state has moved, which mechanically
  favours any model with more state-dependent flexibility. That confound is not
  removed by this test and must be named.
