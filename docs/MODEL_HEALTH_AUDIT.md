# Model health audit — what the data says about the fitted model

Adversarially verified: every finding below was produced by one agent computing on
the data and then re-checked by a second agent that re-ran the computation and was
instructed to refute it. 47 findings survived; the rest were discarded. Severity is
the corrected severity assigned by the verifier, not by the finder.

Panels `76a73ed794af`, channels `dec964873cb2`, the eleven `g2-convective-11`
events. Nothing here required retraining a neural model.

## Clean bill of health: the load-bearing assumption holds

**Assumption 1 — the drift is affine in the state — survives a direct test.**
Under wild cluster bootstrap inference (Rademacher, null imposed, B = 1999, event
clusters), no departure from affineness in `y` is detectable in any geometry
tested: the `y^2` term gives p = 0.46 / 0.82 / 0.78 and a five-hinge spline gives
p = 0.17 / 0.41 / 0.51 at ~200 rows/cell, ~800 rows/cell and pooled. County
clustering (1,566 clusters) agrees: `y^2` coefficient −0.0425, t = −1.05.

A more flexible model does look better in sample — median RSS reduction 8.16% at
k = 200, F-test p < 0.01 in 60% of neighbourhoods — but that is overfitting. Held
out *inside the same neighbourhood*, the flexible model wins in only 15.4%
(k = 200) and 29.4% (k = 800) of cases, with median out-of-sample MSE 15% worse.
The heteroskedasticity-robust rate is 4.1%, against 51% for the naive F-test.

**Methodological trap, recorded so it is not repeated.** The naive event-clustered
asymptotic Wald test on this question has roughly 50% size at nominal 5% with
eleven clusters, and it reports p = 1.24e-05 on the real data. Anyone running it
without calibration would falsely reject the paper's own assumption.

**Scope.** Only 2.81% of rows have `y > 0.1`; mean `(1-y)` is 0.990. The minimum
detectable curvature is 9.6% of the affine drift at `y = 0.05` and 162% at
`y = 1`. The honest phrasing is "no departure detectable at the scale the data can
resolve", with the detection limit attached — not "Assumption 1 holds". These
panels do not validate the susceptible-fraction mechanism specifically, only the
affine line, which damped persistence also produces.

## The cross-county share is a degrees-of-freedom artefact

The published figure — an overall cross-county share of 0.78 with event-cluster
95% interval [0.75, 0.82] — reproduces arithmetically but sits **below its own
county-label permutation null**: observed 0.7848 against permuted 0.7939,
difference −0.0091 with 95% [−0.0230, +0.0057], and only 9 of 26 events above
their own null. The cause is arithmetic: at k = 200 the median neighbourhood holds
**1.20 rows per county**, so a between-group share is nearly forced. The
noise-corrected variance component is 0.0197 mean, −0.0010 median.

At k = 800 (1.66 rows/county) there is a real but small effect: 0.5576 against a
permuted 0.4976, +0.0600 [+0.0517, +0.0696], 26/26 events positive.

**The defensible range is 0.02–0.15, not 0.78.** Pooled homogeneity is better
supported than the draft claims, not worse. The per-family "cross-county" column is
rank-monotone in the median number of distinct counties per neighbourhood
(171 > 165 > 152 > 150 > 147) and sits below its permutation null for four of five
families; it measures county count, not heterogeneity.

## The local Gamma plug-in is an estimator artefact

`experiments/d8_gamma_index.py` reports a median `Gamma` far below 1, driven by
`kappa` being exactly zero in 46.9% of k = 200 neighbourhoods. **Those zeros are
noise, not structure.** Of the cells with `U_hat == 0`, 94.2% have an
unconstrained estimate with `|t| < 2`; only 2.97% of all cells have `U`
significantly negative under HC3, against 35.3% with a negative point estimate.
The nonnegativity constraint is absorbing negative noise draws.

`kappa`'s binding branch is `U^2/B` in 73–86% of cells, and `U_hat^2` is smaller
than its own sampling variance in half to two thirds of them. Subtracting the
estimator noise floor drives the median `Gamma` to exactly zero at every
bandwidth. **A statement of the form "`G = 0` structurally in ~45% of
neighbourhoods, so one flow is exactly sufficient there" is not supported.** The
plug-in is too noisy at these neighbourhood sizes to carry that claim.

## Where the two-flow advantage actually lives

Equal-event mean gain of two flows over the parameter-matched one flow, event
holdout, stratified by the state at the forecast origin:

| stratum | h+24 | h+48 |
|---|---|---|
| `y0 = 0` | **+4.70%** | +1.36% |
| `0 < y0 <= 0.01` | **+5.67%** | +4.69% |
| `y0 > 0.01` | **−5.97%** (4/11) | **−8.90%** (5/11) |

County holdout on `y0 > 0.01`: −4.62% at h+24 (3/11), −7.39% at h+48.

**The advantage lives where the two-flow structure degenerates to one flow.** At
`y0 = 0` the restoration term `R*y` is identically zero, so the model *is*
single-flow there. On rows where both flows are genuinely active the sign
reverses. Any mechanism sentence claiming that concurrent live rates are what buys
the gain is contradicted by the run's own predictions.

**The comparator is boundary-degenerate on 41% of the evaluation set.** The
one-flow arm emits exactly 0.0 on 41.2–42.5% of masked cells — every one of them a
`y0 == 0` row — because a nonpositive signed rate at `y = 0` freezes the trajectory
under `clip(., 0, 1)`. The two-flow arm emits exact zero on 0.00–0.01%. On those
rows the comparison is two flows against a hard clip. This may be the intended
structural point, but it has to be stated, or the gap reads as a clipping artefact.

## Long-horizon performance against trivial baselines

Equal-event mean skill against the all-zero predictor: **+0.0031 at h+24 (7/11
events) and −0.0197 at h+48 (5/11)**. Pooled framing hides this — pooled, the model
beats all-zero by 3.1% at h+48. On pooled RMSE there is no horizon at which the
neural model is best of four: damped persistence wins h+1 and h+6, histogram
gradient boosting wins h+24 and h+48.

At one step, a **two-parameter affine map in `y` with no drivers at all**, fitted
leave-one-event-out, achieves RMSE 0.011787 — beating all three two-flow seeds and
significantly beating both the one-flow arm and gradient boosting.

**The long-horizon failure is a ranking failure, not calibration and not range.**
An oracle transform fitted on the test set buys at most 2.8% RMSE and lifts h+48
R-squared only from 0.016 to 0.052. The oracle affine slope is below 1 at both long
horizons, so the observed range compression (sd ratio 0.315 at h+48) is close to
optimal shrinkage given the correlation, not a defect a variance-matching loss
would repair.

**What the error actually is: onset placement.** Cells that start near zero and
later exceed 0.02 carry 55.4% (h+24) and 60.4% (h+48) of all squared error, and the
model recovers about 15% of the level on them — while the aggregate onset *amount*
is roughly right (0.78x). `U0(x)` gets the amount right and the timing wrong. Under
the affine form, onset at `y ~ 0` is a pure function of `x` with no state pathway
available to help.

**A second methodological trap.** A bias table stratified by decile of the observed
target looks catastrophic (top-decile bias −0.118 at h+24) but is regression to the
mean: an oracle-isotonic control with zero calibration error by construction
reproduces it at −0.121, and damped persistence gives −0.126. Use a reliability
diagram instead, which says the opposite and correct thing.

## The unified pipeline is not a measurement

`results/unified_aistats_v2_final_neural.json` is undertrained to the point of
being uninformative, and should not be cited:

* 62 of 99 fits select epoch 1 or 2. The loop takes one optimiser step per training
  event per epoch, so the median selected checkpoint is **16 gradient steps** for
  the two-flow arm and **8** for the one-flow arm. The epoch cap never binds
  (0/99); patience terminates every fit.
* **Neither arm beats its own two-scalar initialisation.** A constant-rate rollout
  built from the same calibrated `(u0, r0)` the network starts from is on average
  better: the two-flow arm is −1.05% against it at h+24. The fourteen driver
  channels are not measurably used.
* Validation is evaluated only after the first optimiser step, so the
  initialisation is never a candidate checkpoint. 10 of 66 fits select a checkpoint
  worse *on their own selection criterion* than epoch 0; 20 of 66 are worse on test.
* **The structural effect is the size of seed noise.** Two-way ANOVA on log RMSE:
  F_arm = 0.83 at h+6 and 1.01 at h+24. The arm explains 0.01–0.03% of variance,
  the seed 0.71–1.44% — twenty-four to forty-eight times more.
* The headline changes sign with the seed (−6.82 / −0.10 / +0.17% single-seed) and
  is decided by **one fit out of ninety-nine**: dropping the two-flow arm's
  2022-04-13 seed-0 fit moves h+24 from −2.24% to −0.07%. That fit's validation
  score is 0.7% worse than its siblings while its test RMSE is 70% worse, so the
  checkpoint rule cannot see the divergence.
* Its single declared ablation is a no-op: the joint and rollout-only arms produce
  essentially identical fits.

The two pipelines report the same contrast with **opposite signs** on the same
eleven events, same digests and same seeds, and the direction tracks how much
optimisation each performed (median gradient steps: unified 8–16, confirmatory
event split 108–228, confirmatory county split 768–1147).

## Reproduction

The confirmatory event-held-out run reproduces exactly. Re-running
`experiments/cc_event_transfer.py` from scratch returns, to every printed digit,
the archived two-flow and one-flow RMSE at h+24 and h+48, the relative gains, the
positive-event counts, and all eleven event-level gains.
