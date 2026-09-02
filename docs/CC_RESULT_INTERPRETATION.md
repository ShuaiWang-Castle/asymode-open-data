# Result interpretation — corrected branch (living document; append, do not rewrite)

Grades here follow the ledger's scale; "descriptive" means computed on legacy
exports (county folds keyed on the model seed) and not confirmatory.

## 1. Event-level re-analysis of the archived exports (E3, descriptive)

`scripts/event_level_review.py`, 11 convective events, seed-averaged OOF
predictions, event-cluster bootstrap B = 2000 (`results/event_level_*_legacy.json`).

**Gradient boosting vs two-rate (same information).**

| h | mean event ΔMSE (trees − two-rate), % of two-rate MSE | events where trees worse | event-cluster 95% | LOO range |
|---|---|---|---|---|
| 1 | +36.6% | 8/11 | **[−4.9e-6, +1.3e-4] — includes 0** | all positive |
| 6 | −5.6% | 4/11 | [−1.1e-4, +2.7e-5] — includes 0 | all negative |
| 24 | −11.5% | 1/11 | **[−3.1e-4, −6.7e-5]** | all negative |
| 48 | −8.9% | 4/11 | **[−2.3e-4, −2.2e-5]** | all negative |

Reading: the trees' long-horizon advantage survives event clustering (24 h and
48 h intervals exclude zero; leave-one-event-out never changes sign). **The
two-rate model's 1 h advantage does not**: it is large in mean, tiny in median
(+5e-6), and its event interval includes zero — it is carried by a few events.
The ledger's "h+1 −21%, 15/15 folds" therefore **downgrades from [B] to
"descriptive, not established at the event level"** pending the corrected run.

**Concurrency rung (net_scaled vs two-rate).**

| h | mean event ΔMSE, % | events where net_scaled worse | event-cluster 95% | LOO range |
|---|---|---|---|---|
| 1 | −2.5% | 3/11 | [−6.8e-6, −9.9e-7] | negative |
| 6 | −1.4% | 7/11 | includes 0 | mixed |
| 24 | +2.1% | 7/11 | [−4.5e-6, +7.2e-5] — includes 0 | positive |
| 48 | **+4.4%** | 7/11 | **[+1.3e-5, +1.2e-4]** | [+3.7e-5, +6.8e-5] |

Reading: the 48 h concurrency advantage holds at the event level, but 4 of 11
events go the other way and three events (2021-05-04, 2024-05-26, 2021-08-11)
carry most of the mean; the paper shows all eleven. 24 h is not established.

## 2. D-6 local information geometry (C1/C2) — run, zero training

`results/d6_information_geometry_{g2,g3}.json`; kNN in a training-fitted PCA(5)
of the 14-channel block at the first forecast step, k ∈ {50, 200, 800}, plus an
8×8 quantile grid; cells = forecast origins; five folds of the pinned event split.

* **The theorem identities hold on the data**: λ_min(Q̂) is within 1–2% of
  var(y|x) in every family and rule (the eigenvalue is *equal* to the variance
  here, not merely within a factor of two, because λ_max ≈ 1 when y is small).
* **Local information is tiny.** Median var(y|x) is 3e-4 to 2e-3 on convective
  panels and 1e-6 to 4e-4 on the other families; N·var at k = 200 is 0.26 on
  convective and 0.003–0.012 elsewhere. The local variance ratio
  Var(R̂)/Var(Û) = A/B is **~700 on convective and 1.6e4–6e4 on winter, tropical,
  flood and wind** (k = 200). This replaces the "93:1 Kish ESS" statement: the
  correct local statement is that restoration is estimated with two to five
  orders of magnitude more variance than interruption, and the ratio is
  family-dependent.
* **C2 — where the dispersion comes from.** The county decomposition is only
  partly estimable: at k = 50 essentially every neighbourhood row is a different
  county (rows per county ≈ 1), so the "within-county" term is zero by
  construction and the cross-county share is 1.00; at k = 200 the share is 0.78
  over 26 events (event-cluster 95% [0.75, 0.82]; 0.84 on g2), at k = 800 it is
  0.44–0.70. **What the paper may say:** repeats of similar drivers within one
  county are rare in these panels, so the state dispersion that identifies the
  two rates comes overwhelmingly from different counties under similar weather;
  this is a fact about the data's design, stated with the common-rate-function
  assumption, not a causal claim.
* **D-6 (ii) — does local information order the families like H-E? NO.**
  Under the interpretation fixed in `docs/THEORY_PLAN.md`: by median λ_min
  and by N·var the order is convective ≫ winter > tropical ≈ flood > wind
  (k = 200; convective > winter > wind > tropical > flood at k = 800), stable
  across kNN rules. The H-E advantage orders tropical > convective > wind ≈ 0
  > winter (reversed). **The identifiability account does not explain the
  family ordering; the family where the two-rate model wins most (tropical) is
  the one where restoration is locally least identifiable.** Per the fixed
  rule, the paper reports phase-family heterogeneity as an empirical result and
  does not call the identifiability theorem its mechanism. C3 (the Theorem 5
  gain score with cross-fitted rates) remains to be computed from the corrected
  exports, but the state-dispersion factor alone already points the other way.

## 3. What changes in the claims (running list)

| claim | status now |
|---|---|
| "effective sample size for the two rates 93:1" | retracted as a precision statement; replaced by the local variance ratio A/B (family-dependent, 7e2–6e4) |
| "separability is supplied mostly by the cross-section" | supported as a data-design fact (cross-county share 0.78–0.84 at k = 200), with the estimability caveat and the common-function assumption |
| "both consequences are visible in the data" (abstract) | rewrite: the zero-state blind spot and the cross-sectional origin of dispersion are visible; the precision statement is the local A/B |
| "phase separation is the mechanism" | not supported by D-6; report as empirical family heterogeneity |
| "two-rate wins 1 h by 21%" | descriptive only; event interval includes zero |
| "trees win 24/48 h" | holds at the event level (descriptive until the corrected run) |
| "concurrency +2% at 48 h" | holds at the event level, 7/11 events, three events carry it |
| "long-horizon loss is not accumulation" | retracted (Prop. 5 counterexample); replaced by the product-sum bound and a residual-decomposition diagnostic to run |
