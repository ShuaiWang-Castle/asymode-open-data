# Online cross-county two-flow confirmation lock

**Written before computing this estimator on any 2022 or 2024 event.**  
**Public data:** `main@8dd47c5ccd829611f27b69a3d64c274a0a24c400`.  
**Manifest:** `g3-all-26`, digest `db286b4960a4`.

## Motivation

The conservation preflight distinguishes two scales. Highly local cells usually cannot afford a second flow, whereas a globally pooled constant second flow is estimable but misspecified. A storm supplies an intermediate and operationally meaningful scale: before issuing a forecast, observed counties under the same storm provide a cross-section from which a low-dimensional interruption/restoration law can be estimated, and the law can then be applied to held-out counties.

This experiment therefore tests the paper's central claim without requiring a universal rate function to transfer unchanged across unrelated storms:

> At a fixed storm and forecast origin, does cross-county pooling support two opposing flows better than the same exogenous partition restricted to one signed flow?

## Chronology and evidence split

- Development and all model choices use only the fourteen events from 2018--2021.
- Confirmation uses the twelve events from 2022 and 2024 exactly once.
- Event is the inferential unit. Confirmation events, county outcomes, and model results are not inspected before this lock is committed.

## Forecast task

For each event:

1. Define the NOAA county-footprint peak exactly as in the conservation preflight, using the public-weather tie break.
2. Set the forecast origin to `peak - 12 hours`.
3. Require the full 24-hour target to lie inside the published panel; otherwise mark the event unavailable without moving the origin.
4. Deterministically split counties by SHA-256 of `(20260904, FIPS)`: 80% source counties and 20% held-out counties. All rows of a county remain on one side.
5. Use source-county transitions from panel start through the forecast origin only. No transition after the origin enters fitting, clustering, normalization, model selection, or damping.
6. Initialize held-out counties at their observed state at the origin and predict 24 hours open loop using public future weather.

This is storm-conditioned nowcasting with spatial holdout. It is not presented as forecasting an unseen storm without same-storm observations.

## Exogenous feature map

Use exactly 24 features:

- 12 public hourly ERA5 county channels;
- trailing 6-hour maxima of gust, wind speed, and CAPE;
- trailing 12-hour sums of precipitation and snowfall;
- phase relative to the NOAA footprint peak divided by 24;
- UTC clock sine and cosine;
- log land area, county-neighbour degree, latitude, and longitude.

No outage history enters the partition. No 2023 EIA statistic, 2024 customer total, family label, event label, or outcome-derived information score is a feature.

## Locked estimator

For each event independently:

1. Standardize the 24 exogenous features on source-county pre-origin transitions.
2. Fit `MiniBatchKMeans(K=8, random_state=0, n_init=5, batch_size=4096, max_iter=200)`.
3. In each cluster fit the exact constrained two-flow least-squares model
   \[
   \Delta=U_k(1-Y)-R_kY,
   \qquad 0\le U_k\le0.265,
   \quad 0\le R_k\le0.25.
   \]
4. Fit the matched one-flow comparator in the identical cluster by selecting the better constrained ray
   \[
   \Delta=a_k(1-Y),\ a_k\ge0,
   \qquad\text{or}\qquad
   \Delta=-b_kY,
   \ b_k\ge0.
   \]
5. Use the fitted hourly rates without post-fit multiplicative calibration or target-event tuning.

`K=8` was selected from `K in {1,2,4,8,16}` on 2018--2021. It had the lowest equal-event two-flow path MSE, a positive two-flow/one-flow path difference on 11 of 14 development events, and an exact event sign-flip p-value of approximately 0.032. The full development table, including every screened K, must be disclosed.

## Locked endpoints

For each confirmation event:

- primary structural endpoint: held-out-county 24-hour path MSE difference
  \[
  d_e^{\mathrm{path}}=\operatorname{MSE}_{1,e}^{1:24}-\operatorname{MSE}_{2,e}^{1:24};
  \]
- theory-aligned transition endpoint: teacher-forced one-step MSE over held-out counties and the same 24 future transitions;
- secondary lead endpoints: h+6, h+12, and h+24 MSE;
- treatment diagnostics: cluster counts, `U_k`, `R_k`, one-flow branch, local plug-in `Gamma`, `c_k=min(U_k,R_k)`, and RMS `c_k(1-2Y)`;
- optimization is absent: all fits are deterministic convex or exact finite candidate calculations.

Report all event differences, equal-event mean, median, exact two-sided sign-flip test, 50,000-event bootstrap interval, sign count, and leave-one-event influence. Row-level tests are prohibited.

## Baselines

On the same held-out counties and origin, report:

- persistence;
- source-fitted damped persistence;
- exact event-level constant one-flow and two-flow models fitted on the same pre-origin source transitions;
- recursive HistGradientBoosting trained on the identical source-county pre-origin rows and 24 inputs plus current state;
- direct h+24 HistGradientBoosting using the current state and prespecified summaries of the same exogenous future path.

No baseline hyperparameter is tuned on confirmation events.

## Decision rule

The structural result is supported only when:

1. equal-event mean `d_path` is positive;
2. exact sign-flip p-value is below 0.05;
3. at least 8 of 12 available events have positive `d_path`;
4. no single-event removal changes the equal-event mean to non-positive;
5. the teacher-forced transition difference is non-negative in equal-event mean.

Practical competitiveness additionally requires the two-flow path MSE to be lower than persistence and recursive HGB in equal-event mean. Direct h+24 HGB is reported as a separate point-prediction reference and is not required for the structural claim.

## Hard stop

This exact confirmation is run once. No change to `K`, origin, county split, feature set, cap, endpoint, baseline, or event set is permitted after the confirmation tables are materialized. Any later development requires newly frozen storms.