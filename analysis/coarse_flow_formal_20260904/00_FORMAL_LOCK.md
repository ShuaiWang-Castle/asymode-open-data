# Coarse two-flow temporal confirmation lock

**Lock status:** written before any 2022/2024 performance result from the candidate model is computed.  
**Public data:** `main@8dd47c5ccd829611f27b69a3d64c274a0a24c400`.  
**Design-audit reference:** `5f3ff3aa616fac7540eae45415089127816f3199`.  
**Panel manifest:** `g3-all-26`, digest `db286b4960a4`.

## Scientific question

The preflight shows that a global two-flow coefficient is estimable, while a highly local second flow is unaffordable in most `k=200` neighbourhoods. This experiment tests the intermediate regime predicted by that scale dependence:

> Does a deliberately coarse exogenous partition retain enough pooling for interruption and restoration to outperform the identical partition constrained to one signed flow, and does that transition-level gain improve a 24-hour open-loop forecast on future storm events?

The method is a statistical sieve, not a neural network. It changes only the effective pooling scale; it does not add a new physical mechanism.

## Data chronology

- **Exploratory fitting:** events in 2018--2020.
- **Candidate selection:** all six 2021 events.
- **Locked confirmation:** all twelve 2022 and 2024 events. No 2022/2024 outcome, loss, rate estimate, or model comparison may be inspected before this file is committed.
- For the outcome-blind active-48 design, an event is unavailable when the fixed 48-transition NOAA-centred window lies outside the published panel. Such an event remains reported as unavailable and is not shifted or replaced. Full-panel secondary results retain all twelve confirmation events.

## Transition construction

1. Aggregate the 15-minute state to hourly by averaging observed substeps only.
2. Retain `t -> t+1` only when both hourly states are observed and finite.
3. Align weather to `t+1`.
4. Exclude no row by outage state, sign, residual, prior result, or fitted information score.
5. Use the active-48 window already defined in the conservation preflight: 24 transitions before and 24 after the outcome-blind NOAA county-footprint peak, with the fixed public-weather tie break.

## Locked feature map

Each county-hour is represented by exactly 24 exogenous features:

- the 12 public hourly ERA5 county channels;
- trailing 6-hour maxima of gust, wind speed, and CAPE;
- trailing 12-hour sums of precipitation and snowfall;
- phase relative to the NOAA footprint peak, divided by 24;
- UTC clock sine and cosine;
- four time-stable geographic descriptors from `county_statics`: log land area, county-neighbour degree, latitude, and longitude.

No EIA-2023 reliability variable, 2024 customer total, outage-history summary, target state, event label, family label, or learned embedding enters the partition.

## Locked partition and estimators

1. Standardize the 24 features using fitting events only.
2. Fit `MiniBatchKMeans` with `K=8`, `random_state=0`, `n_init=5`, `batch_size=8192`, and `max_iter=200`.
3. Weight fitting rows so every source event has total weight `1/E`.
4. In each cluster fit the exact box-constrained two-flow least-squares model
   \[
   \Delta = U_k(1-Y)-R_kY,\qquad 0\le U_k\le0.265,\;0\le R_k\le0.25.
   \]
5. Fit the matched one-flow comparator in the same cluster and with the same weights by taking the better exact constrained ray
   \[
   \Delta=a_k(1-Y),\;a_k\ge0,
   \qquad\text{or}\qquad
   \Delta=-b_kY,\;b_k\ge0.
   \]
6. No cluster count, feature, cap, weight, rate, or branch is tuned on confirmation events.

`K=8` was selected on the six 2021 events because it gave the best equal-event 24-hour path MSE among the prespecified leakage-safe geographic candidates and was positive for the two-flow/one-flow one-step contrast on all six events. The broader exploratory screen and all candidate results must be disclosed in the final report.

## Locked endpoints

For each held-out confirmation event:

1. **Primary theory-aligned endpoint:** teacher-forced one-step MSE over all legal active-48 transitions,
   \[
   d_e^{\mathrm{step}}=\operatorname{MSE}_{1,e}-\operatorname{MSE}_{2,e}.
   \]
2. **Primary forecast endpoint:** 24-hour recursive path MSE. For every valid origin from peak minus 24 through the peak, initialize with observed `Y_o`, then use only the fitted rates and exogenous future weather. Report
   \[
   d_e^{\mathrm{path24}}=\operatorname{MSE}^{1:24}_{1,e}-\operatorname{MSE}^{1:24}_{2,e}.
   \]
3. **Secondary endpoint:** MSE at lead 24.
4. **Treatment diagnostics:** cluster counts, `Gamma` diagnostics, rates, `c_k=min(U_k,R_k)`, and delivered difference `c_k(1-2Y)`.
5. Full-panel one-step results are a secondary robustness analysis and use all twelve events.

The event is the inferential unit. Report all event differences, equal-event mean, median, sign count, exact two-sided sign-flip test, event bootstrap interval, and leave-one-event influence. No row-level t-test is permitted.

## Locked baselines

- exact global constant two-flow and one-flow fits;
- persistence and a source-fitted damped persistence;
- one-step histogram gradient boosting on the identical 24 inputs plus current state, rolled recursively for the path endpoint;
- direct h+24 histogram gradient boosting using current state and prespecified summaries of the same future exogenous path.

No baseline hyperparameter is tuned on confirmation events.

## Interpretation gates

- The two-flow structural claim requires positive equal-event means for both one-step and path-24, with no single-event removal changing both means to non-positive.
- A positive one-step result without a positive path result supports representation gain but not forecast gain.
- A positive path result without a positive one-step result cannot be attributed to the oracle-gap mechanism.
- The method is practically competitive only if it beats damped persistence and recursive HGB on path-24 and is not worse than direct h+24 HGB by more than 5% in equal-event MSE.
- Failure is reported without changing `K`, features, window, endpoint, event set, or baseline.

## Hard stop

Run this confirmation once. After its tables are materialized, do not alter the model or launch another confirmation on the same 2022/2024 events. Any subsequent model development must use later, newly frozen storm events.