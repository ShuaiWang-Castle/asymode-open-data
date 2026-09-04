# Fifteen-minute online cross-county flow confirmation

**Lock:** committed before this estimator is evaluated on any 2022 or 2024 event.  
**Data:** `main@8dd47c5ccd829611f27b69a3d64c274a0a24c400`; manifest digest `db286b4960a4`.

## Motivation

The hourly preflight finds that interruption and restoration are often separated across hourly cells. The public source state is natively observed on a 15-minute grid, so an hourly aggregation may erase short-lived coexistence while reducing the number of repeated cross-county transitions. This confirmation changes only the temporal observation resolution; the state equation and one-flow/two-flow classes are unchanged.

## Evidence chronology

- Development: the fourteen 2018--2021 events.
- Confirmation: the twelve 2022 and 2024 events, evaluated once after this lock.
- No 2022/2024 result from this 15-minute estimator has been computed before the lock.

## Task

For each event independently:

1. Define the NOAA footprint peak using the same outcome-blind catalogue and public-weather tie-break used by the conservation preflight.
2. Set the forecast origin to 12 hours before that peak and forecast 96 native 15-minute steps (24 hours). An event whose horizon exceeds the panel is reported unavailable and is not shifted.
3. Split counties by deterministic SHA-256 of `(20260904, FIPS)`, retaining 80% source counties and 20% held-out counties.
4. Fit exclusively on source-county transitions before the forecast origin. A transition is legal only when both native 15-minute endpoint states are observed and finite.
5. Assign each 15-minute transition the public hourly exogenous vector of its containing hour. The feature map is the same 24-variable map as the hourly online experiment: 12 ERA5 channels, five causal weather-history summaries, NOAA-peak phase, UTC clock sine/cosine, log land area, neighbour degree, latitude, and longitude.
6. Standardize on source-county pre-origin rows and fit `MiniBatchKMeans(K=8, random_state=0, n_init=5, batch_size=8192, max_iter=200)`.
7. In each cluster fit exact constrained two-flow and matched one-flow models. Fifteen-minute caps are the hourly caps divided by four: `U<=0.06625`, `R<=0.0625`.
8. Roll held-out counties open loop for 96 steps; no post-fit scale, damping, or target-event tuning is allowed.

`K=8` was selected from `K={1,2,4,8,16}` on development data. The 15-minute model gave a positive 24-hour path difference on 13 of 14 development events, an equal-event mean path difference of approximately `3.06e-05`, and an exact event sign-flip p-value of approximately `0.031`. The teacher-forced per-quarter-hour difference was positive in mean but not statistically resolved. Every screened K and all development events must be reported.

## Endpoints

Primary:

\[
d_e^{24\mathrm{h}}=\operatorname{MSE}^{96\mathrm{\ steps}}_{1,e}-\operatorname{MSE}^{96\mathrm{\ steps}}_{2,e}.
\]

Secondary:

- native 15-minute teacher-forced transition MSE;
- one-hour teacher-forced block forecast, obtained by initializing from an observed state and rolling four fitted steps;
- lead 6, 12, and 24 hours;
- persistence and source-fitted damped persistence;
- cluster counts, rates, plug-in Gamma, common flow, and delivered transition difference.

The event is the inferential unit. Report equal-event mean, median, exact two-sided sign-flip test, 50,000-event bootstrap, positive-event count, and leave-one-event influence.

## Confirmation gate

The primary 24-hour path claim requires:

1. positive equal-event mean difference;
2. exact sign-flip p-value below 0.05;
3. at least 8 positive available events;
4. positive mean under every leave-one-event deletion;
5. the 15-minute transition mean difference and one-hour block mean difference are not both negative.

Practical performance against persistence is reported but is not part of the structural gate.

## Hard stop

Run once; do not change the resolution, cluster count, origin, county split, cap, feature set, or endpoint after reading confirmation results. A failure requires newly frozen events for any revised estimator.