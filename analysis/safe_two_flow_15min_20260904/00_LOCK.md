# Safe 15-minute two-flow mixture — locked evaluation

This file is committed before the safe-mixture estimator is evaluated on 2022/2024. The same public data and 15-minute online cross-county task as `analysis/online_cross_county_15min_formal_20260904/00_FORMAL_LOCK.md` are used.

## Only additional operation

The dynamic path is mixed with persistence by a scalar weight selected without test-county or post-origin information.

For each event, deterministically partition counties by the SHA-256 score of `(20260904, FIPS)` into:

- 64% rate-fitting counties;
- 16% deployment-validation counties;
- 20% held-out test counties.

Let the forecast origin be 12 hours before the outcome-blind NOAA footprint peak. Six hours before the origin, fit the `K=8` 15-minute model on rate-fitting counties using only still-earlier transitions. Roll that model over the next six hours on deployment-validation counties and choose

\[
w=\Pi_{[0,1]}
\frac{\sum(\widehat Y^{\rm dyn}-Y^{\rm pers})(Y- Y^{\rm pers})}
{\sum(\widehat Y^{\rm dyn}-Y^{\rm pers})^2}.
\]

Then refit the exact `K=8` one-flow and two-flow models on both fitting and deployment-validation counties using all transitions available before the forecast origin. For each arm, produce its 24-hour dynamic path and report

\[
\widehat Y_h^{\rm safe}
=Y_0+w(\widehat Y_h^{\rm dyn}-Y_0).
\]

The same scalar `w`, selected from the two-flow development trajectory, is applied to both structural arms. Thus the safety layer cannot create a one-flow/two-flow difference; it only controls how much of either dynamic forecast is deployed.

## Fixed configuration

- native 15-minute target;
- 24-hour horizon, 96 steps;
- origin = NOAA peak minus 12 hours;
- six-hour pre-origin deployment-validation window;
- 24 exogenous features from the prior lock;
- `K=8` MiniBatchKMeans;
- caps `U<=0.06625`, `R<=0.0625`;
- no post-test calibration and no hyperparameter search.

The six-hour validation window was chosen on 2018--2021 from the prespecified set `{3,6,12}` because it gave the strongest event-level two-flow/one-flow evidence: positive on 12 of 14 development events with an exact sign-flip p-value about 0.014. All three windows and their full development results must be reported.

## Endpoints and gates

Primary structural endpoint is the held-out-county 24-hour safe-path MSE difference. Practical endpoint is the safe two-flow path MSE relative to persistence. Event is the inferential unit; report all event differences, exact sign-flip tests, 50,000-event bootstrap intervals, and leave-one-event influence.

The structural gate requires positive mean, exact p<0.05, at least 8 positive available events, and positive leave-one-event means. Practical competitiveness requires positive equal-event mean versus persistence; statistical significance versus persistence is reported but is not required to retain the structural finding.

After the evaluation, no alternative validation length, mixture formula, cluster count, or county split may be tried on the same 2022/2024 events.