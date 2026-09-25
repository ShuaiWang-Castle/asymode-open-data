# F1 information audit — parts (c), (d), (e)

Generated 2026-09-25 01:55:03 by `audit_f1.py` (git 3ba37da). Panel `features_w1.npz`, splits `splits_w1.json`, inputs `eih_w1_<v>.npz`: 3098 county-events, 160 features per variant; variants area, pop, quad, mean, pooled, other, quadn. No model was trained; the base is the held-out prediction of runs/geo_weather_20260924/w1_base/fold{fold:02d}/outer.npz. Settings (REG, with its dated revisions) are copied into the JSON next to this file. Parts (a) contrast and (b) redundancy are computed separately.

## (c) Alignment with the base's held-out residual

3098 county-events, 343 fixed-effect blocks (60 singletons), 202 event x state clusters. Residual correlogram: c = 1.00, l = 7 km. Direction +1: higher-information variant above the lower one goes with under-prediction (y - P > 0).

A feature is eligible for a pass only with at least 20 effective clusters in both clusterings; the best eligible feature is shown.

| contrast | features (eligible) | FWER on synthetic fields (homo l / 2l, hetero l / 2l) | null used | best feature | part corr | t | p max-T | p final |
|---|---:|---|---|---|---:|---:|---:|---:|
| C1 quad-pop (sub-grid) | 96 (0) | 0.134 / 0.134 / 0.044 / 0.042 | synthetic field (homo l) |  |  |  |  |  |
| C2 pop-area (exposure weighting) | 160 (12) | 0.092 / 0.112 / 0.030 / 0.031 | synthetic field (homo 2l) | gust_x10*canopy@48 | +0.010 | +0.81 | 0.9950 | 1.0000 |
| C3 pooled-mean (L1g marginal) | 96 (0) | 0.082 / 0.076 / 0.032 / 0.032 | cluster multiplier |  |  |  |  |  |
| C4 quad-pooled (co-location) | 160 (1) | 0.130 / 0.145 / 0.056 / 0.045 | synthetic field (homo 2l) | p_tw-3.0*one@48 | +0.022 | +0.98 | 1.0000 | 1.0000 |
| C5 pooled-mean | other-mean (own beyond donor) | 96 (0) | 0.078 / 0.077 / 0.027 / 0.026 | cluster multiplier |  |  |  |  |  |

**(c) verdict: KILL** — rule: no (contrast, feature) passes at 0.05 after max-T and Bonferroni, in the registered direction, under the null the rule selects.

Hour level (secondary; unit- and lead-demeaned cells; Bonferroni over four contrasts; best feature with at least 20 effective event x state clusters):

| contrast | best feature | eff. clusters | part corr | t (event x state) | p final | t (event) | p final, exact event flip |
|---|---|---:|---:|---:|---:|---:|---:|
| C2 pop-area (exposure weighting) | p_tw+0.0*one@3 | 32 | +0.0093 | +1.30 | 1.0000 | +1.69 | 1.0000 |
| C4 quad-pooled (co-location) | gust_x15*one@0 | 35 | +0.0046 | +2.03 | 1.0000 | +2.06 | 1.0000 |

## (d) Power: minimum detectable effect of each regressor

MDE = 2.80 x the larger of the event x state and county cluster-robust SEs. `gain at MDE` = pooled-RMSE reduction if an effect of exactly that size existed and were captured; `power at 2%` = power against the effect that would move pooled RMSE by 2%. Only regressors with at least 20 effective clusters in both clusterings enter the table and the verdict. t is descriptive only (not lead- or unit-demeaned).

| kind | contrast | regressors (eligible / all) | gain at MDE: min / median / max | share with gain at MDE <= 2% | power at 2%: median | largest |t| (descr.) |
|---|---|---:|---|---:|---:|---:|
| ladder | C1 quad-pop (sub-grid) | 0 / 160 | | | | |
| ladder | C2 pop-area (exposure weighting) | 16 / 160 | 0.016% / 0.055% / 0.141% | 1.00 | 1.00 | 1.83 |
| ladder | C3 pooled-mean (L1g marginal) | 0 / 160 | | | | |
| ladder | C4 quad-pooled (co-location) | 15 / 160 | 0.030% / 0.078% / 0.120% | 1.00 | 1.00 | 1.26 |
| within | quadn-quad [canopy] | 0 / 40 | | | | |
| within | quadn-quad [canopy_leafon] | 0 / 40 | | | | |
| within | quadn-quad [wet] | 0 / 40 | | | | |

**(d) verdict: undetermined (no eligible regressor)** — rule: the pooled-RMSE gain at the minimum detectable effect exceeds the 2% resolution for every within-modulator regressor. An MDE is read against the effect sizes part (a) makes plausible.

## Notes

* (c) The review's donor null is realised by the `other` variant (a same-relief-stratum donor's bands under the county's own weather): C5 asks whether the county's own band distribution aligns beyond the donor's.
* (d) Rates are propagated with the base's own u and r; no restoration time constant is assumed.

