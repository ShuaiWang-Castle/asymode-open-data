# F1 information audit — parts (c), (d), (e)

Generated 2026-09-25 00:47:55 by `audit_f1.py` (git da3b154). 6122 county-events, 160 features per variant; variants area, pop, quad, mean, pooled, other, quadn. No model was trained; the base is the held-out prediction of W+Cin (main design, seed 0). Settings (REG) were fixed before any F1 number existed and are copied into `F1_cde.json`. Parts (a) contrast and (b) redundancy are computed separately.

## (c) Alignment with the base's held-out residual

6122 county-events, 778 fixed-effect blocks (156 singletons), 411 event x state clusters. Residual correlogram: c = 0.68, l = 52 km. Direction +1: higher-information variant above the lower one goes with under-prediction (y - P > 0).

| contrast | features | FWER on synthetic fields (homo l / 2l, hetero l / 2l) | null used | best feature | part corr | t | p max-T | p final |
|---|---:|---|---|---|---:|---:|---:|---:|
| C1 quad-pop (sub-grid) | 96 | 0.086 / 0.081 / 0.025 / 0.033 | cluster multiplier | rain*canopy@3 | +0.099 | +1.42 | 0.7678 | 1.0000 |
| C2 pop-area (exposure weighting) | 160 | 0.082 / 0.078 / 0.037 / 0.040 | cluster multiplier | gust_x10*canopy@0 | +0.048 | +2.73 | 0.0290 | 0.1450 |
| C3 pooled-mean (L1g marginal) | 96 | 0.171 / 0.186 / 0.061 / 0.062 | synthetic field (homo 2l) | rain*wet@48 | +0.055 | +1.92 | 0.4356 | 1.0000 |
| C4 quad-pooled (co-location) | 160 | 0.117 / 0.132 / 0.041 / 0.048 | synthetic field (homo 2l) | gust_x15*canopy_leafon@48 | +0.026 | +2.06 | 0.4665 | 1.0000 |
| C5 pooled-mean | other-mean (own beyond donor) | 96 | 0.180 / 0.187 / 0.054 / 0.067 | synthetic field (homo 2l) | rain*wet@48 | +0.055 | +1.92 | 0.4446 | 1.0000 |

**(c) verdict: KILL** — rule: no (contrast, feature) passes at 0.05 after max-T and Bonferroni, in the registered direction, under the null the rule selects.

Hour level (secondary; unit- and lead-demeaned cells; Bonferroni over four contrasts):

| contrast | best feature | part corr | t (event x state) | p final | t (event) | p final, exact event flip |
|---|---|---:|---:|---:|---:|---:|
| C1 quad-pop (sub-grid) | p_tw+1.5*canopy@0 | +0.0087 | +1.68 | 1.0000 | +1.43 | 1.0000 |
| C2 pop-area (exposure weighting) | p_tw+3.0*one@0 | +0.0038 | +1.92 | 1.0000 | +1.79 | 1.0000 |
| C3 pooled-mean (L1g marginal) | p_tw+1.5*wet@0 | +0.0073 | +1.92 | 1.0000 | +1.53 | 1.0000 |
| C4 quad-pooled (co-location) | gust_x10*canopy_leafon@12 | +0.0296 | +2.96 | 0.0670 | +1.71 | 1.0000 |

## (d) Power: minimum detectable effect of each regressor

MDE = 2.80 x cluster-robust SE (event x state). `gain at MDE` = pooled-RMSE reduction if an effect of exactly that size existed and were captured; `power at 2%` = power against the effect that would move pooled RMSE by 2%. t is descriptive only.

| kind | contrast | regressors | gain at MDE: min / median | share with gain at MDE <= 2% | power at 2%: median | best t (descr.) |
|---|---|---:|---|---:|---:|---:|
| ladder | C1 quad-pop (sub-grid) | 96 | 0.00% / 0.02% | 1.00 | 1.00 | 18.28 |
| ladder | C2 pop-area (exposure weighting) | 160 | 0.00% / 0.02% | 1.00 | 1.00 | 58.52 |
| ladder | C3 pooled-mean (L1g marginal) | 96 | 0.00% / 0.01% | 1.00 | 1.00 | 20.89 |
| ladder | C4 quad-pooled (co-location) | 160 | 0.00% / 0.02% | 1.00 | 1.00 | 3.96 |
| within | quadn-quad [canopy] | 40 | 0.01% / 0.03% | 1.00 | 1.00 | 3.15 |
| within | quadn-quad [canopy_leafon] | 40 | 0.00% / 0.00% | 1.00 | 1.00 | 11.79 |
| within | quadn-quad [wet] | 40 | 0.00% / 0.02% | 1.00 | 1.00 | 6.21 |

**(d) verdict: not killed** — rule: the pooled-RMSE gain at the minimum detectable effect exceeds the 2% resolution for every within-modulator regressor. An MDE is read against the effect sizes part (a) makes plausible.

## Notes

* (c) The review's donor null is realised by the `other` variant (a same-relief-stratum donor's bands under the county's own weather): C5 asks whether the county's own band distribution aligns beyond the donor's.
* (d) Rates are propagated with the base's own u and r; no restoration time constant is assumed.

