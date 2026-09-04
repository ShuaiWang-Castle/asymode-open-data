# Post-confirmation diagnosis: coarse two-flow experiment

**Status:** post hoc diagnostic written after the one-shot confirmation at `9d432801397a97062ef9820894c1c1dbbed09fbb`. Nothing in this directory is part of the frozen confirmation. It must not be used to relabel that run as positive or to tune another model on the same 2022/2024 outcomes.

## Executive finding

The confirmation was computationally reproducible, but its predeclared structural gate failed:

- active-48 one-step difference, one minus two: `-3.5138e-07`, 6/11 positive events, exact sign-flip `p=0.9746`;
- path-24 difference, one minus two: `+3.5317e-05`, 7/12 positive events, exact sign-flip `p=0.5327`, with a negative leave-one-event minimum;
- h+24 difference, one minus two: `+5.5098e-05`, 8/12 positive events, exact sign-flip `p=0.5620`, with a negative leave-one-event minimum.

The null is not primarily a rollout-accumulation failure. Across the eleven events with the one-step endpoint, the event-level one-step difference is strongly aligned with the path-24 and h+24 differences (post-hoc Spearman correlations `0.945` and `0.964`). The failure is already present in the transported one-step law.

An exact target-oracle decomposition on the already revealed confirmation events gives

\[
d_e^{\mathrm{step}}
=
G_{Q,e}^{\mathrm{oracle}}
+T_{1,e}(P\to Q)-T_{2,e}(P\to Q),
\]

where `G_Q^oracle` is the fixed-partition target-event one-flow collapse gap and `T_j` is the excess risk from using the source-fitted arm rather than the target-event oracle in the same class. Numerically:

- mean target-oracle gap: `+9.0285e-07`, positive in 10/11 events;
- mean transfer-penalty difference `T_1-T_2`: `-1.2542e-06`, negative in 11/11 events;
- net observed difference: `-3.5138e-07`.

Thus the fixed `K=8` partition still contains a target-event two-flow representation signal, but the second source-fitted rate is less transportable than the regularized one-flow projection. The source `Gamma` diagnostic priced an extra within-source degree of freedom; it did not price cross-event coefficient drift.

## Critical data defect discovered after the run

The published driver tensors contain a semantic missingness defect that was not covered by the checksum or split validators:

- 2,625 unique counties occur in the 26 panels;
- 1,265 counties have **all twelve ERA5 channels exactly zero at every hour in every event in which they appear**;
- 1,360 counties are always nonzero;
- no county switches between the two statuses across events.

Temperature and surface pressure cannot both be identically zero for a full week. The driver builder initializes every county to zero and only overwrites counties that match rows in the county-weight table. It does not raise an error or write a driver-observation mask when a FIPS has no match. The fixed county pattern therefore identifies an incomplete county-to-ERA5 mapping at driver-build time; the precise upstream cause (stale weight table versus FIPS/mapping omission) still requires direct membership verification.

This defect is material to the formal model because K-means is defined by the weather features. Source cluster 1 has essentially zero weather in every meteorological coordinate. In the worst one-step event, `2024-09-27`, that cluster contains 39.25% of active transitions and contributes `-8.718e-06` of the total `-9.245e-06` one-step difference (94.3%). The source fit assigns cluster 1 both interruption and restoration, while the target-event oracle is interruption-only.

The formal run is therefore a valid execution on the published bytes, but not a scientifically clean confirmation of exogenous-flow transfer. Do not update the manuscript or evidence ledger from it until the weather drivers are rebuilt and versioned under a new manifest.

## What remains encouraging, but is not the structural result

The coarse two-flow estimator beat recursive HGB strongly and beat the direct h+24 HGB in 9/12 events with exact sign-flip `p=0.0127`. These comparisons do not rescue the structural claim: direct HGB was fitted without equal-event sample weights and saw compressed mean/max summaries of the future weather path, whereas the sieve consumed the full hourly path. Damped persistence was not robustly beaten (only 4/12 positive path events, interval crossing zero).

## Immediate decision

1. Freeze the confirmation result as a failed structural confirmation on defective driver bytes.
2. Do not run another architecture or another confirmation on the same 2022/2024 outcomes.
3. Rebuild the county-to-ERA5 mapping after all 26 panels are present; fail closed on every county without weights and never encode missing drivers as physical zero.
4. Produce a new driver manifest and repeat only the zero-training geometry/preflight first.
5. Reframe the next theory/estimator around both representation gain and transport cost. A future untouched confirmation requires newly frozen events; a rerun on 2022/2024 can only be called a repaired replication.

See the remaining files in this directory for the event-level decomposition, driver-coverage table, and reproduction script.