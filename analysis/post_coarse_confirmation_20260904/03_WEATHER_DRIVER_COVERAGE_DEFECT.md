# Semantic weather-driver coverage defect

## Finding

The 26 published driver tensors were scanned county by county across all twelve raw ERA5 channels and all 169 hours.

- unique panel counties: `2,625`;
- counties whose entire 12-channel weekly driver tensor is exactly zero whenever they appear: `1,265` (`48.19%`);
- counties whose driver tensor is nonzero whenever they appear: `1,360`;
- counties that switch status across events: `0`.

This cannot be meteorology. In particular, 2-m temperature and surface pressure cannot both be exactly zero for a full week.

## Code path

`src/asymode/weather.py::apply_weights` initializes

```python
out = np.zeros((len(fips), field.shape[0]), dtype=np.float32)
```

and overwrites only FIPS values present in the county-weight table. A county with no matching weight row silently retains zeros for every field and time. `scripts/build_drivers.py` interpolates temporal NaNs but does not verify that every requested county was populated, does not emit a driver-observation mask, and saves the zero tensor as data.

The fixed county status across all events establishes that the problem is a county-to-grid mapping omission, not a weather episode with small values. The exact upstream cause must be verified directly against `era5_county_weights.parquet`; likely possibilities are that the table was generated before all panel counties were present or that a subset of FIPS failed to match. Until that membership audit is complete, do not assert which of those two caused it.

## Event prevalence

| event | family | all-zero weather counties |
|---|---|---:|
| 2018-01-16 | winter | 34.37% |
| 2018-10-11 | tropical | 46.38% |
| 2019-02-20 | winter | 31.01% |
| 2019-02-24 | wind | 34.16% |
| 2019-11-27 | wind | 40.83% |
| 2020-02-06 | flood | 50.00% |
| 2020-08-04 | tropical | 58.22% |
| 2020-10-29 | tropical | 41.67% |
| 2021-02-15 | winter | 37.90% |
| 2021-05-04 | convective | 0.00% |
| 2021-06-21 | convective | 0.00% |
| 2021-08-11 | convective | 0.00% |
| 2021-12-11 | convective | 0.00% |
| 2021-12-15 | wind | 80.61% |
| 2022-01-16 | winter | 21.76% |
| 2022-03-12 | winter | 27.05% |
| 2022-04-13 | convective | 0.00% |
| 2022-06-08 | convective | 0.00% |
| 2022-06-17 | convective | 0.00% |
| 2022-07-23 | convective | 0.00% |
| 2024-01-09 | winter | 49.58% |
| 2024-01-12 | winter | 49.83% |
| 2024-05-08 | convective | 29.66% |
| 2024-05-26 | convective | 24.19% |
| 2024-06-26 | convective | 38.70% |
| 2024-09-27 | wind | 39.33% |

The corresponding active-48 transition fractions are nearly identical. About 33.2% of source active transitions and 27.7% of confirmation active transitions carry zero weather tensors.

## Effect on the frozen estimator

The K=8 method treats zeros as observed physical features. It therefore allocates missing-driver counties to ordinary K-means cells. Source cluster 1 has an inverse-transformed meteorological center essentially equal to zero for gust, wind speed, CAPE, precipitation, snowfall and their lag summaries. It is a missing-driver cell, not a coherent storm regime.

This cell is the principal cause of the worst confirmation event. In `2024-09-27`:

- 39.25% of active transitions are assigned to cluster 1, matching the event's zero-weather transition fraction;
- the source two-flow fit is `U=0.001537`, `R=0.008572`;
- the target-event oracle is approximately `U=0.006797`, `R=0`;
- cluster 1 contributes 94.3% of the event's negative one-step structural difference.

The driver defect therefore directly enters the structural comparison and cannot be dismissed as a baseline-only issue.

## Sensitivity that does not repair the data

A post-hoc sensitivity excluding all-zero-driver counties does not restore a stable structural result: on the confirmation events, the one-step mean becomes slightly positive but only 5/11 events are positive; path and h+24 remain heterogeneous and fail leave-one-event robustness. This exclusion is not equivalent to rebuilding the missing weather and must not be used as a corrected confirmation. It only shows that the semantic-zero defect is material but not necessarily the sole source of cross-event nontransportability.

## Required repair

1. Rebuild `era5_county_weights.parquet` after the full 26-panel county union is frozen.
2. Assert that every panel FIPS has at least one positive, normalized weight row. Fail closed otherwise.
3. Change `apply_weights` to initialize with `NaN`, not zero; require explicit coverage before writing a county series.
4. Add a `driver_observed`/coverage vector to every driver file or exclude uncovered counties before panel release.
5. Assert physically impossible sentinels are absent: no county may have temperature and pressure identically zero over the panel.
6. Rebuild all 26 `drivers_*.npz`, assign a new manifest and channel/data digest, and leave the original release immutable for audit.
7. Rerun the zero-training conservation and scale preflight before fitting any model.

Because the 2022/2024 outcomes have now been inspected, a run on repaired drivers is a repaired replication, not a fresh confirmation. Untouched confirmation requires newly frozen events.