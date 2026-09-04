# Local development evidence from pinned public data

All numbers below were recomputed from the GitHub Actions artifact for public-data commit `8dd47c5ccd829611f27b69a3d64c274a0a24c400`, whose artifact SHA-256 is `7439bef471cb84e6c791f012c455d07c8b113cc08626c137bf6740e4706ebed5`. No 2022 or 2024 performance was inspected.

## 1. Why a coarse estimator

The conservation preflight found a scale mismatch: event-level constant models have large plug-in selection indices, while most `k=200` local cells have index below one. A model with thousands of local degrees of freedom therefore pays an estimation cost that a two-parameter model does not. The `K=8` sieve occupies the missing intermediate scale: 16 rate coefficients in the two-flow arm versus eight one-flow coefficients.

On the 2018--2020 source events, five of the eight `K=8` cells have `Gamma_neff > 1`; they contain about 66.5% of the effective source weight. Three cells are correctly near one-flow, including one interruption-only cell. The diagnostic is descriptive because equal-event weights and dependence are present, but it verifies that the model does not place a second rate in thousands of unsupported local cells.

## 2. Model selection on the six 2021 events

The table reports equal-event MSE. Positive gain means the matched one-flow model has larger error.

| K | two-flow path-24 MSE | structural path gain | positive events | one-step gain | positive one-step events |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.004135 | 3.70% | 5/6 | 0.41% | 5/6 |
| 2 | 0.004134 | 2.87% | 5/6 | 0.34% | 5/6 |
| 4 | 0.004122 | -0.16% | 1/6 | 0.03% | 2/6 |
| **8** | **0.003997** | **1.31%** | **5/6** | **0.095%** | **5/6** |
| 16 | 0.004013 | 1.79% | 6/6 | 0.22% | 5/6 |

`K=8` is selected by the prespecified practical criterion—lowest equal-event 24-hour path MSE—not by the largest one-flow/two-flow contrast. `K=16` has a larger structural difference but slightly worse absolute forecast risk and is therefore not selected.

## 3. Stability to the fixed K-means seed

Although the formal seed is 0, the development-only sensitivity check used five K-means seeds. The two-flow path gain remained positive for every seed and for 5/6 events under every seed:

| seed | path gain | one-step gain | positive path events |
|---:|---:|---:|---:|
| 0 | 1.31% | 0.095% | 5/6 |
| 1 | 1.62% | 0.290% | 5/6 |
| 2 | 0.60% | 0.078% | 5/6 |
| 3 | 1.39% | 0.153% | 5/6 |
| 4 | 2.53% | 0.204% | 5/6 |

The sign is not an artifact of one cluster initialization.

## 4. Practical baseline check on 2021

| method | equal-event path-24 MSE | equal-event h+24 MSE |
|---|---:|---:|
| **coarse two-flow, K=8** | **0.003997** | **0.005841** |
| matched one-flow, K=8 | 0.004044 | 0.005898 |
| global constant two-flow | 0.004135 | 0.006082 |
| global constant one-flow | 0.004254 | 0.006339 |
| damped persistence | 0.004275 | 0.006325 |
| persistence | 0.004389 | 0.006569 |
| recursive HGB | 0.005084 | 0.007680 |

The coarse two-flow arm has the lowest development-year path risk. Relative to damped persistence it improves the equal-event path MSE by about 6.5%; relative to persistence by about 8.9%; relative to recursive HGB by about 21.4% in the ratio of equal-event means. Event-level signs are 5/6 against persistence and damped persistence and 6/6 against recursive HGB.

A direct h+24 HGB using current state, the current 24-feature vector, and mean/max summaries of the known future 24-hour weather path has equal-event MSE 0.006398, versus 0.005841 for the coarse two-flow trajectory. This direct comparison is heterogeneous (3/6 events favor each), so the confirmation table must report every event rather than only the aggregate.

## 5. Temporal-resolution robustness

A development-only 30-minute rerun, with hourly weather features repeated to half-hour steps and no other model change, gives:

- one-step gain 0.112%, 6/6 events, exact sign-flip `p=0.03125`;
- path-24 gain 1.455%, 5/6 events;
- h+24 gain 1.817%, 5/6 events.

This supports that the hourly result is not created by a single aggregation boundary. The formal run nevertheless stays hourly because that resolution, features, and baseline implementation were locked first and already pass the practical development gate.

## 6. Why the online cross-county candidate is not primary

At `K=8`, the within-storm county-held-out nowcast produced a positive path difference on 14/14 development events (mean gain about 0.54%), but the equal-event teacher-forced one-step difference was slightly negative. It is also a different task because the target storm is observed in source counties before prediction. It can be retained as a future extension, but running it as a second confirmation on the same 2022/2024 events would create avoidable multiplicity and weaken the paper's central unseen-event claim.

## Development conclusion

The coarse temporal-transfer model is hopeful enough for a single locked confirmation because it satisfies four distinct development checks: positive structural one-step and path differences, best absolute path risk among the prespecified K grid, stability across K-means seeds, and superiority in equal-event mean to persistence and both recursive and direct tree baselines. None of these statements uses 2022/2024 outcomes.
