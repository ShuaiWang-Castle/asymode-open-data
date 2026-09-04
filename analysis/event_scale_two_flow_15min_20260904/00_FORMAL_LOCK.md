# Event-scale 15-minute one-flow versus two-flow confirmation

**Status:** canonical confirmation lock, written before the exact estimator below is evaluated on any 2022 or 2024 event.  
**Public data:** `main@8dd47c5ccd829611f27b69a3d64c274a0a24c400`.  
**Manifest:** `configs/panel_manifest_g3-all-26.json`, digest `db286b4960a4`.  
**Design-audit reference:** `5f3ff3aa616fac7540eae45415089127816f3199`.

## 1. Scientific question

The preflight reveals a scale mismatch: a second event-level constant rate is well supported by the large within-event cross-section, whereas a separate second rate in most local `k=200` driver neighbourhoods is not. This experiment tests the scale selected by that diagnosis:

> Before a forecast origin within a storm, do source counties contain enough information to estimate one event-level interruption rate and one event-level restoration rate, and does retaining both rates improve the 24-hour path for held-out counties relative to the matched single signed-flow class?

The experiment is deliberately non-neural. It isolates flow separation from feature learning, capacity, initialization, and nonconvex optimization. It is storm-conditioned cross-county nowcasting, not transfer to an entirely unseen storm.

## 2. Evidence chronology

- **Development and candidate selection:** the fourteen events dated 2018--2021.
- **Locked confirmation:** all twelve events dated 2022 or 2024, evaluated once after this lock.
- No result from the exact event-scale `K=1` estimator on the confirmation events may be inspected before this file is committed.
- Other exploratory or previously locked estimators (`K=8`, 15-minute local clustering, persistence mixtures, neural arms) are not the primary model in this confirmation and may not be substituted after the confirmation is read.

Development-only evidence for this exact specification, computed from the pinned public bytes with the rules below, was:

| endpoint | one-flow equal-event MSE | two-flow equal-event MSE | mean difference `one-two` | positive events | exact two-sided sign-flip p |
|---|---:|---:|---:|---:|---:|
| native 15-minute teacher-forced transition | 2.078185e-4 | 2.077418e-4 | 7.6696e-8 | 11 / 14 | 0.00098 |
| 24-hour open-loop path | 4.916095e-3 | 4.893328e-3 | 2.2767e-5 | 11 / 14 | 0.00098 |
| lead 24 hours | 9.513286e-3 | 9.473760e-3 | 3.9525e-5 | 10 / 14 | 0.00195 |

The two-flow development path MSE was 4.893328e-3 versus 4.976080e-3 for persistence in equal-event mean. This practical comparison was heterogeneous and is not used to select any further model feature.

## 3. Data and outcome-blind storm alignment

For each event:

1. Use the native 15-minute `panel_<event>.npz` state and explicit observation mask.
2. Define the NOAA county-footprint peak exactly as in the conservation preflight: maximize the fraction of panel counties with an active NOAA record; break footprint ties using the positive standardized public-weather composite of gust, wind speed, precipitation, snowfall, and CAPE; break any remaining tie by the earliest hour.
3. Set the forecast origin to exactly twelve hours before the peak.
4. Forecast 96 native 15-minute steps (24 hours). If the origin or horizon falls outside the published panel, mark the event unavailable and do not move it.
5. No outage value, target, prior gain, residual, fitted rate, family label, or information score may select the event, peak, origin, or row population.

## 4. County split and information set

For each event, compute

```text
h(FIPS) = first 64 bits of SHA256("20260904|" + zero-padded FIPS).
```

A county is held out when `h(FIPS) mod 5 == 0`; all other counties are source counties. Thus approximately 80% of counties supply pre-origin transitions and approximately 20% are never used for fitting.

- All rows of a county remain on one side.
- The estimator uses source-county transitions strictly before the forecast origin.
- A transition `t -> t+1` is legal only when both native endpoint states are observed and finite.
- No post-origin outage observation from either source or test counties enters fitting, model selection, damping, or calibration.
- The model uses no event label, county label, weather feature, static feature, or outage-history feature beyond the current state multiplier in the transition law. This is intentional: the target is the event-scale pooling regime identified by the preflight.

## 5. Exact matched estimators

Let the source transition population be `(Y_i, Delta_i)`.

### Two-flow estimator

Fit the exact box-constrained least-squares problem

\[
(\widehat U,\widehat R)
\in\arg\min_{0\le U\le0.06625,\;0\le R\le0.0625}
\sum_i\{\Delta_i-U(1-Y_i)+RY_i\}^2.
\]

The solver must enumerate the feasible interior stationary point, all four edge minimizers, and all four corners.

### One-flow estimator

Fit the matched union of two nonnegative rays and retain the lower source SSE:

\[
\Delta_i=a(1-Y_i),\quad 0\le a\le0.06625,
\]

or

\[
\Delta_i=-bY_i,\quad 0\le b\le0.0625.
\]

There is no learned representation, no optimizer, no early stopping, no initialization, no cluster selection, and no post-fit scale. The only scientific difference is whether interruption and restoration may both be retained.

## 6. Forecast and endpoints

Initialize every eligible held-out county at its observed state at the origin. For each of the next 96 steps, apply the fitted event-level rates recursively:

\[
\widehat Y_{t+1}=\widehat Y_t+\widehat U(1-\widehat Y_t)-\widehat R\widehat Y_t.
\]

No clamp should bind because the locked caps imply `U+R<1`; any out-of-range state is an implementation failure.

For each confirmation event report:

1. **Primary structural endpoint:** observed-cell 24-hour path MSE difference
   \[
   d_e^{\rm path}=\operatorname{MSE}_{1,e}^{1:96}-\operatorname{MSE}_{2,e}^{1:96}.
   \]
2. **Theory-aligned endpoint:** native 15-minute teacher-forced transition MSE difference over held-out counties and the same 24-hour future interval.
3. **Secondary lead endpoint:** MSE at 24 hours.
4. **Treatment diagnostics:** fitted rates, source moments, source plug-in `G` and `Gamma` labelled descriptive, `c=min(U,R)`, RMS delivered difference `c(1-2Y)`, boundary status, and legal row counts.
5. **Practical references:** persistence; source-fitted damped persistence; recursive histogram gradient boosting and direct h+24 histogram gradient boosting may be reported, but neither changes the structural gate.

The event is the inferential unit. Report every event difference, equal-event mean, median, exact two-sided sign-flip test, 50,000-event bootstrap interval, positive-event count, and every leave-one-event mean. Row-level tests are prohibited.

## 7. Confirmation gate

The event-scale two-flow claim is supported only when all of the following hold on the available confirmation events:

1. equal-event mean `d_path > 0`;
2. exact two-sided sign-flip `p < 0.05`;
3. at least 8 of 12 confirmation events have `d_path > 0` (or the corresponding two-thirds threshold if an event is exogenously unavailable);
4. every leave-one-event mean `d_path` remains positive;
5. the equal-event mean teacher-forced transition difference is nonnegative.

Practical competitiveness is reported separately. It requires the two-flow equal-event path MSE to be below persistence; performance against HGB is reported without changing the structural conclusion.

## 8. Required ablation and interpretation

The sole scale ablation is the already development-screened `K=8` exogenous partition. It is secondary and cannot replace the `K=1` primary result. Its purpose is to test the preflight prediction that spending one additional flow parameter per local cell can erase the benefit observed at event scale.

Interpretation is fixed:

- `K=1` positive and `K=8` weak/negative supports a representation--estimation scale tradeoff.
- Both positive supports two-flow value at both scales.
- Transition positive but path nonpositive supports local representation gain without forecast propagation.
- Both nonpositive means the confirmation does not establish event-scale two-flow value, regardless of the development result.
- Beating one flow but losing to persistence/HGB supports a structural result but not practical superiority.

## 9. Hard stop

Run this confirmation once. After any 2022/2024 result for this exact estimator is materialized, do not alter the split, resolution, origin, caps, transition population, endpoint, weighting, or estimator on the same confirmation events. Any later model development requires newly frozen events.