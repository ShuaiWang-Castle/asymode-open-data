# Claim ledger — event-transfer confirmatory task

Status words are `confirmed`, `not confirmed`, `rejected`, `descriptive`. No other
word is used. Every row points at the file that decides it.

| # | claim as it would appear in the paper | status | evidence | wording that is allowed |
|---|---|---|---|---|
| 1 | Two separated rates have a robust event-transfer advantage over a parameter-matched one-rate projection at h+24 | **not confirmed** | `08_EVENT_INFERENCE.json` — mean +4.55%, randomization p = 0.041, bootstrap CI [+0.74, +9.25], every LOEO mean positive, but 8/11 signs against a preregistered 9/11 minimum | "Under leave-one-event-out the two-rate model has a positive mean advantage of 4.6% at 24 h (randomization p = 0.04, event-bootstrap interval above zero), but the advantage is present in only 8 of 11 events and does not meet our preregistered sign criterion." |
| 2 | The same at h+48 | **not confirmed** | mean +3.71%, 6/11, CI [−0.84, +9.20], randomization p = 0.246 | "At 48 h the mean advantage is 3.7% but the event-level interval includes zero and only 6 of 11 events favour the two-rate model." |
| 3 | The advantage is larger under full event holdout than under within-event county holdout | **not confirmed** | `07_SPLIT_DIFFERENCE_RESULTS.csv`, `08_EVENT_INFERENCE.json` — d = +4.11% (h+24, 7/11, CI [−0.29, +9.48]), +2.84% (h+48, 6/11, CI [−2.35, +8.76]) | "The point estimate of the advantage is larger under event holdout than under county holdout at both long horizons, but the paired per-event difference does not exclude zero, so we do not claim strengthening." |
| 4 | A one-rate model is an event-distribution-dependent projection, with an exact transfer-shift term | **confirmed** | `10_THEORY_UNIT_TESTS.json` — 10,000 draws, max relative error 7.4e−13, boundary cases pass, projection independently verified as the minimiser | "The best one-signed-rate approximation is the L2 projection onto an environment-dependent class; its target risk decomposes exactly into an irreducible term and a projection-shift term (Theorem, verified numerically to 1e−12)." |
| 5 | The projection-shift term is present in the open data | **confirmed** | `11_PROJECTION_SHIFT_K16.json`, `12_PROJECTION_SHIFT_K32.json` — event median exceeds the random-split median for both branches at K = 16 and K = 32 | "Partitioning driver space on training rows only, the projection shift between a held-out storm and the remaining storms is an order of magnitude larger than between a random split with the same event mixture, at both resolutions and for both one-rate branches." |
| 6 | A recovery-burden state improves long-horizon performance | **rejected** | `13_RECOVERY_BURDEN_SCREEN.csv` — best horizon −0.49% against a −1.0% gate | "A single learned recovery-burden state fed only to the restoration rate is consistently in the right direction and never harmful, but at half the magnitude we preregistered as the minimum, so we do not adopt it." |
| 7 | The two-rate model beats damped persistence by 7–9% at h+24/48 | **rejected** | `15_HGB_REFERENCE.csv` — equal-event mean is negative at every horizon | Do not state. If damped persistence is discussed: "Under leave-one-event-out the two-rate model does not beat damped persistence on the equal-event mean at any horizon." |
| 8 | The model robustly beats HGB at h+1 | **descriptive** | `15_HGB_REFERENCE.csv` — mean +11.42%, 9/11 events; no event-level interval was preregistered for this comparison | "At 1 h the two-rate model is ahead of gradient boosting on the same information in 9 of 11 events; we report this descriptively because the comparison was not part of the preregistered confirmatory design." |
| 9 | HGB wins at the long horizons | **not confirmed** | mean favours HGB (h+24 −3.29%, h+48 −1.02%) while the median favours the two-rate model (+0.76%, +1.10%) and 6/11 events favour it | "At 24 and 48 hours the two estimators are close: gradient boosting has the lower size-weighted mean while the two-rate model is ahead on the majority of individual events. We do not claim either direction." |
| 10 | Identifiability explains the event-family ordering | **not claimed** | forbidden by the task and untested here | Do not state under any result. |
| 11 | Long-horizon error cannot be error accumulation | **not claimed** | forbidden; the earlier saturation argument was already refuted by counterexample | Do not state. |
| 12 | Seeds or folds are independent storm events | **not claimed** | the event is the unit throughout; seeds are averaged inside events | "Optimization seeds are averaged within each event and are reported only as an optimization-variability diagnostic." |
| 13 | The networks recover causal interruption and restoration hazards | **not claimed** | no sequential-exogeneity argument exists | "The learned functions are conditional predictive transition functions." |

## Provenance note required by the task

The architectural direction tested in row 6 arrived through the controlled channel
as a directional prior; it was registered before implementation and is reported
here on public data only. The paper's provenance statement must say that some
directions were suggested rather than discovered.

## What this task did not test

Family ordering, phase-separation mechanism, information-gated concurrency,
input-asymmetry screens, any new sequence model, denominator or missingness
sensitivity, and any additional architecture. None of these was run, and no
statement about them follows from this package.
