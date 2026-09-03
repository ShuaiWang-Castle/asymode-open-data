# Event-held-out two-rate transfer: confirmatory result

```text
DECISION: NOT CONFIRMED

Core event-held-out result:
- h+24: NOT CONFIRMED. Equal-event mean +4.55%, 8/11 event signs positive,
        bootstrap CI [+0.74, +9.25], randomization p = 0.041, every LOEO mean
        positive. Fails the preregistered 9/11 sign minimum.
- h+48: NOT CONFIRMED. Equal-event mean +3.71%, 6/11 positive,
        bootstrap CI [-0.84, +9.20] includes zero, randomization p = 0.246.

Event-vs-county strengthening:
- h+24: NOT CONFIRMED. d mean +4.11%, 7/11 positive, CI [-0.29, +9.48],
        randomization p = 0.185.
- h+48: NOT CONFIRMED. d mean +2.84%, 6/11 positive, CI [-2.35, +8.76],
        randomization p = 0.386.

Recovery-burden decision:
- SCREEN FAILED
```

## 1. Commit hashes and data digests

See `00_ENVIRONMENT.md`. Work branch `cc-event-transfer-confirmation-20260903`;
`panel_digest 76a73ed794af`, `channel_digest dec964873cb2`, event split digest
`aea2acb10037`, county split digest `f5a428dfa590`; data checksums 60/60.

**One instruction was refused.** Section 1 of the task names
`ShuaiWang-Castle/DMDA_DataChallege` as the source of the model implementation.
That repository is excluded by `FIREWALL.md`; it was not cloned, read or
referenced. The public two-rate implementation used here has always lived in this
repository, so nothing was lost. The exploratory branch `gpt-pretest-20260903`
was listed only; none of its formulas or numbers entered this work.

## 2. Integrity and leakage verdict

**PASS** (`01_INTEGRITY_AUDIT.md`). All 26 panel/driver pairs agree on fips,
county dimension and length; timestamps are strictly increasing UTC at 15-minute
spacing; the scored block is exactly the public 14 channels; the clock was
rebuilt independently from the panel timestamps and matches the harness while
differing from the legacy lead-phase channel; 0.95% of targets are unobserved and
are excluded from every loss, early-stopping decision and reported metric;
normalisation uses training-event rows only; no test event appears in training or
validation in any fold (asserted per fold).

**A defect in the archived material was found and is why nothing was reused.**
The archived county-held-out predictions were produced before the clock fix, so
their `clock_sin/cos` encode forecast lead time modulo 24 h rather than the hour
of day, and their county fold map moved with the model seed. The archived
event-held-out run used 5 balanced event folds and one seed, not leave-one-event-out.
Both protocols were therefore rerun from scratch; **138 new fits**.

## 3. Parameter and budget matching

| model | hidden | parameters | vs two_rate |
|---|---|---|---|
| two_rate | 32 | 3,138 | — |
| net_scaled | 48 | 3,121 | −0.54% |
| recovery_burden | 32 | 3,171 | +33 (+1.05%) |

Shared budget: 60 epochs, patience 12, batch 512, Adam at lr 3e-3, caps 0.25.
Every arm passes through one `fit_arm`; the only arm-dependent statements are the
model factory and the burden state. `tests/test_cc_fairness.py` (6 tests, all
passing) fails if that changes, if the parameter counts drift past 1%, if the
burden increment is not +33, or if the split map stops being leave-one-event-out.

## 4. Event-level gains, all 11 events

Positive means `net_scaled` is worse. Seeds averaged inside each event.

| event | h+1 | h+6 | **h+24** | **h+48** |
|---|---|---|---|---|
| 2021-05-04 | −3.83 | +6.33 | +1.95 | +2.46 |
| 2021-06-21 | −0.99 | +0.23 | +3.80 | −1.14 |
| 2021-08-11 | +1.41 | +1.05 | −2.69 | −0.01 |
| 2021-12-11 | −1.23 | +2.09 | +5.02 | +2.44 |
| 2022-04-13 | +7.41 | +11.48 | +16.11 | +15.99 |
| 2022-06-08 | −2.07 | +0.10 | −0.17 | −8.82 |
| 2022-06-17 | +0.46 | +0.53 | +2.00 | +3.38 |
| 2022-07-23 | +0.15 | −3.24 | −2.09 | −1.08 |
| 2024-05-08 | +3.59 | +9.97 | +22.21 | +24.42 |
| 2024-05-26 | +1.55 | +0.54 | +2.95 | +3.65 |
| 2024-06-26 | +0.56 | +3.29 | +0.94 | −0.47 |
| **equal-event mean** | +0.64 | +2.94 | **+4.55** | **+3.71** |
| **median** | +0.46 | +1.05 | **+2.00** | **+2.44** |
| pooled-cell (secondary) | +0.89 | +2.14 | +2.77 | +3.20 |

The mean exceeds the median at both primary horizons because two events —
2024-05-08 and 2022-04-13 — carry most of it. Leave-one-event-out means stay
positive (h+24 min +2.78%, h+48 min +1.64%), so no single event flips the sign,
but the distribution is not the 10/11 pattern the archived numbers reported.

## 5. Inference

| horizon | mean | sign test p | randomization p | bootstrap 95% | signs | LOEO min |
|---|---|---|---|---|---|---|
| h+24 | +4.55% | 0.227 | **0.041** | [+0.74, +9.25] | 8/11 | +2.78% |
| h+48 | +3.71% | 1.000 | 0.246 | [−0.84, +9.20] | 6/11 | +1.64% |
| h+6 (descriptive) | +2.94% | 0.012 | 0.047 | [+0.57, +5.61] | 10/11 | +2.09% |
| h+1 (descriptive) | +0.64% | 0.549 | 0.531 | [−0.94, +2.42] | 7/11 | −0.04% |

Bootstrap: 50,000 event resamples. Randomization: exact enumeration of all
2^11 = 2,048 model-label swaps. No fold × seed t-test is used anywhere.

**Convergence gate.** No fit reached the epoch cap (0/66), but the median
validation improvement over the last ten checkpoints was 1.55% (two_rate) and
3.67% (net_scaled), above the 0.5% trigger, so the matched 2×-budget probe was
run on the two events with the largest absolute gains. Result: in all 12 probe
fits the best checkpoint was unchanged and the four gains are identical to four
decimals; **the primary sign does not change**. The probe doubles the epoch cap
and patience, and no fit came near the cap (26–50 of 120), so what binds is the
early-stopping rule on a single validation event, not the budget. That is worth
stating plainly: the median best epoch is 3 for two_rate and 6 for net_scaled, so
both models are selected very early on a noisy one-event validation signal. The
probe rules out an under-training explanation of the *sign*; it does not
establish that a richer validation design would leave the magnitudes unchanged.

## 6. County-versus-event direct difference

| horizon | event gain | county gain | d = event − county | signs | CI 95% | rand p |
|---|---|---|---|---|---|---|
| h+24 | +4.55% | +0.44% | +4.11% | 7/11 | [−0.29, +9.48] | 0.185 |
| h+48 | +3.71% | +0.87% | +2.84% | 6/11 | [−2.35, +8.76] | 0.386 |

The event-held-out interval excludes zero at h+24 while the county interval does
not, which is exactly the pattern Section 9 warns against reading as
strengthening. The direct paired test does not pass at either horizon:
**"the two-rate advantage becomes stronger when the entire event is held out" is
not confirmed.**

## 7. Synthetic theorem check — PASS

`10_THEORY_UNIT_TESTS.json`. 10,000 random environment/rate draws: the
interruption identity and the restoration identity both hold with maximum
relative error 7.4 × 10⁻¹³. Each closed-form projection was independently
confirmed to be the minimiser of the risk on P (vanishing numerical derivative,
both neighbours worse) on 500 draws. Boundary cases pass: `R = 0` and `U = 0` give
exactly zero risk for the matching one-rate class; `v_Q = 0` reduces the identity
to the pure projection-shift term; `P = Q` leaves only the irreducible term. The
nonnegative-clipped projection is reported separately: clipping never lowers the
target risk below the free projection. Figure:
`figures/theory_projection_shift.png`.

## 8. Real-data projection shift — PASS

`11_PROJECTION_SHIFT_K16.json`, `12_PROJECTION_SHIFT_K32.json`. Median shift per
unit rate squared, k-means fitted on training rows only, compared against a
pooled random 20% control with the same event mixture:

| K | branch | event-held-out | random split | verdict |
|---|---|---|---|---|
| 16 | interruption | 2.95e−05 | 1.74e−06 | event larger |
| 16 | restoration | 2.06e−02 | 4.10e−03 | event larger |
| 32 | interruption | 3.07e−05 | 2.50e−06 | event larger |
| 32 | restoration | 3.17e−02 | 7.84e−03 | event larger |

Both K values and both branches pass. This tests event-dependent one-rate
projection and nothing else; it is not evidence about event-family ordering.

## 9. Recovery burden — SCREEN FAILED

`13_RECOVERY_BURDEN_SCREEN.csv`. Three preselected events, three seeds, full
budget, same validation rule.

| gate | requirement | observed | pass |
|---|---|---|---|
| 1 | ≥ 1.0% mean improvement at h+24 or h+48 | −0.49% (h+24), −0.40% (h+48) | **no** |
| 2 | other long horizon not worse by > 1.0% | −0.40% | yes |
| 3 | ≥ 2/3 seeds improve on ≥ 2/3 events | 3/3 | yes |
| 4 | no event degrades by > 3.0% | worst −0.04% | yes |
| 5 | not caused by future true `y` | enforced structurally and unit-tested | yes |

The direction is consistently favourable and never harmful, but the magnitude is
half the preregistered threshold. Per Section 11.1 the work stops here: no full
11-event run, no `rho` tuning, no extra inputs, no loss change. The fitted `rho`
stayed at 0.9703–0.9729 against an initialisation of 0.9715, so the model did not
find a different memory timescale.

## 10. Remaining gap to HGB, and damped persistence

Mean over the 11 events of the per-event RMSE:

| arm | h+1 | h+6 | h+24 | h+48 |
|---|---|---|---|---|
| two_rate | 0.01112 | 0.02463 | 0.03726 | 0.03320 |
| net_scaled | 0.01122 | 0.02531 | 0.03873 | 0.03446 |
| HGB, same information | 0.01340 | 0.02555 | 0.03630 | 0.03276 |
| damped persistence | 0.01105 | 0.02433 | 0.03659 | 0.03302 |

two_rate against HGB (positive = HGB worse): h+1 +11.42% (9/11), h+6 +1.89%
(7/11), h+24 −3.29% (median +0.76%, 6/11), h+48 −1.02% (median +1.10%, 6/11). At
the long horizons the mean and the median disagree in sign: HGB wins on average
because of a few events, while the two-rate model is ahead on more events than
not. Neither direction is established at the event level.

**two_rate does not beat damped persistence under this protocol**: the equal-event
mean is negative at every horizon (h+1 −0.72%, h+6 −0.51%, h+24 −2.99%,
h+48 −2.38%), with medians positive at h+6 and h+24. The archived "7–9% over
damped persistence at h+24/48" does not survive leave-one-event-out.

HGB boosting rounds ran 25–225 against a 2,000 cap, so the reference was never
cap-limited. HGB is deterministic at this sample size: its three seeds are one fit
repeated, and are reported as one deterministic fit, not three replicates.

## 11. Claim-by-claim paper wording

See `16_CLAIM_LEDGER.md`.
