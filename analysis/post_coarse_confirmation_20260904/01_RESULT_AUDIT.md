# Result audit of `979adcc -> 9d43280`

## 1. Integrity and chronology

The implementation commit `979adccf1e75dde1eeeb20b3747b2483c6c99248` precedes the result commit `9d432801397a97062ef9820894c1c1dbbed09fbb`. The result commit adds only development-reproduction and confirmation outputs. Independent checks reproduced the paired differences and exact sign-flip tests from `EVENT_METRICS.csv`, used storm event as the bootstrap unit, and confirmed that only 2018--2021 events entered fitting.

Two documentation discrepancies remain and must be disclosed:

1. the prose lock states MiniBatchKMeans `batch_size=8192`, whereas the frozen implementation and reproduced development evidence use `4096` with `reassignment_ratio=0.0`;
2. the formal-lock prose says the 2021 one-step contrast was positive on all six events, whereas the committed development table reports 5/6.

Neither discrepancy was introduced after viewing confirmation outcomes, so neither explains the null. They do mean the implementation commit, rather than the stale prose value, is the exact executed specification.

## 2. Structural result

Primary differences are defined as reference MSE minus two-flow MSE.

| endpoint | events | additive mean difference | ratio of equal-event mean MSEs | median eventwise gain | positive events | exact sign-flip | 95% event bootstrap | leave-one-event minimum |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| active-48 one-step | 11 | `-3.5138e-07` | `-0.117%` | `+0.253%` | 6/11 | `0.9746` | `[-2.322e-06, 9.378e-07]` | `-6.625e-07` |
| path 1:24 | 12 | `+3.5317e-05` | `+0.605%` | `+1.157%` | 7/12 | `0.5327` | `[-6.243e-05, 1.433e-04]` | `-6.572e-06` |
| lead 24 | 12 | `+5.5098e-05` | `+0.635%` | `+2.088%` | 8/12 | `0.5620` | `[-1.300e-04, 2.627e-04]` | `-2.574e-05` |

The predeclared structural gate fails. The one-step endpoint is slightly negative on average, and both rollout endpoints are event-heterogeneous and fail both exact inference and the leave-one-event robustness condition.

The report's `mean_gain_pct` is the arithmetic mean of eventwise percentage gains, not the percentage reduction between equal-event mean MSEs. These quantities can have opposite signs when low-error events carry large percentage changes. For example, path-24 has a positive additive mean and a `+0.605%` ratio-of-means improvement, but the mean eventwise percentage is `-1.018%`. Future tables should make the estimand explicit and use the locked additive event-level difference for inference.

## 3. Influence structure

The path-24 positive mean is not uniform. `2024-05-26` contributes `+4.961e-04`, or 117% of the total positive sum; removing it changes the mean to `-6.57e-06`. `2024-09-27` contributes `-3.415e-04`, offsetting 80.6% of the total sum in the opposite direction. Lead 24 has the same pattern: `2024-05-26` contributes 143% of the total, while `2024-09-27` contributes -98.7%.

The one-step negative mean is dominated by `2024-09-27` (`-9.245e-06`). Removing that event changes the remaining ten-event mean to `+5.380e-07`. This is a diagnostic, not a license to remove the wind event. The event is valid under the frozen protocol and is precisely the type of environment shift the method was supposed to handle.

## 4. Rollout is not the first-order failure

Among the eleven events with active-48 one-step scores, the post-hoc event rankings are strongly aligned:

- Spearman(one-step difference, path-24 difference) = `0.945`;
- Spearman(one-step difference, lead-24 difference) = `0.964`.

Thus events with a transition-level two-flow advantage usually also have a rollout advantage, and events with transition-level harm usually retain it. The main failure occurs before recursive propagation: the transported source-fitted transition law is not consistently better than the one-flow restriction.

## 5. Practical baselines

The two-flow sieve is substantially better than the frozen HGB baselines:

- recursive HGB, path-24: 11/12 positive events, exact `p=0.0415`;
- recursive HGB, lead 24: 11/12, exact `p=0.00586`;
- direct HGB, lead 24: 9/12, exact `p=0.0127`, positive leave-one-event minimum.

These comparisons are encouraging but are not evidence for the structural theorem. The direct HGB fit does not use equal-event sample weights and receives mean/max summaries of future raw weather rather than the full hourly feature path consumed by the sieve. Recursive HGB is visibly unstable under open-loop iteration on several events. Both should be treated as practical references, not as the theorem-matched control.

Damped persistence is the more consequential practical comparator. The two-flow equal-event mean MSE is lower, but only 4/12 path events favor it, the exact test is null, the interval crosses zero, and the leave-one-event minimum is negative. There is no robust claim of superiority to damped persistence.

## 6. Formal conclusion

The result is a valid negative structural confirmation of the frozen implementation on the published data bytes. It does not establish that the two-flow representation has zero target-event value; the post-hoc decomposition in `02_TARGET_ORACLE_DECOMPOSITION.md` shows that target representation gain is usually positive but is outweighed by transport error.