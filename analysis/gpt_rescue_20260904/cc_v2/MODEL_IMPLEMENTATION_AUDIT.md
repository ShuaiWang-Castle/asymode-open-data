# Model implementation audit

Every number here was produced by running the code in this branch. Checks that
passed are reported with their numbers; checks that failed are reported as failures
and no protocol was replaced to make them pass.

## Passed

| check | requirement | measured |
|---|---|---|
| exact bounded two-flow solver | no grid point beats the closed form | 400 random problems, worst deficit `+0.000e+00` |
| exact bounded U-ray solver | same | `+0.000e+00` |
| exact bounded R-ray solver | same | `+0.000e+00` |
| update-0 reproduces the constant class | max abs flow error ≤ `1e-6` | **`2.5e-09`** over all nine jobs |
| update 0 is a checkpoint candidate | scored before any gradient step | yes, scored and eligible in 9/9 jobs |
| state preservation is structural | no clamp activation | **0 clamp events** across all rollouts |
| cap condition | `C_U_main + C_bkg + C_R < 1` | `0.25 + 0.015 + 0.25 = 0.515` |
| shared modules | both arms use identical modules and inputs | one class, one `proposals()`; arms differ only in `collapse()` |
| occurrence block separation | not the magnitude input, not a hidden layer | separate tensor, separate `nn.Linear`, 6 dims vs 32 |
| recovery block | no clock, no simulated state | verified by construction in `features.build_blocks` |
| importance-corrected sampler | natural risk preserved | inverse-probability weights, renormalised per event; **0 stratum fallbacks** |
| every module receives finite gradient | when its regime is present | all six modules have nonzero gradient norms |

## Failed or degenerate — reported, not repaired

### 1. The two interruption heads are exact twins for the entire run (blocker for the main run)

`11_IMPLEMENTATION_AMENDMENT.md` §2 requires every output weight set to zero and
**the same `raw_U_bias` for both interruption MLP heads**. Applied literally, the
two heads start numerically identical, see the same input, and therefore receive
the same gradient at every update.

Measured: identical at update 0, and after 30 Adam updates
`max |head_a − head_b| = 0.000e+00`. In the pilot the two heads' gradient norms are
equal to every printed digit (`gn_head_a` mean `2.306011e-08`, `gn_head_b` mean
`2.306011e-08`).

Averaging two identical functions is arithmetically one function. The competition
invariant that "the second interruption network behaves as an ensemble component"
(`08_EVIDENCE_STATUS_AND_COMPETITION_LESSONS.md` §3.6) is **unreachable** under the
amendment's initialization rule. The two documents conflict.

Breaking the symmetry requires changing the initialization specification — for
example random hidden weights with a zeroed output layer, which preserves the
update-0 identity while making the heads distinct. That is a protocol change and
was **not** made here.

### 2. The learned first-order hold receives no usable gradient

`gn_hold` is nonzero on only 16 of 194 recorded validation checkpoints, with mean
`8.8e-15` and max `5.5e-14`, against `gn_rec` mean `2.2e-04`. This follows from the
same initialization: with all magnitude weights zero the raw logit is constant in
time, so the held logit equals the raw logit for any gate value and the hold has no
first-order effect to learn from. The hold is present and correct, but inert under
this initialization.

### 3. The modular initialization cannot represent `U0 > 0.1325`

With the occurrence gate fixed at `g0 = 0.5`, the pulse pathway emits at most
`g0 · C_U_main = 0.125`, while the amendment's proportional split assigns it
`U0 · (1 − share) = 0.9434 · U0`. The representable ceiling is therefore
`0.1325`, well below the nominal `C_U_main + C_bkg = 0.265`. Above it `logit` would
silently clip and the update-0 identity would break.

A `ValueError` guard now fires instead of clipping. It never triggered in the pilot
because every fitted constant is far below the ceiling.

### 4. The exact constant class is restoration-only on this cohort

Pooled over the fit transitions of the source events:

| pilot test event | `U0` | `R0` | `a_ray` | `b_ray` |
|---|---|---|---|---|
| 2024-05-08 | `2.674e-05` | `2.674e-02` | `0` | `2.667e-02` |
| 2022-03-12 | `0` | `1.285e-02` | `0` | `1.285e-02` |
| 2018-10-11 | `2.674e-05` | `2.674e-02` | `0` | `2.667e-02` |

The bounded interruption ray optimum is **exactly zero** for all three, so the
"interruption-ray start" is the all-zero constant model, and where `U0 = 0` the
two-flow start and the restoration-ray start are the *same point*. The solver is
correct: it returns the interior optimum when feasible and pins at the boundary
exactly when the unconstrained value is negative, verified independently.

### 5. The one-flow collapse removes almost nothing on these events

The one-flow arm deletes exactly `c = min(U_tilde, R_tilde)`. Measured at update 0
over a full 24-hour rollout on each pilot test event:

| event | `U0` | `R0` | mean `c` | `c / R0` | fraction with both flows positive |
|---|---|---|---|---|---|
| 2024-05-08 | `2.674e-05` | `2.674e-02` | `2.674e-05` | 0.1% | 1.00 |
| 2022-03-12 | `0` | `1.285e-02` | `1.4e-09` | 0.0% | 1.00 |
| 2018-10-11 | `2.674e-05` | `2.674e-02` | `2.674e-05` | 0.1% | 1.00 |

Both flows are nominally positive everywhere, but `U0` sits three orders of
magnitude below `R0`, so the removed common component is a negligible share of the
transition. **The two arms are therefore near-identical models on these events by
construction, and the pilot has very little power to separate them.** This is a
statement about the pilot's measurability, not about the science.

### 6. The origin rule is degenerate on this cohort

Recorded in full in `origin_rule_audit.md`: 48 of 78 anchors clip to the legal
range boundary, 24 of 26 panels place the `pre` anchor at index 24 and 25 of 26
place the `post` anchor at 143, so the rule yields a nearly fixed grid
`[24, ~85, 143]` rather than event-centred origins.

## Reproducibility

One complete pilot event was rerun with the same seed; the comparison is recorded in
`PILOT_REPORT.md`.
