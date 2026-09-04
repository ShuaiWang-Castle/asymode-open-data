# V2 implementation pilot report

**Branch** `open-audit-20260904` · **commit at start** `11f893b` · worktree clean at start
**Panel digest** `db286b4960a4` · **fold digest** `beb00a6762ba` · **seed** 0
**Jobs** 3 events × (1 two-flow start + 2 one-flow branch starts) = **9**, plus four
deterministic baselines. No main campaign was run.

This is an implementation check. It is not paper evidence and it does not
adjudicate any scientific claim.

---

## 1. Reproduced legacy evidence

Nothing in this pilot re-ran or re-scored the archived event-held-out comparison,
and nothing here changes its status. It remains **reproduced legacy evidence
pending adjudication**. No entry in `RESULTS_LEDGER.md` was edited, no prior result
was labelled refuted, withdrawn or invalid, and the manuscript was not touched.

For the record, the archived confirmatory result was independently re-run earlier on
this branch and reproduced to every printed digit
(`results/cc_event_repro_core_event.json`): h+24 `0.03726` / `0.03873`, h+48
`0.03320` / `0.03446`, relative gains `+4.55%` / `+3.71%`, sign counts 8/11 and
6/11, and all eleven event-level gains. That reproduction stands and is unaffected
by anything below.

The undertrained unified-v2 output remains present and is marked diagnostic. It was
not deleted and was not averaged with any other result.

---

## 2. Newly verified implementation facts

These are properties of the code, established by running it.

| fact | measurement |
|---|---|
| exact bounded two-flow, U-ray and R-ray solvers are correct | 400 random problems each; no grid point beats the closed form (worst deficit `+0.000e+00`) |
| the update-0 model reproduces the exact constant class | worst absolute flow error **`2.5e-09`** across all nine jobs, against a `1e-6` requirement |
| update 0 is a real checkpoint candidate | scored before any gradient step in 9/9 jobs; it was never selected as final, so training improved on it every time |
| state preservation is structural, not enforced | **0 clamp activations** in every rollout; `C_U_main + C_bkg + C_R = 0.515 < 1` |
| both arms share the same modules and inputs | one class, one `proposals()`; the arms differ only in `collapse()` |
| the occurrence gate reads a separate block | 6 dims vs 32, separate tensor and separate module; not a hidden layer of the magnitude network |
| recovery reads no clock and no simulated state | by construction in `features.build_blocks` |
| the Stage-A sampler preserves the natural risk | inverse-probability weights renormalised per event; **0 stratum fallbacks** |
| the run is numerically reproducible | one complete pilot event rerun at the same seed: max abs difference **`0.000e+00`** on selected update, selected validation, path MSE, teacher-forced MSE and h+24 MSE |

### Four defects found, reported and not repaired

Repairing any of these would change the locked specification, so none was changed.

1. **The two interruption heads are exact twins for the whole run.** The amendment
   requires all output weights zeroed and the *same* bias for both heads, so they
   start identical, receive identical gradients, and never diverge:
   `max |head_a − head_b| = 0.000e+00` after 30 Adam updates, and their pilot
   gradient norms are equal to every printed digit. Averaging two identical
   functions is one function, so the "second interruption network as an ensemble
   component" invariant is **unreachable under the amendment's initialization**.
   The amendment and the competition-lessons document conflict here.

2. **The learned first-order hold is inert.** `gn_hold` is nonzero at only 16 of
   194 checkpoints, mean `8.8e-15`, against `gn_rec` mean `2.2e-04`. With all
   magnitude weights zeroed the raw logit is constant in time, so the hold has no
   first-order effect to learn from.

3. **The modular initialization cannot represent `U0 > 0.1325`**, well below the
   nominal cap of `0.265`, because the gate is fixed at `g0 = 0.5`. A guard now
   raises instead of silently clipping. It never fired in the pilot.

4. **The event-centred origin rule is degenerate on this cohort.** 48 of 78 anchors
   clip to the legal boundary; 24 of 26 panels place `pre` at index 24 and 25 of 26
   place `post` at 143. The rule yields a near-fixed grid `[24, ~85, 143]`, so its
   stated purpose — not overweighting quiet pre-storm and late-recovery periods — is
   not achieved. Full detail in `origin_rule_audit.md`.

---

## 3. Pilot-only observations

Three events, one seed. These are **not** evidence about the scientific question and
must not be quoted as a result.

### 3.1 The comparison has almost no power on these events

The one-flow arm deletes exactly `c = min(U_tilde, R_tilde)`. The exact bounded
constant fit on the source transitions gives an interruption flow three orders of
magnitude below the recovery flow:

| event | `U0` | `R0` | mean `c` | `c / R0` |
|---|---|---|---|---|
| 2024-05-08 | `2.674e-05` | `2.674e-02` | `2.674e-05` | 0.1% |
| 2022-03-12 | `0` | `1.285e-02` | `1.4e-09` | 0.0% |
| 2018-10-11 | `2.674e-05` | `2.674e-02` | `2.674e-05` | 0.1% |

The bounded interruption-ray optimum is **exactly zero** on all three, so the
"interruption-ray start" is the all-zero constant, and where `U0 = 0` the two-flow
start and the restoration-ray start are the same point. On 2022-03-12 the two arms
agree to five decimal places at every horizon and stratum.

**Whatever the main run finds, this pilot could not have separated the arms.**

### 3.2 The differences that do appear

One-flow minus two-flow, relative, positive means two flows better:

| endpoint | 2024-05-08 | 2022-03-12 | 2018-10-11 |
|---|---|---|---|
| teacher-forced one-step, full | +0.07% | +0.00% | −0.05% |
| 24-hour path, full | +0.58% | +0.00% | +0.15% |
| 24-hour path, interior (`y0>0.01`) | **−0.39%** | +0.00% | +0.14% |
| h+24, full | +0.55% | +0.00% | +0.50% |

### 3.3 The absolute-usefulness gate fails on one of three events

24-hour path MSE, full sample:

| model | 2024-05-08 | 2022-03-12 | 2018-10-11 |
|---|---|---|---|
| `asym_two_flow` | **4.188e-05** | **1.327e-05** | 5.339e-04 |
| `asym_one_flow` | 4.212e-05 | 1.327e-05 | 5.347e-04 |
| constant two-flow | 4.980e-05 | 1.452e-05 | **3.107e-04** |
| constant one-flow | 4.959e-05 | 1.452e-05 | 3.102e-04 |
| damped persistence | 4.959e-05 | 1.452e-05 | 3.102e-04 |
| all-zero | 1.311e-04 | 2.606e-05 | 5.659e-03 |

Both neural arms beat all-zero everywhere. On the two convective/winter events they
also beat every constant baseline. **On the tropical event 2018-10-11 both neural
arms are about 72% worse than the constant two-flow baseline**, and the two-flow arm
is `−2.23e-04` against its own update-0 model. Per `09_LOCKED_CC_PLAN_V2.md` §8.3,
the learned rates on that event may not be interpreted.

Selected checkpoints were Stage B in 9/9 jobs (updates 1000–3000); the
restoration-ray start won the source-event validation on all three events.

---

## 4. Scientific claims that remain open for the 26-event main run

None of the following is answered here, and none should be treated as leaning either
way on the basis of three events at one seed.

1. Whether retaining two simultaneous nonnegative components improves held-out
   transitions or 24-hour open-loop forecasts relative to collapsing the same
   proposals. The pilot's effect sizes are inside its own noise and its removed
   component is ≤0.1% of the recovery flow.
2. Whether any gain is interior concurrency or boundary directional decoupling.
   The single interior signal here (`−0.39%` on 2024-05-08) is one event.
3. Whether the process-specific architecture transfers from the competition setting.
   The tropical result says it does not transfer uniformly.
4. Whether a broader, event-balanced cohort changes the earlier conclusion.
5. Whether the neural estimator reaches the local information regime the theory
   describes.

### Blockers that should be resolved before the main run

* **the twin-heads conflict** — the ensemble invariant cannot be realised under the
  current initialization rule; this needs a decision, not a silent fix;
* **the inert hold** — same root cause;
* **the degenerate origin rule** — it does not deliver event-centred origins here;
* **the near-zero interruption constant** — if `U0` remains three orders below `R0`
  on the full cohort, the collapse removes almost nothing and the 45-job campaign
  will measure a difference it cannot resolve.

The pilot stopped here as instructed. No five-fold three-seed campaign was run, no
manuscript file was touched, no ledger entry was changed, and no prior result was
relabelled.
