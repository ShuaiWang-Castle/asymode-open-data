# Negative results, failures and scope not run

This file records every failure, restart and removal in this round, including the ones
that were caught and fixed before any result existed. Outcome-level negative results are
added after evaluation; nothing here is deleted to make the record look cleaner.

## Data problems found at intake

1. **Missing county weather encoded as zero.** 1,265 of 2,625 panel counties carry an
   all-zero driver tensor in every channel and hour. Their windows are excluded for both
   models, removing 30,479 train, 5,231 validation and 16,800 evaluation windows that had
   49 legal states. Task A therefore covers only counties with real weather
   (`intake/SCHEMA_DRIFT.md`, `intake/MISSING_DATA_REQUEST.md`). This is a population
   restriction, not a repair.
2. **Observed but non-finite targets.** 41,297 `observed=True` 15-minute cells have
   non-finite `y`; they are illegal states and are never filled.
3. **Bidirectional gap filling in the driver build.** The weather inputs are reanalysis
   with two-sided temporal interpolation. They cannot be described as forecasts available
   at the origin.

## Implementation failures caught before any result

4. **Corrupted intake documents.** The first `SCHEMA_DRIFT.md` and
   `MISSING_DATA_REQUEST.md` were written through an unquoted shell heredoc, which executed
   every backtick span and expanded `$W`. Both were regenerated from a quoted script
   (`code/intake_docs.py`), and every sibling JSON and CSV was re-parsed and spot-checked
   before commit 1.
5. **A correctness test failed for a real reason.** The NET affine-representation test
   first failed by about 1e-9 per step: the audited core stores `state_scale` as a float32
   buffer before `.double()`, so a scale of 0.04 is not exact. The test now uses 0.0625,
   which is exact in binary. The model code was not changed.
6. **A scoring test compared floats with `==`** and failed; it now uses an absolute
   tolerance of 1e-15.
7. **Absolute paths in pytest output.** `logs/PYTEST.txt` contained scratch paths in
   warning lines and was sanitised before commit 2.
8. **pandas `DatetimeIndex` accessor bug** in the first split build (`.to_numpy()` on an
   ndarray). It failed loudly and was fixed before any lock was written.

## Execution failures during development tuning

9. **3-8x slowdown.** The first development trials ran far slower than profiled because
   the host was shared with other sessions' long jobs and this round's workers ran
   uncapped OpenMP/BLAS pools. Only this round's workers were stopped and relaunched with
   thread caps; completed trials were kept and in-flight trials rerun
   (`logs/RESTARTS.md`). The other sessions' jobs were not signalled.
10. **Two stop attempts failed closed.** Under zsh the PID list did not word-split, and
    macOS bash 3.2 has no `mapfile`; in both cases the per-process guard refused to signal
    anything. The third attempt used plain bash 3.2.

## Selection limitations recorded before any final result

11. **Synthetic selection hit the grid corner for both models.** NET and ASYM each chose
    the largest learning rate (3e-3) with the largest parameter budget (32768). The best
    configuration may lie outside the locked grid; the protocol does not allow extending the
    sweep, so the final grid uses these corner configurations as locked.
12. **NET's synthetic choice is a near-tie.** Its winner leads the runner-up
    (p32768_lr0.001) by 0.039% in mean validation MSE; ASYM's margin is 0.224%. Between-seed
    spread (about 5.8e-5) is two orders of magnitude larger than NET's top-two difference,
    because each development seed draws its own training and validation events and
    validation MSE against realized events is dominated by irreducible noise.

13. **Orchestrator exit codes were logged wrongly.** Three orchestrator scripts wrote
    `exit=$?` after a `$(date)` substitution in the same string, recording the status of `date`.
    `synth_select exit=0` and, at the PI-requested stop, `synth_final exit=0` and `aneel_eval exit=0`
    were therefore uninformative (the stopped jobs were really SIGKILLed, status 137). The log is
    annotated append-only, synthetic selection success is established by the committed lock, and
    the scripts now capture `e=$?` first. A PI-requested pause was also converted into a stop to
    free memory; completed trials were kept (`logs/RESTARTS.md`).

13b. **US selection also sits at the capacity edge, and ASYM peaks very early.** Both models
    chose the largest parameter budget (32768) with lr 1e-3. NET's choice is clear (7.6% ahead
    of its runner-up); its per-seed best updates were 1000, 2750 and 3000, so one seed peaked at
    the 3000-update cap. ASYM's choice is narrow (0.34%); its per-seed best updates were 1750,
    250 and 1000, one of them at the first validation point, which gives a locked final budget of
    only 1000 updates. The locked rule was applied unchanged; this is recorded, not corrected.

## Failures after results existed

14. **The post-hoc descriptive script crashed once** on `cv.update`, which pandas resolves to the
    `DataFrame.update` method rather than the column. It failed before writing any output and now
    indexes the column explicitly. `code/posthoc_descriptives.py` was written after the evaluation
    results had been seen; it adds descriptive references only (persistence, untrained models,
    validation curves, compute totals) and changes no lock, estimand or selection.

## Outcome-level results unfavourable to ASYM

### Task A (US EAGLE-I + ERA5, 2024 retrospective, given reanalysis weather)

- The pooled result favours ASYM, but not everywhere: 198 of 1,240 county-events have Delta < 0,
  and the largest county-level NET advantages (Delta down to -1.62e-02) sit in 2024-09-27, the
  same event that carries most of the pooled gain.
- In four of six events one of the five seeds favours NET.
- In the train-defined cell with current outage above the train 90th percentile and given future
  24 h maximum gust below the train lower tercile, NET is better (window-mean Delta = -6.13e-04,
  4 of 5 seeds). The cell rule uses inputs and train quantiles only; the result is descriptive.
- Post-hoc: on 2022 validation the best development checkpoint of ASYM is, on the seed mean, only
  1.1% below its untrained initialization (a relaxation towards the train source mean), and the
  final fits are above initialization at every recorded update on the seed mean (+10% at the
  locked 1000 updates). On 2024 the trained ASYM is worse than its own initialization in four of
  six events (2024-01-09, 2024-01-12, 2024-05-08, 2024-05-26); its net gain comes from 2024-09-27.
- Post-hoc: in five of six 2024 events the untrained ASYM (relaxation towards the train source
  mean) already beats the untrained NET, by more than the trained Delta in three of them
  (2024-01-12, 2024-05-26, 2024-06-26). Only in 2024-09-27 does training create the ASYM
  advantage, so part of the low-outage-event advantage is present before any learning.
- Only five independent overlap components exist: the exact sign-flip p cannot go below 0.0625,
  and removing 2024-09-27 halves the pooled Delta (+3.23e-04 to +1.54e-04).

### Task B (ANEEL, reused 2019 retrospective evaluation)

- The company-equal path risk favours **NET**: Delta = MSE_NET - MSE_ASYM = -6.44e-07 raw
  (168 h block interval [-1.12e-06, -1.33e-07]) and -8.19e-07 after uniform output clipping
  ([-1.22e-06, -4.38e-07]). Only 1 of 5 paired seeds is positive, and every leave-one-seed-out
  mean stays negative.
- ASYM is worse at every endpoint: 1 h (-5.35e-06), 6 h (-1.59e-06) and 24 h (-1.84e-07).
- The two relative summaries disagree in sign: mean-company-relative gain +0.066% (towards
  ASYM) against ratio-of-means gain -0.38% (towards NET). They are reported separately and
  never substituted for each other.
- The NET advantage is concentrated: the three companies with the largest |Delta| all favour
  NET and carry 115% of the summed company Delta, while 9 of 16 companies and 15 of 24
  collections show small ASYM advantages. Neither direction holds company by company.

### Task C (synthetic)

- Only one of nine main-grid cells (law 2, n = 32) has all five seeds favouring ASYM. The other
  eight have 2-4 positive seeds and exact sign-flip p >= 0.125.
- Law 0 at n = 512 favours NET on the seed mean (-1.01e-06; one seed at -6.07e-06).
- The gamma = 0 control at law 1 favours NET (-5.69e-06, 4 of 5 seeds, p = 0.1875).
- At the 6 h endpoint NET is better in 7 of 9 main-grid cells.
- Calibration-based selection did worse than always-ASYM in two of twelve cells (gamma = 0 law 2,
  and gamma = 0.04 law 2 n = 32), and worse than both fixed choices in gamma = 0 law 2.

## Outcome-level results unfavourable to NET

### Task A

- NET is worse than ASYM on the seed mean in all six 2024 events, in all five seeds and in all
  five overlap components, and stays worse after uniform output clipping.
- NET is worse than a zero-training persistence path at the 1 h (2.08 times) and 6 h (+19%)
  endpoints, and on the whole path in 2024-01-09 and 2024-05-08.
- 33.9% of NET's raw outputs are below 0 (28.8% below -0.001). Clipping recovers only 0.9% of
  its path MSE, so the negative drift is not what separates the two models.

### Task B

- None at the pooled level, where company-equal risk favours NET. At company level, 9 of 16
  companies and 15 of 24 collections show small ASYM advantages.

### Task C

- In law 2 at n = 32 all five seeds favour ASYM (Delta vs mu0 = +2.91e-05, 25% of NET's
  estimation error), and at the 24 h and 32 h endpoints ASYM is better in 9 of 9 main-grid cells.
- NET leaves [0, 1] in 1.9%-12.1% of its synthetic outputs; ASYM cannot by construction.

## Scope not run in this round, by protocol

- No DIRECT, SR, source-rate projection, geographic kernel, neighbour model, intervention
  or certificate work.
- No ANEEL retuning, no DIRECT/SR/one-step ablations, no change to the repaired ANEEL
  configurations, seeds or update counts.
- No driver data rebuild. Full-coverage US results require a repaired driver release.
- No CUDA check: this host has no CUDA device, and every run is CPU FP32.
