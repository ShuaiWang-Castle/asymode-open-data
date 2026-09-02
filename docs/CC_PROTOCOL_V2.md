# Protocol v2 — corrected evaluation protocol (branch `aistats-theory-protocol-repair`)

Everything below is implemented in `experiments/exp05_real_dynamics.py` (reference
harness) and mirrored in `exp06/07/08/10`; unit-tested in `tests/`; enforced by
`scripts/check_comparable.py`, which fails closed on any digest mismatch.

## Randomness namespaces (never overloaded)

| field | decides | default |
|---|---|---|
| `outer_split_seed` | which units are held out | 0 |
| `inner_split_seed` | the early-stopping holdout inside each training set | 0 |
| `model_seed` (`--model-seeds`, alias `--seeds`) | initialisation and data order | 0 1 2 |
| `bootstrap_seed` | resampling in the reviews | 0 |

The outer assignment is computed **once** per run from `outer_split_seed` only
(`exp05.outer_assignment`), persisted to `configs/splits/<unit>_k<k>_s<seed>_<digest>.json`
and named in every result (`outer_split_digest`, `split_file`). Test:
`tests/test_harness_protocol.py::test_outer_assignment_invariant_to_model_seed`.

## Outer protocols

* **Primary — `split_unit = event`.** Whole storm panels held out; a deterministic
  greedy balance on sample counts (largest first, seeded tie-break); no event on
  both sides (asserted). For family runs (`exp06`) k is capped by the number of
  storms in the family (leave-one-event-out for the small families) and the cap
  is recorded.
* **Secondary — `split_unit = county`.** One fixed county → fold map shared by all
  arms and seeds; labelled "unseen-county generalisation within observed event
  families".

Inner early-stopping holdout: by the outer unit (events under the event
protocol, counties under the county protocol), never by row.

## Clock

`--clock utc_hour` (default): sin/cos of the UTC hour of day of each forecast
step, from the panel timestamps (EAGLE-I `run_start_time` is GMT per the release
README; verified in-data on DST days). `--clock none`. `--clock lead_phase_old`
is the legacy channel (lead time mod 24 h) — **diagnostic only, never a paper
arm**; its digest differs, so the checker refuses to compare it with the others.
No crew/daylight interpretation is available without local civil time, which is
not implemented.

## Result schema v2 (`asymode.schema.result_header`)

`experiment_id, created_utc, source_commit_at_launch, dirty_at_launch, panel_ids,
panel_digest, channel_names, channel_digest, clock_definition, clock_digest,
split_unit, outer_split_digest, outer_split_seed, inner_split_seed, model_seeds,
mask_definition, mask_digest, metric_definition, metric_digest, hyperparameters,
wall_time_s, convergence{n_fits, n_at_epoch_cap, max_frac_pred_zero, max_frac_pred_one}`
plus the legacy keys. Source fingerprint pinned at launch (import time).

## Statistical reporting (E3)

* Inferential unit = **event**. `scripts/event_level_review.py` computes per-event
  paired MSE differences from OOF exports on identical samples (asserted), the
  mean/median event effect, an event-cluster bootstrap 95% interval (B = 2000),
  the leave-one-event-out range, every event's effect, and — separately, as an
  optimisation diagnostic — seed sign consistency.
* No fold × seed t-test is presented as confirmatory. Legacy `paired_review`
  t-values are reproduced for audit only.
* Family comparisons show every event; three tropical events are three points.

## Metrics

Primary (locked): all-cell RMSE over observed hourly cells at h = 1/6/24/48.
Secondary, frozen before corrected results are inspected (to implement in the
review script, not in the harness): positive-target/event-conditioned RMSE;
integrated customer-hours-out error over the window; restoration-crossing-time
error (first hour below 0.01 after the peak); trajectory reversal diagnostic
(D-4 statistic). Exploratory: anything else, labelled.

## Gates before any graded run

1. `pytest -q` (currently 1,738 passing) — theory identities, splits, clock,
   schema, comparability, OOF uniqueness;
2. smoke on two panels, k = 2, one seed, one epoch, to `results/smoke/`
   (gitignored) — done for exp05/06/07/08/10, all five mutually comparable;
3. all folds, one model seed, key arms (`results/v2/…_seed0.json`), inspect
   convergence, masks, OOF uniqueness, digests;
4. the registered multi-seed run.

## First corrected arm set (E1)

`susceptible` (two-rate, corrected protocol), `net_scaled`, `transmission`,
`transmission_seed`, `damped_persistence` (exp05), and
`trees_matched` (exp07, same information, cap lifted to 2,000 rounds with early
stopping). Dead architecture hypotheses are not rerun.
