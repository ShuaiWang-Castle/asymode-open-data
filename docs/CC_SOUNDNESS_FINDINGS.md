# Soundness findings — verified independently, with code references

Status of each prompt item R1–R10 after inspection. "Confirmed" means the defect
was reproduced from the code or the text as it stands on `3ec2a60`.

| id | finding | status | where |
|---|---|---|---|
| R1 | clock encodes lead time, not hour of day | **confirmed** | `experiments/exp05_real_dynamics.py:162-169` — `t = np.arange(T)` per window; consumed by exp06:128, exp07:398, exp08 (`clock` family, :87/:145/:161), exp10:151 |
| R2 | outer folds change with the model seed | **confirmed** | `make_folds(..., seed=seed)` inside the seed loop: exp04:110, exp05:372, exp06:155, exp07:417, exp08:415, exp10:158; `evalproto.make_folds:26-32` hashes `f"{seed}:{fips}"`; inner split at `evalproto:54` uses `1000 + 10*seed + fold` |
| R3 | fold × seed cells treated as 15 independent samples | **confirmed** | `scripts/paired_review.py:61` t = mean/(sd/√n) over (seed, fold) units; ledger tables quote these t values |
| R4 | Prop. 5 saturation bound over-strong | **confirmed** by counterexample | `paper/DRAFT.md:87`; `docs/THEORY_PLAN.md` Prop 5; test `test_constant_rate_saturation_bound_fails_without_lower_rate_bound` (linear growth at zero total intensity) |
| R5 | "Fisher information" used without a likelihood | **confirmed** | `paper/DRAFT.md` §4 Prop 2 ("Fisher information determinant"); `docs/THEORY_PLAN.md` Prop 2 |
| R6 | variance ratio called precision ratio | **confirmed** | `paper/DRAFT.md:79` (Cor 2a "precision ratio is large"), `docs/THEORY_PLAN.md:56,108` |
| R7 | Kish ESS presented as the two-rate information ratio | **confirmed** | `paper/DRAFT.md:107` ("exactly as Cor. 2a says"), abstract line 9 ("effective sample size for the two rates 93:1"); `RESULTS_LEDGER.md:551-557` (97:1), :666 (93:1) |
| R8 | cross-county pooling asserted as fact | **confirmed** | `paper/DRAFT.md` abstract ("separability is supplied mostly by the cross-section"; "both are visible in the data"); Cor 2b marked [P] but stated as consequence |
| R9 | exact CT interpretation available; "Euler/ODE" wording | **confirmed** mapping (test 7); wording: repository and title use "ODE" only in the project name and related-work; the draft calls it a "two-rate compartmental model" — keep "discrete-time transition model" consistently (D0) | `src/asymode/dynamics.py:8` |
| R10 | fall/rise ratio is not the theoretical quantity | **confirmed** as a gap | `paper/DRAFT.md` §7 "Reading"; H-E registration uses the duration ratio; the theorem quantity is `v(x)·min(R²/A, U²/B)` — unmeasured |

Additional findings made while auditing:

* **F1 — within a seed, all arms share the same folds.** Paired within-(seed,
  fold) comparisons in the archive are valid *as paired comparisons*; what R2
  breaks is the interpretation of the three seeds as replicates on one test set
  and the pooling of the 15 cells into one t-statistic (R3).
* **F2 — `paired_review` could not reproduce EXP07/EXP10 without `--against`**
  (the two-rate `control` lives in the exp08 file) and could not reproduce EXP06
  at all (per-family rows collide) — fixed by a `--family` filter. Recorded in
  `docs/CC_ARCHIVE_REPRO_AUDIT.md`.
* **F3 — the source fingerprint used to be captured at write time**; fixed at
  `c86497b` (pinned at import). One archived result (`exp08_ha3_g1panels_14ch.json`)
  carries a write-time commit; the ledger entry says so.
* **F4 — clipping is redundant for the two-rate arm** (u, r ≤ 0.25 ⇒ the map is
  affine [0,1]→[0,1], test 6) but not for `net`/`net_scaled`; clamp activations
  are not counted per arm. To add to the schema's convergence block.
