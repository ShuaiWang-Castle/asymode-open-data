# ARIS cycle — what every scheduled wake-up does (2026-09-27)

Long-running work on the designed panel follows the ARIS split (Auto-claude-code-research-in-sleep,
`skills/shared-references/external-cadence.md`). The scheduler only decides **when** to look. It never decides a result:
* keep and discard follow program.md rule 3b and DATASET_DESIGN amendment 2 S10;
* a claim needs an independent reviewer receipt (`aris/CLAIMS.md`).

Every wake-up runs the steps below in order and stops early if nothing changed.

## 1. Look

* Folds done per label (`runs/geo_weather_20260924/<label>/fold0k/DONE.json`), the running job, and the queue logs
  (`logs/queue_v1_*.log`).
* Load average and memory.
* **One training process at a time** (the PI uses the machine). Everything runs at the lowest priority (nice ≥ 15).
  Never touch processes this session did not start.

## 2. Evaluate what finished

For every label with five folds done and no result file, run `evaluate_v1.py` against its registered comparators and
write `results/v1/screen_s<seed>_<arm>_vs_<ref>.json`.

| arm (label) | comparators |
|---|---|
| v1_gcrkopen_s0 (GCRK+Cin, opening unbounded) | host; v1_gcrk_s0 (bounded twin) |
| v1_gcrk_s0 (GCRK+Cin, opening bounded) | host |
| v1_Hmech_era5_s0 (mechanical-load subset, ERA5) | host; v1_Hera5_s0 (full ERA5 dictionary) |
| v1_Hmech_hrrr_s0 (the same subset, HRRR) | host; v1_Hmech_era5_s0 (source twin) |
| v1_Hhrrr_s0 (full HRRR dictionary) | host; v1_Hera5_s0 (source twin) |

## 3. Decide by rule

* **Single-seed screen (seed 0).** Discard an arm if its regime-balanced relative change against the host is ≥ 0.
  Otherwise it is a candidate: queue seeds 1 and 2 for the arm, for its twin, and for the host.
* **Three-seed keep** (seed-averaged predictions). All of the following must hold:
  * the regime-balanced improvement against the host exceeds 1%, with a 95% family-cluster interval that excludes 0;
  * the improvement against the twin is > 0, also with an interval that excludes 0;
  * no non-headline regime gets worse by more than 2%.
* **Stage 0 (DATASET_DESIGN 9.3).** Host seeds 1-2 are needed anyway: they decide the headline regimes. Seeds 3-5 are
  the A/A run, which decides the seed count.

## 4. Queue the next work, in this order (skip what is done)

1. The seed-0 queue `jobs_v1_stageB_s0b.txt`: GCRK open, mechanical ERA5, GCRK bounded, mechanical HRRR, full HRRR.
2. Host seeds 1-2, five folds each (Stage 0).
3. Seeds 1-2 of every candidate and its twin (step 3).
4. The A/A host seeds 3-5.

A new design (a new arm or feature set) is not started by a wake-up. It needs a line in `aris/IDEAS.md` and a reason
written before it runs. The sealed tranche C is never built, read or evaluated.

## 5. Record and push

* One line per decision in `RESEARCH_LOG.md`. `aris/CLAIMS.md` changes only with a reviewer receipt.
* Before every commit, a trace scan for competition names, private paths and the PI's email.
* Commit with the attribution line, then push `research/geo-weather-process-20260924` and fast-forward
  `research/tropical-evidence-20260926`. Never push main.

## 6. Report

Two to five lines to the PI, in Chinese:
* what finished;
* each verdict, with its number and interval;
* what runs next, and when it should finish.

If nothing finished and nothing failed, say nothing.
