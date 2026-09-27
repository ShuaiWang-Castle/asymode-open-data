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

## 3b. After Stage 0 (2026-09-27)

* The headline regimes are tropical and winter. The keep test's regime-balanced gain averages over these two.
  Synoptic wind, convective and heavy rain may not get worse by more than 2% (seed-averaged point estimate).
* Fewer than three regimes qualify, so the next new work is the host itself (idea I11). It needs a written design
  before it runs. The only pathway work that continues is the running candidate test: the HRRR mechanical subset,
  seed 2, against its ERA5 twin and the host.

## 4. Queue the next work, in this order (skip what is done)

1. The seed-0 queue `jobs_v1_stageB_s0b.txt`: GCRK open, mechanical ERA5, GCRK bounded, mechanical HRRR, full HRRR.
2. Host seeds 1-2, five folds each (Stage 0).
3. Seeds 1-2 of every candidate and its twin (step 3). After Stage 0, only the HRRR mechanical subset (see 3b).
4. The A/A host seeds 3-5 (`jobs_v1_aa_s345.txt`, queued after the candidate).
5. Then the host work of I11, once its design is written.

A new design (a new arm or feature set) is not started by a wake-up. It needs a line in `aris/IDEAS.md` and a reason
written before it runs. The sealed tranche C is never built, read or evaluated.

## 5. Record and push

* One line per decision in `RESEARCH_LOG.md`. `aris/CLAIMS.md` changes only with a reviewer receipt.
* Before every commit, a scan for non-public names, private paths and the PI's email.
* Commit with the attribution line, then push `research/geo-weather-process-20260924` and fast-forward
  `research/tropical-evidence-20260926`. Never push main.

## 6. Report

Two to five lines to the PI, in Chinese:
* what finished;
* each verdict, with its number and interval;
* what runs next, and when it should finish.

If nothing finished and nothing failed, say nothing.

## Paper phase (from 2026-09-27 14:20; revised 15:10)

The PI asked for an open-data manuscript with four forecasters on the development tranche: all zero, TimesFM
(zero-shot, weather covariates), AsymODE (host W+Cin) and AsymODE + GCRK. The geography x weather pathway (hazard
features as a competing hazard) was removed by the PI on 2026-09-27: geography enters only through GCRK or a kernel
architecture in the hidden layer, never through hand-built hazard features. Learned models use five initializations,
averaged in forecast space, but only for a design with a single-seed gain over the host. Every wake-up in this phase:

1. **Look.** Check `runs/geo_weather_20260924/timesfm_v1D/` and the running job, if any. The paper queue
   `jobs_v1_paper.txt` is stopped. No GCRK seeds 1-4 run until a GCRK design beats the host at seed 0 (the bounded
   GCRK: +0.7%). The next design is a kernel extension (spatial coupling of the response state), under discussion
   with the PI. It runs only once its design is written in `aris/IDEAS.md` and agreed.
2. **Evaluate.** Run `paper_v1/evaluate_paper.py` whenever a model's set of seeds grows or TimesFM finishes. It
   writes `results/v1/paper_tables.json`: pooled MAE and RMSE at +1, +6, +24 and +48 hours, design-weighted and
   unweighted, by regime, and RMSE against AsymODE with family-cluster intervals.
3. **Figures and text.** Whenever a model's set of seeds grows, rerun, in this order and at nice 15:
   `paper_v1/counterfactual_v1.py` (GCRK counterfactuals, then `summary`), `paper_v1/figures_v1.py`
   (Figures 1-4), `paper_v1/evaluate_paper.py`. They write `paper_v1/generated/*.tex`, and `main.tex` quotes numbers
   only through those macros, so nothing is copied by hand. `describe_v1.py` and `twins_v1.py` depend on the data
   only and are rerun only if the panel changes. Descriptive analyses use the development tranche only. Text that
   states a direction (abstract, Sections 5.2-5.4, conclusion) is reread against the new numbers each time.
4. **Record, push and report**, as above.

## Kernel comparison (from 2026-09-27 15:45; PI: kernels first, the paper waits)

The PI asked to compare plain AsymODE, AsymODE + GCRK (temporal kernel) and AsymODE + spatio-temporal GCRK (I14)
before any more writing, and only then to move on (a recovery-side kernel is the next idea). Every wake-up:

1. **Look.** `jobs_v1_stgcrk_s0.txt` (STGCRK+Cin, seed 0, five event folds; one process, nice 15; queue log
   `logs/queue_v1_stgcrk.log`) and TimesFM (`runs/geo_weather_20260924/timesfm_v1D/`).
2. **Evaluate.** When the five folds of `v1_stgcrk_s0` are done, run `compare_kernels_v1.py --seed 0` (paired seed 0:
   pooled MAE and RMSE at +1/+6/+24/+48 h, relative RMSE with family-cluster intervals overall, by regime, by initial
   state and without the three worst systems, large-outage peaks, the learned coupling per fold) and
   `evaluate_v1.py --arm v1_stgcrk_s0 --host v1_host_s0` and `--host v1_gcrk_s0` for the regime-balanced numbers.
3. **Decide by rule** (single seed first). The spatio-temporal kernel is a candidate if it beats the host and the
   temporal GCRK at seed 0 (regime-balanced and pooled +1 h). Only then: seeds 1-2 of it, of GCRK and of the host, and
   the spatial null (neighbours replaced at matched distance). Otherwise report and discuss the next kernel with the PI.
4. **Record, push and report**, as in sections 5 and 6. When TimesFM finishes, rerun `paper_v1/evaluate_paper.py`
   so its tables are current, but do not edit the manuscript.
