# ASYMODE_TRAJECTORY_MAINLINE_R1_20260909 — executed run

Everything below was actually executed on this host. Nothing is a plan.

## Status

| item | status |
|---|---|
| package integrity (28 files) | **PASS** |
| local data discovery + identity | **PASS** — 192/192 canonical array hashes exact |
| ledger conservation audit | **PASS** — 48 balance pairs, max error 1.19e-16 |
| window counts vs locked table | **PASS** — all six partitions exact |
| packaged preflight | **PASS** |
| §6 correctness gates | **PASS** — 4/4 arms |
| E1 main comparison (DIRECT/NET/ASYM + PERSISTENCE) | **RUN** — 36 + 24 stage trials, 15 final refits |
| A1 SR ablation | **RUN** |
| A2 ASYM_STEP_ONLY ablation | **RUN** |
| E2 controlled case | **RUN** |
| independent re-scoring | **PASS** — max disagreement 2.7e-13 |
| repository push | **NOT DONE** (forbidden by protocol) |

## Headline numbers, 2019 REUSED RETROSPECTIVE EVALUATION

Company-equal path MSE, mean over the five paired final seeds.

| model | path MSE | 1h | 6h | 24h | out-of-bounds |
|---|---|---|---|---|---|
| PERSISTENCE | 3.0558e-04 | 1.3417e-04 | 3.0447e-04 | 3.3635e-04 | 0 |
| DIRECT | 1.69622e-04 | 1.0900e-04 | 1.70130e-04 | 1.77274e-04 | 6.74% |
| NET | 1.69382e-04 | 1.0822e-04 | 1.69659e-04 | 1.76714e-04 | 4.87% |
| **ASYM** | **1.69038e-04** | 1.1255e-04 | 1.70086e-04 | **1.75608e-04** | **0** |

Mean-company-relative gain against DIRECT, computed per seed then aggregated:

| arm | mean | range over seeds | seeds positive |
|---|---|---|---|
| NET | +0.366% | -0.287% .. +0.580% | 4/5 |
| ASYM | +1.242% | +0.843% .. +1.953% | 5/5 |
| ASYM vs NET | +0.878% | +0.267% .. +1.790% | 5/5 |

## Ablations (2019, matched ASYM configuration)

| model | path MSE |
|---|---|
| ASYM (main arm) | 1.69038e-04 |
| ASYM_STEP_ONLY | 1.69023e-04 |
| SR (signed init -0.05, chosen on 2018 only) | 1.82853e-04 |

## Controlled E2 (clean-truth MSE, 5 dataset seeds)

| split | DIRECT | NET | ASYM |
|---|---|---|---|
| source_eval | 1.529e-07 | **3.181e-08** | 9.175e-08 |
| high_eval (shifted Y0) | 1.473e-05 | 6.372e-07 | **1.781e-07** |

ASYM beats DIRECT on the shifted split in 5/5 seeds and NET in 4/5.

## What this does and does not support

ASYM is ahead of both DIRECT and NET on the real cohort in every paired seed, so
§9's first clause applies: this supports a **trajectory structure gain for this
cohort, this information contract and this budget**. It is not evidence of
universal two-flow optimality, and no physical hazard claim follows from
net-only supervision.

The margin is small (+1.24%) and the descriptive 168h block interval for ASYM
(+0.29%, +1.24%) overlaps NET's (+0.24%, +0.82%), so the two structured arms are
not separated by that sensitivity analysis. ASYM is also worse than both
baselines at the 1-hour endpoint. The controlled task shows a much larger and
cleaner structural effect, but only under initial-state extrapolation.

2019 is a repeatedly used retrospective evaluation. It is not a fresh, unseen or
confirmatory holdout, and the five seeds are paired replicates, not independent
samples.

## Files

`MAIN_REAL_RESULTS.csv`, `MAIN_COMPANY_SEED.csv`, `PAIRED_PER_SEED.csv`,
`PAIRED_SUMMARY.csv`, `BLOCK_SENSITIVITY.json`, `ABLATION_RESULTS.csv`,
`CONTROLLED_RESULTS.csv`, `PREDICTIONS_2019.npz` (all raw predictions + truth),
`STAGE1_TRIALS.csv`, `STAGE2_TRIALS.csv`, `TRIAL_REGISTRY.json`,
`LEARNING_CURVES.json`, `DATA_AND_INFORMATION_CONTRACT.json`,
`SPLIT_MANIFEST.json`, `../selection_lock.json`, `resource_plan.json`,
`INDEPENDENT_RESCORE_DIFF.json`, `NEGATIVE_AND_FAILURES.md`,
`appendix_evidence/`, `ENVIRONMENT.json`, `SHA256SUMS.txt`, `../REPRODUCE.sh`.

## Note on the published copy

`PREDICTIONS_2019.npz` (235 MB of raw prediction and truth arrays) exceeds
GitHub's 100 MB per-file limit and is **not** included in this repository copy.
Per §10 the outputs remain fully reconstructible: the fifteen final checkpoints
in `main/` (1.5 MB), the frozen origin file `preflight/origin_hours_L24_H24.npz`,
the verified `data_resolution/` manifest and `REPRODUCE.sh` regenerate the arrays
deterministically. `results/MAIN_COMPANY_SEED.csv` and the per-seed `GROUP_*.csv`
files retain every company- and group-level number the main table is built from.

Absolute filesystem paths in `data_resolution/`, `logs/` and
`DATA_AND_INFORMATION_CONTRACT.json` were replaced with `$D` / `$A` / `$W`
placeholders for this published copy. All SHA-256 values are unchanged.
