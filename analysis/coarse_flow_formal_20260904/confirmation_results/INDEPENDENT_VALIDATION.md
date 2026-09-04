# Independent validation of the one-shot coarse two-flow confirmation

Performed after the confirmation run and before the results commit. Every check
was recomputed independently of the runner's own reporting path.

| # | check | verdict |
|---|---|---|
| 1 | exactly 12 confirmation event IDs in the path table | **PASS** |
| 2 | only 2018--2021 events enter standardization, K-means, rates, damping, HGB | **PASS** |
| 3 | `K=8`, the 24 features, caps, and HGB parameters match the lock | **PASS with one documented discrepancy** |
| 4 | two-flow source training loss no larger than one-flow in every cluster | **PASS to solver tolerance** |
| 5 | event-level paired differences recomputed directly from `EVENT_METRICS.csv` | **PASS** |
| 6 | exact sign-flip p-values obtained by enumerating all sign assignments | **PASS** |
| 7 | bootstrap resamples storm events, not rows | **PASS** |
| 8 | no 2022/2024 outcome influenced a branch, cluster count, feature, origin, baseline or hyperparameter | **PASS** |
| 9 | no online cross-county confirmation was run | **PASS** |
| 10 | no manuscript or evidence ledger was edited | **PASS** |

## 1. Confirmation event set

The path table contains exactly twelve event IDs, all in 2022 or 2024:
`2022-01-16, 2022-03-12, 2022-04-13, 2022-06-08, 2022-06-17, 2022-07-23,
2024-01-09, 2024-01-12, 2024-05-08, 2024-05-26, 2024-06-26, 2024-09-27`.

The active-48 one-step endpoint carries **eleven** events, not twelve.
`2022-06-08` has its NOAA county-footprint peak at panel hour 166, so the fixed
48-transition window falls outside the published panel. Per the lock the event is
retained as unavailable and its window was **not** shifted or replaced. It still
contributes to the full-panel path and h+24 endpoints.

## 2. Source isolation

`RUN_PROVENANCE.json` lists fourteen source events spanning 2018--2021 only. No
2022 or 2024 row entered standardization, K-means, rate fitting, damping, or
either HGB baseline.

## 3. Locked configuration

Verified directly against the committed implementation: `K=8`; 24 features in the
locked order; `CAP_U=0.265`; `CAP_R=0.25`; `MiniBatchKMeans(random_state=0,
n_init=5, max_iter=200)`; recursive HGB `max_iter=200`; direct HGB `max_iter=250`;
both with `learning_rate=0.05`, `max_leaf_nodes=31`, `min_samples_leaf=100`,
`l2_regularization=1e-4`, `random_state=0`.

**Discrepancy, reported rather than repaired.** `00_FORMAL_LOCK.md` specifies
`batch_size=8192` for `MiniBatchKMeans`; the locked implementation uses
`batch_size=4096` (and additionally `reassignment_ratio=0.0`, which the lock does
not mention). The implementation was **not** modified. It is the implementation
that produced the packaged development evidence: the development reproduction
gate reproduced all three locked values to `<= 3.4e-17`, far inside the `1e-10`
tolerance, which would not occur under a different `batch_size`. The lock text
therefore appears to carry a stale parameter value. This is recorded for the
principal investigator to adjudicate; no number in this run was changed by it.

## 4. Nested training loss

The one-flow class is the two-flow box with one rate pinned to zero, so the exact
constrained optimum must satisfy `J_two <= J_one` in every cluster. Recomputed on
the source pool after a deterministic source-only refit:

| cluster | J_two | J_one | J_two − J_one | relative |
|---:|---:|---:|---:|---:|
| 0 | 2.419138e-04 | 2.419138e-04 | −2.711e-20 | −1.1e-16 |
| 1 | 6.222337e-04 | 6.233505e-04 | −1.117e-06 | −1.8e-03 |
| 2 | 3.093496e-04 | 3.097079e-04 | −3.583e-07 | −1.2e-03 |
| 3 | 1.241523e-03 | 1.241523e-03 | +2.168e-19 | +1.8e-16 |
| 4 | 3.800875e-04 | 3.806214e-04 | −5.339e-07 | −1.4e-03 |
| 5 | 5.098967e-05 | 5.098963e-05 | **+3.758e-11** | **+7.4e-07** |
| 6 | 4.024410e-04 | 4.040528e-04 | −1.612e-06 | −4.0e-03 |
| 7 | 1.209862e-04 | 1.211096e-04 | −1.234e-07 | −1.0e-03 |

Seven clusters satisfy the inequality outright. Cluster 5 exceeds it by
`3.8e-11` absolute, `7.4e-07` relative. That cluster's two-flow fit is
`U = 2.93e-07`, `R = 8.23e-03`, i.e. it has already collapsed numerically onto
the one-flow restoration ray, and the gap is convergence slack in
`lsq_linear(..., lsmr_tol="auto")`. It is a solver tolerance artifact, not a
violation of the nesting property, and it is far below any reported effect size.

## 5--6. Recomputed inference

Every paired difference in `PAIRED_RESULTS.csv` was recomputed by pivoting
`EVENT_METRICS.csv` directly; maximum absolute disagreement `9.3e-17`. All ten
sign-flip p-values were reproduced exactly by enumerating the full `2^n` sign
assignment space (`n = 11` or `12`), with no disagreement above `1e-12`.

## 7. Bootstrap unit

`bootstrap_ci` resamples the event-level difference vector, whose length equals
the number of events (11 or 12), with `B = 50,000`. It never indexes transition
rows. The inferential unit is the storm event throughout; no row-level test is
reported.

## 8. Chronology

The implementation and documentation were committed as `979adcc` and pushed to
`open-audit-20260904` **before** the confirmation stage was executed, so the
frozen configuration carries a server-side timestamp preceding any 2022/2024
number. The development reproduction used 2018--2020 sources and 2021 targets
only. No branch, cluster count, feature, origin, baseline, or hyperparameter was
altered at any point after the confirmation output existed.

## 9. Multiplicity

Only the coarse temporal-transfer confirmation was executed. The online
cross-county confirmation locked at `a0b46f47` was not run, and no directory for
it was created. The three 15-minute locks present on the branch were not touched.

## 10. Write scope

The working tree touches only `analysis/coarse_flow_formal_20260904/`. No
manuscript file, no result macro, and no `RESULTS_LEDGER.md` entry was modified.
`00_FORMAL_LOCK.md` was not overwritten.

## Environment

Python 3.11.6 with the pinned stack `numpy==2.3.5`, `pandas==2.2.3`,
`scipy==1.17.0`, `scikit-learn==1.8.0`, `pyarrow==18.1.0`, `pytest==9.0.2`.
`tabulate==0.10.0` was additionally installed: it is a pandas optional dependency
required by `DataFrame.to_markdown` to emit `REPORT.md`, is used only for text
formatting, and enters no computation.

Package integrity: external SHA-256
`1675cfb8aaa556ed25fc0330f1da256fd36f804b7a2592290c1f5970df332e92` as supplied
out of band, matched; internal `SHA256SUMS.txt` 23/23 verified. The copy of
`03_CODEX_GPT_WORK_PROMPT.md` inside the package quotes a different external
SHA-256 (`d4a37dcd...`); an archive cannot contain a correct hash of itself, and
that value is stale from an earlier build. The out-of-band hash was treated as
authoritative and the per-file manifest independently confirms every byte used.

Public data: `main@8dd47c5ccd829611f27b69a3d64c274a0a24c400`, 60/60 files
verified against `data/SHA256SUMS.txt`, manifest digest `db286b4960a4`.
