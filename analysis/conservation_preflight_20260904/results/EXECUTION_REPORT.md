# Execution report — GitHub-only conservation and design preflight

**Task:** zero-training design audit, executed from two clean pinned GitHub clones.
**Branch:** `open-audit-20260904`
**Date of run:** 2026-09-04

This report is an independent audit written after inspecting the three generated
tables. It separates (1) exact algebra verified, (2) empirical assumptions that
hold or fail, (3) descriptive design findings, and (4) claims that still require
an estimator experiment.

It assigns **no** scientific promotion or rejection label. Whether the measured
design is informative enough to justify another estimator experiment is reserved
for Shuai.

---

## 0. What this run did and did not do

Performed: SHA-256 verification of every public file; correct adjacent-observation
hourly transition construction; full-panel and one predeclared outcome-blind
active-48 design; event-, source-fold- and local `k=200` constant-fit geometry;
closure, residual scale, `G`, plug-in `Gamma`, `c = min(U,R)` and `c(1-2Y)`;
balanced-flow interval against county reporting resolution.

Not performed, by instruction: no neural training, no model repair, no pilot
rerun, no five-fold three-seed campaign, no outcome-driven selection of any
event, window, row, horizon, metric or comparator, no `y0` row filtering, no
sweep of window length / `k` / horizon, no manuscript, macro, ledger or
prior-evidence-label edit, and no authorization of a follow-up experiment.

The reproduced legacy event-held-out result and the near-null V2 pilot are both
left exactly as previously labelled. Nothing here relabels either.

---

## 1. Integrity and provenance

| gate | result |
|---|---|
| code clone HEAD | `e60cb5505b0513a4bd7cf603dbcfc8a9a74dc97e` (`open-audit-20260904`) |
| expected starting HEAD | matches exactly |
| public-data clone HEAD | `8dd47c5ccd829611f27b69a3d64c274a0a24c400` (detached) |
| code worktree at start | clean |
| audited base `d6555015…` ancestor of HEAD | yes (9 commits ahead) |
| diff `audited_base..HEAD` over `src/ experiments/ configs/` | **empty** |
| public files verified against `data/SHA256SUMS.txt` | **60 / 60 OK, 0 failures** |
| checker-output SHA-256 | `cd6d486491dde3e13cab0b5d05d83a855de9f0dac93cb48d1f8a2e5cf1231bf5` |
| manifest digest | declared `db286b4960a4`, independently recomputed `db286b4960a4` |
| manifest panels | 26 listed, 26 unique |
| channel digest | declared `dec964873cb2`, recomputed `dec964873cb2` |
| frozen fold digest | declared `beb00a6762ba`, independently recomputed `beb00a6762ba` |

All nine commits between the audited base and HEAD touch only `CC_START_HERE.md`
and `analysis/conservation_preflight_20260904/`. No analysis-relevant source,
experiment or config file changed since the audited base.

Every empirical byte came from the two pinned clones. No pre-existing local
checkout, local dataset, checkpoint, cache, uploaded archive, other branch, or
remembered result was read.

**Recorded deviations (both deliberate, neither affects any number):**

1. The virtual environment was created *outside* the code clone rather than
   inside it, so the code worktree stayed clean throughout and the environment
   could not be committed by accident.
2. The public-data clone is a full clone checked out at the pinned commit rather
   than a `blob:none` sparse clone. This is a strict superset of the sparse
   content at the same commit; every file consumed was verified byte-for-byte
   against the published manifest.
3. `shasum -a 256 -c` was used instead of `sha256sum` (macOS host). This is the
   command `data/README.md` itself documents, and it is the same algorithm.

---

## 2. Exact algebra verified

`pytest -q` → **13 passed**, run before any empirical table was inspected.

| required property | outcome |
|---|---|
| 1. weighted unconstrained LS satisfies `U(1-mu) - R mu = mean_delta` to `1e-11` | PASS, worst residual over 40 random weighted problems below `1e-11` |
| 2. `A*B - C^2 = v` to `1e-12` under arbitrary positive normalized weights | PASS |
| 3. explicit boundary-constrained counterexample violates the unconstrained equality | PASS — unconstrained residual `<1e-11`, box-fit residual `>1e-6`, box fit confirmed optimal against a dense grid |
| 4. under exact zero drift and `mu<=1/2`, `G = v R^2 mu^2 / ((1-mu)^2(mu^2+v)) <= R^2 min(mu^2,v)/(1-mu)^2` | PASS over 300 feasible draws |
| 5. `m_two(y) - m_one(y) = min(U,R)(1-2y)` | PASS, max error `<=1e-15` over 5000 draws |
| 6. hourly rows enter only when `observed[t] & observed[t+1]` | PASS, exact set equality against the brute-force mask |
| 7. no unobserved current state is zero-filled into a teacher-forced transition | PASS — an unobserved hour is NaN and its transitions are dropped, never entered as `0.0` |
| 8. the exogenous window function takes no outage/target argument and never clips an unavailable window | PASS by signature check and by boundary cases `p=23,24,143,144` |

Supporting checks also passed: the independent `to_hourly` reproduces the
documented mean-of-observed-substeps semantics (agreement with
`src/asymode/evalproto.py` to `3e-8`, the difference being only that the repo
casts to `float32` while this audit keeps `float64`); the deterministic hash is
process-independent; the `K=2` band endpoints reproduce a flow ratio of exactly
`2` and `1/2`; and the closure gate rejects a boundary fit even when drift is
tiny.

**On the real tables** (Phase 6 independent validation, all ten checks passed):

- max `|U(1-mu) - R mu - mean_delta|` over all event, fold and local
  unconstrained fits: **`2.04e-17`**;
- max `|A*B - C^2 - v|`: **`6.49e-16`**;
- 5,190 local cells sit on a rate boundary; 4,773 of them have a box-fit
  identity residual above `1e-6`. **This is expected and is not a defect**: a
  constrained optimum obeys KKT inequalities, not the unconstrained normal
  equations. The unconstrained identity still holds exactly on every one of
  those rows.

So Proposition 1 (A1) and the `AB - C^2 = v` identity (A3 machinery) are
confirmed as exact, on synthetic problems and on all 8,071 fitted cells (51 event + 20 fold + 8,000 local).

---

## 3. Empirical assumptions — which hold and which fail

The closed-window corollary `U/R = mu/(1-mu)` is **not** automatic. It needs
measured closure and an interior fit. Both were measured.

`closure_pass` requires closure ratio `<= 0.05`, `0 < mu < 1`, rank two, and an
unconstrained fit strictly inside the fixed box.

### 3.1 Event level

| design | closure_pass | interior | on a boundary |
|---|---|---|---|
| full | **20 / 26** | 25 / 26 | 1 |
| active-48 | **5 / 25** | 21 / 25 | 4 |

Full-design failures and their reasons:

| event | family | why closure fails |
|---|---|---|
| 2018-01-16 | winter | unconstrained fit outside the box (`R_at_cap`) |
| 2019-02-20 | winter | closure ratio `0.856` |
| 2020-10-29 | tropical | closure ratio `0.0972` |
| 2021-02-15 | winter | closure ratio `0.0827` |
| 2024-01-12 | winter | closure ratio `0.0844` |
| 2024-09-27 | wind | closure ratio `0.472` |

### 3.2 Source-fold level

| design | weighting | closure_pass | median closure ratio |
|---|---|---|---|
| full | row-pooled | **1 / 5** | 0.161 |
| full | equal-event | **1 / 5** | 0.129 |
| active-48 | row-pooled | **0 / 5** | 0.693 |
| active-48 | equal-event | **0 / 5** | 0.646 |

**This is the single most consequential empirical finding of Section 3.** The
pooled seven-day source windows are *not* empirically closed under the correct
mask. Only fold 2 passes. The measured departure of the fitted rate ratio from
the closed-window prediction is:

| fold | `U/R` | `mu/(1-mu)` | ratio | closure_pass |
|---|---|---|---|---|
| 0 | 0.02100 | 0.01516 | 1.385 | no |
| 1 | 0.01883 | 0.01342 | 1.403 | no |
| 2 | 0.00892 | 0.00814 | **1.096** | **yes** |
| 3 | 0.02057 | 0.01447 | 1.422 | no |
| 4 | 0.01826 | 0.01415 | 1.290 | no |

Where closure passes the two agree to about 10%; where it fails they differ by
29–42%. The corollary is therefore a good approximation only where the audit
says it is, which is a minority of the pooled fits.

The identity itself holds exactly under **both** weightings (residuals `~1e-17`),
confirming that changing from row-pooled to equal-event weighting moves `mu`,
`mean_delta` and the fitted rates together, exactly as Proposition 1 predicts.
Kish effective size under equal-event weighting is roughly 85% of the row count.

### 3.3 Local `k=200` level

| design | closure_pass | interior | on a boundary |
|---|---|---|---|
| full | **1.75%** | 32.4% | 67.7% |
| active-48 | **1.98%** | 37.9% | 62.1% |

Local closure essentially never holds. The near-closure diagnostic formula was
therefore populated on only 1.75% / 1.98% of local cells, and — per protocol —
was left empty everywhere else and is never described as exact.

**Empirical verdict:** the exact algebra is exact; the closed-window corollary is
an approximation whose validity on this corpus is measured, partial, and worst
exactly where the storm signal is concentrated. The active-48 window is by
construction a *non-closed* interval: net outage accumulates inside it, so mean
drift is materially non-zero, which is why its closure rate collapses.

---

## 4. Descriptive design findings

All `Gamma` values below are **descriptive plug-in** quantities computed from an
exact constant box fit and its local residual scale. They are not neural
selectors, not unbiased estimates of a neural crossover, and not theorems.

### 4.1 Local `k=200` plug-in Gamma, full versus active-48

| design | p25 | p50 | p75 | p90 | p95 | p99 | frac > 1 | frac > 4 | frac exactly 0 |
|---|---|---|---|---|---|---|---|---|---|
| full | 0 | **0** | 0.190 | 0.938 | 1.614 | 3.671 | **9.48%** | 0.58% | 63.6% |
| active-48 | 0 | **0** | 0.440 | 1.543 | 2.294 | 5.616 | **15.18%** | 2.00% | 56.9% |

- **Active/full median plug-in Gamma ratio: undefined.** Both medians are exactly
  `0`, because in the majority of local cells the exact box fit places one of the
  two rates at zero, so `c = min(U,R) = 0` and the one-flow collapse removes
  nothing at all. Reporting a median ratio here would be spurious; the ratio of
  the fraction above one is the meaningful comparison.
- Ratio of the fraction above one, active-48 / full: **1.60**.
- Upper-quantile ratios: p75 `2.31`, p90 `1.64`, p95 `1.42`, p99 `1.53`.

The pattern is stable across all five folds (frac > 1 ranges 7.9–10.9% full,
11.9–17.0% active-48), so it is not driven by one fold.

### 4.2 Event level, full versus active-48

Across the 25 events where both designs exist:

- median `mu` ratio active-48/full: **2.27** — the exogenous window does roughly
  double occupancy, as H2 anticipated;
- median row-count ratio: **0.287** — it also removes about 71% of the rows;
- median plug-in Gamma: full `41.9`, active-48 `29.7`; **median ratio 0.756**.

So at the pooled event level the active-48 window **does not** raise the plug-in
design index: the occupancy gain is more than offset by the loss of rows and the
loss of closure. At the local `k=200` level, where `n` is held fixed at 200 by
construction, the same window **does** raise it (1.60× more cells above one).
The two statements are consistent because only the local comparison holds `n`
fixed; they answer different questions and both are reported.

Event-level plug-in Gamma on the full design is above one for **all 26 events**
(range `8.05`–`75.4`, median `40.7`). Event-level and local-level Gamma are not
comparable: the event cell pools 30k–133k rows, the local cell exactly 200.

### 4.3 Common rate and delivered transition treatment

| scope | median `c` | median `c/R` | median RMS `c(1-2Y)` | median residual `sigma` | median RMS/`sigma` |
|---|---|---|---|---|---|
| event, full | `2.91e-04` | **0.48%** | `2.89e-04` | `8.02e-03` | **2.9%** |
| event, active-48 | `4.79e-04` | **1.20%** | `4.74e-04` | `1.28e-02` | **4.8%** |
| local, full | `0` | 0% | `0` | `3.60e-03` | p90 = 7.7% |
| local, active-48 | `0` | 0.078% | `0` | `6.98e-03` | p90 = 9.6% |

On the pooled source fold used by the pilot (fold 2, full design, row-pooled):
`U = 1.886e-04`, `R = 2.114e-02`, `U/R = 8.92e-03`, `c/R = **0.89%**`.

The pilot report's fold-2 constants gave `c/R = 0.1%`. Measured here on the
**correct** adjacent-observation transition population and the full seven-day
panel, the same fold's pooled ratio is about an order of magnitude larger,
though still under one percent. The difference is attributable to the transition
population, not to the algebra: the pilot fitted only 24-hour windows following
near-fixed anchors and used the incorrect teacher-forced mask.

Either way the delivered structural treatment is small relative to the residual
scale: at the event level the RMS of `c(1-2Y)` is about 3% (full) to 5%
(active-48) of `sigma`.

### 4.4 Balanced-flow interval versus county reporting resolution

`K=2` band, expressed in units of one reporting customer
(band width divided by the median one-customer fraction):

| design | min | median | max | events with band narrower than one customer |
|---|---|---|---|---|
| full | **7.73** | **152** | 5091 | **0 / 26** |
| active-48 | 0 | 342 | 10017 | 2 / 25 |

**H3 is not supported.** On the full design the `K=2` balanced-flow band is never
narrower than one reporting customer; the narrowest is about 7.7 customer units
and the median is about 152. County reporting resolution is therefore *not* the
binding constraint on observing balanced flows in this corpus.

The two active-48 exceptions are a boundary-fit degeneracy, not a resolution
statement: on 2021-02-15 and 2024-09-27 the exact box fit inside the storm window
puts `R = 0` (`R_at_0`), so `rho = U/R` is infinite and the band collapses to the
single point `y = 1`. Inside a 48-hour storm-peak window with net accumulation,
a constant fit with no restoration is the constrained optimum.

The empirical share of states lying inside the band is nonetheless small: median
5.0% (full) and 7.2% (active-48). The states are mostly far from balance, but
that is an occupancy fact, not a resolution limit.

### 4.5 Family-level summaries

Local `k=200`, fraction above one:

| family | full | active-48 | frac `c=0`, full |
|---|---|---|---|
| convective | 10.9% | 16.4% | 61.1% |
| flood | 6.3% | 17.8% | 67.1% |
| tropical | 7.7% | 16.4% | 61.9% |
| wind | 8.7% | 16.1% | 66.5% |
| winter | 9.0% | 12.1% | 66.1% |

The active-48 improvement is present in every family. It is largest for flood
and tropical and smallest for winter. Event level, full design: median plug-in
Gamma by family runs `8.05` (flood, single event) to `63.7` (wind), and the
event-level closure pass rate is lowest for winter (3/7).

No family is selected, promoted or excluded on the basis of these numbers.

---

## 5. Claim-by-claim adjudication of the supplied conservation diagnosis

| claim | status after measurement |
|---|---|
| **A1** weighted constant-fit identity `U(1-mu) - R mu = mean_delta` | **Confirmed exact.** Max residual `2.04e-17` across all 8,071 fitted cells (51 event + 20 fold + 8,000 local) and both weighting schemes. |
| **A2** closed-window corollary `U/R = mu/(1-mu)` | **Confirmed conditional, and the condition frequently fails here.** Holds to ~10% where `closure_pass` is true (fold 2); departs by 29–42% on the four folds where it is false. Pooled full-panel closure passes on only 1 of 5 folds; active-48 on 0 of 5. |
| **A3** closed-window selection bound with the retained `(1-mu)^-2` factor | **Confirmed algebraically** (test 4) and evaluated on data only where `closure_pass` holds, i.e. 1.75%/1.98% of local cells. Reported as a near-closure *diagnostic formula*, never as an exact empirical bound. |
| **A4** `m_2(y) - m_1(y) = min(U,R)(1-2y)` | **Confirmed exact** to `1e-15`. Both `c` and the delivered RMS `c(1-2Y)` are recorded in every table, as required. |
| **H1** the broad seven-day panel makes the constant common component tiny | **Supported in direction, with a corrected magnitude.** On the correct transition population `c/R` is 0.48% median at event level and 0.89% on the pilot's own fold — small, but roughly an order of magnitude larger than the pilot's reported 0.1%. The delivered treatment is ~3% of residual `sigma`. |
| **H2** a storm-conditioned window materially increases the design index | **Partially supported, and the direction depends on what is held fixed.** `mu` roughly doubles (2.27×). At fixed local `n=200` the fraction of cells with plug-in Gamma above one rises 1.60× and the upper quantiles rise 1.4–2.3×. At the pooled event level, where the row loss (0.287×) and the closure loss apply, median plug-in Gamma *falls* to 0.756×. The heuristic `Gamma ~ 1/f` is not reproduced as a scaling law. |
| **H3** county-level resolution is too coarse for balanced flows | **Not supported.** The `K=2` band is never narrower than one reporting customer on the full design (min 7.7, median 152 customer units). Resolution is not the binding constraint. |
| **N1** "the 26-event run must also have `c/R = 0.1%`" | **Not established, and the measured value differs.** Measured `c/R` is 0.48% (event median, full) and 0.89% (fold 2 pooled). More fundamentally, global closure constrains mean flows only; it imposes no pointwise `U(x)/R(x)`. The measured local geometry is strongly heterogeneous — the plug-in Gamma distribution runs from exactly 0 to above 5 at p99 — which is what a mean-flow constraint permits. |
| **N2** "any neural model containing constants satisfies the identity exactly" | **Not established by this audit, and this audit cannot establish it.** Nothing here fits or inspects a neural model. The exact identity is a property of an unconstrained/interior constant least-squares fit. The 5,190 boundary cells measured here are a concrete demonstration that a *constrained* optimum need not satisfy it. |
| **N3** "a correct comparator cannot produce a several-percent MSE difference" | **Not established.** No performance comparison was run and none may be inferred from these tables. The reproduced legacy result retains its existing status, unchanged by this audit. |
| **N4** "the documented implementation defects are irrelevant" | **Not established; the defects remain open.** This preflight measured design geometry only. The dead interruption heads, inert hold, incorrect teacher-forced mask, degenerate origin rule, dead one-flow start, duplicate training fold and missing treatment-dose traces are untouched and still documented. Indeed the mask defect is quantitatively relevant: the pilot's `c/R = 0.1%` and this audit's `0.89%` differ by an order of magnitude on the same fold. |

---

## 6. Exact limitations of the constant and local benchmark

These limits are stated so no number above is over-read.

1. **The plug-in `Gamma` is descriptive.** It substitutes an exact constant box
   fit and its local residual scale into a fixed-design expression. It can be
   biased in either direction and it is not an unbiased estimate of any neural
   crossover point.
2. **`Gamma = 0` means the local *constant* fit is already one-flow**, because
   the exact box optimum put one rate at zero. It does not mean a
   context-dependent `U(x)` is zero on that neighbourhood, and it is not
   evidence that a trained two-flow model must collapse there.
3. **Global closure constrains mean flows, never pointwise rates.** The measured
   moment is `E[U(X)(1-Y)] - E[R(X)Y] = E[Delta]`. A rare weather state can carry
   a large conditional `U(x)` while the full-window average stays small. No
   statement in this report may be read as fixing `U(x)/R(x)`.
4. **A boundary fit is not a failed identity.** Two thirds of local cells lie on
   a rate bound. Their box-fit identity residual is non-zero by construction and
   correctly so; only the unconstrained fit is subject to the equality.
5. **The near-closure formula is a diagnostic, not a bound.** `closure_pass`
   admits a small but non-zero drift, so the zero-drift expression is an
   approximation even where it is populated — which is under 2% of local cells.
6. **Local neighbourhoods are PCA-Euclidean, not causal.** Cells are 200 nearest
   neighbours in a five-dimensional deterministic PCA of 14 standardized public
   channels. A different feature geometry would give different cells. `k`, the
   PCA dimension and the query count were fixed in advance and not swept.
7. **The active-48 window is one predeclared design, not an optimum.** It was
   defined only by NOAA county footprint with a public-weather tie-break. No
   outage value, target, residual or prior gain entered it, it was not searched,
   and the one event whose window falls outside the panel (2022-06-08, peak at
   hour 166) is reported unavailable rather than clipped.
8. **Event remains the inferential unit.** This preflight is descriptive. It
   computes no standard error over events and no test.
9. **Constant fits are not the model.** Everything here is two-variable constant
   least squares. It bounds what a *constant* comparator can see; it does not
   bound what a context-dependent estimator can learn.

---

## 7. Claims still requiring an estimator experiment

None of the following is answered by this audit, and no table above leans either
way on them:

1. whether a correctly implemented two-flow estimator beats its own collapsed
   one-flow comparator on held-out transitions or open-loop forecasts;
2. whether the conditional `U(X)` developed by training is materially larger than
   the constant `U` measured here on rare high-driver states;
3. whether the delivered treatment `c(1-2Y)` remains ~3–5% of residual scale
   after training, or grows;
4. whether the active-48 local improvement (1.60× more cells above one) survives
   as a real estimator gain once row loss and closure loss are paid;
5. whether the seven documented implementation defects, once repaired, change
   the pilot's near-null outcome;
6. the status of the reproduced legacy result, which this audit does not touch.

---

## 8. Output inventory

Written to `analysis/conservation_preflight_20260904/results/`:

| file | contents |
|---|---|
| `EVENT_CONSERVATION_METRICS.csv` | 52 rows: 26 events × 2 fixed designs |
| `FOLD_CONSERVATION_METRICS.csv` | 20 rows: 5 folds × 2 designs × 2 weightings |
| `LOCAL_GAMMA_METRICS.csv` | 8,000 rows: 5 folds × 2 designs × 800 `k=200` cells |
| `PREFLIGHT_SUMMARY.json` | provenance, parameters, checksum audit, compact summaries |
| `PREFLIGHT_REPORT.md` | automatically generated tables, nothing hand-selected |
| `RUN_PROVENANCE.json` | both HEADs, digests, parameters, package versions, seed, checksum results, commands, per-event windows |
| `EXECUTION_REPORT.md` | this audit |

Implementation under `analysis/conservation_preflight_20260904/implementation/`:
`preflight_lib.py`, `preflight_data.py`, `run_preflight.py`,
`test_preflight.py`, `validate_outputs.py`, `make_reports.py`.

Environment: Python 3.11.6 with `numpy==2.1.3`, `pandas==2.2.3`,
`scipy==1.14.1`, `pyarrow==18.1.0`, `pytest==8.3.4`, seed 0.
Total legal adjacent observed transitions: **1,533,413**.
No neural framework was imported at any point.

---

## 9. Gate summary

| gate | status |
|---|---|
| provenance and HEAD gates | PASS |
| audited-base diff empty | PASS |
| public SHA-256 manifest (60 files) | PASS |
| manifest / channel / fold digests reproduced | PASS |
| eight mandatory mathematical tests | PASS (13 tests) |
| Phase 6 independent validation (10 checks) | PASS |
| authorized write scope respected | PASS — only `implementation/` and `results/` |
| no absolute filesystem path in any output | PASS |

No gate failed. No integrity, provenance, algebra or data-path condition
required silent repair.

---

PREFLIGHT_COMPLETE_AWAITING_SHUAI_DECISION
