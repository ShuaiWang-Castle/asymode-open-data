# GitHub-only conservation preflight protocol

## 1. Purpose

Before another neural fit, quantify whether the public county-level data and the chosen time window can resolve the second conditional-mean flow. This task is an analysis of design geometry and residual scale. It is not a model search and not a confirmatory performance experiment.

## 2. Pinned public inputs

| role | repository ref |
|---|---|
| code and prior analyses | branch `open-audit-20260904`; audited implementation base `d6555015cbe1c2b67f5197c725a8c8a785109b51` |
| public data | `main` commit `8dd47c5ccd829611f27b69a3d64c274a0a24c400` |
| panel manifest | `configs/panel_manifest_g3-all-26.json`, digest `db286b4960a4` |
| public-file integrity | `data/SHA256SUMS.txt` from the pinned data commit |
| frozen event folds | `analysis/gpt_rescue_20260904/cc_v2/event_folds_v2.json`, digest `beb00a6762ba` |

The data commit contains 26 analysis-ready `panel_*.npz` files, 26 `drivers_*.npz` files, and the public NOAA county-event table. The analysis verifies all required files against the published SHA-256 manifest before reading them.

## 3. Clean execution environment

Use two separate fresh clones. Never overlay an untracked local data directory onto an old checkout.

1. `code/`: clone `open-audit-20260904`.
2. `public-data/`: sparse clone `data/` and `configs/` at the pinned `main` commit.

The code worktree must remain clean until generated results are copied into the authorized results directory. Local caches, prior checkpoints, server-specific paths, uploaded archives, and files from other branches are prohibited inputs.

## 4. Transition population

For each panel:

1. collapse the 15-minute state and mask to hourly using observed substeps only;
2. retain transition `t -> t+1` only when **both** hourly states are observed;
3. never zero-fill an unobserved teacher-forced current state;
4. align the transition with the public exogenous weather at `t+1`;
5. append only the deterministic UTC sine/cosine clock for the local driver geometry.

Every event is reported separately. Event remains the inferential unit in later work; the current preflight is descriptive.

## 5. Two fixed time designs

### Full panel

All legal adjacent observed transitions in the published seven-day panel.

### Exogenous active-48 window

One fixed 48-hour window centered at an outcome-blind storm-footprint peak:

1. at each hour, compute the fraction of panel counties with an active NOAA event record;
2. choose the hour with maximal county footprint;
3. break footprint ties by a fixed public-weather composite using positive standardized `gust`, `wind_speed`, `precip`, `snowfall`, and `cape` values;
4. break any remaining tie by the earliest hour;
5. use the 48 transition hours centered on that peak;
6. do not clip an unavailable window into the panel; mark it unavailable.

No outage value, model residual, previous gain, or row-level `Y` threshold enters the window definition. The full panel remains the reference; active-48 is one predeclared design comparison, not a searched optimum.

## 6. Event-level calculations

For every event and both time designs, report:

- number of correct adjacent observed transitions;
- `mu`, `v`, `A`, `B`, `C`, and the `AB-C^2=v` residual;
- mean drift and closure ratio;
- unconstrained and exact box-constrained constant fits;
- active boundaries and both conservation residuals;
- residual standard deviation;
- plug-in `G`, `Gamma`, and the noise threshold `sqrt(nG)`;
- near-closure zero-drift formula only where closure/interiority diagnostics pass, with no claim of exactness when mean drift is merely small;
- general cap-based bound;
- `K=2` balanced-flow interval and the share of states in it;
- median one-customer fraction for comparison with the interval width.

## 6.1 Source-fold pooled calculations

For each frozen held-out fold and each time design, compute the source-pool constant fit twice:

1. row-pooled weighting, matching the ordinary pooled-transition geometry and permitting the descriptive Gamma calculation;
2. equal-event weighting, to show how the conservation moment and fitted rates change under the project's inferential weighting.

The equal-event table reports the identity, closure, rates, residual scale, and Kish weight effective size, but deliberately leaves Gamma undefined because its exact weighted finite-sample law requires a sandwich calculation.

## 7. Local design calculation

The primary local neighborhood size is fixed at `k=200`. The existing five-fold event map is used only to prevent a query row from defining normalization and neighborhoods with events in its held-out fold.

For each fold and each time design:

1. use source-event transitions only;
2. cap each event at 12,000 deterministically sampled legal transitions to prevent large events from dominating the geometry;
3. standardize the 12 public weather channels plus UTC clock on source rows;
4. fit a five-dimensional PCA by deterministic SVD;
5. evaluate 800 deterministic query neighborhoods of size 200;
6. report the same constant-fit, closure, residual-scale, and Gamma diagnostics as above.

This is a local fixed-design plug-in diagnostic, not a validated neural selector. No performance result is conditioned on it in this task.

## 8. Required outputs

```text
analysis/conservation_preflight_20260904/results/
    EVENT_CONSERVATION_METRICS.csv
    FOLD_CONSERVATION_METRICS.csv
    LOCAL_GAMMA_METRICS.csv
    PREFLIGHT_SUMMARY.json
    PREFLIGHT_REPORT.md
    RUN_PROVENANCE.json
    EXECUTION_REPORT.md
```

`EXECUTION_REPORT.md` must distinguish:

1. exact algebra verified;
2. empirical assumptions that hold or fail;
3. descriptive design findings;
4. claims still requiring an estimator experiment.

## 9. Decision rule and hard stop

The script does not authorize the next experiment. After the outputs are committed, stop and request Shuai's decision.

In particular:

- a low plug-in Gamma is not called a theorem about neural networks;
- the near-closure zero-drift formula is not populated where closure or interiority fails and is never described as exact when mean drift is merely small;
- an active-48 increase is not used to select a more favorable event subset;
- no legacy result is relabelled;
- no neural model is repaired or trained;
- no manuscript result or conclusion is edited.
