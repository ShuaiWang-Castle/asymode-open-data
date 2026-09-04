# Codex / GPT Work task: GitHub-only conservation and design preflight

You are executing a fail-closed, zero-training audit for the AsymODE AISTATS project. Do not use any pre-existing local checkout, local dataset, cached checkpoint, uploaded archive, or remembered result. Every empirical input must be obtained from the public GitHub repository during this task.

## Scientific background

The observed state is the county fraction of customers without power, with proposed conditional-mean transition

\[
Y_{t+1}=Y_t+U_t(1-Y_t)-R_tY_t.
\]

The paper asks when a single signed flow is sufficient and when interruption and restoration should remain two nonnegative flows. A completed three-event pilot was near-null, but it also contained known implementation and data-path defects. The earlier event-held-out positive result remains reproduced legacy evidence under its own protocol. Neither result is to be relabelled here.

For an unconstrained or interior weighted constant least-squares fit,

\[
\widehat U(1-\widehat\mu)-\widehat R\widehat\mu=\overline{\Delta Y}.
\]

Under an empirically closed window this yields `U/R=mu/(1-mu)` and may make the full-window constant common component small in a low-occupancy corpus. This is exact for the stated fit. It is not automatically a pointwise identity for context-dependent rates, a KKT-boundary identity, or an exact theorem for a bounded early-stopped neural network. Your task is to measure which assumptions hold on the pinned public data and quantify the resulting design geometry before any repair or training.

## Authorized repository and immutable refs

```text
repository:  https://github.com/ShuaiWang-Castle/asymode-open-data
code branch: open-audit-20260904
audited base: d6555015cbe1c2b67f5197c725a8c8a785109b51
public data: main commit 8dd47c5ccd829611f27b69a3d64c274a0a24c400
manifest:    configs/panel_manifest_g3-all-26.json
digest:      db286b4960a4
fold digest: beb00a6762ba
```

## Phase 0 — create two clean clones

Create a new empty task directory. Do not run inside an existing repository.

```bash
set -euo pipefail
ROOT="$(mktemp -d /tmp/asymode-conservation-XXXXXX)"
CODE="$ROOT/code"
DATA="$ROOT/public-data"
REPO="https://github.com/ShuaiWang-Castle/asymode-open-data.git"
CODE_BRANCH="open-audit-20260904"
BASE_SHA="d6555015cbe1c2b67f5197c725a8c8a785109b51"
DATA_SHA="8dd47c5ccd829611f27b69a3d64c274a0a24c400"

git clone --branch "$CODE_BRANCH" --single-branch "$REPO" "$CODE"
git -C "$CODE" status --short
git -C "$CODE" rev-parse HEAD
git -C "$CODE" merge-base --is-ancestor "$BASE_SHA" HEAD

git clone --filter=blob:none --no-checkout "$REPO" "$DATA"
git -C "$DATA" sparse-checkout init --cone
git -C "$DATA" sparse-checkout set data configs
git -C "$DATA" fetch --depth 1 origin "$DATA_SHA"
git -C "$DATA" checkout --detach "$DATA_SHA"
```

Before continuing, report the two HEADs, code worktree status, and:

```bash
git -C "$CODE" diff --name-only "$BASE_SHA"..HEAD -- src experiments configs
```

That diff must be empty. Otherwise stop.

## Phase 1 — mandatory read order

Read completely, in this order:

```text
$CODE/CC_START_HERE.md
$CODE/FIREWALL.md
$CODE/analysis/conservation_preflight_20260904/00_READ_ME_FIRST.md
$CODE/analysis/conservation_preflight_20260904/01_CLAUDE_CLAIM_ADJUDICATION.md
$CODE/analysis/conservation_preflight_20260904/02_CONSERVATION_THEORY_NOTE.md
$CODE/analysis/conservation_preflight_20260904/03_GITHUB_ONLY_PROTOCOL.md
$CODE/analysis/conservation_preflight_20260904/05_OUTPUT_SCHEMA.md
$CODE/analysis/gpt_rescue_20260904/cc_v2/PILOT_REPORT.md
$CODE/analysis/post_pilot_root_cause_20260904/01_ROOT_CAUSE_ANALYSIS.md
$DATA/data/README.md
$DATA/data/SHA256SUMS.txt
$DATA/configs/panel_manifest_g3-all-26.json
```

Then restate in writing:

1. what is algebraically exact;
2. what requires closure and an interior fit;
3. why global mean-flow conservation does not impose a pointwise rate ratio;
4. why this audit precedes but does not erase the implementation repairs;
5. the hard stop.

## Phase 2 — public-data integrity

Run from the pinned data clone:

```bash
cd "$DATA"
sha256sum -c data/SHA256SUMS.txt
python - <<'PY'
import json
from pathlib import Path
x=json.loads(Path('configs/panel_manifest_g3-all-26.json').read_text())
assert x['digest']=='db286b4960a4'
assert len(x['panels'])==26 and len(set(x['panels']))==26
print('manifest PASS', x['digest'], len(x['panels']))
PY
```

Every checksum must pass. Record the output digest and verified-file count. Stop on any failure.

## Phase 3 — isolated environment and implementation

Use Python 3.11. Create a virtual environment inside the clean code clone and install only:

```text
numpy==2.1.3
pandas==2.2.3
scipy==1.14.1
pyarrow==18.1.0
pytest==8.3.4
```

Create only under:

```text
analysis/conservation_preflight_20260904/implementation/
analysis/conservation_preflight_20260904/results/
```

Implement a standalone audit script and tests. Do not edit or import neural-model/training code. Reimplement or independently verify the hourly aggregation semantics from `src/asymode/evalproto.py`; do not call the flawed pilot `pack()` path.

### Mandatory mathematical tests

The test suite must establish:

1. weighted unconstrained least squares satisfies `U*(1-mu)-R*mu=mean_delta` to `1e-11`;
2. `A*B-C^2=v` to `1e-12` under arbitrary positive normalized weights;
3. an explicit boundary-constrained counterexample does **not** satisfy the unconstrained equality;
4. under exact zero drift and `mu<=1/2`,
   \[
   G=\frac{vR^2\mu^2}{(1-\mu)^2(\mu^2+v)}
   \leq
   \frac{R^2\min\{\mu^2,v\}}{(1-\mu)^2};
   \]
5. the output-collapse treatment identity
   \[
   m_2(y)-m_1(y)=\min(U,R)(1-2y)
   \]
   holds numerically;
6. hourly transition rows are included only when `observed[t] & observed[t+1]`;
7. no unobserved current state is zero-filled into the teacher-forced transition table;
8. the exogenous active window never reads outage values and is never clipped into validity.

Run `pytest -q` before reading empirical results.

## Phase 4 — fixed empirical construction

### 4.1 Hourly transitions

For every manifest panel:

1. collapse 15-minute states to hourly by averaging observed substeps only;
2. mark an hour observed if at least one substep is observed;
3. keep transition `t -> t+1` only if both hourly states are observed and finite;
4. use `Delta=Y[t+1]-Y[t]`;
5. align public weather at `t+1`;
6. append only UTC clock sine/cosine for local geometry;
7. preserve event and county identifiers.

### 4.2 Two prespecified time designs

Retain both designs, regardless of their results.

**Full:** all legal adjacent observed transitions in the published panel.

**Active-48:** one 48-transition window centered on an outcome-blind exogenous peak:

1. compute at each hour the fraction of panel counties with an active NOAA event row;
2. select maximal footprint;
3. break ties by the mean positive standardized public-weather composite of `gust`, `wind_speed`, `precip`, `snowfall`, and `cape`;
4. break remaining ties by earliest hour;
5. use 24 transitions before and 24 transitions after the peak;
6. if unavailable, mark that event/window unavailable—do not clip it.

No outage state, outage aggregate, residual, prior gain, row filter, or model output may define the active window.

### 4.3 Exact constant fits

For every event/design and every frozen source-fold/design, compute:

- unconstrained weighted least squares;
- exact two-variable box-constrained least squares with `0<=U<=0.265`, `0<=R<=0.25` by checking the interior point, four edges, and corners;
- `mu,v,A,B,C`, mean drift, both identity residuals, boundary status;
- residual variance and standard deviation;
- mean interruption/restoration flows;
- `c=min(U,R)` and RMS delivered treatment `c(1-2Y)`;
- `K=2` balanced-flow interval and empirical state share inside it;
- median one-customer fraction computed across unique counties.

For each source fold compute both row-pooled and equal-event weights. Report Gamma only for the row-pooled calculation; for equal-event weights report Kish weight effective size and leave Gamma undefined because a weighted sandwich derivation is outside this audit.

Define the empirical closure ratio using the unconstrained fit:

\[
\frac{|\overline{\Delta}|}{|U|(1-\mu)+|R|\mu}.
\]

Mark `closure_pass=True` only when this ratio is at most `0.05`, both unconstrained rates are strictly inside the fixed box, and `0<mu<1`.

### 4.4 Local design geometry

Use the frozen fold map in

```text
$CODE/analysis/gpt_rescue_20260904/cc_v2/event_folds_v2.json
```

with digest `beb00a6762ba`. For each held-out fold and each time design:

1. use source-event transitions only;
2. deterministically cap each source event at 12,000 rows;
3. standardize the 12 public weather channels plus UTC clock on source rows;
4. fit five PCA directions by deterministic SVD;
5. choose 800 deterministic source queries;
6. form `k=200` nearest-neighbour cells;
7. calculate the same constant-fit statistics.

For uniform-weight event and local cells calculate the descriptive plug-in

\[
G=v\min\{R^2/A,U^2/B\},\qquad
\Gamma=nG/\widehat\sigma_\varepsilon^2,
\]

using the exact nonnegative box fit and its residual scale. Also report:

- the general rate-cap bound
  \[
  \Gamma\leq\frac{nv}{\widehat\sigma^2}
  \min\{0.25^2/A,0.265^2/B\};
  \]
- the zero-drift fitted-rate ceiling
  \[
  \frac{nR^2\min(\mu^2,v)}{(1-\mu)^2\widehat\sigma^2}
  \]
  only when `closure_pass=True` and `mu<=1/2`, with the symmetric formula for `mu>1/2`.

Because `closure_pass` allows a small rather than exactly zero drift, label the latter a **near-closure diagnostic formula**, not an exact empirical upper bound.

## Phase 5 — required outputs

Write exactly:

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

`RUN_PROVENANCE.json` must contain both Git HEADs, audited base, manifest/fold digests, complete parameters, package versions, seed, checksum results, and the exact commands.

`EXECUTION_REPORT.md` must include:

- integrity and provenance;
- exact algebra-test results;
- event and source-fold closure/interiority tables;
- full versus active-48 local Gamma quantiles and fraction above one;
- family-level summaries;
- active/full median-Gamma ratio;
- common-rate and delivered-treatment scales;
- balanced-flow band versus one-customer fraction;
- a claim-by-claim adjudication of the supplied conservation diagnosis;
- one and only one descriptive state:
  `LOW_INFORMATION_DESIGN`, `ROOM_REMAINS_AFTER_WINDOWING`, or `INDETERMINATE_BECAUSE_ASSUMPTIONS_FAIL`.

This label does not authorize training.

## Phase 6 — independent validation

Before committing, independently check:

1. maximum unconstrained identity residual is at most `1e-10` in event, fold, and local tables;
2. maximum `abs(A*B-C^2-v)` is at most `1e-10`;
3. zero-drift formula fields are populated only where `closure_pass=True`;
4. boundary fits are not treated as failures of the unconstrained identity;
5. every local row has `k=200` and contains no event from its held-out fold;
6. all 26 events remain in the event table;
7. active-48 windows are outcome-blind and never clipped;
8. full and active-48 designs are both retained regardless of sign;
9. no output says the legacy result was refuted or that a neural result is mathematically impossible;
10. no neural framework has been imported or trained.

## Phase 7 — commit and hard stop

Only files under

```text
analysis/conservation_preflight_20260904/implementation/
analysis/conservation_preflight_20260904/results/
```

may be new or modified. Do not commit the virtual environment, public data, caches, or any change to `src/`, `experiments/`, `configs/`, the manuscript, or `RESULTS_LEDGER.md`.

Commit message:

```text
analysis: run GitHub-only conservation preflight
```

Push to `open-audit-20260904`. If push credentials are unavailable, package only the two authorized directories, provide SHA-256, and do not substitute any non-GitHub empirical input.

Then stop. Do not repair the model, rerun the pilot, start the 45-job campaign, select a favorable window/event/horizon, or propose a second experiment in the same execution. Return the commit SHA and a compact factual summary.
