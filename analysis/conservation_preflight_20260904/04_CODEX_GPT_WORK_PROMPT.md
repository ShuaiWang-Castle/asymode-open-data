# Codex / GPT Work task: GitHub-only conservation and design preflight

You are executing a fail-closed, **zero-neural-training** audit for the AsymODE AISTATS project. Do not use any pre-existing local checkout, local dataset, cached checkpoint, uploaded archive, or remembered result. Every empirical input must be downloaded from the public GitHub repository during this task.

## Scientific background

The observed state is the county fraction of customers without power, with proposed conditional-mean transition

\[
Y_{t+1}=Y_t+U_t(1-Y_t)-R_tY_t.
\]

The paper asks when one signed flow is sufficient and when interruption and restoration should remain two nonnegative flows. A completed three-event pilot was near-null, but it also contained known implementation and data-path defects. The earlier event-held-out positive result remains reproduced legacy evidence under its original protocol. **Neither result may be relabelled in this task.**

For an unconstrained or interior weighted constant least-squares fit,

\[
\widehat U(1-\widehat\mu)-\widehat R\widehat\mu=\overline{\Delta Y}.
\]

Under an empirically closed window this yields `U/R=mu/(1-mu)` and may make the full-window constant common component small in a low-occupancy corpus. This is exact for the stated fit. It is not automatically a pointwise identity for context-dependent rates, a boundary-KKT identity, or an exact theorem for a bounded early-stopped neural network. Your task is to measure which assumptions hold on the pinned public data and quantify the resulting design geometry before any repair or training.

## Immutable inputs

```text
repository:  https://github.com/ShuaiWang-Castle/asymode-open-data
code branch: open-audit-20260904
audited base: d6555015cbe1c2b67f5197c725a8c8a785109b51
public data: main commit 8dd47c5ccd829611f27b69a3d64c274a0a24c400
manifest:    configs/panel_manifest_g3-all-26.json
manifest digest: db286b4960a4
frozen fold digest: beb00a6762ba
```

## Phase 0 — two clean clones

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

Before continuing, report both HEADs, the code worktree status, and:

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

Before implementing, write a short intake note stating:

1. what is algebraically exact;
2. what additionally requires empirical closure and an interior fit;
3. why global mean-flow conservation does not impose a pointwise `U(x)/R(x)` ratio;
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

Every listed checksum must pass. Record the verified-file count and the SHA-256 of the complete checker output. Stop on any failure.

## Phase 3 — isolated implementation

Use Python 3.11. Create a virtual environment inside the clean code clone and install only:

```text
numpy==2.1.3
pandas==2.2.3
scipy==1.14.1
pyarrow==18.1.0
pytest==8.3.4
```

Create or modify files only under:

```text
analysis/conservation_preflight_20260904/implementation/
analysis/conservation_preflight_20260904/results/
```

Implement a standalone audit script and tests. Do not edit or import neural-model/training code. Reimplement or independently verify the hourly aggregation semantics from `src/asymode/evalproto.py`; do not call the pilot `pack()` path.

### Mandatory tests before empirical output is inspected

The test suite must establish:

1. weighted unconstrained least squares satisfies `U*(1-mu)-R*mu=mean_delta` to `1e-11`;
2. `A*B-C^2=v` to `1e-12` under arbitrary positive normalized weights;
3. an explicit boundary-constrained counterexample does not satisfy the unconstrained equality;
4. under exact zero drift and `mu<=1/2`,
   \[
   G=\frac{vR^2\mu^2}{(1-\mu)^2(\mu^2+v)}
   \leq
   \frac{R^2\min\{\mu^2,v\}}{(1-\mu)^2};
   \]
5. `m_two(y)-m_one(y)=min(U,R)*(1-2y)` numerically;
6. hourly transition rows enter only when `observed[t] & observed[t+1]`;
7. no unobserved current state is zero-filled into a teacher-forced transition;
8. the exogenous active-window function accepts no outage array or target argument and never clips an unavailable window into validity.

Run `pytest -q` and save the complete output before reading empirical results.

## Phase 4 — fixed empirical construction

### 4.1 Correct hourly transitions

For every manifest panel:

1. collapse 15-minute states to hourly by averaging observed substeps only;
2. mark an hour observed if at least one substep is observed;
3. keep `t -> t+1` only if both hourly states are observed and finite;
4. set `Delta=Y[t+1]-Y[t]`;
5. align the public weather vector at `t+1`;
6. append only UTC clock sine/cosine for local driver geometry;
7. preserve event, county, physical-hour, and source-time-design identifiers.

### 4.2 Two prespecified time designs

Retain both designs regardless of the resulting Gamma values.

**Full:** all legal adjacent observed transitions in the published seven-day panel.

**Active-48:** one 48-transition window centered on an outcome-blind exogenous peak.

For each event:

1. at every public panel hour, count unique panel counties whose NOAA event interval contains that hour;
2. divide by the number of panel counties to obtain the hourly NOAA footprint;
3. choose the maximal-footprint hour;
4. for tied hours only, standardize `gust`, `wind_speed`, `precip`, `snowfall`, and `cape` within that event over all county-hour cells, take positive parts, average over channels and counties, and choose the largest composite;
5. break any remaining tie by the earliest hour;
6. if the peak state index is `p`, take transitions with current-state indices `p-24,...,p+23`, requiring states through `p+24`;
7. if that complete window is unavailable, mark the event/design unavailable—never clip or replace the peak.

No outage state, outage aggregate, target, residual, prior gain, row filter, or model output may define or break ties in this window.

### 4.3 Event and source-fold constant fits

For every event/design and every frozen source-fold/design, compute:

- unconstrained weighted least squares;
- exact two-variable box-constrained least squares with `0<=U<=0.265`, `0<=R<=0.25`, checking the interior point, four edges, and four corners;
- `mu,v,A,B,C`, mean drift, `A*B-C^2-v`, both identity residuals, and boundary status;
- residual variance `SSE/max(n-rank,1)` and residual standard deviation;
- mean interruption and restoration flows;
- `c=min(U,R)` and RMS delivered transition treatment `c(1-2Y)`;
- the `K=2` balanced-flow interval and empirical share of states inside it;
- the median one-customer fraction across unique counties.

For each source fold compute two weighting schemes:

1. row pooled;
2. equal event, with each source event receiving equal total weight.

Report Gamma only for uniform event-level rows and row-pooled source-fold fits. For equal-event fits, report the Kish weight effective size and leave Gamma undefined because the correct weighted sandwich law is outside this audit.

Define empirical closure ratio from the unconstrained fit:

\[
\mathrm{closure\_ratio}=
\frac{|\overline{\Delta}|}
{|U|(1-\mu)+|R|\mu}.
\]

Set `closure_pass=True` only when the ratio is at most `0.05`, `0<mu<1`, the design is rank two, and both unconstrained rates are strictly inside the fixed box.

### 4.4 Local `k=200` design geometry

Use the frozen fold map:

```text
$CODE/analysis/gpt_rescue_20260904/cc_v2/event_folds_v2.json
```

and verify digest `beb00a6762ba` before use. For each held-out fold and each time design:

1. use source-event transitions only;
2. cap each source event at 12,000 rows using a deterministic hash of `(event, county, physical_hour)`;
3. standardize the 12 public weather channels plus UTC clock on source rows only;
4. fit five PCA directions by deterministic SVD;
5. choose 800 deterministic source query rows using the same hash rule;
6. form Euclidean `k=200` nearest-neighbour cells in PCA space;
7. store the query event and verify that no held-out-fold event entered either PCA fitting or neighbourhood construction;
8. compute the same uniform-weight constant-fit statistics.

For event and local uniform-weight cells, calculate the descriptive plug-in

\[
G=v\min\{R^2/A,U^2/B\},
\qquad
\Gamma_{\mathrm{plugin}}=nG/\widehat\sigma_\varepsilon^2,
\]

using the exact nonnegative box fit and its residual scale. Also report:

1. the general cap-based quantity
   \[
   \Gamma_{\mathrm{cap}}=
   \frac{nv}{\widehat\sigma^2}
   \min\{0.25^2/A,0.265^2/B\};
   \]
2. the zero-drift fitted-rate expression
   \[
   \Gamma_{\mathrm{near\mbox{-}closure}}=
   \frac{nR^2\min(\mu^2,v)}
   {(1-\mu)^2\widehat\sigma^2}
   \]
   only when `closure_pass=True` and `mu<=1/2`, with the symmetric expression when `mu>1/2`.

Because `closure_pass` permits a small rather than exactly zero drift, call the second object a **near-closure diagnostic formula**, not an exact empirical upper bound. Do not call any plug-in quantity an exact neural selector.

## Phase 5 — outputs

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

`RUN_PROVENANCE.json` must contain both Git HEADs, audited base, manifest/fold digests, all parameters, package versions, seed, checksum results, and exact commands.

`EXECUTION_REPORT.md` must report:

- integrity and provenance;
- all mathematical-test outcomes;
- event and source-fold closure/interiority tables;
- full versus active-48 local Gamma quantiles and fraction above one;
- family-level summaries;
- active/full median-Gamma ratio;
- common-rate and delivered-treatment scales;
- balanced-flow interval versus one-customer fraction;
- a claim-by-claim adjudication of the supplied conservation diagnosis;
- exact limitations of the constant/local benchmark for conditional neural rates.

It must end with exactly:

```text
PREFLIGHT_COMPLETE_AWAITING_SHUAI_DECISION
```

or, if a required gate failed:

```text
BLOCKED_<short_reason>
```

Do not automatically promote or reject a later estimator experiment. That decision is reserved for Shuai after inspecting the full tables.

## Phase 6 — independent validation

Before committing, check independently:

1. maximum unconstrained identity residual is at most `1e-10` in event, fold, and local tables;
2. maximum `abs(A*B-C^2-v)` is at most `1e-10`;
3. near-closure fields are populated only when `closure_pass=True`;
4. boundary fits are never treated as failures of the unconstrained identity;
5. every local row has `k=200` and uses no event from its held-out fold;
6. all 26 manifest events appear in the event table;
7. active-48 windows are outcome-blind, use the stated exact index range, and are never clipped;
8. full and active-48 designs are retained regardless of sign;
9. no output says the legacy result was refuted or that a neural outcome is mathematically impossible;
10. no neural framework was imported or trained.

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

Then stop. Do not repair the model, rerun the pilot, launch the full campaign, select a favorable event/window/horizon/metric, or propose a second experiment in the same execution. Return the commit SHA and a compact factual summary only.
