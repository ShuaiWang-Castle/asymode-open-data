# Codex / GPT Work: one-shot coarse two-flow confirmation

You are executing the sole authorized formal real-data experiment for the current AsymODE AISTATS study. The purpose is not to optimize a leaderboard. It is to test, at the pooling scale predicted by the preflight, whether separating interruption and restoration improves unseen-storm transitions and 24-hour open-loop forecasts relative to the identical coarse partition constrained to one signed flow.

## Immutable repository inputs

```text
repository: https://github.com/ShuaiWang-Castle/asymode-open-data
code branch: open-audit-20260904
required ancestor containing preflight: 5f3ff3aa616fac7540eae45415089127816f3199
required ancestor containing formal lock: a0b46f471e6a4689f1e3ab28f90a7a8d5baf756c
public-data commit: 8dd47c5ccd829611f27b69a3d64c274a0a24c400
manifest: configs/panel_manifest_g3-all-26.json
manifest digest: db286b4960a4
```

Start in a new empty temporary directory. Do not use any existing checkout, local data, checkpoint, cached result, or remembered number.

```bash
set -euo pipefail
ROOT="$(mktemp -d /tmp/asymode-coarse-confirm-XXXXXX)"
CODE="$ROOT/code"
DATA="$ROOT/public-data"
REPO="https://github.com/ShuaiWang-Castle/asymode-open-data.git"
git clone --branch open-audit-20260904 --single-branch "$REPO" "$CODE"
git clone --filter=blob:none --no-checkout "$REPO" "$DATA"
git -C "$DATA" sparse-checkout init --cone
git -C "$DATA" sparse-checkout set data configs
git -C "$DATA" fetch --depth 1 origin 8dd47c5ccd829611f27b69a3d64c274a0a24c400
git -C "$DATA" checkout --detach 8dd47c5ccd829611f27b69a3d64c274a0a24c400
```

Before doing anything else, report both HEADs, branch, worktree status, and verify both required ancestors with `git merge-base --is-ancestor`. Stop if either fails.

## Required reading

Read completely in this order:

1. `FIREWALL.md`
2. `analysis/conservation_preflight_20260904/results/EXECUTION_REPORT.md`
3. `analysis/coarse_flow_formal_20260904/00_FORMAL_LOCK.md`
4. `analysis/coarse_flow_formal_20260904/01_LOCAL_DEVELOPMENT_EVIDENCE.md`
5. `analysis/coarse_flow_formal_20260904/02_FORMAL_RUN_LOCK_ADDENDUM.md`
6. `paper/aistats/main.tex` only to understand the claim; do not edit it in this task.

Then restate:

- why event-level and local `Gamma` are not contradictory;
- why `K=8` was selected by 2021 absolute path risk rather than maximum structural gain;
- why 2022/2024 can be inspected only once;
- why the online cross-county experiment is deferred;
- the hard stop.

## Install the implementation

The attached package `AsymODE_CoarseFlow_FormalReady_20260904.zip` contains the implementation. Verify its external SHA-256:

```text
d4a37dcd154e8c30909eda87e73c3bb96afc0e2d63566339756d5774c0b0fa4a
```

Verify its internal `SHA256SUMS.txt`, then copy:

```text
flow_data.py
coarse_flow_formal_runner.py
test_formal_runner.py
```

into:

```text
analysis/coarse_flow_formal_20260904/implementation/
```

and copy the three numbered markdown files plus `development_results/` into `analysis/coarse_flow_formal_20260904/`. Do not overwrite `00_FORMAL_LOCK.md`.

Create a Python 3.11 environment and install exactly:

```text
numpy==2.3.5
pandas==2.2.3
scipy==1.17.0
scikit-learn==1.8.0
pyarrow==18.1.0
pytest==9.0.2
```

Run:

```bash
cd "$CODE/analysis/coarse_flow_formal_20260904/implementation"
python -m pytest -q test_formal_runner.py
python -m py_compile flow_data.py coarse_flow_formal_runner.py
```

All tests must pass. Commit the documentation and implementation **before** viewing any 2022/2024 performance:

```text
analysis: install locked coarse two-flow confirmation
```

## Public-data integrity

From the pinned data clone, run the complete published SHA-256 checker and verify the 26-event manifest digest. Stop on any failure. No file from the code branch may substitute for the pinned data clone.

## Development reproduction gate

Run only the development mode first:

```bash
python coarse_flow_formal_runner.py \
  --stage development \
  --skip-baselines \
  --data-root "$DATA" \
  --out "$CODE/analysis/coarse_flow_formal_20260904/development_reproduction"
```

The following values must reproduce within absolute tolerance `1e-10`:

```text
two-flow equal-event path24 MSE: 0.003996877036459834
one-flow minus two-flow path24 difference: 4.737495563058564e-05
one-flow minus two-flow active48 one-step difference: 4.010410721951202e-07
K: 8
2021 target events: 6
```

If not, write `BLOCKED_DEVELOPMENT_REPRODUCTION_MISMATCH` and stop without reading confirmation outcomes.

## One-shot confirmation

Only after the reproduction gate passes, execute exactly once:

```bash
python coarse_flow_formal_runner.py \
  --stage confirmation \
  --data-root "$DATA" \
  --out "$CODE/analysis/coarse_flow_formal_20260904/confirmation_results"
```

The runner must:

- refit the frozen K=8 partition and all baselines on all 2018--2021 events;
- evaluate all 2022/2024 events, retaining an unavailable active-48 event as unavailable rather than moving its window;
- use the correct adjacent-observation mask;
- use equal source-event weights;
- report the two-flow and matched one-flow sieve, persistence, source-fitted damped persistence, recursive HGB, and direct h+24 HGB;
- write event-level results, source-cluster diagnostics, paired inference, provenance, and a programmatic report.

## Independent checks

Before committing results, independently verify:

1. exactly 12 confirmation event IDs are present in the path table;
2. only 2018--2021 events enter standardization, K-means, rates, damping, or HGB fitting;
3. `K=8`, the 24 features, caps, and HGB parameters match the lock;
4. two-flow source training loss is no larger than one-flow source training loss in every cluster;
5. all event-level paired differences are recomputed directly from `EVENT_METRICS.csv`;
6. exact sign-flip p-values are obtained by enumerating all sign assignments;
7. bootstrap resamples storm events, not rows;
8. no 2022/2024 outcome influenced a branch, cluster count, feature, origin, baseline, or hyperparameter;
9. no online cross-county confirmation was run;
10. no manuscript or evidence ledger was edited.

Write an `INDEPENDENT_VALIDATION.md` recording every check.

Commit only the formal experiment directory with message:

```text
results: run one-shot coarse two-flow confirmation
```

Push to `open-audit-20260904`, then stop.

## Interpretation discipline

- A positive active-48 one-step result tests the theory-aligned representation contrast.
- A positive path result tests whether that gain survives open-loop propagation.
- A path gain without a one-step gain is not attributed to the oracle-gap mechanism.
- A structural gain does not imply practical superiority over HGB or persistence; report those separately.
- A negative result is final for this frozen 2022/2024 confirmation. Do not change K, resolution, features, event set, or endpoints and rerun.

Return only the installation commit SHA, results commit SHA, confirmation output paths, and a compact factual summary of the locked gates.
