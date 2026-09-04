# Claude Code / Codex: start here for `open-audit-20260904`

## Authorized branch

Use only:

```text
open-audit-20260904
```

Before any work, report the checked-out branch, commit SHA, and worktree status.

## Current status

The GitHub-only conservation preflight is complete. The locked coarse `K=8` temporal-transfer confirmation was installed at `979adccf1e75dde1eeeb20b3747b2483c6c99248` and executed once at `9d432801397a97062ef9820894c1c1dbbed09fbb`.

Its predeclared one-flow versus two-flow structural gate failed. The execution and event-level inference were independently reproduced, but a post-confirmation semantic audit found that the published driver tensors encode missing county-to-ERA5 mappings as all-zero physical weather. The confirmation must therefore remain frozen as a result on the published bytes and must not be promoted to manuscript evidence until the driver data are rebuilt and versioned.

No additional model run is authorized in this branch.

## Immediate task

Read, audit, or implement a **data repair only** when Shuai explicitly authorizes it. The current read order is:

1. `CC_START_HERE.md`
2. `FIREWALL.md`
3. `analysis/coarse_flow_formal_20260904/00_FORMAL_LOCK.md`
4. `analysis/coarse_flow_formal_20260904/confirmation_results/REPORT.md`
5. `analysis/coarse_flow_formal_20260904/confirmation_results/INDEPENDENT_VALIDATION.md`
6. `analysis/post_coarse_confirmation_20260904/00_READ_ME_FIRST.md`
7. `analysis/post_coarse_confirmation_20260904/01_RESULT_AUDIT.md`
8. `analysis/post_coarse_confirmation_20260904/02_TARGET_ORACLE_DECOMPOSITION.md`
9. `analysis/post_coarse_confirmation_20260904/03_WEATHER_DRIVER_COVERAGE_DEFECT.md`
10. `analysis/post_coarse_confirmation_20260904/weather_coverage_by_event.csv`
11. `analysis/post_coarse_confirmation_20260904/target_oracle_decomposition.csv`
12. `data/README.md`, `scripts/build_county_weights.py`, `scripts/build_drivers.py`, and `src/asymode/weather.py` from the pinned public-data commit `8dd47c5ccd829611f27b69a3d64c274a0a24c400`.

## Evidence status

- The old event-held-out neural result remains reproduced legacy evidence under its own protocol; this branch does not silently rewrite it.
- The V2 neural pilot remains non-adjudicating because of known estimator and mask defects.
- The coarse `K=8` confirmation is a valid frozen execution, but the structural gate failed.
- The post-hoc target-oracle decomposition is explanatory only. It uses revealed confirmation outcomes and cannot select another model on those events.
- The weather-driver coverage defect is a data-semantic defect, not a model result: 1,265 of 2,625 panel counties have all twelve weather channels exactly zero at every hour whenever they appear.

## Hard restrictions

Until Shuai authorizes a new task:

- no neural training, coarse-model rerun, online cross-county confirmation, or architecture search;
- no change to `K`, features, event set, endpoint, or baseline on the revealed 2022/2024 confirmation events;
- no attempt to rescue the result by deleting `2024-09-27` or any other event;
- no use of target-oracle diagnostics as confirmatory evidence;
- no manuscript, abstract, conclusion, result macro, or `RESULTS_LEDGER.md` edit;
- no overwrite of the published driver files or their manifest;
- no encoding of missing weather as zero.

A valid data repair must create a new immutable driver release and digest. It must rebuild the county-to-ERA5 weight table after the full panel-county union is frozen, fail closed on every unmatched FIPS, and add semantic coverage tests. Any rerun on 2022/2024 after that repair is a repaired replication, not a fresh untouched confirmation. A fresh confirmation requires newly frozen events.