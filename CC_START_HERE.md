# Claude Code: start here for `open-audit-20260904`

## Authorized branch

Use only:

```text
open-audit-20260904
```

Before doing any work, report the checked-out branch, commit SHA, and worktree status.

## Current status

The V2 three-event pilot has completed. Do **not** rerun it and do not launch the full campaign. An independent no-retraining audit found P0 implementation and data-path defects that prevent the pilot from adjudicating the one-flow/two-flow scientific question.

The earlier event-held-out positive result remains reproduced legacy evidence under its original protocol. The completed V2 pilot is recorded as non-adjudicating; it does not authorize a positive or negative manuscript conclusion.

## Required read order

1. `CC_START_HERE.md`
2. `FIREWALL.md`
3. `analysis/post_pilot_root_cause_20260904/00_READ_ME_FIRST.md`
4. `analysis/post_pilot_root_cause_20260904/ROOT_CAUSE_AUDIT_GENERATED.md`
5. `analysis/post_pilot_root_cause_20260904/01_ROOT_CAUSE_ANALYSIS.md`
6. `analysis/post_pilot_root_cause_20260904/02_MINIMAL_REPAIR_GATE.md`
7. `analysis/post_pilot_root_cause_20260904/TRANSITION_DATA_AUDIT_STATUS.md`
8. `analysis/post_pilot_root_cause_20260904/03_CC_REPAIR_ONLY_PROMPT.md`
9. `analysis/gpt_rescue_20260904/cc_v2/PILOT_REPORT.md`
10. `docs/MODEL_HEALTH_AUDIT.md`
11. `paper/aistats/main.tex`

Use `RESULTS_LEDGER.md` only to verify provenance. Do not change evidence labels during the repair task.

## Immediate task

Execute only:

```text
analysis/post_pilot_root_cause_20260904/03_CC_REPAIR_ONLY_PROMPT.md
```

This task runs the raw-panel transition audit, repairs deterministic implementation/data-path defects, writes tests and audits, and **stops before any new accuracy training**.

## Superseded execution files

The following files remain as historical records and must not be executed:

```text
analysis/gpt_rescue_20260904/04_LOCKED_RESCUE_EXPERIMENT.md
analysis/gpt_rescue_20260904/07_CC_NEXT_STEP.md
analysis/gpt_rescue_20260904/09_LOCKED_CC_PLAN_V2.md
analysis/gpt_rescue_20260904/10_CC_EXECUTION_PROMPT_V2.md
analysis/gpt_rescue_20260904/11_IMPLEMENTATION_AMENDMENT.md
```

## Hard restrictions

- Do not rerun the completed V2 pilot before the repair gates pass.
- Do not launch the five-fold, three-seed main campaign.
- Do not edit the manuscript abstract, result macros, conclusion, or legacy evidence labels.
- Do not add a new model family, memory state, gate, transformer, semi-parametric term, or hyperparameter sweep.
- Do not select events or origins using prior gains or residuals.
- Do not infer transition-audit numbers when the raw/interim panel bytes are absent.
- Treat event as the inferential unit; starts and seeds quantify optimization only.

Any blocker must be documented and the task stopped rather than silently replacing the protocol.
