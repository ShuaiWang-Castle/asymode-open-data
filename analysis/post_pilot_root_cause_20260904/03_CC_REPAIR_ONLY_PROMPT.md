# Claude Code prompt: repair only, no accuracy campaign

Use only branch:

```text
open-audit-20260904
```

Before editing, report branch, commit SHA, worktree status, and the files read. Read completely:

```text
CC_START_HERE.md
FIREWALL.md
analysis/post_pilot_root_cause_20260904/00_READ_ME_FIRST.md
analysis/post_pilot_root_cause_20260904/ROOT_CAUSE_AUDIT_GENERATED.md
analysis/post_pilot_root_cause_20260904/01_ROOT_CAUSE_ANALYSIS.md
analysis/post_pilot_root_cause_20260904/04_TRAINING_OBJECTIVE_AND_CONTEXT_AUDIT.md
analysis/post_pilot_root_cause_20260904/02_MINIMAL_REPAIR_GATE.md
analysis/post_pilot_root_cause_20260904/TRANSITION_DATA_AUDIT_STATUS.md
analysis/gpt_rescue_20260904/cc_v2/PILOT_REPORT.md
src/asymode_paper/asymmetric_flows.py
src/asymode_paper/trainer.py
experiments/paper_v2_pilot.py
```

## Governing evidence rule

Do not relabel the legacy event-held-out result as false, invalid, refuted, or withdrawn. The completed V2 pilot is non-adjudicating because the intended estimator and structural treatment were not realized. This task repairs implementation and data-path controls only.

## Task A — run the transition/data audit before changing the code

In the mounted environment containing `data/interim/panel_*.npz`, run:

```bash
PYTHONPATH=src:experiments \
python analysis/post_pilot_root_cause_20260904/transition_data_audit.py
```

Commit its three generated outputs. Report the illegal teacher-forced-row share and the anchor-window versus full-transition composition. Stop if the panel bytes are not available; do not infer the missing numbers.

## Task B — repair the P0 implementation defects

1. The interruption MLP hidden layers must use independent live initialization. Set only the final output weight to zero and set the final bias to reproduce the calibrated update-0 flow. Do not zero every layer.
2. Add parameter-level gradient tests: after the first backward pass the final output weights of both heads must have nonzero and nonidentical gradients; after one optimizer update hidden weights must receive nonzero gradients; after ten updates the heads must differ numerically.
3. Replace independent-row Stage A by ordered teacher-forced sequences that carry the held logit and recovery phase. Add a direct recurrence identity test and verify a nonzero hold gradient on a time-varying synthetic logit sequence.
4. Correct the Stage-A mask to `observed(current) & observed(next)`. No zero-filled unobserved current state may enter teacher forcing.
5. Keep exact zero-ray fits as deterministic baselines, but create prespecified live one-flow branch starts as described in `02_MINIMAL_REPAIR_GATE.md`. Both branch starts must receive usable gradients.
6. Match neural initialization weighting to the equal-event training estimand.
7. Select the deterministic constant one-flow baseline by comparing ray SSEs, not coefficient magnitudes.
8. Compute static imputation from source-fit counties only and apply it unchanged to validation/test data.
9. Reconcile clock normalization with the feature documentation.
10. Save final `U_tilde`, `R_tilde`, `s`, `c`, collapsed `U/R`, occurrence gate, hold gate, raw/held interruption logits, and recovery phase.
11. Use a stage-specific checkpoint criterion: one-step validation for Stage A and rollout validation for Stage B. Do not select a teacher-forced checkpoint with the rollout metric while calling it the best transition model.
12. Resolve transition-context semantics. Either define each unique transition's context at its own physical time, or retain origin-transition pairs with explicit repetition weights. Do not deduplicate by physical transition while keeping the first arbitrary origin-specific history vector.

Do not change widths, caps, feature sets, optimizer, update budgets, or scientific endpoints.

## Task C — repair the pilot design without choosing favorable outcomes

1. Choose one test event from each of three **distinct** frozen outer folds.
2. Replace the min/mid/max NOAA-union anchors by one outcome-blind hourly storm-footprint rule. Freeze it and produce an anchor audit showing no boundary clipping.
3. Use all unique adjacent observed transitions in the storm-conditioned interval for Stage A. The rollout anchors are not the Stage-A transition population.
4. Preserve all 26 events in the frozen main cohort; no event is removed by its outage severity or prior model performance.
5. State and test the time reference of cumulative weather, the first-order hold, and the eight-step recovery schedule. A rolling origin may not silently redefine a mechanism inherited from a fixed-cutoff competition model.

## Task D — run tests and stop before the repaired accuracy pilot

Produce:

```text
analysis/post_pilot_root_cause_20260904/cc_repair/
    REPAIR_IMPLEMENTATION_AUDIT.md
    TRANSITION_DATA_AUDIT.md
    PARAMETER_GRADIENT_TESTS.json
    TEMPORAL_STATE_TESTS.json
    MASK_TESTS.json
    REPAIRED_ORIGIN_AUDIT.md
    THREE_DISTINCT_FOLD_MAP.json
    TREATMENT_DIAGNOSTIC_SCHEMA.md
    STAGE_OBJECTIVE_AUDIT.md
    REPRODUCTION_COMMANDS.md
```

The tests must establish:

- interruption hidden weights are reachable;
- the two interruption heads diverge;
- the hold affects trajectories and receives gradient;
- both one-flow starts are trainable;
- teacher-forced rows always have observed current and next states;
- all three proposed pilot tests correspond to distinct trained estimators;
- no origin is clipped into validity;
- Stage A and Stage B use their declared validation metrics;
- transition context and repeated-transition weighting are explicit;
- update 0 remains exactly reproducible;
- state preservation does not depend on a clamp;
- final treatment-dose diagnostics are implemented.

Then stop. Do not run the nine-job repaired pilot, the five-fold main experiment, HGB, or any new architecture. Report whether each gate passed and list unresolved blockers. Do not edit the paper or `RESULTS_LEDGER.md`.
