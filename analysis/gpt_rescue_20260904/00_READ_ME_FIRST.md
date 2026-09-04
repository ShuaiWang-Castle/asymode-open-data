# Independent audit and adjudication plan: read this first

**Active branch:** `open-audit-20260904`  
**Current task:** preserve the earlier evidence, integrate the final competition-model lessons, and run one controlled implementation pilot before changing the paper.

## Scientific status

The mathematical one-flow/two-flow theory remains intact. The archived event-held-out result also remains a reproduced empirical result under its original protocol. The code, optimization, boundary, and data audits identify issues that a cleaner experiment must adjudicate; they do **not**, by themselves, prove that the prior positive conclusion is false.

Accordingly:

- do not retract or overwrite the earlier result;
- do not promote the undertrained unified-v2 output to paper evidence;
- do not state that initialization, clipping, occupancy, or cohort composition has already explained away the earlier gain;
- do not edit the manuscript conclusion before the adjudicating run is reviewed.

The correct current statement is:

> A positive structural result has been reproduced, several plausible validity threats have been identified, and a controlled experiment is required to determine which interpretation survives.

## Why the selected model path changed

The first rescue proposal used one small shared weather encoder and nested heads. That design was clean, but it discarded an important empirical lesson from the completed data challenge: the strongest model used a highly nonlinear and temporally structured interruption side together with a much simpler recovery side. The occurrence gate, background path, first-order hold, dual interruption heads, recovery GLM, and adequate optimization were not interchangeable decorations; several were load-bearing in the competition study.

The new plan therefore distinguishes:

1. **dynamical asymmetry:** `U(1-Y)-RY`;
2. **flow separation:** one signed flow versus two nonnegative flows;
3. **representation asymmetry:** different inputs, temporal treatment, and capacity for interruption and restoration.

The adjudicating experiment tests flow separation while holding the other two forms of asymmetry fixed.

## Canonical files for new work

Read in this order:

1. `08_EVIDENCE_STATUS_AND_COMPETITION_LESSONS.md` — evidence-preservation rule and the competition-derived architectural invariants;
2. `09_LOCKED_CC_PLAN_V2.md` — authoritative data, model, training, endpoint, and fit-count specification;
3. `10_CC_EXECUTION_PROMPT_V2.md` — executable Claude Code handoff.

These files supersede the action instructions in:

```text
04_LOCKED_RESCUE_EXPERIMENT.md
07_CC_NEXT_STEP.md
```

The older files remain for audit history only. `05_MODEL_V3_REFERENCE.py` is a tested mathematical prototype, not the selected paper architecture.

## Evidence and diagnostic files retained

- `01_CODE_AND_ALGORITHM_AUDIT.md` — source-level questions that the new comparison must control;
- `02_DATA_REPRESENTATIVENESS_AUDIT.md` — cohort, origin, mask, denominator, and weather-aggregation concerns;
- `03_THEORY_IMPLICATIONS.md` — scope limits and one-step/rollout distinction;
- `06_STATIC_RESULT_AUDIT.py` and `STATIC_RESULT_AUDIT.*` — reproducible calculations on one archived result;
- `MODEL_V3_SELF_TEST.txt` — self-test of the earlier nested prototype.

Treat conclusions in these diagnostic files as hypotheses or protocol requirements whenever the V2 files state that adjudication is still open.

## Immediate authorization

Claude Code is authorized to:

- build the 26-panel event-design table and fixed folds;
- implement the clean competition-informed asymmetric scaffold;
- implement exact class-specific initialization and update-0 checkpointing;
- run the nine-job, three-event implementation pilot;
- write all outputs under `analysis/gpt_rescue_20260904/cc_v2/`.

Claude Code is **not** authorized to run the full main campaign, alter paper result macros, select a favorable event subset, add new mechanisms, or reverse prior claims. It must stop after the pilot report.
