# Post-pilot decision: read this first

**Branch:** `open-audit-20260904`  
**Scope:** diagnosis of the completed three-event V2 implementation pilot.  
**Evidence policy:** the pilot is not used to reverse the reproduced legacy result or to edit the manuscript conclusion.

## Decision

The negative or near-null V2 pilot is **not an interpretable estimate of the value of one flow versus two flows**. The reason is not merely low statistical power. The implementation never instantiated the intended competition-informed interruption learner:

1. every weight in both ReLU interruption MLPs was initialized to zero, which permanently disconnects the 32-channel interruption input from the output;
2. the teacher-forced stage resets the temporal state on every row, so it cannot train the first-order hold;
3. the nominal interruption start of the one-flow arm lies at an effectively dead zero/ReLU configuration;
4. two of the three test events belong to the same outer fold and therefore reuse the identical trained models;
5. the anchor rule clips to panel boundaries for most panels and the transition pretraining uses only the three anchor windows;
6. the teacher-forced mask checks the forecast-origin observation rather than the observation of the current intermediate state;
7. the final interruption, restoration, common-component, gate, and hold trajectories were not saved, so the treatment actually delivered by the flow collapse is unmeasured.

The branch contains direct source evidence, a no-retraining gradient probe, and committed pilot outputs supporting these findings. The generated report is `ROOT_CAUSE_AUDIT_GENERATED.md`.

## What the pilot does and does not say

The pilot verifies that the exact constant solvers, state-preserving update, deterministic replay, and output-level collapse execute. It also shows that the learned context map did not transfer uniformly: the fold-2 model improved on the 2024 convective test event but was substantially worse than its own constant initialization on the 2018 tropical test event.

It does **not** determine whether:

- the original two-process state equation is useful;
- a properly trained interruption/restoration model improves on a one-flow collapse;
- the earlier reproduced event-held-out gain survives a repaired comparison;
- any future gain is onset-direction decoupling, interior concurrency, or both.

Accordingly, the earlier event-held-out result remains **reproduced legacy evidence pending adjudication**. The pilot is recorded as **non-adjudicating because the intended estimator and treatment were not realized**.

## Root-cause priority

| priority | cause | consequence |
|---|---|---|
| P0 | all interruption MLP feature weights are permanently dead | the rich interruption representation and two-head ensemble never train |
| P0 | Stage A cannot train the hold; current/next observation mask is wrong | transition pretraining is not the intended theory-aligned estimator |
| P0 | one-flow zero/interruption start is effectively dead | the two-start safeguard reduces to the restoration start |
| P0 | three test events yield only two distinct training problems | the apparent three-event replication is overstated |
| P0 | anchor rule is boundary-degenerate and Stage A samples only anchor windows | interruption episodes can be omitted while quiet/recovery rows dominate |
| P1 | global constant optimum is used as a neural initialization | rare context-dependent interruption is initialized in a saturated near-zero regime |
| P1 | challenge temporal mechanisms reset at arbitrary rolling origins | cumulative hazard, hold, and eight-step recovery no longer have their original cutoff semantics |
| P1 | output-level collapse has an unidentifiable common-mode parameterization | the restricted arm is unnecessarily ill-conditioned even though its function class is valid |
| P1 | final flow/concurrency diagnostics were not stored | the effective treatment dose cannot be measured |

## Hard stop

Do not launch the five-fold, three-seed campaign from the current code. Do not add a new architecture family, memory state, gate, or hyperparameter sweep. Repair the P0 controls, pass the deterministic gradient and data-path gates in `02_MINIMAL_REPAIR_GATE.md`, then repeat a three-**distinct-fold** pilot.

## File order

1. `00_READ_ME_FIRST.md`
2. `ROOT_CAUSE_AUDIT_GENERATED.md`
3. `01_ROOT_CAUSE_ANALYSIS.md`
4. `02_MINIMAL_REPAIR_GATE.md`
5. `root_cause_audit.py`
6. `transition_data_audit.py`

The transition audit requires the mounted `panel_*.npz` files used by the pilot. GitHub Actions did not contain those bytes, so the source-level mask error is established but its contaminated-row share remains to be quantified in the original execution environment.
