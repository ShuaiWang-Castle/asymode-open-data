# Claude Code execution prompt V2

You are taking over a controlled AISTATS experiment on the repository:

```text
ShuaiWang-Castle/asymode-open-data
```

The only branch authorized for this task is:

```text
open-audit-20260904
```

Before editing anything, run:

```bash
git fetch origin
git checkout open-audit-20260904
git pull --ff-only origin open-audit-20260904
git status --short
git rev-parse HEAD
```

Your first response must report:

1. checked-out branch;
2. commit SHA;
3. whether the worktree is clean;
4. which required files you read;
5. a five-sentence summary of the locked scientific question.

## Required read order

Read completely, in this order:

```text
CC_START_HERE.md
FIREWALL.md
analysis/gpt_rescue_20260904/00_READ_ME_FIRST.md
analysis/gpt_rescue_20260904/08_EVIDENCE_STATUS_AND_COMPETITION_LESSONS.md
analysis/gpt_rescue_20260904/09_LOCKED_CC_PLAN_V2.md
docs/MODEL_HEALTH_AUDIT.md
docs/DATA_CARD.md
paper/aistats/main.tex
```

Use `RESULTS_LEDGER.md` only to verify provenance of an existing claim. Do not reconstruct a task from obsolete prompts.

## Governing interpretation

Do **not** begin by declaring the previous positive result false. The existing event-held-out result is reproduced legacy evidence. The initialization, optimization, boundary, and cohort audits identify possible confounds and alternative mechanisms; they do not by themselves negate the earlier conclusion. The new experiment adjudicates what survives after a cleaner comparison.

The final competition model supplies prior architectural evidence that must not be discarded for superficial symmetry. The two physical directions used very different representations: a rich interruption side with two MLP heads, a distinct occurrence gate, an independent background path, and a learned first-order hold; and a simple recovery GLM with a slower update schedule and a different input block. Your new comparison must preserve that process-specific inductive bias in both arms and change only whether simultaneous nonnegative components are retained.

## Scientific contrast

Both arms compute the same nonnegative proposals:

```text
U_tilde = competition-informed interruption proposal
R_tilde = recovery GLM proposal
s = U_tilde - R_tilde
```

They differ only at the output:

```text
asym_two_flow:
    U = U_tilde
    R = R_tilde

asym_one_flow:
    U = relu(s)
    R = relu(-s)
```

Both update the same state equation:

```text
Y_next = Y + U * (1 - Y) - R * Y
```

The one-flow arm therefore removes exactly `min(U_tilde, R_tilde)` while retaining the same feature maps and parameter budget. Do not substitute the earlier one-width-48 `net_scaled` network as the sole primary comparator.

## Task 1 — freeze the data design before fitting

Use all panels in:

```text
configs/panel_manifest_g3-all-26.json
```

Create:

```text
analysis/gpt_rescue_20260904/cc_v2/event_design_table.csv
analysis/gpt_rescue_20260904/cc_v2/event_design_table.md
analysis/gpt_rescue_20260904/cc_v2/event_folds_v2.json
analysis/gpt_rescue_20260904/cc_v2/origin_rule_audit.md
```

The event table and folds must follow Section 3 of `09_LOCKED_CC_PLAN_V2.md`. Do not use prior gains, residuals, outage severity, or target peaks to choose events or folds.

For each panel, create exactly the outcome-blind candidate anchors specified there:

```text
first NOAA event time - 6 h
midpoint of NOAA event interval
last NOAA event time + 6 h
```

Round to the nearest valid hourly origin with 24 h of past context and 24 h of future target. Remove duplicate anchors only. Do not replace a duplicate by searching the outage trajectory.

Write and hash the five-fold event assignment before importing or training the neural model.

## Task 2 — map the competition invariants onto the open data

Create:

```text
analysis/gpt_rescue_20260904/cc_v2/FEATURE_MAP.md
```

The competition dimensions `59/17/43` are not automatically valid on this dataset. Preserve the semantic partitions:

```text
x_u   = rich interruption magnitude context
x_occ = narrow, genuinely distinct occurrence context
x_r   = recovery-specific context
```

For every channel, record source, timestamp availability, whether it is exogenous or pre-origin observed history, normalization population, and missingness treatment.

Hard restrictions:

- `x_occ` may not be the hidden layer or complete input of the interruption magnitude network;
- the background path may not be another output of the main interruption trunk;
- `x_r` has no clock and no current simulated state;
- no county ID embedding;
- no post-origin outage observation;
- no target-derived event or origin feature;
- no GRU, Transformer, recovery burden, secondary damage state, or additional gate in this pass.

Use only causal accumulated-weather features and pre-origin history summaries. Any unavailable competition feature is omitted and documented, not fabricated.

## Task 3 — implement one clean paper module

Create:

```text
src/asymode_paper/__init__.py
src/asymode_paper/features.py
src/asymode_paper/initialization.py
src/asymode_paper/asymmetric_flows.py
src/asymode_paper/trainer.py
experiments/paper_v2_pilot.py
```

Do not add new branches to `src/asymode/dynamics.py` or to the competition model zoo.

### Interruption proposal

Implement exactly one fixed architecture:

```text
two independent two-layer width-32 MLP logits, averaged
learned first-order logit hold
separate linear occurrence gate on x_occ
separate linear low-capacity background path
```

The hold is:

```text
q_t = sigmoid(g_u(x_u_t))
held_logit_t = q_t * held_logit_{t-1} + (1-q_t) * raw_logit_t
```

The interruption proposal combines the held magnitude, occurrence gate, and background path as specified in `09_LOCKED_CC_PLAN_V2.md`.

### Recovery proposal

Implement one logistic GLM on `x_r`. Recompute it every eight forecast steps and hold it between recomputations. Do not add hidden recovery layers.

### State map

Choose fixed rate caps from fit-data transition quantiles before the pilot, write the rule to the audit, and use the same caps in every fold and arm. Require the configured upper bounds to satisfy the state-preservation condition. The primary forward pass must not depend on clipping to remain in `[0,1]`; an assertion clamp may exist only as a fail-closed diagnostic and must report every activation.

## Task 4 — implement exact class initialization

In `initialization.py`, implement and test:

```text
bounded constant two-flow least squares
bounded interruption-ray least squares
bounded restoration-ray least squares
```

The old `u0-r0` rule is forbidden.

For every one-flow fold/seed, train two fixed starts:

```text
interruption-ray start
restoration-ray start
```

Select the start on source-event validation only. Test data never choose the start. Include the calibrated update-0 state as a checkpoint candidate for both model classes.

Unit tests must compare the solvers against brute-force grids or an independent closed-form implementation over random inputs.

## Task 5 — use an adequate, identical optimization schedule

Implement fixed update budgets, not epoch counts:

```text
Stage A: 1,600 teacher-forced transition updates
Stage B: 3,200 24-hour rollout updates
validation every 200 updates
no Stage-B stopping before 1,600 updates
patience 6 checks after the minimum
```

Stage A uses unique hourly transitions; do not multiply a transition by the number of overlapping forecast origins. Sample source events uniformly. When possible, construct each Stage-A minibatch with half `Y_t<=0.01` and half `Y_t>0.01`; log every fallback.

Stage B samples events and the three event-centered origins uniformly. Both arms use the same optimizer, learning rate, normalization, batch size, gradient clipping, update count, and validation rule.

Record, per job:

```text
update-0 validation and test-independent diagnostics
selected update
all validation checkpoints
examples processed
module-wise gradient norms
rate and gate summaries
state-range assertions
training time
selected one-flow start
```

Do not tune width, learning rate, caps, gate form, background cap, hold form, recovery update frequency, objective weights, or history length after seeing pilot performance.

## Task 6 — run the pilot and stop

After Tasks 1–5 pass tests, select three pilot events by exogenous medoid distance only:

```text
one convective event
one winter event
one tropical-or-wind event
```

Use weather-footprint, observation coverage, and county count. Do not use outage severity, old gains, or model residuals.

Run one model seed:

```text
3 events x
[1 asym_two_flow job + 2 asym_one_flow initialization jobs]
= 9 optimization jobs
```

The two one-flow starts produce one validation-selected reported one-flow estimate per event. Run the exact constant baselines, damped persistence, and all-zero. HGB is not needed in the implementation pilot.

Repeat one complete pilot event with the same seed to verify numerical reproducibility.

Write:

```text
analysis/gpt_rescue_20260904/cc_v2/MODEL_IMPLEMENTATION_AUDIT.md
analysis/gpt_rescue_20260904/cc_v2/pilot_results.json
analysis/gpt_rescue_20260904/cc_v2/pilot_event_effects.csv
analysis/gpt_rescue_20260904/cc_v2/pilot_training_diagnostics.csv
analysis/gpt_rescue_20260904/cc_v2/PILOT_REPORT.md
analysis/gpt_rescue_20260904/cc_v2/REPRODUCTION_COMMANDS.md
```

Report teacher-forced one-step MSE and 24-hour path/h+24 MSE for:

```text
all rows
Y_t or Y_0 = 0
0 < Y_t or Y_0 <= 0.01
Y_t or Y_0 > 0.01
```

Also report each trained arm against its own update-0 model and against the constant two-flow baseline.

The pilot is an implementation check. Do not use three events to write a paper conclusion.

## Hard stop

After the pilot deliverables exist, stop. Do not run the five-fold three-seed main campaign. Do not edit `paper/aistats/`, result macros, the abstract, or the conclusion. Do not relabel prior evidence as confirmed, refuted, invalid, or withdrawn.

Your final pilot message must separate four categories:

1. reproduced legacy evidence;
2. newly verified implementation facts;
3. pilot-only observations;
4. scientific claims that remain open for the 26-event main run.

## Failure behavior

Fail closed and stop before training if:

- event metadata cannot support the locked origin rule;
- a feature is unavailable at forecast time;
- fold construction uses any target outcome;
- constant-class initialization does not pass independent numerical checks;
- state preservation requires frequent clipping;
- the two reported arms do not share the same proposal modules and inputs;
- update 0 is not a selectable checkpoint;
- the branch or worktree differs from the reported state.

Do not silently improvise a replacement protocol. Document the blocker in `PILOT_REPORT.md` and stop.
