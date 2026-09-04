# Claude Code next step

## Scope

Do not optimize the existing `cc_event_transfer.py` harness and do not launch another broad experiment campaign. First reproduce and repair the comparison itself.

## Required read order

1. `analysis/gpt_rescue_20260904/00_READ_ME_FIRST.md`
2. `analysis/gpt_rescue_20260904/01_CODE_AND_ALGORITHM_AUDIT.md`
3. `analysis/gpt_rescue_20260904/02_DATA_REPRESENTATIVENESS_AUDIT.md`
4. `analysis/gpt_rescue_20260904/03_THEORY_IMPLICATIONS.md`
5. `analysis/gpt_rescue_20260904/04_LOCKED_RESCUE_EXPERIMENT.md`
6. `analysis/gpt_rescue_20260904/05_MODEL_V3_REFERENCE.py`
7. `analysis/gpt_rescue_20260904/STATIC_RESULT_AUDIT.md`, when present
8. `docs/MODEL_HEALTH_AUDIT.md`
9. `FIREWALL.md`

## Task 1: verify the audit before changing code

Report with exact source locations:

- where the confirmatory harness calibrates all arms as `SUSCEPTIBLE`;
- where `net_scaled` receives `u0-r0`;
- the range of the reconstructed signed initialization in the archived result;
- the exact-zero prediction share of each arm;
- how many checkpoints are selected at epoch 1 or 2;
- why the existing integrity audit did not catch semantic initialization fairness.

Do not proceed if any of these facts cannot be reproduced.

## Task 2: create a clean paper-only implementation

Create a new module rather than modifying the historical model zoo in place. Implement the specification in `05_MODEL_V3_REFERENCE.py`:

```text
SharedContextEncoder
NestedOneFlowModel       # c fixed to zero
NestedTwoFlowModel       # c learned
fit_constant_classes
```

Requirements:

- one shared weather/history/static encoder;
- current outage state enters only through the explicit `(1-y,-y)` multipliers;
- no state clamp in the primary path;
- class-specific exact constant initialization;
- update zero included as a checkpoint candidate;
- reproducible unit tests for state preservation and nestedness.

## Task 3: build the event table and fixed fold map

Using all 26 panels, create the outcome-blind event table specified in `02_DATA_REPRESENTATIVENESS_AUDIT.md`. Build one fixed five-fold event-stratified map from family, year, footprint, coverage, and weather summaries only. Save the map and its digest before any neural fit.

Do not choose events or folds using outage severity, previous errors, or model gains.

## Task 4: implement the two-stage training harness

Use optimizer updates, not epochs:

```text
2,000 transition-pretraining updates
3,000 24-hour rollout fine-tuning updates
validation every 250 updates
```

Sample source events uniformly and sample a minibatch within the chosen event. Validation is the equal-event mean over fixed county-held-out rows from all source events.

The two neural arms must share every data, encoder, optimizer, initialization-procedure, and checkpoint rule. The only scientific switch is whether `c` is learned or fixed to zero.

## Task 5: run only the six-fit implementation pilot

After choosing three outcome-blind family-diverse pilot events from the event table, run:

```text
3 events × 2 neural arms × 1 seed = 6 fits
```

Return:

- update-0 and best validation scores;
- selected update;
- one-step, path-24, and h+24 metrics;
- zero-prediction share;
- mean and quantiles of `s`, `c`, `U`, and `R`;
- exact rerun reproducibility;
- comparison with the no-weather affine transition.

Then stop. Do not run the 30-fit main campaign until Shuai reviews the pilot.

## Prohibited in this stage

- gate, noisy-OR, background-damage, seeded-epidemic, recovery-burden, second-damage, or semi-parametric paths;
- hyperparameter sweeps;
- event removal based on performance;
- using the eleven-event cohort as an untouched confirmation set;
- reporting the two learned heads as physical hazards;
- using local plug-in `Gamma` as a real-data selector;
- interpreting a gain confined to `y0<=0.01` as evidence of concurrent interior flows.
