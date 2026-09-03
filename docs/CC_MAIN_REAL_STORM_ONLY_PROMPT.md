# CC task: run only the locked main real-storm experiment

Read first:

1. `paper/aistats/main_reviewed.tex`
2. `docs/CLAUDE_THEORY_REVIEW_RESPONSE.md`
3. `docs/MAIN_REAL_STORM_EXPERIMENT_LOCK.md`
4. `configs/panel_manifest_g2-convective-11.json`
5. `configs/source_validation_g2.json`

## Task

Implement or minimally adapt the existing event-held-out pipeline to execute exactly the experiment in `docs/MAIN_REAL_STORM_EXPERIMENT_LOCK.md`.

The core comparison is:

```text
two-flow NN: D_theta(x)(1-y) - R_theta(x)y
versus
one-flow NN: [s_theta(x)]_+(1-y) - [-s_theta(x)]_+ y
```

Both arms must use identical inputs, state scaling, masks, source folds, event-balanced 24-hour rollout objective, optimizer, batch budget, stopping rule, and seeds. Parameter counts must differ by less than 1%. Calibrate each arm by the same flow-matching principle in that arm's own parameterization; do not initialize the signed arm by subtracting susceptible-arm rates.

Run:

```text
2 neural arms x 11 held-out events x 3 seeds = 66 neural fits
HGB x 11 events x 2 horizons = 22 fits
damped persistence = negligible
```

Evaluate the neural arms on the held-out event using both:

```text
teacher-forced one-step MSE over valid transitions
recursive h+24 MSE
```

Also report h+6 and complete path-24 as secondary diagnostics. Average neural seeds within each event before inference. Event is the statistical unit.

For H1 and H2 report all eleven event differences, equal-event mean/median, exact two-sided sign-flip p-value, 50,000-resample event bootstrap interval, positive-event count, leave-one-event influence, and seed spread.

Decision rule:

```text
H1 passes: mean d_step > 0 and exact sign-flip p < 0.05
H2 passes: mean d_24   > 0 and exact sign-flip p < 0.05
joint flow-separation claim: H1 and H2 both pass
```

Bootstrap intervals and sign counts are diagnostics, not additional vetoes.

## Forbidden scope expansion

Do not run or add:

- mixed one-step/rollout objectives;
- objective-weight sweeps;
- memory, delayed-damage, gate, or semiparametric modules;
- family or phase-separation campaigns;
- event-shift mechanism tests;
- real-data `Gamma` plug-ins;
- width/depth searches;
- 48-hour campaigns.

The supplied balanced two-state solvable case is the only synthetic experiment and does not need redesign.

## Required output

Produce one versioned result directory containing:

- environment and data digests;
- split and fairness audit;
- raw event/seed rows;
- held-out predictions;
- main four-method table;
- H1/H2 event-level inference;
- optimization diagnostics;
- a short decision: `JOINT SUPPORT`, `TRANSITION ONLY`, `FORECAST ONLY`, or `NO JOINT SUPPORT`.

Stop after archiving this report. Do not launch a follow-up model automatically.
