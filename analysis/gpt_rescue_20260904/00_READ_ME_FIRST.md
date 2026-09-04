# Independent rescue audit: read this first

**Branch audited:** `open-audit-20260904`  
**Audit basis:** committed source code, committed experiment outputs, `docs/MODEL_HEALTH_AUDIT.md`, the data card, and the model genealogy in `DMDA_DataChallege`. No uncommitted raw data were available to this audit.

## Decision

The current experiment does **not** establish that concurrent interruption and restoration flows improve forecasting. The mathematical class-comparison results remain valid, but the empirical contrast is contaminated by a comparator-initialization defect, a boundary-degenerate one-flow implementation, unstable checkpoint selection, an unrepresentative eleven-event cohort, and a zero-dominated forecasting task.

The project is not scientifically dead. Its defensible core is narrower:

> A two-flow conditional-mean class is a strict extension of a one-flow class. Whether that extension helps depends on state dispersion, coactivity, estimation cost, omitted context, and rollout error. The current implementation does not yet isolate those factors.

## Four load-bearing findings

1. **The confirmatory one-flow initialization is semantically wrong.** `cc_event_transfer.py` first calibrates susceptible-flow rates `(u0,r0)` and then initializes the signed `net_scaled` head with `u0-r0`. These quantities multiply different exposure pools, `(1-y)` and `y`, so subtracting the rates is not a net-flow calibration. The committed folds have `u0≈8.3e-4--9.7e-4` and `r0≈0.091--0.107`, which initializes the signed head near `-0.09` in every fold. At `y=0`, a negative signed rate contributes exactly zero gradient and zero state change. This is the most important implementation defect.

2. **The apparent two-flow gain is concentrated at the boundary, not in the active interior.** The branch audit reports positive gains for `y0=0` and `0<y0≤0.01`, but losses for `y0>0.01`. The one-flow model emits exact zero on roughly 41--43% of masked cells, whereas the two-flow model almost never does. This evidence supports an onset/directional-decoupling interpretation, not a claim that simultaneously active physical flows improve prediction.

3. **The training protocol is not stable enough to compare model classes.** A single chronological validation event selects many checkpoints at epoch 1--2. The later event-balanced implementation is even less informative because it performs only one optimizer step per training event per epoch. Neither protocol includes the untrained checkpoint as a candidate. The model-class effect is therefore mixed with optimization and checkpoint noise.

4. **`g2-convective-11` cannot support a general storm claim.** It is an explicitly convective-season convenience cohort. The data card records 436 large-footprint candidate days: 194 convective, 185 winter, 43 wind, 10 flood, and 4 tropical, and explicitly notes that convective-only sampling selects the fastest dynamics. The available `g3-all-26` cohort is a better development sample, although it is still purposive rather than a probability sample.

## What to do next

Do not launch another broad architecture sweep. The next work is one controlled rescue sequence:

1. replace the independent-head comparison by a **nested shared-backbone model**;
2. correct initialization with the same constrained least-squares objective used by the theory;
3. train by fixed optimizer updates with event-balanced minibatches, source-event validation, and epoch-0 checkpointing;
4. use all 26 available events in fixed event-stratified folds;
5. make held-out one-step transition MSE the theorem-aligned primary endpoint and 24-hour rollout MSE the forecasting endpoint;
6. report boundary and active-state strata from the same predictions;
7. add one temporal weather encoder only if the repaired model still fails to beat the no-weather affine transition baseline.

The detailed code audit, data audit, theoretical implications, model specification, and locked experiment are in the remaining files in this directory.
