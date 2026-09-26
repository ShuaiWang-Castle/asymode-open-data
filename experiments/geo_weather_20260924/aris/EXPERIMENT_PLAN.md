# EXPERIMENT_PLAN — the whole system on a designed panel (2026-09-26)

Order of work (ARIS: design -> review -> implement -> multi-seed experiments -> claims with receipts). No step starts
before the previous one has its review receipt.

## Stage A — the panel (design registered)
`DATASET_DESIGN.md` v0 -> three reviews (`notes/LITERATURE_dataset_design.md`, `contrib/REVIEW_dataset_design.md`,
`contrib/REVIEW_dataset_physics.md`) -> **v1, registered**: parent weather systems of five regimes, a sealed
confirmation tranche, pre-window observation gates, near-miss controls, regime-balanced estimand. Build order in its
section 12: frame -> operator exclusions -> draws and coverage audit -> development weather, gates, county sample,
features.

## Stage 0 — the host on the designed panel (registered in DATASET_DESIGN section 9.3)
Host W+Cin, 3 seeds, 5 event folds, plus the A/A run (seeds 0-2 against 3-5). Output: the headline regimes (host beats
zero), the seed floor, and the per-regime error budget. If fewer than three regimes qualify, the host (I11) comes first.

## Stage B — the system, not a mechanism
One model family, evaluated as a whole on every regime with the regime-balanced skill:

* weather source: ERA5 and HRRR (on the ERA5 grid; 3 km only where the resolution audit says it matters), for the host
  inputs as well as for the hazard pathway;
* hazard dictionary covering every regime: gust ramps and exceedance of the local 98th percentile, strong-gust
  duration, convective organisation (shear where HRRR carries it), liquid and frozen precipitation phases (ERA5 ptype,
  HRRR categorical types), temperature-gated accretion loads, heavy-rain accumulations and antecedent wetness; modulators
  (canopy, leaf state from lai_hv, soil drainage x wetness, exposure, developed land) and memories;
* chain reactions through load x trigger slots, formed at the node before the exposure integral;
* geography through modulators and the exposure integral; county context in the host;
* entry into the population balance as a competing hazard, with and without the non-negativity projection (signed
  hazard); GCRK's response kernel with its opening bounded (tanh) and unbounded (the PI's request, I10), as an
  alternative geography pathway on the same panel.

First comparisons, each against the host and its twin, three seeds, event folds: (1) GCRK bounded vs open; (2) the full
EIH pathway, projected vs signed; (3) the same with load x trigger slots; (4) HRRR against ERA5 as the source of the
pathway and of the host inputs.

## Stage C — evaluation rules
* program.md rule 3b: single-seed screens discard only; keeps need 3 seeds on all five folds beside the twin; 5 seeds
  for a confirmation;
* event folds (family-grouped) primary; county folds discard-only; leave-one-region-out and forward-time descriptive;
* the all-zero forecast and the host beside every arm; regime-balanced skill, every regime, the worst regime and the
  frame-weighted value;
* nothing on the sealed tranche before a committed confirmatory registration (at most four tests, fixed sequence).

## Stage D — claims
Every result goes through `aris/CLAIMS.md` with its evidence file and a reviewer receipt (an independent review
session or subagent), and the ledger is updated before anything is written up.
