# EXPERIMENT_PLAN — the whole system on a designed panel (draft, 2026-09-26)

Order of work (ARIS: design -> review -> implement -> multi-seed experiments -> claims with receipts). No step starts
before the previous one has its review receipt.

## Stage A — the panel (in review)
`DATASET_DESIGN.md` v0 -> reviews (`notes/LITERATURE_dataset_design.md`, `contrib/REVIEW_dataset_design.md`,
`contrib/REVIEW_dataset_physics.md`) -> v1, pre-registered selection rules and a sealed confirmation set -> build
(episodes, gates, ERA5 and HRRR, nodes for any new county, splits).

## Stage B — the system, not a mechanism
One model family evaluated as a whole on every regime:

* weather source: ERA5 and HRRR (on the ERA5 grid; 3 km only where the resolution audit says it matters) for the host
  inputs as well as for the hazard pathway;
* hazard dictionary covering every regime (wind ramps and durations, convective proxies, liquid and frozen
  precipitation phases, accretion loads, heavy-rain accumulations and antecedent wetness, heat), modulators
  (canopy, leaf state, soil drainage, exposure, developed land) and memories;
* chain reactions through load x trigger slots; geography through modulators and the exposure integral; county
  context in the host;
* entry into the population balance as a competing hazard, with and without the non-negativity clip (signed hazard);
  GCRK's response kernel with and without the bound on its opening (the PI's request), as alternative geography
  pathways on the same panel.

## Stage C — evaluation rules (from the first night's lessons)
* three seeds for every trained arm, decisions on seed-averaged predictions;
* event-grouped folds primary; county-grouped and leave-one-region-out secondary;
* the all-zero forecast and the host beside every arm; per-regime and pooled metrics, with frequency weights and
  equal-regime weights;
* a screen keep is provisional until all folds agree; nothing on the sealed set before a committed pre-registration.

## Stage D — claims
Every result goes through `aris/CLAIMS.md` with its evidence file and a reviewer receipt (an independent review
session or subagent), and the ledger is updated before anything is written up.
