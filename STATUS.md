# Status snapshot — 2026-09-01, evening

Written at a deliberate pause. Nothing below is a result; results live in
`RESULTS_LEDGER.md` with paths and grades.

## What is running unattended

* **ERA5 fetch** — 25/26 windows landed, `2024-09-27` (Helene) in the CDS queue.
  Process started with `nohup`, survives session restarts. Log:
  `/private/tmp/claude-501/.../era5_strat.log` (this session's scratchpad).
* **exp08 → exp07 chain** on manifest `g2-convective-11` — started by a peer
  session that has since died; the shell chain is orphaned to init and alive.
  Writes `results/exp08_architecture.json` then `results/exp07_learned_baselines.json`.
  Log under the dead session's scratchpad `a9c2fe3c-...`. **Check both outputs
  exist and carry `digest: 76a73ed794af` before reading them.**

## Where the study stands

* Scope fixed to **convective-season events** as the main study
  (`docs/PREREGISTRATION_phase_separation.md`); other families are a
  generalisation test with winter as the natural negative control.
* Panels: 26 built, drivers rebuilt with wind components (12 channels + 2 clock).
  Manifest `g2-convective-11` covers the 11 convective panels only. **A `g3`
  manifest is needed once Helene lands and drivers are rebuilt for it**
  (`scripts/build_drivers.py`, then `scripts/build_panel_manifest.py --generation g3-...`).
* Pre-registered and **not yet run**: H-A1, H-A2, H-B, H-A3' (replacement),
  H-C (registered version), H-E (phase separation). All in `docs/PREREGISTRATION_*.md`.
* Pre-registered and **run**: H-D (dead as written), H-A3 (void), exp01 H2 (void,
  redesigned as exp09 — not yet run).

## Open decisions for the PI

1. Whether to assign a replacement for the dead peer session. A new one needs
   the firewall briefing from scratch (`FIREWALL.md` first, then the four docs
   named in `README.md`).
2. Headline metric. RMSE rewards predicting zero on this target; a
   decision-relevant alternative has published precedent
   (`docs/RELATED_WORK_PRECHECK.md`). Not decided.
3. The `inner_split` seed scheme (`seed*10 + fold + 1000`) is agreed but
   deferred until the g2 run finishes, to keep that run internally consistent.

## Next actions, in order, when work resumes

1. Confirm Helene landed; rebuild its drivers; build `g3` manifest (all 26).
2. Read exp08/exp07 g2 outputs; enter into ledger with grades.
3. Run exp06 (family stratification, tests H-E) on g3.
4. Run the registered asymmetry hypotheses on g2 (main) — H-A1/H-A2/H-B/H-A3'/H-C.
5. Run exp09 (identifiability, state-position sweep with the magnitude control).

Deadlines: abstract 2026-09-29 AoE, full paper with all supplementary 2026-10-06 AoE.
Hard checkpoint 2026-09-10.
