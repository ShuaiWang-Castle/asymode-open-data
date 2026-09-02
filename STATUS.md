# Status snapshot — 2026-09-01, late evening (refreshed)

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

## Since the earlier snapshot

* **Peer session identity.** The experiment lane is `dmda-d5` — the same session
  as before; this session's own address changed on `--resume`. Verified by
  details only the collaborator could know. It **reports to the PI directly**
  and will not accept a change to that from a peer; the PI must say so to it.
  It also **does not commit on a peer's request**; this session's `git add -A`
  carries its changes.
* **H-A3 on g2: RETRACTED.** The ablation removed six channels, not four — the
  wind components I added were in no family. g1 (clean removal) was null; the
  void stands; H-A3' is the replacement. A corrected g2 rerun of the three arms
  is in flight. Hard gate `check_families` now refuses a run whose family map
  does not partition the driver block. **Audit from the archive alone:** compare
  `config.channels` against `config.families` in any result JSON.
* **External priors registered** — `docs/PREREGISTRATION_external_priors.md`,
  seven items from the controlled channel, each carrying the sender's own
  [directional]/[generic] label. The paper's provenance statement must
  acknowledge the directional ones. Design consequences already applied to H-C.
* **D-2 run** (rank ceiling, zero training): ranking is intrinsically
  unpredictable at h+24 (75% of origins below 0.5) and h+48 (95%). Long-horizon
  wins are level-estimation wins; per-horizon reporting is mandatory.
* **H-E amended** before any family fit: comparator is now a **single net-rate
  arm** (`n = cap·tanh(f(x))`), not damped persistence, because the latter's
  difficulty drifts by family in the direction that fakes H-E2. NET arm requested
  from the experiment lane; not yet implemented.
* **OOF predictions**: tier-1 export requested (all arms, 4 horizons, ~49 MB) for
  D-1 and the D-2 level/ranking decomposition. After exp07.

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
* Pre-registered and **run**: H-D (dead on g1 and g2), H-A3 (void on g1; g2
  "confirmation" retracted, corrected rerun pending), exp01 H2 (void,
  redesigned as exp09 — not yet run). D-2 run, [A].
* **[B] as of now: two** — the baseline protocol and "capacity is doing work"
  (both-GLM 0/15 at h+24/h+48 on g2).

## Open decisions for the PI

1. Whether to assign a replacement for the dead peer session. A new one needs
   the firewall briefing from scratch (`FIREWALL.md` first, then the four docs
   named in `README.md`).
2. Headline metric. RMSE rewards predicting zero on this target; a
   decision-relevant alternative has published precedent
   (`docs/RELATED_WORK_PRECHECK.md`). Not decided.
3. The `inner_split` seed scheme (`seed*10 + fold + 1000`) is agreed but
   deferred until the g2 run finishes, to keep that run internally consistent.

## Waiting on

* exp07 (learned baselines, g2) — experiment lane reports on completion.
* H-A3 corrected rerun (3 arms, g2) — decides whether the void holds on g2.
* ERA5 `2024-09-27` (Helene) — last of 26; completion watcher armed.
* NET arm implementation — experiment lane.

## Next actions, in order, when work resumes

1. Confirm Helene landed; rebuild its drivers; build `g3` manifest (all 26).
2. Read exp08/exp07 g2 outputs; enter into ledger with grades.
3. Run exp06 (family stratification, tests H-E) on g3.
4. Run the registered asymmetry hypotheses on g2 (main) — H-A1/H-A2/H-B/H-A3'/H-C.
5. Run exp09 (identifiability, state-position sweep with the magnitude control).

Deadlines: abstract 2026-09-29 AoE, full paper with all supplementary 2026-10-06 AoE.
Hard checkpoint 2026-09-10.
