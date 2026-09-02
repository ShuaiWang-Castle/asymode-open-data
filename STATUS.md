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

* **Peer session identity.** The experiment lane is the same session
  as before; this session's own address changed on `--resume`. Verified by
  details only the collaborator could know. It **reports to the PI directly**
  and will not accept a change to that from a peer; the PI must say so to it.
  It also **does not commit on a peer's request**; this session's `git add -A`
  carries its changes.
* **H-A3 on g2: graded [C]** after the decisive retest (`results/exp08_ha3_retest.json`,
  same digests, ambient = exactly four channels). The first [B] was inflated ~2x
  by two orphaned wind components and stays retracted; the void I then wrote was
  an over-reach the other way and is withdrawn — clean ambient removal has
  signal on g2 (both sides same sign; restoration side 13/15 t=3.03 at h+48,
  interruption side 9/15, below the fold bar). g1's clean ablation was null, so
  the g1/g2 difference is **unexplained**; a decisive rerun (g1's 12 panels x
  14 channels, three arms) is registered with its interpretation fixed. H-A3'
  remains the second control. `check_families` hard gate in place.
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
* **H-E amended twice** (both before any family fit): comparator is
  `net_scaled` at hidden 48 (parameter-matched, 3,121 vs 3,138); a four-rung
  ladder `net → net_scaled → sym_in+sym_arch → two-rate` isolates scaling,
  concurrency, and asymmetry one at a time. Concurrency — both directions active
  in one step — is its own rung, corrected from an earlier draft that conflated
  it with input/capacity symmetry.
* **exp06 registration consolidated** with H-E: H2 void (same question as
  H-E with the drifting comparator); H1's family list corrected (winter is the
  fastest family, not slow); H3 now tests the dominance share of eps, not
  eps > mean.
* **Degeneracy criterion pre-registered** (H-E, third amendment): any arm with
  `frac_pred_zero > 0.9` under the full protocol is `degenerate` — reported as a
  mechanism fact, excluded from quantitative comparison. Motivated by the unscaled
  `net` arm collapsing into the absorbing state at y = 0 in a smoke run.
* **D-1 oracle shrinkage script** (`experiments/d1_oracle_shrinkage.py`)
  written before any prediction exists and **verified on fixtures whose answer
  is known by construction**: calibrated high-SNR predictions give
  λ* = 1.00, a* = 1.00, headroom 0.000; the same predictions doubled give
  a* = 0.500, headroom 1.000; a tampered `fold_of` is rejected (exit 1);
  baseline MSE reconciles with the archive. One real bug was fixed (the
  baseline was computed after clipping, so the "oracle" was measuring the
  clip). **Two commits landed before the self-test passed** — one masked by a
  shell exit-code mistake, one by misreading a low-SNR fixture's large headroom
  as a code fault. Rule: a self-test's exit code is captured directly, never
  through a pipe. **Two further instances the same evening:** a verification
  loop that passed its arguments wrongly (every script exited 2 from argparse
  before executing) was read as a pass, and a `str.replace` that silently
  failed to match left a JSON key unrenamed. Mechanical fixes adopted: every
  scripted `replace` asserts its anchor exists; one `ALL` flag aggregates every
  check and gates the commit; commands in verification are written out
  explicitly, never assembled with `set --` / `${@:n}`. **A fourth instance:** the
  `configs/` copy I added to `build_panel_manifest.py` on the first evening was
  inserted by `rfind("print(")`, which orphaned the script's summary print at
  module scope and referenced a variable name I had guessed; the script was
  broken from that commit until now (g2 was built before it). Repairing it took
  five attempts because each fix was guarded by a check that was itself wrong
  (a regex that missed a parenthesised expression; a "module-level" test that
  missed an import) and one zsh trap (`$VAR` does not word-split). Rule: never
  locate an insertion point by searching for a generic token; anchor on a unique
  full line and assert its count is 1.
  *Reading note for the real run:* the target has median ≈ 0 and ~46% exact
  zeros, so at long horizons the predictor sits in the low-SNR regime; λ* > 1
  with non-trivial headroom there is the registered "shrinkage is MSE-optimal"
  prediction, not a defect.
* **D-4 trajectory-coherence script** (`experiments/d4_trajectory_coherence.py`)
  written before any prediction exists; verified on a coherent fixture (S1
  0.0004) and a per-sample-shuffled one (S1 0.66). Its S2 was first defined as
  raw curvature and **committed before the fixture showed that was wrong**
  (a scrambled trajectory can be flatter than a coherent peaked one); redefined
  as residual roughness `(p − y)` and re-verified. Same lesson as D-1: read the
  fixture before committing.
* **OOF predictions**: tier-1 export requested (all arms, 4 horizons, ~49 MB) for
  D-1 and the D-2 level/ranking decomposition. After exp07.

* **ERA5 complete, 26/26**, 2.9 GB. Drivers for Helene not yet built; g3
  manifest not yet generated — both are the experiment lane's call, after its
  current runs finish.
* **D-2 per family** ([A]): the rank ceiling at h+48 is 0.59 on tropical, 0.36
  wind, 0.29 convective, 0.28 winter — tracks phase separation, third
  independent measurement to do so. Interpretive note added to H-E: a tropical
  win can be a ranking win; a convective/winter win is a level win.

* **Severity-matched control run** ([A]): inside matched peak-severity bands,
  the fall/rise ratio keeps its ordering tropical > {wind, convective} > winter
  (winter 1.0–1.25 at every severity) — phase separation is not a size effect.
  The rank ceiling keeps only tropical-vs-rest; the onset share does not order
  families at all. **"Three independent measurements" is withdrawn**; the count
  is one ordering measurement plus a tropical-vs-rest ceiling. H-E1 amended:
  wind and convective are unordered and not scored.
* **D-3 decomposition script** (`experiments/d3_level_rank_decomp.py`) written
  before any out-of-fold prediction exists and **exercised against a synthetic
  fixture**: fold audit passes and rejects a tampered `fold_of` (non-zero exit),
  level + within reconciles with archived MSE to 1e-3, identity holds to 1e-11.
  Per-cell weighting fixed in the docstring. Waits on the lane's `--save-oof`
  export in the agreed layout (`pred[seed, sample, horizon]` + `fold_of` +
  `origin_id`).

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

## exp07 landed — learned baselines beat the dynamics at h ≥ 6

Trees on identical information: −7.3% / −6.0% at h+24 / h+48, 0/15 folds for
the dynamics; dynamics win h+1 by 20.6%. Trees are cap-limited (edge
understated). **Convergence check done: not under-trained** (natural early stop at 63–102
epochs, 0/5 at cap, net change ≈ 0). **EXP07 graded [B] for the ordering.** Per-horizon two-rate control registered. The
framing decision (narrow the claim / change metric / accept the negative) is
the PI's; no alternative metric was ever registered.

## Experiment lane gone (second time tonight)

Its socket died after exp10; an identity probe found only other projects'
sessions. **All of its work is committed** (workspace clean). The OOF export it
was to write (`--save-oof`) had not been started; this session implemented it
in `exp05_real_dynamics.py` (`--save-oof`), smoke-verified it against the OOF
audit and the archive reconciliation, and runs the six-arm g2 export itself. The decisive H-A3 rerun, Helene drivers, g3 and exp06 remain
unowned until a session is assigned.

## Six-arm export landed; diagnostics run; exp06 launched

EXP05-g2 ladder graded (concurrency rung +2.0% at h+48 only, [B]); D-1 confirms
shrinkage-optimal (no headroom); D-3 shows long-horizon error is within-origin
for every arm (the "level win" wording is corrected); **D-4 shows our own
rollout is not coherent across the four scored horizons (23.6% excess sign
changes)** — the "not a trajectory" framing needs the trees measured, which
requires an EXP07 export (`--save-oof` exists only in exp05). **exp06 (H-E) is
running on g3.** Decisive H-A3 rerun and the exp07 export queue behind it.

## exp07 export verified and committed

`--save-oof` in exp07 passed the scratchpad smoke (k=2, two arms): D-3 audit +
reconciliation 8/8, D-4 runs. The trees' S1 on that smoke was 0.63 — a preview
under tiny budgets, **not a result**. The full exp07 rerun with export is queued
behind exp06 for CPU; D-4 on the trees is decided there. OOF archives
(`results/oof_*.npz`) are regenerable and are not tracked.

## exp06 first attempt lost (2026-09-02, ~00:35)

The two-hour family run finished every fit and died writing its JSON:
`run_arm`/`run_baseline` now return a per-sample `_test_pred` array for the OOF
export, exp05 and exp07 pop it before writing, **exp06 imported the same
functions and did not** — my change, my omission. Fixed (popped for both row
types), and exp06 now has an end-to-end smoke (one family, k=2, 2 epochs) that
must write its JSON before a full run is launched. Relaunched.

## H-E landed — [B]

Two-rate vs parameter-matched single rate by family (h+48): tropical −5.1%
(14/15), convective −2.0%, wind ≈ 0, **winter +2.8% (5/15)**. H-E1 passes,
H-E2 survives — the negative control behaves as one. Convergence probe (400/40, 0/10 at cap): winter +4.3% (0/5), tropical
−3.5% at h+48 but −1.7% at h+24 → **[B] at h+48, [C] at h+24.** **Queue:** exp07 on g2
with export (running) → D-4 on the trees → family convergence probe → decisive
H-A3 rerun.

## Pinning a run to an older panel set

The live manifest is now `g3-all-26`. To rerun anything on g2, pass
`--panels configs/panel_manifest_g2-convective-11.json` — `panels.resolve`
reads any JSON with a `panels` key. `--panels` takes one value; a space list is
rejected. exp07 with `--save-oof` is running pinned this way so D-4 can compare
the trees to the two-rate OOF on identical samples.

## Waiting on
* ~~Per-horizon two-rate fits~~ — done, [B]: closes h+6 fully, not h+24/48;
  the long-horizon gap is structural (EXP10).
* **Order now:** OOF export → D-1/D-3/D-4 (this session) → decisive H-A3 rerun
  → six-arm g2 → Helene drivers → g3 → exp06.
* Per-horizon two-rate fits — ~~runs next~~ (result-driven control for the
  model-count asymmetry; paper must say it was added post hoc). Then the
  decisive H-A3 rerun, then the six-arm run with the OOF export. D-4
  (trajectory coherence) registered against the OOF layout.

* Decisive H-A3 rerun — g1's 12 panels x 14 channels, three ambient arms;
  lane schedules after exp07; must record the source fingerprint.

* exp07 (learned baselines, g2) — experiment lane reports on completion.
* H-A3 corrected rerun (3 arms, g2) — decides whether the void holds on g2.
* ~~ERA5 Helene~~ — landed and verified (188 MB, zip clean, both streams readable). **26/26.**
* **Drivers 26/26 and manifest `g3-all-26` built** (digest `db286b4960a4`,
  14 channels, versioned copy in `configs/`). exp06 (H-E) and the decisive
  H-A3 rerun are now runnable; both wait for the six-arm export to release the
  CPU. **The six-arm run loaded g2 at startup and is unaffected; its result must
  show digest `76a73ed794af`, to be verified on landing.**
* NET arm implementation — experiment lane.

## Standing checks (executable, not remembered)

* `scripts/check_imports.py` — every experiment module must import; run after
  any change to `src/asymode/*` or `experiments/exp05_*`.
* `scripts/check_comparable.py` — two result files are comparable only if their
  panel and channel digests match.
* `scripts/paired_review.py` — the review routine: paired Δ%, fold counts and t
  per horizon against a reference arm, on shared (seed, fold) units only; refuses
  (exit 2) any pair `check_comparable` rejects. Use it; do not re-type the
  comparison. Verified against the hand-computed H-A3 retest table.
* `check_families()` inside `exp08_architecture.py` — the family map must
  exactly partition the driver block, or the run refuses to start.
* `scripts/build_panel_manifest.py` — pins the sample set and writes a versioned
  copy under `configs/`.

## Next actions, in order, when work resumes

1. Confirm Helene landed; rebuild its drivers; build `g3` manifest (all 26).
2. Read exp08/exp07 g2 outputs; enter into ledger with grades.
3. Run exp06 (family stratification, tests H-E) on g3.
4. Run the registered asymmetry hypotheses on g2 (main) — H-A1/H-A2/H-B/H-A3'/H-C.
5. Run exp09 (identifiability, state-position sweep with the magnitude control).

Deadlines: abstract 2026-09-29 AoE, full paper with all supplementary 2026-10-06 AoE.
Hard checkpoint 2026-09-10.
