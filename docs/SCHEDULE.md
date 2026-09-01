# Back-planned schedule

Anchor dates, from the conference's own site (it still labels them "to be posted
as they are confirmed", so re-check weekly until a call for papers appears):

* **abstract — 2026-09-29 AoE**   (4 weeks from 2026-09-01)
* **full paper, including all supplementary material — 2026-10-06 AoE**

The supplementary material has **no grace week**. Everything ships on Oct 6.

## Week 1 — Sep 1 to Sep 7 — get unblocked, prove the mechanism

* [x] project skeleton, firewall, evidence grading
* [x] re-implementation of the two-rate dynamics from scratch
* [x] synthetic identifiability sweep, 6 forcing levels x 3 seeds
* [x] onset comparison, 3 arms x 2 generators x 3 seeds
* [x] storm-event catalog from public bulk CSVs; event-selection rule fixed
* [ ] **PI: Globus account** — blocks the outage panel entirely
* [ ] **PI: Copernicus CDS account** — blocks weather drivers
* [ ] zone-to-county mapping, so tropical cyclones stop being invisible
* [ ] panel v0 for one convective event day

## Week 2 — Sep 8 to Sep 14 — baselines and the evaluation harness

* county-held-out 5-fold split, fixed and versioned before any model sees it
* statistical baselines: persistence, climatology, ARIMAX, Poisson GLM
* tree baselines: random forest, gradient boosting
* deep sequence baselines: LSTM, DLinear, PatchTST
* the proposed model on the same folds, same seeds, same inputs
* horizons t+1 / t+6 / t+24 / t+48

**Hard checkpoint Sep 10.** Continue only if the panel runs end-to-end *and* at
least two baselines have produced 5 folds x 3 seeds. If not, say so that day and
re-target a later venue. Sep 15 is the last possible abort; deciding then leaves
two weeks, which is not enough to write.

## Week 3 — Sep 15 to Sep 21 — the ablations that carry the claim

* dynamical axis: susceptible vs epidemic vs seeded epidemic, on real data
* input axis: same channels to both rates
* capacity axis: same width for both rates
* leave-one-covariate-out attribution
* onset study on real counties: which counties start a storm at zero, and what
  each arm does with them
* **sign gate** — any asymmetry claim must hold in the same direction on all
  three seeds, or it stays graded [C] and never enters the paper

## Week 4 — Sep 22 to Sep 28 — freeze and write the abstract

* every figure regenerated from archived JSON, nothing hand-edited
* `RESULTS_LEDGER.md` frozen; anything still [C] is cut, not softened
* abstract drafted against results that already exist

**Sep 29: abstract submitted.** Title and abstract must be locked while some
experiments are still running, which is normal for this venue but means the
claim set has to be decided in week 3, not week 5.

## Week 5 — Sep 29 to Oct 6 — assemble

* paper assembled by the writing session from the ledger
* appendix: full protocol, all seeds, the identifiability derivation
* code and configuration cleaned for release
* **Oct 6: paper and all supplementary material.**

## Standing risks

1. **Globus is the single point of failure.** No account, no outage data, no
   paper. Everything else has a workaround; this does not.
2. **8 pages is tight** for ten baselines plus three ablation axes plus a
   synthetic study. Plan for the main text to carry the synthetic identifiability
   result and the onset result, with the baseline table compressed and the full
   grid in the appendix.
3. **The abstract locks the claims a week early.** Decide in week 3 which claims
   the paper is making, and do not let week 5 results change the title.
