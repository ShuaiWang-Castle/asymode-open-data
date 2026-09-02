# Pre-registration: is the long-horizon loss an insufficiency of the experiment?

Written 2026-09-02, before any of the runs below. Motivation, stated so it cannot
be re-narrated: the two-rate model loses to gradient boosting on identical
information at h+6/24/48 (EXP07), the per-horizon refit closes h+6 and not
h+24/48 (EXP10), and the ledger currently reads that loss as *structural to the
model form*. That reading rests on one unexamined asymmetry: **the rate networks
see only the current hour's raw meteorology**, while the tree at horizon h sees
the whole driver path x_1..x_h and can synthesise accumulations, lags and
time-since-peak from it. The dynamics cannot — a memoryless map cannot express
restoration that depends on accumulated damage or on time since failure (Prop. 5
of the theory plan). The feature families built for exactly this (`hazard_path`,
`hazard_composites`, `history_level`, `history_clearance`, `freeze_cycle`,
`neighbour_drivers`, statics) and the `state_in_u/state_in_r` switches have never
entered a result. So the loss has never been tested against a version of the
model that *could* represent delayed effects. That is the insufficiency.

## Rules that apply to every experiment here

* **Identical information, both sides.** Any input added to the rate networks is
  added to `trees_matched` as a feature at the same time. A win obtained by
  giving the dynamics something the trees do not get is not a result.
* **The trees are re-fit with their cap lifted** (rounds 400 → 2,000, early
  stopping on the inner split). EXP07 recorded the trees as cap-limited, so any
  narrowing of the gap must be measured against fully trained trees or it is
  an artefact of the cap.
* Primary study = g2 (`76a73ed794af`), 5 county-held-out folds × 3 seeds,
  county-held-out inner split, horizons 1/6/24/48, paired within (seed, fold),
  sign gate ≥ 12/15; every result carries panel, channel and source digests.
* The reference gap is EXP07's, recomputed in the same run (the control arm and
  `trees_matched` on the current 14 channels are re-run alongside, so the
  comparison is within one run, not across archives).
* **Grades.** A result here can reach [B]. Nothing in it can un-kill a hypothesis
  that died as registered (H-D stays dead as stated; new capacity hypotheses may
  be registered separately).

## E1 — memory inputs for the rates (highest expected value)

Add to the driver block, for both rates and for the trees: cumulative gust,
precipitation and snowfall since the forecast origin; hours since peak gust;
exponentially discounted accumulations of gust and precipitation at two fixed
decay rates (6 h, 24 h); freeze–thaw count (identically zero on warm panels —
kept so the same block runs on g3). All are `hazard_path`/`hazard_composites`/
`freeze_cycle` outputs, causal within the window.

*Interpretation, fixed.* Let G_h be the two-rate model's paired loss to the
uncapped trees at h+24 and h+48 in the same run.
- If G_h shrinks by **more than half** at both long horizons (and the sign gate
  holds on the improvement of the dynamics over its own control, ≥ 12/15): the
  long-horizon loss was **input memory, not model form**. The ledger's
  "structural" sentence and the draft's §6.2/§8 are amended; Prop. 5's
  representational reading is narrowed to "without memory inputs".
- If G_h shrinks by less than half, or the trees gain as much as the dynamics:
  the structural reading stands and is now [B] against the strongest
  alternative. Reportable either way.
- Secondary, fixed: which side uses the memory — an ablation removing the memory
  block from U only / R only. If removal from R costs more than from U at
  h+48 (≥ 12/15), that is the first registered *input* asymmetry with a sign
  (restoration is the memory-bearing side), and it goes in the paper as such.

## E2 — state-dependent rates

`state_in_r = True` (restoration rate sees y: crew saturation or
prioritisation), then `state_in_u = True`, then both. Trees already see y0, so
no change on their side.
*Interpretation, fixed.* Improvement over the control ≥ 1% at h+48 with the sign
gate → the paper reports state dependence as part of the model; otherwise
dropped without comment beyond the ledger.

## E3 — age-structured restoration (theory-implied)

Split the interrupted pool into fresh and aged, y = y_f + y_a, with
y_f → y_a at a learned rate a(x) and restoration rates R_f(x), R_a(x); or,
cheaper, give R the inputs "hours since onset (y crossed 0.01)" and "hours
since peak y" computed causally from the observed history. Same inputs to the
trees. Parameter-match the control (hidden 48) as for `net_scaled`.
*Interpretation, fixed.* If the two-compartment form beats the parameter-matched
single-pool control at h+24/48 (≥ 12/15) → the age structure is real and is the
paper's model; and its family pattern is then tested on g3 with H-E's rule
(expected larger on tropical, null on winter — if it *wins on winter* the
mechanism story is wrong and the paper says so). Runs only if E1 leaves a gap
of ≥ 3% at h+48; otherwise it is a follow-up paper.

## E4 — capacity sweep (cheap)

hidden 32 → 64 → 128, one and two layers, control arm only, 3 seeds. The
capacity result (both-GLM loses 3.7–4.8%, 0/15) says capacity matters at the
low end; this asks whether it still does at the high end.
*Interpretation, fixed.* ≥ 1% at h+48 with the gate → the main runs move to the
larger width and E1–E3 use it; otherwise width 32 stands.

## E5 — the registered asymmetry hypotheses, finally run

H-A1 (history level → restoration), H-A2 (neighbours → restoration), H-B
(clearance → interruption), H-A3′ (precipitation + wind speed as the replacement
symmetric control), H-C (gate on statics + hazard composites): as written in
`docs/PREREGISTRATION_asymmetry.md`, with their kill conditions. These are the
experiments the pre-registration planned and the lane never ran. Trees get every
family as features.

## E6 — storm-level phase separation within convective (diagnostic, no training)

D-5 binned county-events; the family effect may ride on storm-system properties.
Bin the 11 convective *panels* by their median fall/rise ratio and recompute the
two-rate vs `net_scaled` advantage per panel from the archived OOF exports.
*Interpretation, fixed.* Spearman between panel ratio and panel advantage ≥ 0.6
on all three seeds → the mechanism resolves at storm grain and the D-5 null is
re-read as county-event noise; otherwise the family-level statement stands.

## E7 — EXP06 at full budget, 3 seeds (upgrade the [C] at h+24)

400 epochs / patience 40 on tropical and winter, all 3 seeds. Interpretation as
in the amended H-E registration. Turns the h+24 row [B] or leaves it [C].

## E8 — the lean architecture (H-H in `PREREGISTRATION_external_priors.md`)

One shared trunk with two rate heads (parameter count ≤ control), plus the
noisy-OR event gate on U; the small-magnitude pathway and the persistence gate
are added only once their mechanism is defined by the PI in public terms.
Interpretation as fixed in H-H. Runs before E1, because E1's memory inputs go
on whichever base architecture E8 selects.

## Order, cost, calendar

E4 (½ day) → E8 lean architecture (1 day) → E1 on the selected base with uncapped trees (1 day compute, 1 day analysis) → E2
(½ day) → E6 (hours) → E7 (1 day) → E5 (2 days) → E3 only if triggered
(2 days). Abstract 2026-09-29, paper 2026-10-06, theory freeze 2026-09-15.
E1's outcome is the one that can change the headline; the framing decision is
therefore deferred to E1's landing (target 2026-09-06) and the draft is
rewritten on whichever branch of E1's fixed interpretation obtains.

## What this cannot fix, stated now

RMSE on a target that is exactly zero in ~45% of cells still rewards predicting
nothing; the winter reversal is the mechanism's positive evidence, not a
negative to remove; H-D as registered stays dead; and the trees' bar rises when
their cap is lifted, so "narrowing the gap" is measured against a stronger
opponent than EXP07's.
