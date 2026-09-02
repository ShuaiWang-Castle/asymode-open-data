# Pre-registration: hypotheses received through the controlled channel

Written on receipt, before any of them has been tested here. Every item below is
a **hypothesis or diagnostic to be run on public data**; none is a finding, and
none may be cited as evidence. The registration exists so that these cannot later
be presented as having been arrived at after seeing this repository's results.

## Provenance, stated plainly

These items arrived through a one-way channel from a session working on a
related, non-public dataset, opened by the PI with an explicit boundary: model
updates and qualitative reasoning about data characteristics may cross; data
details — values, distributions, county sets, variable dictionaries, protocol
constants, any number measured on that dataset — may not. The sender labelled
each item, and the labels are preserved here because they matter for how the
paper describes method provenance:

* **[generic]** — a statistical or methodological fact any careful practitioner
  reaches independently. Negligible provenance risk.
* **[directional]** — even with every number removed, the item specifies a
  direction of exploration. **The paper's account of how the architecture was
  arrived at must acknowledge that some directions were suggested rather than
  discovered.** Evidence for or against them is produced here and only here.

No item below carries a number from the source. Where a number appears, it was
measured in this repository.

---

## H-C refinements — event gate on the interruption rate. [directional]

H-C as registered in `PREREGISTRATION_asymmetry.md` stands. Two refinements are
added **before H-C is run**, both of which change the design:

**H-C.1 — an all-static gate is not a gate.** If the gate reads only county-level
time-invariant inputs, it is constant within a forecast window and degenerates to
a per-county fixed multiplier. That is a different function class from an event
switch, and results from it must not be placed on the same width–effect curve.
*Design consequence:* every gate arm in H-C must include at least one
time-varying input. The registered "identity + hazard composites" gate satisfies
this; a "statics-only" arm is added as a **labelled contrast**, not as a point on
the width sweep.

**H-C.2 — width, not overlap.** The claim under test is that a narrow gate avoids
a bad basin regardless of whether its inputs are a subset of, or disjoint from,
the rate's inputs. *Design consequence:* at one fixed narrow width, run a
strict-subset gate and a disjoint gate. **Kill condition:** if the subset gate
falls into the basin (measured as `frac_gate_closed` or held-out error blow-up
across seeds) and the disjoint gate does not, the mechanism is overlap, not
width, and H-C's framing must change.

**H-C.3 — the basin is seed-dependent.** Prediction: across random
initialisations at a wide gate, a minority of seeds land in a qualitatively worse
held-out error; at a narrow gate, none do. *Measurement:* per-seed held-out error
at each width, reported as a distribution, not a mean. This is the dependent
variable already named for H-C (`frac_gate_closed`), extended to error.

## D-1 — oracle shrinkage bound. [generic] Diagnostic, zero training.

Fit `pred' = a · pred^λ` on the **evaluation** data (deliberately cheating, so
the result is an upper bound on any de-shrinkage method). Under imperfect
ranking, the MSE-optimal prediction shrinks toward the mean; if the oracle λ is
below 1 at short horizons, the model should shrink *more*, not less.

*Why it is registered:* it pre-empts a whole family of interventions — peak
weighting, large-value losses, output-side normalisation — by measuring their
ceiling before any is tried. **Rule: none of those is attempted here until D-1
has been run on the same predictions, and only if D-1 shows headroom.**

*Requires:* saved out-of-fold predictions. The experiment harness does not save
them yet; this is a change request to the experiment lane.

## D-2 — rank ceiling per horizon. [generic] Diagnostic, zero training.

For each scored horizon, estimate how many cells have county-level ranking that
is intrinsically unpredictable, by ranking with an **illegal** information source
(the early part of the future truth) and measuring what even that achieves. Where
the ceiling is low, the MSE-optimal solution is to get each county's level right,
and a model with dynamics is forced toward a constant.

*Prediction it sharpens:* the short-horizon null in EXP05 may be a property of
the metric's structure at those horizons rather than of the model. *Consequence:*
model comparison is reported **per horizon**, and the paper says why.

*Requires:* only the panels. Can run now.

## H-A1 / H-A2 — covariate routing. [directional] Already registered.

Already in `PREREGISTRATION_asymmetry.md`. The channel adds two cautions that are
adopted: (i) the source has **no surviving mechanistic explanation** for why the
families would hurt the interruption side, so no mechanism is asserted here
either — the sign flip is the whole claim; (ii) the two families are different
operations (one time-invariant, one time-varying) and **must be reported
separately even if both go the same way.** They already are.

## H-F — shift-level restoration. [directional]

The restoration rate's logit is recomputed only at the start of each block of N
hours and **held** within the block (sample-and-hold, not block-mean; the two are
not equivalent for a linear map and move in opposite directions). Hypothesis:
repair is scheduled by shift, not re-planned hourly.

**Kill conditions, sharpened by the source's own failed corroboration:**
* The effect must **peak** at some block length and fall off on both sides. A
  monotone improvement with block length is smoothing, not a shift mechanism, and
  kills the hypothesis.
* The block length must be swept over its neighbourhood, never reported as a
  single point.
* An architecture-independent estimator (a linear model with the same
  sample-and-hold applied to its restoration-side inputs) must show the same
  peak. Model-internal ablation alone does not confirm.

## H-G — seed variance is localised in the switch. [directional]

Cross-seed differences in held-out error are predicted to be (a) mostly
under-prediction, and (b) concentrated in a small number of large county-event
waves where a poorly initialised gate did not open.

**Test:** decompose the cross-seed error variance by county-event; report the
sign of the dominant residual and the share of variance carried by the top few
events. **Kill:** if the variance is spread broadly, or is over-prediction, the
localisation claim dies and gate-ensembling is not pursued.

**If it survives:** an in-model gate ensemble (several gate heads averaged) is
compared against a whole-model ensemble of the same cost. This arm is not run
unless H-G survives.

---

## H-H — a leaner rate architecture: one trunk, two rate heads, three gates. [directional]

Received from the PI on 2026-09-02, verbally, as an architectural preference
formed on the related non-public dataset. Recorded here with **every number and
every model name removed**; what crosses is the mechanism only.

*The prior.* A rate architecture built from **one** network rather than two
parallel ones, with four components that each carry an independent mechanism:
(1) a single trunk producing the rate logits; (2) an event gate on the
interruption rate (already registered as H-C, noisy-OR form, `GatedRate`);
(3) a **small-magnitude pathway** — a component dedicated to the low-range
regime of the target; (4) a **persistence gate** — a component that decides
how much of the current state carries forward. The sender's stated reason is
parsimony and narrative cleanliness: fewer parameters, no component that
amounts to averaging two interchangeable networks.

*What it cannot mean here, stated now.* This project's two rate networks are
not interchangeable and are not averaged: one is the interruption rate acting
on (1 − y), the other the restoration rate acting on y, and their separation is
the paper's subject (Prop. 1–2; H-E). "One network" therefore translates to
**one shared trunk with two rate heads** (logit_U, logit_R from the same hidden
layer), which halves the parameter count and keeps the two rates. It does not
translate to one signed rate — that arm exists (`net_scaled`), loses at h+48
(+2.0%, 12/15) and reverses by family, and re-running it under a new name
would establish nothing.

*Components (3) and (4) are not yet defined in public terms.* Their mechanism
must be stated by the PI (or sent through the controlled channel, model-level
only) before they are implemented; nothing is guessed. Candidate readings,
listed so the eventual definition can be matched against them rather than
invented after the fact: a separate low-range output branch active near y ≈ 0
(small-magnitude pathway); a learned convex combination between the rollout
step and the carried-forward state, or a gate on the restoration term
(persistence gate).

*Fixed interpretation.* The lean architecture becomes the main model **only
if**, on g2 under the full protocol: (a) it is not worse than the two-rate
control at any horizon by the sign gate (no horizon with ≥ 12/15 units worse)
and better at ≥ 1 horizon by ≥ 12/15; (b) its parameter count is at most the
control's; (c) on g3 it reproduces the H-E pattern against its own
parameter-matched single-rate counterpart (tropical advantage, winter
reversal) — if the winter reversal disappears, the mechanism claim is
withdrawn, not the model adopted. Otherwise the two-rate control stays the
main model and the lean variant is reported as an ablation. **The paper's
provenance statement must say the direction was suggested.**

## Methodological practices adopted. [generic]

Not hypotheses. Recorded so they are applied consistently.

1. Bitwise comparisons are made in the dtype the predictions were produced in.
   A decimal archive round-tripped through float32 is not the same number.
2. An absolute-threshold bitwise gate has power that jumps across binary
   intervals with the magnitude of the values. A gate that always passes may
   have no power.
3. Corroborating a specific hyperparameter value requires sweeping its
   neighbourhood. A single point cannot distinguish "this value" from "this
   direction". (Applied to H-F.)
4. Align semantics, not names. Before reproducing a structural operator, locate
   the exact line; "every N hours" has two implementations that move in opposite
   directions. (Applied to H-F.)
5. Relabelling is not a placebo; regrouping is. A relabelled grouping is
   bit-identical to the original by construction.
6. "Explains every observation so far" is a hypothesis's entry condition, not
   its pass condition. An explanation that also covers the counter-examples is
   the one most in need of a test.
7. "Untested" and "refuted" are different states. A failed test of low quality
   neither kills nor rescues.
