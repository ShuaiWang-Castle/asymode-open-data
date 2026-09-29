# program.md — how this line of work runs (read before every iteration)

Goal: a principled, general way for many kinds of geography and many kinds of weather to interact in
an outage model, so that geography carries information the weather host cannot get on its own.

## Core research chain (PI clarification, 2026-09-28)

**Weather process -> complex geographic modulation -> impact formation, interaction and accumulation -> outage.**
This is the organizing principle for future data analysis, kernel design and manuscript motivation.
The geographic kernel represents how local conditions transform weather histories into latent impacts;
the existing damage and outage dynamics map those impacts to observed outages, with recovery accounted for.
Conditional outage-response analyses are evidence about symptoms of this chain, not a replacement of it
by an unrestricted geography-to-outage predictor. Keep geography within the hidden kernel and retain the
single damage MLP. A learned impact state is not directly observed physical damage, and outage stock alone
does not identify damage and recovery separately. Distinguish observed associations, structural assumptions
and confirmed mechanisms in every future claim. See `notes/KERNEL_HIGH_DIM_DATA_MOTIVATION_20260928.md`.

**PI clarification: allow general geographic combinations, not named mechanism recipes.** All available
geographic coordinates must be eligible to interact through a learned joint conditional map. Examples such
as terrain and vegetation explain the idea but must not define allowed pairs, required relationships,
signs, semantic interaction masks or a fixed maximum interaction order. Semantic groups may organize
reports, not exclude cross-group or within-group combinations from the candidate map. Preserve access to
the full geographic vector rather than forcing every candidate through the old four-dimensional code.
Finite capacity, regularization and observed support still limit what can be learned: representation access
is not proof that all combinations exist, are identified or generalize. Numerical stability constraints
remain appropriate. The concrete current proposal is
`notes/KERNEL_CONTROLLED_RELAXATION_DESIGN_20260928.md`: all-subset geographic kernel features plus
full-vector access, explicitly separated write/retain/transform/read heads, and four bounded controlled
relaxation states inside the single damage network. It is non-nested with original GCRK. The PI subsequently
authorized full training against no-kernel AsymODE, with an affected-county improvement target near 10%;
the separate execution registration is `notes/I20_CONTROLLED_RESPONSE_SCREEN_20260928.md`. Earlier generic
state-dependent-head sketches remain alternatives, not extra modules silently added to this proposal.

Current operating rules and completed D-stage evidence are in `aris/CYCLE.md`. The campaign findings
below concern the older twelve-event panel; they are historical context, not established facts on current D.
Global prediction error is not a gate for excluding county-level phenomena from exploratory analysis.

## What the last campaign established (experiments/open_gcrk_20260919/RESULTS.md, sections 11-21)

* The host's own ERA5 channels already encode a county's geography (elevation R2 0.997, canopy 0.84,
  median over 40 descriptors 0.72): a county-level geography vector is largely redundant.
* What weather leaves unexplained is a county x event residual (repeatable county effect ICC 0.06).
* Any county-level conditioning of the kernel fingerprints counties (true = permuted = regime-level
  geography, all 2.7-2.9% worse than the same kernel without geography); learned slowly it is harmless.
* What helps: better weather inputs (-2.9%) and county context in the host's first damage layer (-2.4%,
  every cluster interval below zero, seed 0).

The working hypothesis that follows: the useful interaction is local and event-specific - the event's
weather field meets the distribution of local conditions *inside* a county (elevation against the
freezing level, ridges against the gust field, saturated soils under canopy against wind). That is
exactly the county x event residual, and it is not recoverable from county means.

## Rules

1. Metric for iteration (fixed): pooled hourly RMSE of the outage fraction on held-out counties,
   twelve events, round-2 inputs unless a data version says otherwise. Quick screen = outer folds 1-2 of
   the county-grouped design, fixed 900 steps (no early stopping, so arms are compared at one budget),
   seed 0, paired initialisation. Report with the county-cluster bootstrap interval against the base.
2. Keep / discard: a change is kept only if it beats the current base by more than 1% on the screen with
   the county interval below zero, and beats its placebo (same structure, geography permuted across
   counties) - the two-numbers rule: against the base (can it deliver) and against the placebo (is the
   information there).
3a. (Added 2026-09-25 after the cleaning screen: -2.66% on folds 1-2, -0.36% on folds 1-4.) A screen keep, and its placebo, share the screen's two folds, so a keep is provisional until the other three folds (or a second seed on all five) agree; only then is it called a gain.
3. Survivors go to the full protocol (five folds, inner early stopping, refit) before any claim; several
   seeds only after a clear single-seed gain.
3b. (Added 2026-09-26 with DATASET_DESIGN v1 section 9.2; replaces rules 1-3 on the designed panel.) The metric is the
   regime-balanced skill of DATASET_DESIGN section 1, on event folds. A single-seed five-fold screen may discard, never
   keep; a keep needs three seeds (seed-averaged predictions) on all five folds beside its twin; a confirmation uses
   five seeds on all of development; the seed count is recalibrated by the Stage 0 A/A run. Keep (DATASET_DESIGN
   amendment 2 S10): seed-averaged over three seeds on the five event folds, the regime-balanced gain exceeds 1% against
   the host and is positive against the twin, each gain with its 95% family-cluster bootstrap interval above zero, and
   no non-headline regime loses more than 2%. Discard: a single-seed five-fold screen whose regime-balanced gain
   against the host is <= 0.
4. Immutable: targets, masks, folds, the evaluation code, the outer split. The data pipeline is
   versioned (data_v3, ...): a new version is a new directory, never an in-place edit.
5. Every attempt gets a line in RESEARCH_LOG.md (hypothesis, change, result, keep/discard, file).
   Every number in a document points to a file under results/.
6. Public data and public literature only; every input's source, request and checksum is logged in
   data_provenance/.
7. Sync to GitHub (this branch) after each stage that changes the design or the evidence.
