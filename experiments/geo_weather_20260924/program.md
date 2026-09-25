# program.md — how this line of work runs (read before every iteration)

Goal: a principled, general way for many kinds of geography and many kinds of weather to interact in
an outage model, so that geography carries information the weather host cannot get on its own.

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
3. Survivors go to the full protocol (five folds, inner early stopping, refit) before any claim; several
   seeds only after a clear single-seed gain.
4. Immutable: targets, masks, folds, the evaluation code, the outer split. The data pipeline is
   versioned (data_v3, ...): a new version is a new directory, never an in-place edit.
5. Every attempt gets a line in RESEARCH_LOG.md (hypothesis, change, result, keep/discard, file).
   Every number in a document points to a file under results/.
6. Public data and public literature only; every input's source, request and checksum is logged in
   data_provenance/.
7. Sync to GitHub (this branch) after each stage that changes the design or the evidence.
