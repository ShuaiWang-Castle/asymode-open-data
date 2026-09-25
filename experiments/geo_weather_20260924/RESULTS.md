# RESULTS — geo-weather line (exposure-integrated local hazards)

Every number points to a file under `results/` or `data_provenance/`; the attempt log is `RESEARCH_LOG.md`, the
design `DESIGN.md` (v1), the literature `LITERATURE.md`.

Screen protocol (program.md): county-grouped outer folds 1-2 of the twelve-event wind panel (2,489 held-out
county-events), a fixed 900 training steps on all development units, seed 0, paired initialisation; pooled hourly
RMSE of the open-loop rollouts; intervals are cluster-bootstrap 95% (counties; event x state), 2,000 draws.
Relative change = RMSE(arm) / RMSE(base) - 1. The screen resolves about +-3%: a keep needs > 1% with the county
interval below zero and a win over its placebo.

## 1. F0: what the sub-grid structure can act on (no training)

`results/F0/f0_audit.{json,md}`. Exposure-integrated hazard features (DESIGN v1 section 2) on the recoverability
ladder; a difference is material when it exceeds 10% of the feature's 99th percentile.

| contrast | what it isolates | largest share of county-hours (outage-weighted) | features >= 1% |
|---|---|---|---|
| quad - pop | elevation bands inside cells (lapse-rate downscaling) | 0.41% (0.93%) | 0 |
| pooled - mean | the static band distribution under county weather | 0.09% (0.32%) | 0 |
| quad - pooled | co-location of bands with their own cell's weather | 0.17% (0.63%) | 0 |
| pop - area | exposure weighting (population instead of area) | 2.83% (12.17%) | 18 |

On this panel the sub-grid structure has almost nothing to act on: within-county elevation offsets are small
(population-weighted sd of dz: median 16 m, 90th percentile 92 m; `data_provenance/nodes_cs.json`) and the
panel's events are wind and convective storms, where elevation matters only through the rain-snow phase. By the
kill rule of DESIGN section 3 the quadrature is dropped for this panel. Exposure weighting is material, mostly
through the modulators: people live under less canopy than the county's area average (gust x canopy: 12% of the
outage-weighted county-hours differ materially).

**F1: residual structure** (`results/F1/F1_cd.md`, `audit_f1.py`, by the formal contributor). Cluster-robust score
tests of each ladder contrast against the W+Cin held-out residuals, max-T over the 160 features and Bonferroni
over contrasts, with the false-positive rate calibrated on synthetic residual fields: nothing passes (the
strongest, pop - area gust_x10 x canopy, has a final p of 0.145). The power analysis propagates each regressor
through the base's own damage and recovery rates: any of these effects worth 2% of pooled RMSE would have been
detected (power about 1). On this panel the sub-grid and exposure geography channels are absent at a size that
matters, not undetectable.

**F0 on the winter panel W1** (`results/F0_w1/`; section 3). The same audit on eight ice-storm events: the
elevation bands now change eight features by more than the materiality threshold, all near-freezing
precipitation (wet-bulb hats at -1.5, 0, +1.5 C), in up to 1.93% of the outage-weighted county-hours and two to
three events; pop - area reaches 7.2%. Small, but the quadrature passes F0 there, as the phase physics predicts.

## 2. Screens on the wind panel

| arm | change | RMSE | vs base | county interval | event x state interval | events better | file |
|---|---|---|---|---|---|---|---|
| base | W+Cin, round-2 inputs (area weights) | 0.023914 | - | - | - | - | results/screen_S1.json |
| pop | every host input population-weighted (data v3p) | 0.023854 | -0.25% | [-3.66, +2.96] | [-4.60, +3.74] | 5/12 | results/screen_S1.json |
| clean | training loss without EAGLE-I artefact hours (evaluation unchanged) | 0.023278 | -2.66% | [-5.74, -0.06] | [-6.14, +0.16] | 6/12 | results/screen_S2a.json |
| clean placebo | as many hours dropped at random, same event and outage-level bin | 0.024098 | +0.77% | [-0.53, +2.28] | [-0.46, +2.03] | 4/12 | results/screen_S2c.json |
| clean vs its placebo | (second number of the two-numbers rule) | - | -3.40% | [-6.55, -0.63] | [-7.00, -0.37] | 11/12 | results/screen_S2c_vs_placebo.json |
| hazard (area) | W+Cin+H on eih_area (competing hazard, 160 features) | 0.024089 | +0.73% | [-3.48, +4.41] | [-3.36, +4.06] | 6/12 | results/screen_S2b.json |

**Target cleaning** (`build_train_mask.py`; LITERATURE_preprocessing change 4). EAGLE-I stores no zero rows and its
scrapers time out in storms, so hourly series carry artefacts that no weather input explains: one-hour dips and
spikes, plateaus of an identical non-zero count for four days or more (stale maps), fractions at the denominator.
They are 0.40% of the observed forecast hours but 5.1% of the sum of squared targets. Dropping them from the
training loss only, with the evaluation targets and masks unchanged, is the first change of this line that clears
the screen. Its placebo drops as many hours in the same event and outage-level bin at random (3,410 of 3,531
matched; 3.8% of the sum of squared targets). The gain sits on the unflagged hours (-2.89%; the flagged hours
themselves get +0.94% worse, as they are no longer fitted) and in both phases (rise to the peak -3.74%, decay after
it -2.04%; `results/diag/clean_split.json`): the artefacts distorted the learned response everywhere.

(Further rows are added as the screens finish.)
