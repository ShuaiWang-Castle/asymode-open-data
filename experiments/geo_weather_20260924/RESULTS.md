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

## 2. Screens on the wind panel

| arm | change | RMSE | vs base | county interval | event x state interval | events better | file |
|---|---|---|---|---|---|---|---|
| base | W+Cin, round-2 inputs (area weights) | 0.023914 | - | - | - | - | results/screen_S1.json |
| pop | every host input population-weighted (data v3p) | 0.023854 | -0.25% | [-3.66, +2.96] | [-4.60, +3.74] | 5/12 | results/screen_S1.json |

(Further rows are added as the screens finish.)
