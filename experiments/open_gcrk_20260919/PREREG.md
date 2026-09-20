# PREREG — AsymODE host (W) vs AsymODE + GCRK on public county outage data

Written 2026-09-19, before any model was trained and before any forecast-window
(hour >= 72) outage value was inspected. What had been read at this point:
NOAA Storm Events metadata, ERA5 fields, EAGLE-I observation availability (for the
data gates), and prefix outages (hours 0-71; for the fold stratification and the
recovery inputs). The scripts named below are the rule; this file restates them.

## 1. Question

On public data, under the same architecture and training protocol as the paper's
method, does the Geography-Conditioned Response Kernel (GCRK) inside the damage
network improve county outage-fraction trajectories over the same host without it
(W)? Five initialisation seeds per arm, all outer folds. Diagnostics then ask where
any difference comes from (joint training vs the kernel's output, the county's own
geography vs a random county's, a few counties/events vs many).

## 2. Events (`select_events.py`, table `event_selection.csv`)

Candidate pool: the 11 convective days of `configs/panel_manifest_g2-convective-11.json`
plus every "wind"-dominant day of the event-day catalogue in years whose EAGLE-I records
are on disk (2018-2022, 2024) — 40 candidates (one of them, 2022-12-27, lies after the
EAGLE-I archive's actual end on 2022-11-12 and fails F5).

Window: 216 hourly steps from 00 UTC three days before the catalogue day D; hours 0-71
are observed, hours 72-215 (from 00 UTC of D) are forecast by one open-loop rollout
from the observed p_71.

Filters on public metadata only (distinct counties, Storm Events; wind-type =
Thunderstorm Wind, High Wind, Strong Wind, Tornado; convective = Thunderstorm Wind,
Tornado):
F1 gust-driven with convection (wind-type counties >= 50% of outage-relevant counties
in the forecast window, convective reports in >= 100 counties); F2 scale (>= 5 states,
>= 100 counties); F3 calm prefix (prefix wind counties <= 25% of forecast-window wind
counties); F4 multi-day (5-95% span of wind report times >= 24 h); F5 EAGLE-I on disk;
R1 second wave (a later day >= 24 h after the first peak day with >= 25% of its wind
counties, or wet/cold reports in >= 50 counties afterwards). Passing candidates are
accepted in decreasing order of forecast-window wind counties if their windows do not
overlap, at most six.

Result (reasons for every candidate in `event_selection.csv`):

| event (D) | window (UTC) | why selected |
|---|---|---|
| 2019-03-13 | 03-10 00:00 - 03-18 23:00 | gust-driven with convection, calm prefix (0.10), wind then wet/cold (blizzard, flood) reports in 109 counties |
| 2021-03-26 | 03-23 00:00 - 03-31 23:00 | convective outbreak, then squall-line and high-wind days two to three days later |
| 2021-12-11 | 12-08 00:00 - 12-16 23:00 | convective outbreak on D, second wind wave (derecho and high wind) on D+4/D+5 followed by snow |
| 2022-06-08 | 06-05 00:00 - 06-13 23:00 | convective wind on D, second and third convective waves on D+2 and D+4/D+5 |
| 2024-02-27 | 02-24 00:00 - 03-03 23:00 | high wind and a convective line, then a cold-sector snow process (208 counties) |

Not selected, by first failing filter: 17 F1 (synoptic wind, winter or tropical systems
without a convective component), 15 F3 (active prefix), 1 F4 (single burst, 2021-12-15),
1 R1 (no second wave, 2022-04-22), 1 F5; no accepted window overlapped another. Nine of
the 11 convective-pool days fail F3; 2021-12-11 and 2022-06-08 pass everything.

## 3. County set and data (`build_panel216.py`, `panel_gates.csv`)

County set of an event = counties with >= 1 wind-type report starting in its forecast
window, then data gates: G1 EAGLE-I 2024 modelled customers >= 500; G2 state EAGLE-I
coverage >= 70% (coverage history ends in 2022; 2024 uses 2022); G3 hour 71 observed;
G4 >= 90% of prefix hours and >= 90% of forecast hours observed.

| event | footprint | after gates | states |
|---|---:|---:|---:|
| 2019-03-13 | 400 | 321 | 23 |
| 2021-03-26 | 651 | 544 | 39 |
| 2021-12-11 | 1133 | 991 | 41 |
| 2022-06-08 | 470 | 385 | 35 |
| 2024-02-27 | 445 | 419 | 34 |
| total | 3099 | 2660 county-events, 1756 counties | |

Target: p_{i,t} = EAGLE-I customers out / EAGLE-I 2024 modelled customers, the hourly
mean of the observed 15-minute cells (densification and mask rule of
`src/asymode/panel.py`), clipped to [0, 1]. Unobserved cells are excluded from every
loss and metric (99.9% of cells are observed).

Inputs (`build_features.py`):
* damage x^U (40): the 14-channel ERA5 block (12 area-weighted fields + sin/cos UTC
  hour) and 26 causal weather summaries (7 instantaneous hazard composites, 5 path
  summaries cumulated from hour 72, 8 trailing windows, 2 freeze-cycle counts,
  4 direction terms); every summary at hour t uses hours <= t;
* occurrence gate (7): the instantaneous hazard composites;
* recovery x^R (33): the 14 weather channels, 6 county background variables (log
  EAGLE-I customers, USDA ERS RUCC 2023, log Census population density, EIA-861 2023
  cooperative share, log(1 + number of utilities), log(1 + SAIDI)), 5 prefix-outage
  summaries (p_71, prefix max, mean over hours 66-71, p_71 - p_65, share of prefix
  hours with p > 0.005), 8 five-nearest-county weather summaries;
* geography g (enters only through GCRK; `build_geography.py`): terrain from USGS 3DEP
  at 150 m (mean elevation, p95-p5 relief, mean slope, share of slope >= 10 deg,
  ruggedness, 8 aspect shares), canopy and land cover (USFS NLCD tree canopy 2021:
  mean, share >= 50%, mean on developed land; Annual NLCD 2021: forest, developed and
  wetland shares, forest on slopes >= 10 deg), SSURGO soils via Soil Data Access
  (poorly drained, hydric, restrictive layer < 50 cm, annual minimum water table
  < 30 cm, root-limiting = any of the three), windthrow susceptibility = forest share
  x root-limiting share, log land area, and 50 km land-area-weighted smoothed
  versions of relief, canopy, windthrow susceptibility and poor drainage.
  Not rebuilt: soil-forest co-location (the gridded soil services do not return
  values; county shares are multiplied instead, an independence approximation) and a
  national windthrow-hazard interpretation (SSURGO has one for two states only;
  root-limiting soils stand in for it).

The target, the loss and every metric are defined on the outage fraction p alone.

## 4. Models

W and GCRK as in the paper (`src/asymode/asym_host.py`, `src/asymode/gcrk.py`).
Damage: two-hidden-layer MLP (width 32) on x^U, learned logit smoother, occurrence
gate, background rate; recovery: independent two-hidden-layer MLP (width 16) recomputed
every hour; population-balance stock update. GCRK replaces the damage MLP's second
linear layer (same W2, b2) with the single response state, code dimension 4.
Its forward pass equals the frozen reference layer element-wise on synthetic inputs
(`tests/test_gcrk_equivalence.py`: float32 <= 1e-6, float64 <= 1e-12, including
calibration, gating, state, readout, drop-path and gradients).

Deliberate differences from the reference host, all forced by the public setting or the
brief: the loss uses the outage fraction only; inputs are the public analogues above;
the reference's state indicators are replaced by the county background variables (the
panel spans 23-41 states per event); the reference's recovery input family that it
zeroes at input and initialisation is omitted (a no-op); the kernel's private seed
differs (same initialisation recipe, checked against the reference seed in the test).

## 5. Training protocol (frozen reference; `src/asymode/gcrk_train.py`)

Adam, damage and kernel lr 0.003, recovery lr 0.0003. INNER: three county-grouped
inner folds trained in lockstep, evaluated every 10 steps (step 0 included) on their
held-out inner units, pooled masked MSE decides; >= 400 steps, patience 200, <= 1600.
REFIT from scratch on all development units for the selected t*, export OUTER in
evaluation mode. GCRK only: drop-path 0.2 (no rescaling), linear opening over 200
steps, calibration refreshed from the fitting units every 10 steps and before every
evaluation. Loss = selection criterion = masked MSE of p over hours 72-215.

## 6. Splits and seeds (`run.py splits` -> `splits.json`)

* main: 5 county-grouped outer folds (every event of a county held out together);
  inside each, 3 county-grouped inner folds. Assignment: counties ordered by (first
  event, state, maximum prefix p, customers), consecutive blocks receive a seeded
  random permutation of fold labels (split seed 20260919; inner seeds 20260920+).
  Five folds rather than ten: the panel has 11 times the units of the reference study,
  and five folds keep the whole campaign (both designs, both arms, five seeds) inside
  one working day on this machine; the choice is fixed here, before any fit.
* loeo (robustness): each event held out in turn; its counties may appear in the
  development set through other events (event transfer, not county transfer).
* seeds 0-4 initialise the networks; W and GCRK of the same (seed, fold) share the host
  initialisation; splits do not depend on the seed.

## 7. Baselines (no training)

all-zero; persistence (p_t = p_71 for every forecast hour); TimesFM 3.0 zero-shot,
official code and weights run locally (academic evaluation only, weights not
redistributed): context = customers out over hours 0-71 (unobserved hours filled
forward), the 14 ERA5 channels over hours 0-215 as past-and-future covariates, point
forecast = the model's median, clipped to [0, customers], divided by customers. A
history-only TimesFM arm is reported alongside.

## 8. Metrics (pooled over OUTER observed cells, every unit, every fold)

* primary: RMSE of p over forecast hours 72-215; also MAE;
* RMSE by lead l = t - 71: 1-6 h, 7-24 h, 25-48 h, 49-144 h;
* peaks, for units whose observed forecast-window maximum is >= 0.01: peak magnitude
  error max p_hat - max p and peak time error argmax p_hat - argmax p (hours); mean and
  median absolute values;
* secondary threshold diagnostics: share of observed-zero hours with p_hat > 0.001;
  share of active hours (p > 0.01) predicted below half the observed value;
* W and GCRK per seed, mean +- sd over seeds, paired seed-wise differences (GCRK - W)
  and their sign counts; the same by event.

## 9. Interpretation rules

* GCRK is "better" (worse) than W on a metric if the mean paired difference is
  negative (positive) and at least 4 of 5 seeds agree in sign; 5/5 = consistent;
  otherwise "no consistent difference". Relative changes are means over seeds of
  100 (GCRK/W - 1). No other significance claim.
* Kernel exit: the same trained GCRK network evaluated with the response contribution
  removed (a configuration drop-path trains) splits GCRK - W into joint training
  (closed - W) and kernel output (open - closed), per seed.
* Before any result is interpreted, every cell is checked: t* at 0 or 1600, nonfinite
  values, kernel opening |tanh(alpha)| < 1e-3, a fold/seed whose OUTER RMSE exceeds 1.5x
  the median of the same arm and fold. Anomalies are examined and reported, not dropped.

## 10. The PI's three expectations, registered as hypotheses

* H1 (concentration): the GCRK - W gain comes from few counties or events. Test: per
  unit, D_u = SSE_u(W) - SSE_u(GCRK); share of the total from the top 10 units and the
  top 1% of units, and from each event; pooled difference recomputed without the top
  1% of units and without each event. Supported if removing the top 1% reverses the
  sign of the pooled RMSE difference in >= 3 of 5 seeds, or if one event carries
  > 50% of the total gain.
* H2 (geography): a county's own geography is not better than a random county's.
  Test, per seed on the main design: for each OUTER unit, its fold's GCRK model is run
  with its own geography and with those of 200 donor counties (seeded draw from the
  other panel counties, standardised with that fold's statistics); own-geography
  percentile among donors (share of donors with lower RMSE); units whose RMSE varies
  across donors (sd > 1e-5) are tested with a one-sided Wilcoxon signed-rank test of
  percentile - 0.5 < 0. H2 is supported if p >= 0.05 in >= 3 of 5 seeds.
* H3 (initialisation): single-seed county-level differences are dominated by the
  initialisation. Test: Spearman correlation of D_u between the 10 seed pairs, and the
  share of units with |D_u| above its median whose sign agrees in >= 4 of 5 seeds.
  Supported if the mean pairwise correlation is < 0.3.

## 11. Figures 3 and 4: fixed choice rules (Storm Events and observed data only)

* Representative event: the event with the most panel counties that have wind-type
  reports both on wave-A day (first forecast day with >= 50% of the maximum daily wind
  counties) and on a wave-B day (>= 2 days later with >= 25% of that maximum); ties
  broken by the larger ratio of the strongest wave-B day to the wave-A day.
  Computed now: 2019-03-13 0, 2021-03-26 11, 2021-12-11 28, 2022-06-08 28, 2024-02-27 8
  -> 2021-12-11 (ratio 388/406 against 109/197). Wave A = hours 72-95, wave B =
  hours 168-215; the push window of panel (b) is hours 168-215.
* Representative county (Figure 3): among that event's two-wave counties with
  >= 10,000 customers, the largest observed rise of p inside the push window
  (max over hours 168-215 of p minus p at hour 167). Model: its own main-design outer
  fold, seed 0.
* Figure 4 (8 panels): the Figure 3 county; for each event, the county with >= 10,000
  customers and the largest observed forecast-window peak; then the next two-wave
  counties of the representative event by push-window rise until eight are chosen.
  Curves: observed, GCRK (seed 0, own outer fold), TimesFM.

## 12. Compute estimate

Measured on this machine: one GCRK training step costs ~0.3 ms per unit, W ~0.075 ms.
Main design: 25 GCRK + 25 W cells, loeo: 25 + 25. Expected ~30 CPU hours, ~7-8 hours
on five single-thread workers (memory limits the machine to about five); TimesFM and
the diagnostics add about two hours. Below the 24-hour threshold, so the design runs
unreduced.

## 13. Reported regardless of outcome

Every seed, fold, event and arm; negative and null results; the per-unit and per-event
gains behind any pooled number; all pathology checks.

## Amendment 1 (2026-09-19, after the main and leave-one-event-out results; PI request)

Nothing above is changed; the pre-registered analysis and its results stand as run.

1. Erratum to section 3. SSURGO publishes the national windthrow-hazard interpretation
   'FOR - Windthrow Hazard' for survey areas in every state (3,379 survey-area
   interpretations in Soil Data Access); section 3 said it exists for two states only,
   which confused it with the state rules of Vermont and Michigan.
2. Nine further public descriptors are built by `build_geography_ext.py`: soil x land-cover
   co-location on one sample lattice (gNATSGO map units and Annual NLCD 2021): wet-soil
   share, windthrow-hazard share, forest-and-wet share, wet share within forest, hazard
   share within forest; forest within 150 m of developed land; five-point 3DEP elevation
   mean and range; FIA forest-land share. They enter only a new, separately labelled round.
3. Frozen-checkpoint diagnostics (`diagnostics_frozen.py`) are post hoc and descriptive.
4. Any new training round (inputs, geography or model-selection changes) is specified
   here before it is run, with W and GCRK changed together wherever an input both read.

## Amendment 2 (2026-09-19, written before any run below; PI instruction to proceed)

**E2, step-matched W refit (diagnostic).** In every main and LOEO cell, W is refit from
scratch on the cell's development set for exactly the number of steps GCRK selected in that
cell (same seed, data and optimiser; with the same initialisation and full-batch steps this is
W's own refit path stopped at GCRK's t*). Its OUTER forecasts are compared with W at its own
t*, GCRK and GCRK exit closed on RMSE, false activity and the gated-damage pathway. Reading: if
W at GCRK's t* is close to the closed-exit network, the closed-exit gap is a training-length
effect; if it stays close to W at its own t*, it comes from training with the kernel.

**R2, input and protocol round (both arms change together).**
(a) Gust is ERA5 '10m_wind_gust_since_previous_post_processing' (the hour's maximum) instead of
the instantaneous gust, in every channel and summary that uses gust, own county and neighbours.
(b) Gust exceedance energy, wet wind, snow-ice load, near-freeze and cold precipitation are
computed on each ERA5 cell and then area-averaged over the county (their 6 h and 12 h sums
follow); two damage inputs are added: the county's highest cell gust and the share of county
area whose cell gust exceeds 15 m/s.
(c) The five path summaries become trailing 72-hour summaries (maximum gust; sums of gust
exceedance energy, wet wind and snow-ice load; hours since the 72-hour maximum), defined the
same way in the prefix and the forecast window.
(d) GCRK reads the 40 descriptors of Amendment 1.
(e) LOEO selects t* on inner folds that each leave one development event out (four folds);
the main design is unchanged.
Screen: seed 0 only, main and LOEO, both arms (20 cells). Seeds 1-4 follow only if R2's W
improves on round 1's W at seed 0 by more than round 1's seed-to-seed standard deviation of
W's RMSE in at least one design (main 0.00036, LOEO 0.00102); otherwise R2 is reported as not
better and stops. GCRK - W inside R2 at one seed is descriptive only. Evaluation as in round 1,
plus cluster-bootstrap intervals (counties, event x state blocks) and the oracle rescaling.

**E3, more events (data only here).** Candidate pool: the pre-registered one
(`event_selection.csv`). Rules F2, F4, F5 and R1 are kept; F1 (convective gusts) and F3 (calm
prefix) are dropped. The five pre-registered events are kept and the remaining candidates are
added in decreasing order of forecast-window wind-report counties, skipping any window that
overlaps one already taken, until there are twelve events. Panels, gates and descriptors are
built by the same scripts. Which models are trained on the twelve events, and with which
held-out design, is fixed in a further amendment before any such run.

Note to Amendment 2 (E3 data build, before any E3 training): candidate 2024-05-26 passes the
metadata rules but gate G4 leaves no county (an EAGLE-I collection gap of about 9.5 hours on
2024-05-24 puts every county's prefix below 90% observed hours). It fails F5 in substance; the
next candidate in the fixed order, 2024-05-08, takes its place (`select_events_e3.py`).

## Amendment 3 (2026-09-19, written before the R2 screen has any result and before any E3 run)

**E3 training.** Data: the twelve events of Amendment 2 with its note (2024-05-08 in place of
2024-05-26). Inputs: the R2 set if the R2 screen passes its continuation rule, otherwise the
round-1 set; both are built by the same scripts for the twelve events, and GCRK reads the 40
descriptors (built for the added counties by the same scripts). If R2 passes, R2's seeds 1-4 on
the five events are not run: E3 contains those events and supersedes them.
Designs: (i) county-grouped, five outer folds with three county-grouped inner folds, built as in
round 1; (ii) event-grouped, four outer folds of three events each (events in date order, fold =
rank mod 4), each selecting t* on three inner folds of three development events (rank mod 3).
Protocol, optimiser, stopping rule and export as in round 1. Arms W and GCRK, seeds 0-4, run
seed by seed.
Primary contrast: GCRK vs W on the pooled full-rollout RMSE of held-out county-hours in each
design, with 95% cluster-bootstrap intervals over counties and over event x state blocks, the
event-equal-weighted RMSE (mean over events of each event's RMSE) and seed agreement. Reading:
"GCRK better" or "worse" only when the county-cluster interval of the relative RMSE change
excludes zero and at least four of five seeds agree; otherwise "not distinguishable". Also
reported: lead segments, MAE, peak errors, false activity at 0.001 and 0.005, the exit-closed
decomposition, oracle rescaling, the E0 residual test and the effective number of events.
Compute: about 90 CPU hours; the number of workers is set by measured memory, not cores alone.

Note to Amendment 3 (before the twelve-event run): in the R2 screen the leave-one-event-out
design with event-held-out inner folds selects 50-110 steps, i.e. a barely trained model. The
kernel's warm-up is 200 steps, so its opening is still ramping and W, GCRK and GCRK with the exit
closed coincide (0.0297 each at seed 0, against 0.0310 for the round-1 W); RMSE improves while MAE
and false activity get worse. The same is expected in the event-grouped design of the twelve-event
round: it measures how far a shrunken model transfers, and cannot test the kernel. The
county-grouped design remains the test of the kernel. Both are reported, and each event-design cell
also reports its kernel opening at t* (ramp x tanh(alpha)) so that a coincidence of the arms is
visible rather than read as "no kernel effect".

## Amendment 4 (2026-09-20, before the run; diagnostics only, no retraining of W or GCRK)

Three checks of a second external review, on the twelve-event data and the seed-0 cells of E3.
D1 (descriptive). Is the host a calibrated conditional mean? Held-out units in ten quantile bins of
the predicted forecast-window peak: mean predicted and observed peak, share of units with an
observed peak below 1% and above 10%; the same for county-hours in bins of predicted p.
D2 (descriptive). Does a stormy prefix weaken the kernel? By event: GCRK - W, the pre-registered
prefix-to-forecast wind-report ratio, and from frozen replays the kernel's forecast-window deposit
norm and gate occupancy.
D3 (information ceiling). A gradient-boosted regressor (scikit-learn HistGradientBoostingRegressor,
fixed settings: 300 iterations, learning rate 0.05, 15 leaves, min 20 samples per leaf, L2 1.0, no
early stopping, no tuning) predicts each held-out unit's forecast-window peak and mean outage
(raw, and log(x + 0.002)) from unit-level summaries: forecast-window maximum and mean and prefix
mean of the 42 damage inputs, forecast-window maximum of the 8 neighbour summaries, the 6 county
context variables and the 5 prefix-outage summaries. Feature sets: without geography; with the 40
descriptors; with the descriptors permuted within event x state blocks (20 permutations, the
null). Folds: the E3 county-grouped and event-grouped outer folds. Reported: out-of-fold RMSE and
R2, the gain from geography with a county-cluster bootstrap interval and its place in the
permutation null, and the same unit-level targets implied by W and GCRK (seed 0) for reference.
Reading fixed in advance: geography carries usable conditional information at this resolution
only if the gain exceeds the permutation null's 95th percentile in the county-grouped design; if
the regressor predicts unit peaks clearly better than W, the host leaves information unused; if
it does not, the miss of peaks is an information limit of these inputs rather than of the model.
