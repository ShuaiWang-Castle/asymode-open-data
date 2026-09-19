# RESULTS — AsymODE host (W) vs AsymODE + GCRK on public county outage data

Plan and every rule used below: `PREREG.md` (committed before training). Every number in
this file is read from a file under `results/` that a named script wrote from the named
inputs; the provenance table at the end lists them. Grades follow `README.md` of the
repository: [B] = full protocol, >= 3 seeds, sign consistent.

## 0. Bottom line (main design: 5 county-grouped folds x 5 seeds x 2 arms, 50 cells, no failure; leave-one-event-out: 50 more cells)

* Both trained models beat every no-training baseline on RMSE by a wide margin
  (W 0.02630, GCRK 0.02644 against all-zero 0.03049, persistence 0.03207, TimesFM 0.03089).
* **GCRK does not improve on W.** Full-rollout RMSE is higher in 5 of 5 seeds
  (+0.53% on average; paired differences +2.0e-5 to +3.6e-4). By lead time: 1-6 h and
  49-144 h no consistent difference; 7-24 h no consistent difference (2/5 lower);
  25-48 h worse in 5/5 (+3.3%). MAE worse in 5/5 (+6.5%). **[B], negative.**
* GCRK is slightly better on peaks in 4 of 5 seeds: mean absolute peak-magnitude error
  -0.5%, peak-time error 19.3 h -> 18.3 h. It also predicts non-zero outage more often
  in hours that are observed at zero (false-activity share 0.31 -> 0.43, 5/5).
* The kernel's output is used: closing the exit of the trained GCRK network raises RMSE
  to 0.02731 (5/5 above both W and GCRK). The GCRK - W difference splits into a joint-
  training part (closed - W = +1.01e-3) and a kernel-output part (open - closed =
  -0.87e-3) that almost cancel.
* Effects are two-sided and county-concentrated: in every seed GCRK is better in
  1080-1302 of the 2660 county-events and worse in 1358-1580; 1% of the units carry
  52-63% of all gains and 45-60% of all losses; the 2021-12-11 event holds 78% of the
  gains and 80% of the losses. By event, GCRK - W is negative in 2021-03-26 (4/5 seeds) and 2022-06-08 (5/5,
  tiny), positive in 2019-03-13 (0/5 lower) and 2021-12-11 (1/5), mixed in 2024-02-27 (2/5).
* A county's own geography is better than a random donor county's inside the trained
  GCRK (one-sided Wilcoxon p <= 3e-4 in 5/5 seeds; median own-geography percentile
  0.435-0.475), but the RMSE advantage is tiny (own minus median donor between -4.7e-4
  and +6e-6).
* Per-unit GCRK - W differences are weakly reproducible across seeds (mean pairwise
  Spearman 0.23): single-seed county-level differences are dominated by initialisation.
* Leave-one-event-out (robustness, 50 more cells): transferring to an unseen event, both
  trained models lose their edge over the all-zero forecast (W 0.03144, GCRK 0.03063,
  all-zero 0.03049). GCRK is better than W in 4/5 seeds there (-2.5%), and the GCRK network
  with its kernel exit closed is best (0.03003; below all-zero in 4/5 seeds): the
  transferable part is the jointly trained host, not the kernel's output. One pre-registered outlier cell (held-out
  2024-02-27, seed 4) comes from the kernel output extrapolating to 37-70% outages in ten
  counties of the interior West and the Maine coast whose observed peaks were 0-21%.

## 1. What was run

* Data: 5 events chosen by the public-metadata rule (`event_selection.csv`), 2660
  county-events in 1756 counties and 45 states, 216 hours each, 99.94% of county-hours
  observed (`panel_gates.csv`, `results/panel_description.json`).
* Models: W and GCRK re-implemented in `src/asymode/` (GCRK equal to the frozen
  reference layer on synthetic inputs, `tests/test_gcrk_equivalence.py`: float32
  <= 1e-6, float64 <= 1e-12). GCRK adds 574 parameters (31 geographic descriptors).
* Protocol: `src/asymode/gcrk_train.py` (INNER pooled early stopping, REFIT, OUTER export).
  Selected steps t*: W 340-930 (median 620), GCRK 230-640 (median 360); INNER stopped
  at 430-1130; no cell at 0 or 1600 steps (`results/pathology_main.csv`).
* Baselines: all-zero, persistence, TimesFM 3.0 zero-shot with and without ERA5
  covariates (`timesfm_baseline.py`, local weights).
* Compute: 50 main cells, mean 41 min (GCRK) and 15 min (W) on five single-thread
  workers of an 8-core M2 (4 performance + 4 efficiency cores).

## 2. Main table (held-out county-hours pooled over all folds; mean ± sd over seeds 0-4)

382,736 observed county-hours: 15,959 at leads 1-6 h, 47,850 at 7-24 h, 63,797 at
25-48 h, 255,130 at 49-144 h (`results/manuscript_numbers_main.json`).

| model | RMSE all | 1-6 h | 7-24 h | 25-48 h | 49-144 h | MAE |
|---|---:|---:|---:|---:|---:|---:|
| all-zero | 0.03049 | 0.01502 | 0.02874 | 0.02888 | 0.03190 | 0.00452 |
| persistence (p_t = p_71) | 0.03207 | 0.01549 | 0.03046 | 0.03069 | 0.03344 | 0.00500 |
| TimesFM, ERA5 covariates | 0.03089 | 0.01567 | 0.02920 | 0.02884 | 0.03237 | 0.00471 |
| TimesFM, history only | 0.03094 | 0.01552 | 0.03021 | 0.02907 | 0.03223 | 0.00468 |
| W | 0.02630 ± 0.00036 | 0.01340 ± 0.00019 | 0.02628 ± 0.00053 | 0.02470 ± 0.00083 | 0.02728 ± 0.00037 | 0.00563 ± 0.00024 |
| AsymODE + GCRK | 0.02644 ± 0.00030 | 0.01342 ± 0.00013 | 0.02644 ± 0.00040 | 0.02549 ± 0.00034 | 0.02727 ± 0.00035 | 0.00599 ± 0.00029 |

Peaks and thresholds (1609 county-events with an observed forecast-window peak >= 1%):

| model | mean abs. peak magnitude error | mean abs. peak time error (h) | median abs. peak time error (h) | false-activity share | under-half share |
|---|---:|---:|---:|---:|---:|
| all-zero | 0.0987 | n/a (constant) | n/a | 0.000 | 1.000 |
| persistence | 0.0977 | n/a (constant) | n/a | 0.038 | 0.982 |
| TimesFM, ERA5 covariates | 0.0974 | 47.8 | 35.0 | 0.019 | 0.986 |
| W | 0.0718 ± 0.0006 | 19.3 ± 0.9 | 4.0 | 0.309 | 0.605 |
| AsymODE + GCRK | 0.0714 ± 0.0008 | 18.3 ± 0.6 | 3.8 | 0.429 | 0.600 |

False-activity share = share of observed-zero hours with a forecast above 0.001; under-half
share = share of hours with observed p > 0.01 forecast below half the observed value.
On MAE both trained models are worse than the all-zero forecast: the target is zero in most
county-hours, and any positive forecast in those hours adds absolute error.

## 3. Paired seed-wise differences (GCRK - W) and the PREREG 9 verdict

| metric | mean diff | relative | seeds GCRK lower | verdict |
|---|---:|---:|---:|---|
| RMSE all | +1.38e-4 | +0.53% | 0/5 | GCRK worse (5/5) |
| RMSE 1-6 h | +1.7e-5 | +0.13% | 3/5 | no consistent difference |
| RMSE 7-24 h | +1.66e-4 | +0.64% | 2/5 | no consistent difference |
| RMSE 25-48 h | +7.87e-4 | +3.26% | 0/5 | GCRK worse (5/5) |
| RMSE 49-144 h | -9.2e-6 | -0.03% | 3/5 | no consistent difference |
| MAE | +3.63e-4 | +6.51% | 0/5 | GCRK worse (5/5) |
| abs. peak magnitude error | -3.8e-4 | -0.52% | 4/5 | GCRK better (4/5) |
| abs. peak time error (h) | -0.97 | -4.83% | 4/5 | GCRK better (4/5) |
| false-activity share | +0.120 | +38.6% | 0/5 | GCRK worse (5/5) |
| under-half share | -0.0047 | -0.72% | 4/5 | GCRK better (4/5) |

Per-seed values: `results/paired_main.csv`, `results/seeds_main.csv`. Per outer fold, GCRK's
OUTER RMSE is lower than W's in 11 of 25 (seed, fold) cells and its best INNER loss in
10 of 25 (fold 4 is unfavourable to GCRK in four of five seeds).

## 4. Kernel exit: joint training vs kernel output (full-rollout RMSE)

| seed | W | GCRK exit closed | GCRK | closed - W | GCRK - closed | GCRK - W |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.026752 | 0.027663 | 0.026797 | +9.10e-4 | -8.66e-4 | +4.5e-5 |
| 1 | 0.026055 | 0.026811 | 0.026075 | +7.56e-4 | -7.36e-4 | +2.0e-5 |
| 2 | 0.026431 | 0.027006 | 0.026507 | +5.75e-4 | -4.99e-4 | +7.6e-5 |
| 3 | 0.026425 | 0.027823 | 0.026614 | +1.40e-3 | -1.21e-3 | +1.89e-4 |
| 4 | 0.025826 | 0.027232 | 0.026188 | +1.41e-3 | -1.04e-3 | +3.62e-4 |
| mean | 0.026298 | 0.027307 | 0.026436 | +1.01e-3 | -8.71e-4 | +1.38e-4 |

The closed configuration is the trained GCRK network with its response contribution
removed (drop-path exposes it during training). Jointly training the host with the kernel
makes the host alone worse than W; the kernel's output recovers most but not all of it.
The kernel opening tanh(alpha) is -0.83 to +0.88 across cells and its sign is constant
within a seed (seeds 1 and 2 negative, 0, 3 and 4 positive): the state enters through the
same W2 as h, so the sign is absorbed by the weights and is not a pathology.

## 5. By event (full-rollout RMSE; W and GCRK mean ± sd over seeds)

| event | units | all-zero | persistence | TimesFM | W | GCRK | GCRK - W | seeds GCRK lower |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2019-03-13 | 321 | 0.02534 | 0.02528 | 0.02529 | 0.02366 ± 0.00042 | 0.02419 ± 0.00027 | +5.3e-4 | 0/5 |
| 2021-03-26 | 544 | 0.01059 | 0.01094 | 0.01068 | 0.01060 ± 0.00043 | 0.01032 ± 0.00015 | -2.8e-4 | 4/5 |
| 2021-12-11 | 991 | 0.04408 | 0.04692 | 0.04474 | 0.03667 ± 0.00061 | 0.03695 ± 0.00052 | +2.8e-4 | 1/5 |
| 2022-06-08 | 385 | 0.01756 | 0.01755 | 0.01767 | 0.01740 ± 0.00003 | 0.01737 ± 0.00003 | -3.3e-5 | 5/5 |
| 2024-02-27 | 419 | 0.01971 | 0.01989 | 0.01998 | 0.01890 ± 0.00102 | 0.01859 ± 0.00027 | -3.1e-4 | 2/5 |

In 2021-03-26 and 2022-06-08 the trained models are within 3% of the all-zero forecast
(W is 0.1% above it in 2021-03-26); the gain over the baselines comes mostly from
2021-12-11 and 2019-03-13.

## 6. Leave-one-event-out (robustness)

Each event is held out in turn and the other four train (inner folds county-grouped);
every unit is forecast by models that never saw its event. 50 cells (5 events x 5 seeds x
2 arms), no failure; t*: W 220-1280 (median 620), GCRK 220-750 (median 420).

| model | RMSE all | 1-6 h | 7-24 h | 25-48 h | 49-144 h | MAE |
|---|---:|---:|---:|---:|---:|---:|
| all-zero | 0.03049 | 0.01502 | 0.02874 | 0.02888 | 0.03190 | 0.00452 |
| persistence | 0.03207 | 0.01549 | 0.03046 | 0.03069 | 0.03344 | 0.00500 |
| TimesFM, ERA5 covariates | 0.03089 | 0.01567 | 0.02920 | 0.02884 | 0.03237 | 0.00471 |
| W | 0.03144 ± 0.00102 | 0.01416 ± 0.00058 | 0.03117 ± 0.00085 | 0.03292 ± 0.00294 | 0.03187 ± 0.00067 | 0.00692 ± 0.00042 |
| AsymODE + GCRK | 0.03063 ± 0.00140 | 0.01412 ± 0.00044 | 0.02910 ± 0.00144 | 0.02998 ± 0.00132 | 0.03180 ± 0.00160 | 0.00700 ± 0.00049 |
| GCRK, exit closed | 0.03003 ± 0.00064 | | | | | 0.00672 |

* **Under event transfer neither trained model beats the all-zero forecast on RMSE**
  (W 0.03144, GCRK 0.03063 against 0.03049); only the GCRK network with its exit closed
  does, by 1.5% (0.03003; 4/5 seeds below all-zero). By event, the trained models beat all-zero only in 2021-12-11.
* GCRK vs W (paired, PREREG 9): RMSE -2.5% (4/5 seeds lower), 1-6 h -0.2% (4/5),
  7-24 h -6.5% (4/5), 25-48 h -8.4% (4/5): "GCRK better (4/5)"; 49-144 h and MAE no
  consistent difference; peak magnitude -3.2% (4/5); false activity +32% (0/5 lower).
* Kernel exit: closed - W = -1.40e-3 (5/5 negative), open - closed = +6.0e-4 (3/5
  positive). Under event transfer the gain over W comes from the host trained jointly with
  the kernel and drop-path, not from the kernel's output, which on average adds error.
* Pathology (PREREG 9): one cell exceeds 1.5 times its median, held-out 2024-02-27 with
  seed 4 (GCRK 0.0319 against 0.0194 with the exit closed, ratio 1.63). Its kernel output
  lifts the forecast to 37-70% in ten counties, eight in Utah, California, New Mexico and
  Colorado and two on the Maine coast, whose observed peaks are 0-21%; those ten units carry 87% of the excess squared error. Kept.
  Sensitivity without that event (all seeds; not pre-registered): W 0.03317, GCRK 0.03195
  (5/5 lower), exit closed 0.03159 (5/5 lower), all-zero 0.03211
  (`results/sensitivity_loeo_without_2024-02-27.json`).

| held-out event | units | all-zero | persistence | TimesFM | W | GCRK | GCRK - W | seeds GCRK lower |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2019-03-13 | 321 | 0.02534 | 0.02528 | 0.02529 | 0.03624 ± 0.00508 | 0.02999 ± 0.00266 | -6.3e-3 | 4/5 |
| 2021-03-26 | 544 | 0.01059 | 0.01094 | 0.01068 | 0.01297 ± 0.00100 | 0.01142 ± 0.00093 | -1.6e-3 | 5/5 |
| 2021-12-11 | 991 | 0.04408 | 0.04692 | 0.04474 | 0.04233 ± 0.00032 | 0.04192 ± 0.00047 | -4.1e-4 | 4/5 |
| 2022-06-08 | 385 | 0.01756 | 0.01755 | 0.01767 | 0.02091 ± 0.00349 | 0.02164 ± 0.00412 | +7.3e-4 | 1/5 |
| 2024-02-27 | 419 | 0.01971 | 0.01989 | 0.01998 | 0.01977 ± 0.00087 | 0.02190 ± 0.00562 | +2.1e-3 | 4/5 |

The two designs answer different questions. Within events (new counties, seen storms)
GCRK is slightly worse than W; across events (a new storm) it is better than W in four
of five seeds, but both trained models then fall back to about the all-zero forecast.
`results/tables_loeo.md`, `table_main_loeo.csv`, `paired_loeo.csv`, `decomposition_loeo.csv`,
`events_loeo.csv`, `pathology_loeo.csv`, `loeo_cells.csv`.

## 7. Registered hypotheses (PREREG 10)

**H1 — the GCRK - W gain is concentrated in few counties or events.** As registered the
test presumes a net gain; there is none (net SSE change negative in 5/5 seeds), so the
sign-reversal criterion does not apply. What the data show: the per-unit effect is
two-sided and heavy-tailed. GCRK is better in 1080-1302 units and worse in 1358-1580;
the top 1% of units (27) carry 52-63% of all positive gains and 45-60% of all losses;
removing those 27 units moves the pooled RMSE difference from +2e-5..+3.6e-4 to
+6.5e-4..+1.3e-3 (GCRK further behind). The 2021-12-11 event holds 22.6 of 29.1 units of
positive and 25.5 of 31.9 units of negative squared-error change (means over seeds; 78% and
80%).
`results/concentration_summary_main.csv`, `concentration_twosided_main.csv`,
`concentration_by_event_main.csv`, `concentration_units_main.csv`.

**H2 — a county's own geography is not better than a random county's.** Not supported.
Inside each fold's trained GCRK, every held-out unit was run with its own geography and
with 200 donor counties' geographies; among units whose forecast depends on geography
(1839-2489 of 2660 per seed), own geography is better than the median donor in
53.1-57.2% and the one-sided Wilcoxon test of percentile - 0.5 < 0 gives p = 2e-11,
9e-9, 3e-14, 3e-9 and 3e-4 for seeds 0-4. The size is small: mean RMSE of own minus
median donor = -5e-5, -4.7e-4, -1.9e-4, +6e-6 and -4.6e-5.
`results/swap_summary_main.csv`, `swap_units_main_seed*.csv`.

**H3 — single-seed county-level differences are dominated by initialisation.**
Supported. Mean Spearman correlation of per-unit D_u between the 10 seed pairs = 0.23
(< 0.3); among the 1330 units with above-median |D_u|, the sign agrees in >= 4 of 5 seeds
for 59%. `results/h3_main.json`, `seed_pair_correlation_main.csv`.

## 8. Figures

* Figure 1 (`figures/fig1_event_county_impacts_open.*`): observed data only.
* Figure 3 (`figures/fig3_gcrk_interpretation_open.*`): PREREG 11 chose the event
  2021-12-11 (28 two-wave counties, tie with 2022-06-08 broken by the wave-B/wave-A
  ratio) and the county Siskiyou, CA (06093; largest observed wave-B rise, 31.9 pp at
  hour 195; `results/figure_choice.json`). Seed 0, fold 4. Checks: the rebuilt kernel
  state matches the model's to 3.0e-7; row sums of panel (a) match the logit change to
  3.3e-4; the integrated-gradient attributions of (b) sum to the push (-11.72 vs -11.71).
  In this county the kernel lowers the damage logit in the push window (-11.7, mostly
  through snow/ice/freezing inputs, -18.9), but the damage rate is near zero then, so the
  forecasts with the exit open and closed differ by at most 0.07 pp; neither model
  predicts the observed 31.9% peak (GCRK 1.6-3.1% and W 2.1-3.0% at that hour across seeds).
* Figure 3S (`figures/fig3s_gcrk_largest_kernel_effect_posthoc_open.*`): **post hoc,
  chosen from model output** (largest |open - closed| in the push window among the
  event's units, seed 0; `results/figure3s_choice.json`): Oneida, WI (55085). Push +78.9
  (weather +78.9, geography -0.01; attribution sum error 1.5e-3). Seed 0 forecasts 46.5%
  at the observed wave-B peak (44.5%, hour 203) against W 14.8% and exit-closed 9.6%;
  across seeds GCRK gives 19.4-46.5% and W 14.2-31.2% at that hour
  (`results/manuscript_numbers_main.json`). Shown only to illustrate the mechanism.
* Figure 4 (`figures/fig4_county_trajectories_open.*`): eight counties by PREREG 11
  (`results/figure4_choice.json`); GCRK seed 0. All eight start at p_71 = 0. GCRK's peak
  hour lies within 5 h of the observed one in six counties (13 h late in Mason, MI; the
  wrong wave in Clinton, OH); peak magnitudes are underestimated by factors of 5-29 in the
  six largest events and overestimated in Newaygo and Mason, MI (34.0% vs 23.1%, 27.0% vs
  14.6%). TimesFM stays below 2% in all eight. Roscommon, MI reaches the cap p = 1 for
  two hours (4 of the 2660 county-events touch the cap; kept, as PREREG fixed the data).

## 9. Pathology checks (PREREG 9)

No cell selected 0 or 1600 steps; no kernel closed (|tanh(alpha)| >= 0.22); no nonfinite
value; no (seed, fold) OUTER RMSE above 1.04 times the median of its arm and fold
(`results/pathology_main.csv`). Calibration buffers: c = 2.95-3.60, theta = 1.29-1.37.

## 10. Data caveats

* ERA5 reanalysis is used over the whole window, a perfect-weather-forecast setting for
  every model compared; ERA5 gusts at 0.25 degrees do not resolve local convective gusts.
* The denominator is the publisher's modelled 2024 county customer count applied to
  2019-2024; 4 county-events reach p = 1 for a few hours. Sensitivity (not pre-registered):
  without them GCRK - W is +3.0e-4 on full-rollout RMSE, +3.5e-4 at 25-48 h and +3.7e-4 on
  MAE, higher in 5/5 seeds each (`results/sensitivity_no_cap_main.json`); those units
  favoured GCRK, so the main conclusion does not rest on them.
* The county set is each event's wind-report footprint; quiet counties outside it are
  not in the panel.
* Soil-forest co-location and a national windthrow interpretation could not be rebuilt
  from public data (PREREG 3); county-share products and root-limiting soils stand in.

## 11. Provenance of every number

| numbers | file | script (inputs) | seeds |
|---|---|---|---|
| main table, peaks, thresholds | `results/table_main_main.csv`, `results/tables_main.md`, `results/table_main_main.tex` | `evaluate.py main`, `make_tables.py main` (runs/.../main/*/outer.npz, runs/.../timesfm/*.npz, features.npz) | 0-4 |
| per-seed metrics, paired differences, verdicts | `results/seeds_main.csv`, `results/paired_main.csv` | `evaluate.py main` | 0-4 |
| kernel-exit decomposition | `results/decomposition_main.csv` | `evaluate.py main` (outer.npz: P, P_closed) | 0-4 |
| by event | `results/events_main.csv` | `evaluate.py main` | 0-4 |
| selected steps, kernel opening, calibration | `results/pathology_main.csv` | `evaluate.py main` (DONE.json) | 0-4 |
| concentration (H1), seed agreement (H3) | `results/concentration_*_main.csv`, `results/h3_main.json`, `results/seed_pair_correlation_main.csv` | `diagnostics.py concentration` | 0-4 |
| geography swap (H2) | `results/swap_summary_main.csv`, `results/swap_units_main_seed*.csv` | `diagnostics.py swap` (final.pt of every GCRK cell) | 0-4 |
| Figure 3 data and checks | `results/figure_data/fig3_*`, `results/figure_choice.json` | `diagnostics.py figure3` (seed 0 fold-4 GCRK and W final.pt) | 0 |
| Figure 3S data | `results/figure_data/fig3s_*`, `results/figure3s_choice.json` | `diagnostics.py figure3 --unit 1822 --tag s` | 0 |
| case-county numbers, segment cell counts | `results/manuscript_numbers_main.json` | inline script recorded in this file's history (outer.npz, features.npz) | 0-4 |
| panel descriptives | `results/panel_description.*`, `results/weather_twin_pair.json` | `describe_panel.py` (features.npz) | none |
| LOEO tables, decomposition, pathology, per cell | `results/*_loeo.*`, `results/loeo_cells.csv` | `evaluate.py loeo`, `make_tables.py loeo`, inline cell check (runs/.../loeo/*/outer.npz) | 0-4 |
| sensitivities (not pre-registered) | `results/sensitivity_no_cap_main.json`, `results/sensitivity_loeo_without_2024-02-27.json` | inline scripts (outer.npz, features.npz) | 0-4 |
