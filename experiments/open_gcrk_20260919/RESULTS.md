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
  25-48 h worse in 5/5 (+3.3%). MAE worse in 5/5 (+6.5%). **[B], negative.** Seed
  agreement measures initialisation only: resampling counties, the 95% interval of the
  RMSE change is -1.3% to +2.1% (event x state blocks: -1.5% to +2.8%), so over the
  sampled counties and storms GCRK and W are indistinguishable; one event carries 78% of
  the squared outage signal (effective number of events 1.6; section 12).
* GCRK is slightly better on peaks in 4 of 5 seeds: mean absolute peak-magnitude error
  -0.5%, peak-time error 19.3 h -> 18.3 h. It also predicts non-zero outage more often
  in hours that are observed at zero (false-activity share 0.31 -> 0.43, 5/5). The extra
  false activity stays when the kernel exit is closed (0.43); frozen replays place it in
  the gated damage term of the network trained with the kernel (section 11).
* The kernel's output is used: closing the exit of the trained GCRK network raises RMSE
  to 0.02731 (5/5 above both W and GCRK). The GCRK - W difference splits into a joint-
  training part (closed - W = +1.01e-3) and a kernel-output part (open - closed =
  -0.87e-3) that almost cancel. This is an arithmetic split of RMSE between trained
  configurations, not a causal attribution to parameter blocks.
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
  with its kernel exit closed is best (0.03003; below all-zero in 4/5 seeds); arithmetically
  the gain over W sits in the closed-exit network and the kernel's output adds error on
  average. Model selection here used county-grouped inner folds inside the training events
  (as pre-registered), i.e. it tuned t* for new counties of seen storms, not for new storms.
  All three trained forecasts are too large in amplitude under transfer (oracle scale
  factors 0.40 W, 0.54 GCRK, 0.68 closed); one common rescaling removes 83% of their RMSE
  spread, so the ordering mostly reflects how much each overshoots (section 12).
  One pre-registered outlier cell (held-out
  2024-02-27, seed 4) comes from the kernel output extrapolating to 37-70% outages in ten
  counties of the interior West and the Maine coast whose observed peaks were 0-21%.

## 1. What was run

* Data: 5 events chosen by the public-metadata rule (`event_selection.csv`), 2660
  county-events in 1756 counties and 45 states, 216 hours each, 99.94% of county-hours
  observed, where observed means a record or an inferred zero while the county's feed was in
  service (a record within 7 days; `src/asymode/panel.py`) (`panel_gates.csv`,
  `results/panel_description.json`).
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
removed (drop-path exposes it during training). With the exit closed the network trained
with the kernel is worse than W, and the kernel's output recovers most but not all of the
gap. The split describes trained configurations; it does not say which parameter block
causes the difference (section 11 locates where the errors arise).
The kernel opening tanh(alpha) is -0.83 to +0.88 across cells and its sign is constant
within a seed (seeds 1 and 2 negative, 0, 3 and 4 positive). The two signs are different
operators (with W2 shared by h and the state, one amplifies persistent departures and the
other damps them), and alpha has not converged: it starts at 0 and |alpha| / t* averages
2.2e-3 per step, about three quarters of the learning rate, with corr(|alpha|, t*) = 0.73.
Its size at t* reflects the training length more than a learned optimum (section 12).

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
  positive). Under event transfer the gain over W appears in the closed-exit network, and
  the kernel's output adds error on average (arithmetic split, not a causal attribution).
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
  The gust channel is the instantaneous gust at each hour ('instantaneous_10m_wind_gust'),
  not the hour's maximum ('10m_wind_gust_since_previous_post_processing'), and the gust
  exceedance terms are computed after the county area average, which can remove local
  exceedances ([mean gust - 15]_+^2 <= mean([gust - 15]_+^2)).
* The denominator is the publisher's modelled 2024 county customer count applied to
  2019-2024; 4 county-events reach p = 1 for a few hours. Sensitivity (not pre-registered):
  without them GCRK - W is +3.0e-4 on full-rollout RMSE, +3.5e-4 at 25-48 h and +3.7e-4 on
  MAE, higher in 5/5 seeds each (`results/sensitivity_no_cap_main.json`); those units
  favoured GCRK, so the main conclusion does not rest on them.
* The county set is each event's wind-report footprint; quiet counties outside it are
  not in the panel. Storm Events wind reports are partly damage reports, so the panel
  conditions on a damage proxy in the forecast window and leaves out exposed counties
  without reported damage, the contrast geography would most need.
* The pre-registered geography (31 descriptors) has no soil-forest co-location and uses
  root-limiting soils in place of a windthrow rating. PREREG 3 said SSURGO carries a
  windthrow-hazard interpretation for two states only; that was wrong: the national rule
  'FOR - Windthrow Hazard' is published for survey areas in every state (PREREG Amendment 1).
  Both are rebuilt in `build_geography_ext.py` for a separately labelled round; every result
  above uses the 31.

## 11. Frozen-checkpoint diagnostics (post hoc; no retraining)

Every REFIT checkpoint was replayed on its own OUTER units (`diagnostics_frozen.py`; the
replays reproduce the exported forecasts to < 1e-5). Means over seeds 0-4, main design
unless stated; files `results/frozen_{segments,pathways,reach,origin,inner}_{main,loeo}.csv`.

* Where the squared error is (share of total; W / GCRK / GCRK exit closed): observed rises
  39.5% / 39.5% / 41.4%, falls 27.2% / 27.7% / 27.5%, other active hours 30.7% / 30.3% /
  29.1%, hours observed at zero 2.6% / 2.6% / 2.0%. GCRK's extra squared error over W
  (+2.8) is in falls (+2.0) and rises (+1.2). In falls GCRK predicts lower p than W (mean
  0.0223 vs 0.0231, observed 0.0338), and its recovery rate is higher (mean 0.250 vs 0.235).
* False activity (p > 0.001 in hours observed at zero): W 0.309, GCRK 0.429, exit closed
  0.430. Rolling each model forward on its own rates with the background rate removed
  leaves 0.092 (W) and 0.205 (GCRK; 0.201 closed); with the background rate alone, 0.225
  and 0.227. The extra false activity comes from the gated damage term (occurrence gate x
  damage sigmoid) of the network trained with the kernel, not from the background rate,
  the recovery rate or the kernel's output. LOEO: 0.277 / 0.362 / 0.366, the same pattern.
* Observed rises above 1 pp in an hour (3929 hours): the damage rate each model supplies is
  a median 9% (W) and 12% (GCRK) of what the observed step needs. There the damage sigmoid
  averages 0.022 (W), 0.032 (GCRK) and 0.017 (exit closed), and the occurrence gate 0.70 (W)
  and 0.59 (GCRK); the ceiling b + 0.5 x gate is below the need in only 2.7% (W) and 5.0%
  (GCRK) of these hours. Peaks are missed because the damage logit stays low, not because
  of the gate's ceiling; the kernel nearly doubles the damage sigmoid on big rises, which is
  why closing the exit costs most on rises.
* Forecast-origin artifact: the path summaries accumulate from hour 72 (zero before) while
  the kernel's reference is the prefix mean, so the start of accumulation can look like a
  departure. With every OUTER unit's weather frozen at its hour-71 value, the kernel's
  effect on the damage logit (open - closed) averages -0.017 (mean absolute 0.028) with the
  path summaries as trained and -0.003 (0.006) when they accumulate from hour 0; its effect
  on p averages -2e-5 and exceeds 1 pp in 0.04% of units. The artifact is present in the
  trained kernels but far too small to explain the results; a new round should still
  accumulate from hour 0 for W and GCRK alike.
* Model selection: GCRK's best pooled INNER MSE is below W's in 10/25 cells (LOEO 11/25).

## 12. Checks raised by external review (post hoc; no retraining)

`review_checks.py`; files `results/review_*`. Means over seeds 0-4.

* Sampling uncertainty (cluster bootstrap of the seed-mean squared error of each unit,
  5000 draws). Main design, GCRK vs W: +0.52% RMSE, 95% interval -1.3% to +2.1% resampling
  counties (P(worse) 0.73), -1.5% to +2.8% resampling event x state blocks (0.66). Exit
  closed vs W: +3.8% (counties +1.9% to +5.6%; blocks +0.1% to +7.3%); GCRK vs exit closed:
  -3.2% (counties -5.1% to -1.3%; blocks -7.4% to +0.4%). LOEO, GCRK vs W: -2.5% (counties
  -3.9% to -1.2%; blocks -6.7% to +0.9%; five events -13.4% to +6.4%). The 2021-12-11 event
  holds 77.8% of the all-zero squared error; the other four 8.3%, 2.5%, 4.8% and 6.6%
  (effective number of events 1.61).
* Oracle rescaling (each forecast multiplied by the scalar that minimises its own squared
  error). Main: scale factors 0.96 (W), 1.00 (GCRK), 1.21 (exit closed); RMSE after
  rescaling 0.02628, 0.02642, 0.02720, the same order. LOEO: 0.40, 0.54, 0.68; RMSE after
  rescaling 0.02990, 0.02974, 0.02966 (before 0.03144, 0.03063, 0.03003), all below the
  all-zero 0.03049; per-event rescaling 0.02916, 0.02876, 0.02870. Under event transfer the
  models carry pattern information but overshoot in amplitude, and most of the LOEO ordering
  is the size of that overshoot.
* False activity by threshold (main; W / GCRK / exit closed): p > 0.001 0.309 / 0.429 /
  0.430; > 0.005 0.067 / 0.085 / 0.079; > 0.01 0.030 / 0.036 / 0.032; > 0.02 0.011 / 0.013
  / 0.012. The 0.001 threshold sits near the background-rate level, so the pre-registered
  share overstates the difference; the direction holds at every threshold.
* Kernel opening alpha: see section 4 (LOEO: |alpha| / t* 2.1e-3, corr 0.58; sign set by
  seed except in five cells).
* Trained memory lengths (main; median over cells of the per-county values): shortest
  coordinate 9.8 h, longest 24.5 h (ratio 2.9); no coordinate's steady-state gain nu /
  lambda_j is below 0.1, 32% are below 0.5; across counties the memory lengths vary by 9%
  (coefficient of variation). The dampings did not split into very fast and very slow
  coordinates, and geography changes them little.
* Kernel gate (main): share of unit-hours with the gate above 0.5 is 3.6% in the prefix
  (hours 1-71) and 18%, 24%, 22% and 31% at leads 1-6, 7-24, 25-48 and 49-144 h; the mean
  deposit norm rises from 0.10 in the prefix to 0.31 at 49-144 h. Holding one input group
  at its own prefix mean for all hours lowers the forecast-window deposit norm by 43%
  (wind), 31% (path summaries), 28% (slow thermal, snow and soil inputs), 13% (other) and
  10% (precipitation). The kernel mostly integrates a growing drift of the weather away from
  the prefix, not isolated shocks.
* E0, geographic information in W's held-out errors (ridge on the descriptors; 200
  permutations of the descriptors within event x state blocks as the null). Main design,
  county-grouped cross-validation: out-of-fold R2 0.002 for the mean residual (null 95th
  percentile 0.000, p = 0.005) and 0.031 for the log peak ratio (0.017, p = 0.005) with the
  31 descriptors; 0.001 and 0.034 with 40. LOEO, event-grouped: R2 -0.16 to -0.19, below
  the null (p 0.87-0.99). Within seen storms the descriptors carry a small, detectable
  signal about where W misses peaks; nothing of it carries across events, and the nine added
  descriptors add nothing.

## 13. E2: W refit at GCRK's t* (PREREG Amendment 2; diagnostic)

`e2_refit.py`, `e2_evaluate.py`; `results/e2_{main,loeo}.csv`. With the same initialisation and
full-batch steps this is W's own refit path stopped where GCRK's inner selection stopped (GCRK
selects fewer steps: median 420 vs 620 in LOEO, 290-420 vs 620 in main). Means over seeds 0-4.

| design | W, own t* | W at GCRK t* | GCRK exit closed | GCRK |
|---|---:|---:|---:|---:|
| main RMSE | 0.02630 | 0.02703 | 0.02731 | 0.02644 |
| main false activity (0.001) | 0.309 | 0.463 | 0.430 | 0.429 |
| main false activity without background | 0.092 | 0.239 | 0.201 | 0.205 |
| main mean recovery rate | 0.235 | 0.250 | 0.250 | 0.250 |
| LOEO RMSE | 0.03144 | 0.03069 | 0.03003 | 0.03063 |
| LOEO false activity (0.001) | 0.277 | 0.360 | 0.366 | 0.362 |

* Stopped at GCRK's t*, W reproduces the GCRK network's host: RMSE close to the closed exit,
  the same excess false activity from the gated damage term, the same faster recovery. The
  closed-exit gap of section 4 is mostly training length (W at GCRK t* - W = +7.3e-4 of the
  +1.01e-3), not training with the kernel; a small remainder (closed - W at GCRK t* = +2.8e-4,
  5/5 seeds) is.
* At matched training length the kernel helps in the main design: GCRK 0.02644 vs W at GCRK t*
  0.02703 (-2.2%, lower in 5/5 seeds). GCRK's own inner selection stops earlier than W's, and W
  trained to its own t* ends up better than GCRK.
* Under event transfer shorter training is itself the gain: W at GCRK t* (0.03069) matches GCRK
  (0.03063); the kernel adds nothing at matched length, and GCRK's advantage over W in LOEO is a
  training-length (shrinkage) effect.

## 14. R2: better inputs and an event-held-out selection (PREREG Amendment 2; seed 0 screen)

`build_panel216_r2.py`, `build_features_r2.py`, `r2_screen_evaluate.py`; `results/r2/screen*.csv`,
`screen_decision.json`. Round 2 changes both arms together: the hour's maximum ERA5 gust instead of
the instantaneous gust; gust exceedance energy, wet wind, near-freeze, snow-ice load and cold
precipitation computed per ERA5 cell and then area-averaged; trailing 72-hour path summaries in
place of summaries that start at the forecast origin; the county's highest cell gust and its
gust > 15 m/s area share as two more damage inputs; 40 geographic descriptors; and, in the
leave-one-event-out design only, inner folds that leave one development event out. One seed.

| design | round | W | GCRK | GCRK exit closed | W MAE | W false activity |
|---|---|---:|---:|---:|---:|---:|
| main (inputs only) | 1 | 0.02675 | 0.02680 | 0.02766 | 0.00579 | 0.347 |
| main (inputs only) | 2 | 0.02597 | 0.02644 | 0.02788 | 0.00536 | 0.267 |
| LOEO (inputs + selection) | 1 | 0.03096 | 0.03023 | 0.02977 | 0.00682 | 0.313 |
| LOEO (inputs + selection) | 2 | 0.02971 | 0.02973 | 0.02973 | 0.00739 | 0.749 |

* Inputs alone (main design, where selection is unchanged) lower the host's RMSE by 2.9%
  (0.02675 -> 0.02597, against a seed-to-seed standard deviation of 0.00036), every lead segment,
  MAE (0.00579 -> 0.00536) and false activity (0.347 -> 0.267). The pre-registered continuation
  rule passes in both designs (improvement 7.8e-4 main, 1.3e-3 LOEO).
* GCRK gains less from the better inputs than W does (0.02680 -> 0.02644), so at this seed the
  gap to W widens from +0.2% to +1.8%.
* With event-held-out inner folds the selected training length collapses to 50-110 steps (median
  70, against 330 for W in round 1). The kernel's warm-up is 200 steps, so its opening is still
  ramping and W, GCRK and the closed exit coincide at 0.0297. RMSE improves, MAE and false
  activity get worse (0.31 -> 0.75): under event transfer the safest model is a barely trained one,
  and this design cannot test the kernel.
* The twelve-event round therefore uses the round-2 inputs (rule of Amendment 2) and reports the
  kernel's realised opening in every event-design cell.

## 15. E3: twelve events with the round-2 inputs (PREREG Amendment 3; seed 0 of five)

Twelve events (Amendment 2 with its note), 6,122 county-events, 2,409 counties, effective number
of events 4.97 (five-event set: 1.61); round-2 inputs; GCRK reads the 40 descriptors. Two designs:
county-grouped (five outer folds, three county-grouped inner folds) and event-grouped (four outer
folds of three events, three inner folds of three development events). **One seed so far** (the PI
stopped the round after seed 0), so the Amendment 3 verdict rule, which needs five, does not apply.
`evaluate_e3.py`, `timesfm_baseline.py` (twelve-event panels); `results/e3r2/e3_*`.

| design | all-zero | persistence | TimesFM | W | GCRK | GCRK exit closed |
|---|---:|---:|---:|---:|---:|---:|
| county-grouped | 0.02868 | 0.03079 | 0.02890 | 0.02574 | 0.02587 | 0.02649 |
| event-grouped | 0.02868 | 0.03079 | 0.02890 | 0.02800 | 0.02818 | 0.02792 |

| design | contrast | change | 95% counties | 95% event x state | 95% events |
|---|---|---:|---|---|---|
| county-grouped | GCRK vs W | +0.49% | -1.18% to +2.14% | -1.48% to +2.15% | -1.55% to +2.55% |
| county-grouped | closed vs W | +2.93% | +1.38% to +4.40% | -0.01% to +6.40% | -0.21% to +4.55% |
| event-grouped | GCRK vs W | +0.66% | +0.10% to +1.36% | -0.39% to +2.44% | -0.51% to +3.16% |
| event-grouped | closed vs W | -0.27% | -0.41% to -0.13% | -0.71% to +0.28% | -0.81% to +0.39% |

* Zero-shot TimesFM does not beat the all-zero forecast here either (0.02890 with ERA5
  covariates, 0.02901 from history alone).
* County-grouped: GCRK and W are not distinguishable at this seed (+0.5%, the county interval
  covers zero); the kernel's output still pays for joint training (closed is 2.9% worse than W,
  interval above zero). Both beat the all-zero forecast by 10% and persistence by 16%. The kernel
  is fully open in all five folds (tanh(alpha) 0.63-0.83). W again trains about twice as long as
  GCRK (t* 780-1050 against 410-610), so the E2 caveat applies here too: the comparison mixes the
  kernel with the training length each arm's own selection picks.
* Event-grouped: GCRK is 0.7% worse than W; only the county-clustered interval excludes zero, and
  the county clustering is the weakest of the three for a design held out by event. Both trained
  models beat the all-zero forecast pooled (0.0280 and 0.0282 against 0.0287) -- unlike the
  five-event leave-one-event-out round, where they did not -- and the overshoot is much smaller
  (oracle scale 0.80 against 0.40).
* By fold (event-grouped; the kernel's realised opening is reported because event-held-out
  selection can stop inside its warm-up):

  | fold | events | tanh(alpha) | t* | W | GCRK | closed | all-zero |
  |---|---|---:|---:|---:|---:|---:|---:|
  | 1 | 2019-02-24, 2021-08-11, 2022-06-17 | +0.83 | 640 | 0.03107 | 0.03166 | 0.03106 | 0.03229 |
  | 2 | 2019-03-13, 2021-12-11, 2024-02-27 | +0.88 | 680 | 0.03501 | 0.03502 | 0.03480 | 0.03640 |
  | 3 | 2019-11-27, 2022-04-13, 2024-05-08 | +0.07 | 60 | 0.02387 | 0.02387 | 0.02387 | 0.02326 |
  | 4 | 2021-03-26, 2022-06-08, 2024-06-26 | +0.17 | 170 | 0.01512 | 0.01515 | 0.01509 | 0.01452 |

  In the two folds whose held-out events carry the large outages the trained models beat all-zero
  by about 4% and the kernel adds error (+1.9%) or nothing (+0.02%); in the two quieter folds
  selection stops inside the warm-up, the three arms coincide, and all of them are worse than
  all-zero.

## 16. Diagnostics of a second external review (PREREG Amendment 4; twelve events, seed 0; no retraining)

`review2_checks.py` with OPEN_GCRK_ROUND=e3r2; `results/e3r2/review2_*.csv`.

* D1, calibration. Pooled over held-out county-hours, W's forecast is a calibrated conditional mean:
  in bins of predicted p the observed mean follows the prediction (0.0055 -> 0.0058, 0.0170 ->
  0.0165, 0.0493 -> 0.0503, 0.149 -> 0.125), while the outcomes inside a bin are zero-inflated (44%,
  28%, 12% and 5% of the cells are exactly zero) with 5-9% of cells above three times the
  prediction. Missing peaks and false activity are therefore two faces of a mean forecast over
  cases the inputs do not separate. By unit, the predicted peak ranks well (share of units with an
  observed peak above 10%: 3.6% in the lowest decile of predicted peak, 50% in the highest) but is
  below the observed peak in every decile (0.116 vs 0.157 at the top): the peak of a mean path is
  below the mean of the realised peaks.
* D2, prefix. Across the twelve events the relative change GCRK vs W rises with the
  prefix-to-forecast wind-report ratio (Spearman +0.66, p = 0.02): -0.9% on average in the seven
  events with a calm prefix (-5% in the two large synoptic wind events of 2019), +2.1% in the five
  without one. Those five are mostly summer convective events, where the kernel is hardly active
  (gate open in 3-15% of forecast hours against 27-46%), so event type and prefix are confounded;
  the kernel's prefix reference was designed for a calm prefix, which the twelve-event rule dropped.
* D3, information ceiling (gradient-boosted regressor on unit-level summaries, fixed settings,
  same outer folds). County-grouped: the forecast-window mean outage has out-of-fold R2 0.29
  without geography and 0.31 with the 40 descriptors (RMSE -1.1%, county interval -2.5% to +0.3%);
  the peak 0.31 and 0.31 (+0.1%, -0.8% to +1.2%, inside the permutation null). Event-grouped:
  R2 0.04-0.19 only, with a small gain from geography (-0.8% to -2.0%, above the null in three of
  four targets). So geography carries little conditional information at this resolution wherever it
  enters; a test of where to put it is unlikely to separate the options.
* The host leaves unit-level information unused. W's own forecasts imply R2 0.15 for the window
  mean and 0.09 for the peak, against 0.29 and 0.31 for the regressor (0.23 for both with weather
  summaries alone). The most useful non-weather group is county context (customers, rural-urban
  code, density, cooperative share, utilities, SAIDI): without it R2 falls to 0.26 and 0.25. In W
  that context reaches the recovery network only; the damage network reads weather alone.
* Carried to the primary metric (diagnostic, not a model): multiplying each unit's W path by the
  ratio of the regressor's window mean to W's own lowers the hourly RMSE from 0.02574 to 0.02546
  with weather summaries, 0.02501 (-2.8%) with context and prefix outage added, and 0.02483 (-3.5%)
  with the descriptors as well; leads of 25-144 h improve and leads of 1-24 h get worse, because a
  uniform scale also moves the hours anchored by p_71. For comparison, GCRK - W is +0.5%.

## 17. Level versus memory: W+C and W+G (PREREG Amendment 5; twelve events, county-grouped, seed 0)

Two control arms add one zero-initialised linear term to the damage logit, constant over the
window: W+C reads the six county context variables (7 parameters), W+G the 40 geographic
descriptors (41). Both are exactly W at step 0 and share W's initialisation at this seed; the
kernel and the protocol are untouched. Ten cells, no failure. `results/e3r2/level_arms_*.csv`.

| model | RMSE | MAE | 25-48 h | 49-144 h | event-equal | false activity | vs W |
|---|---:|---:|---:|---:|---:|---:|---:|
| W | 0.02574 | 0.00612 | 0.03321 | 0.02322 | 0.02268 | 0.377 | - |
| **W+C** | **0.02553** | **0.00587** | 0.03266 | 0.02306 | 0.02269 | **0.347** | **-0.82%** |
| W+G | 0.02570 | 0.00617 | 0.03285 | 0.02317 | 0.02268 | 0.421 | -0.15% |
| GCRK | 0.02587 | 0.00647 | 0.03266 | 0.02351 | 0.02275 | 0.507 | +0.49% |
| GCRK exit closed | 0.02649 | 0.00626 | 0.03373 | 0.02405 | 0.02299 | 0.509 | +2.93% |

| contrast | change | 95% counties | 95% event x state | 95% events | folds better |
|---|---:|---|---|---|---:|
| W+C vs W | -0.82% | -2.49% to +0.79% | -3.28% to +1.30% | -2.53% to +1.94% | 3/5 |
| W+G vs W | -0.15% | -1.69% to +1.42% | -2.15% to +1.84% | -1.70% to +1.83% | 3/5 |
| GCRK vs W | +0.49% | -1.15% to +2.11% | -1.47% to +2.16% | -1.51% to +2.49% | 2/5 |

* By the rule fixed in Amendment 5 this is the third branch: **no arm's interval excludes zero**,
  so a constant level shift does not reach the unit-level magnitude information the unconstrained
  regressor found (Amendment 4: -2.8% when W's path is rescaled per unit). The ordering of the
  point estimates is the one the level-versus-memory reading predicts (W+C best, W+G near W, the
  memory route worst), and W+C is the only arm that improves MAE (-4.1%) and false activity
  (0.377 -> 0.347) at the same time, but one seed cannot turn that into a result.
* The fitted term is stable and physically readable: all six coefficients keep their sign in all
  five folds. More customers (-0.61), denser population (-0.38) and more utilities (-0.40) lower the
  damage logit; a higher rural-urban code (+0.28) and a worse reliability history (SAIDI, +0.30)
  raise it. The arm is learning exposure and service structure, not weather.
* W+C also trains about as long as W (t* 430-850 against 780-1050), unlike GCRK (410-610), so this
  comparison is not confounded by training length the way the kernel comparison is (section 13).

## 18. Why the kernel does not help on these data (PREREG Amendment 6 with its note; seed 0)

Scripts `review3_*.py`, `evaluate_controls.py`, `make_planted_*.py`, `evaluate_planted.py`; files
`results/review3_inner_curves.csv`, `results/r2/controls_*.csv`, `results/e3r2/review3_*`,
`results/planted_*.{json,csv}`. Four findings, ordered from the most basic.

**1. The geography is already in the weather the host reads.** A regressor given only the prefix and
forecast-window means of the 14 ERA5 channels predicts the descriptors of counties it has never seen:
mean elevation R2 0.997 (0.994 from surface pressure alone), canopy 0.84, forest share 0.79, slope
0.86, relief 0.83, wet-soil share 0.76, hydric share 0.72; median over the 40 descriptors 0.72, 29
above 0.5; only the eight aspect shares (0.14-0.29) and developed land (0.40) are not encoded.
Conditioning on geography therefore gives the model little it does not have, which is what the
unconstrained regressor of section 16 measured (-1.1%, interval across zero).

**2. What weather leaves unexplained is not a property of the county.** In the out-of-fold residuals
of the forecast-window mean outage (weather, neighbours and prefix outage as inputs), the repeatable
county effect over the 1,653 counties seen in two or more events has an intraclass correlation of
0.06 on the scale of the loss (0.17 on the log scale): about 94% of it is specific to the county in
that event. No static county descriptor, of any kind and through any interface, can explain more than
that share. Consistent with it: geography does not predict the timing of the response (lag from gust
peak to outage peak: R2 0.39 without, 0.37 with geography; hours above half peak 0.24 and 0.24; only
the mean-to-peak ratio moves, -1.2%, interval touching zero); two coordinates (latitude, longitude)
help the pooled regressor more than the 40 descriptors (-1.8% against -1.1%); and the descriptors are
45% state (median R2 on state alone; elevation and canopy 70-80%). A guess that descriptors work as a
proxy of one storm's footprint inside a single event was tested and is wrong here (+2.0% on average,
better in 2 of 12 events).

**3. The conditioning identifies counties instead of transferring geography.** Round-1 traces: once
the kernel is open, GCRK's training loss is 10.6% below W's at step 400 (24 of 25 cell pairs) while
its held-out inner loss is 1.5% above. Controls on the five-event panel (round-2 inputs):

| arm | RMSE | vs W | median t* | training / inner loss vs W at step 400 |
|---|---:|---:|---:|---|
| W | 0.02597 | - | 650 | - |
| GCRK | 0.02644 | +1.8% | 360 | -26% / +1.4% |
| GCRK-S, shared kernel (no geography) | **0.02570** | -1.1% | 510 | -15% / -0.6% |
| GCRK-P, descriptors permuted across counties | 0.02658 | +2.3% | 510 | -30% / +6.2% |

GCRK against GCRK-S: +2.9% (county interval +0.8% to +4.9%); GCRK against GCRK-P: -0.5% (-2.2% to
+1.1%). Vectors with no geographic meaning lower the training loss as much as true geography and
generalise as badly; the same kernel without geography is the best arm (interval against W across
zero) and does not stop early.

**4. At this noise and sample size a real geographic effect of 7% could not be learned either.**
Semi-synthetic worlds on the real inputs and folds (truth: a trained GCRK network; log-normal unit and
event-by-state heterogeneity calibrated to the real data; folds 1-3):

| world (planted gap) | W | GCRK-S | GCRK | W+G | share of the planted effect recovered |
|---|---:|---:|---:|---:|---|
| TB, geography-conditioned memory (7.3%) | 0.01006 | 0.01014 | 0.01020 | 0.01028 | GCRK vs GCRK-S +0.01 |
| TA, geography level term (7.2%) | 0.01356 | 0.01358 | 0.01348 | 0.01365 | GCRK vs GCRK-S +0.21; W+G vs W +0.06 |
| TBc, TB without noise | 0.00085 | 0.00079 | 0.00075 | - | see below |

In TB the learned county memory lengths barely vary where training stops early (spread 0.02 and 0.13 h
against 2.9 h in the truth) and, where it runs longer, vary against the truth (Spearman -0.10): what
grows is noise. Without noise (TBc) the order is GCRK < GCRK-S < W and all arms are still improving
at the 1,600-step cap (kernel opening 0.35), so the optimisation path exists but is slow: the
conditioning maps start at zero and an Adam step of 0.003 needs more than a thousand consistent steps
to reach the planted weights, while selection stops at 400-800. TBc also shows finding 1 from another
side: W, which has no geographic input, reaches 0.00085 where the best geography-blind predictor of
the truth's own form has 0.00395, because the weather channels reveal the geography.

Reading. The failure is over-determined: the information the kernel is conditioned on is largely
redundant with its host's inputs (1); the variance left over is not a county property (2); the
conditioning's capacity is spent on identifying counties, which costs about 3% against the same
kernel without geography (3); and effects several times larger than any this panel could contain are
not learnable at its noise level (4), while the evaluation cannot detect differences below about 2%
(section 12). What helps on these data is what changes the information or removes noise: better
weather inputs (-2.9%, section 14), a shared response kernel (-1.1%, not yet distinguishable), county
context on the damage level (-0.8%, not distinguishable, section 17).

## 19. A conditioning that cannot single out a county: GCRK-K8 (PREREG Amendment 8; five events, seed 0)

The unchanged kernel reads the one-hot indicator of the county's geographic regime instead of its 40
descriptors (k-means, K = 8, on the standardised descriptors of all panel counties; regimes of 86 to
413 counties: open plains, forested mountains, wet lowlands, arid high relief, forested hills, ...).
`results/r2/controls_*.csv`.

| arm | RMSE | vs W | median t* | training loss vs W at step 400 |
|---|---:|---:|---:|---:|
| W | 0.02597 | - | 650 | - |
| GCRK-S, shared kernel | **0.02570** | -1.1% | 510 | -15% |
| GCRK-K8, eight regimes | 0.02638 | +1.6% | 360 | -19% |
| GCRK, 40 descriptors | 0.02644 | +1.8% | 360 | -26% |
| GCRK-P, permuted descriptors | 0.02658 | +2.3% | 510 | -30% |

GCRK-K8 against GCRK-S: +2.7% (county interval +0.9% to +4.6%); against GCRK: -0.2% (-1.4% to
+1.0%). By the rule fixed in advance this is the third branch: even conditioning that cannot identify a
county overfits here. It fits the training set better than the shared kernel, its inner loss turns up
at the same early step as the full conditioning, and the whole model is then refit for 360 steps where
the shared kernel gets 510 and W 650 (section 13 measured what that costs the host). All three forms of
conditioning - true, permuted and regime-level geography - are 2.7-2.9% worse than the same kernel
without geography, each with an interval that excludes zero.

## 20. Provenance of every number

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
| frozen-checkpoint diagnostics (section 11) | `results/frozen_{segments,pathways,reach,origin,inner}_{main,loeo}.csv` | `diagnostics_frozen.py` (final.pt of every cell, features.npz) | 0-4 |
| review checks (section 12) | `results/review_*.csv`, `results/review_effective_events_*.json` | `review_checks.py` (outer.npz, final.pt, features.npz, features_geo40.npz) | 0-4 |
| nine further descriptors (PREREG Amendment 1) | `data/interim/open_gcrk/geography_ext.parquet`, `features_geo40.npz`; `data_provenance/geography_ext_{log.jsonl,meta.json}`, `features_geo40_checksum.json` | `build_geography_ext.py`, `build_features.py --geo-ext` | none |
| E2 (section 13) | `results/e2_{main,loeo}.csv` | `e2_refit.py`, `e2_evaluate.py` (runs/.../e2/, final.pt) | 0-4 |
| R2 screen (section 14) | `results/r2/screen*.csv`, `screen_decision.json` | `r2_screen_evaluate.py` (runs/.../r2/) | 0 |
| E3 seed 0 (section 15) | `results/e3r2/e3_*` | `evaluate_e3.py` with OPEN_GCRK_ROUND=e3r2 (runs/.../e3r2/) | 0 |
| second-review diagnostics (section 16) | `results/e3r2/review2_*.csv` | `review2_checks.py` and the two inline ablation / rescaling scripts recorded in the commit message (features_e3r2.npz, runs/.../e3r2/) | 0 |
| level arms (section 17) | `results/e3r2/level_arms_main.csv`, `level_arms_bootstrap.csv` | inline script recorded in the Amendment 5 commit (runs/.../e3r2/main/seed0/W+*) | 0 |
| why the kernel does not help (section 18) | `results/review3_inner_curves.csv`, `results/r2/controls_*.csv`, `results/e3r2/review3_{timing,footprint,redundancy}.csv`, `review3_ceiling.json`, `results/planted_{worlds.json,recovery.csv,maps_TB.csv}` | `review3_*.py`, `evaluate_controls.py`, `make_planted_worlds.py`, `make_planted_clean.py`, `evaluate_planted.py` | 0 (round-1 traces: 0-4) |
| regime-conditioned kernel (section 19) | `results/r2/controls_*.csv` | `evaluate_controls.py` with OPEN_GCRK_ROUND=r2 (runs/.../r2/main/seed0/*/GCRK-K8) | 0 |
