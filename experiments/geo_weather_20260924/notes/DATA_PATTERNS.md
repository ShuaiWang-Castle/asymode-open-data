# Data patterns of the twelve-event public panel, for model design

Status 2026-09-25. Exploratory description; nothing is trained. Script: `analyze_patterns.py` (one process,
under 2 minutes, under 0.7 GB). Every number below is in `results/data_patterns/patterns.json`, under the key
(q1 to q6) of the question it answers.

**Panel.** 6,122 county-events ("units") from 12 events, 2,409 counties, 46 states; forecast hours 72-215 of
each 216-h window, 880,884 observed county-hours (0.08% unobserved). Target: EAGLE-I customers out over EAGLE-I
2024 modelled customers, clipped to [0, 1]. **Base:** W+Cin (host with six county-context inputs;
`experiments/open_gcrk_20260919/RESULTS.md` section 21), out-of-fold predictions of the county-grouped five-fold
design, seed 0: pooled RMSE 0.02513 (all-zero forecast 0.02868, MSE skill 0.23). Residual r = y - P (r > 0:
under-prediction).

**Conventions.** Peak t* = first hour of the observed maximum over the whole window; *active* = peak >= 1%.
Seasons: May-Oct (2021-08-11, 2022-06-08, 2022-06-17, 2024-05-08, 2024-06-26; leaf-on in `build_eih.py`) and
Nov-Apr (the other seven). Statistics are on ranks; partial correlations (rho_p) use event fixed effects; 95%
intervals are county-cluster bootstraps of the residualised products (not refitted, so approximate). Many
features were screened: read |rho| < 0.05 as noise and section 2 as hypothesis-generating.

## 1. Where the base's squared error sits

- **Under-prediction:** 87.8% of the SSE is on cells with y > P.
- **A few county-events:** the top 1% of units (61) carry 46.6% of the SSE, the top 5% (306) 77.4%; 73 units
  carry half. The target is as concentrated (top 1%: 46.9% of the sum of y^2).
- **Peak level:** the 455 units (7.4%) with a forecast-window peak above 20% carry 78.8%, the 61 above 50% carry
  36.5%, the 2,212 below 1% carry 2.1%.
- **County size:** the smallest customer quintile (521-7,437 customers) carries 40.6%, the largest (>= 77,102)
  9.5%. The top-1% units have a median of 7,109 customers and a median peak of 57%.
- **Magnitude more than timing:** over 3,910 active forecast windows the predicted peak is a median 0.45 of the
  observed (0.73 for 1-5% peaks, 0.31 for 5-20%, 0.18 above 20%); median peak-time error +1 h (absolute 6 h).
- **Around the peak:** 54.7% of the SSE lies within +/-6 h of the observed peak (7.4% of cells), 83.1% within
  +/-24 h (26.2%), 11.2% in the peak hour (0.6%). Rise (before t*): 22.7% of the SSE on 31.6% of cells; decay:
  66.0% on 64.9%; 84% and 87% of each is under-prediction. With t* taken inside the forecast window: 23.4% and 65.0%.
- **Level, zeros:** each unit's window-mean error is 22.9% of the SSE; the rest is shape and timing. 57.4% of
  observed cells are exactly zero; they hold 2.7% of the SSE, and the base predicts > 0.001 on 32.5% of them.
- **Where and when:** 7 of 46 states and 11 of 411 event x state blocks carry half the SSE; Michigan 17.5% with
  2.9% of the units, 2021-08-11 x MI 9.9% with 37 units. By lead: 1-24 h 18.8%, 25-48 h 26.9%, 49-72 h 14.0%,
  73-96 h 6.8%, 97-120 h 8.5%, 121-144 h 25.0% (mostly a second storm, section 6).

**Table 1.** Error by event (active = peak >= 1%).

| event | season | units (active) | share of SSE | RMSE | MSE skill vs zero | top-5 units' share of event SSE | event SSE after hour 168 |
|---|---|---|---|---|---|---|---|
| 2019-02-24 | Nov-Apr | 564 (422) | 9.8% | 0.0260 | 0.38 | 39% | 1% |
| 2019-03-13 | Nov-Apr | 321 (230) | 5.0% | 0.0246 | 0.06 | 54% | 16% |
| 2019-11-27 | Nov-Apr | 464 (249) | 1.6% | 0.0117 | -0.06 | 48% | 33% |
| 2021-03-26 | Nov-Apr | 544 (293) | 1.6% | 0.0106 | -0.01 | 32% | 8% |
| 2021-08-11 | May-Oct | 477 (359) | 12.8% | 0.0322 | 0.18 | 45% | 10% |
| 2021-12-11 | Nov-Apr | 991 (724) | 33.5% | 0.0362 | 0.33 | 26% | 71% |
| 2022-04-13 | Nov-Apr | 432 (258) | 3.1% | 0.0167 | 0.11 | 29% | 24% |
| 2022-06-08 | May-Oct | 385 (233) | 3.0% | 0.0174 | 0.02 | 55% | 43% |
| 2022-06-17 | May-Oct | 604 (459) | 10.3% | 0.0257 | 0.20 | 32% | 15% |
| 2024-02-27 | Nov-Apr | 419 (196) | 3.6% | 0.0183 | 0.14 | 63% | 30% |
| 2024-05-08 | May-Oct | 460 (362) | 12.4% | 0.0323 | 0.11 | 39% | 14% |
| 2024-06-26 | May-Oct | 461 (324) | 3.1% | 0.0161 | -0.06 | 31% | 14% |

## 2. What the residual wants from the exposure-integrated hazards

Population variant, tau = 0 and 12 h (80 features). Unit targets: window-mean residual (against each feature's
window mean) and peak residual max y - max P (against its window max). Controls: six summaries of the host's own
area-weighted gust channels (window max and mean, max cell gust, mean share of cells above 15 m/s, max 72-h
excess energy, prefix max) plus event fixed effects.

- **Pooled rankings mostly reflect event contrasts and the host's gust.** Gust ramps top the pooled Spearman
  with the window-mean residual (-0.19 to -0.20), fall to -0.11 to -0.13 within event, and to |rho_p| <= 0.027
  after the host gust (<= 0.034 for the peak residual). Leaf-on canopy features top it for the peak residual
  (+0.15) but are +0.01 within event. The gust ramps add nothing to the host's own gust.
- **Window-mean residual: rain and convection, negative.** 25 features reach |rho_p| >= 0.05: 16 rain or
  convective (all negative; strongest convective*one@12, -0.149 [-0.174, -0.123], partial R^2 0.022, same sign in
  11 of 12 events), two gust x wet (negative), seven leaf-on-modulated (+0.05 to +0.07). At equal wind the base
  over-predicts in rainy, convective counties, but controlling its own window-mean prediction and log customers
  removes this (+0.010): host calibration, not missing information.
- **Peak residual: canopy-modulated wind, positive.** Only five features reach |rho_p| >= 0.05, all gust x canopy;
  gust_x10*canopy@0 gives +0.091 [0.063, 0.119] (partial R^2 0.008, 11 of 12 events), +0.141 with prediction and
  customers controlled, +0.055 on active units. Its area-weighted twin gives +0.092 and the population feature
  adds +0.012 beyond it: canopy carries the signal, not exposure weighting (population- and area-weighted canopy
  correlate at 0.97).
- **Small on the squared-error scale:** no feature adds more than 0.53% to the within-event R^2 of either residual.
- **Modulators** (beyond the same hazard with modulator one). *Canopy* is positive for the peak residual with gust,
  rain and convective hazards (+0.091, +0.087, +0.086), stronger in Nov-Apr than May-Oct (gust_x10 +0.116 vs
  +0.057), near zero for the window mean. *Leaf-on* cannot be separated from canopy (within an event it equals
  canopy or zero), and the canopy effect is not larger in the leaf-on season. *Wet* (poorly drained share) is
  negative for the window mean (gust_x10*wet@0 -0.056 [-0.085, -0.027]) and near zero for the peak: not the
  wet-soil uprooting sign; the share probably marks flat, low-canopy farmland.
- **Physical check on the observed peak.** At equal host wind, the observed peak rises with population-weighted
  canopy: rho_p +0.297 [0.268, 0.324] (Nov-Apr +0.359, May-Oct +0.200); +0.239 [0.210, 0.266] with the base's six
  context inputs also controlled. Within event x gust tercile, the top canopy tercile's median peak is 3.5 times
  the bottom's (86% of 36 cells above 1). The base (no canopy input) leaves +0.101 [0.072, 0.127] of it in the
  peak residual. The association is similar in each gust tercile (+0.266, +0.294, +0.267 at median gust maxima of
  17, 21 and 24 m/s): within the range of these storms canopy acts like a multiplicative susceptibility, with no
  sign of a wind threshold. So yes: more canopy, more damage at equal wind.
- **Hour level** (1,500 sampled units, 215,880 cells; unit and lead fixed effects; seven hourly host gust
  controls): the strongest within-unit rho_p are 12-h rain and convective memories (convective*one@12 -0.181
  [-0.201, -0.161], unit-cluster interval): those hours are over-predicted relative to the unit's other hours.
  Largest R^2 gain: 0.21%.

## 3. Timing

- **The gust clock works only in the cold season.** The outage peak is within +/-6 h of the county's gust peak
  for 56% of active units in Nov-Apr (median lag +2 h, IQR -2 to +7) and 18% in May-Oct (+4 h, IQR -22 to +55);
  median county gust maxima 23.5 and 14.8 m/s.
- **Other clocks** (peak and forcing inside the forecast window, 3,661 units), within +/-6 h in Nov-Apr / May-Oct:
  gust 62% / 23%, convective proxy (CAPE x precipitation) 36% / 41%, precipitation 34% / 38%; maximum-cell gust
  does about as well as area-mean gust. Rain and convection time warm-season peaks better than gust, but none does
  it well; in 2019-02-24 they almost never coincide with the outage peak (5%, 6%).
- **Fast onset:** the largest gust of the preceding 24 h comes a median 3 h (Nov-Apr) or 7 h (May-Oct) before the
  peak; the rise from 10% of the peak to the peak takes a median 2 h (IQR 2-4).
- **Secondary waves** (3-h running median; another peak >= 25% of the main one, with that prominence, >= 12 h
  apart): 31% of active units (17% at 50%); 39% in May-Oct, 26% in Nov-Apr, 53% in 2021-08-11. Half come after the
  main peak, a median 60 h apart; 63% follow a gust >= 0.8 of the main peak's (a second storm), 21% one < 0.6
  (possibly a restoration setback, a sub-grid cell or a report artefact).
- **Antecedent rain.** At equal host wind the observed peak rises with rain in the 48 h before the gust peak
  (rho_p +0.124 [0.096, 0.151]; Nov-Apr +0.163, May-Oct +0.060) and with soil moisture at the gust peak (+0.145).
  The 48-h rain correlates with soil moisture (0.54) and storm-time rain (0.43); given both it keeps +0.033
  [0.007, 0.059]. Within event x gust tercile the top rain tercile's median peak is 2.3 times the bottom's (72% of
  36 cells above 1). The base's peak residual shows none of it (+0.003 [-0.022, 0.029]): the host's rain and
  soil-moisture channels already carry it.

**Table 2.** Timing, waves, recovery and flags by event (lag over the whole window; +/-6 h shares inside the
forecast window).

| event | median county gust max (m/s) | outage-peak lag, median (h) | peak within +/-6 h of gust / convective | multi-wave share | time to 10% of peak (h, peaks >= 5%) | flagged hours |
|---|---|---|---|---|---|---|
| 2019-02-24 | 23.9 | +2 | 61% / 6% | 32% | 23 | 0.32% |
| 2019-03-13 | 26.3 | +3 | 65% / 49% | 24% | 5 | 0.62% |
| 2019-11-27 | 22.0 | +1 | 64% / 15% | 28% | 5 | 0.11% |
| 2021-03-26 | 21.6 | +1 | 57% / 38% | 24% | 6 | 0.54% |
| 2021-08-11 | 13.3 | +12 | 23% / 25% | 53% | 11 | 0.15% |
| 2021-12-11 | 25.6 | +2 | 70% / 52% | 22% | 10 | 0.36% |
| 2022-04-13 | 20.4 | +5 | 38% / 35% | 30% | 7 | 0.36% |
| 2022-06-08 | 14.5 | 0 | 30% / 45% | 33% | 5 | 0.12% |
| 2022-06-17 | 14.8 | +2 | 32% / 60% | 32% | 12 | 0.24% |
| 2024-02-27 | 21.7 | +2 | 58% / 42% | 25% | 8 | 0.11% |
| 2024-05-08 | 15.7 | +4.5 | 13% / 46% | 34% | 10 | 1.84% |
| 2024-06-26 | 15.3 | +9 | 17% / 24% | 42% | 8 | 0.14% |

## 4. EAGLE-I artefact flags (`build_train_mask.py`)

- 3,531 flagged forecast hours (0.40% of observed) in 1,014 units: 2,307 plateau, 944 spike, 257 dip, 23 over
  (fraction >= 0.999).
- They hold 5.5% of the base's SSE, 13.7 times the per-cell average: over 3.1% from 23 hours (the five units with
  such hours hold 7.5% in total), plateau 1.7%, spike 0.7%, dip 0.04%; 99% of it is under-prediction.
- 2024-05-08 has 34.5% of the flags (1.84% of its hours; at most 0.62% in any other event). Five states (KS, WV,
  SD, PA, MD) have half; the block 2024-05-08 x WV alone 10.9%.
- Plateaus are only 20 runs in 20 units (median 128 h, 60% to the window end), mostly low (median 0.05% of
  customers; 15% at >= 1%). Eight runs in MD, OH, PA and WV start at hours 84-88 of 2024-05-08 and run to the end:
  one stale feed, not outages. Two Colorado counties (FIPS 08017, 08063) sit at exactly 20.4% and 21.8% from hour
  112 of 2019-03-13 to the end; 08063 is that event's largest error (18.3% of its SSE).
- The flag rate rises with the outage level: 0.63% of hours at (0, 0.1%], 2.0% at (1%, 5%], 5.9% above 20%.

## 5. Recovery

3,452 eligible units (peak >= 1%, >= 100 customers out, >= 24 h of window after the peak).

- **Fast, fastest for small peaks.** Kaplan-Meier median half-life after the peak: 2 h for 1-5% peaks, 3 h for
  5-20%, 5 h above 20% (at most 0.5% censored); median time to 10% of the peak: 5, 8 and 19 h. By event (peaks
  >= 5%) the time to 10% ranges from 5 to 23 h (Table 2). Of the states with >= 60 eligible units, Michigan is
  the slowest (half-life 6 h, 8.5% of the peak left at 24 h), then West Virginia (7.9%).
- **What slows it** (share left 24 h after the peak, within event): peak size +0.23, county size +0.21, peak
  customers +0.40, continued wind (max gust in the next 24 h: rho_p +0.119 [0.086, 0.149] given peak and
  customers). Cold barely matters: hours below 0 C +0.036 [0.001, 0.069], temperature near zero, snowfall -0.053.
- **Shape: heavier than exponential in aggregate.** On the 2,589 units with >= 96 h after the peak, the
  peak-weighted share left (sum of y(t*+s) over sum of y*) is 47% at 3 h, 23% at 12 h, 11% at 24 h, 4.0% at 48 h,
  2.0% at 72 h, 1.6% at 96 h; the implied half-life grows from 2.8 to 16 h, and an exponential through the 24-h
  value predicts 0.015% at 96 h. For the median unit with a peak above 20%, however, the implied half-life is
  roughly flat (6.1, 6.6, 5.9, 6.0 h at 12, 24, 48, 72 h): large single outages decay close to exponentially after
  the first hours, and the aggregate tail is a mixture of rates (a slow minority). Per unit, a power law in (1 + s)
  fits the log-decay better than an exponential for 65% of fitted units (75% of 1-5% peaks, 36% above 20%).
  Dropping plateau-flagged units changes little (1.4% instead of 1.6% left at 96 h).

## 6. Other things a model designer should know

- **Two storms in one window.** In 2021-12-11 (a third of all SSE), 71% of the event's SSE is at window hours
  >= 168 and 52% of its active units peak there: a second wind event four days after the nominal date. 2022-06-08
  (43%), 2019-11-27 (33%) and 2024-02-27 (30%) also carry much error after hour 168. Lead time is not storm phase.
- **Events hinge on a few counties.** Single county-events hold 35.7% of the SSE of 2024-02-27 (FIPS 36041, NY),
  22.4% of 2022-06-08 (46055, SD) and 18.3% of 2019-03-13 (08063, CO); the top five hold 26-63% of each event's
  SSE. Per-event scores are noisy.
- **Fractions of 1.0.** Five county-events reach 100%: 26001, 26143 (MI), 36041 (NY), 55125 (WI) in 2021-12-11 and
  54101 (WV) in 2019-02-24. All five are among the ten largest errors (36041 twice in the top ten). Their customers
  per resident (0.41-0.60) are ordinary, so the denominator is not obviously wrong.
- **Denominators.** Customers per 2020 resident: median 0.58, 1st-99th percentile 0.25-1.67; 164 counties above
  1.0, 43 below 0.3. The 105 units below 0.3 carry 4.5% of the SSE with 1.7% of the units; the 1,063 above 0.8 have
  less than half the median peak (0.97% vs 2.2%), consistent with inflated denominators (seasonal homes,
  overlapping territories) diluting fractions; not verified. Denominators are one 2024 snapshot for all events.
- **Prefix outages.** 99 units have more than 1% out at the origin (2.5% of the SSE); 11% of active units peak
  before the origin; one unit is constant and non-zero over the whole window.

## 7. Implications for model design

1. **Magnitude is the error budget.** 73 county-events carry half the SSE, peaks above 20% carry 79%, and those
   peaks are predicted at a median 18% of their size. An output layer and loss able to reach high fractions
   (heavy-tailed or count likelihoods, peak-aware terms), evaluated by peak tier, come before new inputs. Tension:
   the smallest customer quintile holds 41% of the SSE, so customer-weighted training moves effort away from where
   this metric's error is. Choose deliberately.
2. **Fast dynamics.** 55% of the SSE is within +/-6 h of the peak; outages rise from 10% to peak in about 2 h and
   halve in 2-5 h. Slow memories alone cannot produce this shape.
3. **A season-dependent clock.** County ERA5 gust times cold-season peaks (62% within +/-6 h), not warm-season ones
   (23%), where rain and convective proxies do somewhat better (41%). Give the damage path convective-timing
   inputs; do not force outages to follow the gust peak.
4. **Heterogeneous restoration rates** that depend on damage size, county size and post-peak wind (not cold). Large
   single outages decay near-exponentially, but in aggregate 1.6% is still out at 96 h where one exponential leaves
   0.015%: use unit-specific rates or a fast plus slow restoration pool.
5. **Geography: canopy x wind for peaks, nothing else yet.** Canopy is the one geographic signal left in the
   residual (+0.10 after the base's context inputs; observed peaks 3.5 times higher across canopy terciles at equal
   wind). It acts multiplicatively across the wind range of these storms and needs no population weighting.
   Poorly drained share has the unphysical sign and leaf-on is not identifiable here: no sign constraint, no
   leaf-on term. Antecedent rain is already absorbed by the host.
6. **Clean and stratify the target.** Mask fraction-1.0 hours and stale plateaus in training (0.4% of hours, 5.5%
   of the SSE), report metrics with and without them, flag single-county dominance per event, and treat windows as
   possibly multi-storm.
