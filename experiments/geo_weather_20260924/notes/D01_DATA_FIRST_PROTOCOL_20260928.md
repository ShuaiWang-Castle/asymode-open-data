# D01: weather sequence, compound exposure and geography evidence

Written before fitting these probes on 2026-09-28 after the PI asked for data-first verification.
This is exploratory development analysis, not an untouched confirmatory preregistration. Earlier D diagnostics
and I18 results have already been seen. It authorizes no new neural arm, seeds or sealed evaluation.

## Question and scope

Do weather overlap, directional ordering and their geographic modulation add reproducible information about
subsequent county outage burden beyond weather severity, individual weather histories, geography main effects,
calendar, county context and already observed outages? Analyze only `features_v1D.npz`, the existing five event
folds and public county statics. Do not change data, masks, model code, the paper or C.

These are interpretable statistical probes, not handcrafted hazard channels for the damage network.
The outcome is observed outage stock, not identified physical damage or restoration rates.
The sample comprises selected public D systems and sampled counties; it is not a national causal sample.

## Units and timing

* Six fixed first-prediction hours a = 72, 96, 120, 144, 168, 192 in each 216-hour county-event.
  Inputs use `[a-24,a)`; current stock is `y_full[a-1]`; outcomes use `[a,a+24)`.
* Require current stock observed and at least 22 observed hours in each history/outcome window. Never treat
  missing values as zero. Report attrition. No window is aligned to a realized outage onset or peak.
* Primary response: mean observed outage fraction in the next 24 hours. Secondary response: whether its
  observed peak exceeds the current fraction by at least 0.01 (a net stock-rise indicator, not a failure label).
* All zero-outage windows stay in the analysis. Repeated windows do not become independent weather events.
  Divide each county-event's design weight by its retained window count; report untrimmed-weight sensitivity.
* This is a rolling-origin historical association task. Later anchors condition on outages that have become
  observed by that anchor. It is NOT the original fixed-origin AsymODE forecast task, and its scores cannot be
  compared numerically with I18 scores. Conditioning on past outage can absorb an earlier weather effect.
* Weather is ERA5 reanalysis, not issue-time archived forecasts. Past-only here describes the time indices,
  not real-time data availability.

## Fixed feature blocks

Use the 12 raw weather channels only, excluding the existing derived hazard/weather features. Apply log1p
to nonnegative precipitation, snowfall and CAPE. Main effects contain each channel's mean, standard deviation,
minimum, maximum, last value, three lag-band means (last 6 h, preceding 6 h, preceding 12 h), and peak age.
Controls include all 40 geography descriptors, six existing county statics, current/mean/max/trend of previous
outage, observed-history coverage, season, year, anchor, system regime and state. All continuous main effects
use a fixed piecewise-linear basis with knots at standardized -1, 0, +1. Scaling is fit-only at every inner and
outer split, with finite median imputation and bounded standardized inputs.

Interaction weather channels, chosen before fitting: gust, precipitation, temperature, soil moisture, snowfall,
CAPE. Interaction geography descriptors: mean elevation, relief, mean canopy, developed fraction,
poorly drained share, forest/wet-soil co-location. Retain all geographic main effects regardless of this subset.

For each of the 15 weather pairs include the product M of their window means, so sustained high/high exposure
is represented even when neither series varies. Also use within-window centered channels. C is their simultaneous covariance;
S is the mean symmetric crossmoment over lags 1 through 12 hours; O is the corresponding antisymmetric
crossmoment. Thus O changes sign under full time reversal, while C and S do not. Weather marginals and
individual lag-band histories are included before O. No lag or pair is selected from outcome results.

Six nested ridge probes:

| name | added block | incremental question |
|---|---|---|
| A | additive main effects and controls | reference |
| B | each selected weather's three lag-band means x each selected geography | ordinary weather/geography modulation |
| C | M, C and S for all weather pairs | sustained overlap and lagged compound exposure |
| D | O for all weather pairs | directional ordering beyond symmetric lagged exposure |
| E | M, C, S x each selected geography | whether compound associations vary with geography |
| F | O x each selected geography | whether directional-order associations vary with geography |

Run a mandatory second specification with unordered summaries (mean, SD, min, max) of the **future** 24-hour
weather added to A. This deliberately uses future environmental observations as a retrospective sensitivity;
it cannot measure operational forecast skill. A vanishing sequence increment here is compatible with the
past sequence merely predicting subsequent weather. It does not by itself rule out a physical mechanism.

Also run an aligned-exposure specification: use the response window's weather trajectory `[a,a+24)` in place
of the preceding window's weather, with the same response and pre-window outage controls. This is retrospective
same-window association, not forecasting. It checks whether conditioning on end-of-exposure stock obscured an
immediate association in the first specification. None of these three specifications is a causal intervention.

## Fitting and falsification

* Preserve the existing five event folds, which join families and nearby systems sharing counties. Inner
  validation uses each of the four remaining existing folds. Select ridge lambda from 0.001, 0.01, 0.1, 1
  separately per model and response; no outer score chooses lambda, features, window or target.
* Optimize equal-regime, within-regime design-weighted error; predict on held-out windows once, clipped to
  [0,1]. All transforms are fitted inside the relevant training partition. Save local OOF arrays separately
  from compact public results.
* For F, three fixed-seed capacity controls replace geography only in the compound/order interactions added
  beyond D
  with one county-level donor vector from the same state. All true lower-order features remain fixed. Donors
  are drawn from fit counties; one recipient county retains one donor across its events. Single-county state
  fallback is a fit-county donor from the entire training set and is counted. Held-out outcomes never enter
  mapping. These controls break the true local correspondence but are not an exact conditional randomization
  test and do not warrant permutation p-values. They may also alter weather/geography support.
* Do not use raw hourly shuffling as sole order evidence: it destroys smoothness, duration and physical
  weather combinations. The present screen separates symmetric vs antisymmetric terms but cannot isolate
  an experimentally manipulable weather-order effect. More narrowly matched trajectory controls would be
  required after a reproducible signal, not invented after a favorable coefficient.

## Reporting and decision discipline

Report paired incremental MSE changes B/A, C/B, D/C, E/D and F/E, plus joint F/D, by regime, five-fold direction counts,
and 2,000 cluster bootstrap intervals, clustering at the merged event group rather than treating windows as
independent. Include a county-cluster sensitivity to repeated county observations. Report raw design-weight
sensitivity, largest-system contribution and the estimate after removing that system. These intervals condition
on fitted OOF models, omit refit uncertainty, and are exploratory rather than multiplicity-adjusted claims.
Resample whole clusters within the regime carrying the largest retained design-weight mass of each cluster;
this preserves cross-regime members of a merged group. Record strata counts. Undefined zero-reference ratios
are null, not zero gain, and bootstrap support is checked. County intervals are a separate dependence sensitivity,
not a spatial generalization test or a joint two-way cluster interval.
The designated primary contrast is F/D on tropical/winter balanced burden MSE; the all-five mean, other
increments and secondary risk target are exploratory diagnostic results. They are not interchangeable endpoints.

An interaction is only a candidate for a focused follow-up if its primary-response error decreases, the merged
event interval excludes zero, at least four folds improve, the direction survives future-weather adjustment
and the largest-system removal, and F (for a joint geography claim) outperforms all three scrambled geography controls.
The F controls test joint geographic correspondence, not order-specific correspondence after preserving E;
an order-only geographic conclusion would require a subsequent matched control retaining true compound/geography terms.
Positive secondary risk results alone cannot satisfy this primary burden criterion. Negative results mean
insufficient evidence for these fixed, low-capacity summaries, not proof that geography or sequence is irrelevant.
Do not select a large kernel architecture until the measured pattern identifies what information it should retain.
