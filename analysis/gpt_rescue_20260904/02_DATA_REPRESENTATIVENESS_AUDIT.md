# Data representativeness audit

## 1. The current eleven-event cohort is not a representative storm sample

The primary manifest is explicitly named `g2-convective-11` and contains eleven dates from 2021, 2022, and 2024. Its own note says that the main experiment uses “convective-season events” and that other families belong to a separate generalization manifest.

The data card defines a much broader sampling frame. Across 2018--2025 it identifies 436 storm days with at least 150 affected counties:

```text
convective  194
winter      185
wind         43
flood        10
tropical      4
```

It also records that the largest-footprint events are winter storms and explicitly warns that a convective-only study samples the fastest dynamics in the record. Therefore the current cohort cannot support claims about US storm outages in general. It supports, at most, a purposive study of eleven large convective-season panels with adequate data coverage.

## 2. Use all 26 built panels for model development and out-of-fold evaluation

The available `g3-all-26` manifest spans 2018--2024 and contains winter, convective, wind, flood, and tropical panels. It is still a hand-built convenience sample, but it is materially better than `g2` for learning and evaluating a model intended to transfer across event regimes.

The next main experiment should use fixed five-fold **event-stratified** cross-validation over all 26 panels. Fold construction may use only:

- event family;
- calendar year;
- county footprint and observation coverage;
- weather summaries computed without outage outcomes.

Each event must appear in exactly one outer test fold. Final metrics are computed per event and then averaged across events. The eleven-event cohort remains a historical diagnostic subset and must not be used to tune a new method and then presented as untouched confirmation.

## 3. Define the target population before claiming representativeness

The defensible target population is not “all power outages.” It is:

> Large-footprint US storm-day panels for which EAGLE-I coverage, a county customer denominator, and matched hourly weather are available.

Every claim should be scoped to that population. A future data release should sample all eligible events or use a documented stratified random sample from the candidate frame.

A particularly valuable external test would use 2025 EAGLE-I events selected by the same NOAA footprint and coverage rule before inspecting outage trajectories. Existing 2018--2024 panels can then be used for development, while 2025 provides a genuinely future, untouched event set.

## 4. Calendar-day aggregation may combine unrelated weather systems

The event catalog correctly notes that NOAA `EPISODE_ID` does not cross state lines, which makes it unsuitable for national synoptic events. The current fallback groups resolved event rows by UTC day. That fixes cross-state fragmentation, but a calendar day can also merge several unrelated systems in distant regions.

This matters because “leave one event out” is meaningful only if the held-out unit is a coherent environment. A mixed national day can give the model several unrelated weather regimes under one event label and can make event-level uncertainty difficult to interpret.

Before expanding the cohort, audit each candidate day for spatial multimodality. A cleaner event construction is a connected-component rule over:

- overlapping event time intervals;
- adjacent counties or contiguous weather footprints;
- compatible event families;
- a maximum temporal gap, for example 12 hours.

The outcome target must not enter this clustering. If a calendar day contains disconnected components, split it into separate storm systems.

## 5. The forecast-origin distribution is dominated by quiet periods

Each panel spans roughly two days before to five days after the event, with origins every 12 hours. This produces many highly overlapping forecast windows and many quiet pre- or post-event origins. The target is exactly zero in a large share of scored cells, and the health audit shows that near-zero origins dominate the error accounting.

The large number of origin rows does not increase the number of independent storm environments. It also causes the training objective to reward conservative near-zero trajectories.

Use an outcome-blind storm-conditioned origin rule. A forecast origin is eligible when the next 24 hours overlap the NOAA event interval or a prespecified weather-exposure window for the corresponding spatial component. Keep a fixed number of origins per event or weight each event-origin block equally. This retains onset prediction before the hazard while preventing long quiet tails from dominating the objective.

The full-window origin set may remain a robustness evaluation, but it should not be the only task if the paper asks about storm dynamics.

## 6. Missing zeros are a structural measurement problem

EAGLE-I omits entries with zero customers out. The project reconstructs zeros when a collection run exists and a county reports within a plus/minus seven-day service window. This is a transparent and reasonable heuristic, but it cannot fully distinguish a true zero from an unobserved county.

The distinction is load-bearing because the empirical two-flow advantage is concentrated at zero and near-zero origins. The paper therefore needs a data sensitivity analysis before interpreting that advantage:

- service windows of plus/minus 3, 7, and 14 days;
- stricter state-year coverage thresholds;
- high-confidence counties with dense reporting;
- reporting the result on explicitly positive transitions separately from reconstructed zeros.

These are data reconstructions, not additional model families. The same fitted models can be evaluated under compatible masks when possible; otherwise run only the final selected model and comparator.

## 7. The denominator is a 2024 snapshot applied to earlier years

The current target divides earlier outage counts by 2024 modeled county customer totals. The data card documents the semantic advantage of using the publisher's own denominator, but also acknowledges temporal drift and extreme county sizes.

At minimum, report:

- equal-county event-level MSE as the primary statistical metric;
- customer-weighted error as an operational secondary metric;
- sensitivity after excluding counties with very small denominators;
- a comparison against annual EIA-861-based county allocations when available.

Do not interpret small changes in outage fraction as physical rate changes without acknowledging denominator error.

## 8. Area-weighted weather is misaligned with a customer target

Weather is currently averaged over county land area, while the target counts customers. In large rural counties, area-weighted gust or precipitation can represent unpopulated land rather than the distribution network serving most customers.

The next driver build should use population- or customer-weighted weather as the primary aggregation. Area-weighted drivers should remain a prespecified sensitivity. This is a single data-design comparison, not a model sweep, and it directly tests whether poor spatial aggregation contributes to onset-timing errors.

## 9. The current data contain little interior-state information

The health audit reports that only 2.81% of rows have `y>0.1`, while the average served-pool multiplier is approximately `1-y=0.990`. Consequently:

- the interruption basis is nearly constant over most rows;
- restoration and turnover are weakly informed;
- tests for non-affinity have low power;
- local plug-in estimates of the flow-selection index are dominated by noise and nonnegativity boundaries.

A broad statement that the two physical rates have been recovered is not supported. The data can support a predictive comparison of two conditional-mean classes, and perhaps an onset-decoupling result, but not direct validation of latent interruption and restoration hazards without additional labeled process data.

## 10. Required data table before the next main run

Create one event-level table for all 26 panels with:

```text
event id
family and year
spatial component id
number of counties
observation coverage
number of eligible origins
zero-origin share
active-origin share
future-onset share
median and upper-quantile outage fraction
weather footprint summaries
county/customer-weighted denominator summaries
```

No model result may be used to choose events or origins. This table should appear in the supplement and should drive the fixed event-fold assignment.
