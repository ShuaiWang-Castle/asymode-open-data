# D02: county-structure association atlas

Frozen before this analysis is run, after D01 was seen. Exploratory D-only analysis;
no global prediction-MSE gate, neural fitting, C reads, or paper work. Use the fixed
outcome-blind six-type county lookup from `county_structure_d02.py` (county/type,
with type6 accepted as an alias). Do not choose partitions from outage patterns.

## Rows, drivers, responses

Reuse D01's 50,302 retained, non-overlapping 24-hour rolling-origin windows and
`w / retained anchors per county-event`. Use response-window weather, so these are
retrospective same-window associations. Retain zeros and the same observation masks.
The three responses are mean stock minus pre-window stock, positive part of observed
window peak minus pre-window stock, and the existing >=1 percentage point rise flag.
None identifies gross physical damage or restoration separately.

The full atlas has 56 drivers: 12 original weather-channel means; 12 last-12-hour
minus first-12-hour means; 15 simultaneous uncentered second moments among gust,
precipitation, temperature, soil moisture, snowfall and CAPE; their 15 centered
antisymmetric lag moments (D01 lags 1--12); and two mean wind projections along and
clockwise-across the county's circular terrain-aspect vector. The simultaneous
moment is E[x_i x_j], preserving both mean-product and covariance information.
Use D01's fixed log1p transform of precipitation, snowfall and CAPE; other channels
keep their existing units. The maps are statistical interactions, not hazard rates.

Terrain east/north vectors are the sine/cosine averages of the eight aspect shares.
Projection uses the unit terrain vector, requires resultant length >=0.1 and
horizontal wind speed >=2 m/s in at least 12 of 24 hours, and averages only those
reliable hours. Report this attrition. Positive along projection means the u/v wind
vector points toward mean terrain aspect, not meteorological wind-from direction.
This is county-average orientation, not an electrical network or storm-track map.

## Association decompositions

Within each of five regimes and six county types, estimate each driver separately:
intercept-only, system-by-anchor fixed effects, county fixed effects, and both fixed
effects. Use weighted alternating projections for the unbalanced two-way case and
report convergence. If the iteration limit is reached, solve the same weighted
dummy projection sparsely, removing one redundant dummy per bipartite connected
component. Verify all original weighted group means below 1e-9 in normalized units
before accepting either solution; report fallback use. Pair each with a sensitivity adjusting pre-window stock and
the four D01 outage-history/coverage controls. Slopes use one fixed, design-weighted
within-regime driver SD and response percentage points; do not standardize separately
by type. These are bivariate partial associations, not independent causal effects.
Before fixed-effect projection, history controls are scaled to unit design-weighted
variance. Partial regression retains residual-control Gram eigenvalues above
max(1e-12, 1e-10 times the largest eigenvalue); the absolute floor prevents inverting
roundoff for completely absorbed controls. Record retained ranks. This numerical
criterion was added after the first completed atlas passed independent projection
checks but exposed this absorbed-control edge case; the pre-correction artifacts
are preserved under ignored runs, with no change to drivers, outcomes or strata.

Window-level county fixed effects mix within-event evolution with cross-event
variation. Therefore also collapse windows to one design-weighted county-event row
and report county-fixed-effect slopes, with and without the same history controls.
This small auxiliary table explicitly targets cross-event variation. Omit the two
terrain projections from this auxiliary table because averaging only their reliable
windows would mismatch the full-window outcome/control averages; preserve their
window-level results. A further
u/v sensitivity adds window mean wind speed and its early/late contrast, with stock
history, rather than interpreting wind-component signs alone as direction effects.

For uncertainty, use merged-event-cluster sandwich scores, G/(G-1) correction and
t critical values with G-1 degrees of freedom, counting clusters with nonnegligible
residual driver variation. These intervals condition on this fixed descriptive
specification and are not simultaneous or causal intervals. County repeated-event
dependence is not jointly eliminated by this event-cluster interval. Also estimate
directions separately in the five original event-fold subsets. These are separate
descriptive fits, not OOF prediction scores or an untouched confirmation split.

## Support, cancellation and complete reporting

Every type/regime cell reports windows, county-events, counties, systems, families,
merged groups, counties repeated across >=2 families and separately >=2 merged groups, weight concentration,
weather P10/P90 and maximum group weight/leverage. Retain sparse cells; grey flags
include <20 counties, <5 merged groups, fewer than 10 repeated-merged-group counties for
cross-event interpretation, or too little residual variation. Flags are evidence
limitations, not exclusion or effect-selection rules.
The common-support and fold-subset fits also retain residual informative group
counts, maximum group leverage and residual driver variance. A nonzero fold sign
without adequate residual support is only a sign, not an independent replication.

For each regime/driver define a common exposure interval as the intersection of
the represented types' weighted P10--P90 intervals (an absent type has fixed q=0).
Repeat intercept-only unadjusted
and two-way/history-adjusted fits inside that interval. Empty/degenerate intervals
or sparse retained support remain explicit. This matches marginal exposure ranges,
not joint weather trajectories or event composition; the FE comparison is still needed.

Fix q_type to the fraction of distinct counties of that type among the regime's
observed counties, equally weighting counties and keeping q unchanged across drivers,
responses and specifications. This describes D county composition, not national
population exposure. Report signed sum sum(q*beta), unsigned sum sum(q*abs(beta)),
and 1-abs(signed)/unsigned, where defined, both before and after common-support restriction.
The cancellation index alone is not evidence: inspect opposite directions, their
intervals, fold replication, event leverage and shared support. Near-zero slopes can
give a large index from noise; do not claim cancellation merely from the index.

Keep every driver/cell/method/response, including null and opposite results. Store
full-precision arrays in ignored runs and rounded complete arrays plus metadata in
`results/v1/county_heterogeneity_d02.json`. No pair, lag, type or direction is chosen
for reruns from this atlas. One process, nice >=15, at most two numerical threads.
