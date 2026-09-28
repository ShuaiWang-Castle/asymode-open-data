# D02: county structure and heterogeneous weather-response symptoms

## PI correction and objective

The PI asked to preserve a broad view of weather/geography complexity rather than narrowing immediately
to a few hand-picked mechanisms after a weak overall score. Different counties may exhibit opposing
responses, different thresholds, time scales or phases, so an overall average can obscure structure.
D01 is retained as the completed test of its shared representation and average predictive increment;
it is not a gate that county-specific patterns must first pass.

D02 is an exploratory atlas of the public D panel. Its aim is to identify and distinguish:

1. Opposing association directions across county structures on comparable exposure support.
2. Responses with the same direction but different timing, producing a blurred pooled trajectory.
3. Nonlinear response curves whose thresholds vary across county structures.
4. Apparent structure differences attributable to counties encountering different weather systems.

The existence of heterogeneity or cancellation is a hypothesis to examine, not an assumed result.
No overall MSE or single headline gate suppresses a county type, weather variable, weak estimate or null result.
This does not change I18's frozen screen, targets, masks, splits, neural architecture or manuscript.

## Three complementary views

### County structure

Use one row per unique county to form six fixed structural types from geography and existing county context,
with balanced feature-block scaling. These types are blind to the current D-event outage outcomes;
historical SAIDI in the context is historical service performance, so they are not purely physical terrain types.
Do not cluster by current outage burden or choose the number of clusters from an outcome result.
Retain all 40 geographic descriptors and six county context variables. Audit within-county consistency,
missingness, event coverage, repeated-family support and sensitivity of the structural partition itself.
Types receive neutral identifiers; feature profiles explain them without assigning causal labels.
The six types are a navigation layer, not an exhaustive set of mechanisms. Retain the continuous
46-dimensional representation, its outcome-blind PCA coordinates and county-level support beside the
type averages; within-type variation must not be replaced by a single centroid.

See `notes/D02_COUNTY_STRUCTURE_SCOPE_20260928.md` and `county_structure_d02.py`.

### Weather-anchored dynamics

Use weather-defined, not outage-defined anchors. Inspect gust, precipitation, cold (negative temperature),
snowfall, soil moisture and CAPE anchors while retaining the aligned profiles of all 12 raw weather channels.
Show county types and regimes, exposure-intensity strata, observed outage stock and observed hourly stock
changes, weather-to-outage lag distributions, and equal-weather-intensity ascending/descending branches.
Account explicitly for window boundaries, missing observations, all-zero/flat weather and zero outage.

Stock hysteresis alone can arise from accumulation and restoration; it is not proof of a new hidden-memory
mechanism. A weather-to-outage lag is not a propagation speed. Plot the complete patterns and support counts,
not just the largest selected effect. Descriptive bootstrap bands are not a multiplicity-corrected discovery test.
The first six anchors and current summaries are an auditable starting coverage, not an exhaustive account
of complexity. A missing signal here cannot rule out longer memory, multi-peak sequences or subcounty
spatial organization.

Implementation: `analyze_county_dynamics_d02.py`, with local caches under ignored `runs/`.

### Association direction and composition

Compare county-type/weather associations under no fixed effects, system/phase fixed effects,
county fixed effects and weighted two-way fixed effects. Use proper iterative projection for the unbalanced
panel, not simple double demeaning. Separate same-county changes from between-county contrasts.
Keep raw direction, early/late differences and pairwise overlap/order summaries across weather channels,
with observed starting stock/history sensitivities rather than one average forecast score.
Display fixed-effects and history-adjustment specifications together: these adjustments can absorb parts
of the process under study, so the most adjusted version is not a universal exclusion gate.

Report all types and regimes with county counts, independent event-group counts, repeated-county support,
weather support, weights and original-fold directional replication. Low-support cells remain visible and
are marked uncertain. Wind u/v and any terrain-aspect projection are local mean flow descriptions, not a
transmission graph or storm-motion observation.

If a cancellation index is reported, its exposure scale, fixed type weights and comparable support must be
explicit. A high index alone does not prove reproducible opposing effects: noisy estimates can also cancel.
Composition effects, opposing signs, shifted lags and different thresholds must not be conflated.
Matching a common exposure interval does not balance the exposure density within that interval. Different
local sampling of one shared nonlinear response can itself produce different linear slopes, including
opposite signs. Preserve that explanation alongside geographic modification; interval overlap is not
evidence of a common joint-weather distribution or identical local contrast.

See `notes/D02_HETEROGENEITY_SCOPE_20260928.md` and `analyze_county_heterogeneity_d02.py`.

## Evidence labels and deliverables

* **Descriptive pattern:** visible in these data, with its counties, events, uncertainty and missingness shown.
* **Repeated association:** direction or dynamic structure recurs across independent event groups/folds;
  disclose weather overlap, leave-group sensitivity and the fixed-effects specification used.
* **Mechanism interpretation:** a hypothesis for subsequent tests, not a causal conclusion from this atlas.

All results remain developmental and exploratory; the same D panel has already been examined.
Do not select a favorable type/weather/lag cell and call its ordinary interval confirmatory evidence.
The atlas keeps the full measured space available. A concise narrative may illustrate patterns but must
also show counterexamples and unsupported cells. No new neural seed, geography NULL or architecture run
starts automatically. C is never read and `paper_v1/` is never edited. Save compact public results,
preserve local caches, scan added material and push both research branches, never main.
