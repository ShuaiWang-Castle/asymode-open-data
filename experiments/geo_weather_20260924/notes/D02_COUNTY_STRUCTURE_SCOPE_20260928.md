# D02: outcome-blind county structure map

Written on 2026-09-28 before computing these county clusters, after D01 results were seen.
This is exploratory development characterization, not an untouched confirmatory protocol.
The purpose is to preserve potentially opposing associations across county structures;
neither clustering nor the choice of K may use outage outcomes, fitted predictions, or their errors.

## Population, inputs and fixed geometry

Use only public D `features_v1D.npz` and the existing D event split metadata. Deduplicate
county FIPS before fitting: each of the expected 2,410 sampled counties receives one vote,
regardless of its event count or sampling weight. Read the 40 geography columns, six
existing county context columns at `xr[:, 0, 14:20]`, and small county/event metadata.
Do not read outage arrays, damage/weather arrays, model predictions, or sealed C.
The compressed `xr` member is streamed to retain only those six static values; its
weather/history values are not materialized as an array. Check all 216 hours of those
six static columns while streaming. Never infer constancy from a column name alone.

Audit each county's values and missingness across events. Require exact equality after
the existing float32 encoding, treating NaNs as equal; stop rather than silently selecting
an event-specific representation when a county is inconsistent. Use its first row only
after this check. Missing values are replaced by the unweighted median over distinct D
counties. Center and scale each imputed column using the unweighted county mean and
population standard deviation. A constant column has scale 1 and contributes zero.
No clipping, outcome screening, feature selection, or whitening is performed.

Partition all 46 columns once, with no duplicates, into five fixed blocks:

* Terrain and extent (17): elevation, relief, slope, steepness, ruggedness, eight aspect
  fractions, log land area, s50 relief, elevation mean5 and relief5.
* Canopy and land cover (7): mean/dense canopy, forest/developed/wetland fractions,
  s50 canopy and FIA forest-land share.
* Soil and drainage (10): poorly drained, hydric, shallow, high water table, root limiting,
  windthrow susceptibility, s50 windthrow/poor drainage, wet-soil share and windthrow hazard.
* Spatial co-location (6): forest on steep land, canopy in developed land, forest/wet
  co-location, wet/hazard in forest and forest near developed land.
* County and service context (6): log customer count, RUCC, log population density,
  cooperative share, log1p number of utilities and log1p SAIDI (the existing 2017 context).

Divide standardized columns in each block by the square root of that block's dimension,
then by sqrt(5). Thus each block has equal aggregate scale rather than giving the largest
block the most weight. Correlated descriptors remain correlated; this geometry does not
claim five statistically independent mechanisms. RUCC is treated as an ordinal numerical
descriptor, as already encoded, and SAIDI is historical service context rather than a
target from the analyzed events.

Fit KMeans with fixed K=6, seed 20260928, n_init=50, max_iter=500, tolerance 1e-4 and
Lloyd updates. Use two numerical-library threads and process niceness at least 15.
Report neutral type IDs 0 through 5, canonicalized by each cluster's smallest FIPS.
They are descriptive partitions of observed county descriptors, not causal county types.

## Continuous geometry extension

Added after the initial structure characterization on 2026-09-28, at the PI's request to
avoid flattening continuous within-type differences. Keep all original types, raw features,
normalization and block weights unchanged. Apply ordinary unweighted PCA to that same
46D county geometry, using full SVD and no whitening. Retain six PC scores and report
their explained-variance ratios and strongest positive/negative component coefficients.
Orient each component so its largest absolute coefficient is positive; this sign convention
has no scientific direction attached to it. Two coordinates are a navigation aid alongside
the full descriptors, never an outcome-based selection or replacement of the 46D geometry.

Publish every county's first two coordinates, neutral type, a few event-support counts,
distance to its assigned centroid, nearest/second-nearest centroid distances, and normalized
gap `(d2-d1)/d2`. Distances use the full 46D geometry; the margin is not a class probability.
The display is a descriptor-space projection, not a geographic map or proof of natural
categories. Report omitted variance so visual proximity cannot silently imply similar
weather exposure, the same mechanism, or full-space similarity. A type mean must not
replace the county points or the original feature values.

For this additive update, the explicit `--augment-existing-geometry` entry point reads
only the existing structure lookup and summary. It preserves all original lookup arrays
exactly, backs up the initial lookup/JSON under ignored `runs/`, and atomically replaces
the expanded outputs. It does not reread event outcomes or refit KMeans. The default
fresh-run path still refuses to overwrite an existing result.

## Structure, coverage and stability reporting

Report per-type county counts; feature means, medians, quartiles and standardized means;
within-type spread and distance to centroid; and the largest standardized contrasts.
Report missingness, silhouette distributions and low-margin assignments so a discrete
label does not hide continuous variation or weak boundaries.

Coverage uses event metadata only: county-event, system, family, merged-event-group,
regime and existing outer-fold counts; numbers of counties repeated across at least
2, 3 and 5 distinct systems and merged groups; and Kish effective county support under
event repetition and accumulated design weights. These effective counts describe
concentration, not independent sample size or a standard-error correction. Give the
type-by-system and type-by-regime support tables for downstream checks.

Fit K=4 and K=8 with the same geometry and fixed seed only to describe structural
coarsening/splitting. Report adjusted Rand indices and membership cross-tabs against
K=6; never choose K using an outage result. Repeat K=6 with seeds 20260929 and 20260930
to assess initialization stability, preserving the designated seed's map regardless.

All D county descriptors define this map. It is outcome-blind but transductive within D;
it is not an independently learned spatial-generalization representation. Downstream
association estimates must retain event-family-heldout logic and repeated-county/event
dependence, disclose sparse type/regime cells, and avoid treating different signs as
established mechanisms merely because aggregate effects cancel.

## Artifacts and boundary

Write the compact public summary to `results/v1/county_structure_d02.json` and the
county-level lookup to ignored `runs/geo_weather_20260924/county_structure_d02/`.
The NPZ interface includes `county`, `type`, `features` (raw deduplicated values),
`feature_names`, `blocks`, standardized/geometry values, preprocessing parameters and
support counts; the extension adds PC scores/components and full-space centroid distances.
The compact JSON additionally keeps all 2,410 county points with six-decimal display values;
full precision stays in the lookup. County FIPS are public identifiers. Store only repository-relative paths
in artifacts. Never overwrite an existing result; require inspection before rerunning.
No new outage predictor, kernel, paper edit, or sealed evaluation is authorized.
