# Literature index for the geo-weather line (maintained; newest notes first)

Three detailed notes sit behind this page; every citation in them was checked against the publisher or an
archive (see each note's verification section). This page keeps what each strand of literature means for the
design, so it can be re-read at a glance.

* `notes/LITERATURE_preprocessing.md` — outage-model preprocessing: targets, weather features, exposure,
  reanalysis limits, finer public weather; ranked list of 12 changes for our inputs.
* `notes/LITERATURE_framework.md` — literature map (differentiable parameter learning, sub-grid tiles and
  downscaling, fragility and accretion, conditioning and shortcut learning, spatio-temporal outage models) and an
  adversarial review of DESIGN v0.
* `contrib/MECHANISMS_structure.md` — physics of six outage mechanisms with ERA5-computable intensity measures,
  bounded parameters and literature constants (lapse rates, phase thresholds, ISO 12494 icing windows, Jones
  glaze model, Klawa–Ulbrich wind loss).
* `contrib/REVIEW_formal.md` — the identity, what is recoverable from county aggregates, identification of
  geography coefficients, one principle (exposure-weighted integral of local fading-memory hazards), falsifiers.

## What each strand says for this line

**Aggregation and exposure.** Nonlinear hazards should be evaluated where weather meets customers and then summed
with exposure weights (Hsiang 2016; Klawa & Ulbrich 2003); fitting a response to aggregated weather smooths it.
The statistical names are ecological bias (Wakefield & Salway 2001) and disaggregation regression (Law et al.
2018); sub-grid tiles and elevation bands are standard in land-surface models (Avissar & Pielke 1989; Giorgi &
Avissar 1997; TopoSUB/TopoSCALE, Fiddes & Gruber 2012, 2014). *Here:* the exposure-integrated hazard features of
DESIGN v1; the F0 audit found the sub-grid (elevation-band) part empty on the wind panel and the exposure
weighting material.

**Local climatology.** Damage adapts to the local wind climate: losses scale with (v/v98 - 1)^3 above the local
98th percentile of daily-maximum gust (Klawa & Ulbrich 2003). *Here:* fixed gust knots for now; a local
climatology (monthly-mean proxy, then daily-maximum v98) replaces them.

**Peaks and durations.** Across utility outage models the robust predictors are the peak and the duration of
damaging wind above thresholds, then precipitation, antecedent wetness, leaf state and, in winter, snow density
and freezing rain (Wanik et al. 2015; Cerrai et al. 2019, 2020; Watson et al. 2021; Guikema et al. 2014).
*Here:* the fixed memory bank (tau = 3, 12, 48 h) over gated features, and the round-2 trailing summaries.

**Targets.** EAGLE-I stores no zero rows, scraper timeouts cluster in storms, stale maps repeat counts for days,
coverage and denominators change by year (Brelsford et al. 2024). *Here:* a training mask drops one-hour dips and
spikes, stale plateaus and saturated fractions from the training loss only (0.4% of hours, 5.1% of sum y^2).

**Phase and elevation.** The rain-snow threshold averages 1.0 C with humidity-aware (wet-bulb) partitioning
doing better (Jennings et al. 2018; Ding et al. 2014; Wang et al. 2019); near-surface lapse rates are shallower
than 6.5 K/km (Minder et al. 2010; Kunkel 1989 as tabulated by Liston & Elder 2006); wet snow accretes at 0..+3 C
(ISO 12494), glaze from freezing rain (Jones 1998); an operational wet-snow alert system does DEM-min/max
downscaling per cell (the WOLF system, cited in the preprocessing note). *Here:* fixed monthly lapse rates, Stull
(2011) wet bulb, precipitation x wet-bulb hat features; ERA5 precipitation type deferred; the winter ice-storm
panel W1 is where these can be tested.

**Learning through a process model.** Differentiable parameter learning (Tsai et al. 2021; Feng et al. 2022,
2023), multiscale parameter regionalisation (Samaniego et al. 2010), universal differential equations
(Rackauckas et al. 2020). Static attributes add little out of sample when the forcing already reveals them
(Heudorfer et al. 2025), the hydrology analogue of the open-data GCRK result. PINN-style residual penalties need
a trusted governing equation, which outages lack; the trusted structure is the balance law plus constitutive
features (Raissi et al. 2019; Krishnapriyan et al. 2021 on failure modes). *Here:* inductive bias through the
population balance and a monotone competing hazard, no residual penalty.

**Fading memory.** A causal time-invariant fading-memory operator is approximated by a bank of exponential
filters followed by a static readout (Boyd & Chua 1985); estimating time constants of exponential sums is
ill-conditioned (Istratov & Vyvenko 1999). *Here:* fixed log-spaced time constants, learned non-negative weights.

**Evaluation.** Random splits inflate outage-model skill; leave-one-event or leave-one-state designs often fail
to beat a null (Essus et al. 2026, preprint; Roberts et al. 2017 on block cross-validation). *Here:* county-grouped
screens with county and event x state cluster intervals; event-grouped checks before any claim.
