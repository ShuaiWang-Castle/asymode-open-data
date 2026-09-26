# Hazard dictionary v2 — the weather features of every regime (data completion before Stage B)

2026-09-26. The PI asked to complete the weather data before running the system experiments. The host's inputs
(`build_features_r2.py`: twelve area-weighted ERA5 channels, hourly-maximum gust, gust excess energy, rain x wind,
snow and ice load proxies, near-freezing and cold precipitation, rolling sums and maxima, freeze-thaw crossings,
wind-direction changes, sub-county gust maxima) miss what the physical review of the panel design lists as missing
(`contrib/REVIEW_dataset_physics.md` section 5). Dictionary v2 adds it, for every regime, from two weather sources.
Code: `panel_v1/hazard_v2.py` (the features), `panel_v1/build_hazard_v2.py` (per system), `panel_v1/assemble_phi_v1.py`
(the pathway inputs). Development tranche only; the sealed tranche is built by the confirmatory script.

## What is added

| feature | why (review item) | ERA5 source | HRRR source |
|---|---|---|---|
| gust exceedance of the local 98th percentile, max(0, g/g98 − 1)³ | damage scales with the local extreme, not an absolute ramp (5.1; Klawa & Ulbrich 2003) | fg10; g98 from one random hour a day, 2008-2017 | GUST, with the ERA5 g98 of the node's cell |
| precipitation by phase: rain, freezing rain and drizzle, ice pellets, snow | the -1.5 °C hat mixes freezing rain and snow (5.2) | ptype on tp; sf | CRAIN, CFRZR, CICEP, CSNOW on APCP; FRZR |
| ice and wet-snow loads, temperature-gated states | glaze persists until a thaw, wet snow sheds on warming; linear memories cannot do it (5.3; Jones 1998; ISO 12494) | state at the node | state at the node |
| load x wind, antecedent wetness x gust exceedance, formed at the node | the county integral of a product is not the product of the integrals (5.4); anchorage fails on wet soils (5.5) | ice·g², wet snow·g², swvl1 at the origin | the same, MSTAV |
| canopy x gust exceedance, canopy x load·wind, poorly drained x wetness·gust, at the node | geography acts on the hazard where the customers are (framework) | node canopy, drainage | node canopy, drainage |
| convective organisation: 0-6 km shear, 2-5 km updraft helicity, composite reflectivity, lightning | organised systems against pulse storms (5.8) | - | VUCSH/VVCSH, MXUPHL, REFC, LTNG |
| warm layer aloft | freezing rain against snow at the same surface temperature (5.2) | - | TMP 850 hPa > 0 °C over a surface below 0 °C |
| leaf state | phenology, not a calendar (5.6) | lai_hv | - |

Node = county x HRRR 3 km cell with its population share, elevation, canopy and poorly drained share
(`panel_v1/build_nodes_geo_v1.py`). ERA5 temperature is moved from the 0.25-degree cell to the node's elevation
(6.5 °C per km against the ERA5 orography), so the phase and the loads near 0 °C follow the terrain. HRRR fields are
the node's own cell.

Per county and hour: the population-weighted mean over the nodes, the maximum over the nodes, the population shares
above fixed thresholds (gust > 25 m/s, exceedance > 0, ice > 6 mm, wet snow > 10 mm, updraft helicity > 75 m²/s²,
reflectivity > 50 dBZ), and the geography x weather products. Missing HRRR hours enter as zeros and are listed
(DATASET_DESIGN amendment 2 S9).

## Constants (fixed before any model uses them)

Ice melts 0.5 mm per degree-hour above 0 °C and sheds 10% per hour with gust above 15 m/s. Wet snow accumulates
snowfall at -1..+3 °C, melts 1 mm per degree-hour above 3 °C, and sheds 10% per hour with gust above 12 m/s or
temperature below -3 °C. Loads x wind are divided by 1,000 (mm·m²/s²). The thresholds of the population shares are
the table's. None of these was tuned on outcomes.

## Not yet in v2

Terrain wind exposure (slope in the wind direction; review 5.9), flood terrain (HAND, TWI; 5.7), soil frost (5.5),
heat and cold load (5.10, regimes not sampled).

## How it enters the system (Stage B)

As the exposure-integrated hazard pathway of the host (competing-hazard entry, with and without the non-negativity
projection), per weather source; HRRR as a source of the host inputs is a separate comparison. Every comparison
follows program.md rule 3b and DATASET_DESIGN amendment 2 S10.
