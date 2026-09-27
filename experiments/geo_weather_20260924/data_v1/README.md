# data_v1 — geography, weather features and development outcomes of the designed panel

2026-09-26. Data products of the designed public panel (`../DATASET_DESIGN.md` v1 with amendments 1-3), built from public
sources only. Only the **development tranche** is here (81 systems, 8,457 county-events). The sealed confirmation tranche
is not built and must not be built outside a registered confirmatory test. Code: `../panel_v1/`. County codes are the
2020 vintage: Connecticut's eight counties, as EAGLE-I reports them through May 2025.

## geography/

| file | rows | content |
|---|---|---|
| `county_geography_v1.parquet` | 3,108 counties (index fips) | terrain from 3DEP at 150 m: mean elevation, relief p95-p5, mean slope, steep share, ruggedness, 8 aspect shares. Vegetation and land cover from USFS tree canopy and NLCD 2021: canopy mean and dense share; forest, developed and wetland shares; forest on steep slopes; canopy in developed land. SSURGO soils: poorly drained, hydric, shallow, high water table and root-limiting shares; windthrow susceptibility. Log land area. 50 km smoothed relief, canopy, windthrow and drainage (`open_gcrk_20260919/build_geography.py`) |
| `county_geography_ext_v1.parquet` | 3,108 | gNATSGO soil wet share and windthrow hazard; forest x wet-soil co-location; wet soil in forest; hazard in forest; forest near developed land; five-point elevation and relief; FIA forest land share (`build_geography_ext.py`) |
| `county_axes.parquet` | 3,108 | the five audit axes of the design (section 4.6): relief (SD of elevation, m), canopy (%), poorly drained share, log customers per km², distance to the coast (km) |
| `county_statics_v1.parquet` | 3,230 | the host's county context: customers, rural-urban continuum code, population density, number of utilities, cooperative share, and SAIDI/SAIFI from EIA-861 **2017** (`saidi_imputed` flags a state median). `saidi_2023` and `saifi_2023` are kept for reference only. They are computed from 2023 interruptions and must not be model inputs on this panel (amendment 2 B1) |
| `nodes_geo_D.parquet` | 738,057 nodes, 2,410 counties | 3 km geography nodes of every development county: county x HRRR cell (hr, hc), population share `w_pop` (WorldPop 2020), population-weighted elevation `z` (m), canopy (%), poorly drained share `wet`, cell-centre lat/lon |
| `nodes3k.parquet` | 797,021 nodes, 3,108 counties | 3 km population nodes of every CONUS county (county x HRRR cell, WorldPop people, population-weighted point) |
| `adjacency2020.parquet` | 18,608 pairs | county adjacency (shared boundary or corner), 2020 boundaries |
| `era5_area_weights_v1.parquet`, `era5_popweights.parquet` | county x ERA5 cell | area weights (the host's drivers; CONUS grid index i, j) and population weights (global row, col) |
| `zone_shares.parquet`, `polygon_shares.parquet` | | exposure a of a county to an NWS forecast zone version and to an initial storm-based warning polygon (frame inputs, section 3.3) |

## weather/

* `era5_fg10_clim_2008_2017.npz`. The local gust climatology: the 90th, 95th, 98th and 99th percentiles and the mean
  of the ERA5 hourly maximum 10 m gust at one random hour a day, 2008-2017. The grid is the CONUS box, 50..24 N and
  -125..-66 E, at 0.25 degrees.
* `hazard_v2/era5/<system>.npz` and `hazard_v2/hrrr/<system>.npz`. The hazard dictionary v2 (`../HAZARD_V2.md`) per
  development system.
  * `X` [counties, 216 hours from the window start, features] (float16), with `names`, `fips` and `missing_hours`.
  * Every feature is computed at the 3 km node and integrated over the county's population: the mean, the maximum and
    threshold shares of gust, gust exceedance of the local 98th percentile, rain, freezing rain, ice pellets, snow, ice
    and wet-snow loads, load x wind, wetness x gust exceedance, and canopy/drainage x hazard. HRRR adds shear, updraft
    helicity, reflectivity, lightning and the warm layer aloft; ERA5 adds leaf area.
  * Hour 72 is the forecast origin.

## outcomes/

* `panel_v1D_outcomes.npz`. For each development county-event: its system, fips, regime, stratum (S1 warned, S2
  advisory or watch, S3 ring) and family, and the window start (UTC).
  * `y` [8,457, 216]: the hourly outage fraction, EAGLE-I customers out over the 2024 modelled customers.
  * `observed`: the observation mask of the pre-window service rule.
  * `customers`.
  * The inclusion probabilities `pi_system` and `pi_county`, and the design weights `w` (trimmed) and `w_raw`.
  * The sampling hazard index `h_c`, and the used flag.
* `development_systems.csv`. The 81 development systems: regime, storm name for tropical systems, family, origin,
  onset, window, region, period, season class, compound flag, used flag, inclusion probability, size measure, and
  stratum sizes.

## Sources and licences

USGS 3DEP, USFS tree canopy cover, NLCD, NRCS SSURGO/gNATSGO, USFS FIA, US Census (boundaries, gazetteer), EIA-861 and
NOAA/NWS products (HRRR, VTEC warnings via the Iowa Environmental Mesonet, HURDAT2) are US public domain. WorldPop 2020
is CC BY 4.0. EAGLE-I (Brelsford et al. 2024, *Scientific Data* 11:271; figshare 10.6084/m9.figshare.24237376 v4) is
CC BY 4.0. The ERA5 fields come from ARCO-ERA5; they contain modified Copernicus Climate Change Service information (2026),
and neither the European Commission nor ECMWF is responsible for any use of it. Downloads with URL, time, size and SHA-256
are in `../data_provenance/downloads.jsonl`.
