# Analysis-ready county outage panels, weather drivers and event catalogue

This directory is the **derived, analysis-ready** dataset behind the AsymODE
study: 26 United States storm events, county-level outage trajectories with an
explicit observation mask, hourly county-aggregated ERA5 weather, county static
covariates and the storm-event catalogue used to select the events. Everything
here is built from public sources by the scripts in `scripts/`; nothing is
redistributed that its source does not permit.

**Raw inputs are not included** (they total about 4 GB, and every one of them is
freely downloadable): `scripts/ingest_eaglei.py`, `scripts/fetch_era5.py`,
`scripts/build_event_catalog.py` and `scripts/build_county_statics.py` fetch
them, and `scripts/build_panel.py`, `scripts/build_drivers.py` and
`scripts/build_county_weights.py` rebuild every file below from them.

Integrity: `data/SHA256SUMS.txt` (verify with `shasum -a 256 -c data/SHA256SUMS.txt`).

## What is here

| files | what | shape |
|---|---|---|
| `panel_<YYYY-MM-DD>.npz` × 26 | one storm event each, windowed 2 days before to 5 days after | `y` (counties × 15-min steps, fraction of customers out), `observed` (bool mask), `denominator` (customers per county), `fips`, `ts` (UTC timestamps) |
| `drivers_<YYYY-MM-DD>.npz` × 26 | hourly county-aggregated weather for the same window and counties | `X` (counties × hours × 12), `channels`, `fips` |
| `county_statics.parquet` | per-county static covariates | area, centroid, rural–urban continuum, neighbour degree, utility reliability indices, customers |
| `eaglei_county_customers_2024.parquet` | the publisher's modelled customers per county (the denominator) | 3,059 counties |
| `eaglei_coverage_history.parquet` | EAGLE-I state coverage by year, used to gate counties in | 2018–2022 |
| `era5_county_weights.parquet` | area weights mapping the ERA5 0.25° grid onto counties | sparse |
| `storm_events_county.parquet` | NOAA Storm Events, county-coded and with zone records expanded to counties | 562,573 rows |
| `county_event_days.parquet`, `event_days_stratified.parquet`, `storm_episodes_ranked.parquet` | the event-selection catalogues (days ranked by county footprint, stratified by dominant event family) | — |

Driver channels: `cape, cloud, gust, precip, pressure, rh, snowfall, soil_moisture, t2m_c, u10, v10, wind_speed`.

## Two things to read before using the panels

**Zeros are omitted at the source, so the mask is not optional.** EAGLE-I does
not record a county-timestamp with zero customers out, so an absent record is
either a true zero or an unobserved cell and the file cannot tell you which. The
panels are densified under an explicit rule — a timestamp counts as a collection
run if any county reports at it; a county counts as in service on a day if it
reports within ±7 days; a cell missing while both hold is set to zero, and
everything else stays missing. `observed` marks the difference. On these windows
95.3–99.8% of cells are marked observed. **Excluding unobserved cells from every
loss and metric, rather than imputing them, is the intended use.**

**Timestamps are UTC.** EAGLE-I's `run_start_time` is documented as GMT and marks
the start of a 15-minute collection run; the driver files are aligned to the same
clock. `y` is `customers_out / total_customers` with the denominator above, which
is modelled by the publisher and described by them as approximate; it is a 2024
snapshot applied to earlier years, so it carries customer drift.

## Sources, licences and required attribution

| source | licence | attribution |
|---|---|---|
| EAGLE-I Power Outage Data 2014–2022, Oak Ridge National Laboratory for the U.S. Department of Energy (DOI 10.13139/ORNLNCCS/1975202) | **CC BY 4.0** (stated in the release README) | Tansakul, V. et al. (2023), *EAGLE-I Power Outage Data 2014–2022* |
| EAGLE-I Power Outage Data 2024 (DOI 10.13139/OLCF/2500278) | release README: "Reuse restrictions placed on the data: None" | Tansakul, V. et al. (2025), *EAGLE-I Power Outage Data 2024* |
| ERA5 single levels, Copernicus Climate Change Service (C3S) Climate Data Store | Licence to use Copernicus Products (redistribution of derived products permitted with attribution) | Hersbach, H. et al., ERA5 hourly data on single levels, C3S CDS. *Neither the European Commission nor ECMWF is responsible for any use of this Copernicus information.* |
| NOAA Storm Events Database, NCEI | U.S. Government work, public domain | NOAA National Centers for Environmental Information |
| NWS zone–county correlation file | U.S. Government work, public domain | NOAA National Weather Service |
| U.S. Census gazetteer, county adjacency, 2023 TIGER/cartographic boundaries | U.S. Government work, public domain | U.S. Census Bureau |
| USDA ERS Rural–Urban Continuum Codes 2023 | U.S. Government work, public domain | USDA Economic Research Service |
| EIA Form 861 (2023) service territory, sales, reliability | U.S. Government work, public domain | U.S. Energy Information Administration |

The weather files here are **derived products**: hourly ERA5 fields
area-weighted onto county polygons, not the ERA5 grid itself.

`poweroutage.us` / `poweroutage.com` are not a source for any file here.

## Citing this dataset

Cite the sources above. If the derived panels themselves are useful, cite this
repository and the accompanying paper.
