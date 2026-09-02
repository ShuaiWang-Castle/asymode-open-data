# Data card

Every source below is public and citable. Nothing here requires a licence
negotiation, and nothing here is derived from a restricted dataset.

## Prediction target

    y_{c,t} = customers_out_{c,t} / customers_{c}   in [0, 1]

the fraction of tracked customers in county `c` without power at 15-minute
timestamp `t`. This is the quantity the data publisher itself defines the
denominator in order to support ("percent of customers without power by county"),
so the target is the source's own intended use, not a construction of ours.

### Numerator -- EAGLE-I, ORNL / DOE

County-level `customers_out`, 15-minute resolution, collected by ETL from
utilities' public outage maps.

| Years | DOI | Notes |
|---|---|---|
| 2014-2022 | `10.13139/ORNLNCCS/1975202` | 8 years; ships state-by-year coverage **and** a modeled county-customer file |
| 2023 | OSTI biblio 2297398 | annual release |
| 2024 | `10.13139/OLCF/2500278` | **carries `total customers per county` in the records themselves** |
| 2025 | `10.13139/ORNLNCCS/3012826` | annual release |
| metadata | `10.13139/ORNLNCCS/1975203` | field-level metadata record |

Landing page: <https://doi.ccs.ornl.gov/dataset/ccec86f0-e144-5de8-aee0-fb26028b26e1>

### Denominator -- NOT in the archive. Open question.

The earlier reading of this was wrong and is corrected here. The portal blurb
promising a "modeled county customer dataset" describes the EAGLE-I *programme*,
not this archive. The 2014-2022 release contains exactly ten files: nine yearly
outage CSVs and `coverage_history.csv`. **There is no county-level customer count
anywhere in it.** The outage records carry five columns -- `fips_code`, `county`,
`state`, `sum`, `run_start_time` -- and no denominator.

`coverage_history.csv` is **state**-level: 280 rows over 2018-2022 with
`total_customers`, `min/max_covered` and `min/max_pct_covered` per state-year.
Useful for gating, useless as a county denominator. Note it also **does not cover
2014-2017**, so coverage-gated work is restricted to 2018 onward.

Three routes to a county denominator, in preference order:

1. **EAGLE-I 2024 release** (`10.13139/OLCF/2500278`), same Globus collection,
   carries `total customers per county` inline. Same programme, same county
   definitions, same meaning of "customer" as the numerator -- the semantic match
   is decisive. Cost: it is a 2024 snapshot applied to earlier years, so it
   carries a few percent of drift, which must be stated.
2. **Modeled County Electric Customers** on OpenEnergyDataPortal. Correct in
   principle, but the portal requires an approved account and states that
   requests from personal email addresses are not approved -- an institutional
   address and a wait. Not on the critical path.
3. **Build it from EIA-861** service territory and customer counts. Fully
   reproducible from primary sources with no account, but allocating a utility's
   state-level customers across the counties it serves needs an assumption of its
   own. Best kept as a robustness check against route 1.

**Resolved — route 1 is in use.** `data/interim/eaglei_county_customers_2024.parquet`
(built by `scripts/ingest_eaglei.py`): the publisher's own modelled per-county
totals from the 2024 release — 3,059 counties, constant within the year for
99.9% of them, median 16,995 customers, range 5 to 3,799,750. Every graded
number rests on it. The provisional population-share stand-in used while the
pipeline was being built was off by a median ratio of 1.17, a typical 23%, and
by more than 2x in 8.3% of counties; it is retained only in the ledger's record
of the comparison (`RESULTS_LEDGER.md`, "Denominator — resolved").

### Coverage

The 2014-2022 release includes EAGLE-I coverage per state per year. Coverage is
the fraction of a state's customers whose utilities were being scraped, and it
changes over time. A county-year whose state coverage is low is not a county with
few outages -- it is a county that was partly unobserved. Coverage must gate
inclusion, and the threshold must be stated in the paper.

## Zeros are omitted, and that is the central data problem

"Entries with 0 customers without power were not included in this dataset." In
2021 the records fill **23.3%** of the (county x 15-minute) grid: 24.8 M rows over
3,045 counties and 34,926 of the 35,040 possible timestamps. The other 76.7% is
either a true zero or an unobserved cell, and the file cannot tell you which.

This is not a nuisance. A county sitting at zero before a storm *is* the onset
case this project is about, and an unobserved county looks exactly the same. The
densification rule is stated in `src/asymode/panel.py` and carries an explicit
observation mask: a timestamp counts as a collection run if any county reports at
it, a county counts as in service on a day if it reports within a week either
side, and only a cell that is missing while both hold is filled with zero.
Everything else stays missing and is excluded from every loss and metric rather
than imputed. On the first eight storm windows built the mask marks 95.3-99.8% of cells
observed; 26 windows are built in total (manifest `g3-all-26`).

## Drivers -- weather

ERA5 single-levels (Copernicus CDS, `cdsapi`), CONUS at 0.25°, hourly, one
request per calendar month per storm window (`scripts/fetch_era5.py`), 26/26
windows acquired. Eleven raw fields (u10, v10, i10fg, t2m, d2m, cape, swvl1,
tcc, sp, tp, sf) are **area-weighted** onto counties with
`scripts/build_county_weights.py` (Census cb_2023 county shapes against the
ERA5 grid) and derived into the driver channels in `src/asymode/weather.py`:
gust, wind speed, u10/v10, 2-m temperature in °C, relative humidity from dew
point, CAPE, surface pressure, cloud, soil moisture, precipitation in mm,
snowfall, plus a diurnal clock — the 14-channel block whose digest
(`dec964873cb2`) every result file carries. Area weighting is the declared
choice: it is the one implemented, and population weighting remains an
untested alternative.

## Coverage gaps that are not obvious from the labels

* **The 2014-2022 archive stops on 2022-11-12**, not 2022-12-31, despite the title.
  The row count matches the published figure exactly, so the file is complete as
  released; the collection simply ends in mid-November. Winter Storm Elliott
  (2022-12-22, the largest county footprint in the whole storm catalog at 938
  counties) is therefore **not obtainable from this release**. Any window must be
  checked against the actual timestamp range, which `scripts/build_panel.py` now
  does rather than silently building a panel out of nothing.
* `coverage_history.csv` spans **2018-2022 only**. Years outside it fall back to
  the nearest available year, which is stated in the log when it happens.

## ERA5 request gotcha — cross-month windows are inflated

A CDS request lists `year`, `month` and `day` as sets and returns their **cross
product**. A ten-day window that crosses a month boundary therefore returns
twenty days (e.g. Sep 24–30 plus Oct 1–3 becomes Sep 1–3, Sep 24–30, Oct 1–3,
Oct 24–30). Five of the 26 windows were affected; the Helene window returned 480
hourly steps instead of 240 and took 45 minutes. **Correctness is unaffected** —
`scripts/build_drivers.py` reindexes every field onto the panel's own hour grid
and the surplus days are dropped — but bandwidth and queue time roughly double.
`scripts/fetch_era5.py` now issues one request per calendar month.

## Event selection -- NOAA Storm Events

Bulk CSVs, no account: <https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/>

Downloaded years 2021, 2022, 2024, 2025. After restricting to county-coded rows,
146,475 event rows; 63,638 county-days carry an outage-relevant event across
3,267 distinct counties.

Two findings that change how selection has to work:

1. **`EPISODE_ID` never crosses a state line.** Episodes are assigned per state,
   so the largest single episode in these four years touches 65 counties in one
   state. Episode is therefore the wrong key for a synoptic event. Keying by UTC
   day instead yields 260-369 counties across 16-32 states on the largest days --
   the scale county-held-out evaluation actually needs.
2. **Tropical cyclones are absent from county-coded rows -- fixed.** Hurricane,
   Tropical Storm, Winter Storm and High Wind events are filed as *zone*
   (`CZ_TYPE == 'Z'`) records, not county records, so a county-only filter drops
   every one of them. Expanding zone rows through the NWS zone-to-county
   correlation file (`data/raw/nws/zone_county.txt`, 4,793 rows) adds 260,544
   resolved rows on top of 302,029 county-coded ones, and takes tropical cyclone
   event rows from **0 to 2,758**. Michael, Isaias, Zeta and Helene are visible
   only after this step. Zone rows are flagged `cz_type == 'Z'` so the two codings
   stay distinguishable downstream.

3. **Event selection was the real constraint, not data volume.** Ranking days by
   county footprint across 2018-2025 yields **436 days with at least 150 counties**:
   194 convective, 185 winter, 43 wind, 10 flood, 4 tropical. The largest events
   are winter storms, not convective ones -- Uri reaches 842 counties against 474
   for the largest convective day. A study that samples only convective days is
   sampling the fastest dynamics in the record and is not representative.

## Static county covariates -- all public, re-pulled from origin

| Field | Source |
|---|---|
| adjacency | Census county adjacency 2023 |
| centroid, land area | Census Gazetteer 2023 |
| rural-urban continuum | USDA ERS RUCC 2023 |
| utility service territory, SAIDI/SAIFI | EIA Form 861 |
| forest / canopy share | USDA FS FIA EVALIDator API, over Census land area |

Built (`scripts/build_county_statics.py`): Census Gazetteer (area, centroid),
Census adjacency (neighbour degree), USDA RUCC 2023, EIA-861 2023 service
territory, sales and SAIDI/SAIFI, and the EAGLE-I 2024 customer totals. Forest
canopy share is listed as a source and has not been pulled. No static enters
any result in the current draft; they exist for the registered input-asymmetry
hypotheses.

## Explicitly excluded

`poweroutage.us` / `poweroutage.com` are not used as a source at any point.
