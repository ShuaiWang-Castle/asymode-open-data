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

**Currently in use:** a *provisional* denominator, state `total_customers` from
`coverage_history.csv` apportioned to counties by 2020 population share. It is
labelled `provisional_state_pop_share` in every artifact so the caveat cannot be
separated from the number. It exists to make the pipeline runnable end to end,
not to support a claim.

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
than imputed. On the eight storm windows built so far the mask marks 95.3-99.8%
of cells observed.

## Drivers -- weather

ERA5 reanalysis (Copernicus CDS) or NOAA URMA. County aggregation is either
area-weighted or population-weighted; **pick one and state it**. Population
weighting is the better default here because outages are counted per customer,
not per unit area, but it must be a declared choice, not an implicit one.

Blocked on a Copernicus CDS account -- see `docs/ACCESS_TODO.md`.

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
2. **Tropical cyclones are absent from county-coded rows.** Hurricane and
   Tropical Storm events are filed as *zone* (`CZ_TYPE == 'Z'`) records, not
   county records, so the county filter drops every one of them. Any hurricane
   case study requires the NWS zone-to-county correlation file first. This is an
   open gap, not a solved step.

## Static county covariates -- all public, re-pulled from origin

| Field | Source |
|---|---|
| adjacency | Census county adjacency 2023 |
| centroid, land area | Census Gazetteer 2023 |
| rural-urban continuum | USDA ERS RUCC 2023 |
| utility service territory, SAIDI/SAIFI | EIA Form 861 |
| forest / canopy share | USDA FS FIA EVALIDator API, over Census land area |

Downloaded so far: Census adjacency, Census Gazetteer.

## Explicitly excluded

`poweroutage.us` / `poweroutage.com` are not used as a source at any point.
