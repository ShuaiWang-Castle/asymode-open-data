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

### Denominator -- resolved, and it differs by release

This was the open question going in. The answer is not uniform across years:

* **2014-2022**: the release includes *"the modeled county customer dataset that
  was created to enable estimations of the percent of customers without power by
  county, with full methods described in Moehl et al."* That modeled county
  customer (MCC) file is the intended denominator and it is what we will use.
  It is a **model output, not a census**, and the paper must say so: county
  customer counts are estimated, carry their own error, and are static within a
  release.
* **2024**: no external file needed -- `total customers per county` is a column.
* **2025**: the record description does not mention per-county totals. Until the
  files are in hand, treat 2025 as needing the MCC file or the 2024 column
  carried forward, and **verify before use**.

Consequence for the panel: the denominator's provenance is not constant across
years, so any multi-year pooling must either restrict to one denominator regime
or carry a per-year indicator. Do not silently concatenate.

### Coverage

The 2014-2022 release includes EAGLE-I coverage per state per year. Coverage is
the fraction of a state's customers whose utilities were being scraped, and it
changes over time. A county-year whose state coverage is low is not a county with
few outages -- it is a county that was partly unobserved. Coverage must gate
inclusion, and the threshold must be stated in the paper.

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
