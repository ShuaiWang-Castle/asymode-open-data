# Feature map: the three process-specific blocks on the open data

The competition dimensions `59/17/43` are not transferable. What transfers is the
*roles*: interruption magnitude, interruption occurrence and recovery read
different blocks, and the occurrence block is neither the magnitude block nor any
hidden representation of it. This file records exactly what exists here.

All normalisation statistics come from the **fit counties of the source events
only**, computed before any gradient step, and are applied unchanged to the
validation counties and to the held-out test event.

## `x_u` — interruption magnitude (32 channels)

| channel | source | availability | kind | normalisation population | missingness |
|---|---|---|---|---|---|
| cape | ERA5 hourly reanalysis | at and before the step it is used | exogenous | training-fit transitions | none observed |
| cloud | ERA5 hourly reanalysis | at and before the step it is used | exogenous | training-fit transitions | none observed |
| gust | ERA5 hourly reanalysis | at and before the step it is used | exogenous | training-fit transitions | none observed |
| precip | ERA5 hourly reanalysis | at and before the step it is used | exogenous | training-fit transitions | none observed |
| pressure | ERA5 hourly reanalysis | at and before the step it is used | exogenous | training-fit transitions | none observed |
| rh | ERA5 hourly reanalysis | at and before the step it is used | exogenous | training-fit transitions | none observed |
| snowfall | ERA5 hourly reanalysis | at and before the step it is used | exogenous | training-fit transitions | none observed |
| soil_moisture | ERA5 hourly reanalysis | at and before the step it is used | exogenous | training-fit transitions | none observed |
| t2m_c | ERA5 hourly reanalysis | at and before the step it is used | exogenous | training-fit transitions | none observed |
| u10 | ERA5 hourly reanalysis | at and before the step it is used | exogenous | training-fit transitions | none observed |
| v10 | ERA5 hourly reanalysis | at and before the step it is used | exogenous | training-fit transitions | none observed |
| wind_speed | ERA5 hourly reanalysis | at and before the step it is used | exogenous | training-fit transitions | none observed |
| gust_max_since_origin | derived from ERA5 over the forecast window | accumulated from the origin forward | exogenous, causal within the horizon | training-fit transitions | none observed |
| wind_max_since_origin | derived from ERA5 over the forecast window | accumulated from the origin forward | exogenous, causal within the horizon | training-fit transitions | none observed |
| cape_max_since_origin | derived from ERA5 over the forecast window | accumulated from the origin forward | exogenous, causal within the horizon | training-fit transitions | none observed |
| precip_cum_since_origin | derived from ERA5 over the forecast window | accumulated from the origin forward | exogenous, causal within the horizon | training-fit transitions | none observed |
| panel_gust_footprint_share | derived from ERA5 across the panel's counties | same step | exogenous | training-fit transitions | none observed |
| clock_sin | panel timestamps (UTC hour of day) | same step | deterministic | not normalised (bounded) | none |
| clock_cos | panel timestamps (UTC hour of day) | same step | deterministic | not normalised (bounded) | none |
| log_area | Census / USDA ERS / EIA-861 county statics | fixed, time invariant | exogenous | training-fit transitions | column median imputation |
| rucc | Census / USDA ERS / EIA-861 county statics | fixed, time invariant | exogenous | training-fit transitions | column median imputation |
| log_pop | Census / USDA ERS / EIA-861 county statics | fixed, time invariant | exogenous | training-fit transitions | column median imputation |
| log_pop_density | Census / USDA ERS / EIA-861 county statics | fixed, time invariant | exogenous | training-fit transitions | column median imputation |
| n_neighbours | Census / USDA ERS / EIA-861 county statics | fixed, time invariant | exogenous | training-fit transitions | column median imputation |
| n_utilities | Census / USDA ERS / EIA-861 county statics | fixed, time invariant | exogenous | training-fit transitions | column median imputation |
| coop_share | Census / USDA ERS / EIA-861 county statics | fixed, time invariant | exogenous | training-fit transitions | column median imputation |
| saidi | Census / USDA ERS / EIA-861 county statics | fixed, time invariant | exogenous | training-fit transitions | column median imputation |
| saifi | Census / USDA ERS / EIA-861 county statics | fixed, time invariant | exogenous | training-fit transitions | column median imputation |
| log_cust | Census / USDA ERS / EIA-861 county statics | fixed, time invariant | exogenous | training-fit transitions | column median imputation |
| log_cust_density | Census / USDA ERS / EIA-861 county statics | fixed, time invariant | exogenous | training-fit transitions | column median imputation |
| lat | Census / USDA ERS / EIA-861 county statics | fixed, time invariant | exogenous | training-fit transitions | column median imputation |
| lon | Census / USDA ERS / EIA-861 county statics | fixed, time invariant | exogenous | training-fit transitions | column median imputation |

## `x_occ` — interruption occurrence (6 channels)

| channel | source | availability | kind | normalisation population | missingness |
|---|---|---|---|---|---|
| gust | ERA5 hourly reanalysis | at and before the step it is used | exogenous | training-fit transitions | none observed |
| precip | ERA5 hourly reanalysis | at and before the step it is used | exogenous | training-fit transitions | none observed |
| log_cust_density | Census / USDA ERS / EIA-861 county statics | fixed, time invariant | exogenous | training-fit transitions | column median imputation |
| rucc | Census / USDA ERS / EIA-861 county statics | fixed, time invariant | exogenous | training-fit transitions | column median imputation |
| n_utilities | Census / USDA ERS / EIA-861 county statics | fixed, time invariant | exogenous | training-fit transitions | column median imputation |
| saidi | Census / USDA ERS / EIA-861 county statics | fixed, time invariant | exogenous | training-fit transitions | column median imputation |

**Honest statement about overlap.** The open-data driver block contains only twelve
weather channels, and `x_u` uses all of them, so no weather feature can be given to
`x_occ` without appearing in `x_u`. Strict disjointness is therefore impossible on
this dataset, and it is not what the plan asks for: the plan specifies
"instantaneous hazard plus a small static block". What is enforced is the part that
matters mechanically:

* `x_occ` has 6 channels against `x_u`'s 32;
* it contains **no** accumulated-hazard path feature, **no** clock, **no** storm
  footprint, and a different four-column static subset;
* it is a separate input tensor read by a separate `nn.Linear` module. It is not
  the magnitude network's input and not any hidden layer of it.

## `x_r` — recovery (23 channels)

| channel | source | availability | kind | normalisation population | missingness |
|---|---|---|---|---|---|
| t2m_c | ERA5 hourly reanalysis | at and before the step it is used | exogenous | training-fit transitions | none observed |
| wind_speed | ERA5 hourly reanalysis | at and before the step it is used | exogenous | training-fit transitions | none observed |
| precip | ERA5 hourly reanalysis | at and before the step it is used | exogenous | training-fit transitions | none observed |
| snowfall | ERA5 hourly reanalysis | at and before the step it is used | exogenous | training-fit transitions | none observed |
| log_area | Census / USDA ERS / EIA-861 county statics | fixed, time invariant | exogenous | training-fit transitions | column median imputation |
| rucc | Census / USDA ERS / EIA-861 county statics | fixed, time invariant | exogenous | training-fit transitions | column median imputation |
| log_pop | Census / USDA ERS / EIA-861 county statics | fixed, time invariant | exogenous | training-fit transitions | column median imputation |
| log_pop_density | Census / USDA ERS / EIA-861 county statics | fixed, time invariant | exogenous | training-fit transitions | column median imputation |
| n_neighbours | Census / USDA ERS / EIA-861 county statics | fixed, time invariant | exogenous | training-fit transitions | column median imputation |
| n_utilities | Census / USDA ERS / EIA-861 county statics | fixed, time invariant | exogenous | training-fit transitions | column median imputation |
| coop_share | Census / USDA ERS / EIA-861 county statics | fixed, time invariant | exogenous | training-fit transitions | column median imputation |
| saidi | Census / USDA ERS / EIA-861 county statics | fixed, time invariant | exogenous | training-fit transitions | column median imputation |
| saifi | Census / USDA ERS / EIA-861 county statics | fixed, time invariant | exogenous | training-fit transitions | column median imputation |
| log_cust | Census / USDA ERS / EIA-861 county statics | fixed, time invariant | exogenous | training-fit transitions | column median imputation |
| log_cust_density | Census / USDA ERS / EIA-861 county statics | fixed, time invariant | exogenous | training-fit transitions | column median imputation |
| lat | Census / USDA ERS / EIA-861 county statics | fixed, time invariant | exogenous | training-fit transitions | column median imputation |
| lon | Census / USDA ERS / EIA-861 county statics | fixed, time invariant | exogenous | training-fit transitions | column median imputation |
| hist_last | EAGLE-I outage panel | strictly before the origin | pre-origin observed history | training-fit transitions | unobserved cells excluded; all-NaN county -> 0 |
| hist_max | EAGLE-I outage panel | strictly before the origin | pre-origin observed history | training-fit transitions | unobserved cells excluded; all-NaN county -> 0 |
| hist_mean | EAGLE-I outage panel | strictly before the origin | pre-origin observed history | training-fit transitions | unobserved cells excluded; all-NaN county -> 0 |
| hist_trend | EAGLE-I outage panel | strictly before the origin | pre-origin observed history | training-fit transitions | unobserved cells excluded; all-NaN county -> 0 |
| neighbour_gust_mean | ERA5 + Census county adjacency 2023 | same step | exogenous | training-fit transitions | counties with no listed neighbour get a zero row |
| neighbour_precip_mean | ERA5 + Census county adjacency 2023 | same step | exogenous | training-fit transitions | counties with no listed neighbour get a zero row |

`x_r` carries **no clock channel and no current simulated state**, as required.
State dependence enters the recovery side only through the `-R_t Y_t` term of the
state map.

## What was deliberately not built

| omitted | reason |
|---|---|
| county identifier embedding | prohibited by the plan |
| any post-origin outage observation | the forecast is open loop |
| target-derived event or origin features | anchors come from NOAA times only |
| wind-direction change, freeze-thaw cycling | not required by the V2 scaffold; adding them would be an unauthorised feature search |
| competition features with no open-data counterpart | omitted and recorded here rather than fabricated |

## Provenance of the primitives

`asymode_paper.features` is new. It reuses only `asymode.features.load_adjacency`
for the Census adjacency file, which is a data reader, not a model component. No
arm was added to `asymode.dynamics`.
