# Outage-model preprocessing in the literature, and what to change in our public inputs

2026-09-25. Literature note for `geo_weather_20260924` (data v3 and the mechanism layer of DESIGN.md).

Every work was checked against OpenAlex/Crossref and the publisher or archive page. "(full text)" = methods read;
"[preprint]" = not peer reviewed; UNVERIFIED = not confirmed. HRRR facts come from the AWS archive inventories
(2026-09-25). Numbers from papers describe their settings, not ours.

## Bottom line

1. The predictors that survive across outage models are the **peak** and **duration** of damaging wind above a
   threshold, then precipitation, antecedent wetness, leaf state and, in winter, snow density and freezing rain
   [12,16,18,22,23]. Our county-mean channels encode none of these directly.
2. Compute nonlinear hazards where weather meets customers, then sum with exposure weights [34,35]. The
   utility-grade models work on 2–4 km cells, tract centroids or storm objects, not county area means
   [12,16,22,27].
3. Static geography rarely helps alone [23,30]; it helps through line proximity, leaf state and elevation
   interacting with the event [12,19,29,52].
4. ERA5 misses convective gusts [38–40]. RTMA and HRRR are the public upgrades, with modest thunderstorm gains [19].
5. Some EAGLE-I "county effects" may be artefacts of ambiguous missing rows, storm-time collection timeouts and
   changing coverage and denominators [1,8].

## 1. Outage targets

**EAGLE-I [1] (full text).**
- Counts are scraped every 15 min from utility outage maps; polygon and zip records are split to counties by area,
  not population.
- Zero rows are not stored; for missing entries the authors state "we do not distinguish between these cases"
  (zero or gap).
- Parser timeouts happen when maps are overloaded, typically in extreme weather, and are usually recovered at the
  next run. A one-interval drop during a storm is more likely a gap than a restoration.
- The team's QA flags counts identical for more than 4 days (stale maps) and abrupt discontinuities; manual
  corrections and FEMA overwrites are in the data unflagged.
- Coverage rises from 2,152 counties (2014) to 3,044 (2022) and from 86% to 92% of customers (2018 to 2022).
  `coverage_history.csv` gives state minimum and maximum coverage per year; the quality index covers only
  2018–2022. The 2014 file is far smaller (1.7 M rows against 12.9 M in 2015; whether it covers a partial year is
  UNVERIFIED). Timestamps are UTC run starts; outages under 15 min are caught only by chance.
- `MCC.csv` models 2022 customers per county: EIA-861 utility totals allocated to LandScan population inside HIFLD
  territories. The 2024 file carries per-county totals [2]; a Fall-2025 file flags modeled versus collected
  counts [3]; the method is in [4].

**Cleaning and event rules elsewhere.** The brief lists Do et al. [5] (full text), Ganz et al. [6] and Flores et
al. [7] as EAGLE-I users, but they use PowerOutage.us or New York DPS data. Their rules still transfer:
- [5]: county-hours; keep counties whose feeds report at least 50% of the time and cover at least 50% of customers;
  denominators from ACS households plus business establishments; outage = at least 0.1% of customers out (their
  90th percentile) for 1+ or 8+ continuous hours.
- [7]: drop localities with fewer than 30 customers or more than 5% missing hours; region-specific 90th-percentile
  thresholds; weather flags at the statewide 97.5th percentile with an 8-h lag.
- [6]: duration runs from the peak until fewer than 5% of customers are out.

Studies that do use EAGLE-I:
- Lee et al. [8] (full text) sum utility rows per county, divide by the national coverage ratio 0.871, group
  overlapping NWS warnings per county, attribute outages from 3 h before to 48 h after each group, and set each
  county's threshold at its mean outage outside events. Outages persist 24 h past warning expiry on average. The
  EAGLE-I team's beta-binomial recovery model uses the same coverage scaling [9, preprint].
- Taylor et al. [10] define events as county × Storm Events episode and validate with chronological folds (R² 0.61
  for non-convective storms, 0.31 for thunderstorms). Their reference list cites a commercial aggregator, so their
  use of EAGLE-I is UNVERIFIED.
- Essus et al. [11, preprint] flag days whose peak exceeds a moving average by 3 SD. They find that fraction
  targets transfer across regions better than counts, and that random splits inflate skill.

**Restoration.** County restoration curves can be built from per-outage failure-time models [13]; the share not
yet restored decays roughly exponentially [14]; a few failures dominate customer-hours [15]. EAGLE-I counts
customers out at each moment, so restorations and new failures cancel (peak versus cumulative: [22]). After the
peak, the target follows restoration, which weather cannot explain.

## 2. Weather features in storm-outage models

| Study | Weather source, grid | Unit | Features that mattered |
|---|---|---|---|
| Wanik 2015 [16] (full text) | WRF 2 km (18/6/2 km nests on GFS), 60-h runs | 2-km cell | max and max 4-h running mean of wind and gust; hours above 9/13/18 m/s wind and 18/27/36/45 m/s gust; storm-total liquid and frozen precipitation; soil moisture; season class for leaf state |
| Cerrai 2019 [12] (full text) | WRF 2 km; 48-h (extratropical) or 36-h (convective) windows | 2-km cell | as above, plus hours above 5/9/13 m/s wind and 9/17 m/s gust and the longest continuous run; MODIS LAI (gap-filled climatology + anomaly) +2–9%; variable sets per storm type |
| Cerrai 2020 [18] | WRF; 4-km ML grid + town GLM | 4-km cell / town | assets, LAI, snow density, freezing-rain amount |
| Watson 2021 [19] (full text) | RTMA 2.5 km + Stage IV, versus WRF 2 km; 24-h storms | 2.5-km cell | mean/max/min/SD/total and 4-h peak means; hours above thresholds; LAI × wind²; SPI at 1/3/12 months plus lags; weather-grid minus line elevation |
| Alpay 2020 [20] (full text) | HRRR 3-km analyses, hourly | town-centre pixel | gust, wind, pressure, RH, CAPE, CIN, reflectivity; 1-h lag optimal; hour-of-day encoding for reporting lag |
| Guikema 2014 [22] (full text) | parametric hurricane wind field | census-tract centroid | max 3-s gust; hours above 20 m/s (wooden-pole design speed); customers per area; missed slow heavy-rain storms without a rain input |
| Quiring 2011 [23]; McRoberts 2018 [24] | hurricane wind-field variables | cell / tract | 37 soil and 20 terrain variables: no significant overall gain [23]; two-step model plus elevation, land cover, soil, precipitation and vegetation: ~17% [24] |
| Kabir 2019 [26]; Shashaani 2018 [25] | NDFD grids for Kabir (per [19]); Shashaani's weather source not checked | not checked | zero inflation handled with mixtures, resampling and cost-sensitive learning [26], or staged classifiers [25] |
| Tervo 2021 [27] | ERA5 0.25° + forest inventory | storm object (15 m/s gust contour) | object-mean wind most informative; stand height and diameter next |
| Taylor 2023 [10] | ERA5, ERA5-Land, ECMWF HRES (per its references) | county × episode | ecoregion, canopy, canopy height, roads |

Patterns:
- Each hazard is summarized three ways: its peak, a 4-h peak mean, and time above thresholds.
- Precipitation, antecedent wetness and leaf state enter explicitly.
- Winter models use snow density and freezing rain, not temperature alone.
- Thunderstorms resist every input:
  - WRF gust and rain added nothing because storms were misplaced (the double penalty [12]);
  - RTMA/Stage IV helped only slightly [19].
- With forecast inputs, gust and precipitation errors dominate, and outage error jumps at 4–5-day lead times [21].
  This matters for our 144-h horizon.

## 3. Vegetation, exposure and aggregation

- **Near lines.**
  - Utility studies sample land cover within 60 m of overhead lines [16,19]. The buffer was chosen because roads
    otherwise dominate the pixels [16].
  - LiDAR vegetation tall enough to strike lines was the best addition for Hurricane Sandy [28].
  - Roadside trees cause ~90% of Northeast storm outages; dropping tree variables lowers AUC from 0.832 to
    0.776 [29].
  - Tree species add ~3% [30].
- **Public proxies.**
  - Roads (OpenStreetMap via OSMnx [31]) serve as the infrastructure proxy in [10].
  - Open medium-voltage line maps exist, with 75% validation accuracy across 14 countries [32].
  - Synthetic distribution networks can be built from public data [33].
  - HIFLD covers transmission lines and service territories.
- **Exposure.**
  - Klawa and Ulbrich [34] (full text) weight by population and scale daily-max gust by the local 98th percentile:
    loss ∝ (v/v98 − 1)³ for v > v98. The reason is that damage adapts to the local wind climate: a coastal station
    above 20 m/s on 20% of days sees little damage.
  - Hsiang [35] (full text, NBER version) evaluates nonlinear effects per pixel-hour and aggregates with exposure
    weights; a response fitted to aggregated weather comes out smoother than the local one.
  - EAGLE-I allocates customers by population [1], and [22] uses population as its customer proxy.
  - Utility-grade models aggregate to 2-km cells, tract centroids, town pixels or storm objects [12,16,20,22,27].

## 4. Reanalysis limits and finer public weather

**ERA5 [36].**
- `10fg` is a parametrized hourly-maximum 3-s gust [37]. Its convective part comes from 850–950 hPa shear, not from
  resolved downdrafts [38].
- Gust errors are largest inland and in mountains, and shrink with an elevation term [39].
- Reanalyses underestimate extremes of CAPE, low-level moisture and shear, worst near coasts and mountains [40].
- Severe-wind environments separate on CAPE × shear (WMAXSHEAR) [41].
- ERA5 stores a precipitation-type diagnostic (`ptype`) derived from the vertical temperature profile, including
  freezing rain, wet snow and ice pellets [37,42]. The exact category list in ERA5 is UNVERIFIED.

**Storm Events.** Wind reports misstate the extent of damage [43], and estimated gusts are inflated, with spikes at
values ending in 0 or 5 [44]. Estimates can be inferred from observed damage, so the reports may partly encode the
target (our inference). NWS warnings (VTEC archive, used by [8]) are issued in real time and do not leak.

**RTMA and Stage IV.** RTMA [45] provides hourly 2.5-km analyses including gust, 2011 to the present, on Earth
Engine. Stage IV provides hourly 4-km radar-gauge precipitation [46].

**HRRR [47,48].**
- 3 km, hourly, radar-assimilating. Forecasts run to 18 h every hour and to 48 h every 6 h, so HRRR cannot drive a
  144-h forecast, but it can replace ERA5 as observed weather.
- Versions: v1 2014-09-30, v2 2016-08-23, v3 2018-07-12, v4 2020-12-02. The AWS archive starts 2014-07-30.
- Variables in the `wrfsfcf01` files:

  | Available from | Variables |
  |---|---|
  | 2014 (v1) | GUST, hourly-max 10-m wind, CFRZR/CICEP/CSNOW/CRAIN, CPOFP, APCP, WEASD, SNOD, CAPE, HPBL, LTNG |
  | v2 | ASNOW, FROZR |
  | v3 | FRZR, MAXUW/MAXVW, HAIL |

- Hourly-max and accumulated fields are degenerate in `f00` files, so read them from `f01`.
- A dozen fields cost 6–9 MB per hour via `.idx` byte ranges (Herbie [50]; archive in [49]), about 0.5–0.8 TB for
  2014–2024 (our estimate).
- HRRR still underestimates strong winds, and its temperature bias tracks station elevation [51].
- Version changes need version flags or quantile mapping per version.

## 5. Sub-county heterogeneity

- **Closest precedent.** WOLF [52] (full text), for each 5-km cell:
  - estimates temperature at the cell's DEM minimum and maximum, assuming 6 °C per km;
  - flags wet-snow accretion where that range crosses 0–2 °C;
  - grows the sleeve as ΔM = P·R·Δt·γ, with snow density 400 kg m⁻³ and sticking 1/V for winds of 1–10 m/s.
- **Accretion windows.**
  - ISO 12494 gives wet snow at 0–3 °C and glaze at −10–0 °C [53].
  - Nygaard et al. [54] identify wet snow by snowflake liquid fraction, and find the standard sticking efficiency
    underestimates accretion. General theory is in [55].
  - Freezing rain: Jones' simple ice-load model [56]; ice-to-liquid ratios from ASOS stations, median 0.72 on
    horizontal surfaces and 0.28 radial [57].
- **Rain versus snow.** The 50% rain–snow threshold averages 1.0 °C (−0.4 to 2.4 °C across stations), and
  humidity-aware partitioning beats temperature-only methods between 0.6 and 3.4 °C [58]. Wet bulb follows from T
  and RH [59].
- **Lapse rates.** Surface lapse rates are usually shallower than 6.5 K per km (annual 3.9–5.2, seasonal
  2.5–7.5 [60]). Rates derived from ERA5's own profiles beat a constant; the western US observes −4.5 K per
  km [61]. ERA5-Land applies such a correction at 9 km [62]. ERA5 ships its orography (`z`) and sub-grid statistics
  (`sdor`, `slor`) [37], and [19] used weather-grid minus line elevation as a feature.
- **Wind.** WRF's sub-grid scheme uses the standard deviation of orography and the terrain Laplacian to model
  hill-top speed-up and valley drag [63].
- **Trees.** Crown snow interception saturates as snowfall accumulates; capacity depends on leaf area, species and
  the existing load; unloading follows [64].

## 6. Ranked changes for our public inputs

| # | What to compute | Public source | Expected benefit (evidence) | Cost | Confidence |
|---|---|---|---|---|---|
| 1 | **Compute hazards first, then aggregate.** Build a lattice of ERA5-cell × county pieces weighted by 2020 block housing units. Compute every nonlinear hazard per piece-hour. County input = customer-weighted mean, p90 and max. (RESEARCH_LOG D4 already moves to population weights; the compute-first half is still open.) | ERA5; Census P.L. 94-171 | Removes the gap between f(mean) and mean(f); puts the hazard where customers are; this is DESIGN layer (a) with no new parameters [34,35,1,22] | Low–Med | High for the method, Med for the size of the gain |
| 2 | **Gust relative to local climate.** Per ERA5 cell, take v98 of daily-max `10fg` over 1991–2013, before the outage record. Hourly terms: (g/v98 − 1)³₊ and (g − 20 m/s)₊ | ERA5 | Encodes adaptation to local wind; cancels part of ERA5's local gust bias; makes geography event-specific [34,22,27,39] | Low | Med–High |
| 3 | **Duration and peak.** Hours with gust above 15/20/25 m/s and wind above 9/13 m/s over trailing 6/12/24/48 h; longest continuous run; max 4-h running mean; hours since hazard onset and peak; storm-to-date rain | ERA5, later RTMA/HRRR | The most consistent predictors across studies [16,12,22,23,19] | Low | High |
| 4 | **Clean the targets, as training weights only**, since targets and masks stay fixed under program.md rule 4. Mask storm-time runs of missing rows and one-interval zeros instead of reading them as 0. Flag counts identical for more than 4 days, peaks above moving average + 3 SD, and fractions above 1. Drop or down-weight 2014 (lowest coverage). Use year-specific denominators (MCC 2022, the 2024 file's per-county totals, EIA-861 growth) and state coverage ratios. Keep the modeled-versus-collected flag | EAGLE-I [1–3]; EIA-861 | Removes restorations and plateaus that no weather input can explain; removes county scale errors that a model could fit only by memorizing counties [1,5,7,8,11] | Low–Med | High that the artefacts exist, Med for the RMSE effect |
| 5 | **Temperature at customer elevation.** Move T and Td from ERA5 `z` to 3DEP elevation at housing units, using an hourly, bounded lapse rate from ERA5 1000–850 hPa. Compute wet bulb and a humidity-aware snow fraction. Features: customer share in 0–3 °C with snow fraction above ½; customer elevation minus `z` | ERA5 single and pressure levels; 3DEP; Census | Places precipitation phase by customer rather than by cell mean; direct precedent [52]; evidence [58–61]; elevation-difference feature [19] | Med | Med |
| 6 | **Load states.** Wet-snow sleeve: ΔM = P·R·Δt·γ, sticking min(1, 1/U), shedding outside the window. Radial freezing-rain ice from Jones' model, or 0.28 × freezing-rain liquid. Canopy snow load: saturating interception × evergreen canopy × LAI. Phase from ERA5 `ptype`, checked against HRRR CFRZR/CSNOW/CICEP (2014–) and FRZR (2018–) | ERA5 `sf`, `lsf`, `tp`, `ptype`; HRRR; NLCD; MODIS LAI | Snow density and freezing rain are top winter predictors [18]; physics in [52,54–57,64] | Med | Med |
| 7 | **Leaf state.** Canopy-weighted 8-day LAI climatology and anomaly; green-up and dormancy dates. Stress terms: LAI × W², and canopy × LAI × gust exceedance | MODIS MCD15A2H, MCD12Q2; USFS canopy | LAI gave +2–9% [12]; skill varies with leaf season [16,17]; LAI × wind² [19] | Low–Med | Med |
| 8 | **Rain and root-zone wetness.** Stage IV hourly rain rate and 24/72-h totals. Antecedent 7/30-day totals or SPI-1/3/12. Soil layers `swvl2`–`swvl3` (7–100 cm) next to `swvl1`. Wet wind = gust exceedance × root-zone saturation | Stage IV; ERA5 / ERA5-Land | Max rain rate was 2nd most important with Stage IV [19]; heavy-rain storms missed without a rain input [22]; soil and rain inputs in [16,24] | Med | Med |
| 9 | **Convective proxies.** Shear over 0–6 km and 850–950 hPa; WMAXSHEAR; share of convective precipitation. Customer-weighted neighbourhood maxima within 50 km of gust, rain rate and WMAXSHEAR | ERA5 pressure levels | Targets ERA5's weakest regime [38–41]; neighbourhood maxima address misplaced storms [12] | Low–Med | Med–Low |
| 10 | **Line-proxy exposure, as interaction weights only.** Canopy % within 30–60 m of TIGER/OSM roads; road km per customer; evergreen share; optionally canopy height | USFS canopy; TIGER/OSM; GEDI | Standard in utility studies [16,19,28,29]; roads are the public proxy [10,31]; expect little as a main effect [23] | Med | Med as interaction, Low as main effect |
| 11 | **Storm Events only for event typing and evaluation strata.** Use real-time warning flags from VTEC instead | Storm Events; IEM VTEC | Avoids target leakage and reporting artefacts [43,44]; warning-to-outage mapping in [8] | Low | Med–High |
| 12 | **Finer gust inputs.** First RTMA hourly GUST/WIND/TMP/DPT. Then HRRR `f01` hourly-max wind, GUST, LTNG and precipitation types. Add version flags and per-cell quantile matching to ERA5 | Earth Engine RTMA; AWS HRRR with Herbie | Better local gusts and convective timing [19,20,47,48]; modest thunderstorm gain [19]; HRRR biases [51] | High (HRRR ~0.5–0.8 TB) | Med–Low |

## References

[1] Brelsford C, Tennille S, Myers A, et al. (2024). A dataset of recorded electricity outages by United States county 2014–2022. *Scientific Data* 11:271. doi:10.1038/s41597-024-03095-5 (full text). Data: doi:10.6084/m9.figshare.24237376
[2] Tansakul V, Myers A, Tennille S, et al. (2025). EAGLE-I Power Outage Data 2024 [dataset]. ORNL. doi:10.13139/OLCF/2500278 (the per-county total's derivation year is UNVERIFIED)
[3] Denman M, Huihui J, Tennille S, et al. (2026). EAGLE-I County Customer Dataset Fall 2025 [dataset]. ORNL. doi:10.13139/ornlnccs/3022751
[4] Moehl J, Tennille S, Denman M, Myers A (2025). Modeling Electric Utility County Customers for Situational Awareness. ORNL technical report. doi:10.2172/3009475
[5] Do V, McBrien H, Flores NM, et al. (2023). Spatiotemporal distribution of power outages with climate events and social vulnerability in the USA. *Nature Communications* 14:2470. doi:10.1038/s41467-023-38084-6 (full text)
[6] Ganz SC, Duan C, Ji C (2023). Socioeconomic vulnerability and differential impact of severe weather-induced power outages. *PNAS Nexus* 2(10):pgad295. doi:10.1093/pnasnexus/pgad295
[7] Flores NM, Northrop AJ, et al. (2024). Powerless in the storm: severe weather-driven power outages in New York State, 2017–2020. *PLOS Climate* 3(5):e0000364. doi:10.1371/journal.pclm.0000364
[8] Lee SM, Chinthavali S, Bhusal N, et al. (2024). Quantifying the power system resilience of the US power grid through weather and power outage data mapping. *IEEE Access* 12:5237–5255. doi:10.1109/ACCESS.2023.3347129 (full text)
[9] Jacobs J, Piburn J, Myers A (2026). Hierarchical spline-based Bayesian beta-binomial regression for estimating time-varying risk in power outages. arXiv:2608.12560 [preprint]
[10] Taylor WO, Cerrai D, Wanik D, Koukoula M, Anagnostou EN (2023). Community power outage prediction modeling for the Eastern United States. *Energy Reports* 10:4148–4169. doi:10.1016/j.egyr.2023.10.073 (abstract and reference list only)
[11] Essus Y, Vatsavai RR, Rachunok B (2026). Data leakage inflates generalizability of power outage prediction models. arXiv:2608.24665 [preprint]
[12] Cerrai D, Wanik DW, Bhuiyan MAE, et al. (2019). Predicting storm outages through new representations of weather and vegetation. *IEEE Access* 7:29639–29654. doi:10.1109/ACCESS.2019.2902558 (full text)
[13] Liu H, Davidson RA, Apanasovich TV (2007). Statistical forecasting of electric power restoration times in hurricanes and ice storms. *IEEE Trans. Power Systems* 22(4):2270–2279. doi:10.1109/TPWRS.2007.907587
[14] Duffey RB (2019). Power restoration prediction following extreme events and disasters. *Int. J. Disaster Risk Science* 10(1):134–148. doi:10.1007/s13753-018-0189-2
[15] Ji C, Wei Y, Mei H, et al. (2016). Large-scale data analysis of power grid resilience across multiple US service regions. *Nature Energy* 1:16052. doi:10.1038/nenergy.2016.52
[16] Wanik DW, Anagnostou EN, Hartman BM, Frediani MEB, Astitha M (2015). Storm outage modeling for an electric distribution network in Northeastern USA. *Natural Hazards* 79(2):1359–1384. doi:10.1007/s11069-015-1908-2 (full text, author copy)
[17] He J, Wanik DW, Hartman BM, et al. (2017). Nonparametric tree-based predictive modeling of storm outages on an electric distribution network. *Risk Analysis* 37(3):441–458. doi:10.1111/risa.12652
[18] Cerrai D, Koukoula M, Watson P, Anagnostou EN (2020). Outage prediction models for snow and ice storms. *Sustainable Energy, Grids and Networks* 21:100294. doi:10.1016/j.segan.2019.100294
[19] Watson PL, Koukoula M, Anagnostou E (2021). Influence of the characteristics of weather information in a thunderstorm-related power outage prediction system. *Forecasting* 3(3):541–560. doi:10.3390/forecast3030034 (full text)
[20] Alpay BA, Wanik D, Watson P, Cerrai D, Liang G, Anagnostou E (2020). Dynamic modeling of power outages caused by thunderstorms. *Forecasting* 2(2):151–162. doi:10.3390/forecast2020008 (full text)
[21] Yang F, Cerrai D, Anagnostou EN (2021). The effect of lead-time weather forecast uncertainty on outage prediction modeling. *Forecasting* 3(3):501–516. doi:10.3390/forecast3030031
[22] Guikema SD, Nateghi R, Quiring SM, Staid A, Reilly AC, Gao M (2014). Predicting hurricane power outages to support storm response planning. *IEEE Access* 2:1364–1373. doi:10.1109/ACCESS.2014.2365716 (full text)
[23] Quiring SM, Zhu L, Guikema SD (2011). Importance of soil and elevation characteristics for modeling hurricane-induced power outages. *Natural Hazards* 58(1):365–390. doi:10.1007/s11069-010-9672-9
[24] McRoberts DB, Quiring SM, Guikema SD (2018). Improving hurricane power outage prediction models through the inclusion of local environmental factors. *Risk Analysis* 38(12):2722–2737. doi:10.1111/risa.12728
[25] Shashaani S, Guikema SD, Zhai C, Pino JV, Quiring SM (2018). Multi-stage prediction for zero-inflated hurricane induced power outages. *IEEE Access* 6:62432–62449. doi:10.1109/ACCESS.2018.2877078
[26] Kabir E, Guikema SD, Quiring SM (2019). Predicting thunderstorm-induced power outages to support utility restoration. *IEEE Trans. Power Systems* 34(6):4370–4381. doi:10.1109/TPWRS.2019.2914214
[27] Tervo R, Láng I, Jung A, Mäkelä A (2021). Predicting power outages caused by extratropical storms. *Nat. Hazards Earth Syst. Sci.* 21(2):607–627. doi:10.5194/nhess-21-607-2021
[28] Wanik DW, Parent JR, Anagnostou EN, Hartman BM (2017). Using vegetation management and LiDAR-derived tree height data to improve outage predictions for electric utilities. *Electric Power Systems Research* 146:236–245. doi:10.1016/j.epsr.2017.01.039
[29] Wedagedara H, Witharana C, Fahey R, et al. (2023). Modeling the impact of local environmental variables on tree-related power outages along distribution powerlines. *Electric Power Systems Research* 221:109486. doi:10.1016/j.epsr.2023.109486
[30] D'Amico DF, Quiring SM, Maderia CM, McRoberts DB (2019). Improving the hurricane outage prediction model by including tree species. *Climate Risk Management* 25:100193. doi:10.1016/j.crm.2019.100193
[31] Boeing G (2017). OSMnx: new methods for acquiring, constructing, analyzing, and visualizing complex street networks. *Computers, Environment and Urban Systems* 65:126–139. doi:10.1016/j.compenvurbsys.2017.05.004
[32] Arderne C, Zorn C, Nicolas C, Koks EE (2020). Predictive mapping of the global power system using open data. *Scientific Data* 7:19. doi:10.1038/s41597-019-0347-4
[33] Zhai C, Chen TYJ, White A, Guikema SD (2021). Power outage prediction for natural hazards using synthetic power distribution systems. *Reliability Engineering & System Safety* 208:107348. doi:10.1016/j.ress.2020.107348
[34] Klawa M, Ulbrich U (2003). A model for the estimation of storm losses and the identification of severe winter storms in Germany. *Nat. Hazards Earth Syst. Sci.* 3(6):725–732. doi:10.5194/nhess-3-725-2003 (full text)
[35] Hsiang S (2016). Climate econometrics. *Annual Review of Resource Economics* 8:43–75. doi:10.1146/annurev-resource-100815-095343 (full text of NBER w22181)
[36] Hersbach H, Bell B, Berrisford P, et al. (2020). The ERA5 global reanalysis. *Q. J. R. Meteorol. Soc.* 146:1999–2049. doi:10.1002/qj.3803
[37] ECMWF/Copernicus. ERA5: data documentation (parameter tables for `10fg`, `ptype`, `z`, `sdor`, `swvl1-4`). https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation
[38] Bechtold P, Bidlot J-R (2009). Parametrization of convective gusts. *ECMWF Newsletter* 119:15–18. doi:10.21957/kfr42kfp8c
[39] Minola L, Zhang F, Azorin-Molina C, et al. (2020). Near-surface mean and gust wind speeds in ERA5 across Sweden: towards an improved gust parametrization. *Climate Dynamics* 55:887–907. doi:10.1007/s00382-020-05302-6
[40] Taszarek M, Pilguj N, Allen JT, et al. (2021). Comparison of convective parameters derived from ERA5 and MERRA-2 with rawinsonde data over Europe and North America. *J. Climate* 34(8):3211–3237. doi:10.1175/JCLI-D-20-0484.1
[41] Taszarek M, Allen JT, Púčik T, Hoogewind KA, Brooks HE (2020). Severe convective storms across Europe and the United States. Part II: ERA5 environments associated with lightning, large hail, severe wind, and tornadoes. *J. Climate* 33(23):10263–10286. doi:10.1175/JCLI-D-20-0346.1
[42] ECMWF. Forecast User Guide §8.1.10, Types of precipitation. https://confluence.ecmwf.int/display/FUG/Section+8.1.10+Types+of+Precipitation+-+charts+and+diagrams
[43] Trapp RJ, Wheatley DM, Atkins NT, Przybylinski RW, Wolf R (2006). Buyer beware: some words of caution on the use of severe wind reports in postevent assessment and research. *Weather and Forecasting* 21(3):408–415. doi:10.1175/WAF925.1
[44] Edwards R, Allen JT, Carbin GW (2018). Reliability and climatological impacts of convective wind estimations. *J. Appl. Meteor. Climatol.* 57(8):1825–1845. doi:10.1175/JAMC-D-17-0306.1
[45] De Pondeca MSFV, Manikin GS, DiMego G, et al. (2011). The Real-Time Mesoscale Analysis at NOAA's National Centers for Environmental Prediction: current status and development. *Weather and Forecasting* 26(5):593–612. doi:10.1175/WAF-D-10-05037.1. Hourly 2.5-km archive, 2011–present: https://developers.google.com/earth-engine/datasets/catalog/NOAA_NWS_RTMA
[46] Nelson BR, Prat OP, Seo D-J, Habib E (2016). Assessment and implications of NCEP Stage IV quantitative precipitation estimates for product intercomparisons. *Weather and Forecasting* 31(2):371–394. doi:10.1175/WAF-D-14-00112.1. Data: doi:10.5065/D6PG1QDD
[47] Dowell DC, Alexander CR, James EP, et al. (2022). The High-Resolution Rapid Refresh (HRRR): an hourly updating convection-allowing forecast model. Part I. *Weather and Forecasting* 37(8):1371–1395. doi:10.1175/WAF-D-21-0151.1. Version dates: https://rapidrefresh.noaa.gov/hrrr/
[48] James EP, Alexander CR, Dowell DC, et al. (2022). The High-Resolution Rapid Refresh (HRRR): an hourly updating convection-allowing forecast model. Part II: forecast performance. *Weather and Forecasting* 37(8):1397–1417. doi:10.1175/WAF-D-21-0130.1
[49] Blaylock BK, Horel JD, Liston ST (2017). Cloud archiving and data mining of High-Resolution Rapid Refresh forecast model output. *Computers & Geosciences* 109:43–50. doi:10.1016/j.cageo.2017.08.005. AWS archive: https://registry.opendata.aws/noaa-hrrr-pds/
[50] Blaylock BK. Herbie: retrieve numerical weather prediction model data [software]. doi:10.5281/zenodo.4567540
[51] Fovell RG, Gallagher A (2020). Boundary layer and surface verification of the High-Resolution Rapid Refresh, version 3. *Weather and Forecasting* 35(6):2255–2278. doi:10.1175/WAF-D-20-0101.1
[52] Bonelli P, Lacavalla M, Marcacci P, Mariani G, Stella G (2011). Wet snow hazard for power lines: a forecast and alert system applied in Italy. *Nat. Hazards Earth Syst. Sci.* 11(9):2419–2431. doi:10.5194/nhess-11-2419-2011 (full text)
[53] ISO 12494:2017, Atmospheric icing of structures. https://www.iso.org/standard/72443.html (paywalled). The ranges were confirmed only through a secondary reproduction of its Table 2 (Thorsson P, Modelling of atmospheric icing: an introduction essay, DiVA diva2:1038705, not peer reviewed)
[54] Nygaard BEK, Ágústsson H, Somfalvi-Tóth K (2013). Modeling wet snow accretion on power lines: improvements to previous methods using 50 years of observations. *J. Appl. Meteor. Climatol.* 52(10):2189–2203. doi:10.1175/JAMC-D-12-0332.1
[55] Makkonen L (2000). Models for the growth of rime, glaze, icicles and wet snow on structures. *Phil. Trans. R. Soc. A* 358(1776):2913–2939. doi:10.1098/rsta.2000.0690
[56] Jones KF (1998). A simple model for freezing rain ice loads. *Atmospheric Research* 46(1–2):87–97. doi:10.1016/S0169-8095(97)00053-7
[57] Sanders KJ, Barjenbruch BL (2016). Analysis of ice-to-liquid ratios during freezing rain and the development of an ice accumulation model. *Weather and Forecasting* 31(4):1041–1060. doi:10.1175/WAF-D-15-0118.1
[58] Jennings KS, Winchell TS, Livneh B, Molotch NP (2018). Spatial variation of the rain–snow temperature threshold across the Northern Hemisphere. *Nature Communications* 9:1148. doi:10.1038/s41467-018-03629-7
[59] Stull R (2011). Wet-bulb temperature from relative humidity and air temperature. *J. Appl. Meteor. Climatol.* 50(11):2267–2269. doi:10.1175/JAMC-D-11-0143.1
[60] Minder JR, Mote PW, Lundquist JD (2010). Surface temperature lapse rates over complex terrain: lessons from the Cascade Mountains. *J. Geophys. Res.* 115:D14122. doi:10.1029/2009JD013493
[61] Dutra E, Muñoz-Sabater J, Boussetta S, et al. (2020). Environmental lapse rate for high-resolution land surface downscaling: an application to ERA5. *Earth and Space Science* 7(5):e2019EA000984. doi:10.1029/2019EA000984
[62] Muñoz-Sabater J, Dutra E, Agustí-Panareda A, et al. (2021). ERA5-Land: a state-of-the-art global reanalysis dataset for land applications. *Earth Syst. Sci. Data* 13:4349–4383. doi:10.5194/essd-13-4349-2021
[63] Jiménez PA, Dudhia J (2012). Improving the representation of resolved and unresolved topographic effects on surface wind in the WRF model. *J. Appl. Meteor. Climatol.* 51(2):300–316. doi:10.1175/JAMC-D-11-084.1
[64] Hedstrom N, Pomeroy JW (1998). Measurements and modelling of snow interception in the boreal forest. *Hydrological Processes* 12(10–11):1611–1625. doi:10.1002/(SICI)1099-1085(199808/09)12:10/11<1611::AID-HYP684>3.0.CO;2-4

Public data sources named in section 6 (URLs checked 2026-09-25):
- MODIS LAI MCD15A2H v061 (Myneni, Knyazikhin, Park 2021): doi:10.5067/MODIS/MCD15A2H.061
- MODIS phenology MCD12Q2 v061 (Friedl, Gray, Sulla-Menashe 2022): doi:10.5067/MODIS/MCD12Q2.061
- USFS/NLCD tree canopy: https://www.mrlc.gov/data/type/tree-canopy. Method: Coulston JW et al. (2012), *PE&RS* 78(7):715–727, doi:10.14358/PERS.78.7.715
- Census 2020 P.L. 94-171 block counts: https://www.census.gov/programs-surveys/decennial-census/about/rdo/summary-files.html
- TIGER/Line roads: https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html
- Microsoft US building footprints: https://github.com/microsoft/USBuildingFootprints
- HIFLD: https://hifld-geoplatform.hub.arcgis.com/
- IEM NWS warning archive (VTEC): https://mesonet.agron.iastate.edu/request/gis/watchwarn.phtml
