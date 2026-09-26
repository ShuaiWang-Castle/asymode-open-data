# Warnings, event selection, exclusions, data gaps and allocation for the public panel

2026-09-26. Literature and data note for DATASET_DESIGN v0 (`geo_weather_20260924`). "(full text)" = methods read;
"our check" = a listing, HEAD or byte-range request made on 2026-09-26 (no bulk download), or a count on the on-disk
copies of the public files; UNVERIFIED = not confirmed. Numbers from papers describe their settings, not ours.

## Bottom line

1. The IEM VTEC archive is an outcome-independent frame for 2015-2025 (3.2 GB), but warnings are local-criteria
   decisions on zones or polygons whose coding changed in 2017, 2021 and 2025. The one EAGLE-I precedent [Lee24] works
   per county and counts any polygon overlap as exposure.
2. Most outage datasets select events on the outcome; event hold-outs remove much of the apparent skill [Ess26].
3. Public records identify operator-initiated outages: PSPS by circuit and county in California, load shedding by
   utility and hour.
4. EAGLE-I files differ by year (zeros, blanks, columns, coverage); the public 2022 file is complete. HRRR hourly
   files are 84-98% complete in 2014-2018 (our sample).
5. Equal allocation per regime is sound if the frame is enumerated first and inclusion probabilities are kept; power
   comes from the number of episodes.

## 1. Episodes from NWS warnings

**What exists.** The Iowa Environmental Mesonet (IEM) holds the only processed, event-level archive; NCEI keeps raw
text without parsed events [IEMv].

| Item | Detail |
|---|---|
| Coverage | CONUS and territories. VTEC rolled out from 2005 and was largely complete by 2008; all types from 12 Nov 2005; tornado and severe-thunderstorm warnings by county back to 1986 (back-computed); official storm-based polygons from 1 Oct 2007 [IEMv, IEMd] |
| Files | `pickup/wwa/{year}_all.zip`: 248-335 MB per year, 3.2 GB for 2015-2025, each a 1.1-1.2 GB shapefile plus an 81-94 MB attribute CSV; `_tsmf.zip` (TO/SV/MA/FF) 36-48 MB; `_tsmf_sbw.zip` (polygons) 5-6.5 MB (our check). The `watchwarn.py` service filters by state, WFO, phenomenon and time; unfiltered requests are capped at one year [IEMc] |
| Fields | ISSUED, EXPIRED, INIT_ISS, INIT_EXP, PHENOM, SIG, GTYPE (P polygon, C county/zone), ETN, STATUS, NWS_UGC, AREA_KM2, H-VTEC fields, POLYBEGIN/END, wind/hail/tornado/damage tags, PRODUCT_ID (our check of the 2015 CSV; definitions in [IEMv]) |
| Codes | SV, TO; HW, WI, EW; WS, WW, IS, BZ, LE; TR, HU, SS; FF, FA, FL; EH (to 2025), XH, HT; FW (Red Flag). SIG: W, A, Y, S [VTEC] |
| Geography | Time-versioned UGC geometries since 2007 (`api/1/nws/ugcs.json?valid=`) [IEMa]; zone-county correlation file, one row per county-zone pair [NWSzc] |

**Use in outage studies.** Lee et al. [Lee24] (full text; EAGLE-I 2018-2022):
- kept warnings and advisories, replacing each polygon by every county it touches;
- keyed events by ISSUED+EXPIRED+PHENOM+SIG+state, dropping the 1.2% of rows with negative duration;
- grouped overlapping warnings per county, typed by the longest member, with outages attributed from 3 h before to
  48 h after group expiry.

VTEC alerts also serve as events [Lee23] and heat flags [Sak25]. A public EAGLE-I x VTEC table (2015-2024) has
columns (new_event_no, evDur_Hr, Matched_Events) suggesting outage events come first [Sak26; our reading]. We found no
study that clusters warnings into multi-county episodes.

**Pitfalls.**

| Pitfall | Evidence | Handling |
|---|---|---|
| Zones are not counties | Winter and non-precipitation (wind, heat) products use zone UGCs [N513, N515]; UGCs are not FIPS codes [IEMv]; a split county has one row per zone [NWSzc]; zones change over time [IEMa] | Use the geometry valid at issuance; expose a county by the customer share inside warned areas, not by any overlap |
| Last-snapshot records | An event upgraded before its start ends before it begins (a negative duration); ETNs are sometimes not unique; polygons shrink in updates [IEMv, IEMa] | Start at INIT_ISS; keep upgraded events ([Lee24] dropped negative durations as errors); key on WFO+PHENOM+SIG+ETN+year |
| Local criteria | Winter, ice, wind and heat thresholds are set locally (high wind typically 40 mph sustained for 1 h, or 58 mph for any duration); lower, impact-based issuance is allowed; winter warnings should not exceed 48 h, so long storms are split [N513, N515] | A warning is a local-climate exceedance plus a decision: stratify by region, merge consecutive warnings |
| Practice changes | Oct 2017 winter consolidation [IWX]; 2012 shorter tornado warnings [BC18]; Aug 2021 damage tags [SVR21]; Mar 2025 EH to XH [SCN]. In about 7,000 polygon rows per year, 74% of 2015 SV.W polygons carry a wind tag, against 100% in 2019 and 2024 (our check) | Crosswalk the codes; keep period strata; no tag features before 2019 |
| Overlap and misses | Tropical cyclones bring TR/HU with TO, FF and FL. Sub-severe storms (40-57 mph) get a Special Weather Statement [N517], which has no VTEC code [VTEC] | Type by parent system; measure what falls outside the frame |

## 2. How published outage datasets choose events and units

| Study (outage data) | Events, units | Selection | Non-damaging cases | Split |
|---|---|---|---|---|
| Wanik 2015 [Wan15] (utility, CT) | 89 storms 2005-2014; 2-km cells, 149 towns | storms affecting the territory | zero cells in storms | repeated random holdout |
| Cerrai 2019 [Cer19] (utility, CT) | 120 storms (76 extratropical, 44 convective) | days above the 95th percentile of daily outages, plus weather-classified storms below it | weather-selected storms | leave-one-storm-out |
| Watson 2021 [Wat21] (5 New England utilities) | 372 thunderstorms; 5,774 cells, 96.3% zeros | thunderstorms in station reports | zero cells | leave-one-date-out |
| Tervo 2021 [Ter21] (Finland) | 24,542 ERA5 gust objects (15 m/s) | hazard objects | 22,179 objects without outages | random vs 1-year block |
| Taylor 2023 [Tay23] (17 Eastern states) | 15,872 county x Storm Events episodes | Storm Events | none stated | chronological 10-fold |
| Lee 2023 [Lee23] (EAGLE-I) | 15,272 / 4,742 / 503 alerts (TX / MI / HI) | every NWS warning or advisory | alerts without outages | chronological 80/20 |
| Do 2023 [Do23] (PowerOutage.us) | 2,447 counties, 2018-2020 | >= 0.1% of customers out | event-free county-days | descriptive |
| Ganz 2023 [Gan23] (PowerOutage.us) | 8 hurricanes, 588 counties | county kept only if >= 5% of customers lost power | none | none |
| Essus 2026 [Ess26] (PowerOutage.us) | 28 dates, 13 East Coast states | >= 40 counties above a 1-year moving average + 3 SD | none | random, leave-one-state-out, leave-one-event-out |

- **Outcome selection is the norm**, by date [Cer19, Ess26], county [Gan23] or event [Do23, Sak25]; Essus et al.
  call hand-picked outage dates the prevailing practice. UConn models over-predict low-impact and under-predict
  high-impact storms [Yan20].
- **Storm Events is damage-based by rule.** It documents storms with "sufficient intensity to cause loss of life,
  injuries, significant property damage" or disruption to commerce, and estimates gusts from toppled trees and power
  lines [N1605]; reports misstate extent [Tra06, Edw18]. County x episode designs [Tay23] inherit this.
- **Hazard-first designs with non-damaging cases exist** [Ter21, Lee23, Cer19]: the model for our frame.
- **The split decides the result.** Random splits reuse correlated cells of one storm [Cer19]. In [Ess26],
  random-split R² of 0.44-0.49 (relative) and 0.32-0.36 (absolute) falls to about the null under state or event
  hold-outs, except hurricanes on the relative target.

## 3. Operator-initiated outages: public records

| Record | Years | Detail | Source |
|---|---|---|---|
| CPUC PSPS post-event reports, all California IOUs | 2017- | Circuit rows: de-energization, all-clear and restoration times, county, customers; GIS in recent reports. Resolution ESRB-8 (12 Jul 2018) extended SDG&E's 2012 duty to all IOUs: report within 10 business days | [CPUCr, ESRB8] |
| CPUC post-season data workbooks; D.21-06-034 post-season reports | 2021-2025 | One row per county per event (first off, last on) and per circuit segment | [CPUCp, D2106] |
| CPUC event table (ArcGIS) | 2018-2026 | 165 events: IOU, de-energization status, start and restoration dates, customers, number (not names) of counties (our check) | [CPUCt] |
| Oregon PUC docket UM 2268; other western states | 2022-2025 (Oregon) | Annual PSPS reports in Oregon; elsewhere commission pages and press releases (e.g. Colorado's first large-scale PSPS, 6-7 Apr 2024, six counties) | [OPUC, CBS24] |
| DOE-417 annual summaries | 2000-2023 (2024-2025 not found) | Start and restoration times, area (state, often counties), alert criterion (e.g. firm load shed >= 100 MW), event type, MW, customers | [OE417] |
| FERC-NERC and ISO reports | events | Feb 2021: ERCOT 20,000 MW peak over 70.5 h, SPP 2,718 MW, MISO South 700 MW. Dec 23-24, 2022: over 5,400 MW (TVA over 3,000, Duke Carolinas 1,000, Duke Progress 961, LG&E/KU 317, Dominion SC 95, Santee Cooper 86). CAISO 14-15 Aug 2020: 491,600 customers on 14 Aug. 2025: SWEPCO 140 MW (26 Apr), Entergy/CLECO about 600 MW (25 May) | [ELL23, ERC21, CAI21, SPP25, MISO25] |

- California IOU shutoffs can be excluded by circuit or county window; publicly owned utilities and co-operatives are
  not covered. Elsewhere PSPS is mostly 2020 or later and must be compiled by hand.
- DOE-417 event types are unreliable: the 2021 ERCOT load-shed rows (e.g. Harris County, 1.39 M customers, 15-18 Feb)
  are typed Severe Weather (our check). Filter on the alert criterion; sheds under 100 MW need not be reported.
- ISO-directed sheds are pro rata without county footprints, so whole footprints must go for the window (all ERCOT
  counties 15-18 Feb 2021; the shedding utilities' territories 23-24 Dec 2022), although these windows also hold
  storm damage.
- EAGLE-I has no cause field [Bre24]; we found no public county-level record of planned outages. EAGLE-I is in UTC,
  the reports in local time.

## 4. Data quality that affects selection

**EAGLE-I** (figshare v4, 2026-02-25, 2014-2025 [Bre24]; our check).

| Year(s) | Finding | Consequence |
|---|---|---|
| 2014 | Starts 2014-11-01 04:00 UTC; 1,072 counties had no data in 2014 against 182 in 2022 [Bre24] | Keep 2014 out |
| 2014-2021 | Every file ends at Dec 31 00:00 UTC | Dec 31 is a gap, not zeros |
| 2017-2021, 2025 | Rows with a blank count: 0.16 M in 2017, 1.1-2.1 M per year in 2018-2021, 2.0 M in 2025 | Blank = unobserved |
| 2022-2025 | Explicit zero rows (3.7-5.0% of rows in 2023-2025), although the paper says zeros are not stored [Bre24] | Define "observed hour" identically every year, or the >= 90% gate loosens after 2022 |
| 2022 | The public file (1,187,596,176 bytes, md5 0cd04f23...) runs to 2022-12-31 23:45. The on-disk copy (1,026,691,516 bytes) stops at 2022-11-12 18:30 | Replace it; this restores Winter Storm Elliott (also a load-shed event) |
| 2023 / 2024 | Count column named `sum` (2023); extra `total_customers` column and rows not time-sorted (2024) | Map columns, sort |

**Coverage** (our check of the public files).
- `coverage_history.csv` covers only 2018-2022, with each state's minimum and maximum coverage per year [Bre24].
- Below 0.8 at maximum: 16 states in 2018 (e.g. MT 0.16, NE 0.45, SD 0.46, ND 0.51); 7 in 2022 (NE 0.48, KS 0.59,
  ND 0.66, WY 0.71, TN 0.75, MT 0.76, MS 0.77); TX 0.63 in 2020.
- Within-year drops: NH 0.17-0.88 (2018), IA 0.44-0.89 (2019), KS 0.30-0.80 (2021).
- The Data Quality Index is lowest for FEMA Region 9 in 2018 (39), Region 7 in 2019 (50) and Region 1 in 2020 (62).
- A nearest-year maximum-coverage gate therefore passes episodes inside a drop, and has no data for 2015-2017 or
  2023-2025.

**HRRR** (AWS archive from 2014-07-30; operational v1 2014-09-30, v2 2016-08-23, v3 2018-07-12, v4 2020-12-02 [HRR]).
Every day folder to 2019 exists except 2016-03-19. The table gives the share of hourly `wrfsfcf01` files (the source
of hourly maxima) present on two sampled days per month (our check):

| 2014 (Aug-Dec) | 2015 | 2016 | 2017 | 2018 | 2019 |
|---|---|---|---|---|---|
| 95.8% | 93.6% | 84.4% | 97.6% | 97.2% | 100% |

Some days are empty (2015-06-20, 2016-03-20). The Google Cloud mirror has the same gaps on five sampled days, and the
AWS registry page mentions none. Gaps reach into 2018.

## 5. Allocation and weights

- **Which allocation.** Proportional allocation is self-weighting for frame totals; Neyman allocation minimises the
  variance of a frame-wide estimate [Ney34]; equal allocation maximises the precision of between-regime comparisons.
  Any is fine if inclusion probabilities are kept: Horvitz-Thompson weights undo the allocation [HT52], as they correct
  case-control samples of rare events [KZ01].
- **What to report.**
  - (a) Regime-balanced: the unweighted mean of per-regime relative skill (vs the all-zero forecast and the host),
    since pooling regimes with different climatologies can manufacture skill [HJ06].
  - (b) Frame-weighted: pooled metrics with weights N_h/n_h, times 1/pi for controls.
  - (c) Per-regime and worst-regime values [Sag20, Koh21].
- **No outcome-conditioned scores.** Scoring only damaging cases penalises skilful forecasts; weight extremes through
  forecasts or covariates instead, e.g. threshold-weighted scores [Ler17, GR11].
- **Power.** Six episodes per regime are six clusters; cluster-robust tests over-reject with few, unequal clusters
  [CGM08, MW17], and the design effect grows with counties per episode [CM15].

## 6. Recommendations for DATASET_DESIGN v1

1. **Enumerate the frame before sampling.** Build every 2015-2025 warning episode, store N_h per stratum, then draw
   the confirmation and development sets with recorded inclusion probabilities. Reason: weights and both metrics need
   N_h fixed before any outcome is read [HT52].
2. **Write the episode rule down.** Regime codes with SIG = W; start at INIT_ISS; keep upgrades; key on
   WFO+PHENOM+SIG+ETN+year; cluster by adjacency of warned areas and a fixed gap (e.g. 12 h); prefix 72 h before the
   first INIT_ISS; crosswalk EH/XH and the 2017 winter codes. Reason: section 1 [IEMv, SCN, IWX].
3. **Expose counties by customer share.** Use the block-population share inside polygons (SV/TO/FF) or zones (others),
   with UGC geometry valid at issuance, and keep the share. Report frame coverage (the share of EAGLE-I county-hours
   above a fixed outage fraction lying inside the frame, by regime and region) as a diagnostic, never a selection
   rule. Reason: zones are not counties, and sub-severe storms fall outside VTEC [NWSzc, N517].
4. **Type by parent system.** Tropical if any TR/HU/SS/EW, else winter, synoptic wind, convective, flood, heat; store
   all labels. Reason: longest-member typing [Lee24] would label a hurricane by its river flood.
5. **Two kinds of controls.** Advisory-only (WI.Y, WW.Y, HT.Y) and no-product counties, each with its own inclusion
   probability. Reason: advisories (e.g. 30-39 mph sustained wind) and sub-severe statements (40-57 mph gusts) cover
   the band just below warning criteria [N515, N517], where the damage threshold has to be learned (our reasoning).
6. **Exclude operator actions from the records, per utility-county window.** Take PSPS from CPUC files and load sheds
   from FERC-NERC, ISO and DOE-417 reports (filtered on the alert criterion). In the West, check every HW and Red Flag
   episode against PSPS lists. Log every exclusion. Reason: section 3.
7. **Fix the EAGLE-I inputs first.** Use the v4 2022 file; harmonise blanks and zeros; treat Dec 31 (2014-2021) as
   missing. Replace the coverage gate with an in-data check (reporting counties in the 30 days before the episode
   against the county customer model), using minimum rather than maximum coverage where it exists. Reason: section 4.
8. **Gate HRRR hours like outage hours**, and report HRRR results separately for 2019-2025 and earlier years. Reason:
   section 4.
9. **Buy episodes, not counties.** Cap counties per episode at about 150 (weighted draw). This affords about 60
   development and 30 confirmation episodes (10 and 5 per regime) within the 9,000 + 4,500 budget. Make per-regime
   confirmatory claims only where 8 effective episodes exist. Reason: [CGM08, MW17] and the design's own threshold.
10. **Report both metrics.** Headline the regime-balanced skill, with the frame-weighted value and the worst regime
    beside it; never a damaging-only subset [HJ06, Ler17].

## References

Warnings and NWS practice
- [IEMv] Iowa Environmental Mesonet. NWS VTEC archives: documentation and FAQ (updated 2025-07-12). https://mesonet.agron.iastate.edu/info/datasets/vtec.html
- [IEMd] IEM watch/warning download page; annual files. https://mesonet.agron.iastate.edu/request/gis/watchwarn.phtml ; https://mesonet.agron.iastate.edu/pickup/wwa/
- [IEMc] IEM watchwarn.py documentation and changelog. https://mesonet.agron.iastate.edu/cgi-bin/request/gis/watchwarn.py?help
- [IEMa] IEM API (/nws/ugcs, /vtec/county_zone, /vtec/sbw_interval). https://mesonet.agron.iastate.edu/api/1/docs
- [VTEC] NWS VTEC explanation v9.0 (Mar 2025). https://www.weather.gov/media/vtec/VTEC_explanation_ver9.pdf ; NWSI 10-1703. https://www.weather.gov/media/directives/010_pdfs/pd01017003curr.pdf
- [NWSzc] NWS zone-county correlation file. https://www.weather.gov/gis/ZoneCounty
- [N513] NWSI 10-513, winter weather products (2024-11-06). https://www.weather.gov/media/directives/010_pdfs/pd01005013curr.pdf
- [N515] NWSI 10-515, non-precipitation products (2025-05-16). https://www.weather.gov/media/directives/010_pdfs/pd01005015curr.pdf
- [N517] NWSI 10-517, multi-purpose products incl. Special Weather Statements (2022-08-01). https://www.weather.gov/media/directives/010_pdfs/pd01005017curr.pdf
- [N1605] NWSI 10-1605, Storm Data preparation (2021-07-26). https://www.weather.gov/media/directives/010_pdfs/pd01016005curr.pdf
- [IWX] NWS Northern Indiana, 2017-2018 winter hazard simplification. https://www.weather.gov/iwx/HazSimpWinterWx
- [SVR21] NWS, damage-threat tags for severe thunderstorm warnings from 2021-08-02. https://www.weather.gov/news/072221-svr-wea
- [SCN] NWS Service Change Notice 24-88 (EH to XH, effective 2025-03-04). https://www.weather.gov/media/tbw/heat/scn24-88_heat_haz_simp.pdf
- [BC18] Brooks HE, Correia J (2018). Long-term performance metrics for NWS tornado warnings. Wea. Forecasting 33:1501-1511. doi:10.1175/WAF-D-18-0120.1
- [Tra06] Trapp RJ et al. (2006). Wea. Forecasting 21:408-415. doi:10.1175/WAF925.1 ; [Edw18] Edwards R et al. (2018). J. Appl. Meteor. Climatol. 57:1825-1845. doi:10.1175/JAMC-D-17-0306.1

Outage data and studies
- [Bre24] Brelsford C et al. (2024). Scientific Data 11:271. doi:10.1038/s41597-024-03095-5 (full text). Data v4 (2014-2025): doi:10.6084/m9.figshare.24237376
- [Lee24] Lee SM et al. (2024). IEEE Access 12:5237-5255. doi:10.1109/ACCESS.2023.3347129 (full text)
- [Lee23] Lee S et al. (2023). Predicting power outage during extreme weather events with EAGLE-I and NWS datasets. IEEE IRI 2023:211-212. doi:10.1109/IRI58017.2023.00042 (full text: https://www.osti.gov/servlets/purl/1997746)
- [Sak25] Saki SA, Sofia G, Kar B, Anagnostou E (2025). Scientific Reports 15:30846. doi:10.1038/s41598-025-15065-x
- [Sak26] Saki SA, Sofia G, Kar B, Anagnostou E (2026). EAGLE-I and NWS VTEC harmonized dataset, v2 [dataset]. doi:10.5281/zenodo.22651795
- [Wan15] Wanik DW et al. (2015). Natural Hazards 79:1359-1384. doi:10.1007/s11069-015-1908-2 (full text)
- [Cer19] Cerrai D et al. (2019). IEEE Access 7:29639-29654. doi:10.1109/ACCESS.2019.2902558 (full text)
- [Yan20] Yang F et al. (2020). Sustainability 12:1525. doi:10.3390/su12041525 (full text)
- [Wat21] Watson PL et al. (2021). Forecasting 3:541-560. doi:10.3390/forecast3030034 (full text)
- [Ter21] Tervo R et al. (2021). NHESS 21:607-627. doi:10.5194/nhess-21-607-2021 (full text)
- [Tay23] Taylor WO et al. (2023). Energy Reports 10:4148-4169. doi:10.1016/j.egyr.2023.10.073 (abstract)
- [Do23] Do V et al. (2023). Nature Communications 14:2470. doi:10.1038/s41467-023-38084-6 (full text)
- [Gan23] Ganz SC, Duan C, Ji C (2023). PNAS Nexus 2:pgad295. doi:10.1093/pnasnexus/pgad295 (full text)
- [Ess26] Essus Y, Vatsavai RR, Rachunok B (2026). arXiv:2608.24665 [preprint] (full text)

Operator-initiated outages
- [CPUCr] CPUC PSPS post-event and post-season reports. https://www.cpuc.ca.gov/consumer-support/psps/utility-company-psps-reports-post-event-and-post-season
- [ESRB8] CPUC Resolution ESRB-8 (2018-07-12). https://docs.cpuc.ca.gov/PublishedDocs/Published/G000/M218/K186/218186823.PDF
- [D2106] CPUC D.21-06-034 (2021-06-24). https://www.cpuc.ca.gov/-/media/cpuc-website/divisions/safety-and-enforcement-division/documents/decision-phase-3-gl.pdf
- [CPUCp] CPUC post-season data workbook, e.g. https://www.cpuc.ca.gov/-/media/cpuc-website/divisions/safety-and-enforcement-division/reports/psdrs-post-season-data-reports/pge_psdr_2025_public.xlsx
- [CPUCt] CPUC PSPS event table. https://services2.arcgis.com/VofPZYDe2pLxSP5G/arcgis/rest/services/survey123_d4f10a2bbdba428283473319f537e4c1/FeatureServer/0
- [OPUC] Oregon PUC docket UM 2268. https://apps.puc.state.or.us/edockets/docket.asp?DocketID=23578 ; [CBS24] CBS Colorado, Xcel's first large-scale shutoffs (Apr 2024). https://cbsnews.com/colorado/news/xcel-energy-colorado-large-scale-power-outages-wind-storm-proactive-safety-shutoffs
- [OE417] DOE-417 annual summaries. https://doe417.pnnl.gov/ (e.g. https://doe417.pnnl.gov/summaries/2021_Annual_Summary.pdf)
- [ELL23] FERC-NERC (2023). Inquiry into bulk-power system operations during December 2022 Winter Storm Elliott (comparison table includes Feb 2021). https://www.nerc.com/globalassets/our-work/reports/event-reports/winter_storm_elliot_report.pdf
- [ERC21] ERCOT (2021-02-24). Review of February 2021 extreme cold weather event. https://www.ercot.com/files/docs/2021/02/24/2.2_REVISED_ERCOT_Presentation.pdf
- [CAI21] CAISO, CPUC, CEC (2021). Final root cause analysis, mid-August 2020 extreme heat wave. https://www.caiso.com/Documents/Final-Root-Cause-Analysis-Mid-August-2020-Extreme-Heat-Wave.pdf
- [SPP25] SPP summary of the 26 Apr 2025 Shreveport-area load shed. https://www.spp.org/documents/74283/spp's%20summary%20of%20the%20april%2026,%202025,%20shreveport-area%20load%20shed%20event.pdf ; [MISO25] MISO May 25 load shed event report (Aug 2025). https://cdn.misoenergy.org/May%2025%20Load%20Shed%20Event%20Report%20-%20August%202025711584.pdf

Weather data
- [HRR] AWS Open Data, NOAA HRRR. https://registry.opendata.aws/noaa-hrrr-pds/ (bucket noaa-hrrr-bdp-pds); versions: https://rapidrefresh.noaa.gov/hrrr/

Sampling and evaluation
- [Ney34] Neyman J (1934). JRSS 97:558-625. doi:10.2307/2342192
- [HT52] Horvitz DG, Thompson DJ (1952). JASA 47:663-685. doi:10.1080/01621459.1952.10483446
- [KZ01] King G, Zeng L (2001). Logistic regression in rare events data. Political Analysis 9:137-163. doi:10.1093/oxfordjournals.pan.a004868
- [HJ06] Hamill TM, Juras J (2006). QJRMS 132:2905-2923. doi:10.1256/qj.06.25
- [Ler17] Lerch S et al. (2017). Forecaster's dilemma: extreme events and forecast evaluation. Statistical Science 32(1). doi:10.1214/16-STS588
- [GR11] Gneiting T, Ranjan R (2011). JBES 29:411-422. doi:10.1198/jbes.2010.08110
- [Sag20] Sagawa S et al. (2020). ICLR. arXiv:1911.08731 ; [Koh21] Koh PW et al. (2021). WILDS. ICML. arXiv:2012.07421
- [CGM08] Cameron AC, Gelbach JB, Miller DL (2008). Rev. Econ. Stat. 90:414-427. doi:10.1162/rest.90.3.414
- [MW17] MacKinnon JG, Webb MD (2017). J. Appl. Econometrics 32:233-254. doi:10.1002/jae.2508
- [CM15] Cameron AC, Miller DL (2015). J. Human Resources 50:317-372. doi:10.3368/jhr.50.2.317
