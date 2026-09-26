# operator_exclusions.csv: operator-initiated outages, contiguous US, 2018-07-01 to 2025-12-31

Compiled 2026-09-26 for `DATASET_DESIGN.md` §7 and amendment 2 (M7). The file lists every rotating load shed and every
public-safety power shutoff (PSPS) that a public record documents, one row per county per record. It is the input of the
§7 rule: a county-event is excluded when [start_utc − 12 h, end_utc + 48 h] of a row for that county overlaps its
forecast window. The rule itself is applied downstream, not here.

Every row was decided from the record's type, footprint and time only. No outage-count data were opened (no EAGLE-I
file was read), and no record was chosen or dropped because of its size or the number of customers affected. Sizes
(MW, customers) appear in `notes` as reported, for information only.

## 1. Columns

| column | content |
|---|---|
| record_id | one id per record (a DOE-417 row, a report-documented shed of one operator, or one PSPS event); `LS-...` load shed, `PSPS-<state>-<operator>-<local start date>` PSPS (`PSPS-CA-PGE-...` is PG&E, `PSPS-OR-PGE-...` is Portland General Electric) |
| kind | `load_shed` or `psps` |
| operator | utility, ISO or balancing authority; DOE-417 summaries do not name the respondent, see §4.1 |
| eia_utility_ids | EIA-861 utility numbers (semicolon list) serving that county inside the record's footprint; empty when the operator is not named |
| state, county_fips | two-letter state and 5-digit county FIPS (2023 Census gazetteer) |
| footprint_rule | `county_list` (the source lists the county), `service_territory` (no county list: the operator's EIA-861 counties), `iso_footprint` (ISO- or BA-directed shed: EIA-861 counties of the member utilities), `state_list` (one DOE-417 row whose respondent could not be identified: every county of the listed state) |
| start_utc, end_utc | ISO 8601 UTC (`Z`); de-energisation or shed start and last restoration (or end of the shed order), see §3 |
| source_url, source_type | the document the row was read from |
| notes | area and times as written in the source, the local clock and zone used, sizes (information only), attribution and expansion steps, `UNVERIFIED: ...` when the row could not be confirmed from a document |
| flag_damage_shed | 1 when the source narrative attributes the shed to damage in the reporting utility's own system (M7: flagged and kept); 0 otherwise (all PSPS rows are 0) |

`state_list` is a fourth footprint value, beyond the three in the task. It is used for one row only (§4.1).

## 2. Sources

All files are under `data/raw/operator/`. Every download has a line (URL, UTC time, bytes, sha256) in
`data_provenance/downloads.jsonl`. Web pages that were only read are cited by URL in `notes`; the ones that decide a
row were saved under `data/raw/operator/web/`.

Load shedding:
* DOE-417 annual summaries 2018-2023 (`https://doe417.pnnl.gov/summaries/{YEAR}_Annual_Summary.xls`; 2024 and 2025
  are not published, which is a stated limit under M7) and the DOE-417 form instructions (`doe417/`).
* CAISO, CPUC and CEC, Final Root Cause Analysis of the mid-August 2020 heat wave (13 Jan 2021; `caiso/`). The CAISO
  host failed TLS verification from this machine, so the PDF was fetched with a web-fetch tool; its first pages were
  checked before it was filed.
* FERC, NERC and Regional Entity staff report on the February 2021 cold weather outages (Nov 2021) and the ERCOT review
  of 24 Feb 2021 (`nerc/`, `ercot/`).
* FERC-NERC inquiry into Winter Storm Elliott (Oct 2023; `nerc/`).
* SPP summary of the 26 Apr 2025 Shreveport-area load shed, SPP state-of-the-grid briefing (May 2025), SPP update to the
  Oklahoma Corporation Commission (June 2025) (`spp/`).
* MISO May 25 load shed event report (Aug 2025) and the Louisiana PSC presentation of the same event (`miso/`, `lpsc/`).
* PUCO staff report on the June 2022 AEP Ohio outages (`puco/`).
* NERC State of Reliability 2021, 2023, 2025 and 2026 technical assessments, and the NERC incident review
  "Load-Pocket Shoulder Season Challenges" (`nerc/`), used to search for other 2024-2025 sheds.
* Same-day utility statements and news reports, used only to name a respondent, an end time or a parish list
  (`web/`): KLTV 18 Aug 2019 (SWEPCO), Hello Woodlands 27 Aug 2020 and KFDM 28 Aug 2020 (Entergy Texas, Hurricane
  Laura), KYMA 5 Sep 2020 (IID), Entergy 17 Feb 2021, KSLA 2 Apr 2025 (SWEPCO), FOX8 25 and 26 May 2025 (Entergy), Cleco
  statement May 2025.
* EIA-861 2023 (`data/raw/eia/861/`, identical to `f8612023.zip`): Service_Territory, Sales_Ult_Cust (BA codes).

PSPS:
* California: CPUC PSPS event table (ArcGIS layer, 165 records; `cpuc/cpuc_psps_event_table.json`), CPUC post-event
  reports 2018-2021, post-season data workbooks (PSDR) 2021-2025, utility press releases where a report had no county
  table (`cpuc/`).
* Oregon: Oregon PUC docket UM 2268 PSPS reports and OPUC filings, PGE and PacifiCorp releases, EWEB and Lane Electric
  pages (`oregon_puc/`, `oregon_pge/`, `pacificorp/`).
* Other states: Colorado PUC proceedings 24M-0173E and 26M-0037E (`colorado_puc/`), Utah PSC and Rocky Mountain Power
  (`utah_psc/`, `rocky_mountain_power/`), PUC of Nevada PSPS dockets (`pucn/`), Idaho Power report filed in UM 2268,
  Avista and BPA wildfire plans (`avista/`, `bpa/`), PNM and Xcel (SPS) releases (`pnm/`). Place-name lookups used the
  USGS GNIS gazetteer (`gazetteer/`).

## 3. Times

* Output times are UTC. Local clock times were converted with the IANA zone of the place (Python `zoneinfo`, daylight
  time applied by date), or with a fixed offset where the source states one: the FERC-NERC February 2021 report gives
  Central (CST, UTC−6); the FERC-NERC Elliott report states that all its times are EST (UTC−5); the MISO May 2025
  report uses EST year-round (UTC−5, equal to CDT that day). An ambiguous fall-back hour takes the first (daylight)
  occurrence.
* DOE-417: the form lets the respondent tick a zone, which the summary drops. Times are read as the prevailing local
  time of the reported area. This is confirmed where a timeline exists: CAISO (row 2020-228 starts 18:25 = PG&E
  6:25 PM PDT in the RCA) and ERCOT (2021 rows start 01:20 and 01:23 = the FERC-NERC CST orders). Two MISO rows
  (2021-R097, 2021-R098) match the MISO timeline exactly one hour later, so they are read as EST. The TVA row
  (2022-R363) is read as Central (TVA spans two zones; ±1 h).
* Only dates given: local day boundaries 00:00 to 24:00 in each county's own zone (a county's zone is its state's,
  with the split states handled county by county: TX, KS, NE, SD, ND, TN, KY, ID, OR). Each such row says so in
  `notes`.
* Unknown ends: DOE-417 rows with restoration "Unknown" take the end from the matched ISO timeline (ERCOT cancelled its
  last shed orders 00:42 CST 18 Feb 2021; MISO and SPP restored all shed load by 10:10 and 10:07 CST 16 Feb 2021; MISO
  cancelled the Laura order around midnight 27-28 Aug 2020; PUCO gives 04:51 EDT 16 Jun 2022 for AEP Ohio; TVA released
  its last order 11:30 EST 24 Dec 2022), else the end of the local day. PSPS events with no published end take the end
  of the start day and are marked UNVERIFIED.
* PSPS end = last restoration of the county (or of the event) where the source gives it.

## 4. Load shedding

### 4.1 DOE-417 rows, 2018-2023

The filter is the alert criterion "Firm load shedding of 100 Megawatts or more implemented under emergency operational
policy" (with its truncated spelling), not the event type. 58 rows meet it: none in 2018, 3 in 2019, 12 in 2020, 28
in 2021, 8 in 2022 and 7 in 2023. The 7 rows of 2023 are LUMA Energy (Puerto Rico, December 2023), outside the
contiguous US, and are left out, so 51 DOE-417 records remain.

The summaries do not name the respondent. Footprints:
* The row lists counties: those counties (`county_list`, 16 records). The operator is named only where a document matches
  the row (for example SCE in 2020-R226 and R230, whose customer counts equal the SCE rows of the CAISO RCA tables).
* The row lists states only: the row is matched by its time, states and MW to a documented shed, and expanded to the
  EIA-861 counties of that operator or balancing area in the listed states. ERCOT rows (TRE, Feb 2021) go to the
  ERCOT footprint; MRO rows to the SPP footprint; MISO rows as in the table; 2019-R184 is SWEPCO (KLTV, 18 Aug 2019);
  the Hurricane Laura rows (27 Aug 2020) are the MISO-directed shed in the Entergy Texas area (MISO balancing-area
  utilities in Texas); 2022-R166 is AEP Ohio (PUCO); 2022-R363 is TVA; 2022-R374 and R375 are Duke Energy Progress and
  Duke Energy Carolinas (FERC-NERC MW and times). Where the time falls in both the SPP and the MISO sheds of
  16 Feb 2021 (R091, R107, R110, R119), both balancing areas in the listed states are used.
* 2022-R152 (California, 5 Jun 2022, "Generation Inadequacy", MW and customers unknown) could not be matched to any
  public document. Every California county is used (`state_list`); the row says so.
* 2021-R119 repeats R110 (same MW, clock time and states) but is dated 19 Feb, after the last SPP and MISO sheds in the
  FERC-NERC report. It is kept as given and noted as a possible date error.

### 4.2 Sheds documented by FERC-NERC and ISO reports

* 14-15 Aug 2020, CAISO: one record per utility and day for PG&E, SCE and SDG&E, the utilities the RCA names as having
  shed (the RCA says contacted non-CPUC entities did not). Times from RCA Tables 3.1 and 3.2; SDG&E has a duration only,
  so the CAISO Stage 3 window is used.
* 15-18 Feb 2021, ERCOT: 01:20 CST 15 Feb to 00:42 CST 18 Feb, the whole ERCOT footprint (ERCOT's 17 transmission
  operators with shed orders are listed in `notes`). SPP: 12:04-14:00 CST 15 Feb and 06:44-10:07 CST 16 Feb, the whole
  SPP footprint. MISO South: local transmission sheds in the WOTAB load pocket and western Louisiana (04:55 CST 15 Feb
  to 10:10 CST 16 Feb; MISO utilities in TX and LA) and the EEA3 shed of 18:40-20:41 CST 16 Feb (MISO utilities in AR,
  LA, MS, TX). The reports name areas, not member utilities, so EIA-861 balancing-area codes define the members.
* 23-24 Dec 2022 (Winter Storm Elliott): TVA (two sheds), LG&E/KU, Duke Energy Carolinas, Duke Energy Progress, Dominion
  Energy South Carolina, Santee Cooper, with the report's EST times. The report names the balancing authority only;
  rows cover the named utility's own territory (`service_territory`) and the other EIA-861 utilities with the same BA
  code (`iso_footprint`), because a BA-ordered shed is carried out by the load-serving utilities in its area (for TVA,
  its local power companies; TVA's own retail territory is three counties).
* 2024-2025, all found: SPS (Xcel) 31 Mar 2025 (122 MW instructed 01:49 CT, southwest SPS; end not published, end of
  local day; SPS territory); SWEPCO 2 Apr 2025 (start not published, start of local day; restored by 17:00 CDT; SWEPCO
  territory); SWEPCO 26 Apr 2025 (15:12-21:23 CDT; Caddo and Bossier parishes, as SPP states); Entergy Louisiana and
  Entergy New Orleans 25 May 2025 (16:20-22:43 EST; Orleans, Jefferson, St. Bernard, Plaquemines from the FOX8
  outages-by-parish list, and St. Tammany for Slidell, which the MISO report names); Cleco 25 May 2025 (16:28-19:00
  EST; St. Tammany, and Washington for Franklinton, from the LPSC presentation and Cleco's statement). NERC reports no
  operator-initiated load shed in the contiguous US in 2024 (its only 2024 entry is a Western Interconnection event,
  the Alberta shed of 5 Apr 2024, outside the US) and none in 2025 tied to an energy emergency.

### 4.3 Damage flag (M7)

`flag_damage_shed = 1` for 7 records: 2019-R184 (SWEPCO substation incident near Longview), the four Hurricane Laura
rows 2020-R258, R262, R265, R266 (MISO-directed shed after Laura damaged the Entergy transmission system), 2022-R166
(PJM-directed sheds after AEP Ohio 138 kV lines failed in storms) and LS-SPP-20250402-SWEPCO (storm-forced transmission
outages; the reports do not say who owns the failed lines). Not flagged: the MISO WOTAB shed of 15 Feb 2021 (icing line
trips together with generation outages; the report gives no ownership), the MISO shed of 25 May 2025 (a 500 kV line out
since a March tornado, but the shed day had no storm), and all energy-emergency sheds.

### 4.4 Territory expansion

* EIA-861 2023 is the only year on disk, so it is used for every event (2019-2025); no closer year was available. Known
  consequences: Lubbock (SPP in Feb 2021, ERCOT by 2023) sits in the 2023 ERCOT footprint, not SPP's, though both
  February 2021 windows overlap; utility territories list every county with any customers, so they over-cover.
* County names in `Service_Territory_2023.xlsx` (sheet Counties_States) are matched to FIPS through the 2023 Census
  gazetteer. All names used here match (the unmatched names are in Alaska and the old Connecticut counties).
* Balancing-area members: utilities with that `BA Code` in `Sales_Ult_Cust_2023.xlsx`, rows of service type Bundled or
  Delivery (energy-only retail sellers have no wires territory). Resulting footprints: ERCOT 199 Texas counties, SPP 605
  counties in 14 states, MISO South (AR, LA, MS, TX) 229, CAISO IOUs 57 California counties, TVA 200.
* PSPS events without a county list (M7) take the operator's EIA-861 territory in that state (`service_territory`).
  Where an event lists some counties and says other areas were also shut off without naming them, the unnamed part is
  expanded the same way.

### 4.5 Load-shed records

| record_id | operator | start_utc | end_utc | footprint_rule | states | counties | flag_damage_shed |
|---|---|---|---|---|---|---|---|
| LS-DOE417-2019-R084 | not named in DOE-417 summary | 2019-04-26T01:03:00Z | 2019-04-26T01:32:00Z | county_list | AZ | 1 | 0 |
| LS-DOE417-2019-R118 | not named in DOE-417 summary | 2019-06-12T21:56:00Z | 2019-06-12T22:50:00Z | county_list | CA | 2 | 0 |
| LS-DOE417-2019-R184 | Southwestern Electric Power Co | 2019-08-18T20:59:00Z | 2019-08-19T04:00:00Z | service_territory | LA,TX | 36 | 1 |
| LS-DOE417-2020-R222 | not named in DOE-417 summary | 2020-08-13T19:51:00Z | 2020-08-13T22:27:00Z | county_list | TX | 1 | 0 |
| LS-DOE417-2020-R223 | CAISO balancing area | 2020-08-15T01:36:00Z | 2020-08-15T03:42:00Z | iso_footprint | CA | 57 | 0 |
| LS-DOE417-2020-R225 | CAISO balancing area | 2020-08-15T00:15:00Z | 2020-08-16T04:00:00Z | iso_footprint | CA | 57 | 0 |
| LS-DOE417-2020-R226 | Southern California Edison Co | 2020-08-15T01:45:00Z | 2020-08-15T04:12:00Z | county_list | CA | 9 | 0 |
| LS-DOE417-2020-R228 | CAISO balancing area | 2020-08-16T01:25:00Z | 2020-08-16T02:44:00Z | iso_footprint | CA | 57 | 0 |
| LS-DOE417-2020-R229 | CAISO balancing area | 2020-08-15T21:53:00Z | 2020-08-16T03:00:00Z | iso_footprint | CA | 57 | 0 |
| LS-DOE417-2020-R230 | Southern California Edison Co | 2020-08-15T22:00:00Z | 2020-08-16T02:45:00Z | county_list | CA | 8 | 0 |
| LS-DOE417-2020-R258 | MISO-directed shed in the Entergy Texas area | 2020-08-27T17:06:00Z | 2020-08-28T03:49:00Z | iso_footprint | TX | 36 | 1 |
| LS-DOE417-2020-R262 | MISO-directed shed in the Entergy Texas area | 2020-08-27T17:02:00Z | 2020-08-28T03:54:00Z | iso_footprint | TX | 36 | 1 |
| LS-DOE417-2020-R265 | not named in DOE-417 summary | 2020-08-27T18:24:00Z | 2020-08-28T05:00:00Z | county_list | TX | 2 | 1 |
| LS-DOE417-2020-R266 | MISO-directed shed in the Entergy Texas area | 2020-08-27T17:06:00Z | 2020-08-28T05:00:00Z | iso_footprint | TX | 36 | 1 |
| LS-DOE417-2020-R277 | Imperial Irrigation District | 2020-09-06T00:25:00Z | 2020-09-06T03:55:00Z | county_list | CA | 1 | 0 |
| LS-DOE417-2021-R050 | SPP balancing area | 2021-02-15T02:00:00Z | 2021-02-18T15:28:00Z | iso_footprint | KS,ND,NE,OK,SD,TX | 435 | 0 |
| LS-DOE417-2021-R054 | not named in DOE-417 summary | 2021-02-15T07:20:00Z | 2021-02-18T06:42:00Z | county_list | TX | 2 | 0 |
| LS-DOE417-2021-R055 | not named in DOE-417 summary | 2021-02-15T07:20:00Z | 2021-02-18T06:02:00Z | county_list | TX | 1 | 0 |
| LS-DOE417-2021-R057 | ERCOT | 2021-02-15T06:00:00Z | 2021-02-18T06:42:00Z | iso_footprint | TX | 199 | 0 |
| LS-DOE417-2021-R058 | ERCOT | 2021-02-15T11:36:00Z | 2021-02-18T06:42:00Z | iso_footprint | TX | 199 | 0 |
| LS-DOE417-2021-R059 | ERCOT | 2021-02-15T19:00:00Z | 2021-02-20T16:00:00Z | iso_footprint | TX | 199 | 0 |
| LS-DOE417-2021-R060 | ERCOT | 2021-02-15T07:54:00Z | 2021-02-18T05:56:00Z | iso_footprint | TX | 199 | 0 |
| LS-DOE417-2021-R061 | ERCOT | 2021-02-15T07:20:00Z | 2021-02-19T15:00:00Z | iso_footprint | TX | 199 | 0 |
| LS-DOE417-2021-R064 | not named in DOE-417 summary | 2021-02-15T07:54:00Z | 2021-02-19T15:00:00Z | county_list | TX | 1 | 0 |
| LS-DOE417-2021-R067 | ERCOT | 2021-02-15T07:23:00Z | 2021-02-19T11:30:00Z | iso_footprint | TX | 199 | 0 |
| LS-DOE417-2021-R073 | ERCOT | 2021-02-15T11:40:00Z | 2021-02-16T19:11:00Z | iso_footprint | TX | 199 | 0 |
| LS-DOE417-2021-R077 | not named in DOE-417 summary | 2021-02-15T13:52:00Z | 2021-02-17T20:15:00Z | county_list | TX | 14 | 0 |
| LS-DOE417-2021-R078 | SPP balancing area | 2021-02-15T12:51:00Z | 2021-02-16T19:19:00Z | iso_footprint | OK | 77 | 0 |
| LS-DOE417-2021-R083 | SPP balancing area | 2021-02-15T18:18:00Z | 2021-02-15T19:22:00Z | iso_footprint | KS,MO | 152 | 0 |
| LS-DOE417-2021-R086 | ERCOT | 2021-02-15T11:36:00Z | 2021-02-18T06:42:00Z | iso_footprint | TX | 199 | 0 |
| LS-DOE417-2021-R087 | ERCOT | 2021-02-15T06:00:00Z | 2021-02-18T06:42:00Z | iso_footprint | TX | 199 | 0 |
| LS-DOE417-2021-R088 | SPP balancing area | 2021-02-16T13:03:00Z | 2021-02-21T04:00:00Z | iso_footprint | KS,MO | 152 | 0 |
| LS-DOE417-2021-R091 | SPP or MISO South member | 2021-02-16T12:48:00Z | 2021-02-16T16:10:00Z | iso_footprint | LA | 64 | 0 |
| LS-DOE417-2021-R096 | not named in DOE-417 summary | 2021-02-16T13:20:00Z | 2021-02-16T16:02:00Z | county_list | NE | 13 | 0 |
| LS-DOE417-2021-R097 | MISO | 2021-02-16T10:30:00Z | 2021-02-16T19:00:00Z | iso_footprint | IL,LA,TX | 191 | 0 |
| LS-DOE417-2021-R098 | MISO South | 2021-02-17T00:43:00Z | 2021-02-17T02:41:00Z | iso_footprint | AR,LA,MS,TX | 229 | 0 |
| LS-DOE417-2021-R099 | ERCOT | 2021-02-17T01:40:00Z | 2021-02-17T07:00:00Z | iso_footprint | TX | 199 | 0 |
| LS-DOE417-2021-R105 | MISO South member in Louisiana | 2021-02-16T12:24:00Z | 2021-02-16T16:41:00Z | iso_footprint | LA | 63 | 0 |
| LS-DOE417-2021-R107 | SPP or MISO member in Missouri | 2021-02-16T12:10:00Z | 2021-02-16T16:10:00Z | iso_footprint | MO | 102 | 0 |
| LS-DOE417-2021-R109 | not named in DOE-417 summary | 2021-02-16T13:22:00Z | 2021-02-16T14:00:00Z | county_list | LA | 3 | 0 |
| LS-DOE417-2021-R110 | SPP or MISO South member | 2021-02-16T12:48:00Z | 2021-02-16T16:10:00Z | iso_footprint | LA,TX | 169 | 0 |
| LS-DOE417-2021-R111 | ERCOT | 2021-02-16T10:26:00Z | 2021-02-18T06:42:00Z | iso_footprint | TX | 199 | 0 |
| LS-DOE417-2021-R119 | SPP or MISO South member | 2021-02-19T12:48:00Z | 2021-02-20T06:00:00Z | iso_footprint | LA,TX | 169 | 0 |
| LS-DOE417-2022-R152 | unidentified | 2022-06-05T22:20:00Z | 2022-06-06T03:12:00Z | state_list | CA | 58 | 0 |
| LS-DOE417-2022-R160 | not named in DOE-417 summary | 2022-06-11T03:27:00Z | 2022-06-11T04:30:00Z | county_list | CA | 1 | 0 |
| LS-DOE417-2022-R161 | not named in DOE-417 summary | 2022-06-11T03:20:00Z | 2022-06-11T03:28:00Z | county_list | CA | 1 | 0 |
| LS-DOE417-2022-R166 | Ohio Power Co | 2022-06-14T17:46:00Z | 2022-06-16T08:51:00Z | service_territory | OH | 54 | 1 |
| LS-DOE417-2022-R363 | Tennessee Valley Authority balancing area | 2022-12-22T15:04:00Z | 2022-12-25T06:00:00Z | iso_footprint/service_territory | AL,KY,MS,TN | 178 | 0 |
| LS-DOE417-2022-R370 | not named in DOE-417 summary | 2022-12-23T14:31:00Z | 2022-12-24T16:30:00Z | county_list | TN | 1 | 0 |
| LS-DOE417-2022-R374 | Duke Energy Progress | 2022-12-24T11:35:00Z | 2022-12-24T21:10:00Z | service_territory | NC | 56 | 0 |
| LS-DOE417-2022-R375 | Duke Energy Carolinas | 2022-12-24T09:45:00Z | 2022-12-24T20:45:00Z | service_territory | NC,SC | 63 | 0 |
| LS-CAISO-20200814-PGE | CAISO-directed rotating outage: Pacific Gas & Electric Co | 2020-08-15T01:38:00Z | 2020-08-15T04:08:00Z | iso_footprint | CA | 50 | 0 |
| LS-CAISO-20200814-SCE | CAISO-directed rotating outage: Southern California Edison Co | 2020-08-15T01:56:00Z | 2020-08-15T02:59:00Z | iso_footprint | CA | 16 | 0 |
| LS-CAISO-20200814-SDGE | CAISO-directed rotating outage: San Diego Gas & Electric Co | 2020-08-15T01:38:00Z | 2020-08-15T03:38:00Z | iso_footprint | CA | 2 | 0 |
| LS-CAISO-20200815-PGE | CAISO-directed rotating outage: Pacific Gas & Electric Co | 2020-08-16T01:25:00Z | 2020-08-16T02:55:00Z | iso_footprint | CA | 50 | 0 |
| LS-CAISO-20200815-SCE | CAISO-directed rotating outage: Southern California Edison Co | 2020-08-16T01:43:00Z | 2020-08-16T01:51:00Z | iso_footprint | CA | 16 | 0 |
| LS-CAISO-20200815-SDGE | CAISO-directed rotating outage: San Diego Gas & Electric Co | 2020-08-16T01:28:00Z | 2020-08-16T01:48:00Z | iso_footprint | CA | 2 | 0 |
| LS-ERCOT-20210215 | ERCOT | 2021-02-15T07:20:00Z | 2021-02-18T06:42:00Z | iso_footprint | TX | 199 | 0 |
| LS-SPP-20210215 | SPP | 2021-02-15T18:04:00Z | 2021-02-15T20:00:00Z | iso_footprint | AR,CO,IA,KS,LA,MN,MO,MT,ND,NE,NM,OK,SD,TX | 605 | 0 |
| LS-SPP-20210216 | SPP | 2021-02-16T12:44:00Z | 2021-02-16T16:07:00Z | iso_footprint | AR,CO,IA,KS,LA,MN,MO,MT,ND,NE,NM,OK,SD,TX | 605 | 0 |
| LS-MISOS-20210215-LTE | MISO South | 2021-02-15T10:55:00Z | 2021-02-16T16:10:00Z | iso_footprint | LA,TX | 99 | 0 |
| LS-MISOS-20210216-EEA3 | MISO South | 2021-02-17T00:40:00Z | 2021-02-17T02:41:00Z | iso_footprint | AR,LA,MS,TX | 229 | 0 |
| LS-TVA-20221223 | Tennessee Valley Authority balancing area | 2022-12-23T15:31:00Z | 2022-12-23T17:43:00Z | iso_footprint/service_territory | AL,GA,KY,MS,NC,TN,VA | 200 | 0 |
| LS-TVA-20221224 | Tennessee Valley Authority balancing area | 2022-12-24T10:51:00Z | 2022-12-24T16:30:00Z | iso_footprint/service_territory | AL,GA,KY,MS,NC,TN,VA | 200 | 0 |
| LS-LGEKU-20221223 | LG&E and KU | 2022-12-23T22:58:00Z | 2022-12-24T03:11:00Z | service_territory | KY,VA | 85 | 0 |
| LS-DEC-20221224 | Duke Energy Carolinas | 2022-12-24T11:27:00Z | 2022-12-24T20:45:00Z | iso_footprint/service_territory | GA,NC,SC | 72 | 0 |
| LS-DEP-20221224 | Duke Energy Progress | 2022-12-24T11:25:00Z | 2022-12-24T13:43:00Z | iso_footprint/service_territory | NC,SC | 72 | 0 |
| LS-DESC-20221224 | Dominion Energy South Carolina | 2022-12-24T13:00:00Z | 2022-12-24T13:09:00Z | service_territory | SC | 13 | 0 |
| LS-SANTEE-20221224 | Santee Cooper | 2022-12-24T12:18:00Z | 2022-12-24T12:33:00Z | iso_footprint/service_territory | SC | 38 | 0 |
| LS-SPP-20250331-SPS | Southwestern Public Service Co | 2025-03-31T06:49:00Z | 2025-04-01T06:00:00Z | service_territory | NM,TX | 51 | 0 |
| LS-SPP-20250402-SWEPCO | Southwestern Electric Power Co | 2025-04-02T05:00:00Z | 2025-04-02T22:00:00Z | service_territory | AR,LA,TX | 50 | 1 |
| LS-SPP-20250426-SWEPCO | Southwestern Electric Power Co | 2025-04-26T20:12:00Z | 2025-04-27T02:23:00Z | county_list | LA | 2 | 0 |
| LS-MISO-20250525-ENTERGY | Entergy Louisiana and Entergy New Orleans, MISO-directed | 2025-05-25T21:20:00Z | 2025-05-26T03:43:00Z | county_list | LA | 5 | 0 |
| LS-MISO-20250525-CLECO | Cleco Power, MISO-directed | 2025-05-25T21:28:00Z | 2025-05-26T00:00:00Z | county_list | LA | 2 | 0 |

## 5. PSPS

### 5.1 California IOUs

114 de-energisation events 2018-2025 (PG&E 31, SCE 61, SDG&E 15, PacifiCorp 2, Liberty 5, Bear Valley 0), 638 county
rows. Every "Yes" event of the CPUC table in the window is covered; the "No" events checked show no customers
de-energised in the reports. Sources: post-season data workbooks (county sheet, first de-energisation and last
restoration per county) for 2021-2025; post-event reports for 2018-2020, with PG&E press releases where the report had
no county table; SDG&E counties before 2021 from the community names in the report tables. 389 rows have county times,
249 event- or wave-level times. Event dates follow the first customer-affecting de-energisation, which often differs
from the event table's date (the table often dates the watch period). Bear Valley reports no PSPS in 2021-2025. Liberty's
20 Nov 2024 event was an upstream NV Energy shutoff (Liberty's report).

### 5.2 Oregon and other states

34 events (listed in the table below). Oregon (UM 2268 and utility releases): PGE 7 Sep 2020 (Mt. Hood corridor), PGE
"eight high-risk fire areas" Sept 2020 (UNVERIFIED, see §6), PacifiCorp, PGE, EWEB and Lane Electric on 9 Sep 2022.
UM 2268 reports none for PGE or PacifiCorp in 2023-2025 and none for Idaho Power in Oregon. Colorado: Xcel 6-7 Apr 2024
(6 counties), 17 Dec and 19 Dec 2025. Utah: Rocky Mountain Power 17 Jun 2022 and 21 Jun 2025. Nevada: NV Energy 15
events 2020-2025 (mostly Mt. Charleston, Clark County). Idaho: Idaho Power 24 Jul 2024. Washington: Avista 29 Sep 2024.
New Mexico and Texas: Xcel (SPS) 3, 14 and 18 Mar 2025; PNM 17 Apr 2025. None were found for Arizona (APS's first PSPS
was in 2026), Wyoming, Montana, Oklahoma or Kansas (Evergy states it has no PSPS authority).

Counties that were derived rather than stated: PGE 2020 and 2022, EWEB and Lane Electric (from the places each source
names, via USGS GNIS); Xcel SPS 18 Mar 2025 (Quay NM, Briscoe and Gaines TX, from the named towns); Rocky Mountain
Power 2025 (from news); Idaho Power 2024 Ada County (from the report's damage table and the county that received the
notices). Each row says so in `notes`.

### 5.3 Events per state, operator and year (by start_utc)

| state operator | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | total |
|---|---|---|---|---|---|---|---|---|---|
| CA LIBERTY | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 1 | 5 |
| CA PACIFICORP | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 2 |
| CA PGE | 1 | 7 | 6 | 5 | 0 | 2 | 6 | 4 | 31 |
| CA SCE | 1 | 10 | 10 | 8 | 3 | 5 | 11 | 13 | 61 |
| CA SDGE | 3 | 3 | 4 | 1 | 0 | 0 | 2 | 2 | 15 |
| CO XCEL | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 2 | 3 |
| ID IPC | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 |
| NM PNM | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| NM SPS | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 3 |
| NV NVE | 0 | 0 | 1 | 3 | 8 | 0 | 1 | 2 | 15 |
| OR EWEB | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
| OR LANEEC | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
| OR PAC | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
| OR PGE | 0 | 0 | 2 | 0 | 1 | 0 | 0 | 0 | 3 |
| TX SPS | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 2 |
| UT PAC | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 2 |
| WA AVA | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 |

## 6. Not found, not verified, not included

* DOE-417 summaries for 2024 and 2025 are not published; 2024-2025 sheds rest on NERC and ISO reports (M7 limit).
* UNVERIFIED rows (174 rows, 16 PSPS records): SCE 2020 counties that the reports contradict (Inyo 6 Nov and 7 Dec,
  Riverside 26 Nov); SCE 10 Jun and 5 Nov 2025 counties that appear only in grouped workbook rows (Riverside, Ventura);
  NV Energy 21 Oct and 1 Nov 2022, 20 Nov 2024, 5 Nov and 19 Dec 2025 (dockets downloaded but not read; counties and
  times from notification letters or news); Xcel SPS 14 Mar 2025 (NM and TX; no county list or times) and the unnamed
  areas of 18 Mar 2025; the CORE Electric members cut by Xcel's transmission shutoff on 19 Dec 2025; PGE "eight
  high-risk fire areas" (areas, counties and times never published; window set to 7-10 Sep 2020 from the windstorm
  start and the last release). The territory-expanded ones of these carry the operator's EIA-861 counties (M7).
* SCE 2024-2025 workbook rows that group several counties list more counties than SCE's own dashboard for 12 events.
  Two were checked against the post-event reports and marked UNVERIFIED; the others are kept (conservative) and flagged
  in `notes`. Parsing the county tables of SCE's 2024-2025 post-event reports is the first thing to redo.
* Left out for lack of a day or a footprint: BPA PSPS de-energisations listed only as "September 2022, WA" and
  "August 2024, WA", with no customer load lost; two further SPS load-pocket sheds in 2024-2025 that the NERC incident
  review describes without dates (one in April, about 150 MW); the MISO local shed of 140 MW at 06:52 CST 16 Feb 2021,
  location not given (the DOE-417 row 2021-R097 lists Illinois and may be it); the 29 MW local shed in southwest
  Missouri on 22 Dec 2022 (SPP footprint, FERC-NERC), below the design's list and without a county.
* Flagged, kept (not exclusions, per §7): de-energisation for active fires or at fire agencies' request (Pacific Power
  Medford lines Sept 2020, Mountain Parks Oct 2020, Beckwourth and Tamarack 2021, PSE Skykomish and Snohomish PUD Sept
  2022, Seattle City Light 2023, LPEA Oak Fire Aug 2025 and July 2025 fires, Garkane and Rocky Mountain Power Willard
  Peak 2025, SCE fire-department-requested shutoffs 25 Nov 2019). SCE 12 Oct 2019 includes one Santa Barbara circuit
  switched off for an active fire (kept inside the PSPS event, noted). One possible flood shutoff (LPEA Vallecito) is
  unconfirmed. Planned shutoffs that were called off are not included (SPS near Post TX 4 Mar 2025 and 1 Apr 2025, PNM
  6, 14 and 18 Mar 2025).
* Coverage not finished (search budget): Oregon consumer-owned utilities other than EWEB and Lane Electric, Wyoming,
  Montana and plains co-operatives, Puget Sound Energy June-December 2025, southern Idaho co-operatives, and
  New Mexico, Texas, Oklahoma and Kansas co-operatives. California publicly owned utilities and co-operatives are not
  covered by the CPUC records.
* Could not be opened: two original SCE 2019 post-event report links (error pages; the amended reports were used); the
  circuit appendix of SCE's 27 Oct-4 Nov 2019 report (missing from the public PDF); Xcel's PSPS map and event pages
  (script-loaded, not archived); the New Mexico PRC e-docket search; SPS and Oncor/AEP plan files over 50 MB (not
  downloaded); the Idaho PUC and NorthWestern Energy sites (TLS failures; read with a web-fetch tool, no file saved);
  some news pages that refuse direct download (KGW, Seattle Times, Renewable Energy World). Not identifiable at all:
  planned outages and sheds below the DOE-417 thresholds (stated limitation, §7).

## 7. Counts

Years are the UTC year of start_utc. Row counts only; no outage numbers are used.

Records (events) by kind and year:

| kind | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | total |
|---|---|---|---|---|---|---|---|---|---|
| load_shed | 0 | 3 | 18 | 33 | 15 | 0 | 0 | 5 | 74 |
| psps | 5 | 20 | 24 | 18 | 16 | 7 | 27 | 31 | 148 |

Rows (county x record) by kind and year:

| kind | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | total |
|---|---|---|---|---|---|---|---|---|---|
| load_shed | 0 | 39 | 493 | 5763 | 1092 | 0 | 0 | 110 | 7497 |
| psps | 14 | 155 | 163 | 94 | 71 | 26 | 128 | 251 | 902 |

Rows by footprint rule:

| kind | county_list | iso_footprint | service_territory | state_list |
|---|---|---|---|---|
| load_shed | 70 | 6821 | 548 | 58 |
| psps | 693 | 0 | 209 | 0 |

Total: 8399 rows, 222 records (74 load shed, 148 PSPS); 1625 distinct counties; rows with flag_damage_shed = 1: 250 (7 records).
