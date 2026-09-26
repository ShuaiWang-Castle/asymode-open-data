# DATASET_DESIGN v1 — how the public panel chooses weather systems, counties and tranches

Status: **registered** 2026-09-26 (the commit time is the registration time), before any frame, draw, weather window,
panel or outcome of this design exists. v0 (commit db631a6) was a draft for review. v1 integrates three reviews:
statistical (`contrib/REVIEW_dataset_design.md`), physical (`contrib/REVIEW_dataset_physics.md`), and literature and
data (`notes/LITERATURE_dataset_design.md`). Changes after this commit are dated amendments at the end, never edits in
place. §13 lists what changed from v0, and why; §14 lists the review suggestions not adopted.

## 0. Decisions at a glance

| question | decision |
|---|---|
| unit of independence | the parent weather system; systems are grouped into families, and a family never straddles a split |
| frame | every warning-defined system of five damage regimes, 2018-07-12 to 2025-12-31, CONUS, listed before any draw |
| regimes | convective, synoptic wind, tropical cyclone, winter, heavy rain (heat, cold and fire weather are not sampled) |
| states | every state that passes a strict coverage gate; no state is chosen by hand |
| sizes | development 75 systems (15 per regime), sealed confirmation 100 (20 per regime), at most 140 counties per system |
| draw | PPS on customers under warnings, shoulder-season and compound systems at double size, implicit stratification by period, region and date; every inclusion probability kept |
| counties | warned, advisory or watch, and no-product ring counties, drawn by PPS on a weather hazard index, so near misses are in |
| observation gates | only data from before the window; unobserved forecast hours are masked, never a reason to drop a county |
| estimand | regime-balanced skill: equal-weight mean over regimes of design-weighted MSE skill; the loss has the same form |
| operator outages | excluded from public records of type, footprint and time only |

## 1. Target, claim population, estimands

**Forecast target.** A county's hourly outage fraction y (EAGLE-I customers out over the 2024 modelled customers) over
the 144 h after an origin, given the 72 h observed before it and the observed weather of the whole window.

**Claim population.** County-events (county x parent system) in the domain (§5) of systems of the five regimes whose
windows lie in 2018-07-12..2025-12-31, in states passing G2, after the operator exclusions (§7). Outside it: heat waves,
cold waves, fire weather, storms that left no warning in the archive (sub-severe storms get a Special Weather Statement,
which has no VTEC code), states failing G2.

**Estimands.**

* **Primary: regime-balanced skill** Φ(f) = (1/R) Σ_r [1 − MSE_r(f) / MSE_r(ref)], with MSE_r the design-weighted mean
  squared error over the observed county-hours of regime r (weights §5.4), ref = the all-zero forecast (skill) or the
  host (gain), and r over the headline regimes of §9.3.
* **Secondary:** the frame-weighted pooled relative MSE (system weights N_h / n_h inside the design weights), every
  per-regime value, and the worst regime.
* **Descriptive:** the unweighted pooled RMSE. It weights regimes by their squared outcome mass, so it mostly measures
  the largest tropical and winter systems.
* **Never:** scores on a subset chosen by the outcome (for example damaging cases only). Breakdowns are by covariates:
  the stratum of §5.1 and the hazard class (h_c ≥ 1 or < 1).
* **The loss has the estimand's form.** County-event i of regime r weighs w_i / (R Z_r), with
  Z_r = Σ_{i ∈ r, training folds} w_i Σ_t y_it² (the all-zero SSE of r in the training folds), so training does not
  undo the equal allocation.

## 2. Why a designed subset, not all data or all states

* **Compute.** A 900-step run handles 10-12 k county-events in about 40 minutes, about 100 runs a day on three workers.
  Every county of every system in 7.5 years is two orders of magnitude more.
* **Disk.** About 30 GB are free, and one CONUS ERA5 window takes 236 MB.
* **Labels.** State coverage is below 0.8 in 9 of the 48 lower states even in 2022 (by the year's minimum), the EAGLE-I
  files differ by year (§6), and operator actions look like storm damage in the data (§7).
* **Power.** It comes from independent systems, not from counties. Beyond about 140 counties, a system adds cost, not
  power (statistical review §3).

So every system is listed, a sample is drawn with known probabilities, and weights let the sample stand for the frame.
States leave only through the coverage gate.

## 3. The frame: parent weather systems from NWS warnings

### 3.1 Sources (all public)

IEM VTEC archives (annual attribute tables 2018-2025 and storm-based polygons); NHC HURDAT2 (Atlantic and eastern North
Pacific best tracks); the NWS zone-county correlation and IEM UGC geometries valid at issuance; the Census county
adjacency file; the EAGLE-I 2024 modelled customers and coverage history; the WorldPop population nodes of the 3 km
node set.

### 3.2 Products

| regime | defining products (SIG = W) | typing priority |
|---|---|---|
| tropical cyclone | TR, HU, SS, EW, and the HURDAT2 domain (§3.4) | 1 |
| winter | WS, IS, BZ, LE | 2 |
| synoptic wind | HW | 3 |
| convective | SV, TO | 4 |
| heavy rain | FF, FA | 5 |

* Watches (SIG = A) and advisories (SIG = Y) of these phenomena, and WI.Y, WW.Y, ZR.Y, LE.Y and FA.Y, only place
  counties in stratum S2. They neither create nor link systems.
* Event key: WFO + PHENOM + SIG + ETN + year. An event starts at INIT_ISS and ends at its last EXPIRED. Upgrades are
  kept: the negative durations they produce are upgrades, not errors.
* The meaning of the time fields is checked on the 2018 table before the frame is built; any difference from the IEM
  documentation is recorded as an amendment.

### 3.3 County exposure

* A county's exposure a to a product is the share of its population (3 km nodes) inside the product's area:
  * for polygon products (SV, TO, FF, EW), the storm-based polygon;
  * for zone products, the zones, with geometry valid at issuance. A county that coincides with one zone has a = 1.
* If a year's zone geometries cannot be obtained, every county mapped to a warned zone gets a = 1 (recorded).
* A county is **warned (S1)** by a W product with a ≥ 0.1. A W product with a < 0.1 places it in S2.

### 3.4 Parent systems

1. **Tropical cyclones first.**
   * A storm's span is the time its HURDAT2 centre (any status, post-tropical included; positions interpolated hourly)
     lies within 300 km of a CONUS county centroid.
   * Its domain is the set of counties within 300 km of a track position in the span.
   * A W event joins the TC system if its INIT_ISS lies in [span start − 24 h, span end + 12 h] and at least half of its
     counties are in the domain.
2. **Other W events** form a graph. Two events are linked when their time intervals are within 12 h of each other and
   their county sets overlap or are adjacent. The connected components are **families**.
3. **Segments.** A family longer than 96 h (first INIT_ISS to last expiry) is cut into consecutive 96 h segments by
   INIT_ISS. Each segment is a **system**, and segments keep their family.

### 3.5 Typing

* Every county-event keeps a multi-label vector: all regimes of its W products.
* Its label is the highest-priority regime in the vector (tropical inside a TC system).
* A system's regime is the label with the largest share of S1 customers, ties broken by priority.
* A system is **compound** if a second label holds at least 20% of its S1 customers.

### 3.6 Origin and window

* Expected onset E_s: the earliest begin time, max(ISSUED, INIT_ISS), among the system's W events. For a TC system,
  the first hour of its span.
* Origin t_s = floor_hour(E_s) − 6 h. The lead before the expected hazard is then the same in every regime; a warning's
  own lead is not (minutes for convective warnings, a day or more for tropical ones).
* Window: [t_s − 72 h, t_s + 144 h). Restorations longer than the 144 h horizon are truncated (a stated limit).

### 3.7 System gates (outcome-independent)

* **S-a.** The window lies in 2018-07-12..2025-12-31, with 30 days of EAGLE-I record before it.
* **S-b.** National collection runs (the timestamps with at least 5 reporting counties, `panel.collection_timestamps`)
  cover at least 90% of the window's hours.
* **S-c.** HRRR `wrfsfcf01` files are present for at least 95% of the window's hours (AWS listing).
* **S-d.** At least one S1 county passes G1-G2 (§6) and the operator exclusions (§7).

### 3.8 Stratum variables

* **Period.** P1 from 2018-07-12 to 2020-12-01 (HRRR v3); P2 from 2020-12-02 (HRRR v4).
* **Region.** The plurality of S1 customers among five regions (Northeast, Southeast, Central, Plains and Mountain
  West, Pacific; the state lists are in the frame code).
* **Season class.** Core months per regime: convective May-Aug, synoptic wind Nov-Mar, winter Dec-Feb, tropical
  Aug-Oct, heavy rain Jun-Sep. Every other month is shoulder season. Leaf state is otherwise confounded with the regime
  (convection mostly leaf-on, synoptic wind mostly leaf-off).
* **Compound flag** (§3.5).

### 3.9 Frame file

The frame file and its hash are committed before any draw. It holds every system (family, regime, labels, S1 and S2
counties with exposure, onset, origin, size, strata) and the counts N_h per regime x period x region x season class.

## 4. Tranches, allocation and the draw

### 4.1 Tranches

* **C:** confirmation. Sealed, inference only.
* **D:** development.
* **P:** prospective, the systems of the next EAGLE-I release, which becomes the next confirmation tranche.
* **U:** unused.

### 4.2 Used windows

A system is **used** if it shares a county with a window whose outcomes were read in this repository before this
design, and the two windows start within 16 days of each other. The used windows are compiled from the event files of
the earlier panels into `data_provenance/used_windows.json` at the frame step, every listed event included. Used
systems can be in D, never in C.

### 4.3 Size measure

M_s = the 2024 modelled customers in the S1 counties that pass G1-G2. M_s is doubled for a shoulder-season system and
doubled again for a compound one. This oversamples, with known probabilities, the cases the physical review asked for
(leaf-on snow, leaf-off convection, ice then wind).

### 4.4 Draw

1. Within each regime, the frame is sorted by period, region and origin.
2. Systematic PPS with a seeded random start. Units with π ≥ 1 are taken with certainty and removed, iteratively.
3. **C is drawn first** (seed 20260926), from systems that are not used.
4. Systems blocked by C leave the D frame: those in the same family as a C system, or sharing a county with one and
   starting within 16 days of it.
5. **D is drawn next** (seed 20260927).

π is recorded per system and tranche. D's π are conditional on the C draw.

### 4.5 Allocation

* C gets 20 systems per regime and D 15.
* The tropical frame is small. Let N be the number of tropical systems eligible for C. Then C = min(20, ⌊0.55 N⌋)
  and D = min(15, the tropical systems left for D).

### 4.6 Coverage audit, before any weather is fetched

The audit is outcome-independent. It runs over the domain counties (§5.1) of each regime's D systems.

* **Geography.** Every tercile of five axes holds at least 15% of the county-events. The axes:
  * relief: the SD of 3DEP elevation within the county;
  * tree canopy cover;
  * share of poorly drained soils;
  * customer density;
  * distance to the coast.

  Terciles are taken over the gated CONUS counties.
* **Season.** Shoulder-season systems are at least 25% of the regime's D systems.
* **Compound.** D as a whole holds at least 5 compound systems.

If a condition fails, one supplementary PPS draw (seed 20260928) adds up to 3 systems of that regime from the deficient
cell, with π = 1 − (1 − π₁)(1 − π₂). C is never supplemented.

The axes are the same for every regime. The model is meant to learn how every geography modulates every regime, so no
regime gets a geography of its own.

## 5. Counties within a system

### 5.1 Domain

The domain holds the counties that pass G1-G5 (§6) and are not operator-excluded. It has three strata:

* **S1, warned:** a ≥ 0.1 under a W product of the system.
* **S2, advisory, watch or touched.**
* **S3, ring:** no product of the system, and no W product of any system during the forecast window; centroid within
  200 km of an S1 centroid (for TC systems, also the TC domain).

### 5.2 Hazard index

h_c is computed from ERA5 over the forecast window, after the weather fetch and before any outcome of the system is
extracted. It is the maximum of four ratios, each a customer-weighted county mean of cell values:

* the maximum hourly 10 m gust over 20 m/s;
* the maximum 24 h precipitation over 50 mm;
* the maximum 24 h freezing precipitation (hours whose ptype is freezing rain or freezing drizzle) over 5 mm;
* the maximum 24 h snowfall water equivalent over 15 mm.

h_c sets sampling probabilities only; it is never a model feature.

### 5.3 Sample

* n_s = min(|domain|, 140), split 90 / 25 / 25 over S1 / S2 / S3.
* Capacity a stratum cannot use passes on in the order S1 → S2 → S3 → S1.
* Within a stratum: systematic PPS on max(h_c, 0.25), sorted by h_c, with a seeded start (20260929 + the system's
  ordinal). Certainty for π ≥ 1.

### 5.4 Weights

* w = 1 / (π_s π_c|s), trimmed at 10 times the regime's median weight in each tranche.
* The trimmed weights are primary for the loss and the metric; the untrimmed weights are secondary.

## 6. Observation: county gates and EAGLE-I harmonisation

* **G1.** At least 500 modelled customers (2024).
* **G2.** State coverage ≥ 0.8 by the year's **minimum**, in the most recent coverage year not after the system's year
  (2022 for 2023-2025). 24 of the lower 48 states pass in 2018 and 39 in 2022.
* **G3.** At least one positive record in the 30 days before the window starts (`panel.in_service_before`; explicit
  zero rows and blank counts do not count). If G3 removes more than 5% of D's otherwise gated domain counties, the
  lookback becomes 90 days. That is decided from gate counts, which read no outcome.
* **G4.** At least 90% of the prefix hours observed.
* **G5.** The origin hour observed.
* **Forecast window.** Hours without a national collection run, or with only blank counts, are masked in the loss and
  the metric. They never remove a county.

**EAGLE-I harmonisation.**

* The local 2022 file stops at 2022-11-12; it is replaced by the public v4 file, which runs to 2022-12-31 (size and
  md5 recorded).
* Explicit zero rows (published from 2022) are dropped before gating, so a record means a positive count in every year.
* Blank counts are unobserved.
* Dec 31 of 2018-2021 is a collection gap.
* 2023's `sum` column is mapped, and 2024's rows are sorted.

**Audit before training.** G3-G5 drop rates by stratum (S1, S2, S3) in D.

Artefact flags stay a training mask only; evaluation keeps every observed hour.

## 7. Operator-initiated outages

Exclusions are decided from each record's type, footprint and time, never from its size or from EAGLE-I. They are
compiled into `data_provenance/operator_exclusions.csv`, with a source per row, before any gate output exists, and they
apply to every tranche. A county-event is excluded when an exclusion window [start − 12 h, end + 48 h] overlaps its
forecast window.

* **Load shedding.**
  * DOE-417 rows with firm load shed of at least 100 MW, 2018-2023. The filter is the alert criterion, not the event
    type, which is unreliable (the February 2021 rotating outages are typed Severe Weather).
  * FERC-NERC and ISO reports:
    * 14-15 Aug 2020: CAISO;
    * 14-20 Feb 2021: ERCOT, SPP, MISO South;
    * 23-25 Dec 2022: TVA, Duke Energy Carolinas and Progress, LG&E/KU, Dominion Energy SC, Santee Cooper;
    * 26 Apr 2025: SWEPCO;
    * 25 May 2025: Entergy and CLECO.

  Where no county list exists, all counties of the utility's or ISO's service territory (EIA-861) are excluded for the
  window.
* **Public-safety power shutoffs.**
  * California IOUs: CPUC post-event reports and event table, 2018-2025, with county lists.
  * Oregon: docket UM 2268 reports.
  * Other western events, each with a source (for example Colorado, 6-7 Apr 2024).

  Every western high-wind system is checked against these lists.
* **Flagged, kept:** pre-emptive flood or surge de-energisation, and outages from fire damage.
* **Not identifiable:** planned outages, and sheds below the reporting thresholds. This is a stated limitation.

## 8. Weather and storage

* **ERA5 (ARCO)** for each D window: the main set (u10 v10 i10fg t2m d2m cape swvl1 tcc sp tp sf), the gust set (fg10)
  and the extra set (ptype sd lai_hv cp swvl2), cropped to the domain box plus 1°.
* **HRRR `wrfsfcf01`**, on the ERA5 grid and at 3 km. Categorical precipitation types and 0-6 km shear are added to the
  existing fields if the files carry them (checked on one file first). Missing HRRR hours are recorded and masked like
  unobserved outage hours. HRRR results are also reported by period (P1, HRRR v3; P2, HRRR v4).
* **Storage.** Raw windows are deleted once their features and checksums are written (they can be fetched again). C's
  weather is fetched only after a confirmatory registration.

## 9. Splits, seeds and Stage 0

### 9.1 Splits

* **Event folds (primary).** Fold groups are families, joined whenever two systems share a county and their origins
  are within 16 days of each other. No held-out outcome can then sit inside a training window. Within each regime (a
  group takes the regime of its largest system), groups are sorted by origin, and fold = rank mod 5.
* **County folds:** discard-only screens.
* **Leave one region out:** descriptive.
* **Forward in time** (origins before, and from, 2023-07-01): secondary and descriptive.

### 9.2 Seeds

This amends program.md rule 3 for the designed panel.

* A single-seed, five-fold screen may discard a change, never keep it.
* A keep needs 3 seeds (seed-averaged predictions) on all five folds, beside its twin.
* A confirmation uses 5 seeds trained on all of D.
* **A/A run.** Stage 0 runs the host with seeds 0-2 against seeds 3-5 (seed-averaged, event folds). If the A/A
  difference on the primary estimand exceeds 1% in absolute value, keeps need 5 seeds.

### 9.3 Stage 0 (registered now)

* The host W+Cin: 900 steps, 3 seeds, the 5 event folds of D.
* **Headline regimes.** A regime is a headline regime if the host beats the all-zero forecast there: the
  seed-averaged, design-weighted relative MSE against zero is below 0, and so is the upper end of its 80%
  family-cluster bootstrap interval.
* **Non-inferiority strata.** In every other regime, an arm may not raise the MSE by more than 2% against the host.
* If fewer than three regimes qualify, the host itself (idea I11) is the next object of work, before any pathway arm.

## 10. Sealing and multiplicity

* **Extraction.** C's forecast-window outcomes are extracted only by the confirmatory script, after its registration
  is committed. Until then, C's builder writes only inputs: weather, prefix and gates.
* **What counts as a read of C.** Any outcome-derived number on C outside a registered test: per-system errors,
  outcome-weighted audits, Storm Events for C, gate counts from forecast-window records, y-weighted effective clusters.
  Eligibility on C uses only features, pre-window data or the host's predicted Σŷ².
* **Budget.**
  * At most K = 4 confirmatory tests per tranche, in a fixed sequence, each one-sided at α = 0.05. Testing stops at the
    first non-rejection, and only the registered statistics are revealed.
  * A spent tranche joins D, and P becomes the next confirmation tranche. Important claims must replicate there.
* **Power.** Power is planned at one third of the exploratory effect: the one confirmed effect so far came in at 0.30 of
  its exploratory size. At a per-system SD of 9%, the minimum detectable effect is:

  | effective systems | 5 | 8 | 15 | 20 | 30 |
  |---|---|---|---|---|---|
  | MDE | 12% | 8.8% | 6.1% | 5.2% | 4.2% |

  C has about 9-12 effective systems per regime and about 46-60 regime-balanced (statistical review §3). C can
  therefore confirm regime-balanced gains of about 3-4%, not per-regime gains of that size.
* **Earlier registrations.** H2b is closed (failed at seed 0, verdict final; PREREG_W2). H1a is withdrawn (PREREG_W2
  amendment 8): the new winter stratum is not an enlargement of the earlier winter panels.

## 11. Diagnostics (never selection rules)

* **Frame coverage.** The share of county-hours with y ≥ 0.01 in gated counties, 2018-07..2025, that fall inside frame
  windows, by regime, region and month. C's windows are removed from numerator and denominator.
* **Typing check** against Storm Events, on D only.
* **Gate drop rates** by stratum (§6).

## 12. Build order

Each step is committed before the next starts.

1. The frame (§3), the used windows (§4.2) and the frame counts.
2. The operator exclusions (§7).
3. The draws of C and D, and the coverage audit (§4).
4. D's weather, gates, county sample, panels and features, with the gate audit.
5. Stage 0 (§9.3).
6. Stage B (`aris/EXPERIMENT_PLAN.md`).

## 13. Changes from v0

| v0 | v1 | reason |
|---|---|---|
| episodes: warning clusters of one regime | parent systems in families; tropical from best tracks; multi-label county typing | compound systems leak across folds and the seal (stat. 6.2; phys. 3; lit. 6.4) |
| six regimes, heat included | five damage regimes; heat and cold deferred | a different mechanism class, largely operator action (phys. 1; stat. 3) |
| 36 + 18 episodes | 75 D + 100 C systems, at most 140 counties each | three per regime cannot reach 8 effective systems (stat. 3) |
| equal allocation, random within strata | PPS on customers, shoulder and compound at double size, implicit stratification, π kept | power and balance with known weights (stat. 3; lit. 5) |
| controls: a random 25% of the region | S2 (advisory, watch) and S3 (ring) strata, PPS on a weather hazard index | near misses teach, calm distant counties do not (stat. 2; phys. 4; lit. 6.5) |
| pooled RMSE | regime-balanced skill, a loss of the same form, the worst regime | pooled MSE is the metric of one regime (stat. 2) |
| gates G3/G4 from the centred ±7-day rule | a pre-window service rule; the forecast window only masked | the centred rule reads the forecast window (stat. 5; verified in `src/asymode/panel.py`) |
| maximum coverage in the nearest year | minimum coverage | a drop within a year passes a maximum gate (lit. 4) |
| 2015-2025 | 2018-07-12 to 2025-12-31 | coverage imputed and HRRR gaps before it; HRRR v3 starts 2018-07-12 (stat. 5; lit. 4) |
| 144 h after the first warning | origin 6 h before the expected onset, every regime | warning leads differ by regime (stat. 2c) |
| macro-region strata | region as a sort key; five geography axes audited in every regime; a shoulder-season quota | geography x weather needs geographic spread inside each regime (phys. 2) |
| 18 sealed episodes, rules open | C sealed by an extraction rule, K ≤ 4, a prospective tranche, used windows kept out | the file drawer and contamination (stat. 4) |
| operator outages "from records" | a registered list from type, footprint and time | (stat. 5; lit. 3) |

## 14. Review suggestions not adopted, and why

* **Regime-specific windows** (phys. 4). The horizon stays 144 h, because the forecast problem has one horizon.
  Truncated restorations are a stated limit.
* **A geography stratum specific to each regime** (phys. 2). It is replaced by the same five axes audited in every
  regime, so that no regime-geography pair is favoured.
* **Allocation above a floor by N_h^0.5** (stat. 3). With 15 and 20 systems per regime, the budget leaves nothing above
  the floor. The frame counts N_h enter the secondary estimand instead.
* **The cube method** (stat. 3). Systematic PPS with implicit stratification balances on the sort keys and gives exact
  first-order π; the audit of §4.6 covers the rest.
* **Keeping canopy-poor states that fail coverage, with coverage weights** (phys. 2). Not adopted: an under-counted
  numerator is a bias that no weight removes. The low-canopy tercile is guaranteed by §4.6 from states that pass.

## Amendment 1 (2026-09-26 05:08 EDT): frame construction details, before any draw

Written after a first, outcome-blind build of the frame (`panel_v1/build_frame.py`; no outage value read, no draw
made). Each item either pins down an implementation detail that §3 left open or corrects a rule that the build showed
to be wrong.

1. **Tropical products join their storm by event number.** The VTEC event number of a TR, HU, SS or EW product
   names its storm: 1000 + n for Atlantic storm n, 2000 + n for eastern Pacific storm n (95% of the frame's tropical
   W events match a HURDAT2 storm this way). Such an event joins that storm's TC system whatever its time or distance.
   A storm that has such events but no span forms a TC system from its events alone, with onset at their earliest
   begin; this covers potential tropical cyclones that were never named, and tracks that stay beyond 300 km. Other
   phenomena keep the rule of §3.4. Reason: tropical warnings come up to two days before tropical-storm winds, and
   often while the centre is beyond 300 km or the track has ended. In the first build, the time-and-distance rule
   split Helene (2024), Lee (2023), Henri (2021), Hilary (2023) and others into a TC system plus a separate "tropical"
   system. A wider time window (72 h) fixed Helene and Henri, but not Lee (the storm passed offshore) or Hilary (its
   track ends at dissipation).
2. **County vintage.** The units are the 2020 counties (Census Gazetteer and cartographic boundaries 2020), which
   EAGLE-I reports through May 2025; Connecticut keeps its eight counties. EAGLE-I switches Connecticut to planning
   regions between 2025-05-29 and 2025-06-02, and the two vintages do not nest. Connecticut county-events whose window
   ends after 2025-05-29 are therefore excluded. VTEC keeps the eight Connecticut county codes through 2025. County
   adjacency (shared boundary or corner) is computed from the 2020 boundaries, since no 2020 adjacency file is
   published.
3. **Exposure.**
   * SV, TO and FF use their initial polygon (STATUS NEW in `<year>_tsmf_sbw.zip`).
   * Other products use their UGC rows. County UGCs give a = 1. Zone UGCs use the IEM UGC snapshot of the month of
     INIT_ISS; a zone missing from the snapshot falls back to the NWS zone-county correlation file with a = 1. The
     share of fallback rows is recorded.
   * The 3 km nodes are the WorldPop 2020 1 km cells aggregated to HRRR cells within each county (`build_nodes3k.py`,
     all 3,108 counties).
4. **Time fields.** Checked on the 2018 and 2019 tables: ISSUED is the VTEC event begin and INIT_ISS the first
   issuance (WS.W median ISSUED − INIT_ISS 13 h in 2018, 14.7 h in 2019), as the IEM documentation states.
5. **Gate S-b** reads only the national set of collection-run timestamps (timestamps with at least 5 reporting
   counties), a property of the calendar, not of any county's records.
6. **Build order.** Gate S-d needs the operator exclusions (§7), so steps 1 and 2 of §12 are committed together: the
   frame file is built and committed only after `operator_exclusions.csv` is.

## Amendment 2 (2026-09-26 05:37 EDT): the v1 review, before any draw

From the independent review `contrib/REVIEW_dataset_design_v1.md` (items B1, S1-S13, M1-M14). No draw has been made
and no outcome read. Every item below is adopted unless it says otherwise.

**B1 (blocking): no input from outage records inside the frame.** The host's county context takes SAIDI from EIA-861
**2017**, the last year before the frame; the 2023 value is computed from 2023 interruptions. The construction is the
same as for 2023 (first SAIDI column, utility mean, counties of the utility's service territory, county mean). A county
whose utilities did not report reliability in 2017 (275 counties) takes its state's median 2017 county value, flagged
(`panel_v1/statics_v1.py`). No model input may be computed from outage records dated inside 2018-07-12..2025-12-31,
other than the county-event's own prefix.

**Sets and sources (S1).**
* In §4.2 and §4.4(4), a system's counties are its S1, S2 and S3 counties in the frame file (before G3-G5 and before
  county sampling). A used window's counties are its listed footprint.
* In §9.1, a system's counties are its sampled counties. In §3.4(2), they are its S1 counties.
* The used windows are every event in `experiments/open_gcrk_20260919/selected_events*.json` and
  `experiments/geo_weather_20260924/selected_events*.json`, and every main-line `data/interim/panel_<day>.npz` (only its
  county list and timestamps are read), as compiled by `panel_v1/used_windows.py`: 125 windows.

**Frame rules (S2, S4, S6, S12, M1, M2).**
* **Compound (S2).** A non-TC system is compound if at least 20% of its S1 customers are in counties whose multi-label
  vector holds a regime other than the system's regime. This covers ice then wind in the same county.
* **Coverage (S4).** S-d and M_s use the year's **maximum** coverage ≥ 0.8. At the county-sample step (step 4 for D, C's
  build for C), a state whose minimum is below 0.8 keeps its counties only if at least 80% of its 2024 modelled customers
  are in counties with a positive record in the 30 days before the window. That read is pre-window and takes the C mask
  (S8). The frame lists every state-year concerned. G2 of §6 is replaced accordingly.
* **S-e (S6).** National collection runs cover the origin hour and at least 90% of the prefix hours.
* **Linking (M1).** For linking, an event's interval ends at min(last EXPIRED, INIT_ISS + 168 h). Long areal flood
  warnings and a few record errors chained unrelated storms over up to 92 days.
* **TC families (M2).** A graph family linked by §3.4(2) to an event of a TC system joins that TC system's family, not
  its system.
* **As built (S12).**
  * TC systems are not segmented.
  * S2 counties are ring counties, either touched by the system (a < 0.1) or under a listed A or Y product of any system
    during the forecast window. Advisory counties outside the ring are not in the domain.
  * A W event with no county at a ≥ 0.1 creates no system and acts as an advisory.
  * M_s leaves out operator-excluded counties.
  * Typing and region use all S1 customers, ungated.
  * The season class uses the origin month (UTC). Sort ties keep frame order.
  * S1 membership includes zone extensions, so S1 partly reacts to damage reports. The stratum breakdown of §1 is
    therefore descriptive, not a covariate breakdown. S1 remains a sampling stratum with known π, so the domain estimand
    stays unbiased.

**Coverage audit (S3, M8).**
* **(i) Counties.** The audit uses the S1-S3 counties that pass G1-G2 and are not operator-excluded.
* **(ii) Cells.** A system is in a deficient geography cell if at least half of those counties lie in the deficient
  tercile.
* **(iii) Supplements.** Conditions are checked in the order relief, canopy, drainage, density, coast, season,
  compound. Each failing condition gets one PPS draw of 3 systems (fewer if the cell has fewer), from the cell's D-frame
  systems not yet in D. A D-wide compound failure is supplemented in the regime whose D frame holds the most compound
  systems. A condition that still fails is recorded, and nothing more is drawn.
* **(iv) π.** Each system's D π is the share of 10,000 replays of the D draw, the audit and the supplements that
  include it (uniform starts, the realised C draw fixed). This replaces 1 − (1 − π₁)(1 − π₂).
* **Axes (M8).** Computed by `panel_v1/county_axes.py` for all 3,108 counties:
  * relief: the SD of 3DEP elevation over land pixels, 150 m;
  * canopy: mean USFS tree canopy cover;
  * drainage: the share of gNATSGO map units drained somewhat poorly, poorly or very poorly;
  * density: log modelled customers per km² of land (Gazetteer 2020);
  * coast: km from the Gazetteer internal point to the Natural Earth 1:10m coastline.

**C's population (S5).** C's inference population is the frame minus the used systems. The draw log reports, per
regime, the used share of systems and of M. D results shown beside C results are also reported on D minus used
systems. §1's claim population for C says so.

**Estimand and loss (S7, M9-M13).**
* **(a) Regime of a county-event.** In MSE_r, Z_r and the loss, a county-event belongs to its system's regime. Labels
  are used for breakdowns only.
* **(b) R.** Every loss, Stage 0's included, averages over the five regimes (R = 5). A regime with Z_r = 0 in a run's
  training folds is left out of that run. The primary estimand averages over the headline regimes.
* **(c) Frame weighting.** The frame-weighted estimand pools the regimes with the design weights w of §5.4. N_h enter
  only a post-stratified check, with weights w · N_h / Σ_{sampled s in h} 1/π_s.
* **Stage 0 details (M9).** Bootstrap: 2,000 draws, seed 20260924. Non-inferiority is judged on the seed-averaged point
  estimate. The worst regime is taken over all five.
* **Trimming (M10).** The median is over the county-event weights of the regime in the tranche. Trimmed and untrimmed
  values are both reported per regime.
* **Groups (M11).** The "largest system" of a group has the most sampled county-events; a group's origin is its
  earliest origin.
* **Power (M12).** The Kish design effect of w on C's sample is recomputed, outcome-free, before the confirmatory
  registration.
* **Loss weighting (M13).** The Z_r-normalised loss matches skill against zero. Gain against the host weights regimes by
  1/SSE_r(host) instead.

**Sealing (S8, S11, S13, M4).**
* **Skipped reads (S8).** Every EAGLE-I read outside the confirmatory script (G3 of D, the §11 diagnostic, any audit)
  skips the county-hours of C windows: C's S1-S3 counties × [window start, window end + 7 d]. A lookback cut this way
  uses the hours that remain. No table spanning a C window is materialised. The G3 lookback switch of §6 is decided on
  D's pre-window records and then applies to every tranche.
* **National collection runs (M4).** The national collection-run set (S-b, S-e, G4, G5, the forecast mask) counts every
  row, zero and blank rows included, and is not a read of any tranche. A 5-county threshold detects only a total
  collection failure, not a partial one (a stated limit).
* **Spent (S11).** One confirmatory registration per tranche. A tranche is spent when that registration's sequence
  stops.
* **Committed files (S13).**
  * The frame, the exclusions and the tranche list (system, regime, family, tranche, π) are force-added under
    `data_provenance/frame_v1/`, each with its SHA-256, at the steps §12 names.
  * `panel_v1/draw.py` is committed before it runs, and it runs once.

**Weather and screening (S9, S10).**
* **Missing HRRR hours (S9).** They enter the HRRR inputs as zeros, as in H2b, and are listed. They never change the
  loss or metric mask. This replaces the masking sentence of §8.
* **Keep and discard (S10).** Program.md rule 3b gets this criterion.
  * Keep: seed-averaged over three seeds on the five event folds, the regime-balanced gain exceeds 1% against the host
    and is positive against the twin. Each of the two gains needs its 95% family-cluster bootstrap interval above zero.
    No non-headline regime may lose more than 2%.
  * Discard: a single-seed, five-fold screen whose regime-balanced gain against the host is ≤ 0.

**Operator exclusions (M7).**
* Rows whose narrative attributes the shed to damage in the reporting utility's own system are flagged and kept.
* A PSPS event without a county list excludes the IOU's EIA-861 territory for its window.
* All times are converted to UTC.
* The 2024-2025 gap in DOE-417 summaries is a stated limit.

**Diagnostics and limits (M5, M6, M14).**
* **§11 addition (M5).** The share of blank-masked county-hours for h_c ≥ 1 against h_c < 1, on D.
* **Outside the claim population (M6).** Windows containing Dec 31 of 2018-2021 (S-b); HRRR-gap windows (S-c);
  Connecticut after 2025-05-29; used systems (for C); operator-excluded county-events.
* **Storage (M14).** A CONUS window is 365 MB (main 236 + extra 108 + gust 22), so deletion after features is mandatory
  at about 28 GB free.
* **P (M14).** P is the whole next EAGLE-I release (all qualifying systems), not a draw.
* **SQ.W (M14).** Snow squall warnings (polygons from 2018) are not sampled (brief; §14).
* **EAGLE-I versions (M14).** The local 2018-2021 files differ from the v4 files only in the header name of the count
  column (`sum` for `customers_out`, 10 bytes). The first and last 200 bytes were compared for 2018, and the sizes for
  every year. The 2024 file matches v4 exactly.

**§14 additions (review section 4), not adopted and why.**
* **Ring π graded by distance.** Replaced by PPS on h_c, which grades by weather, not distance.
* **Heat and flood-only negative-control strata.** Heat is not sampled; flood-only systems are full heavy-rain systems,
  judged by the Stage 0 rule.
* **2015-2018 as a secondary frame.** Dropped: coverage is imputed, and HRRR has gaps.
* **The host must beat zero in every regime.** At least three headline regimes are required; the rest are
  non-inferiority strata.
* **A frequency-weighted headline (physical review).** The statistical review's regime-balanced headline is followed.
  The frame-weighted value is secondary.
* **A hazard vector per county-hour.** Kept per county-event; S2 fixes the compound case.
* **Damaging and quiet sides reported separately.** Replaced by h_c classes, never outcome subsets.
* **Sub-regimes (derecho against pulse storms, wet against dry snow).** Not stratified; the dictionary is meant to
  separate them.
* **HRRR results before and after 2019.** Split at HRRR v4 (2020-12-02) instead.
* **Membership from products issued before the hazard (statistical revision 3).** Not adopted: zone extensions count
  (S12).
