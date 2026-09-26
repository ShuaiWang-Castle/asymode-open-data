# DATASET_DESIGN v0 — which events, states and counties the public panel should contain, and why

Status: draft for review (2026-09-26). Nothing here has been built or tested. Written after the first night, whose
panels were chosen ad hoc (a wind panel from wind-report footprints; ice-storm panels from ice-storm reports) and
whose tests narrowed onto one or two hazards. The goal now is one model of the whole county-level outage system under
many weather regimes, so the panel must be designed for that goal before any test is run on it.

## 1. What the panel is for

Inference target: the hourly outage fraction of US counties over 144 forecast hours after a weather episode begins,
under *any* damaging weather regime (convective wind, synoptic wind, tropical cyclones, winter snow and ice, heavy
rain, heat), with the geography x weather interactions shared across regimes. A panel serves this target if:

1. it covers the regimes and regions the model will be used on, in proportions that let every regime be learned and
   tested (a panel dominated by one regime teaches that regime);
2. it does not select on the outcome, or selects in a way that can be undone (weights, controls);
3. its labels are clean enough (EAGLE-I coverage, denominators, collection artefacts) and its outage mechanism is
   weather damage, not operator action;
4. it supports held-out tests of the claims we want to make (new counties, new episodes, new regions), with enough
   effective clusters for the tests to have power;
5. it fits the compute (a training run on this machine handles about 10,000-12,000 county-events in 40 minutes).

## 2. Why not all data and all states

* Compute: the full frame (2015-2025, all CONUS counties, all episodes) is two orders of magnitude beyond one machine.
* Labels: EAGLE-I coverage is below 80% of customers in several states (2022 maximum share: NE 0.48, KS 0.59, ND 0.66,
  WY 0.71, TN 0.75, MT 0.76, MS 0.77); denominators are one 2024 snapshot; collection artefacts are heaviest in storms.
* Mechanisms: operator-initiated outages (public-safety power shutoffs during fire weather, rotating load shedding in
  grid emergencies) and planned outages are not weather damage; they are a different system.
* Diminishing returns: most county-events are small convective days; after a point, more of them adds little and
  unbalances the regimes.

## 3. Sampling frame and episode definition (outcome-independent)

* **Episodes from NWS warnings, not from damage reports.** Storm Events reports partly encode the damage (many wind
  reports are "trees down" reports), so selecting on them selects on the outcome. NWS watches and warnings (the VTEC
  archive of the Iowa Environmental Mesonet) are issued in real time before the damage: an episode is a cluster of
  warnings of one regime in space and time; its window starts 72 h before its first warning (prefix) and its forecast
  window is the next 144 h, as before.
* Regime of an episode = the warning types that dominate it: Severe Thunderstorm / Tornado (convective), High Wind
  (synoptic wind), Tropical Storm / Hurricane (tropical), Winter Storm / Ice Storm / Blizzard / Winter Weather
  (winter), Flash Flood / Flood (heavy rain), Excessive Heat (heat). Storm Events is kept only as a label for
  diagnostics, never for selection.
* Years 2015-2025 (EAGLE-I on disk: 2014-11 onwards, with 2022-11-13..2022-12-31 missing locally; HRRR from 2014-07,
  with gaps in 2015-2017).

## 4. Stratification and allocation

* Strata: regime (6) x macro-region (5: Northeast, Southeast, Central, Great Plains + Mountain West, Pacific) x period
  (2015-2019, 2020-2025). Not every cell exists (tropical in the Pacific); empty cells are recorded, not filled.
* Allocation: equal numbers of episodes per regime (not proportional to frequency), spread across regions and
  periods; within a stratum, episodes are drawn at random from the whole size distribution (warned area terciles), not
  only the largest, so ordinary storms and quiet outcomes are represented.
* Target size: about 36 development episodes (6 per regime) and 18 sealed confirmation episodes (3 per regime), each
  with about 150-400 counties after the gates: about 9,000 development and 4,500 confirmation county-events.

## 5. Counties within an episode

* Exposed counties: those under a warning of the episode's regime during the forecast window.
* Controls: a random 25% extra sample of counties in the same region with no such warning (seeded), so the model
  sees weather without damage and the metric sees quiet counties; they carry a known sampling weight.
* Data gates as before (denominator >= 500 customers, state coverage >= 0.8 in the nearest coverage year, the origin
  hour observed, >= 90% of prefix and forecast hours observed), plus: exclude a state-episode if an operator-initiated
  shutoff or rotating outage is on public record for it (utility or regulator reports), decided from the record, not
  from the outcome.

## 6. Splits and sealing

* The 18 confirmation episodes are drawn first, by the same stratified rule and a separate seed, and sealed: their
  outcomes are not read until a pre-registered confirmatory test is committed.
* Development designs: event-grouped folds (whole episodes held out; primary, because the claim is about the system
  on new weather), county-grouped folds (secondary), leave-one-region-out (transfer).
* Every trained comparison uses three seeds and seed-averaged predictions; the all-zero forecast and the host are
  reported beside every arm.

## 7. What changes for the model

The dictionary of the exposure-integrated hazard must cover every regime (gust, synoptic wind duration, convective
proxies, liquid and frozen precipitation phases, accretion loads, heavy-rain accumulations, heat), and HRRR becomes a
candidate weather source for the whole system (host inputs as well as the pathway), since the audits found the
information in the weather source rather than in sub-county geography.

## 8. Open questions for the reviewers

1. Warnings as the episode definition: does it remove the outcome selection enough, and how should overlapping
   regimes (a tropical storm with tornado warnings) be typed?
2. Controls at 25%: enough to teach the quiet side without swamping the damaging side; how should the metric weight
   them?
3. Equal allocation per regime vs the operational frequency: which should the headline metric use (report both)?
4. Is 36 + 18 episodes enough for event-level power (effective events >= 8 per claim), given the uneven sizes?
5. Which public records identify operator-initiated shutoffs reliably enough to exclude them?
