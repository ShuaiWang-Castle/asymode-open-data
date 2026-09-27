# Tropical evidence (热带证据)

2026-09-26. The first model result on the designed public panel (`../DATASET_DESIGN.md` v1 with amendments 1-3): the
host (W+Cin) forecasts unseen tropical-cyclone systems better than the all-zero forecast, and no other regime clearly.
Everything here is on the **development tranche** (81 systems, 8,457 county-events). The sealed confirmation tranche
was not read. **One seed** (seed 0), five event folds (whole systems held out), 900 steps, design-weighted loss and metric
(`../evaluate_v1.py`). Descriptive, not a registered test: Stage 0's headline rule needs three seeds (DATASET_DESIGN
section 9.3).

## 1. Host against the all-zero forecast, by regime

Design-weighted MSE over the held-out county-hours. 80% family-cluster bootstrap intervals (2,000 draws, seed
20260924; `stage0_host_s0.json`).

| regime | county-events | host MSE / zero MSE − 1 | 80% interval |
|---|---|---|---|
| tropical cyclone | 1,633 | **−11.5%** | [−12.9, −9.4] |
| winter | 1,899 | −2.7% | [−5.3, +0.1] |
| convective | 1,727 | −2.7% | [−5.6, +0.3] |
| synoptic wind | 1,824 | +12.7% | [−1.2, +31.7] |
| heavy rain | 1,374 | +6.3% | [+0.5, +19.7] |

Regime-balanced skill against zero: −0.45%. Only the tropical interval lies below zero.

## 2. Where the tropical skill comes from

Per system (`tables.md`): the gain rests on the five major landfalling hurricanes of the development tranche.

| storm | landfall region | host / zero − 1 | counties with peak outage ≥ 10% |
|---|---|---|---|
| Michael 2018 | Southeast | −14.2% | 36% |
| Isaias 2020 | Northeast | −6.4% | 29% |
| Delta 2020 | Southeast | −9.8% | 32% |
| Zeta 2020 | Southeast | −10.6% | 37% |
| Milton 2024 | Southeast | −14.6% | 31% |

On the weak systems (Alex 2022, Danny 2021, Ophelia 2023, Alberto 2024, an unnamed potential tropical cyclone of
September 2024), the host is worse than zero. These are false alarms on storms with almost no outage, and they matter
little to the design-weighted MSE.

**The skill is not confined to warned counties.**

| stratum | county-events | host / zero − 1 |
|---|---|---|
| S1, warned | 720 | −11.5% |
| S2, advisory or watch | 507 | −12.0% |
| S3, ring, no product | 406 | −9.4% |

**Geography inside the tropical regime.** Terciles are national, over all CONUS counties (`tables.md`).
* The skill is nearly uniform across canopy, drainage and customer density (−10% to −14% in every tercile).
* It is weaker in high-relief counties (−6.9%, against −12% to −13% in flatter terrain).
* 86% of the tropical county-events lie within 303 km of the coast.

## 3. The weather of tropical systems

ERA5, development tranche.
* 24 h maximum precipitation: median 28 mm (90th percentile 76 mm), the most of any regime.
* 15% of tropical county-events have a county-mean rain rate above 10 mm/h in some hour, against 0-3% in the other
  regimes (`weather_v2_by_regime.txt`).
* Gust above the local 2008-2017 98th percentile: 70% of county-events.
* Warm and moist: minimum temperature 16 °C, CAPE about 1,350 J/kg.
* ERA5 at 0.25 degrees smooths the eyewall. The median ERA5 maximum gust is only 16.5 m/s (90th percentile 26.6), so
  the 3 km HRRR source is expected to matter most here.

## 4. The geography of tropical counties

Medians over the development county-events (`geography_by_regime.md`).

| regime | tree canopy % | relief m | poorly drained share | customers per km² | distance to coast km |
|---|---|---|---|---|---|
| tropical | **49** | 17 | **0.24** | **26** | **93** |
| convective | 39 | 26 | 0.18 | 15 | 409 |
| heavy rain | 31 | 27 | 0.22 | 20 | 359 |
| winter | 29 | 39 | 0.15 | 14 | 604 |
| synoptic wind | 16 | 46 | 0.07 | 9 | 724 |

Tropical systems land on dense-canopy, flat, poorly drained, densely served coastal counties. Synoptic wind lands on
open, rugged, well-drained, sparse inland counties. Geography and regime are therefore strongly confounded. Geography x
weather has to be learned within each regime, which is why the panel design audits the geography terciles of every
regime (DATASET_DESIGN section 4.6).

## 5. What this does and does not show

* **What it shows.** On unseen tropical systems, the host's error is about 11% below that of forecasting no outage.
  The gain comes from the large hurricanes, in warned, advisory and ring counties alike.
* **What it does not show.**
  * Whether the result holds over seeds: Stage 0 needs seeds 1-2.
  * Whether it holds on the sealed tranche: not read.
  * Whether geography adds anything: the host uses no physical geography. Its county context is customers,
    rural-urban class, population density, cooperative share, number of utilities and SAIDI 2017.
* **Next.** The pathway arms with hazard dictionary v2, which includes geography x weather at the 3 km nodes
  (`../HAZARD_V2.md`), from ERA5 and then from HRRR. Also the geography-conditioned kernel (GCRK) with its opening
  bounded and unbounded.

Files: `build_tropical_evidence.py` (the tables, from the run outputs), `tables.md`, `tables.json`,
`stage0_host_s0.json`, `weather_v2_by_regime.txt`, `geography_by_regime.md`.
