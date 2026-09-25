# PREREG W2 — one hypothesis for the elevation-phase mechanism on a larger ice-storm panel

Written 2026-09-25 05:25, before any W2 panel, feature or outcome exists. Changes after this point are appended as
dated amendments, never edited in place.

## Why

On W1 (8 ice-storm events) the sub-grid (elevation-band) contrast of the exposure-integrated hazard features is
material only for near-freezing precipitation (F0, `results/F0_w1/`), the screens are dominated by a few
county-events (fold and seed swings of about +-10%, `RESEARCH_LOG.md` W1a-W1f), and the residual-alignment audit
F1 finds the contrast not testable: no feature reaches 20 effective event x state clusters (`results/F1_w1/`).
The strongest feature there, descriptively and not as evidence, was the 48-h memory of precipitation under the
wet-bulb hat at -1.5 C. W2 tests that single feature on more independent events.

## Panel W2 (metadata rules only)

* Pool: ice-storm episodes from NOAA Storm Events 2018-2024 (the years on disk): for day d, the counties with an
  Ice Storm report beginning in [d, d + 3 days); greedy local maxima at least 10 days apart with at least **20**
  counties (W1 used 40).
* Exclusions as W1: windows overlapping the twelve-event wind panel; 2021-02-13 (outages dominated by operator
  load shedding); windows without an EAGLE-I record on disk (2022-11-13 to 2022-12-31: the local 2022 file ends
  on 2022-11-12).
* Window, footprint (winter report types in forecast hours 72-215), data gates G1-G4, the node-availability rule,
  round-2 inputs with area weights, county-grouped splits (seed 20260919): exactly as W1 (`build_winter.py`).

## Hypothesis H1 (single feature, one-sided)

Feature f = `p_tw-1.5*one@48` of `build_eih.py`. Contrast C1 = f(quad) - f(pop): the elevation-band nodes against
the cells without bands, both population-weighted.

* **H1a (no training).** On W2, C1 aligns positively with the held-out residual (y - P) of the base W+Cin (seed 0,
  900 steps, five county-grouped folds): the cluster-robust score test of `audit_f1.py` part (c) for this one
  feature (no max-T or Bonferroni: one registered test), clusters = event x state, one-sided alpha = 0.05.
  **Eligibility:** at least 20 effective event x state clusters for this feature (as audit_f1 revision_2 counts
  them); if fewer, the result is reported as "not testable" and not as a pass or a fail.
* **H1b (trained, only if H1a passes).** A competing-hazard arm with this single feature (W+Cin+H, one beta >= 0,
  slow clock) using the quad variant beats the same arm with the pop variant and the base, on the county-grouped
  screen (folds 1-2) averaged over seeds 0-2 (seed-averaged predictions, `evaluate_seeds.py`), and on the
  event-grouped design: each number with an interval below zero.

## What would falsify it

H1a not passing with >= 20 effective clusters; or H1b's quad arm not beating its pop twin (the elevation bands add
nothing beyond the cell-level feature); or a gain that appears only in the county-grouped design.

## Amendment 1 (2026-09-25 05:40, before any W2 panel or outcome; after the formal contributor's checklist)

1. **Disjoint from W1.** The feature was chosen on W1, so the confirmatory panel W2 excludes W1's eight events
   (overlap would carry the winner's curse). W1 + W2 pooled is secondary only. Disjoint candidates under the rule
   above: 2018-02-19, 2019-01-10, 2020-11-09, 2020-12-30, 2022-02-22 (Storm Events 2018-2024 on disk), plus the
   2025 episodes with >= 20 counties (2025-01-03, 2025-03-28; EAGLE-I 2025 is public on figshare), plus 2014-2017
   episodes once those Storm Events years are added with the same code. Seven known so far.
2. **Feature frozen:** variant quad against pop (contrast C1), hat centre -1.5 C, half-width 1.5 C, unit-gain
   exponential filter tau = 48 h, modulator one, as in build_eih.py at commit 6a9aa5a; test code audit_f1.py at
   commit 6a9aa5a plus its announced --single mode.
3. **Base frozen:** W+Cin, 900 steps, seed 0, county-grouped main design of W2 (five outer folds, seed 20260919);
   out-of-fold predictions of one seed.
4. **Statistic:** one-sided (positive), alpha 0.05, a single test; nuisance = event x state x relief-tercile blocks
   + base window-mean prediction + log customers + the pop variant's own window-mean intensity of the feature.
5. **Eligibility (decided before reading):** >= 20 effective clusters at event x state and at county level, and
   >= 8 effective events (G_eff,event, as audit_f1 counts it); the exact event sign-flip p is reported beside the
   multiplier p, and a pass needs both below 0.05. Otherwise the result is "not testable".
6. **Null rule:** if the synthetic-field false-positive rate of this single test exceeds 0.10, the synthetic-field
   p replaces the multiplier p (as in F1).
7. **Power before outcomes:** plant the feature's effect in W2 at half of W1's descriptive part correlation
   (0.045) and require power >= 0.8 under this design; otherwise W2 is declared uninformative before reading.
8. **Decision table:** pass -> the elevation-phase channel is supported on independent events and H1b (trained,
   seeds 0-2, event-grouped) is run; fail with power >= 0.8 -> the channel is absent at a size that matters on
   ice storms too, and the framework's claim rests on its interface and audits only; not testable -> more
   independent near-freezing events are needed, no claim either way.

## Amendment 2 (2026-09-25 09:25, before any W2d feature exists)

* Test code frozen at commit 239fd7d (`audit_f1.py`, sha256 72ebcfcd...6814 of the working-tree file as sent by the
  formal contributor): `--single "C1:p_tw-1.5*one@48" --min-eff-events 8 --require-event-flip --power-target 0.045`,
  raw scale. Command: `audit_f1.py --features data/interim/geo_weather/features_w2d.npz --splits
  experiments/geo_weather_20260924/splits_w2d.json --base "runs/geo_weather_20260924/w2d_base/fold{fold:02d}/outer.npz"
  --eih-prefix eih_w2d_ --single "C1:p_tw-1.5*one@48" --min-eff-events 8 --require-event-flip --power-target 0.045
  --out experiments/geo_weather_20260924/results/F1_w2d`.
* W2d events after the gates (no outcome read): 12 of the 18 selected, with 106 to 696 county-events each; the three
  early-2014 episodes have no EAGLE-I record (the 2014 file starts 2014-11-01) and 2014-12-31, 2015-12-26 and
  2020-12-30 keep no county after the gates.
* Known before any feature (design only, from the formal contributor): with these event sizes the effective
  number of events is capped at 9.28 for any feature and reached only by a feature spread in proportion to units;
  on W1 the registered feature had 3.5 effective events of 8. The likely outcome is therefore "not testable".
  **The criteria of amendment 1 are kept unchanged**; relaxing the event rule would leave power as the binding
  constraint, and only more independent near-freezing events would raise both.

## Erratum on times (2026-09-25 03:45 EDT)

The clock times written above ("05:25" in the header, "05:40" in amendment 1, "09:25" in amendment 2) were
estimated without reading the system clock and are wrong. The authoritative times are the commits (EDT): the
header c8dc7d4 at 02:02, amendment 1 2283a4e at 02:03, the frozen test code 239fd7d and amendment 2 b96d9f5 at
03:06. The order is as registered: the W2d features were built after 03:10 (log experiments/geo_weather_20260924/
logs/chain_w2d.log), and the test ran at 03:42 with the frozen code.

## Outcome of H1a on W2d (recorded, not an amendment)

`results/F1_w2d/H1a.md` (formal contributor, audit_f1.py at 239fd7d, HEAD 2645983): 4,049 county-events, 12
events, 272 event x state clusters; the contrast is nonzero in 261 blocks and in all 12 events; effective clusters
event x state 30.5, county 114.3, events **7.5 (< 8)**: **not testable**, decided from the design; power and the
test were not computed, and no residual-derived number exists. By the decision table: more independent
near-freezing events are needed, no claim either way. The W2d residual alignment stays unread, so an enlarged
panel that contains W2d can still serve as the confirmatory test if its amendment (new events, criteria
unchanged) is committed before anything is read.
