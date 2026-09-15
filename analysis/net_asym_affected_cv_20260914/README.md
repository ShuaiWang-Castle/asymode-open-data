# NET / ASYM affected-county grouped cross-validation (2026-09-14)

This round compares two recursions on US EAGLE-I outages with given ERA5 reanalysis weather: a single
net-flow recursion (NET) and a two-flow source-pool recursion (ASYM). It was redesigned on the PI's
decisions of 2026-09-14 (`PROTOCOL_ZH.md`, `CONFIG_LOCK.json`):

- Training and inner validation use affected county-events only (event-panel peak >= 1%). The test covers
  every legal window and is reported for all, affected and unaffected county-events.
- Grouped 5-fold cross-validation over the 26 events, stratified by type group, with overlapping events in
  one fold.
- ASYM is split into separate damage and recovery networks, with a damage:recovery ratio of 1:1, 2:1 or 3:1.
  NET has 2, 3 or 4 hidden layers. Every candidate is within 1% of 32,768 parameters.
- Both models use the same learning-rate grid, update budget and early stopping, and both start near
  persistence.
- 180 DEV runs, a per-fold selection locked before any final fit, and 50 final fits. Every checkpoint
  replays bit-exactly.

## Outcome

Delta = MSE_NET - MSE_ASYM, so positive favours ASYM. The decision uses component sign-flip p < 0.05 with a
consistent sign. Details are in `RESULTS_ZH.md`.

| grouping | affected | all windows | unaffected |
|---|---|---|---|
| all 26 events | undecided (-3.89e-04, p = 0.11) | undecided (-2.77e-04, p = 0.21) | supports ASYM (+2.62e-04, p < 5e-06) |
| winter (7 events) | undecided (+6.03e-05, p = 0.75) | undecided (+1.55e-04, p = 0.125) | supports ASYM (+3.03e-04, p = 0.031) |
| wind and tropical (7) | undecided (-1.60e-03, p = 0.063) | undecided (-1.31e-03, p = 0.11) | supports ASYM (+4.23e-04, p = 0.016) |
| convective and flood (12) | undecided (+5.30e-05, p = 0.17) | supports ASYM (+7.09e-05, p = 0.024) | supports ASYM (+1.45e-04, p = 0.0005) |

- **Large wind and tropical events favour NET.** ASYM is worse than persistence in the three largest of
  them.
- **Convective events and six of seven winter events favour ASYM.**
- **Unaffected counties.** Both models predict spurious outages there: NET's MSE is about 320 times
  persistence and ASYM's about 75 times. The ASYM decision reflects smaller spurious outages, not a better
  response.
- **RMSE**, the secondary metric, agrees in direction with MSE in every grouping and population.

## Layout

| path | contents |
|---|---|
| `PROTOCOL_ZH.md`, `CONFIG_LOCK.json` | design, decision rule, revision log |
| `locks/` | `FOLDS.json`, `CV_WINDOW_INDEX.csv.gz`, `SELECTION_LOCK.json` |
| `code/` | fold data, models, training, replay and the pre-registered analysis |
| `tests/` | correctness tests |
| `checkpoints/`, `predictions/` | final checkpoints and test prediction shards per fold, model and seed; truth |
| `results/`, `figures/` | analysis outputs |
| `logs/` | DEV and final run logs, PI decisions and notes, replay record, orchestration scripts |

## Replay and analysis without training

Run these from the repository root:

```bash
python3.11 analysis/net_asym_affected_cv_20260914/code/cv_replay.py --threads 2
python3.11 analysis/net_asym_affected_cv_20260914/code/cv_analyze.py --threads 2
```

## Licences

EAGLE-I 2014-2022: CC BY 4.0. EAGLE-I 2024: no reuse restrictions. ERA5: Copernicus licence, derived
product with attribution. Event types come from the NOAA Storm Events catalogue.
