# Open-data replication: AsymODE host (W) vs AsymODE + GCRK

Plan and rules: `PREREG.md` (fixed before any training). Results and the provenance of
every number: `RESULTS.md`. Figure captions: `CAPTIONS.md`. What changes in the paper:
`RESULTS_FOR_MANUSCRIPT.md`.

## Pipeline (run from the repository root with `./.venv/bin/python`)

| step | script | output |
|---|---|---|
| 1 event selection (public metadata only) | `select_events.py` | `event_selection.csv`, `selected_events.json` |
| 2 ERA5 for windows not on disk | `fetch_era5_windows.py --events ...` | `data/raw/era5/`, `data_provenance/era5_fetch_log.jsonl` |
| 3 216-hour panels + drivers | `build_panel216.py` | `data/interim/open_gcrk/panel216_*.npz`, `panel_gates.csv` |
| 4 geography from public rasters and soils | `build_geography.py` | `data/interim/open_gcrk/geography.parquet`, `data_provenance/geography_*` |
| 5 model inputs | `build_features.py` | `data/interim/open_gcrk/features.npz` |
| 6 splits | `run.py splits` | `splits.json` |
| 7 training (W, GCRK; seeds 0-4) | `run.py queue --design main`, `run.py queue --design loeo` | `runs/open_gcrk_20260919/<design>/seed*/fold*/<arm>/` |
| 8 TimesFM zero-shot | `timesfm_baseline.py --runtime <TimesFM code+weights>` (a Python with torch, safetensors) | `runs/open_gcrk_20260919/timesfm/` |
| 9 tables | `evaluate.py main`, `evaluate.py loeo` | `results/` |
| 10 diagnostics (inference only) | `diagnostics.py concentration|swap|figure3` | `results/`, `results/figure_data/` |
| 11 figures | `figures.py` | `figures/*.pdf`, `figures/*.png` |
| 12 source checksums | `record_sources.py` | `data_provenance/SOURCES.md` |

The model code lives in `src/asymode/` (`asym_host.py`, `gcrk.py`, `gcrk_train.py`); the
kernel's equivalence to the frozen reference layer is `tests/test_gcrk_equivalence.py`.

## Large files (not in git)

* `data/raw/era5/`, `data/raw/geography/` — raw public downloads (checksums in `data_provenance/`)
* `data/interim/open_gcrk/` — panels, geography table, model inputs
* `runs/open_gcrk_20260919/` — per cell: `final.pt` (weights, optimiser-free snapshot with the
  fitting-set standardisation and the kernel's calibration), `outer.npz` (OUTER hourly
  rollouts with the kernel exit open and closed), `selection_trace.csv`, `DONE.json`;
  `timesfm/timesfm_*.npz`; worker logs under `<design>/logs/`.

Every script is single-threaded (OMP/OpenBLAS/MKL/vecLib = 1); the queue runs at most five
workers on this 8-core machine.
