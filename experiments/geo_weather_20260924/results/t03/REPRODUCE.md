# Reproduce T03

From the repository root, with the existing public `data_v1` panel and weather caches available:

```bash
export T03_ARTIFACTS=/tmp/asymode_t03_reproduction
python3 experiments/geo_weather_20260924/t03_poster_trajectory.py build
python3 experiments/geo_weather_20260924/run_t03.py
python3 experiments/geo_weather_20260924/score_t03.py
python3 experiments/geo_weather_20260924/verify_t03.py
python3 experiments/geo_weather_20260924/report_t03.py
```

Use a clean checkout to preserve published aggregate results: score/report write to `results/t03` and its linked results note. Private model weights, predictions and job logs go to `T03_ARTIFACTS` (default: `t03_artifacts` beside the repository). The runner verifies the compressed cache CRC and SHA256, skips completed prediction files, and exits nonzero if any job fails. Four subprocesses run at once, each with one CPU computation thread. No GPU is required.

Dependencies: Python, NumPy, pandas, PyArrow, PyTorch, SciPy, LightGBM, Matplotlib. Runtime versions and source hashes are recorded in `verification.json`; input hashes, feature lists and missing-hour counts are in `build_audit.json`. The repository's `src/asymode/asym_host.py` and `gcrk.py` are used directly. A successful verification replays four neural and two tree checkpoints from fold 0 and checks their held-out predictions against the saved files.

Read the T03 protocol before interpreting results. This is a conditional hindcast on examined public development data, not a six-day operational forecast or an independent confirmation set. The code uses all 15 tropical windows and global county grouping. Exact +1/+6/+24/+48 lead metrics are not competition suffix metrics. No county-level predictions or model checkpoints are included in this public result directory.

Poster exports: `poster_trajectory_results.pdf` (vector), `.png`, and `poster_table.tex` (requires `booktabs` and `multirow`). The full baseline, candidate, per-event, per-seed, weighting and severity results remain in the CSV tables; the chart shows trained candidates. A stronger ensemble is not by itself evidence that GCRK improves a matched weather-memory model.
