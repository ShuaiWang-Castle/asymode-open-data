# T06 reproduction

Use the authorized public development inputs and immutable T03 cache; raw data,
county forecasts and trained weights remain outside the git repository.

```bash
python experiments/geo_weather_20260924/t06_featured_subset.py build
python experiments/geo_weather_20260924/t06_featured_subset.py preflight
python experiments/geo_weather_20260924/run_t06.py
python experiments/geo_weather_20260924/score_t06.py
python experiments/geo_weather_20260924/audit_t05_t06.py t06
python experiments/geo_weather_20260924/report_t06.py
```

Do not rebuild an existing registered cache. Training skips completed jobs only
after checking source/cache receipts. A partial checkpoint/receipt set is rejected
for inspection rather than overwritten. Any budget other than 600 steps, alternate
seeds, altered selection or changed source hashes are separate experiments.

The dataset selection uses coverage, wind and geography only. Keep all four
selected windows, all five model arms and all three seeds. This is a county-held-out,
supplied-weather hindcast on examined development data, not event-held-out or sealed
confirmation. CRK retains the full pre-existing controlled-response architecture.
CRK_SHUFFLE uses separate within-state geography mappings in each fold role.

Runtime versions and input/source SHA256 are recorded in build_audit.json and
continuation_environment.json. This
continuation used Python 3.12, PyTorch 2.14.1+cpu, NumPy 2.3.5; pandas/scipy versions
are available from continuation_environment.json. Controlled-kernel tests pass in this
environment. Aggregates and all negative results are committed; private forecast
and checkpoint files reside in the sibling t06_artifacts directory.
