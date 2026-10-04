# T04 reproduction

Use the same environment and authorized local data/cache as T03. No downloads or training data are bundled here.

```bash
python experiments/geo_weather_20260924/t04_simple_geography.py audit
python experiments/geo_weather_20260924/t04_simple_geography.py check
python experiments/geo_weather_20260924/run_t04.py
python experiments/geo_weather_20260924/score_t04.py
python experiments/geo_weather_20260924/report_t04.py
```

Audit/scope rules were committed before fitting. Defaults read the sibling t03_artifacts/data.npz; T03_ARTIFACTS can override that input path. Private T04 checkpoints/predictions/logs are written to sibling t04_artifacts. Complete jobs are resumed. Reproducing the original run requires the cache SHA in verification.json and the same T03 source/dependencies. All trained arms, seeds and predeclared scopes are retained.
