# T05 repeated-weather experiment

Use the T03/T04 Python environment and existing authorized `data_v1` development inputs. The original data, county predictions and trained checkpoints are not included in this results directory.

```bash
python experiments/geo_weather_20260924/t05_screen.py
python experiments/geo_weather_20260924/t05_repeated_weather.py build
python experiments/geo_weather_20260924/run_t05.py
python experiments/geo_weather_20260924/score_t05.py
python experiments/geo_weather_20260924/report_t05.py
```

`t05_screen.py` audits all 81 existing development windows against fixed weather/coverage thresholds. `selection.json` records every source file hash and the selected eight windows. Training uses seven arms, five global county folds and three seeds, 600 steps per fit. Completed jobs resume from saved predictions. Private artifacts are in the sibling `t05_artifacts` directory, outside the git repository. All comparisons retain every seed and selected window.

The plot uses the four windows with the most cross-cutoff weather peak pairs, chosen independently of model accuracy. It displays weather medians/IQRs and observed county-average outage fractions, not model predictions. The experiment is a conditional hindcast with supplied weather; multiple weather peaks do not by themselves establish distinct named storms. The 2018 Riley/Quinn case is a sourced future data-construction candidate, not part of this trained cohort.
