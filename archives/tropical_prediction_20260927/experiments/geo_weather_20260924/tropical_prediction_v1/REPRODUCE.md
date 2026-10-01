# Reproduce the tropical prediction campaign

Run commands from the repository root. The fixed training source and protocol
were committed at `5119ec898baa2f44fd095c07e6806aefe91cc15b`, before fitting.
The output scorer verifies the source SHA-256, prepared-data SHA-256, every
outer row, all 145 fit records, neural seed averaging and inner blend choices.

## Environment

Python 3.12, NumPy, pandas, SciPy, PyTorch, LightGBM and PyArrow are required.
Matplotlib and a LaTeX installation with booktabs/multirow are needed only for
the figures/report. Exact observed package versions and execution limitations
are recorded in `runs/tropical_prediction_v1/environment.json` in the handoff.
Training used CPU execution with one thread per fit. Limit concurrency to two
folds on an 8 GiB machine; the two large pooled tree feature matrices are kept
in memory during each fold.

For a single command that resumes all five folds with two workers and then
verifies/summarizes the completed campaign:

```bash
python experiments/geo_weather_20260924/tropical_prediction_v1/run_all.py --workers 2
```

Do not start this command while another copy of the campaign is running.

## Data and fitting

The full repository contains the public source products needed to rebuild:

```bash
python experiments/geo_weather_20260924/tropical_prediction_v1/prepare.py
```

The handoff also includes the prepared tropical development NPZ, preserving
its original hash. With that artifact, no download or source-data rebuild is
necessary. The confirmation tranche is neither required nor included.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for fold in 1 2 3 4 5; do
  python experiments/geo_weather_20260924/tropical_prediction_v1/run_campaign.py --fold "$fold"
done
```

Completed fits are cached. Re-running an unchanged completed campaign skips
the fits. To train anew, use a separate checkout/output directory or move the
existing `runs/tropical_prediction_v1` directory aside. Do not change the
training source while using existing caches: cache identities include the
source hash. There are 70 neural and 75 tree fits in total, without selecting
or dropping seeds on outer performance.

## Verify and regenerate deliverables

```bash
python experiments/geo_weather_20260924/tropical_prediction_v1/summarize_campaign.py
python experiments/geo_weather_20260924/tropical_prediction_v1/replay_checkpoint.py --tag f1_outer_host_s0
python experiments/geo_weather_20260924/tropical_prediction_v1/replay_checkpoint.py --tag f1_outer_fusion_s0
python experiments/geo_weather_20260924/tropical_prediction_v1/plot_campaign.py
python experiments/geo_weather_20260924/tropical_prediction_v1/build_paper_report.py
cd experiments/geo_weather_20260924/tropical_prediction_v1/results
pdflatex -interaction=nonstopmode -halt-on-error development_report.tex
pdflatex -interaction=nonstopmode -halt-on-error development_report.tex
```

`overall_metrics.csv` contains full-path metrics, including weighting
sensitivities. `horizon_metrics.csv` contains point horizons, not rolling
future-start scores. `paired_family_bootstrap.csv` uses positive percentages
for RMSE improvement. All tables retain the zero forecast and weak systems.
`ensemble_weights.csv` records one fixed trajectory weight vector per outer
fold. The report describes uncertainty and the conditional-hindcast scope.

Saved predictions and checkpoints are reproduction artifacts, not extra
training data. Loading PyTorch checkpoints requires trusting their source;
the replay script is intended only for this campaign's locally generated
checkpoints. File hashes in the handoff manifest identify the exact artifacts.

## Branch integration

The handoff patch is based on commit
`493aa54fb9233f27c4b6a10b5bacf3c1592f33ba` of
`research/tropical-evidence-20260926`. It includes the earlier pilot archive,
the new data preparation, the fixed campaign and its completed outputs. Apply
it only to a clean checkout at that base using `git am`; if the remote branch
has moved, review/rebase the patch rather than overwriting its new commits.
The archive itself does not grant repository write access.
