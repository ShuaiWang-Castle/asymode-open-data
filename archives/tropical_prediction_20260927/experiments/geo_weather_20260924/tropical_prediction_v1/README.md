# Pure tropical prediction: continuation on data_v1

## Completed campaign (September 27, 2026)

All 70 neural and 75 tree fits completed, covering five nested event folds and
three final neural seeds. See [the Chinese report](results/REPORT_zh.md),
[the PDF report](results/development_report.pdf),
[the fixed protocol](RUN_PROTOCOL.md) and [reproduction instructions](REPRODUCE.md).

For the complete 144-hour path, design-weighted pooled RMSE is 0.05805 for
zero, 0.04858 for the matched host, 0.05073 for the joint GCRK-rate model,
0.04813 for the dual-weather residual tree and 0.04584 for the inner-selected
ensemble. The ensemble's RMSE gain is 21.03% over zero and 5.65% over the host;
the latter's paired family-bootstrap 95% interval is -4.24% to +17.02%.
Its MAE is 0.01108, versus 0.01078 for the host. The absolute-loss dual-weather
tree has the lowest MAE, 0.00954. All candidates and unfavorable comparisons
are retained. These are conditional-hindcast development results, not sealed
confirmation or operational-weather results.

The PI requested that the completed three-event pilot be brought onto
`research/tropical-evidence-20260926`, and that the tropical prediction work
continue from this branch. This new development track uses the committed
`data_v1` rather than the older daily catalogue. It does not amend the parent
five-regime experiment or its registered confirmation criteria.

## What has been integrated

The previous pilot's scripts, pre-fit protocol, training records, selection
records and results are preserved at:

- `experiments/tropical_prediction_pilot.py`
- `experiments/summarize_tropical_pilot.py`
- `docs/TROPICAL_PREDICTION_PILOT.md`
- `docs/TROPICAL_PILOT_REPRODUCE.md`
- `results/tropical_prediction_pilot/`

That pilot has three events, 24-hour history and rolling 48-hour forecasts.
Its unweighted point-horizon results are an exploratory archive, not extra
folds or a directly comparable baseline for `tropical_evidence/`.
The archived data source commit's `src/asymode/panel.py` uses a centred
plus/minus-seven-day service rule. That can depend on future outage reporting.
The pilot did not rebuild those panels with this branch's pre-window gates.
Therefore its metrics must not be promoted to prospective or confirmation
evidence. The three old storm windows have no county/time intersection with
the C sealing windows, checked from metadata only in `legacy_exposure_audit.json`.

## New development population and evaluation

- Include **all 15 tropical systems and 1,633 county-events** of development
  tranche D, including weak and near-zero-outage systems. No outcome filtering.
- Restrict training, validation and evaluation to tropical systems: this differs
  from the existing all-regime-trained HOST's tropical subgroup evaluation.
- Inherit every existing event outer fold and source-row identity from
  `splits_v1D.json`. Keep 72 historical hours, then the original 144-hour target
  and observation mask. No county, hour or target is relabelled.
- Hyperparameter tuning, if used, gets four inner event folds inside each
  outer development set. Do not rely on one quiet storm as the only validation
  event. Refit selected settings on all four outer development folds.
- Primary score: design-weighted pooled 144-hour MSE and RMSE. Report MAE,
  untrimmed-weight sensitivity, each storm and S1/S2/S3 strata as well. Report
  +1/+6/+24/+48 point-horizon metrics separately, with valid-cell counts.
- All transforms, imputers, rate initializations, calibration and ensemble
  weights must fit within their corresponding training/validation data.
- The sealed C outcomes remain unread. This is development after observing
  tropical subgroup performance; it is not a registered confirmatory result.

## Models to continue with

1. Refit the existing W+Cin HOST on tropical D only, seeds 0/1/2 and five event
   folds. Keep the parent's 900-step budget as the first matched reference.
2. Re-evaluate direct/residual trees on this same population and 144-hour
   information boundary. The older pilot's numerical gains do not transfer.
3. Train the existing geography/kernel variants and joint rate adapters from
   scratch, using actual geography and ERA5/HRRR hazard dictionaries. Use
   matched seeds and HOST inputs when isolating geography's contribution.
4. Select calibration and blends using inner event predictions. Judge both
   RMSE and MAE, including weak storms where false positive outages matter.

Neither archive weights nor old fitted normalizers are reused. Future ERA5 and
rolling HRRR source fields are conditional hindcast inputs; the presence of
HRRR does not by itself establish operational 144-hour forecast accuracy.

## Prepared data

Run from repository root:

```bash
python experiments/geo_weather_20260924/tropical_prediction_v1/prepare.py
```

This joins only committed D outcomes, 26 ERA5 features, 34 HRRR features,
physical geography and historical HOST context. SAIDI 2023 / SAIFI 2023 are
excluded. Missing geography stays missing for training-fold-only imputation.
Weather availability is carried separately from outcome observation masks:
Milton (S00037) has 59 missing HRRR hours. Preserve the upstream missing-value
convention and use the explicit indicator; do not drop target cells to make
HRRR comparisons look better. Any alternative ERA5 fallback must be declared
as a new input recipe. These feature files are derived summaries, not raw HRRR
or the exact original HOST feature tensors.
It writes an ignored local data artifact, plus versioned `splits.json` and
`preparation_audit.json`. These are data preparation outputs, not new model
performance results. The existing HOST result in `tropical_evidence/` is still
single-seed evidence; its 11.5% improvement is in MSE against zero.
