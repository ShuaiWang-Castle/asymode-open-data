# Tropical prediction campaign v1 (fixed before fitting)

This campaign follows the PI's request to continue toward a prediction paper.
Only the 15 tropical development systems are used. No confirmation outcomes
are read. Prior tropical subgroup results already informed this direction, so
all present results are development evidence, including the nested evaluation.

## Shared information and target

Use `prepare.py`'s development artifact with its inherited five event folds,
72 historical hours and fixed 144-hour forecast. Keep every weak storm and all
original target masks and design weights. Output MAE, RMSE and MSE on the full
path, point horizons 1/6/24/48, individual systems and exposure strata. Primary
comparison pools weighted errors over cells before computing RMSE; report
untrimmed and unweighted sensitivities separately.

The original `features_v1D.npz` HOST tensor is not part of the committed data
products. We therefore reuse the actual `asym_host.AsymODE` / `gcrk.GCRKLayer`
implementations but reconstruct matched model inputs from the available hazard
dictionary. This HOST is a new matched reference, not a numerical reproduction
of the earlier 11.5% MSE result. Inputs for both neural arms: non-geography-
multiplied ERA5 and HRRR hazard features, weather availability flags, clock,
causal weather summaries, the six historical county context fields, and
summaries of the observed pre-origin outage history. Weather missing values
keep the upstream convention plus an explicit indicator. Imputation and
standardization fit on training rows only. No future outage is a covariate.

## Fixed candidates

- Zero, persistence, and a single training-fitted exponential-decay persistence.
- Matched hourly HOST, existing widths 32 (damage) / 16 (recovery).
- Joint fusion: same HOST initialization, actual GCRK with physical geography,
  and zero-output-initialized 16-unit damage/recovery rate residual heads.
  All parameters train jointly; no frozen or imported weights.
- ERA5 residual LightGBM (squared loss), dual-source residual LightGBM (squared
  loss), and dual-source direct LightGBM (absolute loss). Tree inputs include
  physical geography and causal/known-weather summaries; all time leads share
  one predictor. Fixed 300 trees, 15 leaves, minimum 80 samples/leaf, learning
  rate 0.035, L2 regularization 20; no test-driven configuration selection.

Neural fitting: 900 Adam updates, batch 128, damage/kernel learning rate 0.003,
recovery 0.0003, gradient norm cap 1. GCRK uses the repository's bounded opening,
200-step warmup and training-only drop path. Calibration buffers refresh from
a fixed fitting-only sample (up to 256 rows) every 10 steps. Fit all normalization
and geography imputation anew per fitting set. Three final seeds 0/1/2 are
averaged at the prediction level. Batch sampling means this is not the old
full-batch 900-step experiment; the budget is matched between new neural arms.

For each outer fold, generate four inner event-fold predictions for every
candidate. Neural inner fits use seed 0, same 900-step budget. Choose a single
nonnegative, sum-to-one vector of blend weights using the pooled design-weighted
inner MSE. That vector is fixed across all 144 forecast hours. Refit on all outer
development rows; neural components use the three-seed prediction mean. Report
every candidate and the validation-selected blend. Differences between one-seed
inner forecasts and three-seed final forecasts are a declared approximation.

No candidate, seed, county or event will be dropped based on its outer score.
The full training manifest and all inner blend choices will be saved before
the campaign summary is inspected. Cache identities include fitting rows,
prediction rows, configuration and source hash. Improvements of the joint arm
would concern the combined predictor, not isolated causal geography effects.

ERA5 and rolling HRRR inputs remain conditional hindcast information. A real
operational-weather test and an independently registered confirmation are
separate requirements before claiming deployable forecast skill.
