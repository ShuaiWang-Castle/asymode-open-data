# D08 input-reader storage amendment

Original selector/scope registration: `6a64a12cc3515c1c8dc34b905e14dc12c1dbd2e7`.
The first execution stopped before selection: the compressed `geo.npy` member is
Fortran-ordered `[8457,40]` float32, whereas the initial row reader accepts C order.
The `xu.npy` member is C-ordered `[8457,216,42]` float32. Only NPY headers were used
to diagnose the mismatch. No roster, input cutoff, outcome, prediction or score was produced.

Preserve the first empty runtime directory `runs/geo_weather_20260924/d08_panel_20260929`
and failure log `experiments/geo_weather_20260924/logs/d08_input_panel_20260929.log`.
The initial process exited with status1; there is no live queue to duplicate.

Repair only static two-dimensional Fortran geography input: stream coordinate columns,
copy the specified FIT rows, and retain all40 dimensions. No OUTER values enter scaling,
imputation, distance or selection. Unexpected weather ordering remains an error.
Add a synthetic Fortran static-matrix test with selected-row identity and layout guards.
All input-only selection settings, distance definitions, group quotas, seeds, probabilities,
original data, labels, model artifacts and the original D08 scope remain unchanged.

Commit the repaired selector/test and this amendment before re-execution. The new exclusive
runtime directory is `runs/geo_weather_20260924/d08_panel_20260929_iofix1`; pass it using
`--out`. Record the new source/registration hashes. The label audit passes this directory
explicitly using `--runtime`, after the new input FROZEN marker has been independently checked.
No training, new forward, extra sampling seed, cutoff relaxation or outcome-driven reselection.

HT reporting clarification from independent review: only the additive weighted sums have
fixed-predictor design-unbiasedness. Weighted quantiles, RMSE square roots, sample positive-unit
counts and sample top10% positive-gain concentration do not estimate those population features
without additional uncertainty/definitions; report them as reweighted-panel descriptions.
Accepted pairs may reuse endpoints and are dependent comparisons, not independent causal replicates.
