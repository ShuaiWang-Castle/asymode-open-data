# T03: complete tropical trajectory benchmark for poster development

Fixed before model execution, continuing branch 9495be9. User authorized retraining and multiple accuracy-improvement methods. This finite experiment uses all 15 existing tropical development events, not a selected favorable event. It does not require cross-event generalization. Previously examined D is development evidence, not unseen confirmation.

## Task and information boundary

72 observed-history hours; predict the next 144 hours with no future outage inputs. Global five-fold county grouping (SHA256 county assignment seed 20261003), including every occurrence of a county in the same outer fold. Within each outer-development set, use a fixed county-grouped ~20% validation subset to select checkpoints, tree settings and convex blend weights. No refit on validation; all arms use the same fit/validation/outer rows. Preprocessing and GCRK calibration use fit rows only.

Weather is existing ERA5 plus HRRR hourly f01 fields. Future HRRR cycles were issued after the outage origin: this is a conditional hindcast with supplied weather, NOT a six-day operational forecast. Static geographic products also have snapshot dates and are not asserted to have been available contemporaneously. Exclude explicit canopy/drainage product channels from both weather sources. Missing HRRR hours are represented by NaN plus an availability channel. All original outcome masks and design weights are retained; no window or target changes.

## Fixed candidate budget

- Three training seeds 0, 1, 2; all five outer folds.
- HOST: repository AsymODE with hourly recovery; weather and history summaries as inputs.
- DOSE: HOST plus causal weather averages (6/24/72 hours), maxima since the beginning of the 216-hour window, and EWMA (0.9/0.98) for gust, excess gust, rain and wet-wind channels from both sources.
- GCRK: DOSE plus the original repository geography-conditioned hidden response kernel (geo40). Train all parameters jointly, from paired HOST initialization.
- GEO_MLP: DOSE with geo40 directly added to its first damage layer through a zero-initialized linear projection. This is a direct geographic-input comparator, not a kernel.
- Each neural fit: 600 Adam steps, batch up to 64 county-events, host LR 0.003, recovery LR 0.0003, gradient norm cap 1.0. Check validation every 50 steps plus initialization; retain lowest pooled weighted MSE checkpoint. No changes after outer scores.
- Two LightGBM trajectory candidates, with/without geo40, use the same weather/history/dose information plus forecast lead. Three fixed (leaves,min_child_samples,L2) configurations: (7,120,30), (15,80,20), (31,80,30), LR .035, up to 500 rounds, validation early stopping 50. Predict a residual over a training-selected damped-persistence trajectory; validate and score after clipping to [0,1]. Select configuration by the same validation MSE.
- Baselines: zero, persistence, training-selected exponential damping (rate grid 0..0.2, 101 values).
- Convex blend of HOST/DOSE/GCRK/GEO_MLP/TREE_W/TREE_G, weights selected using validation predictions; neural inputs to blending are three-seed mean predictions. Same weights for all 144 hours. Keep standalone results beside the blend.

## Evaluation and reporting

Primary: full 144-hour design-weighted pooled RMSE and MAE over all observed county-event-hours. Secondary: unweighted metrics, exact fixed-origin leads +1/+6/+24/+48 (forecast array indices 0/5/23/47), per-event metrics, individual seeds, and severe vs non-severe county-event metrics. These exact-lead metrics differ from competition suffix metrics. Report zero/persistence/damped and every trained candidate, including negative results.

Paired 2000-resample county-cluster bootstrap for full-path RMSE improvement versus HOST; repeat counties across events stay together. These intervals describe this development panel conditional on the fitted models/split, not uncertainty from full retraining or dataset selection. GCRK-vs-DOSE and GCRK-vs-GEO_MLP determine whether a kernel-specific improvement exists. A favorable blend alone cannot establish geographic-kernel value.

Validation: exact OOF coverage, county separation including inner partitions, training-only statistics, finite predictions in [0,1], paired neural initialization, source/input checksums, checkpoint/parameter-update audit, and reconstruction of pooled metrics from fold SSE/SAE and weight totals. Model artifacts and predictions remain local; push only code, protocol, aggregate tables, choices, validation summary, and an accurate poster-ready conclusion. All candidates and this fixed budget are reported; do not iterate on outer results within T03.
