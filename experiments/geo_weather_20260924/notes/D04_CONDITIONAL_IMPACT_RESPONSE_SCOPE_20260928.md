# D04: conditional weather–geography response estimation

Written before response fitting, following the PI's authorization to execute the high-dimensional analysis.
This is exploratory statistical work on public D. D01–D03 and their negative/weak evidence remain intact.
It does not train a new neural outage arm, read C, change the paper, or replace the original targets/folds.

## Research chain and estimand

The durable core is **weather -> complex geographic modulation -> impact formation, interaction and
accumulation -> outage**. This analysis looks for observable conditional-response symptoms of that chain.
The proposed impact state remains latent; hourly outage stock does not separately identify physical damage
and recovery. A successful conditional regression is not proof of an impact mediator or geographic causation.

Use all 8,457 public-D county-events, including `used=False` and zero-outage units. Let p(t) be the existing
hourly average of available 15-minute customer-outage fractions. A valid hour may contain only one available
quarter-hour value; its four-point coverage is not in the panel. Target y(t)=p(t)-p(t-1) requires both hourly
observations and finite values. Negative changes and exact zeros remain. Missing `y_full` values are NaN,
not zeros. Do not bridge observation gaps. Differences can amplify reporting noise.

Fit at fixed times 72,78,...,210: this samples one-hour changes every six hours, not six-hour changes.
This gives 202,311 valid rows from 202,968 proposals, covering all units. Evaluate held-out events on **all**
valid times 72..215 (1,213,761 rows) and separately all six clock phases, so training-phase aliasing is visible.
These support counts were inspected before response fitting. For each support set divide each county-event's
design weight by its actual number of retained rows. Original raw weights are retained for sensitivity.

Weather is strictly earlier: lag ell uses x(t-1-ell), ell=0..47. No target-hour or later weather is used.
ERA5 is nevertheless realized/reanalysis weather, not an operational forecast assessment. Context is fixed
county/service information; initial stock p(71) and rolling p(t-1) are observation controls in the primary fit.
This is a conditional one-hour diagnostic with observed previous stock, not an autonomous 144-hour forecast.
Do not cumulatively sum its predictions and label the result a free-running outage trajectory.

## Fixed representation, broad scope

Retain the original 12 weather channels. Apply the existing log1p nonnegative transforms to CAPE,
precipitation and snowfall. Weather scales and weighted q50/q90 knots use only training rows' 48-hour
histories. Three fixed lag bands are past hours 1–6, 7–24 and 25–48.

The 774 weather coordinates are:

* 144 nonlinear main coordinates: each channel/band averages z, z², positive(z−q50), positive(z−q90).
* 198 synchronous coordinates: mean of the same-hour product for all 66 different-channel pairs, in all
  three bands. This is not the product of two mean exposures; same-channel squares are already in main.
* 234 symmetric cross-band coordinates: all 78 pairs including 12 self pairs, for each of the three band
  pairs; average of the two assignments. Self pairs represent repeated same-weather exposure.
* 198 antisymmetric cross-band coordinates: all 66 different-channel pairs in the three band pairs;
  positive means channel a in the earlier band paired with channel b in the later band, minus the reverse,
  divided by two. Same-channel antisymmetric terms are identically zero and are not duplicated.

This is a finite, coarse lag approximation. It does not preserve every within-band sequence, every
48×48 lag pair, third-order combination, or history longer than 48 hours. Failure cannot exclude those
structures. No pair, county type or regime is selected based on D02's promising cells.

Retain geo40 continuously. Training-unique-county median imputation and scaling precede the four semantic
block scaling used by D02. Geographic basis: 40 linear, 40 positive-part, and 64 fixed-seed tanh projections
of all block-scaled descriptors. All original directions remain; this is not PCA-based feature selection.
Final basis columns are standardized on training rows, so equal semantic weighting of the input projection
must not be confused with equal total penalty on every final semantic block. Seed 20260929 defines a basis,
not a neural training seed. Context6 separately gets linear and positive-part coordinates (12 columns).

All arms contain the **entire same additive geography basis**, context basis, regime-specific linear/quadratic
event time, fixed clock harmonics, origin season/year, and partial-pooled county intercepts. Primary historical
controls are p(t-1), its square/square root, and fixed p(71). Consequently an added interaction does not merely
introduce the 64 nonlinear additive geography coordinates absent from the baseline.

All models also have a rank-2 context×weather term using the weather coordinates present in that model;
physical geography is not allowed to be the only modifier competing with service/context differences.
These are conditional statistical nuisance controls, not claims about service mechanisms.

## Jointly fitted models

For weather design X, nuisance N, geography G and separate context C, fit

    yhat = N betaN + X betaX
           + sum_r (X U_geo)_r (G V_geo)_r
           + sum_s (X U_context)_s (C V_context)_s + a_county.

All components are fitted jointly. This is low rank in the geography×weather coefficient matrix, not
ordinary reduced-rank regression on a small number of outputs. Separate main/pair contrasts do not identify
unique additive mechanisms under correlated inputs. Include pure interactions even if marginal effects vanish.

Four fixed structures:

| arm | weather | geography interaction |
|---|---|---|
| A_shared_main | 144 main coordinates | none |
| B_shared_pairs | all 774 coordinates | none |
| C_geo_main | 144 main coordinates | rank 2 or 4 |
| D_geo_pairs | all 774 coordinates | rank 2 or 4 |

Comparisons C/A and D/B assess conditional geographic modulation at two weather representations. B/A
and D/C describe adding composite histories. Their magnitudes are not causal component contributions.
The main motivation remains local response shape and support; an overall MSE loss is no exclusion gate.

Use weighted squared loss, normalizing each of the five regimes equally within training. Preserve original
within-regime design proportions. Scale y by its training weighted standard deviation; do not use held-out
outcomes to set any scale, knot, feature transform, intercept, rank, or regularization.

Use symmetric L2 penalties on the two interaction factors and ordinary coefficients; only the intercept
is unpenalized. Profile county effects exactly in each objective: a_c=sum(w residual)/(sum(w)+0.1/C_train),
with normalized weights and C_train unique training counties. Unseen counties receive zero county effect;
their geography and context remain available. This is shrinkage, not unpenalized county fixed effects.
No held-out system/phase nuisance effect is fitted from held-out y. Event confounding remains a limitation;
within-event/time contrasts will be reported separately as descriptive validation, not new OOF forecasts.

## Selection and numerical budget

Keep the original five event folds and verify no merged group crosses them. For outer fold k, inner
validation is original fold k%5+1 and inner training is the remaining three folds. Every transformation
is re-fitted on that inner training subset. Inner configurations: ridge 0.01/0.1, geography rank 2/4 when
present, one fixed nonzero start (11), at most 60 L-BFGS iterations. Select by five-regime balanced inner
net-change MSE; exact ties prefer stronger shrinkage then lower rank. Rank-zero arms also select ridge.

Re-fit each selected arm on the four outer training folds using two nonzero starts (11,29), at most 120
L-BFGS iterations each; select the lower **training objective**, never outer performance. Record both starts,
objectives, gradients, finite status, iterations and stopping reason. An iteration budget is not convergence
or a global optimum. The same fits/outputs are retained if a comparison is unfavorable. If optimization is
materially unstable, report that limitation and stop inference before redesigning; do not silently relabel
the run as a failed geography hypothesis. This is a finite maximum of 100 statistical fits, not an open sweep.

One process, nice >=15, two numerical threads. Process chunks and do not create the full row-wise G×X
matrix. Preserve fold directories including partial/empty ones; use a lock and source hashes to prevent
duplicate or mixed-code queues. Completed fold checkpoints and finite OOF exports are immutable.

## Evaluation and response diagnostics

Report un-clipped one-hour change error against zero change (persistence), all four arms and all five regimes;
keep tropical/winter balance, all-five balance and pooled measures distinct. A clipped next-stock diagnostic
is secondary, because it uses the observed previous stock. Scores here are not the original neural screen
or its acceptance gates. Keep original evaluation files unchanged.

Use merged-event resampling for intervals on fixed OOF score differences, with repeated-county sensitivity
reported separately. These score intervals do not include fitting/selection uncertainty. Preserve all regimes,
zero-increment rows, large positive/negative changes, six clock phases, peak/boundary/missing support and
unseen-county coverage. Peak diagnostics refer to observed **net increments**, not physical impact peaks.

Use the selected folds to inspect reconstructed coefficient operators and local weather/lag sensitivities,
geographic variation, and stability across held-out events. Latent factors can rotate; interpret reconstructed
operators and observed-path contrasts. A single-coordinate sensitivity can violate u/v/speed/gust dependencies;
label it model sensitivity and use natural joint-path neighbors for observational comparisons.

Support diagnostics must use joint weather/history/context neighborhoods and event concentration; marginal
range overlap is insufficient. Approximate-neighbor support in a finite representation is not proof of full
weather-path exchangeability. Unsupported regions remain explicit. Show local patterns and uncertainty even
if the aggregate score is weak. Do not invent fitted-surface confidence intervals from score bootstraps.

After the primary five-fold results, a separately marked full-D fit without the rolling/prefix stock controls
will examine whether adjustment absorbs historical pathways. Its configuration will be the modal selected
configuration per arm (ties stronger shrinkage/lower rank), two training-objective starts, no new OOF claim.
All-D adjusted fits at the same configurations provide an apples-to-apples descriptive counterpart. This is
eight additional full-D fits per history specification (16 total), not another selection sweep.

Outputs: finite run status/checkpoints and held-out predictions in the ignored D04 directory; compact support,
selection, numerical, score and conditional-response summaries in results/v1; a Chinese result note and
updated research ledgers. Scan public additions and explicitly commit/push only this experiment to the two
research branches. No automatic neural training or reopening I18 follows any statistical result.
