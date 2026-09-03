# Unified AISTATS v2 metric lock

This note resolves the final ambiguity in `docs/UNIFIED_AISTATS_V2_PROTOCOL.md`.

## Primary inferential quantity

For held-out event `e`, model `m`, and horizon `h` in `{6,24}`, compute masked event-level mean squared error

\[
R_{m,e,h}
=
\frac{\sum_{i\in e}M_{i,h}(\widehat Y_{m,i,h}-Y_{i,h})^2}
{\sum_{i\in e}M_{i,h}}.
\]

For the structural comparison, define

\[
d_{e,h}=R_{\mathrm{net\_scaled},e,h}-R_{\mathrm{two\_rate},e,h}.
\]

Positive values favor the proposed two-rate model. The primary estimand is the equal-event mean

\[
\bar d_h=\frac1{11}\sum_{e=1}^{11}d_{e,h}.
\]

This MSE difference is the direct empirical analogue of the squared-risk quantities in the projection theorem.

## Inference

For each primary horizon report:

- the eleven paired event differences;
- the equal-event mean and median difference;
- a 50,000-resample event bootstrap percentile interval for `mean(d_e,h)`;
- an exact two-sided sign-flip/randomization p-value over all `2^11` assignments;
- the number of positive event differences;
- all leave-one-event-out means.

A structural advantage is supported at a horizon when the bootstrap interval for the absolute MSE difference lies above zero and the exact two-sided randomization p-value is below 0.05. No additional sign-count veto is imposed. Sign counts remain descriptive evidence of heterogeneity.

## Presentation metric

The main table also reports

\[
\operatorname{RMSE}_{m,h}
=
\frac1{11}\sum_e\sqrt{R_{m,e,h}}
\]

for readability. RMSE percentages are not used to prove the squared-risk theorem or to determine statistical significance.

## Seeds

For neural models, MSE is first averaged across seeds within each event and only then compared across events. Seeds never increase the inferential sample size.
