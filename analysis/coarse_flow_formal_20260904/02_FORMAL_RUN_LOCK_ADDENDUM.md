# Formal-run lock addendum

This note clarifies implementation details that are implicit in the previously committed formal lock. It does not change `K`, features, events, outcomes, or model classes.

## Final refit

After `K=8` was selected using 2018--2020 fitting and 2021 validation, the one-shot confirmation estimator is refit on **all 2018--2021 events**. All standardization, K-means centers, one-flow branches, two-flow rates, damping, and baseline fitting use only those source years. No 2022/2024 row enters fitting or selection.

## Main data design

- hourly state obtained by the mean of observed 15-minute substeps;
- transition retained iff both current and next hourly states are observed and finite;
- active-48 is the already locked outcome-blind NOAA-footprint window;
- the model is fitted on source-event active-48 transitions;
- primary path forecast uses origins from peak minus 24 hours through peak, inclusive, and 24 open-loop steps;
- an unavailable active-48 event remains unavailable for the active-window endpoint and is not shifted.

## Direct HGB baseline

The direct h+24 baseline uses exactly 50 inputs:

1. current outage state (1);
2. current 24-feature exogenous vector (24);
3. mean of each of the 12 raw weather channels over the next 24 hours (12);
4. maximum of each raw weather channel over the next 24 hours (12);
5. origin phase relative to the NOAA peak (1).

It uses `HistGradientBoostingRegressor(max_iter=250, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=100, l2_regularization=1e-4, early_stopping=True, random_state=0)`.

The recursive HGB predicts the next state using the current 24-feature vector and current predicted state; it is trained with `max_iter=200`, the same learning rate, leaf count, minimum leaf size, regularization, and random state, with early stopping disabled.

## Confirmation claims

The structural claim requires the active-48 one-step and path-24 mean differences to be positive and the leave-one-event minimum mean to remain positive. Exact sign-flip and event bootstrap are reported. The paper may describe statistical support only if the exact sign-flip threshold in the original formal lock is met.

Practical forecasting claims are separate. The two-flow estimator must be compared with persistence, source-fitted damped persistence, recursive HGB, and direct h+24 HGB. A structural result is not automatically described as state-of-the-art forecasting.

## Multiplicity

The online cross-county confirmation is deferred. Only the coarse temporal-transfer confirmation may inspect 2022/2024 outcomes in this analysis cycle.
