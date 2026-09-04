# Output interpretation

## `EVENT_CONSERVATION_METRICS.csv`

One row per event and fixed time design. This is the place to inspect empirical closure, rate-boundary status, residual scale, balanced-flow support, and county reporting resolution.

## `FOLD_CONSERVATION_METRICS.csv`

One row per frozen source fold, time design, and weighting scheme. It directly checks the pooled constant-fit identity underlying the pilot diagnosis. Row-pooled rows include descriptive Gamma; equal-event rows report the changed conservation moment and rates but leave Gamma undefined pending a weighted sandwich derivation.

## `LOCAL_GAMMA_METRICS.csv`

One row per deterministic `k=200` source-event neighborhood. `Gamma_plugin` is a descriptive local constant-fit plug-in. It is not an unbiased estimate of a neural crossover and is never used without the closure and boundary columns beside it.

## `PREFLIGHT_SUMMARY.json`

Machine-readable provenance and compact summaries. The public-data commit, manifest digest, fold digest, parameters, and checksum audit must be present.

## `PREFLIGHT_REPORT.md`

Automatically generated tables. It contains no manually selected event or family.

## `RUN_PROVENANCE.json`

Exact run configuration and verified public-file hashes.

## `EXECUTION_REPORT.md`

Independent human-readable audit written after checking the generated tables. It must distinguish exact algebra, measured empirical assumptions, descriptive design findings, and claims that still require an estimator experiment.

The report must end with exactly one execution status:

```text
PREFLIGHT_COMPLETE_AWAITING_SHUAI_DECISION
```

or, if any required integrity/data/algebra/output gate failed:

```text
BLOCKED_<short_reason>
```

It must not assign a scientific promotion/rejection label automatically. Whether the measured design is sufficiently informative for another estimator experiment is a decision reserved for Shuai after reviewing the full tables.
