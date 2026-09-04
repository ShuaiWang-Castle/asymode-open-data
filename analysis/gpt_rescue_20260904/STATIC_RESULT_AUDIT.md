# Static audit of the committed event-transfer result

This report is computed without retraining from `results/cc_event_repro_core_event.json`.

## Comparator initialization

The actual signed initialization reconstructed as `u_init-r_init` is negative in **33/33** one-flow fits.

- range: `-0.106630` to `-0.089985`;
- median: `-0.092057`;
- median magnitude relative to the susceptible interruption initialization: `98.7x`.

The two operands are rates on different exposure pools, so this subtraction is not a valid flow-matching initialization.

## Optimization and boundary summaries

| arm | parameters | median best epoch | best epoch <=2 | mean exact-zero prediction share | mean seed RMSE SD h24 |
|---|---:|---:|---:|---:|---:|
| net_scaled | 3121 | 6.0 | 8/33 | 41.10% | 0.001797 |
| two_rate | 3138 | 3.0 | 14/33 | 0.00% | 0.001407 |

## Event-level comparison

| test event | h24 gain (%) | h48 gain (%) | one-flow exact-zero share | signed initialization |
|---|---:|---:|---:|---:|
| 2021-05-04 | +1.95 | +2.46 | 40.8% | -0.0900 |
| 2021-06-21 | +3.80 | -1.14 | 38.4% | -0.0956 |
| 2021-08-11 | -2.69 | -0.01 | 27.1% | -0.1003 |
| 2021-12-11 | +5.02 | +2.44 | 53.3% | -0.0937 |
| 2022-04-13 | +16.11 | +15.99 | 43.9% | -0.0902 |
| 2022-06-08 | -0.17 | -8.82 | 46.5% | -0.0911 |
| 2022-06-17 | +2.00 | +3.38 | 38.3% | -0.0921 |
| 2022-07-23 | -2.09 | -1.08 | 41.1% | -0.0912 |
| 2024-05-08 | +22.21 | +24.42 | 40.7% | -0.1066 |
| 2024-05-26 | +2.95 | +3.65 | 38.0% | -0.1066 |
| 2024-06-26 | +0.94 | -0.47 | 44.1% | -0.0904 |

Equal-event descriptive summary:

- h+24: mean `+4.55%`, median `+2.00%`, positive in `8/11` events;
- h+48: mean `+3.71%`, median `+2.44%`, positive in `6/11` events;
- correlation between the one-flow exact-zero share and gain: `+0.223` at h+24 and `-0.005` at h+48.

The correlation is descriptive with eleven events and is not a causal or inferential result. The load-bearing finding is the deterministic reconstruction of the initialization and the large boundary-degenerate prediction share.
