## Main table (loeo)

Pooled over held-out observed county-hours of all folds; W and GCRK = mean ± sd over seeds 0-4.
Source: `evaluate.py loeo` -> `results/table_main_loeo.csv`.

| model | RMSE all | 1-6 h | 7-24 h | 25-48 h | 49-144 h | MAE |
|---|---:|---:|---:|---:|---:|---:|
| all-zero | 0.03049 | 0.01502 | 0.02874 | 0.02888 | 0.03190 | 0.00452 |
| persistence | 0.03207 | 0.01549 | 0.03046 | 0.03069 | 0.03344 | 0.00500 |
| TimesFM | 0.03089 | 0.01567 | 0.02920 | 0.02884 | 0.03237 | 0.00471 |
| TimesFM (history only) | 0.03094 | 0.01552 | 0.03021 | 0.02907 | 0.03223 | 0.00468 |
| W | 0.03144 ± 0.00102 | 0.01416 ± 0.00058 | 0.03117 ± 0.00085 | 0.03292 ± 0.00294 | 0.03187 ± 0.00067 | 0.00692 ± 0.00042 |
| GCRK | 0.03063 ± 0.00140 | 0.01412 ± 0.00044 | 0.02910 ± 0.00144 | 0.02998 ± 0.00132 | 0.03180 ± 0.00160 | 0.00700 ± 0.00049 |

| model | |peak magnitude error| | |peak time error| (h) | false activity share | under-half share |
|---|---:|---:|---:|---:|
| all-zero | 0.0987 | n/a (constant forecast) | 0.000 | 1.000 |
| persistence | 0.0977 | n/a (constant forecast) | 0.038 | 0.982 |
| TimesFM | 0.0974 | 47.8 | 0.019 | 0.986 |
| TimesFM (history only) | 0.0979 | 57.2 | 0.012 | 0.993 |
| W | 0.0832 ± 0.0019 | 22.1 ± 2.1 | 0.277 | 0.768 |
| GCRK | 0.0804 ± 0.0021 | 21.5 ± 1.3 | 0.362 | 0.748 |

## Paired seed-wise differences, GCRK - W (loeo)

Source: `results/paired_loeo.csv` (per-seed values in `results/seeds_loeo.csv`).

| metric | W | GCRK | mean diff | rel. change | seeds GCRK lower | per-seed diff | verdict (PREREG 9) |
|---|---:|---:|---:|---:|---:|---|---|
| rmse | 0.031438 | 0.030631 | -8.070e-04 | -2.54% | 4/5 | -7.361e-04;-8.643e-04;-2.734e-03;-3.593e-04;+6.586e-04 | GCRK better (4/5) |
| rmse 1-6 h | 0.014157 | 0.014116 | -4.153e-05 | -0.24% | 4/5 | -1.297e-04;-9.627e-06;-4.776e-04;+4.777e-04;-6.844e-05 | GCRK better (4/5) |
| rmse 7-24 h | 0.031168 | 0.029099 | -2.069e-03 | -6.54% | 4/5 | -3.580e-03;-2.030e-03;-4.235e-03;-1.171e-03;+6.697e-04 | GCRK better (4/5) |
| rmse 25-48 h | 0.032918 | 0.02998 | -2.938e-03 | -8.36% | 4/5 | -2.105e-03;-6.702e-04;-7.455e-03;+3.187e-04;-4.777e-03 | GCRK better (4/5) |
| rmse 49-144 h | 0.031875 | 0.031798 | -7.693e-05 | -0.26% | 3/5 | +7.508e-05;-7.304e-04;-1.372e-03;-4.111e-04;+2.054e-03 | no consistent difference |
| mae | 0.0069178 | 0.0069994 | +8.157e-05 | +1.22% | 2/5 | -2.498e-04;+5.785e-05;-2.367e-04;+3.614e-04;+4.751e-04 | no consistent difference |
| peak_mag_abs_mean | 0.083161 | 0.08043 | -2.731e-03 | -3.23% | 4/5 | -1.275e-03;-2.672e-03;-7.732e-03;-3.296e-03;+1.323e-03 | GCRK better (4/5) |
| peak_time_abs_mean | 22.126 | 21.499 | -6.273e-01 | -2.43% | 3/5 | +9.994e-01;-6.520e-01;-3.058e+00;+5.469e-02;-4.804e-01 | no consistent difference |
| false_activity_share | 0.27742 | 0.36194 | +8.452e-02 | +31.72% | 0/5 | +7.287e-02;+5.395e-02;+1.640e-01;+1.228e-01;+8.937e-03 | GCRK worse (5/5) |
| under_half_share | 0.7684 | 0.74842 | -1.999e-02 | -2.57% | 4/5 | +1.853e-02;-4.072e-02;-4.670e-02;-4.108e-03;-2.693e-02 | GCRK better (4/5) |

## Kernel exit decomposition of the full-rollout RMSE (loeo)

closed = the trained GCRK network with its response contribution removed. Source: `results/decomposition_loeo.csv`.

| seed | W | GCRK exit closed | GCRK | joint training (closed - W) | kernel output (GCRK - closed) | total |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.030963 | 0.029775 | 0.030227 | -1.19e-03 | +4.52e-04 | -7.36e-04 |
| 1 | 0.030684 | 0.030008 | 0.029819 | -6.76e-04 | -1.89e-04 | -8.64e-04 |
| 2 | 0.032621 | 0.030053 | 0.029887 | -2.57e-03 | -1.66e-04 | -2.73e-03 |
| 3 | 0.030468 | 0.029290 | 0.030109 | -1.18e-03 | +8.19e-04 | -3.59e-04 |
| 4 | 0.032453 | 0.031039 | 0.033111 | -1.41e-03 | +2.07e-03 | +6.59e-04 |
| mean | 0.031438 | 0.030033 | 0.030631 | -1.40e-03 | +5.98e-04 | -8.07e-04 |

## By event: full-rollout RMSE (loeo)

Source: `results/events_loeo.csv`.

| event | units | all-zero | persistence | TimesFM | W | GCRK | GCRK - W | seeds GCRK lower |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2019-03-13 | 321 | 0.02534 | 0.02528 | 0.02529 | 0.03624 ± 0.00508 | 0.02999 ± 0.00266 | -6.25e-03 | 4/5 |
| 2021-03-26 | 544 | 0.01059 | 0.01094 | 0.01068 | 0.01297 ± 0.00100 | 0.01142 ± 0.00093 | -1.55e-03 | 5/5 |
| 2021-12-11 | 991 | 0.04408 | 0.04692 | 0.04474 | 0.04233 ± 0.00032 | 0.04192 ± 0.00047 | -4.10e-04 | 4/5 |
| 2022-06-08 | 385 | 0.01756 | 0.01755 | 0.01767 | 0.02091 ± 0.00349 | 0.02164 ± 0.00412 | +7.28e-04 | 1/5 |
| 2024-02-27 | 419 | 0.01971 | 0.01989 | 0.01998 | 0.01977 ± 0.00087 | 0.02190 ± 0.00562 | +2.14e-03 | 4/5 |
