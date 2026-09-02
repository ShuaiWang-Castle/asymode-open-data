# Archive reproduction audit (A2) — no retraining

All commands run from the repository root with `./.venv/bin/python`; exit codes
captured directly. Source files carry `panel_digest 76a73ed794af` (g2) /
`db286b4960a4` (g3) / `1c2bc7bfdfa6` (g1) and `channel_digest dec964873cb2`.

| command | exit | reproduced values | vs ledger |
|---|---|---|---|
| `scripts/paired_review.py results/exp05_g2_sixarm.json --ref susceptible --arms net_scaled transmission_seed --horizons 1 6 24 48` | 0 | net_scaled −1.68/−1.30/+0.89/**+2.03% 12/15 t=3.39**; transmission_seed +1.39/+1.78/+0.02/+0.28 | identical |
| `scripts/paired_review.py results/exp07_g2_oof.json --against results/exp08_architecture.json --ref control --arms trees_matched trees_lookback linear_matched --horizons 1 6 24 48` | 0 | trees_matched **+25.93% 15/15 / −2.41 2/15 / −7.32 0/15 / −5.98 0/15**; trees_lookback +19.34/−4.39/−7.14/−6.92; linear_matched +109.71/+17.44/+5.03/+4.19 | identical |
| `scripts/paired_review.py results/exp10_per_horizon.json --against results/exp08_architecture.json --ref control --horizons 1 6 24 48` | 0 | per_horizon −1.11 (2/15 worse) / **−2.42 (2/15, t=−4.73)** / −0.15 (7/15) / +0.49 (8/15) | identical |
| `scripts/paired_review.py results/exp06_by_family.json --family {tropical,convective,wind,winter} --ref net_scaled --arms susceptible transmission damped_persistence --horizons 24 48` | 0 ×4 | susceptible: tropical **−3.94/−5.15**, convective −0.88/−1.99, wind +0.03/−0.08, winter **+2.53/+2.82** | identical |
| `scripts/paired_review.py results/exp06_convergence_probe.json --family {tropical,winter} --ref net_scaled --arms susceptible --horizons 24 48` | 0 ×2 | tropical −1.71/−3.45 (1/5 worse); winter +4.18/+4.34 (5/5) | identical |
| `scripts/paired_review.py results/exp08_ha3_g1panels_14ch.json --ref control --arms in_ambient_u_only in_ambient_r_only in_ambient_none --horizons 24 48` | 0 | u_only +0.73 11/15 / +1.24 13/15; r_only +0.68/+1.04; none +1.06/+1.14 | identical |
| `scripts/check_comparable.py results/exp05_g2_sixarm.json results/exp07_g2_oof.json results/exp08_architecture.json results/exp10_per_horizon.json` | 0 | same panels, channels, horizon, stride, k, seeds | — |
| `scripts/make_figures.py` | 0 | fig01–fig05 regenerated (one benign all-NaN warning in fig03) | — |

**Discrepancies / hidden arguments found and fixed on this branch.**
1. EXP07 and EXP10 tables need `--against results/exp08_architecture.json`
   because the two-rate `control` is not in those files; the ledger did not say
   so. Recorded here.
2. EXP06 tables were produced by the experiment's own summary, not by
   `paired_review`, which failed on per-family row collisions. `--family` added
   (`scripts/paired_review.py`); tables now reproduce from the archive.
3. Legacy `paired_review` prints t-statistics over (seed, fold) cells (R3).
   They are reproduced here for audit only and are **not** confirmatory.

**Can the scripts run from a clean checkout?** Yes for every command above:
they read tracked JSON only. The OOF-based diagnostics (D-1/D-3/D-4/D-5) read
`results/oof_*.npz`, which are gitignored exports and must be regenerated with
`--save-oof` (documented in the ledger); they were not re-run here.
