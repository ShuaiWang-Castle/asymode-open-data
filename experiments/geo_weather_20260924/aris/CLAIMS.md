# Claims ledger (ARIS style: every claim has a status, evidence and a reviewer receipt)

Status values: **verified** (pre-registered test passed, or a no-training audit with an independent reviewer's receipt),
**refuted** (a registered or replicated test contradicts it), **unproven** (evidence inconclusive or single-seed),
**retired** (no longer pursued). Evidence points to files under `results/` or to `PREREG_W2.md`.

| id | claim | status | evidence | reviewer receipt |
|---|---|---|---|---|
| C01 | On the twelve-event wind panel, sub-county geography through ERA5 (elevation bands, co-location, static band distribution) changes the hazard features in < 1% of outage-weighted county-hours | verified | results/F0/f0_audit.json | formal contributor, F1 (results/F1/F1_cd.md) |
| C02 | On the wind panel, no sub-county or exposure-weighting contrast aligns with the host's residuals; effects worth 2% of pooled RMSE would have been detected | verified | results/F1/F1_cd.md | formal contributor |
| C03 | Dropping EAGLE-I artefact hours from the training loss improves held-out RMSE | refuted | results/screen_clean_5fold.json (+0.01% over five folds; the two-fold -2.66% was a false positive) | - |
| C04 | Population-weighted host inputs improve RMSE | refuted (screen) | results/screen_S1.json (-0.25%, +-3%) | - |
| C05 | Canopy as a county context, or gust x canopy as gated inputs, improves RMSE | refuted (screen, folds 1-2) | results/screen_S4_canopy.json | - |
| C06 | HRRR changes the hazard features far more than ERA5 downscaling, and almost all of the difference is the weather source, not resolution | verified (audit) | results/F0_w1_hrrr_split, results/F0_hrrr_split | formal contributor (C6-C8, results/F1_hrrr, results/F1_w1_hrrr) |
| C07 | HRRR-source near-freezing precipitation on the ERA5 grid, minus ERA5's, aligns with where the ERA5-driven host under-predicts, on independent winter storms | **verified** (pre-registered H2a, W2e, 23 storms) | results/F1_w2e_hrrr/H2a.md; PREREG_W2 amendments 4-5 | formal contributor (ran the frozen test) |
| C08 | The HRRR-source pathway lowers RMSE on unseen ice storms | unproven (seed 0 -18.7%, seed 1 +13.4%, seed-averaged -4.2% [-11.1, +2.6]) | results/w1e_hrrr_seeds.json | - |
| C09 | The HRRR-source pathway beats its ERA5 twin on the wind panel | unproven (one seed: -1.95%, intervals below zero) | results/screen_S5_hrrr_vs_twin.json | - |
| C10 | ERA5 elevation-band near-freezing precipitation aligns with host residuals (H1) | unproven (W2d not testable; W2e uninformative, power 0.795) | results/F1_w2d, results/F1_w2e | formal contributor |
| C11 | The HRRR-source pathway lowers RMSE on 23 unseen winter storms (H2b) | pending (seed 0 done, unread; seeds 1-2 paused by the PI) | PREREG_W2 amendments 6-7 | - |
