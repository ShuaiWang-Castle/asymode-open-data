# Decisions and notes recorded during the run

## 2026-09-14, during DEV (41 of 180 DEV runs complete; no final run or test score existed)

Observation on inner-validation data only:

- Fold 0 validation (the affected county-events of fold 1). Persistence path MSE was 2.7284e-03. The untrained
  NET logged 2.737e-03 at update 0, which is near persistence. The untrained ASYM logged 2.6526e-03, 2.78%
  below persistence. ASYM's initialization relaxes toward the fold's training source mean at 0.2% per hour.
  That already captures part of the recovery that dominates error in affected counties.
- Fold 1 validation. Persistence was 7.4682e-04; the same relaxation gave 7.2370e-04, 3.10% below.
- Fold 0 DEV comparison. The best ASYM configuration (r2, lr 3e-4) was 3.4% below the best NET configuration
  (d2, lr 3e-4), about the size of this head start. Relative to their own starting points, ASYM improved 7.7%
  and NET 7.1%.
- Other fold 0 observations:
  - Both models overfit after about 750 to 1000 updates.
  - NET's validation curve is noisier than ASYM's.
  - NET's pre-clip gradient norms were 10 to 24 against about 1 for ASYM.
  - At lr 1e-3 and 3e-3, 7 of 12 NET runs never improved on their starting point.

Options put to the PI:

1. Give NET the identical initial path y + 0.002 (pi - y) and restart DEV.
2. Lower ASYM's eps to 1e-4 and restart.
3. Continue as locked.

PI decision: no modification ("没必要修改"). The locked protocol continues unchanged. The pre-registered analysis
already reports the untrained (update-0) models per event and population, so the starting-point difference is
reported next to every trained comparison.

## 2026-09-14, after the first DEV run and before any selection or final run

- The machine rebooted at 18:47 local time. The working copy lived in the session's temporary directory and was
  cleared. Lost, none of it pushed: the local commit with this directory's protocol, fold lock, code and tests;
  all DEV logs; and the separate local Helene seed-extension commit.
- The task notification reported DEV finished before the reboot, but the selection and final phases had not been
  started (an operator error), so no selection lock and no test score existed.
- The PI had asked why RMSE is not used. Answer given: for training, selection and ranking, MSE and RMSE are
  equivalent (monotone transform). They differ in how events are averaged: event-equal MSE is dominated by
  high-outage events.
- PI decisions:
  1. Rebuild in a persistent working directory and push to the research branch after each stage.
  2. Rerun this round only; the Helene seed extension is not rebuilt.
  3. Add RMSE as a secondary metric, reported beside MSE; decisions stay on MSE.
- Rebuild: code, tests, protocol and configuration were recovered from the session record exactly as written,
  and the one patch made before the first DEV was re-applied. The fold lock was regenerated with the same
  script. Before the DEV rerun, the RMSE secondary metric was added to `code/cv_analyze.py`, and revision notes
  were added to `PROTOCOL_ZH.md` and `CONFIG_LOCK.json`.

## 2026-09-15, after the results write-up

- PI feedback on reporting: do not split results by forecast lead, and do not report counts of events favouring
  a model within each event type. The question to answer is how the data features defined in the paper decide
  which model to use.
- Options put to the PI:
  - Decision unit: input-defined condition cells, events, or both.
  - Estimation of the paper's signal S and noise nu^2: the solvable benchmark with the Appendix F cross-fitted
    estimator, the readable features only, or both.
- PI decisions:
  - Unit: event level.
  - Estimation: report both. The solvable-benchmark estimates make the decision; the conditional path dispersion
    sigma_x and the replicate count n explain it.
- Status: this analysis is written after the CV results were seen. It is labelled post-hoc and descriptive, as the
  paper requires for real-data conditional-law diagnostics.
