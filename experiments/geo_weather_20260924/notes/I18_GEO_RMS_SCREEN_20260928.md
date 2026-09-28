# I18: retain geographic magnitude inside GCRK (2026-09-28)

Status: registered before training. The PI authorized this chat to run experiments
and prioritized the geographic kernel over the proposed recovery work. This is a
single-factor damage-side GCRK screen. I15-I17 remain deferred; no recovery changes.

## Hypothesis and exact change

With the existing standardized, clipped geography g and fitting mean
`center = mean_fit tanh(g/3)`, let `x = tanh(g/3) - center`.

- Existing: `z = tanh(U x / sqrt(0.1^2 + ||x||^2))`.
- I18: `s_geo = sqrt(0.1^2 + mean_fit ||x||^2)` and
  `z = tanh(U x / s_geo)`.

The scalar averages unweighted fitting county-event rows (repeated counties kept,
as for the existing center); the 40 coordinate squares are summed, not averaged.
It is fitted once after existing training-fold standardization, missing-value
handling and clipping, then stored as `geo_rms_scale` in the checkpoint. It never
uses held-out rows or changes with the hidden-state calibration.

This retains radial geographic deviations instead of nearly normalizing every
county to a unit vector. It adds no learned parameter and leaves the 4-dimensional
code, lambda/a/Omega maps, memory bounds, response recurrence, W2 readout, opening,
drop-path, optimizer and rate networks unchanged. Both arms initialize all host
weights identically; alpha=0 and zero conditional maps preserve exact step-0
host equivalence. The existing warmup remains unchanged, including its zero
opening gradient at step 0; this experiment does not change initialization clocks.

This is motivated by the measured reference sensitivity and narrow conditional
variation (notes/KERNEL_STATE_DIAGNOSTICS_20260928.md), and by the initialization/
normalization emphasis in LRU and stable state-space literature (notes/
LITERATURE_KERNELS_20260928.md). The specific RMS change is our hypothesis, not a
published guarantee. It does not make standardized g=0 equal latent z=0, prove
geographic information value, fix missing weather, or add spatial coupling.

Because z remains coordinatewise bounded by tanh, and lambda>0, ||a||<1/2 and
1/2<Omega<2 are unchanged, the existing response unit bound and stock [0,1] bound
continue to apply. Runtime remains O(N T d); the new fit-only computation is O(NG).

## Frozen experiment and decisions

- Data: existing public `features_v1D.npz`, 81 D systems / 8,457 county-events.
  Existing family-held-out event folds only. C never built, read or evaluated.
- Arm `GCRK+Cin-georms`; label `v1_gcrk_georms_s0`; seed 0; folds 1-5; 900 steps;
  existing regime-normalized design-weighted MSE; CPU, two threads per process.
- Comparators: completed `v1_host_s0` and `v1_gcrk_s0`, paired initialization.
  No additional seed or trained NULL runs during this screen.
- Primary: tropical/winter balanced MSE change. Also report all-five balanced MSE,
  each regime, worst regime, raw vs trimmed weights, and pooled MAE/RMSE at
  +1/+6/+24/+48 h with the existing 2,000 family-cluster draws, seed 20260924.
- A **clear candidate for further discussion** must improve the primary by more
  than 1% versus the host, improve it versus original GCRK, improve pooled +1 h
  RMSE versus both, have all-five balanced MSE change <=0 versus host, and have no
  non-headline regime worse than 2% versus host. This prospectively declared screen
  is deliberately conservative; it does not rewrite the registered three-seed keep
  test or retroactively reclassify old arms.
- Failure of these screening gates means no multi-seed expansion of I18. Passing
  is not a keep or an information claim: a survivor next needs seed averaging and
  a pre-specified, same-architecture trained geography NULL under the existing
  two-numbers rule. Freeze the NULL before its first run; frozen constant-input
  counterfactuals do not substitute for it.
- Partial folds may be described only on common completed held-out units; no
  selection, parameter adjustment or early stopping by their results. All five
  registered folds complete unless a software/resource failure prevents it.

## Correctness and operation

Before training: old equivalence/space tests, new geography-chain gradients and
full recurrence/reference gradient check, step-0 host equality, extreme-input
state bounds, fit/held-out separation, and checkpoint reload parity.

Start one low-priority process (nice >=15), inspect its resident memory and CPU,
then use at most two concurrent processes if the machine retains headroom. Never
touch another session's process. Run folders, training logs and detailed outputs
stay untracked; only code, design and compact result summaries enter Git.

Score only complete 900-step runs with matching label/data/arm/seed/fold metadata,
finite predictions and exactly one held-out prediction per unit. A finite runner
will evaluate both comparators under both headline definitions after all folds
finish, write a uniquely named pooled comparison, and stop. It must not overwrite
existing checkpoints, `kernels_s0.json`, paper files or another queue's output.
