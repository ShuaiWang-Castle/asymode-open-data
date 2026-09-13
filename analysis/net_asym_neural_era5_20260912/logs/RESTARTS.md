# Execution restarts

## 2026-09-13T00:45:20+00:00: development workers restarted with thread caps

- Symptom: the first development trials ran 3-8x slower than the 100-update profile
  (US about 300 s against about 90 s; synthetic NET about 160 s against about 20 s), with a host
  load average near 100 on 8 cores.
- Cause: the host is shared with long-running jobs from other sessions, which this round did not
  start and did not signal. This round's workers also ran uncapped OpenMP/BLAS thread pools
  (10-17 threads per process) on top of the torch thread setting.
- Action: only this round's development workers were stopped, then relaunched with
  OMP/OPENBLAS/VECLIB/MKL caps equal to the torch thread count (US 2, synthetic 1). Trials already
  recorded as COMPLETED were kept and skipped on resume; trials in flight were discarded and rerun
  from scratch.
- Two earlier stop attempts signalled nothing and failed closed: under zsh the PID list did not
  word-split, and macOS bash 3.2 has no `mapfile`. The third attempt used plain bash 3.2.
- Scope: execution environment only. No candidate, seed, update budget, split, loss or selection
  rule changed, and no evaluation result existed.
