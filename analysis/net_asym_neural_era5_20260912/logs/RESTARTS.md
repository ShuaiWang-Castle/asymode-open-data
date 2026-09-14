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

## 2026-09-13T01:30:36Z: paused at the PI's request

- The PI asked to free the machine for another session's jobs. This round's Python workers
  (PIDs 48965 48966 48967 51141 51142 50998) were suspended with SIGSTOP; nothing was killed and no state was discarded.
- Resume is SIGCONT on the same processes, so every in-flight trial continues from where it
  stopped with identical arithmetic.
- Consequence: the recorded wall_seconds of trials in flight at this moment include the pause
  and overstate their compute cost; the cost table must be read with this entry.

## 2026-09-13T01:34:28+00:00: pause converted to a full stop at the PI's request

- To free memory as well as CPU, this round's suspended workers were terminated (SIGKILL,
  since stopped processes do not act on SIGTERM) after their orchestrator scripts were stopped
  first, so no orchestrator could advance to a selection or evaluation step.
- Kept: every COMPLETED trial record (US development 21/36, synthetic final 17/120). All
  registry lines parse. On relaunch, the resume logic skips these.
- Lost and to be rerun from scratch: the trials in flight (3 US development trials, 2 synthetic
  final fits) and the whole ANEEL evaluation, which was in its spot retrain. The ANEEL
  replay had already matched all 10 repair checkpoints bitwise; it reruns in full.
- Artifacts of in-flight synthetic fits without a registry record: none. They are
  not results; a rerun of the same fit overwrites them deterministically.
- No evaluation table was produced before the stop, and no scientific setting changed.

- Correction (2026-09-13T01:35:01+00:00): the stop script's orchestrator pattern matched nothing, because the
  orchestrators had been relaunched with a relative script path. They were therefore not
  signalled; each exited on its own once its workers died. Verified afterwards: no process of
  this round remains, the US orchestrator skipped selection because its workers exited
  non-zero (no US selection lock and no selection log exist), and no ANEEL result table or
  US final record exists.

- Exit-code logging bug found at this stop (2026-09-13T01:35:53+00:00): three orchestrators logged `exit=$?` after a
  `$(date)` substitution inside the same string, so they recorded the status of `date`, not of the
  job. The affected lines in `logs/EXIT_CODES.log` are annotated there (append-only). The
  orchestrators now capture `e=$?` before logging. The US development worker lines were unaffected
  and correctly show 137.
