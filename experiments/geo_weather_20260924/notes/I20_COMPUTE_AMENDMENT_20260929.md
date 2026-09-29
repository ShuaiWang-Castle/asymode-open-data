# I20 compute-only amendment (2026-09-29)

After I20 source registration `6b6b964` and the first fold's launch, the PI explicitly
freed CPU and authorized multiple processes/threads. Use at most **three folds in
parallel, two numerical threads each, nice>=15**, on the eight-core machine. Resource
checks before this amendment show 57% system memory free by the macOS pressure query;
the registered one-update preflight peaked at 1.82 GiB, and the first live fold uses
about 1.1 GiB RSS. Continue to inspect actual CPU/RSS and memory pressure. The separate
`PARALLEL_WORKERS` control can reduce future concurrency to 1 or 2 without stopping
healthy training. Do not increase beyond 3 under this amendment.

This changes scheduling only. The seed, 900 full-fit Adam updates, county chunk512,
two-thread arithmetic, data, loss, model, outputs, affected-cohort definition and
evaluation remain fixed. Preserve all sources in the original SCREEN_RUN manifest,
including the serial runner and original registration. A new `run_cr_parallel.py`
imports the registered validators and evaluator, preserving their strict checks.

Handoff verifies the serial coordinator and its sole live child's PID, start time,
command and parent relationship. It records the original state and compute-source
hashes before sending SIGTERM **only to the lightweight serial coordinator's PID**.
It never signals the training child or the process group. After taking the original
runner lock, it monitors the same live fold without restarting it, and starts only
unstarted folds. Existing/empty fold directories remain protected by atomic refusal.
An adopted fold needs both process exit and valid complete exports before receiving
the original artifact/source receipt. Child process failures stop new launches and
leave other healthy processes untouched for inspection.

The original manifest's workers=1 describes the initial execution. The independent
COMPUTE_AMENDMENT.json records workers=3 and adoption provenance; it does not rewrite
history or assign a new model/source configuration to an already-running fold.
The current RUN_STATUS records all active PIDs. The same five-fold-only evaluation
runs once all training processes have exited and all five receipts pass.

No new scientific hypothesis, training run, seed, NULL comparison or evaluation
selection is introduced. No C access or manuscript work is authorized. The existing
I20 monitor follows this newer compute authorization and pauses after result delivery.

Before handoff, 24 synthetic scheduling/evaluation checks pass, including verified
parent-only signalling, PID reuse detection, child reparenting, bounded worker
settings, nonduplicated fold ownership, frozen evaluation and partial-output refusal.
