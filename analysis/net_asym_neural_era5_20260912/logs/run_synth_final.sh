#!/usr/bin/env bash
# Task C final grid: 90 main fits + 30 gamma=0 control fits on 2 workers x 1 thread. Requires the committed synthetic selection lock.
set -u
# Synthetic workers capped to 1 thread; ANEEL keeps the uncapped 8-thread environment of the repair run.
WORK="$1"; cd "$WORK"
pids=()
for w in 0 1; do
  env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 python3.11 code/synth_train.py --phase final --work "$WORK" --worker $w --n-workers 2 --threads 1 > logs/synth_final_w$w.log 2>&1 &
  pids+=($!)
done
for i in 0 1; do wait "${pids[$i]}"; echo "$(date -u +%FT%TZ) synth_final worker=$i exit=$?" >> logs/EXIT_CODES.log; done
