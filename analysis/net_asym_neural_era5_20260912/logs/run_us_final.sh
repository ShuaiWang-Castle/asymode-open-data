#!/usr/bin/env bash
# Task A final fits: 10 runs (2 models x 5 seeds) on 3 workers x 2 threads. Requires the committed US selection lock.
set -u
# Thread pools capped to the torch thread count: the host is shared with other sessions' jobs.
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 MKL_NUM_THREADS=2
ROOT="$1"; WORK="$2"; cd "$WORK"
pids=()
for w in 0 1 2; do
  python3.11 code/us_train.py --phase final --root "$ROOT" --work "$WORK" --worker $w --n-workers 3 --threads 2 > logs/us_final_w$w.log 2>&1 &
  pids+=($!)
done
for i in 0 1 2; do wait "${pids[$i]}"; e=$?; echo "$(date -u +%FT%TZ) us_final worker=$i exit=$e" >> logs/EXIT_CODES.log; done
