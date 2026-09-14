#!/usr/bin/env bash
# Task A development tuning: 36 runs on 3 workers x 2 threads, then the selection lock (2022 validation only).
set -u
# Thread pools capped to the torch thread count: the host is shared with other sessions' jobs.
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 MKL_NUM_THREADS=2
ROOT="$1"; WORK="$2"; cd "$WORK"
pids=()
for w in 0 1 2; do
  python3.11 code/us_train.py --phase dev --root "$ROOT" --work "$WORK" --worker $w --n-workers 3 --threads 2 > logs/us_dev_w$w.log 2>&1 &
  pids+=($!)
done
rc=0
for i in 0 1 2; do wait "${pids[$i]}"; e=$?; echo "$(date -u +%FT%TZ) us_dev worker=$i exit=$e" >> logs/EXIT_CODES.log; [ $e -ne 0 ] && rc=1; done
if [ $rc -eq 0 ]; then
  python3.11 code/us_train.py --phase select --root "$ROOT" --work "$WORK" > logs/us_select.log 2>&1
  e=$?; echo "$(date -u +%FT%TZ) us_select exit=$e" >> logs/EXIT_CODES.log
fi
