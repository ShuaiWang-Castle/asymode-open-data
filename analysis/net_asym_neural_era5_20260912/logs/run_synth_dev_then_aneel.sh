#!/usr/bin/env bash
# Task C development tuning (2 workers x 1 thread) and selection lock, then Task B evaluation (8 threads).
set -u
# Synthetic workers capped to 1 thread; ANEEL keeps the uncapped 8-thread environment of the repair run.
ROOT="$1"; WORK="$2"; REPAIR="$3"; SCRATCH="$4"; cd "$WORK"
pids=()
for w in 0 1; do
  env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 python3.11 code/synth_train.py --phase dev --work "$WORK" --worker $w --n-workers 2 --threads 1 > logs/synth_dev_w$w.log 2>&1 &
  pids+=($!)
done
rc=0
for i in 0 1; do wait "${pids[$i]}"; e=$?; echo "$(date -u +%FT%TZ) synth_dev worker=$i exit=$e" >> logs/EXIT_CODES.log; [ $e -ne 0 ] && rc=1; done
if [ $rc -eq 0 ]; then
  env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 python3.11 code/synth_train.py --phase select --work "$WORK" > logs/synth_select.log 2>&1
  e=$?; echo "$(date -u +%FT%TZ) synth_select exit=$e" >> logs/EXIT_CODES.log
fi
python3.11 code/aneel_eval.py --root "$ROOT" --work "$WORK" --repair-dir "$REPAIR" --scratch "$SCRATCH" --threads 8 > logs/aneel_eval.log 2>&1
e=$?; echo "$(date -u +%FT%TZ) aneel_eval exit=$e" >> logs/EXIT_CODES.log
