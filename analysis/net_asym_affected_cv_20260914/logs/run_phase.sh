#!/usr/bin/env bash
# Orchestrator for the dev and final phases: 3 workers x 2 threads.
# Usage: run_phase.sh <repo root> dev|final
set -uo pipefail
ROOT="$1"; PHASE="$2"; W="$(cd "$(dirname "$0")/.." && pwd)"
case "$PHASE" in dev|final) ;; *) echo "phase must be dev or final"; exit 2 ;; esac
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 MKL_NUM_THREADS=2
cd "$W"
echo "$PHASE launch=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> logs/EXIT_CODES.log
pids=()
for w in 0 1 2; do
  python3.11 code/cv_train.py --phase "$PHASE" --root "$ROOT" --work "$W" --worker "$w" --n-workers 3 --threads 2 >> "logs/${PHASE}_w$w.log" 2>&1 &
  pids+=($!)
done
status=0
for i in 0 1 2; do
  wait "${pids[$i]}"; e=$?
  echo "$PHASE worker $i exit=$e finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> logs/EXIT_CODES.log
  [ "$e" -eq 0 ] || status=1
done
n=$(cat logs/${PHASE}_w*.jsonl 2>/dev/null | grep -c '"status": "COMPLETED"')
echo "$PHASE finished: status=$status completed_runs=$n"
exit $status
