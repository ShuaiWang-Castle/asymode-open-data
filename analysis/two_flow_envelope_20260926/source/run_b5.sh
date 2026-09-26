#!/usr/bin/env bash
# Launch B4' workers (resumable: completed fits are skipped) and wait for all of them.
set -u
B="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$HOME/asymode-open-data-work"
WORKERS=${WORKERS:-6}
cd "$B" || exit 1
echo "B5 launch $(date -u +%FT%TZ) workers=$WORKERS" >> logs/b4_run.log
pids=()
for w in $(seq 0 $((WORKERS - 1))); do
  python3.11 source/b5_experiment.py --repo "$REPO" --worker "$w" --workers "$WORKERS" --threads 1 >> "logs/b5_worker$w.log" 2>&1 &
  pids+=($!)
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
echo "B5 finished $(date -u +%FT%TZ) status=$status fits=$(ls results/b5_fits/*.json 2>/dev/null | wc -l)" >> logs/b4_run.log
exit $status
