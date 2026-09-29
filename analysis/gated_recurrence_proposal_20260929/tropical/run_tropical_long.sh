#!/usr/bin/env bash
# Launch the tropical fits (resumable) and wait for all workers.
set -u
cd "$(dirname "$0")" || exit 1
PANEL="${PANEL:-$HOME/Desktop/DMDA/AsymODE-Public/data/interim/panel_v1/features_v1D.npz}"
SPLITS="${SPLITS:-$HOME/Desktop/DMDA/AsymODE-Public/experiments/geo_weather_20260924/splits_v1D.json}"
WORKERS=${WORKERS:-3}
echo "tropical-long launch $(date -u +%FT%TZ) workers=$WORKERS" >> logs/run.log
pids=()
for w in $(seq 0 $((WORKERS - 1))); do
  nice -n 10 python3.11 tropical_run_long.py --panel "$PANEL" --splits "$SPLITS" --worker "$w" --workers "$WORKERS" >> "logs/long_worker$w.log" 2>&1 &
  pids+=($!)
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
echo "tropical-long finished $(date -u +%FT%TZ) status=$status fits=$(ls results/fits_long/*.json 2>/dev/null | wc -l)" >> logs/run.log
exit $status
