#!/usr/bin/env bash
# Tropical v2: inner stage (learning-rate and step selection on held-out storms), then final refits with five seeds.
set -u
cd "$(dirname "$0")" || exit 1
PANEL="${PANEL:-$HOME/Desktop/DMDA/AsymODE-Public/data/interim/panel_v1/features_v1D.npz}"
SPLITS="${SPLITS:-$HOME/Desktop/DMDA/AsymODE-Public/experiments/geo_weather_20260924/splits_v1D.json}"
WORKERS=${WORKERS:-4}
for stage in inner final; do
  echo "v2 $stage launch $(date -u +%FT%TZ) workers=$WORKERS" >> logs/run.log
  pids=()
  for w in $(seq 0 $((WORKERS - 1))); do
    nice -n 10 python3.11 tropical_v2.py --panel "$PANEL" --splits "$SPLITS" --stage $stage --worker "$w" --workers "$WORKERS" >> "logs/v2_${stage}_worker$w.log" 2>&1 &
    pids+=($!)
  done
  status=0
  for p in "${pids[@]}"; do wait "$p" || status=1; done
  echo "v2 $stage finished $(date -u +%FT%TZ) status=$status files=$(ls results/v2/$stage/*.json 2>/dev/null | wc -l)" >> logs/run.log
  [ $status -eq 0 ] || exit $status
done
