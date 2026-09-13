#!/usr/bin/env bash
# Full rerun of this round from the pinned repository data. Retrains every model.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PY=python3.11
R1="$ROOT/analysis/asymode_trajectory_r1_20260909"
cd "$HERE"
# 1. intake and locks
$PY code/us_audit.py --root "$ROOT" --out intake
$PY code/aneel_ledger_identity.py --ledger "$R1/data_resolution/rebuilt_selected_ledgers.npz" \
    --reference "$R1/_package/ASYMODE_CC_STANDALONE_20260909/metadata/LEDGER_REFERENCE.json" --out intake/ANEEL_LEDGER_IDENTITY.json
$PY code/aneel_selection_recompute.py --stage1 "$R1/results/STAGE1_TRIALS.csv" --stage2 "$R1/results/STAGE2_TRIALS.csv" \
    --origins "$R1/preflight/origin_hours_L24_H24.npz" --out locks
$PY code/us_data.py --root "$ROOT" --locks locks --intake intake
$PY code/synth_generator.py --out results/synthetic
# 2. correctness gates
$PY -m pytest -q tests
# 3. Task A (US ERA5): tuning, selection lock, final fits
for w in 0 1 2; do $PY code/us_train.py --phase dev --root "$ROOT" --work "$HERE" --worker $w --n-workers 3 --threads 2 & done; wait
$PY code/us_train.py --phase select --root "$ROOT" --work "$HERE"
for w in 0 1 2; do $PY code/us_train.py --phase final --root "$ROOT" --work "$HERE" --worker $w --n-workers 3 --threads 2 & done; wait
# 4. Task C (synthetic): tuning, selection lock, final grid
for w in 0 1; do $PY code/synth_train.py --phase dev --work "$HERE" --worker $w --n-workers 2 --threads 1 & done; wait
$PY code/synth_train.py --phase select --work "$HERE"
for w in 0 1; do $PY code/synth_train.py --phase final --work "$HERE" --worker $w --n-workers 2 --threads 1 & done; wait
# 5. Task B (ANEEL): the repair checkpoints must be supplied via --repair-dir to reproduce the replay step
echo "Task B: run code/aneel_eval.py --root $ROOT --work $HERE --repair-dir <dir with main/ and predictions/> --scratch \$(mktemp -d)"
# 6. analysis, figures, replay
$PY code/analyze.py --root "$ROOT" --work "$HERE"
./REPLAY_ONLY.sh
