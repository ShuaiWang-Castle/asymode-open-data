#!/usr/bin/env bash
# Reproduce ASYMODE_TRAJECTORY_MAINLINE_R1_20260909 end to end.
set -euo pipefail
D="${D:?set D to the project working directory}"
A="${A:-$HOME/Desktop/DMDA/ANEE}"          # local root holding the ANEEL annual ZIPs
W="$D/runs/asymode_trajectory_r1_20260909"
P="$W/_package/ASYMODE_CC_STANDALONE_20260909"
PY="${PY:-python3.11}"

"$PY" "$P/code/verify_package.py" --root "$P"
"$PY" "$P/code/prepare_local_data.py" --data-root "$D" --data-root "$A" --output "$W/data_resolution"
"$PY" "$P/code/preflight.py" --data-manifest "$W/data_resolution/DATA_MANIFEST.json" --output "$W/preflight"
"$PY" "$W/correctness_gates.py" --package-code "$P/code" --output "$W/gates"

for phase in main ablation controlled package; do
  "$PY" "$W/run_campaign.py" \
    --data-manifest "$W/data_resolution/DATA_MANIFEST.json" \
    --protocol "$P/CONFIG_LOCK.json" --work "$W" --phase "$phase" \
    --package-code "$P/code" --preflight "$W/preflight" 2>&1 | tee "$W/logs/$phase.log"
done

"$PY" "$W/independent_rescore.py" --work "$W"
"$PY" "$W/paired_analysis.py" --work "$W"
