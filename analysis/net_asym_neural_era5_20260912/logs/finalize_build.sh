#!/usr/bin/env bash
# Step 1 of finalisation: completeness gates, analysis from saved predictions, checkpoint-only replay per task.
# Usage: finalize_build.sh <repo root> <round dir>
set -euo pipefail
ROOT="$1"; WORK="$2"; cd "$WORK"
us=$(cat logs/us_final_w*.jsonl 2>/dev/null | wc -l | tr -d ' ')
sy=$(cat logs/synth_final_w*.jsonl 2>/dev/null | wc -l | tr -d ' ')
echo "completeness: US final $us/10, synthetic final $sy/120, ANEEL summary $([ -f results/aneel/ANEEL_PAIRED_SUMMARY.json ] && echo present || echo absent)"
[ "$us" = 10 ] && [ "$sy" = 120 ] && [ -f results/aneel/ANEEL_PAIRED_SUMMARY.json ] || { echo "STOP: runs incomplete"; exit 1; }
python3.11 code/analyze.py --root "$ROOT" --work "$WORK" > logs/analyze.log 2>&1
echo "analysis exit=0"; tail -4 logs/analyze.log
# replay each task at the thread count it was trained and predicted with
./REPLAY_ONLY.sh --task us        --threads 2 --out logs/REPLAY_US.json        > logs/replay_us.log 2>&1
./REPLAY_ONLY.sh --task synthetic --threads 1 --out logs/REPLAY_SYNTHETIC.json > logs/replay_synthetic.log 2>&1
./REPLAY_ONLY.sh --task aneel     --threads 8 --out logs/REPLAY_ANEEL.json     > logs/replay_aneel.log 2>&1
python3.11 - "$WORK" <<'PY'
import json, sys
from pathlib import Path
W = Path(sys.argv[1]); gates = json.loads((W / 'logs/CORRECTNESS_GATES.json').read_text()); out = {}
for task, f in (('us_era5', 'REPLAY_US.json'), ('synthetic', 'REPLAY_SYNTHETIC.json'), ('aneel', 'REPLAY_ANEEL.json')):
    r = json.loads((W / 'logs' / f).read_text())
    out[task] = {'replayed_checkpoints': r['replayed_checkpoints'], 'worst_max_abs_diff_vs_shard': r['worst_max_abs_diff_vs_shard'], 'exact': r['exact']}
gates['checkpoint_only_replay'] = {'status': 'COMPLETED', 'per_task': out, 'expected_counts': {'us_era5': 10, 'synthetic': 120, 'aneel': 10},
    'pass': all(out[t]['replayed_checkpoints'] == n for t, n in (('us_era5', 10), ('synthetic', 120), ('aneel', 10))) and all(v['exact'] for v in out.values())}
(W / 'logs/CORRECTNESS_GATES.json').write_text(json.dumps(gates, indent=1))
print('replay gate:', json.dumps(gates['checkpoint_only_replay']))
PY
