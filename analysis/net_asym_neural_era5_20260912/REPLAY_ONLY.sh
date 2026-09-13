#!/usr/bin/env bash
# Checkpoint-only replay: loads checkpoints and scores them. Never trains.
# Usage: ./REPLAY_ONLY.sh [--task us|synthetic|aneel|all] [--max-models N]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
SCRATCH="$(mktemp -d)"
python3.11 "$HERE/code/replay.py" --root "$ROOT" --work "$HERE" --scratch "$SCRATCH" "$@"
