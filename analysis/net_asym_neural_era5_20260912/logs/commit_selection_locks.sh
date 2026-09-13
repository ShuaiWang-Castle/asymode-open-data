#!/usr/bin/env bash
# Commit and push one task's selection lock with its development registries, before that task's final fits.
# Usage: commit_selection_locks.sh <repo root> us|synthetic
set -euo pipefail
ROOT="$1"; TASK="$2"; cd "$ROOT"
R=analysis/net_asym_neural_era5_20260912
case "$TASK" in
  us)        LOCK="$R/locks/US_SELECTION_LOCK.json";    PAT="us_dev_w";    SEL="$R/logs/us_select.log";    RUN="$R/logs/run_us_dev.sh $R/logs/run_us_final.sh";
             WHAT="the 2022 validation split" ;;
  synthetic) LOCK="$R/locks/SYNTH_SELECTION_LOCK.json"; PAT="synth_dev_w"; SEL="$R/logs/synth_select.log"; RUN="$R/logs/run_synth_dev_then_aneel.sh $R/logs/run_synth_final.sh";
             WHAT="new validation events in the middle synthetic law (n=128, gamma=0.04)" ;;
  *) echo "task must be us or synthetic"; exit 1 ;;
esac
files="$LOCK $SEL $RUN $R/logs/commit_selection_locks.sh $R/logs/RESTARTS.md $R/locks/RESOURCE_PLAN.json $(ls $R/logs/${PAT}*.jsonl)"
for f in $files; do [ -f "$f" ] || { echo "MISSING $f"; exit 1; }; done
git add -- $files
if git diff --cached --name-only | grep -v "^$R/"; then echo "out of scope"; exit 1; fi
# The pattern is assembled from fragments so this script cannot match its own source.
LEAK="/Us""ers/castle""wang|/priv""ate/tmp"
if git diff --cached --name-only | xargs grep -I -l -E "$LEAK" 2>/dev/null; then echo "absolute path leak"; exit 1; fi
git -c user.name="$(git config user.name || echo 'Shuai Wang')" -c user.email="$(git config user.email || echo 'shuaiwangacademic@gmail.com')" \
    commit -q -m "locks: $TASK selection fixed before any final fit or evaluation" -m "Each model's configuration and final update count come from its own development runs on $WHAT. The development trial registry is committed with the lock so the choice can be recomputed. A thread-cap restart during tuning is recorded in logs/RESTARTS.md; it changed the execution environment only." -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
echo "LOCK COMMIT ($TASK): $(git rev-parse HEAD)"
git -c http.postBuffer=524288000 push origin research/net-asym-neural-era5-20260912 2>&1 | tail -1
