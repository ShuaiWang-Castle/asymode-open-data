#!/usr/bin/env bash
# After DEV: completeness check, selection lock, stage commit and push, then the final phase.
# Usage: after_dev.sh <repo root>
set -uo pipefail
ROOT="$1"; R=analysis/net_asym_affected_cv_20260914; BR=research/net-asym-affected-cv-20260914
W="$ROOT/$R"; cd "$W" || exit 1
n=$(cat logs/dev_w0.jsonl logs/dev_w1.jsonl logs/dev_w2.jsonl 2>/dev/null | grep -c '"status": "COMPLETED"')
bad=$(grep -E '^dev worker [0-9] exit=' logs/EXIT_CODES.log | grep -vc 'exit=0 ')
echo "DEV completed runs: $n/180; worker exits other than 0: $bad"
[ "$n" -eq 180 ] && [ "$bad" -eq 0 ] || { echo "STOP: DEV incomplete"; exit 1; }
[ -e locks/SELECTION_LOCK.json ] && { echo "STOP: selection lock already exists"; exit 1; }
python3.11 code/cv_train.py --phase select --root "$ROOT" --work "$W" > logs/select.log 2>&1 || { cat logs/select.log; exit 1; }
cat logs/select.log
rm -rf code/__pycache__
LEAK="/Us""ers/castle""wang|/priv""ate/tmp"
if grep -r -I -l -E "$LEAK" logs locks >/dev/null 2>&1; then
  echo "STOP: absolute paths in:"; grep -r -I -l -E "$LEAK" logs locks; exit 1
fi
cd "$ROOT" || exit 1
git ls-files --others --modified --exclude-standard -- "$R" | grep -v '__pycache__\|\.pytest_cache' | xargs git add --
if git diff --cached --name-only | grep -v "^$R/"; then echo "STOP: out-of-scope path staged"; exit 1; fi
git -c user.name="$(git config user.name || echo 'Shuai Wang')" -c user.email="$(git config user.email || echo 'shuaiwangacademic@gmail.com')" commit -q -F - <<'MSG'
locks: affected-county CV selection fixed before any final fit or test score

All 180 DEV runs (5 outer folds x 2 models x 9 configurations x seeds 9101 and 9102) completed.
Per outer fold and model, the configuration with the lowest mean best inner-validation path MSE
is locked in locks/SELECTION_LOCK.json. DEV logs included. No final run and no test score exist.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
MSG
echo "SELECTION COMMIT: $(git rev-parse HEAD)"
git push origin "$BR" 2>&1 | tail -2
remote=$(git ls-remote origin "$BR" | cut -c1-40)
[ "$remote" = "$(git rev-parse HEAD)" ] || { echo "STOP: push not confirmed (remote $remote)"; exit 1; }
echo "pushed: $remote"
bash "$W/logs/run_phase.sh" "$ROOT" final
