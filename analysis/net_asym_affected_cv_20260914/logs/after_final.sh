#!/usr/bin/env bash
# After final: completeness check, checkpoint-only replay, pre-registered analysis, stage commit and push of raw results.
# The write-up (README.md, RESULTS_ZH.md) and SHA256SUMS follow in finalize_results.sh.
# Usage: after_final.sh <repo root>
set -uo pipefail
ROOT="$1"; R=analysis/net_asym_affected_cv_20260914; BR=research/net-asym-affected-cv-20260914
W="$ROOT/$R"; cd "$W" || exit 1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 MKL_NUM_THREADS=2
n=$(cat logs/final_w0.jsonl logs/final_w1.jsonl logs/final_w2.jsonl 2>/dev/null | grep -c '"status": "COMPLETED"')
bad=$(grep -E '^final worker [0-9] exit=' logs/EXIT_CODES.log | grep -vc 'exit=0 ')
echo "final completed runs: $n/50; worker exits other than 0: $bad"
[ "$n" -eq 50 ] && [ "$bad" -eq 0 ] || { echo "STOP: final incomplete"; exit 1; }
python3.11 code/cv_replay.py --root "$ROOT" --threads 2 --out logs/REPLAY_CV.json > logs/replay_cv.log 2>&1 || { tail -20 logs/replay_cv.log; exit 1; }
tail -1 logs/replay_cv.log
python3.11 code/cv_analyze.py --root "$ROOT" --threads 2 > logs/analyze.log 2>&1 || { tail -30 logs/analyze.log; exit 1; }
cat logs/analyze.log
rm -rf code/__pycache__ tests/__pycache__
LEAK="/Us""ers/castle""wang|/priv""ate/tmp"
if grep -r -a -l -E "$LEAK" . --exclude-dir=__pycache__ --exclude-dir=.pytest_cache >/dev/null 2>&1; then
  echo "STOP: absolute paths in:"; grep -r -a -l -E "$LEAK" . --exclude-dir=__pycache__ --exclude-dir=.pytest_cache; exit 1
fi
big=$(find . -type f -size +95M ! -path '*/__pycache__/*'); [ -z "$big" ] || { echo "STOP: files over 95 MB: $big"; exit 1; }
cd "$ROOT" || exit 1
git ls-files --others --modified --exclude-standard -- "$R" | grep -v '__pycache__\|\.pytest_cache' | xargs git add --
if git diff --cached --name-only | grep -v "^$R/"; then echo "STOP: out-of-scope path staged"; exit 1; fi
echo "staged: $(git diff --cached --name-only | wc -l | tr -d ' ') files, $(git diff --cached --name-only | xargs du -ch 2>/dev/null | tail -1 | cut -f1)"
git -c user.name="$(git config user.name || echo 'Shuai Wang')" -c user.email="$(git config user.email || echo 'shuaiwangacademic@gmail.com')" commit -q -F - <<'MSG'
results: affected-county CV final fits, checkpoint replay and pre-registered analysis (raw outputs)

Fifty final fits (5 outer folds x 2 models x seeds 9201-9205) under the committed selection
lock, per-seed early stopping on inner validation, test predictions for every legal window of
each outer fold, checkpoint-only replay of all checkpoints, and the output of the analysis
script committed before any DEV run. The written interpretation follows in a separate commit.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
MSG
echo "RESULTS COMMIT: $(git rev-parse HEAD)"
git -c http.postBuffer=1048576000 push origin "$BR" 2>&1 | tail -2
remote=$(git ls-remote origin "$BR" | cut -c1-40)
[ "$remote" = "$(git rev-parse HEAD)" ] || { echo "STOP: push not confirmed (remote $remote)"; exit 1; }
echo "pushed: $remote"
