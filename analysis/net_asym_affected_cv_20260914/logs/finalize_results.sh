#!/usr/bin/env bash
# Finalise the affected-county CV round: checksums, size and leak checks, scoped staging, local results commit. Does not push.
# Usage: finalize_results.sh <repo root>
set -euo pipefail
ROOT="$1"; R=analysis/net_asym_affected_cv_20260914
cd "$ROOT/$R"
for f in README.md RESULTS_ZH.md results/CV_SUMMARY.json results/CV_GROUP_SUMMARY.csv logs/REPLAY_CV.json logs/RESULTS_COMMIT_MESSAGE.txt; do
  [ -s "$f" ] || { echo "STOP: missing $f"; exit 1; }
done
find . -type f ! -name SHA256SUMS.txt ! -path '*/__pycache__/*' ! -path '*/.pytest_cache/*' -print0 | sort -z \
  | xargs -0 shasum -a 256 | sed 's|  \./|  |' > SHA256SUMS.txt
echo "SHA256SUMS: $(wc -l < SHA256SUMS.txt | tr -d ' ') entries, verified $(shasum -a 256 -c SHA256SUMS.txt 2>/dev/null | grep -c ': OK$')"
big=$(find . -type f -size +95M ! -path '*/__pycache__/*'); [ -z "$big" ] || { echo "STOP: files over 95 MB: $big"; exit 1; }
LEAK="/Us""ers/castle""wang|/priv""ate/tmp"
if grep -r -a -l -E "$LEAK" . --exclude-dir=__pycache__ --exclude-dir=.pytest_cache >/dev/null 2>&1; then
  echo "STOP: absolute paths in:"; grep -r -a -l -E "$LEAK" . --exclude-dir=__pycache__ --exclude-dir=.pytest_cache; exit 1
fi
cd "$ROOT"
LIST="$(mktemp)"
git ls-files --others --modified --exclude-standard -- "$R" | grep -v '__pycache__\|\.pytest_cache' > "$LIST"
echo "staging $(wc -l < "$LIST" | tr -d ' ') files under $R"
xargs git add -- < "$LIST"
if git diff --cached --name-only | grep -v "^$R/"; then echo "STOP: out-of-scope path staged"; exit 1; fi
echo "staged size: $(git diff --cached --name-only | xargs du -ch 2>/dev/null | tail -1 | cut -f1)"
git -c user.name="$(git config user.name || echo 'Shuai Wang')" -c user.email="$(git config user.email || echo 'shuaiwangacademic@gmail.com')" \
    commit -q -F "$R/logs/RESULTS_COMMIT_MESSAGE.txt"
echo "LOCAL COMMIT: $(git rev-parse HEAD) (not pushed)"
