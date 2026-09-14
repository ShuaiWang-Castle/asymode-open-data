#!/usr/bin/env bash
# Step 2 of finalisation, after RESULTS_ZH.md is written: checksums, scoped staging, results commit, push, fresh-clone verification.
# Usage: finalize_publish.sh <repo root> <round dir> <scratch dir>
set -euo pipefail
ROOT="$1"; WORK="$2"; SCRATCH="$3"
R=analysis/net_asym_neural_era5_20260912
BR=research/net-asym-neural-era5-20260912
cd "$WORK"
for f in RESULTS_ZH.md README.md logs/NEGATIVE_AND_FAILURES.md logs/TRIAL_REGISTRY.jsonl logs/CORRECTNESS_GATES.json results/COST_TABLE.csv; do
  [ -s "$f" ] || { echo "STOP: missing $f"; exit 1; }
done
find . -type f ! -name SHA256SUMS.txt ! -path '*/__pycache__/*' ! -path '*/.pytest_cache/*' -print0 | sort -z \
  | xargs -0 shasum -a 256 | sed 's|  \./|  |' > SHA256SUMS.txt
echo "SHA256SUMS: $(wc -l < SHA256SUMS.txt | tr -d ' ') entries, verified $(shasum -a 256 -c SHA256SUMS.txt 2>/dev/null | grep -c ': OK$')"
big=$(find . -type f -size +95M ! -path '*/__pycache__/*'); [ -z "$big" ] || { echo "STOP: files over 95 MB: $big"; exit 1; }
cd "$ROOT"
git ls-files --others --modified --exclude-standard -- "$R" | grep -v '__pycache__\|\.pytest_cache' > "$SCRATCH/results_stage_list.txt"
echo "staging $(wc -l < "$SCRATCH/results_stage_list.txt" | tr -d ' ') files under $R"
xargs git add -- < "$SCRATCH/results_stage_list.txt"
if git diff --cached --name-only | grep -v "^$R/"; then echo "STOP: out-of-scope path staged"; exit 1; fi
LEAK="/Us""ers/castle""wang|/priv""ate/tmp"
leaks=$(git diff --cached --name-only | xargs grep -I -l -E "$LEAK" 2>/dev/null || true)
[ -z "$leaks" ] || { echo "STOP: absolute paths in: $leaks"; exit 1; }
echo "staged size: $(git diff --cached --name-only | xargs du -ch 2>/dev/null | tail -1 | cut -f1)"
git -c user.name="$(git config user.name || echo 'Shuai Wang')" -c user.email="$(git config user.email || echo 'shuaiwangacademic@gmail.com')" \
    commit -q -F "$WORK/logs/RESULTS_COMMIT_MESSAGE.txt"
echo "RESULTS COMMIT: $(git rev-parse HEAD)"
git -c http.postBuffer=1048576000 push origin "$BR" 2>&1 | tail -2
remote=$(git ls-remote origin "$BR" | cut -c1-40)
[ "$remote" = "$(git rev-parse HEAD)" ] || { echo "STOP: remote $remote does not match local HEAD"; exit 1; }
V="$SCRATCH/verify_fresh_clone"; rm -rf "$V"
git clone -q --branch "$BR" --depth 1 https://github.com/ShuaiWang-Castle/asymode-open-data "$V"
cd "$V/$R"
echo "fresh clone HEAD: $(git -C "$V" rev-parse HEAD)"
echo "fresh clone checksums OK: $(shasum -a 256 -c SHA256SUMS.txt 2>/dev/null | grep -c ': OK$') / $(wc -l < SHA256SUMS.txt | tr -d ' ')"
./REPLAY_ONLY.sh --task us        --max-models 1 --threads 2 --out "$SCRATCH/fresh_replay_us.json"
./REPLAY_ONLY.sh --task synthetic --max-models 1 --threads 1 --out "$SCRATCH/fresh_replay_synthetic.json"
./REPLAY_ONLY.sh --task aneel     --max-models 1 --threads 8 --out "$SCRATCH/fresh_replay_aneel.json"
