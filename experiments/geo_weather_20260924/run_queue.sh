#!/bin/zsh
# Run screen jobs (one line = the arguments of one screen.py call) with N parallel workers.
# usage: run_queue.sh jobs.txt N
cd ${0:A:h}
JOBS=$1; N=${2:-3}
cat $JOBS | grep -v '^#' | grep -v '^$' | xargs -P $N -L 1 zsh -c 'lab=$(echo "$@" | sed -E "s/.*--label ([^ ]+).*/\1/"); f=$(echo "$@" | sed -E "s/.*--folds ([0-9]+).*/\1/"); ../../.venv/bin/python -u screen.py "$@" > logs/screen_${lab}_f${f}.log 2>&1' _
