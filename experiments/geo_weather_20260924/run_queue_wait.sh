#!/bin/zsh
# Run screen jobs (one line = the arguments of one screen.py call), starting a job only while fewer than N screen.py
# runs of this experiment are alive, so the queue can follow runs that were started elsewhere without exceeding N.
# usage: run_queue_wait.sh jobs.txt N
cd ${0:A:h}
setopt nobgnice
JOBS=$1; N=${2:-3}
grep -v '^#' $JOBS | grep -v '^$' | while IFS= read -r line; do
  while (( $(pgrep -f "screen.py --label" | wc -l) >= N )); do sleep 20; done
  lab=$(echo "$line" | sed -E "s/.*--label ([^ ]+).*/\1/"); f=$(echo "$line" | sed -E "s/.*--folds ([0-9]+).*/\1/")
  ../../.venv/bin/python -u screen.py ${=line} > logs/screen_${lab}_f${f}.log 2>&1 &
  sleep 20
done
wait
