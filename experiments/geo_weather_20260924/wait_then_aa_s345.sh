#!/bin/zsh
# the registered A/A run (host seeds 3-5), after the first candidate's seeds (one training process)
cd ${0:A:h}
while kill -0 64696 2>/dev/null; do sleep 30; done
nice -n 15 zsh run_queue.sh jobs_v1_aa_s345.txt 1
