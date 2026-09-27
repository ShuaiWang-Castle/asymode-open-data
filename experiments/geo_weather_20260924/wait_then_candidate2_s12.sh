#!/bin/zsh
# seeds 1-2 of the full HRRR dictionary (a second candidate) and its ERA5 twin, after the first candidate's seeds
cd ${0:A:h}
while kill -0 64696 2>/dev/null; do sleep 30; done
nice -n 15 zsh run_queue.sh jobs_v1_candidate2_s12.txt 1
