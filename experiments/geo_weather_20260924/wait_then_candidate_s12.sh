#!/bin/zsh
# seeds 1-2 of the host, the HRRR mechanical-load candidate and its ERA5 twin, after the seed-0 queue (one training process)
cd ${0:A:h}
while kill -0 24375 2>/dev/null; do sleep 30; done
nice -n 15 zsh run_queue.sh jobs_v1_candidate_s12.txt 1
