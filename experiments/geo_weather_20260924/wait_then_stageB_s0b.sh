#!/bin/zsh
# start the reordered seed-0 queue once the running GCRK-open fold finishes (one training process at a time)
cd ${0:A:h}
while kill -0 14404 2>/dev/null; do sleep 30; done
nice -n 15 zsh run_queue.sh jobs_v1_stageB_s0b.txt 1
