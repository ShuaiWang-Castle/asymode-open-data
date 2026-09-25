#!/bin/zsh
cd ${0:A:h}
until grep -qE "CHAIN_W2E_DONE|fold . missing|Traceback" logs/chain_w2e.log; do sleep 30; done
./run_queue.sh jobs_w1e_hrrr_seeds.txt 1
