#!/bin/zsh
# H2b seeds 1-2 (PREREG_W2 amendment 7), after the seed-0 runs.
cd ${0:A:h}
until [ $(ls ../../runs/geo_weather_20260924/h2b_{base,Hc,Hp}/fold0*/DONE.json 2>/dev/null | wc -l) -ge 15 ]; do sleep 60; done
./run_queue.sh jobs_h2b_seeds.txt 3
echo H2B_SEEDS_DONE
