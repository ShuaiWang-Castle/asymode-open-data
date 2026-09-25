#!/bin/zsh
cd ${0:A:h}
until [ -f ../../runs/geo_weather_20260924/w2e_base/fold03/DONE.json ] && [ -f ../../runs/geo_weather_20260924/w2e_base/fold01/DONE.json ]; do sleep 30; done
./run_queue.sh jobs_w1e_hrrr_coarse.txt 1
