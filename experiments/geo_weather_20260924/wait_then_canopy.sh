#!/bin/zsh
cd ${0:A:h}
until [ -f ../../runs/geo_weather_20260924/H_area/fold02/DONE.json ]; do sleep 30; done
./run_queue.sh jobs_canopy.txt 1
./run_queue.sh jobs_confirm.txt 1
