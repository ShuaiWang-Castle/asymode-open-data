#!/bin/zsh
cd ${0:A:h}
until [ -f ../../runs/geo_weather_20260924/w1_base/fold05/DONE.json ]; do sleep 30; done
./run_queue.sh jobs_w1_event.txt 1
