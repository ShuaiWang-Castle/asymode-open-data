#!/bin/zsh
# Start the five-fold confirmation of the cleaning once the wind hazard arm's second fold is done.
cd ${0:A:h}
until [ -f ../../runs/geo_weather_20260924/H_area/fold02/DONE.json ]; do sleep 30; done
./run_queue.sh jobs_confirm.txt 1
