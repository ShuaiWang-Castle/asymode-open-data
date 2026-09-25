#!/bin/zsh
cd ${0:A:h}
PY=../../.venv/bin/python
$PY -u build_eih_hrrr.py --feat data/interim/geo_weather/features_w1.npz --events-file experiments/geo_weather_20260924/selected_events_winter_build.json --tag w1_ --workers 16 || exit 1
$PY check_eih_layout.py ../../data/interim/geo_weather/eih_w1_hrrr.npz || exit 1
$PY audit_f0.py --feat data/interim/geo_weather/features_w1.npz --prefix w1_ --out F0_w1_hrrr --pairs hrrr-pop hrrr-quad quad-pop
./run_queue.sh jobs_w1_hrrr.txt 2
echo HRRR_W1_DONE
