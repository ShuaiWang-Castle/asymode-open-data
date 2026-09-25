#!/bin/zsh
cd ${0:A:h}
PY=../../.venv/bin/python
$PY -u build_eih_hrrr_coarse.py --feat data/interim/geo_weather/features_w1.npz --events-file experiments/geo_weather_20260924/selected_events_winter_build.json --tag w1_ --workers 8 || exit 1
$PY check_eih_layout.py ../../data/interim/geo_weather/eih_w1_hrrr_coarse.npz || exit 1
$PY audit_f0.py --feat data/interim/geo_weather/features_w1.npz --prefix w1_ --out F0_w1_hrrr_split --pairs hrrr-hrrr_coarse hrrr_coarse-pop hrrr-pop quad-pop
echo COARSE_W1_DONE
