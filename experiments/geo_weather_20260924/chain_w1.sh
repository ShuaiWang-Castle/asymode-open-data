#!/bin/zsh
# After build_winter.py: training mask (+ placebo), EIH features and the F0 audit for panel W1.
cd ${0:A:h}
until [ -f ../../data/interim/geo_weather/features_w1.npz ] && [ -f splits_w1.json ]; do sleep 20; done
PY=../../.venv/bin/python
$PY build_train_mask.py --feat data/interim/geo_weather/features_w1.npz --tag w1
$PY build_train_mask_placebo.py --feat data/interim/geo_weather/features_w1.npz --tag w1
$PY -u build_eih.py --feat data/interim/geo_weather/features_w1.npz --events-file experiments/geo_weather_20260924/selected_events_winter_build.json --tag w1_ --variants area pop quad mean pooled other quadn
$PY audit_f0.py --feat data/interim/geo_weather/features_w1.npz --prefix w1_ --out F0_w1
echo CHAIN_DONE
