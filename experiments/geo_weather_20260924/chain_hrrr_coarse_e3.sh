#!/bin/zsh
cd ${0:A:h}
PY=../../.venv/bin/python
until grep -qE "COARSE_W1_DONE|Traceback|FAILED" logs/chain_hrrr_coarse_w1.log; do sleep 20; done
$PY -u build_eih_hrrr_coarse.py --feat data/interim/open_gcrk/features_e3r2.npz --events-file experiments/open_gcrk_20260919/selected_events_e3.json --tag "" --workers 8 || exit 1
$PY check_eih_layout.py ../../data/interim/geo_weather/eih_hrrr_coarse.npz || exit 1
$PY audit_f0.py --prefix "" --out F0_hrrr_split --pairs hrrr-hrrr_coarse hrrr_coarse-pop hrrr-pop quad-pop
echo COARSE_E3_DONE
