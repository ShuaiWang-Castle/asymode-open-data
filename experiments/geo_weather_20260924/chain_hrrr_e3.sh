#!/bin/zsh
# HRRR hazard features for the twelve-event wind panel, after the W1 HRRR build; then the audit and two arms.
cd ${0:A:h}
PY=../../.venv/bin/python
until grep -qE "saved|Traceback" logs/queue_w1_hrrr.log; do sleep 20; done
$PY -u build_eih_hrrr.py --feat data/interim/open_gcrk/features_e3r2.npz --events-file experiments/open_gcrk_20260919/selected_events_e3.json --tag "" --nodes nodes_hrrr_e3.parquet --workers 16 || exit 1
$PY check_eih_layout.py ../../data/interim/geo_weather/eih_hrrr.npz || exit 1
$PY audit_f0.py --prefix "" --out F0_hrrr --pairs hrrr-pop hrrr-quad quad-pop
until grep -qE "CHAIN_W2E_DONE|fold . missing|Traceback" logs/chain_w2e.log; do sleep 30; done   # the registered W2e base first
./run_queue.sh jobs_e3_hrrr.txt 2
echo HRRR_E3_DONE
