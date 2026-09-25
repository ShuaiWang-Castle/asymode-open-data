#!/bin/zsh
# Disjoint ice-storm panel W2d (PREREG_W2.md): wait for sources, build, features, base out-of-fold (5 folds).
cd ${0:A:h}
PY=../../.venv/bin/python
until grep -q SOURCES_DONE logs/fetch_w2_sources.log; do sleep 30; done
[ -f ../../data/interim/eaglei_outages_2025.parquet ] || $PY ingest_eaglei_year.py 2025
until ! pgrep -f "ingest_eaglei.py build" > /dev/null; do sleep 30; done
until ! pgrep -f "fetch_arco_windows.py --windows-file experiments/geo_weather_20260924/selected_events_w2d.json" > /dev/null; do sleep 30; done
$PY -u build_winter.py --events-file selected_events_w2d.json --tag w2d || exit 1
$PY build_train_mask.py --feat data/interim/geo_weather/features_w2d.npz --tag w2d
$PY -u build_eih.py --feat data/interim/geo_weather/features_w2d.npz --events-file experiments/geo_weather_20260924/selected_events_w2d_build.json --tag w2d_ --variants quad pop area mean
$PY audit_f0.py --feat data/interim/geo_weather/features_w2d.npz --prefix w2d_ --out F0_w2d
printf -- "--label w2d_base --data w2d --arm W+Cin --folds %s\n" 1 2 3 4 5 > jobs_w2d_base.txt
./run_queue.sh jobs_w2d_base.txt 2
echo CHAIN_W2D_DONE
