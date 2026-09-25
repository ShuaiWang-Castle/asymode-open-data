#!/bin/zsh
# Enlarged confirmatory panel W2e (PREREG_W2.md amendment 3): build, features, base out-of-fold (5 folds).
# The registered test is run afterwards, once, with the frozen code (239fd7d).
cd ${0:A:h}
PY=../../.venv/bin/python
until ! pgrep -f "fetch_arco_windows.py --windows-file experiments/geo_weather_20260924/selected_events_w2e.json" > /dev/null; do sleep 30; done
$PY -u build_winter.py --events-file selected_events_w2e.json --tag w2e || exit 1
$PY -u build_eih.py --feat data/interim/geo_weather/features_w2e.npz --events-file experiments/geo_weather_20260924/selected_events_w2e_build.json --tag w2e_ --variants quad pop || exit 1
printf -- "--label w2e_base --data w2e --arm W+Cin --folds %s\n" 1 2 3 4 5 > jobs_w2e_base.txt
./run_queue.sh jobs_w2e_base.txt 3
for k in 1 2 3 4 5; do [ -f ../../runs/geo_weather_20260924/w2e_base/fold0$k/DONE.json ] || { echo "fold $k missing"; exit 1; }; done
echo CHAIN_W2E_DONE
