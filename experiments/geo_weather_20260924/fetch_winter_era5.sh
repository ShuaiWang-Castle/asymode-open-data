#!/bin/zsh
# ERA5 main fields and hourly-maximum gust for the winter windows, one CDS request at a time (the CDS limits
# queued requests per dataset); retries after a rejection. Skips files on disk.
cd ${0:A:h}/../open_gcrk_20260919
W=experiments/geo_weather_20260924/selected_events_winter.json
EVS=(2023-02-20 2022-02-02 2018-11-13 2019-02-05 2020-10-25 2022-12-13 2023-01-30 2023-12-24 2024-12-13)
for ev in $EVS; do
  for mode in main fg10; do
    for try in 1 2 3 4 5 6 7 8; do
      if [[ $mode == main ]]; then
        ../../.venv/bin/python -u fetch_era5_windows.py --windows-file $W --events $ev && break
      else
        ../../.venv/bin/python -u fetch_era5_windows.py --windows-file $W --variables 10m_wind_gust_since_previous_post_processing --out-dir data/raw/era5_fg10 --events $ev && break
      fi
      echo "retry $ev $mode after rejection ($try)"; sleep 180
    done
  done
done
echo ALL_DONE
