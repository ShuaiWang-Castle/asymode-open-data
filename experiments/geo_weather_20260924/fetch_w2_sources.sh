#!/bin/zsh
# Public sources for the disjoint ice-storm panel W2 (PREREG_W2.md amendment 1):
#  NOAA NCEI Storm Events details 2014-2017 (into a separate folder; the canonical 2018+ files stay as they are)
#  EAGLE-I 2025 (figshare 10.6084/m9.figshare.24237376 v4, CC BY 4.0, file 62164877)
set -e
ROOT=${0:A:h:h:h}
PROV=$ROOT/experiments/geo_weather_20260924/data_provenance/downloads.jsonl
SE=$ROOT/data/raw/storm_events_2014_2017; mkdir -p $SE
for f in StormEvents_details-ftp_v1.0_d2014_c20260323.csv.gz StormEvents_details-ftp_v1.0_d2015_c20260323.csv.gz StormEvents_details-ftp_v1.0_d2016_c20260323.csv.gz StormEvents_details-ftp_v1.0_d2017_c20260519.csv.gz; do
  [[ -f $SE/$f ]] && continue
  curl -sSL --retry 3 https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/$f -o $SE/$f.part && mv $SE/$f.part $SE/$f
  echo "{\"file\": \"data/raw/storm_events_2014_2017/$f\", \"source\": \"https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/$f\", \"bytes\": $(stat -f %z $SE/$f), \"sha256\": \"$(shasum -a 256 $SE/$f | cut -d' ' -f1)\", \"downloaded_utc\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" >> $PROV
  echo saved $f
done
DST=$ROOT/data/raw/eaglei/eaglei_outages_2025.csv; MD5=cd2feb1282a42fb048cb6885398bc1cc
if [[ ! -f $DST ]]; then
  curl -sSL --retry 3 https://ndownloader.figshare.com/files/62164877 -o $DST.part
  [[ $(md5 -q $DST.part) == $MD5 ]] || { echo "md5 mismatch"; exit 1; }
  mv $DST.part $DST
  echo "{\"file\": \"data/raw/eaglei/eaglei_outages_2025.csv\", \"source\": \"figshare 10.6084/m9.figshare.24237376 v4, file 62164877\", \"licence\": \"CC BY 4.0\", \"md5\": \"$MD5\", \"bytes\": $(stat -f %z $DST), \"downloaded_utc\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" >> $PROV
  echo saved eaglei 2025
fi
echo SOURCES_DONE
