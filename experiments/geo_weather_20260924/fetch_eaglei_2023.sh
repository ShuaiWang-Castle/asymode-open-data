#!/bin/zsh
# EAGLE-I recorded outages 2023 (figshare 10.6084/m9.figshare.24237376 v4, CC BY 4.0), file id 44574907.
# The 2023 file names its outage count column `sum` (customers_out in the other years).
set -e
ROOT=${0:A:h:h:h}
DST=$ROOT/data/raw/eaglei/eaglei_outages_2023.csv
MD5=4ef3f9290884b4151a43333d28f67369
[[ -f $DST ]] && { echo exists; exit 0; }
curl -sSL --retry 3 https://ndownloader.figshare.com/files/44574907 -o $DST.part
[[ $(md5 -q $DST.part) == $MD5 ]] || { echo "md5 mismatch"; exit 1; }
mv $DST.part $DST
SHA=$(shasum -a 256 $DST | cut -d' ' -f1); BYTES=$(stat -f %z $DST)
echo "{\"file\": \"data/raw/eaglei/eaglei_outages_2023.csv\", \"source\": \"figshare 10.6084/m9.figshare.24237376 v4, file 44574907 (https://ndownloader.figshare.com/files/44574907)\", \"licence\": \"CC BY 4.0\", \"md5\": \"$MD5\", \"sha256\": \"$SHA\", \"bytes\": $BYTES, \"downloaded_utc\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" >> $ROOT/experiments/geo_weather_20260924/data_provenance/downloads.jsonl
echo saved $BYTES
