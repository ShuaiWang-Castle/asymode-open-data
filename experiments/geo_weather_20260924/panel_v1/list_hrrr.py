"""Which hours of 2018-07-01..2025-12-31 have an HRRR wrfsfcf01 file on AWS (noaa-hrrr-bdp-pds), for system gate S-c
of DATASET_DESIGN v1 (section 3.7). One S3 listing per day (ListObjectsV2, paged). Output:
data/interim/panel_v1/hrrr_hours.parquet (valid hour = cycle + 1 h, present) and a provenance line."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "data" / "interim" / "panel_v1"
BUCKET = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/"
NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
PAT = re.compile(r"hrrr\.t(\d{2})z\.wrfsfcf01\.grib2$")
S = requests.Session()
S.mount("https://", requests.adapters.HTTPAdapter(pool_connections=16, pool_maxsize=16, max_retries=5))


def day(d: pd.Timestamp) -> list[int]:
    cycles, token = set(), None
    while True:
        q = {"list-type": "2", "prefix": f"hrrr.{d:%Y%m%d}/conus/hrrr.t", "max-keys": "1000"}
        if token:
            q["continuation-token"] = token
        r = S.get(BUCKET, params=q, timeout=120)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        for k in root.findall("s:Contents/s:Key", NS):
            m = PAT.search(k.text)
            if m:
                cycles.add(int(m.group(1)))
        if root.findtext("s:IsTruncated", namespaces=NS) != "true":
            return sorted(cycles)
        token = root.findtext("s:NextContinuationToken", namespaces=NS)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    days = pd.date_range("2018-07-01", "2025-12-31", freq="D")
    rows = []
    with ThreadPoolExecutor(16) as ex:
        for i, (d, cyc) in enumerate(zip(days, ex.map(day, days))):
            rows += [dict(valid=d + pd.Timedelta(hours=c + 1), present=True) for c in cyc]
            if i % 200 == 0:
                print(d.date(), len(cyc), flush=True)
    have = pd.DataFrame(rows).set_index("valid")
    full = pd.DataFrame(index=pd.date_range("2018-07-01 01:00", "2026-01-01 00:00", freq="h"))
    full["present"] = have["present"].reindex(full.index).fillna(False).astype(bool)
    full.index.name = "valid"
    f = OUT / "hrrr_hours.parquet"
    full.reset_index().to_parquet(f, index=False)
    rec = dict(file=str(f.relative_to(ROOT)), source=BUCKET + " (ListObjectsV2 per day, key hrrr.tHHz.wrfsfcf01.grib2)",
               bytes=f.stat().st_size, sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
               downloaded_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
               hours=int(len(full)), present_share=float(full["present"].mean()))
    with open(HERE.parent / "data_provenance" / "downloads.jsonl", "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(rec, flush=True)


if __name__ == "__main__":
    main()
