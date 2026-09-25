"""EAGLE-I yearly CSV (figshare v4, data/raw/eaglei/eaglei_outages_<year>.csv) to the interim parquet layout of
scripts/ingest_eaglei.py (fips, ts, customers_out, state, county). The 2023 file names its outage count column `sum`;
2025 names it `customers_out`.  usage: python ingest_eaglei_year.py <year>"""
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
import sys
YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 2023
SRC = ROOT / "data" / "raw" / "eaglei" / f"eaglei_outages_{YEAR}.csv"
DST = ROOT / "data" / "interim" / f"eaglei_outages_{YEAR}.parquet"
ref = pq.read_schema(ROOT / "data" / "interim" / "eaglei_outages_2022.parquet")
tmp = DST.with_suffix(".parquet.tmp")
w, n = None, 0
for ch in pd.read_csv(SRC, dtype={"fips_code": str, "county": str, "state": str}, chunksize=2_000_000):
    ch = ch.rename(columns={"fips_code": "fips", "sum": "customers_out", "run_start_time": "ts"})
    ch = ch[[c for c in ("fips", "ts", "customers_out", "state", "county") if c in ch.columns]]
    ch["fips"] = ch["fips"].str.zfill(5).astype("string")
    ch["ts"] = pd.to_datetime(ch["ts"]).astype("datetime64[us]")
    ch["customers_out"] = pd.to_numeric(ch["customers_out"], errors="coerce").astype("Int64")
    ch["state"] = ch["state"].astype("string"); ch["county"] = ch["county"].astype("string")
    t = pa.Table.from_pandas(ch[["fips", "ts", "customers_out", "state", "county"]], preserve_index=False).cast(ref)
    w = w or pq.ParquetWriter(tmp, ref)
    w.write_table(t); n += len(ch)
w.close(); tmp.rename(DST)
print("rows", n, "->", DST.name)
