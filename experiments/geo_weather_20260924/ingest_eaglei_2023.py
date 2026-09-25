"""EAGLE-I 2023 (figshare v4, file eaglei_outages_2023.csv) to the interim parquet layout of scripts/ingest_eaglei.py
(fips, ts, customers_out, state, county). The 2023 file names its outage count column `sum`."""
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "raw" / "eaglei" / "eaglei_outages_2023.csv"
DST = ROOT / "data" / "interim" / "eaglei_outages_2023.parquet"
ref = pq.read_schema(ROOT / "data" / "interim" / "eaglei_outages_2022.parquet")
tmp = DST.with_suffix(".parquet.tmp")
w, n = None, 0
for ch in pd.read_csv(SRC, dtype={"fips_code": str, "county": str, "state": str}, chunksize=2_000_000):
    ch = ch.rename(columns={"fips_code": "fips", "sum": "customers_out", "run_start_time": "ts"})
    ch["fips"] = ch["fips"].str.zfill(5).astype("string")
    ch["ts"] = pd.to_datetime(ch["ts"]).astype("datetime64[us]")
    ch["customers_out"] = pd.to_numeric(ch["customers_out"], errors="coerce").astype("Int64")
    ch["state"] = ch["state"].astype("string"); ch["county"] = ch["county"].astype("string")
    t = pa.Table.from_pandas(ch[["fips", "ts", "customers_out", "state", "county"]], preserve_index=False).cast(ref)
    w = w or pq.ParquetWriter(tmp, ref)
    w.write_table(t); n += len(ch)
w.close(); tmp.rename(DST)
print("rows", n, "->", DST.name)
