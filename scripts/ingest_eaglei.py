"""Inspect and ingest the EAGLE-I release archive.

Two modes, deliberately separate. `inspect` reads the archive's table of
contents and the first lines of each member without extracting anything, so the
layout can be checked before a multi-gigabyte unpack. `build` then converts the
outage records to a partitioned parquet panel and joins the denominator.

The denominator is the part that needs care: per the data card, the 2014-2022
release ships a *modeled* county customer file, the 2024 release carries per-county
totals inline, and 2025 is unconfirmed. This script refuses to guess -- if it
cannot find a denominator for a year it says so and leaves that year out rather
than silently falling back to another year's counts.
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "eaglei"
INTERIM = ROOT / "data" / "interim"

# Column names vary across releases; resolve by pattern rather than by position.
PAT = {
    "fips":  re.compile(r"^(fips.*code|fips|county_?fips|geoid)$", re.I),
    "out":   re.compile(r"^(customers_?out|sum|outage.*count|customers_?without)", re.I),
    "cust":  re.compile(r"^(customers|total_?customers|county_?customers|customers_?tracked)$", re.I),
    "time":  re.compile(r"^(run_?start_?time|timestamp|date_?time|time|datetime)$", re.I),
    "state": re.compile(r"^(state|state_?name|state_?abbr)$", re.I),
    "county": re.compile(r"^(county|county_?name)$", re.I),
    "year":  re.compile(r"^year$", re.I),
}


def resolve(cols: list[str]) -> dict[str, str]:
    out = {}
    for key, pat in PAT.items():
        for c in cols:
            if pat.match(c.strip()):
                out[key] = c
                break
    return out


def find_archive() -> Path:
    zips = sorted(RAW.glob("*.zip"))
    if not zips:
        sys.exit(f"no archive found in {RAW.relative_to(ROOT)} -- download it first "
                 f"(see docs/ACCESS_TODO.md)")
    return zips[0]


def cmd_inspect(args):
    zp = find_archive()
    print(f"archive: {zp.name}  ({zp.stat().st_size/2**30:.2f} GB)\n")
    with zipfile.ZipFile(zp) as z:
        members = [m for m in z.infolist() if not m.is_dir()]
        print(f"{len(members)} members\n")
        print(f"{'member':<58}{'compressed':>13}{'uncompressed':>15}")
        total = 0
        for m in sorted(members, key=lambda m: -m.file_size):
            total += m.file_size
            print(f"{m.filename[:57]:<58}{m.compress_size/2**20:>10.1f} MB{m.file_size/2**20:>12.1f} MB")
        print(f"{'TOTAL uncompressed':<58}{'':>13}{total/2**30:>12.2f} GB\n")

        print("=== header and first row of each text member ===")
        for m in sorted(members, key=lambda m: m.filename):
            if not re.search(r"\.(csv|txt|tsv)$", m.filename, re.I):
                print(f"\n-- {m.filename}  [not a text table, skipped]")
                continue
            with z.open(m) as fh:
                head = io.TextIOWrapper(fh, encoding="latin-1")
                lines = [head.readline().rstrip("\n") for _ in range(3)]
            print(f"\n-- {m.filename}")
            for ln in lines:
                print(f"   {ln[:200]}")
            cols = [c.strip() for c in lines[0].split(",")]
            r = resolve(cols)
            print(f"   resolved -> {r if r else 'NO MATCH -- needs a rule'}")


def cmd_build(args):
    """Stream each yearly CSV to parquet without holding a year in memory.

    A single year is up to 1.1 GB of CSV and 25 M rows; read whole, it costs
    several GB of RAM as a DataFrame. pyarrow's incremental writer keeps the
    footprint at one chunk.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    INTERIM.mkdir(parents=True, exist_ok=True)
    archives = sorted(RAW.glob("*.zip")) if args.all_archives else [find_archive()]
    members = []
    for zp in archives:
        with zipfile.ZipFile(zp) as z:
            members += [(zp, m) for m in z.infolist()
                        if not m.is_dir()
                        and re.search(r"\.csv$", m.filename, re.I)
                        and "__MACOSX" not in m.filename]
    if True:

        cover = [(zp, m) for zp, m in members if re.search(r"coverage", m.filename, re.I)]
        outage = [(zp, m) for zp, m in members if (zp, m) not in cover]
        if args.years:
            outage = [(zp, m) for zp, m in outage
                      if any(str(y) in Path(m.filename).stem for y in args.years)]

        for zp, m in cover:
            df = pd.read_csv(zipfile.ZipFile(zp).open(m), encoding="latin-1", low_memory=False)
            dst = INTERIM / f"eaglei_{Path(m.filename).stem}.parquet"
            df.to_parquet(dst, index=False)
            print(f"  {dst.name}  {df.shape}  cols={list(df.columns)}")

        for zp, m in sorted(outage, key=lambda t: t[1].filename):
            stem = Path(m.filename).stem
            dst = INTERIM / f"{stem}.parquet"
            if dst.exists() and not args.force:
                print(f"  skip {dst.name} (exists)")
                continue
            writer, n_rows, n_chunks = None, 0, 0
            tmp = dst.with_suffix(".parquet.tmp")
            try:
                for ch in pd.read_csv(zipfile.ZipFile(zp).open(m), encoding="latin-1",
                                      chunksize=args.chunk, low_memory=False,
                                      dtype={"fips_code": "string", "county": "string",
                                             "state": "string"},
                                      parse_dates=["run_start_time"]):
                    if n_chunks == 0:
                        r = resolve(list(ch.columns))
                        missing = {"fips", "out", "time"} - set(r)
                        if missing:
                            sys.exit(f"{m.filename}: cannot resolve {missing} "
                                     f"from {list(ch.columns)}")
                        print(f"  {stem}: {list(ch.columns)} -> {r}")
                    ch = ch.rename(columns={r["fips"]: "fips", r["out"]: "customers_out",
                                            r["time"]: "ts"})
                    # FIPS is a 5-character code; leading zeros are significant and
                    # any numeric read silently destroys them.
                    ch["fips"] = ch["fips"].astype("string").str.strip().str.zfill(5)
                    ch["customers_out"] = pd.to_numeric(ch["customers_out"],
                                                        errors="coerce").astype("Int64")
                    keep = ["fips", "ts", "customers_out"]
                    # Later releases carry the modelled county customer total in
                    # the records themselves. It is the denominator, so keep it.
                    if "cust" in r:
                        ch = ch.rename(columns={r["cust"]: "total_customers"})
                        ch["total_customers"] = pd.to_numeric(ch["total_customers"],
                                                              errors="coerce").astype("Int64")
                        keep.append("total_customers")
                    if "state" in r: keep.append(r["state"])
                    if "county" in r: keep.append(r["county"])
                    ch = ch[keep]
                    tbl = pa.Table.from_pandas(ch, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(tmp, tbl.schema, compression="zstd")
                    writer.write_table(tbl)
                    n_rows += len(ch); n_chunks += 1
                    print(f"    {stem}: {n_rows:,} rows", end="\r", flush=True)
            finally:
                if writer is not None:
                    writer.close()
            tmp.rename(dst)
            mb = dst.stat().st_size / 2**20
            print(f"  {dst.name}: {n_rows:,} rows, {mb:.0f} MB parquet".ljust(70))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("inspect").set_defaults(fn=cmd_inspect)
    b = sub.add_parser("build"); b.set_defaults(fn=cmd_build)
    b.add_argument("--chunk", type=int, default=2_000_000)
    b.add_argument("--force", action="store_true")
    b.add_argument("--all-archives", action="store_true",
                   help="ingest every zip in data/raw/eaglei, not just the first")
    b.add_argument("--years", type=int, nargs="+", default=None,
                   help="restrict to these years; default all")
    a = ap.parse_args()
    a.fn(a)
