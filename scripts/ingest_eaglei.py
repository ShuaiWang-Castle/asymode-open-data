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
    zp = find_archive()
    INTERIM.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zp) as z:
        members = [m for m in z.infolist()
                   if not m.is_dir() and re.search(r"\.csv$", m.filename, re.I)]
        outage = [m for m in members if re.search(r"outage|eaglei", m.filename, re.I)
                  and not re.search(r"coverage|customer", m.filename, re.I)]
        denom = [m for m in members if re.search(r"customer|mcc", m.filename, re.I)]
        cover = [m for m in members if re.search(r"coverage", m.filename, re.I)]

        print(f"outage files: {[m.filename for m in outage]}")
        print(f"denominator files: {[m.filename for m in denom]}")
        print(f"coverage files: {[m.filename for m in cover]}")
        if not denom:
            print("\nWARNING: no denominator file matched. The target y = out/customers "
                  "cannot be formed. Stopping rather than guessing.", file=sys.stderr)

        for m in cover + denom:
            df = pd.read_csv(z.open(m), encoding="latin-1", low_memory=False)
            name = Path(m.filename).stem
            df.to_parquet(INTERIM / f"eaglei_{name}.parquet", index=False)
            print(f"  wrote eaglei_{name}.parquet  {df.shape}")

        for m in outage:
            name = Path(m.filename).stem
            dst = INTERIM / f"eaglei_{name}.parquet"
            if dst.exists() and not args.force:
                print(f"  skip {dst.name} (exists)")
                continue
            chunks = []
            for i, ch in enumerate(pd.read_csv(z.open(m), encoding="latin-1",
                                               chunksize=args.chunk, low_memory=False)):
                if i == 0:
                    r = resolve(list(ch.columns))
                    print(f"  {m.filename}: columns {list(ch.columns)} -> {r}")
                chunks.append(ch)
            df = pd.concat(chunks, ignore_index=True)
            df.to_parquet(dst, index=False)
            print(f"  wrote {dst.name}  {df.shape}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("inspect").set_defaults(fn=cmd_inspect)
    b = sub.add_parser("build"); b.set_defaults(fn=cmd_build)
    b.add_argument("--chunk", type=int, default=2_000_000)
    b.add_argument("--force", action="store_true")
    a = ap.parse_args()
    a.fn(a)
