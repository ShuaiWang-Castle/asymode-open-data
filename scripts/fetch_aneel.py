"""Download the ANEEL distribution-network interruption records.

Brazil's electricity regulator (ANEEL) publishes every interruption recorded on
the country's distribution networks, one row per interruption event, with the
**start and end timestamp of each event**. That is the property that makes this
dataset worth carrying alongside county-aggregated outage counts: the restoration
time is observed directly rather than inferred from a falling aggregate.

The archives are large (about 1.8 GB compressed, roughly 15 GB as CSV), so they
are not redistributed here. This script fetches them from the official portal and
verifies each file against `data/aneel/MANIFEST.json`.

    python scripts/fetch_aneel.py --years 2017 2018 --format parquet
    python scripts/fetch_aneel.py --all --format zip --verify-only

Parquet is the better choice for analysis and is what the portal recommends; the
zip archives contain a single semicolon-delimited, Latin-1 CSV each.

Source, licence and required attribution: see `data/aneel/README.md`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/aneel/MANIFEST.json"
CHUNK = 1 << 20


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, tmp.open("wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        got = 0
        while True:
            b = r.read(CHUNK)
            if not b:
                break
            f.write(b)
            got += len(b)
            if total:
                print(f"\r  {dest.name}: {got / 1e6:.0f}/{total / 1e6:.0f} MB", end="", flush=True)
    print()
    tmp.rename(dest)


def main() -> None:
    man = json.loads(MANIFEST.read_text())
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--format", choices=["zip", "parquet"], default="parquet")
    ap.add_argument("--out", default="data/raw/aneel")
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args()

    years = sorted(man["years"]) if (a.all or not a.years) else sorted(a.years)
    unknown = [y for y in years if y not in man["years"]]
    if unknown:
        raise SystemExit(f"no such year in the manifest: {unknown}; have {sorted(man['years'])}")
    out = Path(a.out).expanduser()
    out = out if out.is_absolute() else ROOT / out

    def show(q):
        try:
            return q.relative_to(ROOT)
        except ValueError:
            return q
    key = "zip_url" if a.format == "zip" else "parquet_url"
    bad = 0
    for y in years:
        rec = man["years"][y]
        url = rec.get(key)
        if not url:
            print(f"{y}: no {a.format} resource on the portal, skipping")
            continue
        dest = out / url.rsplit("/", 1)[-1]
        if not dest.exists():
            if a.verify_only:
                print(f"{y}: missing ({show(dest)})")
                bad += 1
                continue
            print(f"{y}: downloading {a.format}")
            download(url, dest)
        want = rec.get("sha256") if a.format == "zip" else None
        if want:
            got = sha256(dest)
            ok = got == want
            print(f"{y}: {'sha256 OK' if ok else 'sha256 MISMATCH'}  {show(dest)}")
            bad += (not ok)
        else:
            print(f"{y}: present, no recorded checksum for this format  {show(dest)}")
    if bad:
        print(f"\n{bad} file(s) missing or failed verification", file=sys.stderr)
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
