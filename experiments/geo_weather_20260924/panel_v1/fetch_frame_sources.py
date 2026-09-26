"""Public sources of the DATASET_DESIGN v1 frame (section 3.1), each logged to data_provenance/downloads.jsonl with its
URL, size, SHA-256 and UTC time.

  --eaglei2022  EAGLE-I 2022 from figshare v4 (the local copy stops at 2022-11-12; section 6); md5 checked
  --vtec        IEM VTEC annual archives 2018-2025: the attribute CSV member of pickup/wwa/<year>_all.zip, read by
                HTTP range requests (the 1 GB shapefile member is not downloaded), stored gzipped; and the storm-based
                polygons <year>_tsmf_sbw.zip
  --hurdat      NHC HURDAT2, Atlantic and eastern North Pacific, the newest files listed on the NHC data page
  --ugc         IEM UGC geometry snapshots (api/1/nws/ugcs.geojson?valid=), first of each month 2018-07..2025-12;
                each UGC geometry is stored once per distinct version (WKB, keyed by its hash) with a snapshot index
  --coast       Natural Earth 1:10m coastline (public domain), for the distance-to-coast audit axis (section 4.6)
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import io
import json
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
RAW = ROOT / "data" / "raw"
PROV = EXP / "data_provenance" / "downloads.jsonl"
S = requests.Session()
S.mount("https://", requests.adapters.HTTPAdapter(pool_connections=8, pool_maxsize=8, max_retries=3))


def log(path: Path, source: str, **extra) -> None:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    rec = dict(file=str(path.relative_to(ROOT)), source=source, bytes=path.stat().st_size, sha256=h.hexdigest(),
               downloaded_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **extra)
    with open(PROV, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print("saved", rec["file"], rec["bytes"], flush=True)


def stream(url: str, dst: Path, md5: str | None = None) -> None:
    tmp = dst.with_suffix(dst.suffix + ".part")
    h = hashlib.md5()
    with S.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for b in r.iter_content(1 << 22):
                f.write(b); h.update(b)
    if md5 and h.hexdigest() != md5:
        raise RuntimeError(f"md5 mismatch for {url}: {h.hexdigest()} != {md5}")
    tmp.rename(dst)


class HttpFile(io.RawIOBase):
    """A read-only, seekable view of a remote file through HTTP range requests (zip central directory + one member)."""

    def __init__(self, url: str):
        self.url, self.pos = url, 0
        self.size = int(S.head(url, timeout=60).headers["Content-Length"])

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        r = S.get(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"}, timeout=300)
        r.raise_for_status()
        b[:len(r.content)] = r.content
        self.pos += len(r.content)
        return len(r.content)


def eaglei2022() -> None:
    dst = RAW / "eaglei" / "eaglei_outages_2022.csv"
    url = "https://ndownloader.figshare.com/files/42547897"
    stream(url, dst, md5="0cd04f23ac90ec4ec20ce3a63dcfa25a")
    log(dst, f"figshare 10.6084/m9.figshare.24237376 v4, file 42547897 ({url})", licence="CC BY 4.0",
        md5="0cd04f23ac90ec4ec20ce3a63dcfa25a")


def vtec(years) -> None:
    out = RAW / "nws" / "vtec"
    out.mkdir(parents=True, exist_ok=True)
    for y in years:
        url = f"https://mesonet.agron.iastate.edu/pickup/wwa/{y}_all.zip"
        z = zipfile.ZipFile(io.BufferedReader(HttpFile(url), 1 << 22))
        name = [i.filename for i in z.infolist() if i.filename.endswith(".csv")][0]
        dst = out / f"{Path(name).stem}.csv.gz"
        with z.open(name) as src, gzip.open(dst, "wb") as g:
            for b in iter(lambda: src.read(1 << 22), b""):
                g.write(b)
        log(dst, f"IEM VTEC archive {url}, member {name} (range read; stored gzipped)")
        url = f"https://mesonet.agron.iastate.edu/pickup/wwa/{y}_tsmf_sbw.zip"
        dst = out / f"{y}_tsmf_sbw.zip"
        stream(url, dst)
        log(dst, f"IEM VTEC storm-based polygons {url}")


def hurdat() -> None:
    out = RAW / "nws" / "hurdat2"
    out.mkdir(parents=True, exist_ok=True)
    page = S.get("https://www.nhc.noaa.gov/data/hurdat/", timeout=60).text
    def key(name):                      # (season, release date); release dates are mmddyy or mmddyyyy
        season, d = name[:-4].split("-")[-2:]
        return int(season), dt.datetime.strptime(d, "%m%d%Y" if len(d) == 8 else "%m%d%y")

    for basin in ("hurdat2-1851", "hurdat2-nepac-1949"):
        names = sorted(set(re.findall(basin + r"-\d{4}-\d+\.txt", page)), key=key)
        url = "https://www.nhc.noaa.gov/data/hurdat/" + names[-1]
        dst = out / names[-1]
        stream(url, dst)
        log(dst, url)


def ugc(start="2018-07-01", end="2025-12-01") -> None:
    out = RAW / "nws" / "ugc"
    out.mkdir(parents=True, exist_ok=True)
    from shapely.geometry import shape
    dates = pd.date_range(start, end, freq="MS")

    def one(d):
        url = f"https://mesonet.agron.iastate.edu/api/1/nws/ugcs.geojson?valid={d:%Y-%m-%dT00:00Z}"
        r = S.get(url, timeout=300)
        r.raise_for_status()
        return d, url, r.content

    versions, index = {}, []
    with ThreadPoolExecutor(3) as ex:
        for d, url, content in ex.map(one, dates):
            fc = json.loads(content)
            for ft in fc["features"]:
                p = ft["properties"]
                if ft.get("geometry") is None:
                    continue
                wkb = shape(ft["geometry"]).wkb
                k = hashlib.sha1(wkb).hexdigest()
                versions.setdefault(k, dict(ugc=p["ugc"], state=p.get("state"), wfo=p.get("wfo"), vhash=k, wkb=wkb))
                index.append(dict(snapshot=d, ugc=p["ugc"], vhash=k))
            with open(PROV, "a") as f:
                f.write(json.dumps(dict(source=url, bytes=len(content), sha256=hashlib.sha256(content).hexdigest(),
                                        downloaded_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                                        note="UGC snapshot, reduced to data/raw/nws/ugc/ugc_versions.parquet")) + "\n")
            print("ugc snapshot", d.date(), len(fc["features"]), "versions so far", len(versions), flush=True)
    pd.DataFrame(list(versions.values())).to_parquet(out / "ugc_versions.parquet", index=False)
    pd.DataFrame(index).to_parquet(out / "ugc_snapshot_index.parquet", index=False)
    log(out / "ugc_versions.parquet", "IEM api/1/nws/ugcs.geojson monthly snapshots 2018-07..2025-12 (distinct geometries)")
    log(out / "ugc_snapshot_index.parquet", "IEM api/1/nws/ugcs.geojson monthly snapshots (ugc -> geometry version)")


def coast() -> None:
    out = RAW / "geography" / "coast"
    out.mkdir(parents=True, exist_ok=True)
    url = "https://naciscdn.org/naturalearth/10m/physical/ne_10m_coastline.zip"
    dst = out / "ne_10m_coastline.zip"
    stream(url, dst)
    log(dst, url, licence="public domain (Natural Earth)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k in ("eaglei2022", "vtec", "hurdat", "ugc", "coast"):
        ap.add_argument("--" + k, action="store_true")
    ap.add_argument("--years", default="2018-2025")
    a = ap.parse_args()
    y0, y1 = map(int, a.years.split("-"))
    if a.hurdat: hurdat()
    if a.coast: coast()
    if a.vtec: vtec(range(y0, y1 + 1))
    if a.ugc: ugc()
    if a.eaglei2022: eaglei2022()
