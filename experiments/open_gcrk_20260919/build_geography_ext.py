"""Further public geographic descriptors: an amendment to PREREG 3 made after the main results.

The 31 pre-registered descriptors (`build_geography.py`, `geography.parquet`) are left as they
are. This script adds nine more, all rebuilt from public origin, so that soil x land-cover
co-location, a national windthrow-hazard rating, a five-point elevation sample and an
inventory-based forest share are also available to the response kernel:

  soils x land cover, on one sample lattice per county (the 80 m overview of the 10 m gNATSGO
  map-unit raster, July 2020; each sample takes the Annual NLCD 2021 class of the 60 m pixel
  that contains it; samples inside the county polygon, not open water, with a mapped soil):
    soil_wet_share            dominant-condition drainage class (muaggatt.drclassdcd) somewhat
                              poorly, poorly or very poorly drained
    soil_windthrow_hazard     NRCS interpretation 'FOR - Windthrow Hazard' (root rule): component-
                              percent-weighted share rated moderate or severe, divided by the share
                              rated at all (map units without a rating are left out, not zeroed)
    forest_wet_coloc          share of samples that are forest (NLCD 41, 42, 43, 90) and wet
    wet_in_forest             share of forest samples that are wet
    hazard_in_forest          windthrow-hazard share over rated forest samples
  land cover only (60 m land pixels):
    forest_near_developed     share of land pixels that are forest within 150 m of developed
                              land (NLCD 21-24)
  terrain (USGS 3DEP, bilinear on the 150 m grid of the pre-registered terrain descriptors) at the
  Census Gazetteer internal point and 0.06 degrees north, south, east and west of it:
    elev_mean5, relief5       mean and max - min of the five elevations (m)
  forest inventory (USDA Forest Service FIA EVALIDator, the most recent evaluation of each
  state, 'Area of forest land, in acres' by county):
    fia_forest_land_share     forest-land acres / Gazetteer land acres, capped at 1; counties
                              absent from a state's county list have no forest-land estimate
                              and get 0

Every download is cached under data/raw/geography/ and logged (URL, parameters, UTC time,
bytes, SHA-256) in data_provenance/geography_ext_log.jsonl. Output:
data/interim/open_gcrk/geography_ext.parquet (the nine descriptors plus sample counts).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_geography as BG  # noqa: E402  (county polygons, Gazetteer path, NLCD service)

ROOT = BG.ROOT
RAW = BG.RAW
OUT = BG.OUT / "geography_ext.parquet"
LOG = HERE / "data_provenance" / "geography_ext_log.jsonl"
RES_LC = 60.0                     # NLCD export resolution (m)
OV_FACTOR, OV_LEVEL = 8, 2        # gNATSGO overview used: 8x (80 m); rasterio overview_level index
TILE_ORIGIN_X, TILE_ORIGIN_Y = -2356155.0, 2399905.0   # any gNATSGO tile corner; all tiles share its grid
STEP = 10.0 * OV_FACTOR
MAX_PX = 4096
NEAR_M = 150.0
FOREST = (41, 42, 43, 90)
DEVELOPED = (21, 22, 23, 24)
WET = ("Somewhat poorly drained", "Poorly drained", "Very poorly drained")
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SIGN = "https://planetarycomputer.microsoft.com/api/sas/v1/sign"
FIA = "https://apps.fs.usda.gov/fiadb-api/fullreport"
FIA_WC = "https://apps.fs.usda.gov/fiadb-api/fullreport/parameters/wc"
NODATA_MUKEY = 2147483647
_lock = threading.Lock()


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def log(rec: dict):
    rec.setdefault("utc", dt.datetime.now(dt.timezone.utc).isoformat())
    with _lock:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG, "a") as f:
            f.write(json.dumps(rec) + "\n")


def get(session, url, params=None, timeout=180, tries=6, post_json=None):
    import requests
    for attempt in range(tries):
        try:
            r = (session.post(url, json=post_json, timeout=timeout) if post_json is not None
                 else session.get(url, params=params, timeout=timeout))
            if r.status_code == 200:
                return r
        except requests.RequestException:
            pass
        time.sleep(4 * (attempt + 1))
    raise RuntimeError(f"request failed: {url}")


# ---------------------------------------------------------------- NLCD 2021 at 60 m
def lc_grid(geom):
    x0, y0, x1, y1 = geom.bounds
    x0, y0 = np.floor(x0 / RES_LC) * RES_LC, np.floor(y0 / RES_LC) * RES_LC
    return x0, y0, int(np.ceil((x1 - x0) / RES_LC)), int(np.ceil((y1 - y0) / RES_LC))


def fetch_lc(fips, geom, session) -> np.ndarray:
    """Annual NLCD 2021 class codes on the county's 60 m grid (row 0 = north), exported in chunks."""
    from PIL import Image
    x0, y0, W, H = lc_grid(geom)
    d = RAW / "nlcd60"
    d.mkdir(parents=True, exist_ok=True)
    out = np.zeros((H, W), np.uint8)
    s = BG.SOURCES["nlcd"]
    for r0 in range(0, H, MAX_PX):
        for c0 in range(0, W, MAX_PX):
            h, w = min(MAX_PX, H - r0), min(MAX_PX, W - c0)
            f = d / (f"{fips}.tif" if (H <= MAX_PX and W <= MAX_PX) else f"{fips}_r{r0}_c{c0}.tif")
            if not f.exists():
                bx0 = x0 + c0 * RES_LC
                by1 = y0 + (H - r0) * RES_LC
                p = dict(bbox=f"{bx0},{by1 - h * RES_LC},{bx0 + w * RES_LC},{by1}", bboxSR=5070, imageSR=5070,
                         size=f"{w},{h}", format="tiff", pixelType=s["pixelType"], noData=s["noData"],
                         interpolation=s["interpolation"], compression="LZW", f="image",
                         mosaicRule=json.dumps(s["mosaic"]))
                r = get(session, s["url"], p)
                if not r.headers.get("content-type", "").startswith("image/tiff"):
                    raise RuntimeError(f"nlcd60 {fips}: not a tiff")
                tmp = f.with_suffix(".part")
                tmp.write_bytes(r.content)
                tmp.rename(f)
                log(dict(file=str(f.relative_to(ROOT)), source=s["url"], params=p, bytes=len(r.content),
                         sha256=sha256(r.content)))
            a = np.array(Image.open(io.BytesIO(f.read_bytes())))
            if a.shape != (h, w):
                raise ValueError(f"{f}: shape {a.shape} != {(h, w)}")
            out[r0:r0 + h, c0:c0 + w] = a
    return out


# ---------------------------------------------------------------- gNATSGO map units
def gnatsgo_items(session) -> list[dict]:
    p = RAW / "gnatsgo_items.json"
    if p.exists():
        return json.loads(p.read_text())
    items, body = [], {"collections": ["gnatsgo-rasters"], "bbox": [-125, 24, -66, 50], "limit": 500}
    url = STAC
    while True:
        r = get(session, url, post_json=body, timeout=120)
        d = r.json()
        for it in d["features"]:
            if not it["id"].startswith("conus_"):
                continue
            items.append(dict(id=it["id"], bbox=it["properties"]["proj:bbox"], href=it["assets"]["mukey"]["href"]))
        nxt = [l for l in d.get("links", []) if l.get("rel") == "next"]
        if not nxt:
            break
        body = nxt[0].get("body", body)
        url = nxt[0].get("href", STAC)
    p.write_text(json.dumps(items))
    log(dict(file=str(p.relative_to(ROOT)), source=STAC, params={"collections": ["gnatsgo-rasters"]},
             n_items=len(items), sha256=sha256(p.read_bytes())))
    return items


class Signer:
    def __init__(self, session):
        self.s, self.q, self.t = session, None, 0.0

    def __call__(self, href: str) -> str:
        with _lock:
            if self.q is None or time.time() - self.t > 1800:
                signed = get(self.s, SIGN, {"href": href}, timeout=60).json()["href"]
                self.q, self.t = signed.split("?", 1)[1], time.time()
            return href + "?" + self.q


def lattice(geom):
    """County bounding box on the global 80 m lattice shared by every gNATSGO tile overview."""
    x0, y0, x1, y1 = geom.bounds
    i0 = int(np.floor((x0 - TILE_ORIGIN_X) / STEP)); i1 = int(np.ceil((x1 - TILE_ORIGIN_X) / STEP))
    j0 = int(np.floor((TILE_ORIGIN_Y - y1) / STEP)); j1 = int(np.ceil((TILE_ORIGIN_Y - y0) / STEP))
    return i0, j0, i1 - i0, j1 - j0          # column/row offsets from the origin, width, height


def fetch_mukey(fips, geom, items, signer) -> np.ndarray:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import Window
    d = RAW / "gnatsgo80"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{fips}.npz"
    i0, j0, W, H = lattice(geom)
    if f.exists():
        z = np.load(f)
        assert tuple(z["lattice"]) == (i0, j0, W, H)
        return z["mukey"]
    gx0, gy1 = TILE_ORIGIN_X + i0 * STEP, TILE_ORIGIN_Y - j0 * STEP
    gx1, gy0 = gx0 + W * STEP, gy1 - H * STEP
    out = np.full((H, W), NODATA_MUKEY, np.int64)
    used = []
    env = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
               GDAL_HTTP_MAX_RETRY="6", GDAL_HTTP_RETRY_DELAY="3", VSI_CACHE="TRUE")
    for it in items:
        tx0, ty0, tx1, ty1 = it["bbox"]
        ax0, ax1, ay0, ay1 = max(gx0, tx0), min(gx1, tx1), max(gy0, ty0), min(gy1, ty1)
        if ax0 >= ax1 or ay0 >= ay1:
            continue
        c0, r0 = int(round((ax0 - tx0) / STEP)), int(round((ty1 - ay1) / STEP))
        w, h = int(round((ax1 - ax0) / STEP)), int(round((ay1 - ay0) / STEP))
        for attempt in range(5):
            try:
                with rasterio.Env(**env):
                    href = signer(it["href"])
                    with rasterio.open(href, overview_level=OV_LEVEL) as src:
                        exact = abs(src.res[0] - STEP) < 1e-6 and abs(src.res[1] - STEP) < 1e-6
                        if exact:
                            a = src.read(1, window=Window(c0, r0, w, h))
                    if not exact:
                        # tiles whose width is not a multiple of 8 (CONUS edge): their 8x overview is not
                        # on the 80 m lattice, so sample the full-resolution raster onto it (nearest)
                        with rasterio.open(href) as src:
                            a = src.read(1, window=Window(c0 * OV_FACTOR, r0 * OV_FACTOR, w * OV_FACTOR, h * OV_FACTOR),
                                         out_shape=(h, w), resampling=Resampling.nearest, boundless=True,
                                         fill_value=NODATA_MUKEY)
                break
            except Exception:
                with _lock:
                    signer.q = None
                time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(f"gnatsgo {fips} {it['id']}: read failed")
        oc, orow = int(round((ax0 - gx0) / STEP)), int(round((gy1 - ay1) / STEP))
        out[orow:orow + h, oc:oc + w] = a
        used.append(dict(item=it["id"], window=[c0, r0, w, h], overview_exact=bool(exact),
                         sha256=sha256(np.ascontiguousarray(a).tobytes())))
    tmp = f.with_name(f.stem + ".part")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, mukey=out, lattice=np.array([i0, j0, W, H]))
    tmp.rename(f)
    log(dict(file=str(f.relative_to(ROOT)), source="gnatsgo-rasters mukey (Planetary Computer, soils/gnatsgo/july2020)",
             overview_factor=OV_FACTOR, reads=used, sha256=sha256(f.read_bytes())))
    return out


# ---------------------------------------------------------------- per-county sampling
def near_footprint() -> np.ndarray:
    """Pixel offsets whose centres lie within NEAR_M of the centre pixel on the 60 m grid."""
    r = int(NEAR_M // RES_LC)
    o = np.arange(-r, r + 1)
    return (RES_LC ** 2 * (o[:, None] ** 2 + o[None, :] ** 2)) <= NEAR_M ** 2


def county_samples(fips, geom, items, signer, session) -> dict:
    from rasterio.features import geometry_mask
    from rasterio.transform import Affine
    from scipy.ndimage import binary_dilation
    lc = fetch_lc(fips, geom, session)
    x0, y0, W, H = lc_grid(geom)
    inside = ~geometry_mask([geom], out_shape=(H, W), transform=Affine(RES_LC, 0, x0, 0, -RES_LC, y0 + H * RES_LC))
    land = inside & (lc > 0) & (lc != 11)
    forest = np.isin(lc, FOREST)
    dev = np.isin(lc, DEVELOPED)
    near = binary_dilation(dev, structure=near_footprint())
    out = dict(fips=fips, n_land_pixels_60m=int(land.sum()),
               forest_near_developed=float((forest & near)[land].mean()) if land.any() else np.nan)
    del near, dev, inside
    mk = fetch_mukey(fips, geom, items, signer)
    i0, j0, Wm, Hm = lattice(geom)
    gx0, gy1 = TILE_ORIGIN_X + i0 * STEP, TILE_ORIGIN_Y - j0 * STEP
    in80 = ~geometry_mask([geom], out_shape=(Hm, Wm), transform=Affine(STEP, 0, gx0, 0, -STEP, gy1))
    col = np.floor((gx0 + (np.arange(Wm) + 0.5) * STEP - x0) / RES_LC).astype(int)
    row = np.floor((y0 + H * RES_LC - (gy1 - (np.arange(Hm) + 0.5) * STEP)) / RES_LC).astype(int)
    cok, rok = (col >= 0) & (col < W), (row >= 0) & (row < H)
    samp = np.zeros((Hm, Wm), bool)
    fo80 = np.zeros((Hm, Wm), bool)
    samp[np.ix_(rok, cok)] = land[np.ix_(row[rok], col[cok])]
    fo80[np.ix_(rok, cok)] = forest[np.ix_(row[rok], col[cok])]
    samp &= in80
    keys, fo = mk[samp], fo80[samp]
    u, inv = np.unique(keys, return_inverse=True)
    out.update(n_samples=int(samp.sum()), _mukey=u, _n_all=np.bincount(inv, minlength=len(u)).astype(np.int64),
               _n_forest=np.bincount(inv, weights=fo, minlength=len(u)).astype(np.int64))
    return out


# ---------------------------------------------------------------- map-unit attributes (SDA)
def sda(session, sql: str) -> list[list]:
    r = get(session, BG.SDA, post_json={"query": sql, "format": "JSON+COLUMNNAME"}, timeout=300)
    d = r.json().get("Table", [])
    return d


def mapunit_table(session, mukeys: np.ndarray) -> pd.DataFrame:
    p = RAW / "gnatsgo_mapunits.parquet"
    if p.exists():
        t = pd.read_parquet(p)
        if set(mukeys.tolist()) <= set(t.index.tolist()):
            return t
    keys = sorted(set(int(k) for k in mukeys if k != NODATA_MUKEY))
    agg, comp = [], []

    def batch(b):
        import requests
        s = requests.Session()
        chunk = ",".join(str(k) for k in keys[b:b + 1000])
        q1 = f"SELECT mukey, drclassdcd, brockdepmin, wtdepannmin FROM muaggatt WHERE mukey IN ({chunk})"
        q2 = ("SELECT c.mukey, c.cokey, c.comppct_r, ci.interphrc FROM component c LEFT JOIN cointerp ci "
              "ON ci.cokey = c.cokey AND ci.mrulename = 'FOR - Windthrow Hazard' AND ci.ruledepth = 0 "
              f"WHERE c.mukey IN ({chunk})")
        res = []
        for q in (q1, q2):
            rows = sda(s, q)
            res.append(rows[1:] if rows else [])
            log(dict(source=BG.SDA, query=q[:160] + f" ... ({len(keys[b:b + 1000])} mukeys from {keys[b]})",
                     rows=max(len(rows) - 1, 0), sha256=sha256(json.dumps(rows).encode())))
        return res

    with ThreadPoolExecutor(3) as ex:
        for r1, r2 in ex.map(batch, range(0, len(keys), 1000)):
            agg.extend(r1); comp.extend(r2)
    print(f"map-unit attributes: {len(keys)} map units, {len(agg)} muaggatt rows, {len(comp)} component rows", flush=True)
    a = pd.DataFrame(agg, columns=["mukey", "drclassdcd", "brockdepmin", "wtdepannmin"])
    a["mukey"] = a.mukey.astype(int)
    a = a.drop_duplicates("mukey").set_index("mukey")
    c = pd.DataFrame(comp, columns=["mukey", "cokey", "comppct_r", "interphrc"])
    c["mukey"] = c.mukey.astype(int)
    c = c.drop_duplicates("cokey")
    c["w"] = pd.to_numeric(c.comppct_r, errors="coerce").clip(lower=0).fillna(0)
    den = c.groupby("mukey").w.sum().replace(0, np.nan)
    rated = (c.w * c.interphrc.isin(["Slight", "Moderate", "Severe"])).groupby(c.mukey).sum() / den
    modsev = (c.w * c.interphrc.isin(["Moderate", "Severe"])).groupby(c.mukey).sum() / den
    t = pd.DataFrame(index=pd.Index(keys, name="mukey"))
    t["wet"] = a.drclassdcd.reindex(t.index).isin(WET).astype(float)
    t.loc[a.drclassdcd.reindex(t.index).isna(), "wet"] = np.nan
    t["rated"] = rated.reindex(t.index).fillna(0.0)
    t["modsev"] = modsev.reindex(t.index).fillna(0.0)
    t.to_parquet(p)
    log(dict(file=str(p.relative_to(ROOT)), n_mukeys=len(t), sha256=sha256(p.read_bytes())))
    return t


def soil_descriptors(s: dict, mu: pd.DataFrame) -> dict:
    """County shares from per-map-unit sample counts (all samples, forest samples)."""
    u, na, nf = s.pop("_mukey"), s.pop("_n_all").astype(float), s.pop("_n_forest").astype(float)
    mapped = u != NODATA_MUKEY
    n_samples = na.sum()
    u, na, nf = u[mapped], na[mapped], nf[mapped]
    m = mu.reindex(u)
    wet, rated, ms = m.wet.to_numpy(), m.rated.fillna(0).to_numpy(), m.modsev.fillna(0).to_numpy()
    ok = np.isfinite(wet)
    w = np.where(ok, wet, 0.0)
    n_ok, nf_ok = na[ok].sum(), nf[ok].sum()
    rs, rsf = (na * rated).sum(), (nf * rated).sum()
    s.update(n_soil_samples=int(na.sum()), soil_mapped_share=float(na.sum() / n_samples) if n_samples else np.nan,
             soil_wet_share=float((na * w).sum() / n_ok) if n_ok else np.nan,
             soil_windthrow_hazard=float((na * ms).sum() / rs) if rs > 0 else np.nan,
             forest_wet_coloc=float((nf * w).sum() / n_ok) if n_ok else np.nan,
             wet_in_forest=float((nf * w).sum() / nf_ok) if nf_ok >= 20 else np.nan,
             hazard_in_forest=float((nf * ms).sum() / rsf) if rsf >= 5 else np.nan,
             windthrow_rated_share=float(rs / na.sum()) if na.sum() else np.nan)
    return s


# ---------------------------------------------------------------- five-point elevation
def _bilinear(a, fx, fy):
    i, j = int(np.floor(fx)), int(np.floor(fy))
    if i < 0 or j < 0 or i + 1 >= a.shape[1] or j + 1 >= a.shape[0]:
        return np.nan
    wx, wy = fx - i, fy - j
    q = a[j:j + 2, i:i + 2]
    return float((1 - wy) * ((1 - wx) * q[0, 0] + wx * q[0, 1]) + wy * ((1 - wx) * q[1, 0] + wx * q[1, 1]))


def elevation5(session, gaz: pd.DataFrame, fips: list[str], C) -> pd.DataFrame:
    """3DEP elevation at the Gazetteer internal point and 0.06 degrees N, S, E, W of it, bilinear on the
    same 150 m 3DEP grid as the pre-registered terrain descriptors (the county exports of
    build_geography.py; a point outside its county's export gets a 4 x 4 pixel export of its own)."""
    import pyproj
    tr = pyproj.Transformer.from_crs(4326, 5070, always_xy=True)
    d = RAW / "dem_points"
    rows = []
    for f in fips:
        la, lo = float(gaz.loc[f, "INTPTLAT"]), float(gaz.loc[f, "INTPTLONG"])
        pts = [(lo + dx, la + dy) for dx, dy in [(0, 0), (0, 0.06), (0, -0.06), (0.06, 0), (-0.06, 0)]]
        xs, ys = tr.transform([q[0] for q in pts], [q[1] for q in pts])
        x0, y0, W, H = BG.grid(C.geometry[f])
        dem = BG.read(RAW / "dem" / f"{f}.tif", H, W, -9999)
        v = []
        for k, (x, y) in enumerate(zip(xs, ys)):
            val = _bilinear(dem, (x - x0) / BG.RES - 0.5, (y0 + H * BG.RES - y) / BG.RES - 0.5)
            if not np.isfinite(val):
                bx0, by0 = np.floor(x / BG.RES) * BG.RES - BG.RES, np.floor(y / BG.RES) * BG.RES - BG.RES
                t = d / f"{f}_{k}.tif"
                if not t.exists():
                    d.mkdir(parents=True, exist_ok=True)
                    sdef = BG.SOURCES["dem"]
                    prm = dict(bbox=f"{bx0},{by0},{bx0 + 4 * BG.RES},{by0 + 4 * BG.RES}", bboxSR=5070, imageSR=5070,
                               size="4,4", format="tiff", pixelType=sdef["pixelType"], noData=sdef["noData"],
                               interpolation=sdef["interpolation"], compression="LZW", f="image")
                    r = get(session, sdef["url"], prm)
                    t.write_bytes(r.content)
                    log(dict(file=str(t.relative_to(ROOT)), source=sdef["url"], params=prm, bytes=len(r.content),
                             sha256=sha256(r.content)))
                small = BG.read(t, 4, 4, -9999)
                val = _bilinear(small, (x - bx0) / BG.RES - 0.5, (by0 + 4 * BG.RES - y) / BG.RES - 0.5)
            v.append(val)
        v = np.array(v, float)
        rows.append(dict(fips=f, elev_mean5=float(np.nanmean(v)), relief5=float(np.nanmax(v) - np.nanmin(v)),
                         n_elev_points=int(np.isfinite(v).sum())))
    return pd.DataFrame(rows).set_index("fips")


# ---------------------------------------------------------------- FIA forest land
def fia_forest(session, gaz: pd.DataFrame, fips: list[str]) -> pd.DataFrame:
    d = RAW / "fia"
    d.mkdir(parents=True, exist_ok=True)
    pw = d / "wc_parameters.html"
    if not pw.exists():
        r = get(session, FIA_WC)
        pw.write_bytes(r.content)
        log(dict(file=str(pw.relative_to(ROOT)), source=FIA_WC, bytes=len(r.content), sha256=sha256(r.content)))
    html = pw.read_text()
    # columns: STATE, STATECD, EVALID, EVAL_GRP, REPORT_YEAR_NM, GROWTH_ACCT, INVENTORY, MOST_RECENT
    wc = {}
    for chunk in html.split("<th scope=row")[1:]:
        cells = [c.strip() for c in re.findall(r">([^<]*)</td>", chunk)]
        if len(cells) >= 8 and cells[7] == "Y" and cells[1].isdigit() and cells[3].isdigit():
            wc[int(cells[1])] = max(wc.get(int(cells[1]), 0), int(cells[3]))
    acres = {}
    for sc in sorted({int(f[:2]) for f in fips}):
        if sc not in wc:
            continue
        f = d / f"forest_land_{wc[sc]}.json"
        prm = dict(wc=wc[sc], snum=2, rselected="County code and name", cselected="State code", estOnly="Y",
                   outputFormat="NJSON")
        if not f.exists():
            r = get(session, FIA, prm, timeout=300)
            f.write_bytes(r.content)
            log(dict(file=str(f.relative_to(ROOT)), source=FIA, params=prm, bytes=len(r.content), sha256=sha256(r.content)))
        for e in json.loads(f.read_text())["estimates"]:
            m = re.match(r"`?(\d{5})", str(e["GRP1"]))
            if m:
                acres[m.group(1)] = float(e["ESTIMATE"])
    out = []
    for f in fips:
        land_acres = float(gaz.loc[f, "ALAND"]) / 4046.8564224
        covered = int(f[:2]) in wc
        v = acres.get(f, 0.0 if covered else np.nan)
        out.append(dict(fips=f, fia_forest_land_share=min(v / land_acres, 1.0) if np.isfinite(v) else np.nan,
                        fia_state_evaluation=wc.get(int(f[:2]), -1)))
    return pd.DataFrame(out).set_index("fips")


# ---------------------------------------------------------------- main
def main():
    import requests
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="first N counties only (smoke test)")
    a = ap.parse_args()
    base = pd.read_parquet(BG.OUT / "geography.parquet")
    fips = [str(f) for f in (base.index if base.index.name == "fips" else base["fips"])]
    if a.limit:
        fips = fips[:a.limit]
    C = BG.counties().loc[fips]
    gaz = pd.read_csv(BG.GAZ, sep="\t", dtype={"GEOID": str})
    gaz.columns = [c.strip() for c in gaz.columns]
    gaz = gaz.set_index("GEOID")
    session = requests.Session()
    items = gnatsgo_items(session)
    signer = Signer(session)
    t0, rows = time.time(), []
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(county_samples, f, C.geometry[f], items, signer, requests.Session()): f for f in fips}
        for n, fu in enumerate(as_completed(futs), 1):
            rows.append(fu.result())
            if n % 50 == 0 or n == len(fips):
                print(f"{n}/{len(fips)} counties sampled, {time.time() - t0:.0f}s", flush=True)
    allk = np.unique(np.concatenate([r["_mukey"] for r in rows]))
    mu = mapunit_table(session, allk)
    soil = pd.DataFrame([soil_descriptors(r, mu) for r in rows]).set_index("fips")
    e5 = elevation5(session, gaz, fips, C)
    fia = fia_forest(session, gaz, fips)
    out = soil.join(e5).join(fia).loc[fips]
    out.index.name = "fips"
    if not a.limit:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(OUT)
        meta = dict(n_counties=len(out), sha256=sha256(OUT.read_bytes()),
                    nan_counts={c: int(out[c].isna().sum()) for c in out.columns},
                    built_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        (HERE / "data_provenance" / "geography_ext_meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(out.describe().T.round(3).to_string())


if __name__ == "__main__":
    main()
