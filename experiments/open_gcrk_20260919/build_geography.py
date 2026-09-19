"""Public geographic descriptors g_i for every panel county, rebuilt from origin.

Raster sources, exported per county on one common 150 m grid (NAD83 / CONUS
Albers, EPSG:5070; the county's bounding box snapped to 150 m):
  DEM      USGS 3DEP, ImageServer 3DEPElevation (bilinear)            elevation, m
  canopy   USFS NLCD Tree Canopy Cover v2025-6, year 2021 (nearest)   percent
  cover    USGS Annual NLCD Land Cover C1, year 2021 (nearest)        class code
Soil source (tabular): USDA NRCS Soil Data Access, SSURGO county overlap tables;
shares are component-percent x map-unit-overlap acreage.

Descriptors (land pixels = inside the county polygon and not open water):
  terrain   elev_mean, relief_p95_p5, slope_mean_deg, steep_frac (slope >= 10 deg),
            ruggedness_tri (mean |dz| to the 8 neighbours), aspect_<sector> for 8
            compass sectors (share of pixels with slope >= 2 deg facing each way;
            1/8 each when fewer than 200 such pixels exist)
  canopy    canopy_mean, canopy_dense_frac (>= 50%), canopy_in_developed (mean
            canopy on developed pixels), forest_frac, developed_frac,
            wetland_frac, forest_on_steep_frac
  soils     poorly_drained_share, hydric_share, shallow_share (restrictive layer
            < 50 cm), high_water_table_share (annual minimum < 30 cm),
            root_limiting_share (restrictive layer < 50 cm, or annual minimum
            water table < 30 cm, or poorly / very poorly drained: the soil
            conditions under which shallow rooting makes trees prone to windthrow;
            SSURGO carries windthrow-hazard interpretations for a few states
            only, so no national interpretation is used)
  composite windthrow_susceptibility = forest_frac x root_limiting_share
            (county shares multiplied: soil and forest are not co-located here,
            which the gridded soil services do not allow)
  area      log_land_area_km2 (Census Gazetteer 2023)
  smoothed  s50_<x> for relief, canopy_mean, windthrow_susceptibility and
            poorly_drained_share: land-area-weighted mean over counties whose
            centroid lies within 50 km (the county itself included)

Every downloaded file is cached under data/raw/geography/ and logged with its URL,
request parameters, UTC time, size and SHA-256 in data_provenance/geography_log.jsonl.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RAW = ROOT / "data" / "raw" / "geography"
OUT = ROOT / "data" / "interim" / "open_gcrk"
LOG = HERE / "data_provenance" / "geography_log.jsonl"
SHP = ROOT / "data/raw/census/cb_county/cb_2023_us_county_500k.shp"
GAZ = ROOT / "data/raw/census/2023_Gaz_counties_national.txt"
RES = 150.0
SOURCES = {
    "dem": dict(url="https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage",
                pixelType="F32", noData="-9999", interpolation="RSP_BilinearInterpolation", mosaic=None),
    "tcc": dict(url="https://imagery.geoplatform.gov/iipp/rest/services/Vegetation/USFS_EDW_NLCD_TCC_CONUS/ImageServer/exportImage",
                pixelType="U8", noData="255", interpolation="RSP_NearestNeighbor",
                mosaic={"mosaicMethod": "esriMosaicLockRaster", "lockRasterIds": [76]}),   # 2021
    "nlcd": dict(url="https://imagery.geoplatform.gov/iipp/rest/services/Vegetation/USFS_EDW_NLCD_Landcover_CONUS/ImageServer/exportImage",
                 pixelType="U8", noData="0", interpolation="RSP_NearestNeighbor",
                 mosaic={"mosaicMethod": "esriMosaicLockRaster", "lockRasterIds": [37]}),  # 2021
}
SDA = "https://sdmdataaccess.sc.egov.usda.gov/Tabular/post.rest"
SECTORS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
SMOOTH = ["relief_p95_p5", "canopy_mean", "windthrow_susceptibility", "poorly_drained_share"]
STEEP_DEG, SLOPED_DEG, SMOOTH_KM, MIN_SLOPED_PIXELS = 10.0, 2.0, 50.0, 200


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def log(rec: dict):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")


def counties() -> "gpd.GeoDataFrame":
    import geopandas as gpd
    c = gpd.read_file(SHP)
    c["fips"] = c.STATEFP + c.COUNTYFP
    return c[["fips", "STUSPS", "geometry"]].to_crs(5070).set_index("fips")


def grid(geom):
    x0, y0, x1, y1 = geom.bounds
    x0, y0 = np.floor(x0 / RES) * RES, np.floor(y0 / RES) * RES
    W, H = int(np.ceil((x1 - x0) / RES)), int(np.ceil((y1 - y0) / RES))
    return x0, y0, W, H


def fetch(kind: str, fips: str, geom, session) -> Path:
    import requests
    dst = RAW / kind / f"{fips}.tif"
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    s = SOURCES[kind]
    x0, y0, W, H = grid(geom)
    p = dict(bbox=f"{x0},{y0},{x0 + W * RES},{y0 + H * RES}", bboxSR=5070, imageSR=5070, size=f"{W},{H}",
             format="tiff", pixelType=s["pixelType"], noData=s["noData"], interpolation=s["interpolation"],
             compression="LZW", f="image")
    if s["mosaic"]:
        p["mosaicRule"] = json.dumps(s["mosaic"])
    for attempt in range(5):
        try:
            r = session.get(s["url"], params=p, timeout=180)
            if r.status_code == 200 and r.headers.get("content-type", "").startswith("image/tiff") and len(r.content) > 200:
                break
        except requests.RequestException:
            pass
        time.sleep(5 * (attempt + 1))
    else:
        raise RuntimeError(f"{kind} {fips}: export failed")
    tmp = dst.with_suffix(".part")
    tmp.write_bytes(r.content)
    tmp.rename(dst)
    log(dict(file=str(dst.relative_to(ROOT)), source=s["url"], params=p,
             downloaded_utc=dt.datetime.now(dt.timezone.utc).isoformat(), bytes=len(r.content),
             sha256=sha256(r.content)))
    return dst


def read(path: Path, H: int, W: int, nodata) -> np.ndarray:
    from PIL import Image
    a = np.array(Image.open(io.BytesIO(path.read_bytes())))
    if a.shape != (H, W):
        raise ValueError(f"{path}: shape {a.shape} != {(H, W)}")
    a = a.astype(np.float64)
    a[a == float(nodata)] = np.nan
    return a


def raster_stats(fips: str, geom) -> dict:
    import shapely
    x0, y0, W, H = grid(geom)
    dem = read(RAW / "dem" / f"{fips}.tif", H, W, -9999)
    tcc = read(RAW / "tcc" / f"{fips}.tif", H, W, 255)
    lc = read(RAW / "nlcd" / f"{fips}.tif", H, W, 0)
    xs = x0 + (np.arange(W) + 0.5) * RES
    ys = y0 + H * RES - (np.arange(H) + 0.5) * RES              # row 0 is the north edge
    X, Y = np.meshgrid(xs, ys)
    inside = shapely.contains_xy(geom, X.ravel(), Y.ravel()).reshape(H, W)
    land = inside & np.isfinite(dem) & (lc != 11)
    tcc = np.where(tcc > 100, np.nan, tcc)
    dzdy_row, dzdx = np.gradient(dem, RES)                   # axis 0 runs south, so d/dy = -d/drow
    dzdy = -dzdy_row
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    aspect = (np.degrees(np.arctan2(-dzdx, -dzdy)) + 360.0) % 360.0     # bearing of the downslope direction
    pad = np.pad(dem, 1, mode="edge")
    tri = np.nanmean(np.stack([np.abs(pad[1 + di:H + 1 + di, 1 + dj:W + 1 + dj] - dem)
                               for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj]), 0)
    ok = land & np.isfinite(slope)
    z = dem[ok]
    out = dict(fips=fips, n_land_pixels=int(ok.sum()),
               elev_mean=float(np.mean(z)), relief_p95_p5=float(np.percentile(z, 95) - np.percentile(z, 5)),
               slope_mean_deg=float(np.mean(slope[ok])), steep_frac=float(np.mean(slope[ok] >= STEEP_DEG)),
               ruggedness_tri=float(np.nanmean(tri[ok])))
    sl = ok & (slope >= SLOPED_DEG)
    sector = ((aspect[sl] + 22.5) // 45).astype(int) % 8
    for k, name in enumerate(SECTORS):
        # flat counties: too few sloped pixels for a meaningful aspect mix -> uniform
        out[f"aspect_{name}"] = float(np.mean(sector == k)) if sl.sum() >= MIN_SLOPED_PIXELS else 0.125
    c = tcc[land]
    out["canopy_mean"] = float(np.nanmean(c))
    out["canopy_dense_frac"] = float(np.nanmean(c >= 50))
    lcl = lc[land]
    forest = np.isin(lcl, (41, 42, 43)); dev = np.isin(lcl, (21, 22, 23, 24)); wet = np.isin(lcl, (90, 95))
    out["forest_frac"] = float(forest.mean()); out["developed_frac"] = float(dev.mean())
    out["wetland_frac"] = float(wet.mean())
    out["forest_on_steep_frac"] = float(np.mean(forest & (slope[land] >= STEEP_DEG)))
    cd = c[dev]
    out["canopy_in_developed"] = float(np.nanmean(cd)) if np.isfinite(cd).sum() >= 50 else np.nan
    return out


SOIL_SQL = """
SELECT lo.areasymbol AS areasymbol,
  SUM(mo.areaovacres * c.comppct_r / 100.0) AS acres,
  SUM(CASE WHEN c.drainagecl IN ('Poorly drained','Very poorly drained') THEN mo.areaovacres * c.comppct_r / 100.0 ELSE 0 END) AS poor,
  SUM(CASE WHEN c.hydricrating = 'Yes' THEN mo.areaovacres * c.comppct_r / 100.0 ELSE 0 END) AS hydric,
  SUM(CASE WHEN r.mindep < 50 THEN mo.areaovacres * c.comppct_r / 100.0 ELSE 0 END) AS shallow,
  SUM(CASE WHEN m.wtdepannmin < 30 THEN mo.areaovacres * c.comppct_r / 100.0 ELSE 0 END) AS hiwt,
  SUM(CASE WHEN r.mindep < 50 OR m.wtdepannmin < 30 OR c.drainagecl IN ('Poorly drained','Very poorly drained')
           THEN mo.areaovacres * c.comppct_r / 100.0 ELSE 0 END) AS rootlimit
FROM laoverlap lo
JOIN muaoverlap mo ON mo.lareaovkey = lo.lareaovkey
JOIN component c ON c.mukey = mo.mukey
LEFT JOIN muaggatt m ON m.mukey = mo.mukey
OUTER APPLY (SELECT MIN(cr.resdept_r) AS mindep FROM corestrictions cr WHERE cr.cokey = c.cokey) r
WHERE lo.areatypename = 'County or Parish' AND lo.areasymbol IN ({areas})
GROUP BY lo.areasymbol
"""


def sda(query: str) -> list[dict]:
    import requests
    for attempt in range(6):
        try:
            r = requests.post(SDA, json=dict(query=query, format="JSON+COLUMNNAME"), timeout=300)
            if r.status_code == 200 and r.text.strip().startswith("{"):
                d = r.json().get("Table", [])
                return [dict(zip(d[0], row)) for row in d[1:]] if d else []
        except requests.RequestException:
            pass
        time.sleep(20 * (attempt + 1))
    raise RuntimeError("SDA query failed")


def soils(areas: list[str]) -> pd.DataFrame:
    rows = []
    for k in range(0, len(areas), 40):
        chunk = areas[k:k + 40]
        q = SOIL_SQL.format(areas=",".join(f"'{a}'" for a in chunk))
        res = sda(q)
        rows += res
        log(dict(source=SDA, query_sha256=sha256(q.encode()), n_areas=len(chunk), n_rows=len(res),
                 downloaded_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
        print(f"  SDA {k + len(chunk)}/{len(areas)}", flush=True)
    d = pd.DataFrame(rows)
    for c in d.columns[1:]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    import requests
    import shapely
    OUT.mkdir(parents=True, exist_ok=True)
    panel_fips = sorted({str(f) for p in OUT.glob("panel216_*.npz") for f in np.load(p)["fips"]})
    cty = counties()
    gz = pd.read_csv(GAZ, sep="\t", dtype={"GEOID": str}, encoding="latin-1")
    gz.columns = [c.strip() for c in gz.columns]
    gz = gz.assign(fips=gz.GEOID.str.zfill(5)).set_index("fips")
    gz = gz[gz.index.isin(cty.index)]
    lat, lon = np.radians(gz.INTPTLAT.to_numpy(float)), np.radians(gz.INTPTLONG.to_numpy(float))
    ids = gz.index.to_numpy()

    def km(i):
        return 6371.0 * 2 * np.arcsin(np.sqrt(np.sin((lat - lat[i]) / 2) ** 2
                                              + np.cos(lat[i]) * np.cos(lat) * np.sin((lon - lon[i]) / 2) ** 2))
    pos = {f: i for i, f in enumerate(ids)}
    within = {f: ids[km(pos[f]) <= SMOOTH_KM].tolist() for f in panel_fips}
    need = sorted({g for f in panel_fips for g in within[f]} & set(cty.index))
    print(f"panel counties {len(panel_fips)}, with 50 km neighbours {len(need)}", flush=True)
    sess = requests.Session()
    jobs = [(k, f) for f in need for k in ("dem", "tcc", "nlcd")]
    with ThreadPoolExecutor(a.workers) as ex:
        for i, _ in enumerate(ex.map(lambda kf: fetch(kf[0], kf[1], cty.geometry[kf[1]], sess), jobs)):
            if i % 300 == 0:
                print(f"  rasters {i}/{len(jobs)}", flush=True)
    stats = pd.DataFrame([raster_stats(f, cty.geometry[f]) for f in need]).set_index("fips")
    areas = sorted({cty.STUSPS[f] + f[2:] for f in need})
    soil = soils(areas)
    soil["fips"] = [next((f for f in need if cty.STUSPS[f] + f[2:] == s), None) for s in soil.areasymbol]
    soil = soil.dropna(subset=["fips"]).set_index("fips")
    tot = soil.acres.where(soil.acres > 0)
    stats["poorly_drained_share"] = soil.poor / tot
    stats["hydric_share"] = soil.hydric / tot
    stats["shallow_share"] = soil.shallow / tot
    stats["high_water_table_share"] = soil.hiwt / tot
    stats["root_limiting_share"] = soil.rootlimit / tot
    stats["windthrow_susceptibility"] = stats.forest_frac * stats.root_limiting_share
    stats["log_land_area_km2"] = np.log(gz.loc[stats.index, "ALAND"].astype(float) / 1e6)
    area = gz.loc[stats.index, "ALAND"].astype(float)
    for c in SMOOTH:
        v = stats[c]
        stats[f"s50_{c}"] = [np.nansum(v.reindex(within.get(f, [f])) * area.reindex(within.get(f, [f])))
                             / area.reindex(within.get(f, [f]))[v.reindex(within.get(f, [f])).notna()].sum()
                             if f in within else np.nan for f in stats.index]
    stats = stats.loc[panel_fips]
    stats.attrs = {}
    stats.to_parquet(OUT / "geography.parquet")
    meta = dict(columns=[c for c in stats.columns if c != "n_land_pixels"], n_counties=len(stats),
                missing=stats.isna().sum()[lambda s: s > 0].to_dict(),
                resolution_m=RES, canopy_year=2021, landcover_year=2021)
    (HERE / "data_provenance" / "geography_meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
