# Public data recipes for sub-county inputs (contribution to DESIGN v0)

Checked 2026-09-24. Every URL, size and field name below was confirmed that day through one of:
- HEAD requests;
- the `.idx` index files;
- metadata APIs (figshare, PyPI, the CDS process description, ARCO `.zmetadata`);
- range reads of a few hundred bytes.

No bulk data was downloaded for this note and no package was installed. The Python routes rest on PyPI wheel metadata for macOS 13 arm64; they were not run end to end here. Sketches marked *tested* were run against the live services.

All sources are public. Licences are listed per item.

## 0. Summary

Volume assumes ~2,400 CONUS counties and 12 windows × 216 h ≈ 2,600 valid hours.

| Need | Recommended source | Access | Volume | Account | Licence |
|---|---|---|---|---|---|
| (a) where customers live | Census 2020 blocks: `HOUSING20`, `POP20`, internal point | HTTPS zip per state | 10.1 GB of zips for CONUS (the `.dbf` is all you need) | no | public domain |
| (a) lightweight alternative | Census 2020 block-group centres of population | one national text file | 10.6 MB | no | public domain |
| (a) year-matched raster | WorldPop Global2 R2025A, constrained, 100 m or 1 km | HTTPS GeoTIFF | 1.51 GB (100 m) or 37 MB (1 km) per year | no | CC BY 4.0 |
| (a) node elevation / TPI | USGS 3DEP `getSamples` (1 arc-second layer) | REST JSON | ~1,000 points per request | no | public domain |
| (b) line proxy | TIGER/Line roads, MTFCC S1200 + S1400 | HTTPS zip per county | ~3.6 GB | no | public domain |
| (b) mapped power lines | OpenStreetMap (Geofabrik extracts) | `.osm.pbf` | 12.2 GB US, or 0.1–0.7 GB per state | no | **ODbL** |
| (b) canopy along proxy | NLCD Tree Canopy Cover v2025-6, per-year raster, `getSamples` | REST JSON | 4.8 M points ≈ 4,800 requests ≈ 35 min | no | public domain |
| (b) underground share | GUDS county underground fraction (2020) | REST JSON (Data Commons) | ~2,400 values | no | see provider |
| (c) weather at 3 km | HRRR `wrfsfc` on AWS, byte ranges from `.idx` | HTTPS / S3 anonymous | ~12 MB per hour → ~31 GB (full files ~390 GB) | no | NOAA open data |
| (d) ERA5 extras | CDS `reanalysis-era5-single-levels`; or ARCO-ERA5 Zarr | `cdsapi` / anonymous GCS | invariants and LAI: MB; hourly snow for all windows: ≤ 0.5 GB on CDS | CDS: yes | CC BY 4.0 |
| (e) county customers, coverage | EAGLE-I on figshare: `MCC.csv` (2022), `coverage_history.csv`, 2024 inline totals | HTTPS (figshare) | KB-scale (MCC, coverage) | no | CC BY 4.0 |

## Environment: macOS 13 arm64, no conda

The newest wheels of several geospatial packages target macOS 14 or 15, so on macOS 13:
- **Wheel-only packages** such as `eccodeslib`: pip silently falls back to an older wheel.
- **Packages that also ship an sdist** (`rasterio`, `pyproj`, `pygrib`): pip tries to compile and fails without GDAL, PROJ or ecCodes.

Install with `--only-binary=:all:` so pip picks the newest *compatible* wheel instead:

```bash
/opt/homebrew/bin/python3.11 -m venv .venv && . .venv/bin/activate      # 3.11 or 3.12; 3.14 has no rasterio/pyproj wheel for macOS 13
pip install --only-binary=:all: eccodes eccodeslib cfgrib xarray        # GRIB (HRRR)
pip install --only-binary=:all: rasterio pyproj pyshp pyogrio osmium    # rasters, shapefiles, OSM
pip install --only-binary=:all: zarr numcodecs s3fs gcsfs               # HRRR-Zarr / ARCO-ERA5
```

Latest versions with a macOS ≤ 13 arm64 wheel (PyPI metadata, 2026-09-24):

| Package | Version | Python | Note |
|---|---|---|---|
| `eccodeslib` | 2.48.0.26 | cp310–cp314 | ≥ 2.48.3 is macOS 15 only |
| `rasterio` | 1.4.1 | ≤ cp313 | |
| `pyproj` | 3.6.1 | ≤ cp312 | |
| `pyogrio` | 0.13.0 | abi3 | |
| `osmium` (pyosmium) | 4.3.1 | all | |
| `numcodecs` | 0.16.5 / 0.17.0 | cp311 / cp312+ | |
| `zarr` | 3.x / 2.18.x | Python ≥ 3.12 / 3.11 | |

- **`pygrib`**: only 2.1.4 (cp310) has such a wheel. Use eccodes/cfgrib instead.
- **`herbie-data`** requires `pyproj>=3.7`, which has no macOS 13 arm64 wheel. It will not install wheels-only; the byte-range recipe in (c) does the same job.

## (a) Where customers live

### a1. Census 2020 blocks — recommended

- **URL**: `https://www2.census.gov/geo/tiger/TIGER2020/TABBLOCK20/tl_2020_{SS}_tabblock20.zip`, one per state. Sizes run 5–746 MB (NC 348 MB, TX 746 MB); 48 states + DC total 10.1 GB.
- **Fields**: `GEOID20`, `ALAND20`, `INTPTLAT20`, `INTPTLON20`, **`HOUSING20`**, `POP20`, `UR20`.
- **Why it fits**: housing units are the closest public proxy for residential meters. The internal point gives a node location for every block, so the quadrature needs no raster at all.

```python
import io, zipfile, requests, shapefile, pandas as pd        # pyshp: attributes only, no GDAL
st = "37"
z = zipfile.ZipFile(io.BytesIO(requests.get(f"https://www2.census.gov/geo/tiger/TIGER2020/TABBLOCK20/tl_2020_{st}_tabblock20.zip", timeout=900).content))
r = shapefile.Reader(dbf=io.BytesIO(z.read(f"tl_2020_{st}_tabblock20.dbf")))
b = pd.DataFrame(r.records(), columns=[f[0] for f in r.fields[1:]])
b = b[b.HOUSING20 > 0].assign(fips=b.GEOID20.str[:5], lat=b.INTPTLAT20.astype(float), lon=b.INTPTLON20.astype(float))
```

Pitfalls:
- **Use `HOUSING20` as the weight, not `POP20`.** Block housing counts were held invariant by the 2020 disclosure-avoidance system; block populations carry injected noise, including population in blocks with no housing.
- **Keep only the `.dbf`.** The zips are mostly polygon geometry. Stream to disk for the large states rather than holding them in memory.
- **Connecticut FIPS changed.** 2020 blocks use the old CT county FIPS (09001–09015); Census geography from 2022 uses planning regions (09110–09190). Crosswalk if CT is in the sample.

### a2. Block-group centres of population — lightweight

- **URL**: `https://www2.census.gov/geo/docs/reference/cenpop2020/blkgrp/CenPop2020_Mean_BG.txt`. National, 10,583,505 bytes, about 240k block groups.
- **Columns**: `STATEFP`, `COUNTYFP`, `TRACTCE`, `BLKGRPCE`, `POPULATION`, `LATITUDE`, `LONGITUDE`.
- Good enough for about one node per 1,500 people. It has no housing counts.
- Read with `encoding="utf-8-sig"` and keep FIPS as strings (`dtype=str`).

### a3. WorldPop — year-matched rasters

- **Global2 R2025A, constrained, 2015–2030 yearly**:
  - 100 m: `https://data.worldpop.org/GIS/Population/Global_2015_2030/R2025A/{YYYY}/USA/v1/100m/constrained/usa_pop_{YYYY}_CN_100m_R2025A_v1.tif`, 1.51 GB per year, DOI 10.5258/SOTON/WP00839.
  - 1 km: `.../R2025A/{YYYY}/USA/v1/1km_ua/constrained/usa_pop_{YYYY}_CN_1km_R2025A_UA_v1.tif`, 37 MB, DOI 10.5258/SOTON/WP00840.
- **Older Global 2000–2020, unconstrained**:
  - 100 m: `.../Global_2000_2020/{YYYY}/USA/usa_ppp_{YYYY}.tif`, 4.01 GB, DOI 10.5258/SOTON/WP00645.
  - 1 km: `.../Global_2000_2020_1km/2020/USA/usa_ppp_2020_1km_Aggregated.tif`, 53 MB.
- **Discovery API**: `https://hub.worldpop.org/rest/data/pop/{alias}?iso3=USA`. Aliases include `wpgp`, `G2_CN_POP_R25A_100m`, `G2_CN_POP_R25A_1km`.
- **Licence**: CC BY 4.0.

```python
import rasterio                                   # after downloading the file once (see pitfalls)
from rasterio.windows import from_bounds
with rasterio.open("usa_pop_2022_CN_100m_R2025A_v1.tif") as src:
    w = from_bounds(-78.99, 35.52, -78.25, 36.08, src.transform)     # one county's bbox
    pop = src.read(1, window=w, masked=True); tr = src.window_transform(w)
```

Pitfalls:
- **data.worldpop.org ignores HTTP Range on GET.** HEAD advertises `Accept-Ranges: bytes`, but a ranged GET returns HTTP 200 with the whole file. `/vsicurl/` windowed reads therefore pull the full 1.5–4 GB. Download once, then read windows locally.
- **Constrained vs unconstrained**: "constrained" puts people only in built-settlement pixels; "unconstrained" spreads them over all land.
- **Blocks are better for the 2020 geography.** WorldPop is a modelled disaggregation of census totals. Use it for 2019–2024 year matching or when a raster form is needed.

### a4. LandScan

- **LandScan Global 2000–2022** (30″ ≈ 1 km, ambient 24-hour population) is on figshare: DOI 10.6084/m9.figshare.28439699. The record's licence is CC0. Zips are about 135–145 MB per year, e.g. `landscan-global-2020-assets.zip` = `https://ndownloader.figshare.com/files/52669904`.
- **LandScan USA** (3″ ≈ 90 m, separate day and night layers; the night layer is the residence proxy) is only on the ORNL portal `https://landscan.ornl.gov/`. The portal is a JavaScript app; its sub-pages returned 403 to scripted requests, so the download is manual.
- **Pitfall**: LandScan Global is *ambient* population, not residences. Prefer blocks (a1) for customer location.

### a5. Elevation and TPI at the nodes — USGS 3DEP (*tested*)

- **Endpoint**: `https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/getSamples`.
- **TPI**: sample each node plus 8 points on a 1-km ring; TPI = z(centre) − mean z(ring).

```python
import json, requests, numpy as np
DEM = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/getSamples"
def dem(lonlat):                                   # <= 1000 points, spatially compact (one county)
    d = {"geometry": json.dumps({"points": np.round(lonlat, 6).tolist(), "spatialReference": {"wkid": 4326}}),
         "geometryType": "esriGeometryMultipoint", "returnFirstValueOnly": "true", "f": "json",
         "pixelSize": json.dumps({"x": 90, "y": 90, "spatialReference": {"wkid": 3857}})}   # -> 1 arc-second layer
    z = np.full(len(lonlat), np.nan)
    for s in requests.post(DEM, data=d, timeout=180).json()["samples"]:
        try: z[s["locationId"]] = float(s["value"])
        except (TypeError, ValueError): pass       # 'NoData'
    return z
```

Pitfalls:
- **Without `pixelSize`** the service reads the 1-m lidar mosaic and returns HTTP 504 above roughly 50 points per request.
- **Batches must be spatially compact.** 1,000 points inside a 0.5° box return in about 3 s. 500 points spread over 10° × 7° take 25 s, and 1,000 dispersed points hit the ~30 s gateway limit. Sort points by county before batching.
- **Downscaling needs the model's own terrain as z_c**: ERA5 `z/9.80665` for ERA5, or HRRR `HGT:surface` for HRRR. The 2-m temperatures refer to the model's smoothed terrain, not to a DEM average.
- **Volume**: block-group centres × 9 points ≈ 2.2 M points ≈ 2,200 requests. Every block × 9 is roughly 50 M points; use block centres without rings there, or a local DEM.

## (b) Distribution-line proxy and canopy along it

### b1. TIGER/Line roads

- **URL**: `https://www2.census.gov/geo/tiger/TIGER{YYYY}/ROADS/tl_{YYYY}_{SSCCC}_roads.zip`. TIGER2023, 2024 and 2025 are online.
- **Size**: in a 60-county CONUS sample, median 1.03 MB and mean 1.51 MB (max 12.9 MB). About 3.6 GB for 2,400 counties.
- **Proxy classes**: S1200 (secondary) + S1400 (local/rural streets). Also present are S1100 (primary/limited-access, rarely lined), S1740 (private service roads: logging, oil fields, ranches), S1730 (alleys), S1640 (service drives) and S1500/S171x–S183x (trails, walkways, stairways, paths).

```python
import numpy as np, shapefile                      # pyshp reads the zipped shapefile directly
def road_points(zip_path, n=1000, off_m=25, keep=("S1200", "S1400"), seed=0):
    r = shapefile.Reader(zip_path); a, b = [], []
    for sh, rec in zip(r.iterShapes(), r.iterRecords()):
        if rec["MTFCC"] in keep:
            for s, e in zip(sh.parts, list(sh.parts[1:]) + [len(sh.points)]):     # split multipart lines
                p = np.asarray(sh.points[s:e]); a.append(p[:-1]); b.append(p[1:])
    a, b = np.vstack(a), np.vstack(b); k = np.cos(np.deg2rad(a[:, 1]))
    dx, dy = (b[:, 0] - a[:, 0]) * 111.32 * k, (b[:, 1] - a[:, 1]) * 110.57; L = np.hypot(dx, dy)   # km
    rng = np.random.default_rng(seed); i = rng.choice(len(L), n, p=L / L.sum()); t = rng.random(n)[:, None]
    p = a[i] + t * (b[i] - a[i]); o = off_m / 1000 * np.c_[-dy[i] / L[i] / (111.32 * k[i]), dx[i] / L[i] / 110.57]
    return np.vstack([p + o, p - o]), L.sum()     # 2n points (both sides) and proxy length (km)
```

Pitfalls:
- **Split multipart shapes by `parts`.** Otherwise segments jump between parts.
- **Convert degree lengths with cos(latitude).**
- **One vintage is enough.** Roads change little over 2019–2024; use a single vintage for all years.
- **Urban streets carry underground lines.** Weight the proxy by (1 − underground share), see b4.
- *Sampling logic tested earlier with a pure-Python shapefile reader; the pyshp form is untested here.*

### b2. OpenStreetMap power lines

- **Geofabrik extracts**: `https://download.geofabrik.de/north-america/us-latest.osm.pbf` (12.2 GB, snapshot 2026-09-23) or per state `.../us/{state}-latest.osm.pbf` (KS 116 MB, NC 429 MB, TX 722 MB).
- **Tags**: `power=line` (mostly transmission and sub-transmission), `power=minor_line` (distribution), `power=cable` (underground).

```bash
brew install osmium-tool                           # or: pip install osmium (pyosmium 4.3.1 has macOS 13 arm64 wheels)
curl -LO https://download.geofabrik.de/north-america/us/north-carolina-latest.osm.pbf
osmium tags-filter north-carolina-latest.osm.pbf w/power=line,minor_line,cable -o nc_power.osm.pbf
osmium export nc_power.osm.pbf -f geojsonseq -o nc_power.geojsonseq      # then length per county / per HRRR cell
```

Pitfalls:
- **Distribution coverage is incomplete and uneven** across US counties. A missing line is not proof there is none. Use OSM to validate the road proxy in well-mapped counties, not as a covariate everywhere.
- **Licence is ODbL.** Attribution "© OpenStreetMap contributors" is required. Committing per-county OSM-derived *tables* to a public repo may make them a Derivative Database under share-alike; decide before pushing.
- **`-latest` is a redirect** to a dated file (e.g. `north-carolina-260923.osm.pbf`). Record the dated name.
- **HIFLD transmission lines**: the portal closed in August 2025; archives exist, e.g. the Data Rescue Project. They cover transmission only, which is not the outage-relevant network.

### b3. Canopy along the proxy — NLCD Tree Canopy Cover (*tested*)

- **Service**: `https://imagery.geoplatform.gov/iipp/rest/services/Vegetation/USFS_EDW_NLCD_TCC_CONUS/ImageServer`. This is USFS TCC v2025-6 at 30 m, with one raster per year from 1985 to 2025.
- **Raster id = year − 1945** (2019 → 74, …, 2024 → 79, 2025 → 80). Verified via `/query`.

```python
TCC = "https://imagery.geoplatform.gov/iipp/rest/services/Vegetation/USFS_EDW_NLCD_TCC_CONUS/ImageServer/getSamples"
def tcc(lonlat, year):                             # <= 1000 compact points; returns % canopy
    d = {"geometry": json.dumps({"points": np.round(lonlat, 6).tolist(), "spatialReference": {"wkid": 4326}}),
         "geometryType": "esriGeometryMultipoint", "returnFirstValueOnly": "true", "f": "json",
         "mosaicRule": json.dumps({"mosaicMethod": "esriMosaicLockRaster", "lockRasterIds": [year - 1945]})}
    v = np.full(len(lonlat), np.nan)
    for s in requests.post(TCC, data=d, timeout=180).json()["samples"]:
        try: v[s["locationId"]] = float(s["value"])
        except (TypeError, ValueError): pass       # 'NoData'
    return np.where((v >= 0) & (v <= 100), v, np.nan)
# county descriptors: mean(tcc), share(tcc >= 50), proxy_km * share(tcc >= 50) * (1 - underground) / customers
```

Pitfalls:
- **Lock the raster.** The default mosaic (Northwest) returns one raster, not necessarily the event year.
- **The USFS-hosted copy** of the service on `apps.fs.usda.gov/fsgisx01` returned HTTP 403 in September 2026. The geoplatform mirror works.
- **Offset from the centreline.** A 30-m pixel on the centreline is mostly pavement; sample 20–30 m to both sides and average.
- **NLCD vs Science TCC.** The NLCD version masks water and non-tree cropland; the "Science" version keeps them.
- **TCC is annual cover, not phenology.** Leaf-on state has to come from season, ERA5 LAI (d) or HRRR `VEG`/`LAI`.
- **Throughput**: about 2.5 requests/s with 4 threads at 1,000 points per request. That is ~35 min for 2,400 counties × 2,000 points.
- **Alternative**: the earlier campaign exported county TCC rasters with `exportImage` (see its `SOURCES.md`). If their pixel size is 30 m, sample them locally instead.

### b4. Underground share, optional — GUDS (*tested*)

- **Source**: Stanford Data Commons observation API. Variable `distri_ug_rate`, county entities `geoId/{SSCCC}`, date `2020`. From the Grid Underground Distribution Statistics (Sun et al., arXiv:2402.06668).

```python
p = {"select": ["entity", "variable", "date", "value", "facet"], "date": "2020", "variable": {"dcids": ["distri_ug_rate"]},
     "entity": {"dcids": [f"geoId/{f}" for f in fips_batch]}}            # explicit ids, batches of a few hundred
v = requests.post("https://datacommons.stanford.edu/core/api/v2/observation", json=p, timeout=60).json()["byVariable"]["distri_ug_rate"]["byEntity"]
```

- **Pitfall**: the "all counties in the USA" expression (`country/USA<-containedInPlace+{typeOf:County}`) returned only 481 counties. Query explicit ids.

## (c) HRRR 3 km — AWS `noaa-hrrr-bdp-pds` (*tested*)

**Files**
- Pattern: `https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{YYYYMMDD}/conus/hrrr.t{HH}z.wrfsfcf{FF}.grib2`, plus `.grib2.idx`. Access is anonymous; S3 URI `s3://noaa-hrrr-bdp-pds/...`.
- Mirror: `https://storage.googleapis.com/high-resolution-rapid-refresh/...`, same paths.
- Sizes of `wrfsfcf01`: 121 MB (148 records, 2019-06-01 12z) and 150 MB (170 records, 2024-05-26 12z). `wrfprsf01` is 410 MB and `wrfnatf01` is 716 MB.

**Fields**

Exact `.idx` strings in `wrfsfcf01`. They are identical in 2019 and 2024; the record numbers differ, so match by name.

| Field | `var:level:time` in f01 | MB |
|---|---|---|
| gust (instantaneous at valid time) | `GUST:surface:1 hour fcst` | 1.2 |
| hourly max 10 m wind speed | `WIND:10 m above ground:0-1 hour max fcst` | 1.2 |
| (hourly max u/v) | `MAXUW` / `MAXVW:10 m above ground:0-1 hour max fcst` | 1.3 each |
| 10 m u/v (grid-relative) | `UGRD` / `VGRD:10 m above ground:1 hour fcst` | 1.2–2.4 each |
| precipitation, hour-ending | `APCP:surface:0-1 hour acc fcst` | 0.2 |
| 2 m temperature / dew point | `TMP` / `DPT:2 m above ground:1 hour fcst` | 1.2 each |
| precipitation type (0/1) | `CRAIN` / `CFRZR` / `CSNOW` / `CICEP:surface:1 hour fcst` | < 0.1 |
| freezing rain, snowfall (1 h acc) | `FRZR:surface:0-1 hour acc fcst`, `ASNOW:surface:0-1 hour acc fcst` | < 0.1 |
| % frozen precipitation | `CPOFP:surface:1 hour fcst` | < 0.1 |
| soil moisture availability | `MSTAV:0 m underground:1 hour fcst` | 1.5 |
| **volumetric soil water** | **not in wrfsfc**: `SOILW:{0,0.01,0.04,0.1,0.3,0.6,1,1.6,3}-… m below ground` in `wrfprs`/`wrfnat` | 1.3 each |
| CAPE, model terrain | `CAPE:surface`, `HGT:surface` | 0.6, 2.2 |
| vegetation fraction | `VEG:surface` (present in 2024, absent in 2019) | 0.9 |

**Volume**
- The core set (GUST, WIND max, UGRD/VGRD, APCP, TMP, DPT, 4 precipitation types) is 7.3 MB per hour in 2019 and 9.8 MB in 2024.
- Adding MSTAV, CAPE, FRZR and ASNOW gives about 12 MB per hour, roughly 31 GB for 2,600 hours. Full `wrfsfc` files would be about 390 GB.
- Five `SOILW` levels from `wrfprs` add about 17 GB.

```python
import requests
B = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
KEEP = ("GUST:surface", "WIND:10 m above ground", "APCP:surface", "TMP:2 m above ground", "DPT:2 m above ground",
        "CRAIN:surface", "CFRZR:surface", "CSNOW:surface", "CICEP:surface", "FRZR:surface", "MSTAV:0 m underground")
def hrrr_subset(ymd, cycle, out, fxx=1):          # valid time = cycle + fxx hours
    url = f"{B}/hrrr.{ymd}/conus/hrrr.t{cycle:02d}z.wrfsfcf{fxx:02d}.grib2"
    rec = [l.split(":") for l in requests.get(url + ".idx", timeout=60).text.splitlines()]
    off = [int(r[1]) for r in rec] + [int(requests.head(url, timeout=60).headers["Content-Length"])]
    names = [f"{r[3]}:{r[4]}" for r in rec if f"{r[3]}:{r[4]}" in KEEP]
    with open(out, "wb") as f:                     # concatenated GRIB2 messages form a valid GRIB2 file
        for i, r in enumerate(rec):
            if f"{r[3]}:{r[4]}" in KEEP:
                f.write(requests.get(url, headers={"Range": f"bytes={off[i]}-{off[i+1]-1}"}, timeout=120).content)
    return names
```

The range logic is *tested*: each fetched record starts with `GRIB`, ends with `7777`, and its length field matches.

**Decoding** without conda uses eccodes (see Environment). Messages come back in `.idx` order:

```python
import eccodes, numpy as np
def decode(path, names):
    out = {}
    with open(path, "rb") as f:
        for name in names:
            h = eccodes.codes_grib_new_from_file(f)
            out[name] = eccodes.codes_get_values(h).reshape(1059, 1799)   # row 0 = southern edge
            eccodes.codes_release(h)
    return out
```

With `cfgrib`, open with `cfgrib.open_datasets(path)` (a list). A single `open_dataset` fails on mixed `typeOfLevel` / `stepType`.

**Grid and county mapping** (*tested*, no pyproj needed). Parameters were read from section 3 of a live message:
- template 3.30, Lambert conformal;
- Nx = 1799, Ny = 1059, Dx = Dy = 3000 m;
- La1 = 21.138123, Lo1 = 237.280472;
- LoV = 262.5, Latin1 = Latin2 = LaD = 38.5;
- earth = sphere, R = 6,371,229 m;
- scan mode 0x40 (rows go south to north);
- u/v flag 0x08 = **grid-relative**.

```python
R, n = 6371229.0, np.sin(np.deg2rad(38.5)); F = np.cos(np.deg2rad(38.5)) * np.tan(np.pi/4 + np.deg2rad(38.5)/2)**n / n
rho = lambda lat: R * F / np.tan(np.pi/4 + np.deg2rad(lat)/2)**n
def lcc(lat, lon):
    th = n * np.deg2rad(np.asarray(lon) + 97.5); return rho(np.asarray(lat)) * np.sin(th), rho(38.5) - rho(np.asarray(lat)) * np.cos(th)
x0, y0 = lcc(21.138123, -122.719528)
def hrrr_ij(lat, lon):                            # HRRR cell (row, col) of any node, e.g. a census block point
    x, y = lcc(lat, lon); return np.rint((y - y0) / 3000).astype(int), np.rint((x - x0) / 3000).astype(int)
# earth-relative wind (standard Lambert-conformal rotation): a = n*deg2rad(lon + 97.5); u_e = cos(a)*u + sin(a)*v; v_e = -sin(a)*u + cos(a)*v
```

The mapping reproduces the published corners: SW → (0, 0); NE (47.84220, −60.91719) → (1058, 1798). Mapping each customer node to its cell this way needs no polygon operations.

Pitfalls:
- **f00 holds no hourly accumulations or maxima.** Its `APCP` / `WIND max` are zero-length ("0-0 day acc/max fcst"). For hour-ending precipitation, freezing rain and max wind at valid hour t, use cycle t−1, `f01`. Reading all fields from that one file gives a coherent set; instantaneous fields then are 1-h forecasts.
- **GUST is instantaneous.** `wrfsfc` has no hourly max gust; the hourly maximum available is `WIND 0-1 hour max` (10 m wind speed).
- **Winds are grid-relative.** Rotate before using direction. Speed is unaffected.
- **HRRR version change**: v3 ran from 2018-07-12 and v4 from 2020-12-02, so the 2019–2024 windows straddle a model change. Add a version indicator or check for shifts.
- **Use `HGT:surface` as z_c** for lapse-rate downscaling of `TMP`/`DPT`. It is the smoothed 3 km model terrain.
- **Missing cycles occur.** Handle 404 and fall back to the GCS mirror.
- **Merge requests.** Nearby records can share one Range request if you accept the small gaps between them; e.g. one ~0.3 MB request covers `APCP` through `CRAIN` (records 84–93 in 2024).

**No-GRIB alternative: HRRR-Zarr** (University of Utah, `s3://hrrrzarr`, anonymous).
- Layout: `sfc/{YYYYMMDD}/{YYYYMMDD}_{HH}z_anl.zarr/{level}/{VAR}/{level}/{VAR}`, e.g. `.../10m_above_ground/UGRD/10m_above_ground/UGRD`. Zarr v2, **float16**, 150 × 150 chunks, blosc-lz4, shape (1059, 1799). Present for 2019 and 2024.
- The `fcst.zarr` holds `APCP_1hr_acc_fcst`, `WIND_1hr_max_fcst`, `FRZR_1hr_acc_fcst`, `LAI`, `VEG`, etc. as (lead, y, x) with **all leads in one chunk** (48 for synoptic cycles, 18 otherwise). Reading lead 1 pulls every lead of a tile.
- Use it for instantaneous analysis fields read by tile; use GRIB byte ranges for the hour-ending fields.
- float16 quantises 2 m temperature (in K) to about 0.125–0.25 K.

## (d) ERA5 fields not yet pulled

CDS names, verified on the public process description of `reanalysis-era5-single-levels`:
- `geopotential` (surface; orography = z / 9.80665);
- `leaf_area_index_high_vegetation`, `leaf_area_index_low_vegetation`;
- `snow_depth` (+ `snow_density`);
- `high_vegetation_cover`, `low_vegetation_cover`, `type_of_high_vegetation`, `type_of_low_vegetation`, `land_sea_mask`;
- **sub-grid orography**: `standard_deviation_of_orography`, `standard_deviation_of_filtered_subgrid_orography`, `slope_/anisotropy_/angle_of_sub_gridscale_orography`. These are free sub-grid terrain descriptors on the ERA5 grid.

Licence: CC BY 4.0 (CDS catalogue).

```python
import cdsapi                                      # existing account; one small request each
c = cdsapi.Client(); area = [50, -125, 24, -66]    # N, W, S, E
inv = ["geopotential", "land_sea_mask", "standard_deviation_of_orography", "slope_of_sub_gridscale_orography",
       "high_vegetation_cover", "low_vegetation_cover", "type_of_high_vegetation", "type_of_low_vegetation"]
c.retrieve("reanalysis-era5-single-levels", {"product_type": ["reanalysis"], "variable": inv, "year": ["2020"], "month": ["01"],
           "day": ["01"], "time": ["00:00"], "area": area, "data_format": "netcdf", "download_format": "unarchived"}, "era5_invariants.nc")
c.retrieve("reanalysis-era5-single-levels", {"product_type": ["reanalysis"], "variable": ["leaf_area_index_high_vegetation",
           "leaf_area_index_low_vegetation"], "year": ["2020"], "month": [f"{m:02d}" for m in range(1, 13)], "day": ["15"],
           "time": ["00:00"], "area": area, "data_format": "netcdf", "download_format": "unarchived"}, "era5_lai_monthly.nc")
# snow: ["snow_depth", "snow_density"] hourly inside each window, one request per calendar month (as for the other fields)
```

No-account alternative: ARCO-ERA5 at `gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3`, anonymous (`storage_options={"token": "anon"}`). Covers 1940-01-01 to 2026-06-30 (ERA5T to 2026-09-18).
- Names: `geopotential_at_surface`, `leaf_area_index_high_vegetation`, `leaf_area_index_low_vegetation`, `snow_depth` (units "m of water equivalent"), `snow_density`, `standard_deviation_of_orography`, …
- Every variable, the invariants included, has a `time` dimension, and a chunk is one global hour (721 × 1440 float32).
- Latitude runs descending; longitude runs 0–360.

Pitfalls:
- **ERA5 and ERA5-Land LAI is a monthly climatology.** It has no inter-annual variability (ERA5-Land data documentation), so 12 values are the whole field. For event-year leaf-on, use HRRR `VEG`/`LAI` or MODIS.
- **"snow_depth" means different things in the two datasets.**
  - ERA5: `snow_depth` is `sd` in **m of water equivalent**; physical depth = sd × 1000 / `snow_density`.
  - ERA5-Land: `snow_depth` is `sde`, the **physical** depth in m; `snow_depth_water_equivalent` is `sd`.
- **Orography**: `z/9.80665` is ERA5's smoothed grid-box model terrain. It is the correct z_c for downscaling ERA5 `t2m`; in mountainous terrain it can differ from a DEM mean by tens to hundreds of metres.
- **Invariants**: request one time step. Multiple times return identical copies.

## (e) EAGLE-I county customers and coverage

Figshare record 10.6084/m9.figshare.24237376, **v4 (2026-02-25)**, CC BY 4.0: *EAGLE-I Recorded Electricity Outages 2014-2025*. Direct links are `https://ndownloader.figshare.com/files/{id}`.

| File | id | Bytes | Content |
|---|---|---:|---|
| `MCC.csv` | 42547708 | 40,584 | modelled electric customers per county **as of 2022**: `County_FIPS`, `Customers` |
| `coverage_history.csv` | 42547714 | 11,965 | state × year 2018–2022: `year, state, total_customers, min_covered, max_covered, min_pct_covered, max_pct_covered` |
| `DQI.csv` | 42547705 | 4,338 | FEMA region × year 2018–2022 data-quality index and components |
| `eaglei_outages_{2014…2025}.csv` | e.g. 2024 = 53581661 | 0.08–1.44 GB | 15-min county records |

County customer counts by year:

| Year | Source | Access |
|---|---|---|
| 2019–2021 | none published per year | scale 2022 MCC by the state `total_customers` in `coverage_history.csv` (EIA-861-based, 2018–2022); map FIPS → postal code with the Census Gazetteer |
| 2022 | `MCC.csv` | figshare, no account |
| 2023 | "Modeled County Electric Customers 2023", ORNL Open Energy Data Portal (`openenergyhub.ornl.gov/explore/dataset/modeled-county-customers-2023/`) | **HTTP 404 on 2026-09-24**; earlier described as needing an approved account |
| 2024 | `total_customers` column inside `eaglei_outages_2024.csv` (also in the ORNL 2024 release, 10.13139/OLCF/2500278) | figshare, no account |
| 2025 | *EAGLE-I County Customer Dataset Fall 2025*, DOI 10.13139/ORNLNCCS/3022751 (released 2026-05-06): FIPS, customers, type (modelled / collected / mixed). Built from 2023 EIA-861, 2021 HIFLD territories, 2021 LandScan and 2025 outages | Globus only (login) |

```python
import io, requests, pandas as pd
get = lambda fid: io.BytesIO(requests.get(f"https://ndownloader.figshare.com/files/{fid}", timeout=300).content)
mcc = pd.read_csv(get(42547708), encoding="utf-8-sig", dtype={"County_FIPS": str})
mcc = mcc[mcc.County_FIPS.str.isdigit()].assign(fips=lambda d: d.County_FIPS.str.zfill(5))   # drops 'Grand Total'
cov = pd.read_csv(get(42547714)); cov["year"] = pd.to_datetime(cov.year, format="%m/%d/%y").dt.year
```

This sketch is *tested*: 3,233 county rows, 154.5 M customers, no duplicate FIPS.

Pitfalls:
- **`MCC.csv` ends with a `Grand Total` row** (154,451,840). Summing without dropping it doubles the total.
- **MCC FIPS format**: `County_FIPS` has no leading zeros (312 four-digit codes) and the file starts with a UTF-8 BOM.
- **The outage count column changes name**: `customers_out` in 2014–2022, 2024 and 2025, but **`sum`** in 2023. Only 2024 adds `total_customers`.
- **Coverage tables stop at 2022.** No per-state coverage table ships for 2023–2025.
- **The ORNL Constellation copy of 2014–2022** (10.13139/ORNLNCCS/1975202) has 10 files: no `MCC.csv`, no `DQI.csv`. The figshare copy has them.
- **Download client matters.** `ndownloader` answers with a 302 to a pre-signed S3 URL valid for 10 s:
  - plain `python-requests` works, including Range;
  - a browser-like User-Agent received an HTTP 202 bot-check page;
  - `curl -L -r …` returned 404 in one test.
- **Zeros are omitted** from the outage records: a missing cell is either zero or not collected.

## Sources

- [TIGER/Line 2020 tabulation blocks](https://www2.census.gov/geo/tiger/TIGER2020/TABBLOCK20/) and [2020 TIGER/Line technical documentation](https://www2.census.gov/geo/pdfs/maps-data/data/tiger/tgrshp2020/TGRSHP2020_TechDoc.pdf)
- [Census 2020 centers of population](https://www.census.gov/geographies/reference-files/2020/geo/2020-centers-population.html)
- [TIGER/Line roads](https://www2.census.gov/geo/tiger/)
- [WorldPop hub and REST API](https://hub.worldpop.org/rest/data/pop) and [WorldPop licence](https://hub.worldpop.org/data/licence.txt)
- [LandScan Global 2000–2022 on figshare](https://doi.org/10.6084/m9.figshare.28439699), its [data descriptor](https://www.nature.com/articles/s41597-025-04817-z), and the [LandScan portal](https://landscan.ornl.gov/)
- [USGS 3DEP ImageServer](https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer)
- [USFS NLCD Tree Canopy Cover image service](https://imagery.geoplatform.gov/iipp/rest/services/Vegetation/USFS_EDW_NLCD_TCC_CONUS/ImageServer) and the [USFS TCC gateway](https://data.fs.usda.gov/geodata/rastergateway/treecanopycover/)
- [Geofabrik US extracts](https://download.geofabrik.de/north-america/us.html), the [ODbL licence](https://opendatacommons.org/licenses/odbl/), and the [HIFLD archive (Data Rescue Project)](https://portal.datarescueproject.org/datasets/hifld-open-transmission-lines/)
- [GUDS paper, arXiv:2402.06668](https://arxiv.org/abs/2402.06668) and the [Stanford Data Commons API](https://datacommons.stanford.edu/)
- [HRRR on AWS](https://noaa-hrrr-bdp-pds.s3.amazonaws.com/index.html) and [HRRR-Zarr documentation](https://mesowest.utah.edu/html/hrrr/zarr_documentation/html/zarr_HowToDownload.html)
- [ERA5 single levels (CDS)](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels), [ERA5-Land data documentation](https://confluence.ecmwf.int/display/CKB/ERA5-Land:+data+documentation), and [ARCO-ERA5](https://github.com/google-research/arco-era5)
- [EAGLE-I on figshare](https://doi.org/10.6084/m9.figshare.24237376), its [data descriptor (Brelsford et al. 2024, Sci Data 11:271)](https://doi.org/10.1038/s41597-024-03095-5), and the [EAGLE-I County Customer Dataset Fall 2025](https://doi.org/10.13139/ORNLNCCS/3022751)
- PyPI metadata for wheel availability: [eccodeslib](https://pypi.org/project/eccodeslib/), [rasterio](https://pypi.org/project/rasterio/), [pyproj](https://pypi.org/project/pyproj/), [pygrib](https://pypi.org/project/pygrib/), [herbie-data](https://pypi.org/project/herbie-data/), [osmium](https://pypi.org/project/osmium/)
