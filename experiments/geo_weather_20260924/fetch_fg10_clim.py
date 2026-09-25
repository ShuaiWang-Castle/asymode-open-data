"""Local wind climatology proxy: ERA5 monthly means of the hourly maximum 10 m gust
('10m_wind_gust_since_previous_post_processing'), 2010-2019, every calendar month, on the panel's 0.25-degree grid.
Used as a scaled stand-in for the local 98th percentile of the Klawa & Ulbrich (2003) storm-loss index.
Writes data/raw/era5_extra/era5_fg10_monthly_2010_2019.nc and logs source, request, size and SHA-256."""
import datetime as dt, hashlib, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from fetch_era5 import AREA  # noqa: E402
import cdsapi
dst = ROOT / "data" / "raw" / "era5_extra" / "era5_fg10_monthly_2010_2019.nc"; dst.parent.mkdir(parents=True, exist_ok=True)
req = {"product_type": ["monthly_averaged_reanalysis"], "variable": ["10m_wind_gust_since_previous_post_processing"],
       "year": [str(y) for y in range(2010, 2020)], "month": [f"{m:02d}" for m in range(1, 13)], "time": ["00:00"],
       "area": AREA, "data_format": "netcdf", "download_format": "unarchived"}
if not dst.exists():
    cdsapi.Client().retrieve("reanalysis-era5-single-levels-monthly-means", req, str(dst.with_suffix(".part")))
    dst.with_suffix(".part").rename(dst)
rec = dict(file=str(dst.relative_to(ROOT)), source="Copernicus Climate Data Store, reanalysis-era5-single-levels-monthly-means",
           request=req, downloaded_utc=dt.datetime.now(dt.timezone.utc).isoformat(), bytes=dst.stat().st_size,
           sha256=hashlib.sha256(dst.read_bytes()).hexdigest())
with open(Path(__file__).resolve().parent / "data_provenance" / "downloads.jsonl", "a") as f:
    f.write(json.dumps(rec) + "\n")
print("saved", dst.name, rec["bytes"])
