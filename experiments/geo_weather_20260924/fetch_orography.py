"""ERA5 surface geopotential (orography) on the panel's 0.25-degree grid; one time step (the field is invariant).
Writes data/raw/era5_extra/era5_orography.nc and logs source, request, size and SHA-256."""
import datetime as dt, hashlib, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from fetch_era5 import AREA  # noqa: E402
import cdsapi
dst = ROOT / "data" / "raw" / "era5_extra" / "era5_orography.nc"; dst.parent.mkdir(parents=True, exist_ok=True)
req = {"product_type": ["reanalysis"], "variable": ["geopotential"], "year": ["2022"], "month": ["01"], "day": ["01"],
       "time": ["00:00"], "area": AREA, "data_format": "netcdf", "download_format": "unarchived"}
if not dst.exists():
    cdsapi.Client().retrieve("reanalysis-era5-single-levels", req, str(dst.with_suffix(".part"))); dst.with_suffix(".part").rename(dst)
rec = dict(file=str(dst.relative_to(ROOT)), source="Copernicus Climate Data Store, reanalysis-era5-single-levels", request=req,
           downloaded_utc=dt.datetime.now(dt.timezone.utc).isoformat(), bytes=dst.stat().st_size, sha256=hashlib.sha256(dst.read_bytes()).hexdigest())
with open(Path(__file__).resolve().parent / "data_provenance" / "downloads.jsonl", "a") as f:
    f.write(json.dumps(rec) + "\n")
print("saved", dst.name, rec["bytes"])
