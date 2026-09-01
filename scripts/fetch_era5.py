"""Fetch ERA5 single-level fields for each archived storm window.

One request per storm window rather than one per year: the panel only ever looks
at a week around each event, and pulling whole years would cost two orders of
magnitude more for data that is never read.

Requires ~/.cdsapirc and an accepted ERA5 licence. `--check` verifies both
without downloading anything.
"""
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "era5"

AREA = [50.0, -125.0, 24.0, -66.0]          # N, W, S, E -- CONUS with a margin

# Every channel the rates may read is a named function of these, see
# asymode.weather.derive_channels. Nothing enters the model unnamed.
VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "instantaneous_10m_wind_gust",
    "2m_temperature",
    "2m_dewpoint_temperature",
    "total_precipitation",
    "snowfall",
    "convective_available_potential_energy",
    "volumetric_soil_water_layer_1",
    "total_cloud_cover",
    "surface_pressure",
]


def windows(pad_days: int = 1):
    """Storm windows from the onset audit, padded so lags have history."""
    import pandas as pd
    audit = json.loads((ROOT / "results/panel_onset_audit.json").read_text())
    out = []
    for d in audit["days"]:
        t0, t1 = pd.Timestamp(d["window"][0]), pd.Timestamp(d["window"][1])
        out.append((d["event_day"], t0 - pd.Timedelta(days=pad_days),
                    t1 + pd.Timedelta(days=pad_days)))
    return out


def check() -> int:
    """Verify credentials and licence without downloading a field."""
    rc = Path.home() / ".cdsapirc"
    if not rc.exists():
        print("~/.cdsapirc missing"); return 1
    body = rc.read_text()
    if "PASTE_YOUR_TOKEN_HERE" in body:
        print("~/.cdsapirc still has the placeholder token -- replace it"); return 1
    import cdsapi
    try:
        c = cdsapi.Client(wait_until_complete=False)
    except Exception as e:
        print(f"client init failed: {e}"); return 1
    print("credentials parsed. submitting a 1-hour, 1-variable probe request...")
    try:
        r = c.retrieve("reanalysis-era5-single-levels", {
            "product_type": ["reanalysis"], "variable": ["2m_temperature"],
            "year": ["2022"], "month": ["06"], "day": ["17"], "time": ["12:00"],
            "area": [41.0, -88.0, 40.0, -87.0], "data_format": "netcdf",
            "download_format": "unarchived"})
        print(f"accepted: request id {getattr(r, 'request_uid', '?')}")
        print("authentication and licence are OK.")
        return 0
    except Exception as e:
        msg = str(e)
        print(f"request rejected: {msg[:400]}")
        if "licence" in msg.lower() or "403" in msg or "required licences" in msg.lower():
            print("\n-> the ERA5 licence has not been accepted. Open\n"
                  "   https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download\n"
                  "   and accept the terms at the bottom of the page.")
        return 1


def fetch(pad_days: int, force: bool):
    import cdsapi
    RAW.mkdir(parents=True, exist_ok=True)
    c = cdsapi.Client()
    for day, t0, t1 in windows(pad_days):
        dst = RAW / f"era5_{day}.nc"
        if dst.exists() and not force:
            print(f"skip {dst.name} ({dst.stat().st_size/2**20:.0f} MB)"); continue
        days = [d for d in _daterange(t0, t1)]
        req = {
            "product_type": ["reanalysis"], "variable": VARIABLES,
            "year": sorted({d[:4] for d in days}),
            "month": sorted({d[5:7] for d in days}),
            "day": sorted({d[8:10] for d in days}),
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": AREA, "data_format": "netcdf", "download_format": "unarchived",
        }
        print(f"\n{day}: {t0.date()} .. {t1.date()}  ({len(days)} days x 24 h x "
              f"{len(VARIABLES)} vars)")
        t = time.time()
        c.retrieve("reanalysis-era5-single-levels", req, str(dst))
        print(f"  -> {dst.name}  {dst.stat().st_size/2**20:.0f} MB  "
              f"{time.time()-t:.0f}s")


def _daterange(t0, t1):
    import pandas as pd
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(t0, t1, freq="D")]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--pad-days", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    sys.exit(check() if a.check else (fetch(a.pad_days, a.force) or 0))
