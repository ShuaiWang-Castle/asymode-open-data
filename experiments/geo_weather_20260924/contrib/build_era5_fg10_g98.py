"""Local wind climatology for the Klawa-Ulbrich exceedance.

Prepared by the input contributor; the download has NOT been run.

Output: data/interim/geo_weather/era5_fg10_g98.npz
  g98        [105, 237]     98th percentile of the daily maximum of ERA5 10 m gust, 2010-2019 (m s-1)
  g98_season [4, 105, 237]  same, by season DJF, MAM, JJA, SON (month of the day, pooled over the ten years)
  lat [105] (50 -> 24), lon [237] (-125 -> -66), years [10]
  cell index: i = rint((50 - lat) / 0.25), j = rint((lon + 125) / 0.25)

Route: CDS 'derived-era5-single-levels-daily-statistics'.
  - The server computes the daily maximum of the hourly '10m_wind_gust_since_previous_post_processing'
    (frequency 1_hourly, time_zone utc+00:00).
  - 120 monthly requests of about 3 MB each, about 0.36 GB in total.
  - The anonymous ARCO-ERA5 route would move about 280 GB (87,648 global hourly chunks of about 3.2 MB each).
Needs a CDS account (~/.cdsapirc) that has accepted the ERA5 licence.
Idempotent: monthly files that already exist are skipped.
Each download appends one line to experiments/geo_weather_20260924/data_provenance/downloads.jsonl.

Day = UTC calendar day. The 00 UTC value of 10fg is the maximum over 23-00 UTC of the previous day.
Percentile: numpy default (linear interpolation).

Usage:
  python build_era5_fg10_g98.py --selftest   # synthetic check of the percentile code, no network
  python build_era5_fg10_g98.py --fetch      # download the missing monthly files
  python build_era5_fg10_g98.py --build      # needs all 120 files
"""
import calendar
import datetime as dt
import hashlib
import json
import pathlib
import sys
import zipfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
RAW = ROOT / "data/raw/era5_fg10_daily"
OUT = ROOT / "data/interim/geo_weather/era5_fg10_g98.npz"
PROV = ROOT / "experiments/geo_weather_20260924/data_provenance/downloads.jsonl"
AREA = [50, -125, 24, -66]
YEARS = list(range(2010, 2020))
LAT = 50 - 0.25 * np.arange(105)
LON = -125 + 0.25 * np.arange(237)
SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
DATASET = "derived-era5-single-levels-daily-statistics"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def month_file(y, m):
    return RAW / f"era5_fg10_dmax_{y}{m:02d}.nc"


def fetch():
    import cdsapi
    c = cdsapi.Client()
    RAW.mkdir(parents=True, exist_ok=True)
    for y in YEARS:
        for m in range(1, 13):
            p = month_file(y, m)
            if p.exists():
                continue
            req = {"product_type": "reanalysis", "variable": ["10m_wind_gust_since_previous_post_processing"],
                   "year": str(y), "month": [f"{m:02d}"],
                   "day": [f"{d:02d}" for d in range(1, calendar.monthrange(y, m)[1] + 1)],
                   "daily_statistic": "daily_maximum", "time_zone": "utc+00:00", "frequency": "1_hourly", "area": AREA}
            tmp = p.with_suffix(".part")
            c.retrieve(DATASET, req, str(tmp))
            if tmp.read_bytes()[:2] == b"PK":  # zipped response: keep the single netCDF inside
                with zipfile.ZipFile(tmp) as z:
                    nc = [n for n in z.namelist() if n.endswith(".nc")]
                    assert len(nc) == 1, nc
                    p.write_bytes(z.read(nc[0]))
                tmp.unlink()
            else:
                tmp.rename(p)
            with open(PROV, "a") as f:
                f.write(json.dumps({"file": str(p.relative_to(ROOT)), "source": f"Copernicus Climate Data Store, {DATASET}",
                                    "request": req, "downloaded_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                                    "bytes": p.stat().st_size, "sha256": sha256(p)}) + "\n")
            print("fetched", p.name, flush=True)


def climatology(x, months):
    """x [n_days, 105, 237] daily maxima; months [n_days] month of each day -> (g98, g98_season, n_season)."""
    g98 = np.nanpercentile(x, 98, axis=0).astype(np.float32)
    sel = [np.isin(months, ms) for ms in SEASONS.values()]
    gs = np.stack([np.nanpercentile(x[s], 98, axis=0) for s in sel]).astype(np.float32)
    return g98, gs, np.array([int(s.sum()) for s in sel])


def load_month(p):
    import xarray as xr
    with xr.open_dataset(p) as ds:
        name = [v for v in ds.data_vars if ds[v].ndim == 3]
        assert len(name) == 1, (p.name, list(ds.data_vars))
        da = ds[name[0]]
        tdim = [d for d in da.dims if d not in ("latitude", "longitude")][0]
        lat = da["latitude"].values
        lon = da["longitude"].values
        lon = np.where(lon > 180, lon - 360, lon)
        i = np.rint((50 - lat) / 0.25).astype(int)
        j = np.rint((lon + 125) / 0.25).astype(int)
        assert sorted(i) == list(range(105)) and sorted(j) == list(range(237)), f"{p.name}: grid is not the 0.25 deg CONUS box"
        v = da.transpose(tdim, "latitude", "longitude").values.astype(np.float32)
        out = np.empty_like(v)
        out[:, i[:, None], j[None, :]] = v  # reorder onto lat 50->24, lon -125->-66
        return out, np.asarray(da[tdim].values, dtype="datetime64[D]")


def build():
    missing = [month_file(y, m).name for y in YEARS for m in range(1, 13) if not month_file(y, m).exists()]
    if missing:
        sys.exit(f"{len(missing)} monthly files missing, e.g. {missing[:3]}; run --fetch first")
    xs, days = zip(*(load_month(month_file(y, m)) for y in YEARS for m in range(1, 13)))
    x, days = np.concatenate(xs), np.concatenate(days)
    assert len(days) == len(np.unique(days)) == 3652, len(days)
    months = days.astype("datetime64[M]").astype(int) % 12 + 1
    g98, gs, n_season = climatology(x, months)
    assert np.isfinite(g98).all() and 3 < np.nanmin(g98) and np.nanmax(g98) < 80, (np.nanmin(g98), np.nanmax(g98))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, g98=g98, g98_season=gs, lat=LAT.astype(np.float32), lon=LON.astype(np.float32),
                        years=np.array(YEARS), season_names=np.array(list(SEASONS)), n_days=np.array(len(days)),
                        n_days_season=n_season)
    print(f"wrote {OUT.relative_to(ROOT)}: g98 {np.nanmin(g98):.1f}-{np.nanmax(g98):.1f} m/s, days {len(days)}, by season {n_season.tolist()}")


def selftest():
    rng = np.random.default_rng(0)
    days = np.arange("2010-01-01", "2020-01-01", dtype="datetime64[D]")
    months = days.astype("datetime64[M]").astype(int) % 12 + 1
    x = rng.gamma(4.0, 3.0, (len(days), 105, 237)).astype(np.float32) + (months == 1)[:, None, None] * 10
    g98, gs, n = climatology(x, months)
    assert g98.shape == (105, 237) and gs.shape == (4, 105, 237) and n.sum() == len(days) == 3652
    assert np.allclose(g98[3, 7], np.percentile(x[:, 3, 7], 98)) and (gs[0] > gs[2]).mean() > 0.99  # DJF carries the January shift
    assert np.array_equal(np.rint((50 - LAT) / 0.25), np.arange(105)) and np.array_equal(np.rint((LON + 125) / 0.25), np.arange(237))
    print("selftest ok:", g98.shape, gs.shape, "days per season", n.tolist())


if __name__ == "__main__":
    args = set(sys.argv[1:]) or {"--selftest"}
    if "--selftest" in args:
        selftest()
    if "--fetch" in args:
        fetch()
    if "--build" in args:
        build()
