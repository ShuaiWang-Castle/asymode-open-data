"""ERA5 precipitation type for the twelve e3 windows.

Prepared by the input contributor; the download has NOT been run.

What:   ERA5 'precipitation_type' (instantaneous, GRIB code table 4.201) at 0.25 deg over the CONUS box
        [50, -125, 24, -66]. Each window is 216 h starting at window_start_utc in
        experiments/open_gcrk_20260919/selected_events_e3.json.
Route:  one CDS request per calendar month; a request's year/month/day sets are crossed.
        About 16 requests and about 0.26 GB.
Output: data/raw/era5_ptype/era5_<event>.nc
        Windows that cross a month boundary give .m0.nc and .m1.nc instead, the convention of data/raw/era5/.
Needs:  ~/.cdsapirc. Every download appends a line to experiments/geo_weather_20260924/data_provenance/downloads.jsonl.

Codes:
  0 none, 1 rain, 3 freezing rain, 5 snow, 6 wet snow, 7 rain-snow mix, 8 ice pellets, 12 freezing drizzle.

After download, --check counts values outside that set.
  - Non-code values mean the field was interpolated as if continuous; do not threshold them.
  - Aggregate categorical fields to counties by class shares, never by averaging codes.

Usage:
  python fetch_era5_ptype_windows.py --plan    # print the requests, no network
  python fetch_era5_ptype_windows.py --fetch
  python fetch_era5_ptype_windows.py --check   # needs xarray
"""
import datetime as dt
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
EVENTS = json.loads((ROOT / "experiments/open_gcrk_20260919/selected_events_e3.json").read_text())["events"]
OUT = ROOT / "data/raw/era5_ptype"
PROV = ROOT / "experiments/geo_weather_20260924/data_provenance/downloads.jsonl"
AREA = [50, -125, 24, -66]
CODES = {0, 1, 3, 5, 6, 7, 8, 12}
HOURS = 216


def plan():
    """[(event, path, year, month, days)] covering HOURS hours from each window start."""
    out = []
    for e in EVENTS:
        t0 = dt.datetime.fromisoformat(e["window_start_utc"])
        groups = {}
        for h in range(HOURS):
            t = t0 + dt.timedelta(hours=h)
            groups.setdefault((t.year, t.month), set()).add(t.day)
        for k, ((y, m), days) in enumerate(sorted(groups.items())):
            name = f"era5_{e['event']}.nc" if len(groups) == 1 else f"era5_{e['event']}.m{k}.nc"
            out.append((e["event"], OUT / name, y, m, sorted(days)))
    return out


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fetch():
    import cdsapi
    c = cdsapi.Client()
    OUT.mkdir(parents=True, exist_ok=True)
    for event, p, y, m, days in plan():
        if p.exists():
            continue
        req = {"product_type": ["reanalysis"], "variable": ["precipitation_type"], "year": [str(y)], "month": [f"{m:02d}"],
               "day": [f"{d:02d}" for d in days], "time": [f"{h:02d}:00" for h in range(24)], "area": AREA,
               "data_format": "netcdf", "download_format": "unarchived"}
        tmp = p.with_suffix(".part")
        c.retrieve("reanalysis-era5-single-levels", req, str(tmp))
        tmp.rename(p)
        with open(PROV, "a") as f:
            f.write(json.dumps({"file": str(p.relative_to(ROOT)), "source": "Copernicus Climate Data Store, reanalysis-era5-single-levels",
                                "event": event, "request": req, "downloaded_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                                "bytes": p.stat().st_size, "sha256": sha256(p)}) + "\n")
        print("fetched", p.name, flush=True)


def check():
    import numpy as np
    import xarray as xr
    for event, p, y, m, days in plan():
        with xr.open_dataset(p) as ds:
            v = ds[[n for n in ds.data_vars if ds[n].ndim == 3][0]].values
        ok = np.isin(v, list(CODES)) | np.isnan(v)
        n_steps = v.shape[0]
        print(f"{p.name}: steps {n_steps} (expected {24 * len(days)}), non-code values {100 * (1 - ok.mean()):.3f}%, "
              f"classes present {sorted(set(np.unique(v[ok & ~np.isnan(v)]).astype(int)))}")


if __name__ == "__main__":
    args = set(sys.argv[1:]) or {"--plan"}
    if "--plan" in args:
        P = plan()
        for event, p, y, m, days in P:
            print(f"{event}: {p.name:28s} {y}-{m:02d} days {days[0]:02d}-{days[-1]:02d} ({len(days)} d)")
        print(f"{len(P)} requests for {len(EVENTS)} windows")
    if "--fetch" in args:
        fetch()
    if "--check" in args:
        check()
