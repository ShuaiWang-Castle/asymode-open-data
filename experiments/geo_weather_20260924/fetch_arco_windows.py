"""ERA5 event windows from the public ARCO-ERA5 store (Google Cloud, anonymous HTTPS; Carver et al. 2023,
gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3), for when the CDS queue is congested.

The store holds ERA5 on the 0.25-degree grid, one chunk per variable and hour (721 x 1440, blosc-lz4). Each
chunk is fetched, decoded and cut to the panel's box (50..24 N, -125..-66 E), and the window is written as the
CDS files are (valid_time x latitude x longitude, CDS short names), so the panel builders read it unchanged:
  data/raw/era5/era5_<event>.nc       u10 v10 i10fg t2m d2m cape swvl1 tcc sp tp sf
  data/raw/era5_fg10/era5_<event>.nc  fg10
`--set extra` writes data/raw/era5_extra2/era5_<event>.nc (ptype, sd, lai_hv, cp, swvl2; ptype codes are kept as
stored, no interpolation). `--verify <event>` compares a few hours of every variable with an event already fetched from the CDS.
Provenance lines go to data_provenance/downloads.jsonl.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
MAIN = {"10m_u_component_of_wind": "u10", "10m_v_component_of_wind": "v10", "instantaneous_10m_wind_gust": "i10fg",
        "2m_temperature": "t2m", "2m_dewpoint_temperature": "d2m", "convective_available_potential_energy": "cape",
        "volumetric_soil_water_layer_1": "swvl1", "total_cloud_cover": "tcc", "surface_pressure": "sp",
        "total_precipitation": "tp", "snowfall": "sf"}
GUST = {"10m_wind_gust_since_previous_post_processing": "fg10"}
EXTRA = {"precipitation_type": "ptype", "snow_depth": "sd", "leaf_area_index_high_vegetation": "lai_hv",
         "convective_precipitation": "cp", "volumetric_soil_water_layer_2": "swvl2"}
R0, R1, C0, C1 = 160, 265, 940, 1177          # rows 50..24 N, columns 235..294 E (= -125..-66)
EPOCH = pd.Timestamp("1900-01-01")
_session = requests.Session()
_session.mount("https://", requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=32, max_retries=3))


def chunk(var: str, t: pd.Timestamp) -> np.ndarray:
    import numcodecs
    k = int((t - EPOCH) / pd.Timedelta(hours=1))
    r = _session.get(f"{BASE}/{var}/{k}.0.0", timeout=120)
    r.raise_for_status()
    a = np.frombuffer(numcodecs.Blosc().decode(r.content), dtype="<f4").reshape(721, 1440)
    return a[R0:R1, C0:C1].copy()


def window(varmap: dict, times: pd.DatetimeIndex, workers: int = 24) -> dict:
    jobs = [(v, t) for v in varmap for t in times]
    with ThreadPoolExecutor(workers) as ex:
        arrs = list(ex.map(lambda j: chunk(*j), jobs))
    out, i = {}, 0
    for v in varmap:
        out[varmap[v]] = np.stack(arrs[i:i + len(times)]); i += len(times)
    return out


def write(path: Path, data: dict, times: pd.DatetimeIndex, source_note: str):
    import xarray as xr
    lat = 50.0 - 0.25 * np.arange(R1 - R0); lon = -125.0 + 0.25 * np.arange(C1 - C0)
    ds = xr.Dataset({k: (("valid_time", "latitude", "longitude"), v.astype(np.float32)) for k, v in data.items()},
                    coords=dict(valid_time=times.values, latitude=lat, longitude=lon),
                    attrs=dict(source=source_note, institution="ECMWF ERA5 via ARCO-ERA5 (Google Research)"))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    ds.to_netcdf(tmp, engine="h5netcdf")
    tmp.rename(path)
    rec = dict(file=str(path.relative_to(ROOT)), source=f"ARCO-ERA5 {BASE}", variables=sorted(data),
               hours=[str(times[0]), str(times[-1])], downloaded_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
               bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    with open(HERE / "data_provenance" / "downloads.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


def verify(event: str, t0: str):
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from build_drivers import load_window
    ref = load_window(sorted((ROOT / "data/raw/era5").glob(f"era5_{event}*.nc"))[0])
    refg = load_window(sorted((ROOT / "data/raw/era5_fg10").glob(f"era5_{event}*.nc"))[0])
    times = pd.DatetimeIndex(ref["valid_time"].values)[[5, 30, 77]]
    a = window({**MAIN, **GUST}, times, workers=8)
    rep = {}
    for k, v in a.items():
        r = (refg if k == "fg10" else ref)[k].sel(valid_time=times).values
        rep[k] = dict(max_abs_diff=float(np.nanmax(np.abs(v - r))), ref_scale=float(np.nanmax(np.abs(r))))
    print(json.dumps(rep, indent=1))
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", nargs="*", default=[])
    ap.add_argument("--windows-file", default="experiments/geo_weather_20260924/selected_events_winter.json")
    ap.add_argument("--verify", default=None, help="an event already fetched from the CDS")
    ap.add_argument("--set", default="main", choices=["main", "extra"],
                    help="main = the panel's fields + hourly-maximum gust; extra = ptype, snow depth, LAI, cp, swvl2")
    a = ap.parse_args()
    if a.verify:
        rep = verify(a.verify, "")
        (HERE / "data_provenance" / "arco_verify.json").write_text(json.dumps(dict(event=a.verify, report=rep), indent=1) + "\n")
        return
    sel = {e["event"]: e for e in json.loads((ROOT / a.windows_file).read_text())["events"]}
    for ev in a.events:
        times = pd.date_range(pd.Timestamp(sel[ev]["window_start_utc"]), periods=216, freq="h")
        sets = ((MAIN, "era5"), (GUST, "era5_fg10")) if a.set == "main" else ((EXTRA, "era5_extra2"),)
        for varmap, sub in sets:
            dst = ROOT / "data" / "raw" / sub / f"era5_{ev}.nc"
            if list(dst.parent.glob(f"era5_{ev}*.nc")):
                print("exists", sub, ev, flush=True); continue
            t = dt.datetime.now()
            write(dst, window(varmap, times), times, "ARCO-ERA5 analysis-ready 0.25 deg, hourly")
            print("saved", sub, ev, f"{(dt.datetime.now() - t).total_seconds():.0f} s", flush=True)


if __name__ == "__main__":
    main()
