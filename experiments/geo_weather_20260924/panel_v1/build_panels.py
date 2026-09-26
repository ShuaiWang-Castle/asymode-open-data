"""Panels of DATASET_DESIGN v1 (step 4 of section 12) for the systems of one tranche (development D; the sealed tranche
C is built only by the confirmatory script). For each system and its sampled counties (county_sample_<tranche>.parquet):

1. the outage fraction over the 216 h window from EAGLE-I (explicit zero rows dropped; C's county-hours skipped,
   amendment 2 S8), the observation mask of the pre-window service rule (`panel.build_panel(service_rule="pre_window")`;
   a forecast hour without a national collection run, or with only blank counts, is masked, never a reason to drop a
   county), the 2024 modelled customers as denominator;
2. the round-1 drivers and neighbour drivers of build_panel216.py (area weights; five nearest counties by internal point,
   2020 vintage), written in the panel216 layout;
3. the round-2 host inputs of build_v3p.panel (hourly-maximum gust, cell-level hazards, support channels);
4. a cropped copy of the system's ERA5 fields (domain box + 1 degree; float16, pressure and precipitation float32,
   ptype int8) for later pathway features, after which
   the three CONUS raw files are deleted (section 8).
Outputs: data/interim/panel_v1/panels/{panel216_<system>.npz, panel216w_<system>.npz}, data/raw/era5_v1/<system>.nc.
usage: python build_panels.py --tranche D [--systems S00001 ...]"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OG = ROOT / "experiments" / "open_gcrk_20260919"
sys.path.insert(0, str(OG)); sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(ROOT / "src"))
import build_panel216 as BP  # noqa: E402
import build_v3p as V  # noqa: E402
from asymode.panel import attach_denominator, build_panel  # noqa: E402

import fetch_era5_v1 as FE  # noqa: E402
import gates as GT  # noqa: E402
from frame_common import EXP, INTERIM, OUT, PREFIX_H, HORIZON_H, RAW, counties  # noqa: E402

PANELS = OUT / "panels"
ARCHIVE = RAW / "era5_v1"
T = PREFIX_H + HORIZON_H


def area_weights() -> pd.DataFrame:
    """ERA5 area weights of build_panel216 (2023 boundaries) plus Connecticut's eight 2020 counties."""
    f = OUT / "era5_area_weights_v1.parquet"
    if f.exists():
        return pd.read_parquet(f)
    from asymode.weather import county_weights
    w = BP.all_weights()
    ct = [x for x in counties().fips if x.startswith("09") and x not in set(w.fips)]
    lats = np.arange(50.0, 24.0 - 1e-9, -0.25); lons = np.arange(-125.0, -66.0 + 1e-9, 0.25)
    import tempfile, zipfile
    with tempfile.TemporaryDirectory() as td:
        zipfile.ZipFile(RAW / "census" / "cb_2020_us_county_500k.zip").extractall(td)
        wc = county_weights(lats, lons, Path(td) / "cb_2020_us_county_500k.shp", fips=ct)
    w = pd.concat([w, wc], ignore_index=True)
    w.to_parquet(f, index=False)
    return w


def neighbours(k: int = 5) -> dict[str, list[str]]:
    c = counties()
    lat, lon, ids = c.lat.to_numpy(), c.lon.to_numpy(), c.fips.to_numpy()
    out = {}
    for i, f in enumerate(ids):
        d = np.sqrt((lat - lat[i]) ** 2 + ((lon - lon[i]) * np.cos(np.radians(lat[i]))) ** 2)
        d[i] = np.inf
        out[f] = [str(x) for x in ids[np.argsort(d)[:k]]]
    return out


def eaglei_window(w0: pd.Timestamp, w1: pd.Timestamp, cmask) -> pd.DataFrame:
    lo = w0 - pd.Timedelta(days=GT.LOOKBACK_D)
    parts = [pd.read_parquet(INTERIM / f"eaglei_outages_{y}.parquet", columns=["fips", "ts", "customers_out"],
                             filters=[("ts", ">=", lo), ("ts", "<", w1)]) for y in sorted({lo.year, w1.year})]
    df = pd.concat(parts, ignore_index=True)
    df["fips"] = df.fips.astype(str).str.zfill(5)
    df = df[(df.ts >= lo) & (df.ts < w1)]
    df = df[~(df.customers_out.fillna(-1) == 0)]
    return cmask.drop(df) if cmask is not None else df


def crop_archive(system: str, fips: list[str]) -> Path:
    """Float16 copy of the system's ERA5 fields over its domain box + 1 degree (section 8)."""
    import xarray as xr
    c = counties().set_index("fips").loc[fips]
    la0, la1, lo0, lo1 = c.lat.min() - 1.5, c.lat.max() + 1.5, c.lon.min() - 1.5, c.lon.max() + 1.5
    dst = ARCHIVE / f"{system}.nc"
    dst.parent.mkdir(parents=True, exist_ok=True)
    parts = []
    for p in FE.paths(system):
        ds = xr.open_dataset(p, engine="h5netcdf")
        parts.append(ds.sel(latitude=slice(la1, la0), longitude=slice(lo0, lo1)).load())
        ds.close()
    ds = xr.merge(parts, compat="override")
    wide = {"sp", "tp", "sf", "cp"}          # pressure overflows float16; small precipitation amounts need float32
    enc = {v: dict(dtype="float32" if v in wide else "float16", zlib=True, complevel=4)
           for v in ds.data_vars if v != "ptype"}
    enc["ptype"] = dict(dtype="int8", zlib=True, complevel=4, _FillValue=-1)
    ds = ds.assign(ptype=ds.ptype.fillna(-1).round())
    ds.to_netcdf(dst, engine="h5netcdf", encoding=enc)
    return dst


def build_one(s: str, sy, dr, cs, w, nn, denom, cmask) -> dict:
    t0 = pd.Timestamp(sy.at[s, "window_start"]); w1 = t0 + pd.Timedelta(hours=T)
    d = cs[(cs.system == s) & cs.sampled]
    fips = d.fips.tolist()
    df = eaglei_window(t0, w1, cmask)
    p = build_panel(df, t0, w1 - pd.Timedelta(minutes=15), fips=fips, service_rule="pre_window",
                    lookback_days=GT.LOOKBACK_D)
    p = attach_denominator(p, denom, "eaglei_2024_modelled")
    assert p["fips"] == fips, "a sampled county lost its denominator"
    y15 = np.where(p["observed"], p["y"], np.nan).reshape(len(fips), T, 4)
    obs = np.isfinite(y15).any(-1)
    with np.errstate(all="ignore"):
        y = np.where(obs, np.nanmean(y15, -1), np.nan)
    FE.fetch(s, t0)
    fields = BP.era5_fields(s)
    X = BP.drivers(fields, t0, fips, w)
    nbr = sorted({g for f in fips for g in nn[f]})
    Xn = BP.drivers(fields, t0, nbr, w)
    PANELS.mkdir(parents=True, exist_ok=True)
    r1 = dict(event=s, fips=np.array(fips), y=y.astype(np.float32), observed=obs, denominator=p["denominator"],
              X=X.astype(np.float32), channels=np.array(BP.CHANNELS + ["clock_sin", "clock_cos"]),
              ts=np.array(pd.date_range(t0, periods=T, freq="h").astype(str)), nbr_fips=np.array(nbr),
              X_nbr=Xn.astype(np.float32), nbr_of=np.array([nn[f] for f in fips]))
    np.savez_compressed(PANELS / f"panel216_{s}.npz", **r1)
    BP.OUT = PANELS                                   # build_v3p.panel reads the round-1 panel from BP.OUT
    r2 = V.panel(s, t0, w)
    np.savez_compressed(PANELS / f"panel216w_{s}.npz", **r2)
    crop_archive(s, sorted(set(cs[cs.system == s].fips)))
    for f in FE.paths(s):
        f.unlink()
    return dict(system=s, counties=len(fips), observed_share=float(obs[:, PREFIX_H:].mean()),
                prefix_observed_share=float(obs[:, :PREFIX_H].mean()),
                sha256=hashlib.sha256((PANELS / f"panel216w_{s}.npz").read_bytes()).hexdigest())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tranche", default="D")
    ap.add_argument("--systems", nargs="*", default=None)
    a = ap.parse_args()
    assert a.tranche != "C", "the sealed tranche is built only by the confirmatory script"
    sy = pd.read_parquet(OUT / "systems.parquet").set_index("system")
    dr = pd.read_parquet(OUT / "draws.parquet")
    cs = pd.read_parquet(OUT / f"county_sample_{a.tranche}.parquet")
    systems = a.systems or sorted(dr[dr.tranche == a.tranche].system)
    w = area_weights()
    nn = neighbours()
    denom = pd.read_parquet(INTERIM / "eaglei_county_customers_2024.parquet")["customers"]
    cmask = GT.CMask()
    log = EXP / "data_provenance" / f"panels_v1{a.tranche}.jsonl"
    for s in systems:
        if (PANELS / f"panel216w_{s}.npz").exists():
            continue
        rec = build_one(s, sy, dr, cs, w, nn, denom, cmask)
        with open(log, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(rec, flush=True)


if __name__ == "__main__":
    main()
