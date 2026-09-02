"""ERA5 fields -> county driver series aligned to each storm panel.

The archive the CDS returns is a zip of two netCDF files: instantaneous fields and
accumulated ones (precipitation, snowfall) are delivered as separate streams. They
share a grid and a time axis, so they are merged on read.

Accumulated variables need care. ERA5 reports them as a total over the preceding
hour, so they are already a rate per hour and are used as such -- but they are
NOT differenced, and treating them as running totals would be wrong.
"""
import argparse, io, re, sys, zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from asymode.weather import apply_weights, derive_channels   # noqa: E402

RAW = ROOT / "data" / "raw" / "era5"
INTERIM = ROOT / "data" / "interim"


def load_window(path: Path):
    """Merge the instantaneous and accumulated streams into one dataset."""
    import xarray as xr
    if zipfile.is_zipfile(path):
        z = zipfile.ZipFile(path)
        parts = [xr.open_dataset(io.BytesIO(z.read(m)), engine="h5netcdf")
                 for m in z.infolist() if m.filename.endswith(".nc")]
        ds = xr.merge(parts, compat="override")
    else:
        ds = xr.open_dataset(path, engine="h5netcdf")
    for c in ("number", "expver"):
        if c in ds.coords:
            ds = ds.drop_vars(c)
    return ds.squeeze()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    w = pd.read_parquet(INTERIM / "era5_county_weights.parquet")
    files = sorted(RAW.glob("era5_*.nc"))
    if not files:
        sys.exit("no ERA5 windows downloaded yet")
    # A window that crossed a month boundary is fetched as several archives,
    # `era5_<day>.m0.nc`, `era5_<day>.m1.nc`, ...  Group every archive by its
    # storm day so the pieces are merged into one window rather than mistaken
    # for windows of their own.
    by_day = {}
    for f in files:
        day = re.sub(r"\.m\d+$", "", f.stem.replace("era5_", ""))
        by_day.setdefault(day, []).append(f)
    print(f"{len(files)} ERA5 archives -> {len(by_day)} windows, weight table {len(w):,} pairs")

    for day, pieces in sorted(by_day.items()):
        panel_f = INTERIM / f"panel_{day}.npz"
        dst = INTERIM / f"drivers_{day}.npz"
        if not panel_f.exists():
            print(f"  {day}: no matching panel, skipped"); continue
        if dst.exists() and not a.force:
            print(f"  {day}: exists, skipped"); continue

        import xarray as xr
        ds = (load_window(pieces[0]) if len(pieces) == 1
              else xr.concat([load_window(pc) for pc in pieces], dim="valid_time")
                     .sortby("valid_time"))
        ch = derive_channels(ds)
        names = sorted(ch)
        times = pd.to_datetime(ds["valid_time"].values)

        panel = np.load(panel_f, allow_pickle=True)
        fips = panel["fips"].tolist()
        p_ts = pd.to_datetime(panel["ts"])
        # The panel is 15-minute; the drivers are hourly. Align on the hourly grid
        # the evaluation uses, taking the panel's hours as the reference.
        p_hours = pd.DatetimeIndex(p_ts[::4]).floor("h")

        arrs = []
        for n in names:
            county = apply_weights(np.asarray(ch[n], dtype=np.float32), w, fips)  # (C, T)
            s = pd.DataFrame(county.T, index=times).reindex(p_hours)
            if s.isna().any().any():
                s = s.interpolate(limit_direction="both")
            arrs.append(s.to_numpy(dtype=np.float32).T)
        X = np.stack(arrs, axis=-1)                                   # (C, T, F)
        np.savez_compressed(dst, X=X, channels=np.array(names),
                            fips=np.array(fips), ts=np.array(p_hours.astype(str)))
        print(f"  {day}: {X.shape[0]} counties x {X.shape[1]} hours x "
              f"{X.shape[2]} channels -> {dst.name}")
        for n, k in zip(names, range(X.shape[2])):
            v = X[:, :, k]
            print(f"      {n:<14} mean {v.mean():>10.3f}  p99 {np.percentile(v,99):>10.3f}  "
                  f"max {v.max():>10.3f}")


if __name__ == "__main__":
    main()
