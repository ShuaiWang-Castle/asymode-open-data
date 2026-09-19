"""Round-2 weather for the 216-hour panels (PREREG Amendment 2, R2 (a)-(b)).

For every event the county set, outage series, denominators and neighbour lists are copied from
the round-1 panel (`panel216_<event>.npz`; the data gates do not depend on weather). Only the
weather is rebuilt:

  X      [C, 216, 14]  round-1 channels, with gust = county area mean of the ERA5 hourly maximum
                       gust ('10m_wind_gust_since_previous_post_processing', data/raw/era5_fg10/)
  Hg     [C, 216, 5]   gust exceedance energy [g - 15]_+^2, wet wind [g - 12]_+ p (1 + soil),
                       near-freeze exp(-(t/3.5)^2), snow-ice load (s + p [t <= 0.5]) nf (0.25 + rh/100)
                       and cold precipitation p nf, each computed on every ERA5 cell and then
                       area-averaged over the county (round 1 computed them from county means)
  S      [C, 216, 2]   the county's highest cell gust, and the share of its area whose cell gust
                       exceeds 15 m/s
  X_nbr  [N, 216, 14]  as X for the five-nearest-county neighbours

Output: data/interim/open_gcrk/panel216r2_<event>.npz; checksums in
data_provenance/panel216r2_checksums.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import build_panel216 as BP  # noqa: E402
from asymode.weather import derive_channels  # noqa: E402
from build_drivers import load_window  # noqa: E402

FG10 = ROOT / "data" / "raw" / "era5_fg10"
T = BP.T


def fields_r2(event: str, t0: pd.Timestamp):
    """Cell-level channels for the 216 window hours with gust = the hourly maximum gust."""
    import xarray as xr
    main = [p for p in sorted(BP.RAW_ERA5.glob(f"era5_{event}*.nc")) if not p.name.endswith(".part")]
    gust = [p for p in sorted(FG10.glob(f"era5_{event}*.nc")) if not p.name.endswith(".part")]
    if not main or not gust:
        raise FileNotFoundError(f"ERA5 files missing for {event}: main {len(main)}, fg10 {len(gust)}")
    cat = lambda ps: (load_window(ps[0]) if len(ps) == 1 else
                      xr.concat([load_window(p) for p in ps], dim="valid_time").sortby("valid_time"))
    ds, dg = cat(main), cat(gust)
    ds = ds.drop_vars([v for v in ("i10fg",) if v in ds.data_vars])
    ds = ds.sel(valid_time=slice(t0, t0 + pd.Timedelta(hours=T - 1)))
    dg = dg.reindex(valid_time=ds["valid_time"])
    if "fg10" not in dg.data_vars:
        raise KeyError(f"{event}: no fg10 in {[p.name for p in gust]}: {list(dg.data_vars)}")
    if not np.isfinite(dg["fg10"].values).all():
        raise ValueError(f"{event}: hourly maximum gust missing for some window hours or cells")
    ds["fg10"] = dg["fg10"]
    ch = derive_channels(ds)
    assert "gust" in ch and np.array_equal(ch["gust"], ds["fg10"].values, equal_nan=True)
    return pd.to_datetime(ds["valid_time"].values), ch, [p.name for p in main + gust]


def cell_hazards(ch: dict) -> dict:
    g = ch["gust"].astype(np.float64)
    p = np.clip(ch["precip"], 0, None).astype(np.float64)
    s = np.clip(ch["snowfall"], 0, None).astype(np.float64)
    soil = np.clip(np.nan_to_num(ch["soil_moisture"], nan=0.0), 0, None).astype(np.float64)
    t2, rh = ch["t2m_c"].astype(np.float64), ch["rh"].astype(np.float64)
    nf = np.exp(-(t2 / 3.5) ** 2)
    return {"gust_excess_energy": np.clip(g - 15.0, 0, None) ** 2,
            "wet_wind": np.clip(g - 12.0, 0, None) * p * (1.0 + soil),
            "near_freeze": nf,
            "snow_ice_load": (s + p * (t2 <= 0.5)) * nf * (0.25 + rh / 100.0),
            "cold_precip": p * nf}


def to_hours(county: np.ndarray, times, t0) -> np.ndarray:
    hours = pd.date_range(t0, periods=T, freq="h")
    s = pd.DataFrame(county.T, index=times)
    s = s[~s.index.duplicated()].reindex(hours)
    if s.isna().any().any():
        raise ValueError("ERA5 does not cover every window hour")
    return s.to_numpy(np.float64).T


def support(g: np.ndarray, w: pd.DataFrame, fips: list[str], times, t0) -> np.ndarray:
    """(C, 216, 2): highest cell gust and the gust > 15 m/s area share of each county."""
    out = np.zeros((len(fips), len(times), 2))
    sub = w[w.fips.isin(set(fips))]
    pos = {f: k for k, f in enumerate(fips)}
    for f, grp in sub.groupby("fips"):
        v = g[:, grp["i"].to_numpy(), grp["j"].to_numpy()]
        out[pos[f], :, 0] = v.max(1)
        out[pos[f], :, 1] = (v > 15.0).astype(float) @ grp["w"].to_numpy()
    return np.stack([to_hours(out[..., k], times, t0) for k in range(2)], -1)


def build(event: str, t0: pd.Timestamp, w: pd.DataFrame) -> dict:
    r1 = dict(np.load(BP.OUT / f"panel216_{event}.npz", allow_pickle=True))   # our own round-1 file
    r1["ts"] = np.asarray(r1["ts"]).astype(str)
    fips, nbr = [str(f) for f in r1["fips"]], [str(f) for f in r1["nbr_fips"]]
    fields = fields_r2(event, t0)
    times, ch, names = fields
    X = BP.drivers(fields, t0, fips, w)
    Xn = BP.drivers(fields, t0, nbr, w)
    hz = cell_hazards(ch)
    Hg = np.stack([to_hours(BP.apply_weights(hz[k].astype(np.float32), w, fips).astype(np.float64), times, t0)
                   for k in ("gust_excess_energy", "wet_wind", "near_freeze", "snow_ice_load", "cold_precip")], -1)
    S = support(ch["gust"].astype(np.float64), w, fips, times, t0)
    k = list(r1["channels"]).index("gust")
    keep = [c for c in range(X.shape[-1]) if c != k]
    assert np.allclose(X[..., keep], r1["X"][..., keep], atol=1e-4), "non-gust channels must equal round 1"
    return dict(event=event, fips=r1["fips"], y=r1["y"], observed=r1["observed"], denominator=r1["denominator"],
                X=X.astype(np.float32), channels=r1["channels"], ts=r1["ts"], nbr_fips=r1["nbr_fips"],
                X_nbr=Xn.astype(np.float32), nbr_of=r1["nbr_of"], Hg=Hg.astype(np.float32), S=S.astype(np.float32),
                hazard_names=np.array(["gust_excess_energy", "wet_wind", "near_freeze", "snow_ice_load", "cold_precip"]),
                support_names=np.array(["gust_cell_max", "gust_exceed15_share"]), era5_files=np.array(names),
                gust_r1_mean=float(r1["X"][..., k].mean()), gust_r2_mean=float(X[..., k].mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", nargs="+", default=None, help="default: the five pre-registered events")
    a = ap.parse_args()
    sel = {e["event"]: e for f in ("selected_events.json", "selected_events_e3.json")
           for e in json.loads((HERE / f).read_text())["events"]}
    cand = pd.read_csv(HERE / "event_selection.csv").set_index("day")
    events = a.events or list(sel)
    w = BP.all_weights()
    prov = []
    for ev in events:
        t0 = pd.Timestamp(sel[ev]["window_start_utc"] if ev in sel else cand.loc[ev, "window_start_utc"])
        d = build(ev, t0, w)
        f = BP.OUT / f"panel216r2_{ev}.npz"
        np.savez_compressed(f, **d)
        prov.append(dict(file=str(f.relative_to(ROOT)), sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
                         bytes=f.stat().st_size, era5_files=list(d["era5_files"])))
        print(ev, len(d["fips"]), f"mean county gust: instantaneous {d['gust_r1_mean']:.2f} -> hourly max "
              f"{d['gust_r2_mean']:.2f} m/s; hazard means", np.round(d["Hg"].mean((0, 1)), 4),
              "support means", np.round(d["S"].mean((0, 1)), 3), flush=True)
    p = HERE / "data_provenance" / "panel216r2_checksums.json"
    old = json.loads(p.read_text()) if p.exists() else []
    keep = [r for r in old if r["file"] not in {x["file"] for x in prov}]
    p.write_text(json.dumps(keep + prov, indent=1) + "\n")


if __name__ == "__main__":
    main()
