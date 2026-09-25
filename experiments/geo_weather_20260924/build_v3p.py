"""Data v3p: the round-2 inputs of the twelve-event panel with population weights instead of area weights
(REVIEW_formal L1w': where the hazard meets the customers; no new parameter).

1. Population weights w(fips, i, j) over ERA5 0.25-degree cells: WorldPop 2020 1 km people inside the county
   (Census 2023 polygons, rasterised on the WorldPop grid) and inside the cell, plus 1% of the county's
   population spread by the area weights (so a cell of the county without people keeps a small weight);
   normalised per county. Every CONUS county, so the neighbour drivers of the recovery inputs change too.
2. Panels: the round-2 builder (open_gcrk_20260919/build_panel216_r2.py) with these weights: 14 channels,
   cell-level hazards and support channels, neighbour drivers; outage series, denominators, neighbour lists
   copied from the round-1 panels as before.
3. Features: build_features_r2.py unchanged in content, reading these panels.
Unit order, targets and masks equal features_e3r2.npz (asserted), so its splits apply unchanged.
Outputs: data/interim/geo_weather/{era5_county_popweights_conus.parquet, panel216v3p_<event>.npz, features_v3p.npz}.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OG = ROOT / "experiments" / "open_gcrk_20260919"
sys.path.insert(0, str(OG))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import build_features as BF  # noqa: E402
import build_features_r2 as BR  # noqa: E402
import build_panel216 as BP  # noqa: E402
import build_panel216_r2 as P2  # noqa: E402

OUT = ROOT / "data" / "interim" / "geo_weather"
POP = ROOT / "data" / "raw" / "population" / "usa_ppp_2020_1km_Aggregated.tif"
LAT0, LON0, STEP, NLAT, NLON = 50.0, -125.0, 0.25, 105, 237
EPS = 0.01


def pop_weights() -> pd.DataFrame:
    f = OUT / "era5_county_popweights_conus.parquet"
    if f.exists():
        return pd.read_parquet(f)
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize
    from rasterio.windows import from_bounds
    area = BP.all_weights()
    shp = gpd.read_file(BP.SHP)
    shp = shp[shp.GEOID.isin(set(area.fips))].to_crs(4326).reset_index(drop=True)
    with rasterio.open(POP) as src:
        win = from_bounds(-125.25, 23.75, -65.75, 50.25, src.transform).round_offsets().round_lengths()
        pop = src.read(1, window=win).astype(np.float64)
        tr = src.window_transform(win)
    pop = np.where(pop > 0, pop, 0.0)
    ids = rasterize(((g, k + 1) for k, g in enumerate(shp.geometry)), out_shape=pop.shape, transform=tr,
                    fill=0, dtype="int32", all_touched=False)
    rr, cc = np.nonzero((ids > 0) & (pop > 0))
    lon = tr.c + (cc + 0.5) * tr.a; lat = tr.f + (rr + 0.5) * tr.e
    ci = np.rint((LAT0 - lat) / STEP).astype(int); cj = np.rint((lon - LON0) / STEP).astype(int)
    d = pd.DataFrame(dict(fips=shp.GEOID.to_numpy()[ids[rr, cc] - 1], i=ci, j=cj, pop=pop[rr, cc]))
    d = d.groupby(["fips", "i", "j"], as_index=False)["pop"].sum()
    m = area.merge(d, on=["fips", "i", "j"], how="outer").fillna({"w": 0.0, "pop": 0.0})
    tot = m.groupby("fips")["pop"].transform("sum")
    m["w"] = m["pop"] + EPS * tot * m["w"] + 1e-9 * m["w"]
    m = m[m.w > 0]
    m["w"] = m["w"] / m.groupby("fips")["w"].transform("sum")
    m = m[["fips", "i", "j", "w"]].sort_values(["fips", "i", "j"]).reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    m.to_parquet(f, index=False)
    lost = sorted(set(area.fips) - set(m.fips))
    print("pop weights:", m.fips.nunique(), "counties;", len(m), "county-cells; outside-cell pixels merged:",
          int((~d.set_index(["fips", "i", "j"]).index.isin(area.set_index(["fips", "i", "j"]).index)).sum()),
          "; counties lost:", lost[:5], flush=True)
    return m


def panel(event: str, t0: pd.Timestamp, w: pd.DataFrame) -> dict:
    r1 = dict(np.load(BP.OUT / f"panel216_{event}.npz", allow_pickle=True))
    r1["ts"] = np.asarray(r1["ts"]).astype(str)
    fips, nbr = [str(f) for f in r1["fips"]], [str(f) for f in r1["nbr_fips"]]
    fields = P2.fields_r2(event, t0)
    times, ch, names = fields
    X = BP.drivers(fields, t0, fips, w)
    Xn = BP.drivers(fields, t0, nbr, w)
    hz = P2.cell_hazards(ch)
    Hg = np.stack([P2.to_hours(BP.apply_weights(hz[k].astype(np.float32), w, fips).astype(np.float64), times, t0)
                   for k in ("gust_excess_energy", "wet_wind", "near_freeze", "snow_ice_load", "cold_precip")], -1)
    S = P2.support(ch["gust"].astype(np.float64), w, fips, times, t0)
    return dict(event=event, fips=r1["fips"], y=r1["y"], observed=r1["observed"], denominator=r1["denominator"],
                X=X.astype(np.float32), channels=r1["channels"], ts=r1["ts"], nbr_fips=r1["nbr_fips"],
                X_nbr=Xn.astype(np.float32), nbr_of=r1["nbr_of"], Hg=Hg.astype(np.float32), S=S.astype(np.float32),
                era5_files=np.array(names))


def features(events: list[dict], prefix: str = "panel216v3p") -> dict:
    geo = pd.read_parquet(BF.OUT / "geography_e3.parquet")
    geo_cols = [c for c in geo.columns if c != "n_land_pixels"]
    geo = geo.join(pd.read_parquet(BF.OUT / "geography_ext_e3.parquet")[BF.GEO_EXT], how="left")
    geo_cols = geo_cols + BF.GEO_EXT
    parts = {k: [] for k in ("xu", "xr", "xo", "geo", "y0", "y", "m", "y_full", "obs_full", "cust", "fips", "event")}
    for e in events:                                   # as build_features_r2.main, from the v3p panels
        z = np.load(OUT / f"{prefix}_{e['event']}.npz")
        fips = [str(f) for f in z["fips"]]
        X, Xn = z["X"].astype(np.float64), z["X_nbr"].astype(np.float64)
        assert list(z["channels"]) == BF.CH
        wf = BR.weather_features_r2(X, z["Hg"], z["S"])
        xu = np.concatenate([X, np.stack([wf[k] for k in BR.DAMAGE_R2[len(BF.CH):]], -1)], -1)
        pos = {str(g): i for i, g in enumerate(z["nbr_fips"])}
        nb = np.stack([[pos[str(g)] for g in row] for row in z["nbr_of"]])
        nbr = []
        for c in ("gust", "wind_speed", "precip", "soil_moisture"):
            v = Xn[nb, :, BF.CH.index(c)]
            nbr += [v.mean(1), v.max(1)]
        st = BF.statics(fips)
        hist = BF.hist_features(z["y"], z["observed"])
        xr = np.concatenate([X, np.broadcast_to(st[:, None], (len(fips), BF.T, st.shape[1])),
                             np.broadcast_to(hist[:, None], (len(fips), BF.T, hist.shape[1])), np.stack(nbr, -1)], -1)
        xo = np.stack([wf[k] for k in BF.HAZARD], -1)
        y, obs = z["y"].astype(np.float64), z["observed"]
        parts["xu"].append(xu); parts["xr"].append(xr); parts["xo"].append(xo)
        parts["geo"].append(geo.reindex(fips)[geo_cols].to_numpy(float))
        parts["y0"].append(y[:, 71]); parts["y"].append(np.nan_to_num(y[:, BF.ORIGIN:]))
        parts["m"].append(obs[:, BF.ORIGIN:].astype(np.float64))
        parts["y_full"].append(y); parts["obs_full"].append(obs)
        parts["cust"].append(z["denominator"]); parts["fips"] += fips; parts["event"] += [e["event"]] * len(fips)
    arr = {k: (np.concatenate(v).astype(np.float32) if k not in ("fips", "event", "obs_full") else
               (np.concatenate(v) if k == "obs_full" else np.array(v))) for k, v in parts.items()}
    assert np.isfinite(arr["xu"]).all() and np.isfinite(arr["xo"]).all() and np.isfinite(arr["y0"]).all()
    return dict(**arr, damage_features=np.array(BR.DAMAGE_R2), recovery_features=np.array(BF.RECOVERY_FEATURES),
                occurrence_features=np.array(BF.HAZARD), geo_features=np.array(geo_cols), weather_channels=np.array(BF.CH))


def main():
    w = pop_weights()
    events = json.loads((OG / "selected_events_e3.json").read_text())["events"]
    for e in events:
        f = OUT / f"panel216v3p_{e['event']}.npz"
        if not f.exists():
            np.savez_compressed(f, **panel(e["event"], pd.Timestamp(e["window_start_utc"]), w))
        print("panel", e["event"], flush=True)
    arr = features(events)
    ref = np.load(ROOT / "data" / "interim" / "open_gcrk" / "features_e3r2.npz")
    for k in ("fips", "event", "y", "m", "y0", "cust", "geo"):
        a, b = arr[k], ref[k]
        assert np.array_equal(a, b, equal_nan=a.dtype.kind == "f"), k
    f = OUT / "features_v3p.npz"
    np.savez_compressed(f, **arr)
    d = {k: float(np.mean(np.abs(arr[k] - ref[k]))) for k in ("xu", "xr", "xo")}
    rec = dict(file=str(f.relative_to(ROOT)), sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
               units=int(len(arr["fips"])), mean_abs_change_vs_e3r2=d,
               per_channel_mean_abs_change_xu=dict(zip(arr["damage_features"].tolist(),
                                                       np.round(np.abs(arr["xu"] - ref["xu"]).mean((0, 1)), 5).tolist())))
    (HERE / "data_provenance" / "features_v3p.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec)[:2000])


if __name__ == "__main__":
    main()
