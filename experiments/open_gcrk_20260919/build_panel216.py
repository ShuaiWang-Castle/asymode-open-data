"""Hourly 216-hour county panels for the selected events: outage fraction + ERA5 drivers.

For each event in selected_events.json (window = 216 hourly steps from 00 UTC three
days before the catalogue day):

  outage fraction  EAGLE-I 15-minute county records, densified with the project's
                   explicit observation rule (src/asymode/panel.py: a cell is a true
                   zero only if collection ran at that timestamp and the county
                   reported within +/-7 days; otherwise unobserved), divided by the
                   EAGLE-I 2024 modelled county customers, clipped to [0, 1];
                   hourly value = mean of the observed 15-minute cells in the hour,
                   unobserved if none.
  drivers          ERA5 single levels, area-weighted onto Census 2023 county
                   polygons (src/asymode/weather.py), 12 channels, plus sin/cos of
                   the UTC hour: the project's 14-channel block.

County set: the event's Storm Events footprint (select_events.py) after these data
gates, applied in this order and counted per event:
  G1 a denominator exists and is >= 500 customers;
  G2 the state's EAGLE-I coverage for the year is >= 70% (coverage history spans
     2018-2022; later years use 2022, the nearest year);
  G3 hour 71 (the forecast origin's initial condition) is observed;
  G4 >= 90% of prefix hours (0-71) and >= 90% of forecast hours (72-215) observed.
Nothing here reads anything but these public files; no threshold depends on an
outcome in the forecast window beyond observation availability.

Also writes the 5-nearest-neighbour county list (Census centroids, all CONUS
counties) and the drivers of those neighbours, used by the recovery inputs.
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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from asymode.panel import attach_denominator, build_panel  # noqa: E402
from asymode.weather import apply_weights, county_weights, derive_channels  # noqa: E402
from build_drivers import load_window  # noqa: E402

INTERIM = ROOT / "data" / "interim"
OUT = INTERIM / "open_gcrk"
RAW_ERA5 = ROOT / "data" / "raw" / "era5"
SHP = ROOT / "data/raw/census/cb_county/cb_2023_us_county_500k.shp"
GAZ = ROOT / "data/raw/census/2023_Gaz_counties_national.txt"
T = 216
MIN_CUSTOMERS, MIN_COVERAGE, MIN_OBS = 500, 0.70, 0.90
CHANNELS = ["cape", "cloud", "gust", "precip", "pressure", "rh", "snowfall", "soil_moisture",
            "t2m_c", "u10", "v10", "wind_speed"]
STATE_ABBR = {"alabama": "AL", "arizona": "AZ", "arkansas": "AR", "california": "CA", "colorado": "CO",
              "connecticut": "CT", "delaware": "DE", "district of columbia": "DC", "florida": "FL",
              "georgia": "GA", "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
              "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD", "massachusetts": "MA",
              "michigan": "MI", "minnesota": "MN", "mississippi": "MS", "missouri": "MO", "montana": "MT",
              "nebraska": "NE", "nevada": "NV", "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM",
              "new york": "NY", "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
              "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
              "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT",
              "virginia": "VA", "washington": "WA", "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY"}
FIPS_STATE = {"01": "AL", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT", "10": "DE", "11": "DC",
              "12": "FL", "13": "GA", "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY",
              "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS", "29": "MO",
              "30": "MT", "31": "NE", "32": "NV", "33": "NH", "34": "NJ", "35": "NM", "36": "NY", "37": "NC",
              "38": "ND", "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD",
              "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI",
              "56": "WY"}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def all_weights() -> pd.DataFrame:
    f = OUT / "era5_county_weights_conus.parquet"
    if f.exists():
        return pd.read_parquet(f)
    lats = np.arange(50.0, 24.0 - 1e-9, -0.25)
    lons = np.arange(-125.0, -66.0 + 1e-9, 0.25)
    w = county_weights(lats, lons, SHP, fips=None)
    w = w[w.fips.str[:2].isin(FIPS_STATE)]
    w.to_parquet(f, index=False)
    return w


def neighbours(fips_all: list[str], k: int = 5) -> dict[str, list[str]]:
    gz = pd.read_csv(GAZ, sep="\t", dtype={"GEOID": str}, encoding="latin-1")
    gz.columns = [c.strip() for c in gz.columns]
    gz["GEOID"] = gz.GEOID.str.zfill(5)
    gz = gz[gz.GEOID.isin(set(fips_all))].set_index("GEOID")
    lat, lon = gz.INTPTLAT.to_numpy(float), gz.INTPTLONG.to_numpy(float)
    ids = gz.index.to_list()
    out = {}
    for i, f in enumerate(ids):
        d = np.sqrt((lat - lat[i]) ** 2 + ((lon - lon[i]) * np.cos(np.radians(lat[i]))) ** 2)
        d[i] = np.inf
        out[f] = [ids[j] for j in np.argsort(d)[:k]]
    return out


def coverage(year: int) -> dict[str, float]:
    cov = pd.read_parquet(INTERIM / "eaglei_coverage_history.parquet")
    cov["yr"] = pd.to_datetime(cov["year"], format="%m/%d/%y").dt.year
    use = year if year in set(cov.yr) else int(min(cov.yr, key=lambda y: abs(y - year)))
    c = cov[cov.yr == use]
    return dict(zip(c.state, c.max_pct_covered)), use


def era5_fields(event: str):
    """(valid times, derived channel fields) for one event window, streams merged."""
    import xarray as xr
    pieces = [p for p in sorted(RAW_ERA5.glob(f"era5_{event}*.nc")) if not p.name.endswith(".part")]
    if not pieces:
        raise FileNotFoundError(f"no ERA5 archive for {event}")
    ds = (load_window(pieces[0]) if len(pieces) == 1
          else xr.concat([load_window(p) for p in pieces], dim="valid_time").sortby("valid_time"))
    return pd.to_datetime(ds["valid_time"].values), derive_channels(ds), [p.name for p in pieces]


def drivers(fields, t0: pd.Timestamp, fips: list[str], w: pd.DataFrame) -> np.ndarray:
    """(C, 216, 14): the 12 area-weighted channels plus sin/cos of the UTC hour."""
    times, ch, _ = fields
    hours = pd.date_range(t0, periods=T, freq="h")
    arrs = []
    for n in CHANNELS:
        county = apply_weights(np.asarray(ch[n], dtype=np.float32), w, fips)
        s = pd.DataFrame(county.T, index=times)
        s = s[~s.index.duplicated()].reindex(hours)
        if s.isna().any().any():
            raise ValueError(f"ERA5 does not cover every window hour for {n}")
        arrs.append(s.to_numpy(np.float32).T)
    X = np.stack(arrs, -1)
    hod = hours.hour.to_numpy(float)
    clock = np.stack([np.sin(2 * np.pi * hod / 24), np.cos(2 * np.pi * hod / 24)], -1).astype(np.float32)
    return np.concatenate([X, np.broadcast_to(clock, (len(fips), T, 2))], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events-file", default="selected_events.json")
    ap.add_argument("--events", nargs="+", default=None, help="subset of the file's events")
    ap.add_argument("--gates-out", default="panel_gates.csv")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    sel = json.loads((HERE / a.events_file).read_text())["events"]
    if a.events:
        sel = [e for e in sel if e["event"] in set(a.events)]
    denom = pd.read_parquet(INTERIM / "eaglei_county_customers_2024.parquet")["customers"]
    w = all_weights()
    conus = sorted(w.fips.unique())
    nn5 = neighbours(conus)
    gates, prov = [], []
    for e in sel:
        ev = e["event"]
        t0 = pd.Timestamp(e["window_start_utc"])
        t1 = t0 + pd.Timedelta(hours=T - 1)
        year = t0.year
        pq = INTERIM / f"eaglei_outages_{year}.parquet"
        df = pd.read_parquet(pq, columns=["fips", "ts", "customers_out"],
                             filters=[("ts", ">=", t0 - pd.Timedelta(days=9)),
                                      ("ts", "<=", t1 + pd.Timedelta(days=9))])
        df["fips"] = df.fips.astype(str).str.zfill(5)
        foot = [f for f in e["footprint_fips"] if f in set(conus)]
        g = dict(event=ev, footprint=len(e["footprint_fips"]), in_conus_grid=len(foot))
        keep = [f for f in foot if f in denom.index and denom[f] >= MIN_CUSTOMERS]
        g["G1_denominator"] = len(keep)
        cov, cov_year = coverage(year)
        keep = [f for f in keep if cov.get(FIPS_STATE.get(f[:2], ""), 0.0) >= MIN_COVERAGE]
        g["G2_coverage"] = len(keep); g["coverage_year_used"] = cov_year
        p = build_panel(df, t0, t1 + pd.Timedelta(minutes=45), fips=keep, freq="15min")
        p = attach_denominator(p, denom, "eaglei_2024_modelled")
        y15 = np.where(p["observed"], p["y"], np.nan).reshape(len(p["fips"]), T, 4)
        obs = np.isfinite(y15).any(-1)
        with np.errstate(all="ignore"):
            y = np.where(obs, np.nanmean(y15, -1), np.nan)
        ok3 = obs[:, 71]
        ok4 = (obs[:, :72].mean(1) >= MIN_OBS) & (obs[:, 72:].mean(1) >= MIN_OBS)
        g["G3_origin_observed"] = int(ok3.sum()); g["G4_observed_share"] = int((ok3 & ok4).sum())
        idx = np.where(ok3 & ok4)[0]
        fips = [p["fips"][i] for i in idx]
        fields = era5_fields(ev)
        X = drivers(fields, t0, fips, w)
        nbr = sorted({n for f in fips for n in nn5[f]})
        Xn = drivers(fields, t0, nbr, w)
        g["era5_files"] = ";".join(fields[2])
        np.savez_compressed(OUT / f"panel216_{ev}.npz", event=ev, fips=np.array(fips), y=y[idx].astype(np.float32),
                            observed=obs[idx], denominator=p["denominator"][idx],
                            X=X.astype(np.float32), channels=np.array(CHANNELS + ["clock_sin", "clock_cos"]),
                            ts=np.array(pd.date_range(t0, periods=T, freq="h").astype(str)),
                            nbr_fips=np.array(nbr), X_nbr=Xn.astype(np.float32),
                            nbr_of=np.array([nn5[f] for f in fips]))
        g["final"] = len(fips); g["states"] = len({f[:2] for f in fips})
        g["observed_share_final"] = float(obs[idx].mean())
        gates.append(g)
        print(g, flush=True)
    pd.DataFrame(gates).to_csv(HERE / a.gates_out, index=False)
    for f in sorted(OUT.glob("panel216_*.npz")) + [OUT / "era5_county_weights_conus.parquet"]:
        prov.append(dict(file=str(f.relative_to(ROOT)), sha256=sha(f), bytes=f.stat().st_size))
    (HERE / "data_provenance").mkdir(exist_ok=True)
    (HERE / "data_provenance" / "panel216_checksums.json").write_text(json.dumps(prov, indent=1) + "\n")


if __name__ == "__main__":
    main()
