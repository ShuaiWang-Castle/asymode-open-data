"""Exposure-integrated hazard features from HRRR (3 km, hourly; NOAA High-Resolution Rapid Refresh, public AWS bucket
noaa-hrrr-bdp-pds), variant `hrrr` of build_eih.py with the same dictionary, modulators and memory bank.

Nodes: (county, HRRR cell) with population weights (build_nodes_hrrr.py). Weather at valid hour t from cycle t-1,
forecast hour 1 (`wrfsfcf01`, so that hour-ending precipitation exists): TMP and DPT at 2 m, GUST at the surface
(instantaneous; ERA5's gust is the hourly maximum, so the gust ramps are comparable, not identical), APCP (0-1 h
accumulation), CAPE (surface), plus HGT (model terrain) for the downscaling reference. Downscaling D as build_eih.py,
from the HRRR terrain to the node's population-weighted elevation. Only the needed GRIB2 records are read, by byte
range from the .idx index (contrib/DATA_SOURCES_input.md section (c)); nothing is stored but the county features.
Output: data/interim/geo_weather/eih_<tag>hrrr.npz (phi [U,144,160] float16, names, fips, event, missing_hours).
usage: python build_eih_hrrr.py --feat data/interim/geo_weather/features_w1.npz \
       --events-file experiments/geo_weather_20260924/selected_events_winter_build.json --tag w1_
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from build_eih import MOD, NAMES, ORIGIN, PSI, T, TAUS, filt, psi  # noqa: E402

OUT = ROOT / "data" / "interim" / "geo_weather"
MIRRORS = ("https://noaa-hrrr-bdp-pds.s3.amazonaws.com", "https://storage.googleapis.com/high-resolution-rapid-refresh")
KEYS = {"TMP:2 m above ground:1 hour fcst": "t", "DPT:2 m above ground:1 hour fcst": "td", "GUST:surface:1 hour fcst": "g",
        "APCP:surface:0-1 hour acc fcst": "p", "CAPE:surface:1 hour fcst": "cape"}
HGT_KEY = "HGT:surface:1 hour fcst"
_s = requests.Session()
_s.mount("https://", requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=32, max_retries=3))


def fetch(valid: pd.Timestamp, keys: dict, rows, cols):
    """{short name: values at the node cells} for one valid hour, or None if no mirror has the file."""
    import eccodes
    cyc = valid - pd.Timedelta(hours=1)
    path = f"hrrr.{cyc:%Y%m%d}/conus/hrrr.t{cyc:%H}z.wrfsfcf01.grib2"
    for base in MIRRORS:
        try:
            r = _s.get(f"{base}/{path}.idx", timeout=60)
            if r.status_code != 200:
                continue
            rec = [l.split(":") for l in r.text.splitlines() if l]
            offs = [int(x[1]) for x in rec]
            out = {}
            for i, x in enumerate(rec):
                k = ":".join(x[3:6])
                if k not in keys:
                    continue
                end = offs[i + 1] - 1 if i + 1 < len(offs) else ""
                b = _s.get(f"{base}/{path}", headers={"Range": f"bytes={offs[i]}-{end}"}, timeout=120).content
                h = eccodes.codes_new_from_message(b)
                v = eccodes.codes_get_values(h).reshape(1059, 1799)
                eccodes.codes_release(h)
                out[keys[k]] = v[rows, cols].astype(np.float32)
            if len(out) == len(keys):
                return out
        except requests.RequestException:
            continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", required=True)
    ap.add_argument("--events-file", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--nodes", default=None, help="default nodes_hrrr_<tag without _>.parquet")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    F = np.load(ROOT / a.feat)
    fips_u, ev_u = F["fips"].astype(str), F["event"].astype(str)
    nodes = pd.read_parquet(OUT / (a.nodes or f"nodes_hrrr_{a.tag.rstrip('_')}.parquet"))
    by = {f: d.reset_index(drop=True) for f, d in nodes.groupby("fips")}
    sel = {e["event"]: e for e in json.loads((ROOT / a.events_file).read_text())["events"]}
    phi = np.zeros((len(fips_u), T - ORIGIN, len(NAMES)), np.float16)
    missing = {}
    for ev in sorted(set(ev_u)):
        t0e = time.time()
        units = np.where(ev_u == ev)[0]
        nd = pd.concat([by[fips_u[u]].assign(unit=j) for j, u in enumerate(units)], ignore_index=True)
        rows, cols = nd.hr.to_numpy(), nd.hc.to_numpy()
        w = nd.w_pop.to_numpy(); uid = nd.unit.to_numpy()
        month = int(ev[5:7]); leaf = float(5 <= month <= 10)
        can = nd.canopy.to_numpy() / 100.0
        mods = np.stack([np.ones(len(nd)), can, can * leaf, nd.wet.to_numpy()], -1)            # [n, A]
        times = pd.date_range(pd.Timestamp(sel[ev]["window_start_utc"]), periods=T, freq="h")
        hgt = None
        for tt in times:                                                                       # model terrain once
            z = fetch(tt, {HGT_KEY: "hgt"}, rows, cols)
            if z is not None:
                hgt = z["hgt"]; break
        dz = nd.z.to_numpy() - hgt
        from scipy.sparse import csr_matrix
        S = csr_matrix((w, (uid, np.arange(len(nd)))), shape=(len(units), len(nd)))              # exposure integral
        with ThreadPoolExecutor(a.workers) as ex:
            got = list(ex.map(lambda tt: fetch(tt, KEYS, rows, cols), times))
        miss = [str(t) for t, g in zip(times, got) if g is None]
        missing[ev] = miss
        inst = np.zeros((T, len(units), len(MOD), len(PSI)), np.float64)          # (modulator, psi): psi fastest, as NAMES
        for s, g in enumerate(got):
            if g is None:
                continue
            tc, tdc = g["t"] - 273.15, g["td"] - 273.15
            a_, b_ = 17.625, 243.04
            rh = 100.0 * np.exp(a_ * tdc / (b_ + tdc) - a_ * tc / (b_ + tc))
            ps = psi(tc[None], tdc[None], rh[None], g["p"][None], g["g"][None], g["cape"][None], dz, month)[0]   # [n, B]
            inst[s] = (S @ (mods[:, :, None] * ps[:, None, :]).reshape(len(nd), -1)).reshape(len(units), len(MOD), len(PSI))
        x = inst.reshape(T, len(units), -1)                                    # psi fastest, then modulator (= NAMES)
        for j, u in enumerate(units):
            phi[u] = np.concatenate([filt(x[:, j], tau)[ORIGIN:] for tau in TAUS], -1).astype(np.float16)
        print(ev, len(units), "units", len(nd), "nodes; missing hours", len(miss), f"{time.time() - t0e:.0f} s", flush=True)
    np.savez(OUT / f"eih_{a.tag}hrrr.npz", phi=phi, names=np.array(NAMES), fips=fips_u, event=ev_u,
             missing_hours=np.array(json.dumps(missing)))
    (HERE / "data_provenance" / f"eih_{a.tag}hrrr.json").write_text(json.dumps(dict(
        source="NOAA HRRR wrfsfcf01 (AWS noaa-hrrr-bdp-pds; GCS mirror), byte ranges of the records "
               + ", ".join(list(KEYS) + [HGT_KEY]), missing_hours={k: len(v) for k, v in missing.items()}), indent=1) + "\n")
    print("saved", phi.shape)


if __name__ == "__main__":
    main()
