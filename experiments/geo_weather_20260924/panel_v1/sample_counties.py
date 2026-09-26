"""Counties within a system (DATASET_DESIGN v1, section 5): the gated domain in strata S1 / S2 / S3, the weather hazard
index h_c from the system's ERA5 window (forecast hours only; computed before any outcome of the system is
extracted), and a systematic PPS sample of at most 140 counties on max(h_c, 0.25).

h_c = max over the forecast window of four ratios of population-weighted county means of ERA5 cell values (the
population of the 3 km nodes stands in for customers, which are not gridded):
  max hourly 10 m gust (fg10) / 20 m/s; max 24 h precipitation / 50 mm; max 24 h freezing precipitation (tp in hours
  whose ptype is freezing rain 3 or freezing drizzle 12) / 5 mm; max 24 h snowfall water equivalent / 15 mm.
Output: data/interim/panel_v1/county_sample_<tranche>.parquet
usage: python sample_counties.py --tranche D"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import xarray as xr

import gates as GT
from frame_common import EXP, OUT, PREFIX_H, ROOT

N_MAX, SHARES, STRATA = 140, (90, 25, 25), ("S1", "S2", "S3")   # section 5.3
H_FLOOR = 0.25
SEED0 = 20260929
THRESH = dict(gust=20.0, rain=50.0, frz=5.0, snow=15.0)             # section 5.2


def pop_weights() -> pd.DataFrame:
    f = OUT / "era5_popweights.parquet"
    if f.exists():
        return pd.read_parquet(f)
    n = pd.read_parquet(OUT / "nodes3k.parquet")
    n["row"] = np.rint((90.0 - n.lat) / 0.25).astype(int)
    n["col"] = np.rint((n.lon % 360.0) / 0.25).astype(int)
    w = n.groupby(["fips", "row", "col"], as_index=False)["pop"].sum()
    w["w"] = w["pop"] / w.groupby("fips")["pop"].transform("sum")
    w[["fips", "row", "col", "w"]].to_parquet(f, index=False)
    return w


def hazard_index(system: str, fips: list[str], W: pd.DataFrame) -> pd.DataFrame:
    ds = xr.open_dataset(ROOT / "data" / "raw" / "era5_v1" / f"{system}.nc", engine="h5netcdf")
    lat, lon = ds.latitude.values, ds.longitude.values
    r0, c0 = int(round((90.0 - lat[0]) / 0.25)), int(round((lon[0] % 360.0) / 0.25))
    sl = slice(PREFIX_H, None)                                    # forecast hours only
    fg = ds.fg10.values[sl]; tp = ds.tp.values[sl] * 1000.0; sf = ds.sf.values[sl] * 1000.0
    pt = np.rint(ds.ptype.values[sl])
    frz = np.where(np.isin(pt, (3, 12)), tp, 0.0)
    ds.close()
    out = []
    for f in fips:
        g = W[W.fips == f]
        i, j = g.row.to_numpy() - r0, g.col.to_numpy() - c0
        ok = (i >= 0) & (i < len(lat)) & (j >= 0) & (j < len(lon))
        i, j, w = i[ok], j[ok], g.w.to_numpy()[ok]
        w = w / w.sum()

        def cm(a):
            return (a[:, i, j] * w[None, :]).sum(1)

        def roll24(x):
            c = np.concatenate([[0.0], np.cumsum(x)])
            return (c[24:] - c[:-24]).max() if len(x) >= 24 else x.sum()

        r = dict(fips=f, gust=cm(fg).max(), rain=roll24(cm(tp)), frz=roll24(cm(frz)), snow=roll24(cm(sf)))
        r["h"] = max(r["gust"] / THRESH["gust"], r["rain"] / THRESH["rain"], r["frz"] / THRESH["frz"],
                     r["snow"] / THRESH["snow"])
        out.append(r)
    return pd.DataFrame(out)


def allocate(sizes: dict) -> dict:
    """90 / 25 / 25 over S1 / S2 / S3; capacity a stratum cannot use passes on S1 -> S2 -> S3 -> S1."""
    n = min(sum(sizes.values()), N_MAX)
    a = {s: min(t, sizes[s]) for s, t in zip(STRATA, SHARES)}
    left = n - sum(a.values())
    for s in ("S2", "S3", "S1", "S2", "S3"):
        add = min(left, sizes[s] - a[s])
        a[s] += add; left -= add
    return a


def pps_within(h: np.ndarray, n: int, rng) -> tuple[np.ndarray, np.ndarray]:
    from draw import pps_systematic
    order = np.argsort(-h, kind="stable")                         # sorted by h_c
    sel, pi = pps_systematic(np.maximum(h[order], H_FLOOR), n, rng)
    s_, p_ = np.zeros(len(h), bool), np.zeros(len(h))
    s_[order], p_[order] = sel, pi
    return s_, p_


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tranche", default="D")
    ap.add_argument("--exclusions", default="data_provenance/operator_exclusions.csv")
    a = ap.parse_args()
    import build_frame as BF
    dr = pd.read_parquet(OUT / "draws.parquet")
    sy = pd.read_parquet(OUT / "systems.parquet").set_index("system")
    sc = pd.read_parquet(OUT / "system_counties.parquet")
    st = GT.FixedGates(BF.load_exclusions(a.exclusions))
    cmask = GT.CMask() if a.tranche != "C" else None       # amendment 2 S8: D reads skip C's county-hours
    W = pop_weights()
    rows = []
    for s in dr[dr.tranche == a.tranche].system:
        origin, w1 = pd.Timestamp(sy.at[s, "origin"]), pd.Timestamp(sy.at[s, "window_end"])
        d = sc[sc.system == s].copy()
        g = pd.DataFrame([st(f, origin, w1) for f in d.fips], index=d.index)
        d = pd.concat([d, g], axis=1)
        dyn = GT.dynamic_gates(origin, d.fips.to_list(), cmask).set_index("fips")
        d = d.join(dyn, on="fips")
        d["gated"] = d.G1 & d.G2 & d.G2_in_data & ~d.excluded & d.G3 & d.G4 & d.G5
        hz = hazard_index(s, d.fips.to_list(), W).set_index("fips")
        d = d.join(hz, on="fips")
        dom = d[d.gated]
        alloc = allocate({k: int((dom.stratum == k).sum()) for k in STRATA})
        rng = np.random.default_rng(SEED0 + int(s[1:]))
        d["sampled"], d["pi_c"] = False, 0.0
        for k in STRATA:
            idx = dom.index[dom.stratum == k]
            if alloc[k] == 0 or len(idx) == 0:
                continue
            sel, pi = pps_within(d.loc[idx, "h"].to_numpy(), alloc[k], rng)
            d.loc[idx, "pi_c"] = pi
            d.loc[idx[sel], "sampled"] = True
        rows.append(d)
        print(s, sy.at[s, "regime"], "domain", len(d), "gated", int(d.gated.sum()), "alloc", alloc,
              "sampled", int(d.sampled.sum()), flush=True)
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(OUT / f"county_sample_{a.tranche}.parquet", index=False)
    drop = out.groupby("stratum")[["G1", "G2", "G2_in_data", "G3", "G4", "G5"]].apply(lambda x: (~x.astype(bool)).mean()).round(4)
    ex = out.groupby("stratum").excluded.mean().round(4)
    (EXP / "data_provenance" / f"gate_audit_{a.tranche}.json").write_text(json.dumps(
        dict(drop_rates=drop.to_dict(orient="index"), excluded=ex.to_dict(),
             g3_drop_share_of_otherwise_gated=float(((out.G1 & out.G2 & ~out.excluded) & ~out.G3).sum()
                                                    / max((out.G1 & out.G2 & ~out.excluded).sum(), 1))), indent=1) + "\n")
    print(drop.to_string())


if __name__ == "__main__":
    main()
