"""Winter panel W1 (select_events_winter.py): panels, round-2 inputs, splits, training mask, for the events whose
ERA5 files are on disk.

Steps (every rule as in the wind panel):
  0. data gate: ERA5 on disk and an EAGLE-I record that covers the window (2022-12-13 fails: the local 2022 file
     ends on 2022-11-12)
  1. footprint = the event's winter-report counties that have sub-grid nodes (the 2,409 counties whose rasters are
     built; a data-availability rule, independent of outcomes)
  2. round-1 panels with the data gates G1-G4 (open_gcrk_20260919/build_panel216.py, unchanged)
  3. round-2 inputs with area weights (the wind panel's e3r2 construction: hourly-maximum gust, cell-level
     hazards, support channels), then the features of build_features_r2.py
  4. county-grouped splits: five outer folds, three inner folds (open_gcrk_20260919/run.py rules, seed 20260919)
Outputs: data/interim/geo_weather/{panel216w_<event>.npz, features_w1.npz}, splits_w1.json,
selected_events_winter_build.json, panel_gates_winter.csv, data_provenance/features_w1.json.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OG = ROOT / "experiments" / "open_gcrk_20260919"
sys.path.insert(0, str(OG)); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "src"))
import build_panel216 as BP  # noqa: E402
import build_v3p as V  # noqa: E402

OUT = ROOT / "data" / "interim" / "geo_weather"
K_OUTER, K_INNER, SPLIT_SEED = 5, 3, 20260919


def era5_ready(ev: str) -> bool:
    main = [p for p in (ROOT / "data/raw/era5").glob(f"era5_{ev}*.nc")]
    fg = [p for p in (ROOT / "data/raw/era5_fg10").glob(f"era5_{ev}*.nc")]
    return bool(main) and bool(fg)


def eaglei_covers(e: dict) -> bool:
    """Data gate: the EAGLE-I file on disk for the window's year must reach one day past the window's end
    (the local 2022 file ends on 2022-11-12, so the 2022-12-13 episode has no outage record here)."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    t0 = pd.Timestamp(e["window_start_utc"]); t1 = t0 + pd.Timedelta(hours=215)
    fs = [ROOT / "data" / "interim" / f"eaglei_outages_{y}.parquet" for y in sorted({t0.year, t1.year})]
    if not all(f.exists() for f in fs):
        return False
    ok = pd.Timestamp(pc.max(pq.read_table(fs[-1], columns=["ts"]).column("ts")).as_py()) >= t1 + pd.Timedelta(days=1)
    if not ok:
        print("EAGLE-I gate: no outage record for", e["event"], flush=True)
    return ok


def blocks(order, k, rng):
    lab = {}
    for s in range(0, len(order), k):
        blk = order[s:s + k]
        for f, j in zip(blk, rng.permutation(k)[:len(blk)]):
            lab[f] = int(j)
    return lab


def county_order(F, units):
    d = pd.DataFrame(dict(fips=F["fips"][units], event=F["event"][units],
                          pre=np.nanmax(np.where(F["obs_full"][units, :72], F["y_full"][units, :72], np.nan), 1),
                          cust=F["cust"][units]))
    g = d.groupby("fips").agg(event=("event", "min"), pre=("pre", "max"), cust=("cust", "max")).reset_index()
    g["state"] = g.fips.str[:2]
    return g.sort_values(["event", "state", "pre", "cust", "fips"]).fips.tolist()


def splits(F) -> dict:
    n = len(F["fips"]); allu = np.arange(n)
    lab = blocks(county_order(F, allu), K_OUTER, np.random.default_rng(SPLIT_SEED))
    of = np.array([lab[f] for f in F["fips"]])
    main = {}
    for k in range(K_OUTER):
        held, dev = allu[of == k], allu[of != k]
        il = blocks(county_order(F, dev), K_INNER, np.random.default_rng(SPLIT_SEED + 1 + k))
        fl = np.array([il[f] for f in F["fips"][dev]])
        main[str(k + 1)] = dict(outer=held.tolist(), dev=dev.tolist(),
                                inner=[dict(fit=dev[fl != j].tolist(), val=dev[fl == j].tolist()) for j in range(K_INNER)])
    return dict(k_outer=K_OUTER, split_seed=SPLIT_SEED, n_units=n, main=main)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--events-file", default="selected_events_winter.json")
    ap.add_argument("--tag", default="w1")
    a = ap.parse_args()
    sel = json.loads((HERE / a.events_file).read_text())["events"]
    have = set(np.load(OUT / "nodes_e3.npz")["fips"].astype(str))
    ready = [e for e in sel if era5_ready(e["event"]) and eaglei_covers(e)]
    print("ERA5 ready:", [e["event"] for e in ready], flush=True)
    build = [dict(e, footprint_fips=[f for f in e["footprint_fips"] if f in have]) for e in ready]
    bf = f"selected_events_{a.tag}_build.json" if a.tag != "w1" else "selected_events_winter_build.json"
    (HERE / bf).write_text(json.dumps(dict(events=build), indent=1) + "\n")
    todo = [e["event"] for e in build if not (BP.OUT / f"panel216_{e['event']}.npz").exists()]
    if todo:
        ck = OG / "data_provenance" / "panel216_checksums.json"
        keep = ck.read_bytes()
        subprocess.run([sys.executable, "-u", "build_panel216.py", "--events-file",
                        f"../geo_weather_20260924/{bf}", "--events", *todo,
                        "--gates-out", f"../geo_weather_20260924/panel_gates_{a.tag}.csv"], cwd=OG, check=True)
        ck.write_bytes(keep)                     # the wind panel's provenance file stays as committed
    w = BP.all_weights()
    for e in build:
        f = OUT / f"panel216w_{e['event']}.npz"
        if not f.exists():
            np.savez_compressed(f, **V.panel(e["event"], pd.Timestamp(e["window_start_utc"]), w))
        print("round-2 panel", e["event"], flush=True)
    arr = V.features(build, prefix="panel216w")
    f = OUT / f"features_{a.tag}.npz"
    np.savez_compressed(f, **arr)
    sp = splits(arr)
    (HERE / f"splits_{a.tag}.json").write_text(json.dumps(sp) + "\n")
    rec = dict(file=str(f.relative_to(ROOT)), sha256=hashlib.sha256(f.read_bytes()).hexdigest(), units=int(len(arr["fips"])),
               counties=int(len(set(arr["fips"]))), events={e: int((arr["event"] == e).sum()) for e in sorted(set(arr["event"]))},
               outer_sizes={k: len(v["outer"]) for k, v in sp["main"].items()})
    (HERE / "data_provenance" / f"features_{a.tag}.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
