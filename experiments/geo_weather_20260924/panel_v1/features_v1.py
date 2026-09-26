"""Host features of the designed panel (DATASET_DESIGN v1) and its splits.

Features: build_v3p.features, unchanged in content, reading the v1 panels, with the geography descriptors of every
CONUS county (geography_v1.parquet, geography_ext_v1.parquet) and the v1 county statics (SAIDI from EIA-861 2017,
amendment 2 B1). Added per unit: system, regime (the system's, amendment 2 S7a), family, stratum, pi_s, pi_c, design
weight w = 1 / (pi_s pi_c) trimmed at 10x the regime's median (section 5.4, amendment 2 M10; w_raw untrimmed), h_c,
used (section 4.2).

Splits (section 9.1, amendment 2 S1, M11):
  event: fold groups = families, joined when two systems share a sampled county and their origins are within 16 days;
         within each regime (a group's regime = that of its system with the most sampled county-events), groups sorted
         by their earliest origin, fold = rank mod 5
  main:  county folds, the counties shuffled with seed 20260926 and dealt into 5 folds (discard-only screens)
Outputs: data/interim/panel_v1/features_v1<tranche>.npz, experiments/geo_weather_20260924/splits_v1<tranche>.json"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "open_gcrk_20260919")); sys.path.insert(0, str(HERE.parent))
import build_features as BF  # noqa: E402
import build_v3p as V  # noqa: E402
from frame_common import EXP, OUT, REGIMES  # noqa: E402

TRIM = 10.0
K = 5


def statics_v1(fips: list[str]) -> np.ndarray:
    st = pd.read_parquet(OUT / "county_statics_v1.parquet").set_index("fips")
    s = pd.DataFrame(index=fips)
    s["log_cust"] = st.log_cust.reindex(fips)
    s["rucc"] = st.rucc.reindex(fips)
    s["log_pop_density"] = st.log_pop_density.reindex(fips)
    s["coop_share"] = st.coop_share.reindex(fips)
    s["log1p_n_utilities"] = np.log1p(st.n_utilities.reindex(fips))
    s["log1p_saidi"] = np.log1p(st.saidi.reindex(fips))
    return s[BF.STATIC].to_numpy(float)


def features(tranche: str) -> dict:
    dr = pd.read_parquet(OUT / "draws.parquet").set_index("system")
    cs = pd.read_parquet(OUT / f"county_sample_{tranche}.parquet")
    cs = cs[cs.sampled]
    systems = sorted(set(cs.system))
    # build_v3p.features reads the panels from V.OUT and the geography from BF.OUT; point them at the v1 files
    V.OUT = OUT / "panels"
    BF.statics = statics_v1
    g = OUT / "geography_v1.parquet"; gx = OUT / "geography_ext_v1.parquet"
    orig = pd.read_parquet

    def rp(path, *a, **k):
        name = Path(path).name
        if name == "geography_e3.parquet":
            return orig(g, *a, **k)
        if name == "geography_ext_e3.parquet":
            return orig(gx, *a, **k)
        return orig(path, *a, **k)
    V.pd.read_parquet = rp
    try:
        arr = V.features([dict(event=s) for s in systems], prefix="panel216w")
    finally:
        V.pd.read_parquet = orig
    key = cs.set_index(["system", "fips"])
    rows = key.loc[list(zip(arr["event"], arr["fips"]))]
    pi_s = dr.loc[arr["event"], "pi"].to_numpy(float)
    pi_c = rows.pi_c.to_numpy(float)
    w_raw = 1.0 / (pi_s * pi_c)
    regime = dr.loc[arr["event"], "regime"].to_numpy()
    w = w_raw.copy()
    for r in set(regime):
        m = regime == r
        w[m] = np.minimum(w[m], TRIM * np.median(w_raw[m]))
    arr.update(system=np.array(arr["event"]), regime=regime.astype(str), family=dr.loc[arr["event"], "family"].to_numpy().astype(str),
               stratum=rows.stratum.to_numpy().astype(str), pi_s=pi_s.astype(np.float32), pi_c=pi_c.astype(np.float32),
               w=w.astype(np.float32), w_raw=w_raw.astype(np.float32), h_c=rows.h.to_numpy(np.float32),
               used=dr.loc[arr["event"], "used"].to_numpy(bool),
               origin=np.array(pd.to_datetime(dr.loc[arr["event"], "origin"]).astype(str)))
    assert np.isfinite(arr["xr"]).all(), "non-finite recovery inputs (statics or history)"
    return arr


def splits(arr: dict) -> dict:
    sysv, fam, reg = arr["system"], arr["family"], arr["regime"]
    orig = pd.to_datetime(arr["origin"])
    info = pd.DataFrame({"system": sysv, "family": fam, "regime": reg, "origin": orig, "fips": arr["fips"]})
    by_sys = info.groupby("system").agg(family=("family", "first"), regime=("regime", "first"),
                                        origin=("origin", "first"), n=("fips", "size"),
                                        fips=("fips", lambda x: set(x)))
    parent = {s: s for s in by_sys.index}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    for _, g in by_sys.groupby("family"):
        ss = list(g.index)
        for x in ss[1:]:
            union(ss[0], x)
    ss = list(by_sys.index)
    for i, a in enumerate(ss):
        for b in ss[i + 1:]:
            if abs(by_sys.at[a, "origin"] - by_sys.at[b, "origin"]) <= pd.Timedelta(days=16) and by_sys.at[a, "fips"] & by_sys.at[b, "fips"]:
                union(a, b)
    by_sys["group"] = [find(s) for s in by_sys.index]
    grp = by_sys.groupby("group").agg(origin=("origin", "min"),
                                      regime=("regime", lambda x: x.iloc[int(np.argmax(by_sys.loc[x.index, "n"]))]))
    fold_of_group = {}
    for r, g in grp.groupby("regime"):
        for rank, gid in enumerate(g.sort_values("origin").index):
            fold_of_group[gid] = rank % K + 1
    fold = np.array([fold_of_group[by_sys.at[s, "group"]] for s in sysv])
    rng = np.random.default_rng(20260926)
    cty = np.array(sorted(set(arr["fips"])))
    cf = {c: i % K + 1 for i, c in enumerate(rng.permutation(cty))}
    cfold = np.array([cf[c] for c in arr["fips"]])
    n = len(sysv)
    sp = {"n_units": n, "event": {}, "main": {}}
    for k in range(1, K + 1):
        sp["event"][str(k)] = dict(dev=np.where(fold != k)[0].tolist(), outer=np.where(fold == k)[0].tolist())
        sp["main"][str(k)] = dict(dev=np.where(cfold != k)[0].tolist(), outer=np.where(cfold == k)[0].tolist())
    sp["event_groups"] = {s: int(fold_of_group[by_sys.at[s, "group"]]) for s in by_sys.index}
    sp["event_fold_sizes"] = {str(k): {r: int(((fold == k) & (reg == r)).sum()) for r in REGIMES} for k in range(1, K + 1)}
    return sp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tranche", default="D")
    a = ap.parse_args()
    arr = features(a.tranche)
    f = OUT / f"features_v1{a.tranche}.npz"
    np.savez_compressed(f, **arr)
    sp = splits(arr)
    (EXP / f"splits_v1{a.tranche}.json").write_text(json.dumps(sp) + "\n")
    rec = dict(file=str(f.relative_to(ROOT)), sha256=hashlib.sha256(f.read_bytes()).hexdigest(), units=int(len(arr["fips"])),
               systems=int(len(set(arr["system"]))), by_regime={r: int((arr["regime"] == r).sum()) for r in REGIMES},
               event_fold_sizes=sp["event_fold_sizes"])
    (EXP / "data_provenance" / f"features_v1{a.tranche}.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
