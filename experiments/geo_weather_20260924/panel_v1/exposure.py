"""County exposure a to warning areas (DATASET_DESIGN v1, section 3.3): the share of a county's population (3 km
nodes, build_nodes3k.py) inside a storm-based polygon or inside a forecast zone.

  zone_shares()     for every distinct zone geometry version of the IEM UGC snapshots: (vhash, fips, a)
  polygon_shares()  for every initial storm-based polygon (SV, TO, FF; <year>_tsmf_sbw.zip): (key, fips, a), key =
                    WFO|PHENOM|SIG|ETN|year of INIT_ISS
Both cached under data/interim/panel_v1/."""
from __future__ import annotations

import numpy as np
import pandas as pd
import shapely
from shapely import STRtree

from frame_common import OUT, RAW


def _nodes():
    n = pd.read_parquet(OUT / "nodes3k.parquet")
    pts = shapely.points(n.lon.to_numpy(), n.lat.to_numpy())
    cpop = n.groupby("fips")["pop"].sum()
    return n, pts, STRtree(pts), cpop


def _shares(geoms: np.ndarray, keys: np.ndarray, n, tree, cpop) -> pd.DataFrame:
    gi, pi = tree.query(geoms, predicate="intersects")
    d = pd.DataFrame({"key": keys[gi], "fips": n.fips.to_numpy()[pi], "pop": n["pop"].to_numpy()[pi]})
    d = d.groupby(["key", "fips"], as_index=False)["pop"].sum()
    d["a"] = (d["pop"] / d.fips.map(cpop)).clip(upper=1.0)
    return d[["key", "fips", "a"]]


def zone_shares() -> pd.DataFrame:
    f = OUT / "zone_shares.parquet"
    if f.exists():
        return pd.read_parquet(f)
    v = pd.read_parquet(RAW / "nws" / "ugc" / "ugc_versions.parquet")
    v = v[v.ugc.str[2] == "Z"]
    n, _, tree, cpop = _nodes()
    geoms = shapely.from_wkb(v.wkb.to_numpy())
    d = _shares(geoms, v.vhash.to_numpy(), n, tree, cpop).rename(columns={"key": "vhash"})
    d.to_parquet(f, index=False)
    return d


def polygon_shares(years) -> pd.DataFrame:
    f = OUT / "polygon_shares.parquet"
    if f.exists():
        return pd.read_parquet(f)
    import geopandas as gpd
    n, _, tree, cpop = _nodes()
    out = []
    for y in years:
        g = gpd.read_file(f"zip://{RAW / 'nws' / 'vtec' / f'{y}_tsmf_sbw.zip'}")
        g = g[g.PHENOM.isin(["SV", "TO", "FF"]) & (g.SIG == "W") & (g.STATUS == "NEW")].copy()
        init = pd.to_datetime(g.INIT_ISS, format="%Y%m%d%H%M")
        g["key"] = g.WFO + "|" + g.PHENOM + "|" + g.SIG + "|" + g.ETN.astype(int).astype(str) + "|" + init.dt.year.astype(str)
        g = g.sort_values("POLY_BEG").drop_duplicates("key")          # the initial polygon of each event
        out.append(_shares(g.geometry.to_numpy(), g.key.to_numpy(), n, tree, cpop))
        print("polygons", y, len(g), flush=True)
    d = pd.concat(out, ignore_index=True)
    d.to_parquet(f, index=False)
    return d
