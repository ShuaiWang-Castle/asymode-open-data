"""Constants and county tables of the DATASET_DESIGN v1 frame (sections 3-6). Every number here is a registered value
of the design; the section is given beside it."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
OUT = INTERIM / "panel_v1"

# section 3.2: defining products (SIG = W) and typing priority (1 = highest)
REGIME_OF = {"TR": "tropical", "HU": "tropical", "SS": "tropical", "EW": "tropical",
             "WS": "winter", "IS": "winter", "BZ": "winter", "LE": "winter",
             "HW": "synoptic_wind",
             "SV": "convective", "TO": "convective",
             "FF": "heavy_rain", "FA": "heavy_rain"}
PRIORITY = {"tropical": 1, "winter": 2, "synoptic_wind": 3, "convective": 4, "heavy_rain": 5}
REGIMES = sorted(PRIORITY, key=PRIORITY.get)
W_PHENOM = set(REGIME_OF)
S2_PRODUCTS = {(p, "A") for p in W_PHENOM} | {("WI", "Y"), ("WW", "Y"), ("ZR", "Y"), ("LE", "Y"), ("FA", "Y")}
POLYGON_PHENOM = {"SV", "TO", "FF"}        # initial storm-based polygons in <year>_tsmf_sbw.zip
TROPICAL_PHENOM = {"TR", "HU", "SS", "EW"}  # amendment 1 (1): join a TC system from 72 h before its span
CT_SWITCH = pd.Timestamp("2025-05-29 00:00")  # amendment 1 (2): EAGLE-I reports Connecticut planning regions after

A_WARNED = 0.1          # section 3.3: S1 needs exposure a >= 0.1
LINK_GAP_H = 12         # section 3.4: events within 12 h ...
LINK_CAP_H = 168        # amendment 2 M1: for linking, an interval ends at min(last EXPIRED, INIT_ISS + 168 h)
SEGMENT_H = 96          # ... families longer than 96 h are cut into segments
TC_RADIUS_KM = 300      # section 3.4: TC span and domain
RING_KM = 200           # section 5.1: S3 ring
LEAD_H = 6              # section 3.6: origin = floor_hour(onset) - 6 h
PREFIX_H, HORIZON_H = 72, 144
FRAME_START = pd.Timestamp("2018-07-12 00:00")   # section 3.7 S-a (HRRR v3)
FRAME_END = pd.Timestamp("2026-01-01 00:00")     # exclusive end of the last window hour
P2_START = pd.Timestamp("2020-12-02 00:00")      # section 3.8: HRRR v4
COMPOUND_SHARE = 0.2    # section 3.5
MIN_CUSTOMERS = 500     # section 6, G1
MIN_COVERAGE = 0.8      # section 6, G2 (the year's minimum)
COLLECTION_SHARE = 0.9  # section 3.7 S-b
HRRR_SHARE = 0.95       # section 3.7 S-c

# section 3.8: core months per regime; every other month is shoulder season
CORE_MONTHS = {"convective": {5, 6, 7, 8}, "synoptic_wind": {11, 12, 1, 2, 3}, "winter": {12, 1, 2},
               "tropical": {8, 9, 10}, "heavy_rain": {6, 7, 8, 9}}

# section 3.8: five regions by state
REGION_STATES = {
    "northeast": "ME NH VT MA RI CT NY NJ PA DE MD DC",
    "southeast": "VA WV NC SC GA FL AL MS TN KY AR LA",
    "central": "OH MI IN IL WI MN IA MO",
    "plains_mountain": "ND SD NE KS OK TX MT WY CO NM ID UT AZ NV",
    "pacific": "WA OR CA",
}
REGION_OF_STATE = {s: r for r, ss in REGION_STATES.items() for s in ss.split()}
NOT_CONUS = {"AK", "HI", "PR", "VI", "GU", "AS", "MP"}


CB2020 = "zip://" + str(RAW / "census" / "cb_2020_us_county_500k.zip")


def counties() -> pd.DataFrame:
    """CONUS counties of the 2020 vintage, the units EAGLE-I reports through 2024 (Connecticut's eight counties, not
    the 2022 planning regions): fips, state (USPS), state_fips, lat, lon (Census Gazetteer 2020 internal points),
    region, customers (EAGLE-I 2024 modelled customers, 0 if absent)."""
    g = pd.read_csv(RAW / "census" / "2020_Gaz_counties_national.txt", sep="\t", dtype=str)
    g.columns = [c.strip() for c in g.columns]
    g = g[~g.USPS.isin(NOT_CONUS)]
    c = pd.DataFrame({"fips": g.GEOID, "state": g.USPS, "state_fips": g.GEOID.str[:2],
                      "lat": g.INTPTLAT.astype(float), "lon": g.INTPTLONG.astype(float)})
    c["region"] = c.state.map(REGION_OF_STATE)
    cust = pd.read_parquet(INTERIM / "eaglei_county_customers_2024.parquet")["customers"]
    c["customers"] = c.fips.map(cust).fillna(0.0).astype(float)
    return c.reset_index(drop=True)


def adjacency(fips: set[str]) -> dict[str, set[str]]:
    """County adjacency of the 2020 vintage (shared boundary or corner), from the Census cartographic boundaries
    (no 2020 adjacency file is published); cached in data/interim/panel_v1/adjacency2020.parquet."""
    f = OUT / "adjacency2020.parquet"
    if not f.exists():
        import geopandas as gpd
        c = gpd.read_file(CB2020)[["GEOID", "geometry"]].to_crs(5070)
        c["geometry"] = c.geometry.buffer(1.0)
        j = gpd.sjoin(c, c, predicate="intersects")
        j = j[j.GEOID_left != j.GEOID_right]
        OUT.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"fips": j.GEOID_left.to_numpy(), "nbr": j.GEOID_right.to_numpy()}).to_parquet(f, index=False)
    a = pd.read_parquet(f)
    a = a[a.fips.isin(fips) & a.nbr.isin(fips)]
    nb: dict[str, set[str]] = {x: set() for x in fips}
    for x, n in zip(a.fips, a.nbr):
        nb[x].add(n)
    return nb


def coverage_min() -> dict[tuple[str, int], float]:
    """(state, year) -> the year's minimum coverage share (EAGLE-I coverage_history, 2018-2022)."""
    c = pd.read_parquet(INTERIM / "eaglei_coverage_history.parquet")
    c["yr"] = c["year"].str.slice(-2).astype(int) + 2000
    return {(s, int(y)): float(v) for s, y, v in zip(c.state, c.yr, c.min_pct_covered)}


def coverage_max() -> dict[tuple[str, int], float]:
    """(state, year) -> the year's maximum coverage share (amendment 2 S4: S-d and M_s use the maximum)."""
    c = pd.read_parquet(INTERIM / "eaglei_coverage_history.parquet")
    c["yr"] = c["year"].str.slice(-2).astype(int) + 2000
    return {(s, int(y)): float(v) for s, y, v in zip(c.state, c.yr, c.max_pct_covered)}


def coverage_year(year: int) -> int:
    """Section 6, G2: the most recent coverage year not after the system's year (2022 for 2023-2025)."""
    return min(max(year, 2018), 2022)


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dl = np.deg2rad(np.asarray(lon2) - np.asarray(lon1))
    h = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
