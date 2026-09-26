"""County gates of DATASET_DESIGN v1 (section 6). G1-G2 and the exclusions are static; G3-G5 read EAGLE-I rows from
the 30 days before the window and from the 72 h prefix only. Nothing at or after the origin is loaded, so the gates
never read a forecast-window outcome.

  G1  >= 500 modelled customers (2024)
  G2  state coverage >= 0.8 by the year's maximum, most recent coverage year not after the system's year; a state whose
      minimum is below 0.8 keeps its counties only if >= 80% of its 2024 modelled customers are in counties with a
      positive record in the 30 days before the window (amendment 2 S4)
  G3  a positive record in the 30 days before the window (zero rows and blank counts do not count)
  G4  >= 90% of prefix hours observed (a national collection run in the hour and a non-blank value)
  G5  the origin hour (the last prefix hour) observed
  exclusions: operator-initiated outages (section 7) and Connecticut after 2025-05-29 (amendment 1 (2))
Every read skips the county-hours of the sealed tranche's windows (C's S1-S3 counties x [window start, window end + 7 d],
amendment 2 S8): `CMask`."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from frame_common import (CT_SWITCH, INTERIM, MIN_COVERAGE, MIN_CUSTOMERS, OUT, PREFIX_H, ROOT, counties,
                          coverage_max, coverage_min, coverage_year)

sys.path.insert(0, str(ROOT / "src"))
from asymode.panel import build_panel  # noqa: E402

LOOKBACK_D = 30
MIN_PREFIX_OBS = 0.9


class CMask:
    """Amendment 2 S8: the county-hours of C's windows, C's S1-S3 counties x [window start, window end + 7 d]."""

    def __init__(self):
        dr = pd.read_parquet(OUT / "draws.parquet")
        sc = pd.read_parquet(OUT / "system_counties.parquet")
        sy = pd.read_parquet(OUT / "systems.parquet").set_index("system")
        c = dr[dr.tranche == "C"].system
        m = sc[sc.system.isin(set(c))][["system", "fips"]].drop_duplicates()
        m["s"] = m.system.map(sy.window_start)
        m["e"] = m.system.map(sy.window_end) + pd.Timedelta(days=7)
        self.m = m[["fips", "s", "e"]].reset_index(drop=True)

    def drop(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        lo, hi = df.ts.min(), df.ts.max()
        m = self.m[(self.m.s <= hi) & (self.m.e >= lo)]
        if m.empty:
            return df
        j = df.reset_index().merge(m, on="fips", how="inner")
        bad = j[(j.ts >= j.s) & (j.ts <= j.e)]["index"].unique()
        return df.drop(index=bad)


class FixedGates:
    def __init__(self, exclusions=None):
        c = counties()
        self.cust = dict(zip(c.fips, c.customers))
        self.state = dict(zip(c.fips, c.state))
        self.cov = coverage_max()                     # amendment 2 S4
        self.excl = exclusions

    def __call__(self, f: str, origin: pd.Timestamp, w1: pd.Timestamp) -> dict:
        g1 = self.cust.get(f, 0.0) >= MIN_CUSTOMERS
        g2 = self.cov.get((self.state.get(f, ""), coverage_year(origin.year)), 0.0) >= MIN_COVERAGE
        ex = (f.startswith("09") and w1 > CT_SWITCH) or (self.excl is not None and self.excl(f, origin, w1))
        return dict(G1=g1, G2=g2, excluded=bool(ex))


def eaglei_before(origin: pd.Timestamp, cmask: CMask | None) -> pd.DataFrame:
    """EAGLE-I rows in [window start - 30 d, origin): the lookback and the prefix, nothing later, C's county-hours
    skipped."""
    w0 = origin - pd.Timedelta(hours=PREFIX_H)
    lo, hi = w0 - pd.Timedelta(days=LOOKBACK_D), origin
    parts = []
    for y in sorted({lo.year, hi.year}):
        parts.append(pd.read_parquet(INTERIM / f"eaglei_outages_{y}.parquet", columns=["fips", "ts", "customers_out"],
                                     filters=[("ts", ">=", lo), ("ts", "<", hi)]))
    df = pd.concat(parts, ignore_index=True)
    df["fips"] = df.fips.astype(str).str.zfill(5)
    df = df[(df.ts >= lo) & (df.ts < hi)]
    assert df.ts.max() < origin
    df = df[~(df.customers_out.fillna(-1) == 0)]          # explicit zero rows dropped (section 6)
    if cmask is not None:
        df = cmask.drop(df)
    return df


_CMIN = None


def in_data_coverage(df: pd.DataFrame, w0: pd.Timestamp, year: int) -> dict[str, bool]:
    """Amendment 2 S4: for each state whose minimum coverage (coverage year of `year`) is below 0.8, whether at least 80%
    of its 2024 modelled customers are in counties with a positive record in the 30 days before the window."""
    global _CMIN
    _CMIN = _CMIN or coverage_min()
    c = counties()
    cy = coverage_year(year)
    lo = w0 - pd.Timedelta(days=LOOKBACK_D)
    seen = set(df[(df.ts >= lo) & (df.ts < w0) & (df.customers_out.fillna(0) > 0)].fips)
    out = {}
    for st, g in c.groupby("state"):
        if _CMIN.get((st, cy), 0.0) < MIN_COVERAGE:
            tot = g.customers.sum()
            out[st] = bool(tot > 0 and g.customers[g.fips.isin(seen)].sum() / tot >= 0.8)
    return out


def dynamic_gates(origin: pd.Timestamp, fips: list[str], cmask: CMask | None) -> pd.DataFrame:
    """G3-G5 and the in-data coverage check for the counties of one system (origin = forecast origin)."""
    w0 = origin - pd.Timedelta(hours=PREFIX_H)
    df = eaglei_before(origin, cmask)
    p = build_panel(df, w0, origin - pd.Timedelta(minutes=15), fips=fips, service_rule="pre_window",
                    lookback_days=LOOKBACK_D)
    obs15 = p["observed"] & np.isfinite(p["counts"])
    obs_h = obs15.reshape(len(fips), PREFIX_H, 4).any(-1)
    g3 = p["in_service"][:, 0]
    ok = in_data_coverage(df, w0, origin.year)
    st = dict(zip(counties().fips, counties().state))
    g2b = np.array([ok.get(st.get(f, ""), True) for f in fips])
    return pd.DataFrame({"fips": fips, "G3": g3, "prefix_obs": obs_h.mean(1), "G4": obs_h.mean(1) >= MIN_PREFIX_OBS,
                         "G5": obs_h[:, -1], "G2_in_data": g2b})
