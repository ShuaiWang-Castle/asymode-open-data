"""NOAA Storm Events Database -> county-level event windows.

Public bulk CSVs, no account required:
  https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/

Used to choose which storms and which counties enter the panel, so that event
selection is a stated, reproducible rule rather than a hand-picked list.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd

# Event types that plausibly interrupt distribution-level service. Kept broad on
# purpose: narrowing it is a modelling choice that belongs in a config, not here.
OUTAGE_RELEVANT = {
    "Thunderstorm Wind", "High Wind", "Strong Wind", "Tornado", "Hurricane",
    "Hurricane (Typhoon)", "Tropical Storm", "Ice Storm", "Winter Storm",
    "Heavy Snow", "Blizzard", "Lightning", "Hail", "Flash Flood", "Flood",
    "Extreme Cold/Wind Chill", "Winter Weather",
}


def _read_one(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt", encoding="latin-1") as fh:
        df = pd.read_csv(fh, low_memory=False)
    df.columns = [c.strip().upper() for c in df.columns]
    return df


def load_details(paths: list[Path]) -> pd.DataFrame:
    """Concatenate yearly detail files and derive a county FIPS and UTC window."""
    frames = [_read_one(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)

    # STATE_FIPS/CZ_FIPS are the county code only when CZ_TYPE == 'C'. Zone rows
    # ('Z') are forecast zones, which do not map one-to-one onto counties and are
    # dropped rather than approximated.
    df = df[df["CZ_TYPE"].astype(str).str.upper() == "C"].copy()
    df["fips"] = (df["STATE_FIPS"].astype("Int64").astype(str).str.zfill(2)
                  + df["CZ_FIPS"].astype("Int64").astype(str).str.zfill(3))

    for col, out in (("BEGIN_DATE_TIME", "t_begin"), ("END_DATE_TIME", "t_end")):
        df[out] = pd.to_datetime(df[col], format="%d-%b-%y %H:%M:%S", errors="coerce")

    # Storm Events timestamps are local; CZ_TIMEZONE like 'EST-5' gives the offset.
    off = df["CZ_TIMEZONE"].astype(str).str.extract(r"(-?\d+)")[0].astype(float)
    off = off.fillna(0.0)
    df["t_begin_utc"] = df["t_begin"] - pd.to_timedelta(off, unit="h")
    df["t_end_utc"] = df["t_end"] - pd.to_timedelta(off, unit="h")

    keep = ["EPISODE_ID", "EVENT_ID", "STATE", "fips", "CZ_NAME", "EVENT_TYPE",
            "t_begin_utc", "t_end_utc", "MAGNITUDE", "MAGNITUDE_TYPE",
            "DEATHS_DIRECT", "INJURIES_DIRECT", "DAMAGE_PROPERTY"]
    return df[[c for c in keep if c in df.columns]].dropna(subset=["t_begin_utc"])


def county_event_days(df: pd.DataFrame, types: set[str] | None = None) -> pd.DataFrame:
    """One row per (fips, UTC day) with counts of outage-relevant events."""
    d = df[df["EVENT_TYPE"].isin(types or OUTAGE_RELEVANT)].copy()
    d["day"] = d["t_begin_utc"].dt.floor("D")
    g = (d.groupby(["fips", "day"])
           .agg(n_events=("EVENT_ID", "size"),
                n_types=("EVENT_TYPE", "nunique"),
                states=("STATE", "first"))
           .reset_index())
    return g


def rank_episodes(df: pd.DataFrame, types: set[str] | None = None,
                  min_counties: int = 20) -> pd.DataFrame:
    """Rank storm episodes by county footprint -- the event-selection rule.

    Footprint, not damage dollars: the panel needs many counties observed under
    one synoptic system, which is what makes county-held-out evaluation mean
    anything. Damage is recorded for context only.
    """
    d = df[df["EVENT_TYPE"].isin(types or OUTAGE_RELEVANT)].copy()
    g = (d.groupby("EPISODE_ID")
           .agg(n_counties=("fips", "nunique"),
                n_states=("STATE", "nunique"),
                n_events=("EVENT_ID", "size"),
                t_start=("t_begin_utc", "min"),
                t_stop=("t_end_utc", "max"),
                types=("EVENT_TYPE", lambda s: ",".join(sorted(set(s))[:4])),
                states=("STATE", lambda s: ",".join(sorted(set(s))[:6])))
           .reset_index())
    g["dur_h"] = (g["t_stop"] - g["t_start"]).dt.total_seconds() / 3600.0
    g = g[g["n_counties"] >= min_counties]
    return g.sort_values("n_counties", ascending=False).reset_index(drop=True)
