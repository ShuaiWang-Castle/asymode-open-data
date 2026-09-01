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


def load_zone_county(path: Path) -> pd.DataFrame:
    """NWS forecast zone to county correlation, pipe delimited, no header.

    Needed because tropical cyclones, winter storms and high-wind events are filed
    against forecast *zones*, not counties. A county-only filter silently discards
    every hurricane in the record, which is the opposite of what a study of slow
    outage dynamics wants.
    """
    cols = ["state", "zone", "cwa", "name", "state_zone", "county", "fips",
            "tz", "fe_area", "lat", "lon"]
    z = pd.read_csv(path, sep="|", names=cols, dtype=str, encoding="latin-1")
    z["fips"] = z["fips"].str.strip().str.zfill(5)
    z["state_zone"] = z["state_zone"].str.strip().str.upper()
    z["state"] = z["state"].str.strip().str.upper()
    return z[["state_zone", "fips", "state"]].dropna().drop_duplicates()


def load_details(paths: list[Path], zone_county: Path | None = None) -> pd.DataFrame:
    """Concatenate yearly detail files and derive a county FIPS and UTC window.

    County-coded rows (`CZ_TYPE == 'C'`) carry the county directly. Zone-coded rows
    (`'Z'`) are expanded to every county the zone covers, which duplicates the event
    across counties -- correct for footprint counting, and flagged in `cz_type` so a
    downstream user can tell the two apart.
    """
    frames = [_read_one(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df["CZ_TYPE"] = df["CZ_TYPE"].astype(str).str.upper()

    cty = df[df["CZ_TYPE"] == "C"].copy()
    cty["fips"] = (cty["STATE_FIPS"].astype("Int64").astype(str).str.zfill(2)
                   + cty["CZ_FIPS"].astype("Int64").astype(str).str.zfill(3))

    if zone_county is not None and Path(zone_county).exists():
        zc = load_zone_county(Path(zone_county))
        # Storm Events spells the state out; the zone table abbreviates it. The
        # zone table's own county FIPS supplies the bridge, so no hard-coded
        # state list is needed and none can drift.
        abbr = (zc.assign(sf=zc["fips"].str[:2])
                  .drop_duplicates("sf").set_index("sf")["state"])
        zon = df[df["CZ_TYPE"] == "Z"].copy()
        sf = zon["STATE_FIPS"].astype("Int64").astype(str).str.zfill(2)
        zon["state_zone"] = (sf.map(abbr).fillna("")
                             + zon["CZ_FIPS"].astype("Int64").astype(str).str.zfill(3))
        zon = zon.merge(zc[["state_zone", "fips"]], on="state_zone", how="inner")
        df = pd.concat([cty, zon], ignore_index=True)
    else:
        df = cty

    for col, out in (("BEGIN_DATE_TIME", "t_begin"), ("END_DATE_TIME", "t_end")):
        df[out] = pd.to_datetime(df[col], format="%d-%b-%y %H:%M:%S", errors="coerce")

    # Storm Events timestamps are local; CZ_TIMEZONE like 'EST-5' gives the offset.
    off = df["CZ_TIMEZONE"].astype(str).str.extract(r"(-?\d+)")[0].astype(float)
    off = off.fillna(0.0)
    df["t_begin_utc"] = df["t_begin"] - pd.to_timedelta(off, unit="h")
    df["t_end_utc"] = df["t_end"] - pd.to_timedelta(off, unit="h")

    df = df.rename(columns={"CZ_TYPE": "cz_type"})
    keep = ["EPISODE_ID", "EVENT_ID", "STATE", "fips", "CZ_NAME", "EVENT_TYPE",
            "cz_type", "t_begin_utc", "t_end_utc", "MAGNITUDE", "MAGNITUDE_TYPE",
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
