"""Build the public storm-event catalog that drives panel construction."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from asymode.storm_events import load_details, rank_episodes, county_event_days

raw = sorted((ROOT / "data" / "raw" / "storm_events").glob("StormEvents_details*.csv.gz"))
print(f"reading {len(raw)} yearly files")
df = load_details(raw)
print(f"county-typed event rows: {len(df):,}  span {df.t_begin_utc.min()} .. {df.t_begin_utc.max()}")

out = ROOT / "data" / "interim"; out.mkdir(parents=True, exist_ok=True)
df.to_parquet(out / "storm_events_county.parquet", index=False)

ep = rank_episodes(df, min_counties=30)
ep.to_parquet(out / "storm_episodes_ranked.parquet", index=False)
ced = county_event_days(df)
ced.to_parquet(out / "county_event_days.parquet", index=False)

print(f"\nepisodes with >=30 counties: {len(ep):,}")
print(f"county-days with outage-relevant events: {len(ced):,}  "
      f"distinct counties: {ced.fips.nunique():,}")
print("\n=== top 15 episodes by county footprint ===")
with pd.option_context("display.width", 200, "display.max_colwidth", 42):
    print(ep.head(15)[["EPISODE_ID", "n_counties", "n_states", "dur_h",
                       "t_start", "types", "states"]].to_string(index=False))
