"""Winter event panel W1: ice-storm episodes, selected from NOAA Storm Events metadata only (no outage data is
read here), for the mechanisms the twelve-event wind panel cannot test (glaze and wet-snow loading, where the
elevation of the customers relative to the model orography moves the phase).

Rules, fixed before any panel is built:
  pool      catalogue days 2018-01-01..2024-12-31 (the EAGLE-I years on disk); for day d, the counties with an
            'Ice Storm' report beginning in [d, d + 3 days)
  rank      by that county count; greedy local maxima at least 10 days apart; keep counts >= 40
  exclude   windows that overlap the twelve wind-panel windows; episodes whose outages are dominated by operator
            load shedding (rotating outages ordered in an energy emergency): 2021-02-13 (ERCOT, SPP and MISO
            rotating outages, 2021-02-15..18)
  window    216 hourly steps from 00 UTC three days before the catalogue day (origin = the catalogue day, as in
            the wind panel)
  footprint the counties with a winter report (Ice Storm, Winter Storm, Heavy Snow, Winter Weather, Sleet,
            Freezing Fog, High Wind, Strong Wind) beginning in forecast hours 72-215; the data gates of
            build_panel216.py apply later
Output: selected_events_winter.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SE = ROOT / "data" / "interim" / "storm_events_county.parquet"
MIN_COUNTIES, GAP_DAYS = 40, 10
EXCLUDE = {"2021-02-13": "outages dominated by operator load shedding (ERCOT, SPP and MISO rotating outages, "
                         "2021-02-15..18)"}
FOOT = ["Ice Storm", "Winter Storm", "Heavy Snow", "Winter Weather", "Sleet", "Freezing Fog", "High Wind", "Strong Wind"]


def main():
    se = pd.read_parquet(SE)
    se["t"] = pd.to_datetime(se.t_begin_utc)
    se = se[(se.t >= "2018-01-01") & (se.t < "2025-01-01")]
    ice = se[se.EVENT_TYPE == "Ice Storm"]
    days = pd.date_range("2018-01-01", "2024-12-31", freq="D")
    cnt = {d: ice[(ice.t >= d) & (ice.t < d + pd.Timedelta(days=3))].fips.nunique() for d in days}
    order = sorted((d for d in days if cnt[d] >= MIN_COUNTIES), key=lambda d: (-cnt[d], d))
    wind = json.loads((ROOT / "experiments/open_gcrk_20260919/selected_events_e3.json").read_text())["events"]
    wwin = [(pd.Timestamp(e["window_start_utc"]), pd.Timestamp(e["window_start_utc"]) + pd.Timedelta(hours=215)) for e in wind]
    taken, log = [], []
    for d in order:
        if any(abs((d - t).days) < GAP_DAYS for t in taken):
            continue
        day = d.strftime("%Y-%m-%d")
        s, e = d - pd.Timedelta(days=3), d - pd.Timedelta(days=3) + pd.Timedelta(hours=215)
        if day in EXCLUDE:
            log.append(dict(day=day, ice_counties=int(cnt[d]), decision="excluded: " + EXCLUDE[day])); taken.append(d); continue
        if any(s <= b and a <= e for a, b in wwin):
            log.append(dict(day=day, ice_counties=int(cnt[d]), decision="excluded: overlaps a wind-panel window")); taken.append(d); continue
        taken.append(d)
        fc = se[(se.t >= d) & (se.t < d + pd.Timedelta(hours=144)) & se.EVENT_TYPE.isin(FOOT)]
        log.append(dict(day=day, ice_counties=int(cnt[d]), decision="selected", window_start_utc=str(s),
                        window_end_utc=str(e), footprint_counties=int(fc.fips.nunique()),
                        footprint_states=int(fc.fips.str[:2].nunique()), footprint_fips=sorted(fc.fips.unique().tolist()),
                        top_types=fc.EVENT_TYPE.value_counts().head(4).index.tolist()))
    ev = [dict(event=r["day"], window_start_utc=r["window_start_utc"], window_end_utc=r["window_end_utc"],
               footprint_counties=r["footprint_counties"], top_types=r["top_types"], footprint_fips=r["footprint_fips"])
          for r in log if r["decision"] == "selected"]
    out = dict(rule=__doc__.split("Rules, fixed before any panel is built:")[1].split("Output:")[0].strip(),
               events=sorted(ev, key=lambda r: r["event"]), log=log)
    (HERE / "selected_events_winter.json").write_text(json.dumps(out, indent=1) + "\n")
    for r in log:
        print({k: v for k, v in r.items() if k != "footprint_fips"})


if __name__ == "__main__":
    main()
