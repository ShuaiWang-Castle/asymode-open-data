"""Panel W2e (PREREG_W2.md amendment 3): W2d plus new near-freezing winter-storm episodes, Storm Events metadata only.

For day d in 2014-11-01..2025-12-31: ice = counties with an Ice Storm report beginning in [d, d + 3 days); snow =
counties with a Heavy Snow, Winter Storm or Blizzard report in the same span; the day qualifies if ice >= 10 or
snow >= 300. Episodes are greedy local maxima of the count of counties with any of the four types, >= 10 days
apart. Exclusions: windows that overlap the wind panel, W1 or W2d; load-shedding episodes (2021-02-13, 2022-12-21).
Window and footprint as select_events_winter.py. The EAGLE-I coverage gate and the data gates are applied by
build_winter.py. Output: selected_events_w2e.json (the W2d events followed by the new ones).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ICE, SNOW = ["Ice Storm"], ["Heavy Snow", "Winter Storm", "Blizzard"]
FOOT = ["Ice Storm", "Winter Storm", "Heavy Snow", "Winter Weather", "Sleet", "Freezing Fog", "High Wind", "Strong Wind"]
EXCLUDE = {"2021-02-13": "outages dominated by operator load shedding (ERCOT, SPP, MISO rotating outages)",
           "2022-12-21": "outages dominated by operator load shedding in part of the footprint (TVA and Duke rotating "
                         "outages on 2022-12-24)"}


def main():
    se = pd.concat([pd.read_parquet(ROOT / "data/interim/storm_events_county_2014_2017.parquet"),
                    pd.read_parquet(ROOT / "data/interim/storm_events_county.parquet")], ignore_index=True)
    se["t"] = pd.to_datetime(se.t_begin_utc)
    se = se[(se.t >= "2014-11-01") & (se.t < "2026-01-01")]
    days = pd.date_range("2014-11-01", "2025-12-31", freq="D")

    def count(types):
        s = se[se.EVENT_TYPE.isin(types)]
        return {d: s[(s.t >= d) & (s.t < d + pd.Timedelta(days=3))].fips.nunique() for d in days}
    ice, snow, anyw = count(ICE), count(SNOW), count(ICE + SNOW)
    qual = [d for d in days if ice[d] >= 10 or snow[d] >= 300]
    order = sorted(qual, key=lambda d: (-anyw[d], d))
    prev = []
    for f in ("../open_gcrk_20260919/selected_events_e3.json", "selected_events_winter.json", "selected_events_w2d.json"):
        prev += json.loads((HERE / f).read_text())["events"]
    busy = [(pd.Timestamp(e["window_start_utc"]), pd.Timestamp(e["window_start_utc"]) + pd.Timedelta(hours=215)) for e in prev]
    w2d = json.loads((HERE / "selected_events_w2d.json").read_text())["events"]
    taken, log, new = [], [], []
    for d in order:
        if any(abs((d - t).days) < 10 for t in taken):
            continue
        taken.append(d)
        day = d.strftime("%Y-%m-%d")
        s0 = d - pd.Timedelta(days=3); e0 = s0 + pd.Timedelta(hours=215)
        rec = dict(day=day, ice_counties=int(ice[d]), snow_counties=int(snow[d]), winter_counties=int(anyw[d]))
        if day in EXCLUDE:
            log.append(dict(rec, decision="excluded: " + EXCLUDE[day])); continue
        if any(s0 <= b and a <= e0 for a, b in busy):
            log.append(dict(rec, decision="excluded: overlaps the wind panel, W1 or W2d")); continue
        fc = se[(se.t >= d) & (se.t < d + pd.Timedelta(hours=144)) & se.EVENT_TYPE.isin(FOOT)]
        ev = dict(event=day, window_start_utc=str(s0), window_end_utc=str(e0), footprint_counties=int(fc.fips.nunique()),
                  top_types=fc.EVENT_TYPE.value_counts().head(4).index.tolist(), footprint_fips=sorted(fc.fips.unique().tolist()))
        new.append(ev); log.append(dict(rec, decision="selected (new)"))
    out = dict(rule=__doc__.split("\n\n")[0], events=w2d + sorted(new, key=lambda r: r["event"]), log=log)
    (HERE / "selected_events_w2e.json").write_text(json.dumps(out, indent=1) + "\n")
    for r in log:
        print(r)
    print(len(w2d), "W2d events +", len(new), "new")


if __name__ == "__main__":
    main()
