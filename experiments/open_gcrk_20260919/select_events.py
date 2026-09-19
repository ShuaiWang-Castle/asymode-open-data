"""Event selection for the open-data GCRK study: a fixed rule on public metadata only.

Inputs are NOAA Storm Events (county-coded, zone records expanded to counties) and
the event-day catalogue built by scripts/build_event_catalog.py. No outage value is
read here, before or after the forecast origin, and no model output exists yet.

Candidate pool
  * the 11 convective days of configs/panel_manifest_g2-convective-11.json;
  * every day of data/interim/event_days_stratified.parquet whose dominant family is
    "wind", restricted to years whose EAGLE-I outage records are on disk
    (2018-2022 up to the archive end on 2022-11-12, and 2024).

Window (fixed before any outcome is seen): 216 hourly steps starting at 00 UTC
three days before the catalogue day D.  Hours 0-71 are the observed prefix; hours
72-215 (00 UTC of D onward) are the forecast window.

Wind-type reports: Thunderstorm Wind, High Wind, Strong Wind, Tornado.
Convective reports: Thunderstorm Wind, Tornado.
Wet/cold reports: Heavy Rain, Flash Flood, Flood, Heavy Snow, Winter Storm,
Winter Weather, Ice Storm, Blizzard, Lake-Effect Snow, Sleet.
All counts are distinct counties (zone-expanded rows would otherwise inflate
report counts of zone-coded types).

Hard filters
  F1 gust-driven: >= 50% of the counties with an outage-relevant report in the
     forecast window have a wind-type report, and convective reports cover >= 100
     counties (gust fronts / squall lines present, not a purely synoptic wind);
  F2 scale: wind-type reports in >= 5 states and >= 100 counties (forecast window);
  F3 calm prefix: wind-type counties in the prefix <= 25% of those in the forecast;
  F4 multi-day: the 5th-95th percentile span of wind-type report start times in the
     forecast window is >= 24 h;
  F5 data: the window lies inside the EAGLE-I record on disk.
Preference
  R1 second wave: a UTC day >= 24 h after the first peak day of wind-type counties
     with >= 25% of that peak, or wet/cold reports in >= 50 counties after the
     first peak day.
Selection: candidates passing F1-F5 and R1, in decreasing order of forecast-window
wind-type counties, accepted greedily when their window does not overlap an already
accepted one; at most six.

County set of an accepted event (data gates are applied later, in build_panel216.py):
every county with >= 1 wind-type report starting inside the forecast window.

Writes event_selection.csv (every candidate, its metrics, flags and a one-line
reason) and selected_events.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INTERIM = ROOT / "data" / "interim"

WIND = {"Thunderstorm Wind", "High Wind", "Strong Wind", "Tornado"}
CONVECTIVE = {"Thunderstorm Wind", "Tornado"}
WET_COLD = {"Heavy Rain", "Flash Flood", "Flood", "Heavy Snow", "Winter Storm", "Winter Weather",
            "Ice Storm", "Blizzard", "Lake-Effect Snow", "Sleet"}
OUTAGE_RELEVANT = WIND | WET_COLD | {"Hail", "Lightning", "Tropical Storm", "Hurricane",
                                     "Hurricane (Typhoon)", "Extreme Cold/Wind Chill"}
PREFIX_H, WINDOW_H = 72, 216
EAGLEI_END = {2022: pd.Timestamp("2022-11-12")}
MAX_EVENTS = 6


def window_start(day: pd.Timestamp) -> pd.Timestamp:
    return day.normalize() - pd.Timedelta(days=3)


def eaglei_years() -> set[int]:
    return {int(p.stem.split("_")[-1]) for p in INTERIM.glob("eaglei_outages_*.parquet")}


def candidates() -> pd.DataFrame:
    g2 = json.loads((ROOT / "configs/panel_manifest_g2-convective-11.json").read_text())["panels"]
    cat = pd.read_parquet(INTERIM / "event_days_stratified.parquet")
    years = eaglei_years()
    wind = cat[(cat.dominant == "wind") & cat.yr.isin(years)]
    rows = [dict(day=pd.Timestamp(d), pool="g2 convective") for d in g2]
    rows += [dict(day=pd.Timestamp(d), pool="catalogue wind") for d in wind.day]
    out = pd.DataFrame(rows).drop_duplicates("day").sort_values("day").reset_index(drop=True)
    out["eaglei_on_disk"] = out.day.dt.year.isin(years)
    return out


def metrics(se: pd.DataFrame, day: pd.Timestamp) -> dict:
    s = window_start(day)
    w = se[(se.t_begin_utc >= s) & (se.t_begin_utc < s + pd.Timedelta(hours=WINDOW_H))
           & se.EVENT_TYPE.isin(OUTAGE_RELEVANT)].copy()
    w["h"] = ((w.t_begin_utc - s).dt.total_seconds() // 3600).astype(int)
    pre, fc = w[w.h < PREFIX_H], w[w.h >= PREFIX_H]
    fw, pw = fc[fc.EVENT_TYPE.isin(WIND)], pre[pre.EVENT_TYPE.isin(WIND)]
    fconv = fc[fc.EVENT_TYPE.isin(CONVECTIVE)]
    daily = [fw[(fw.h >= PREFIX_H + 24 * k) & (fw.h < PREFIX_H + 24 * (k + 1))].fips.nunique() for k in range(6)]
    first_peak = int(np.argmax(daily))
    later = daily[first_peak + 1:]
    second_wind = bool(later) and max(later) >= 0.25 * daily[first_peak]
    wetcold_after = fc[fc.EVENT_TYPE.isin(WET_COLD) & (fc.h >= PREFIX_H + 24 * (first_peak + 1))].fips.nunique()
    span = float(np.subtract(*np.percentile(fw.h, [95, 5]))) if len(fw) else 0.0
    n_rel = fc.fips.nunique()
    return dict(window_start_utc=str(s), window_end_utc=str(s + pd.Timedelta(hours=WINDOW_H - 1)),
                prefix_wind_counties=pw.fips.nunique(), fc_wind_counties=fw.fips.nunique(),
                fc_wind_states=fw.STATE.nunique(), fc_convective_counties=fconv.fips.nunique(),
                fc_relevant_counties=n_rel,
                fc_wind_county_share=round(fw.fips.nunique() / n_rel, 3) if n_rel else 0.0,
                prefix_to_fc_ratio=round(pw.fips.nunique() / max(fw.fips.nunique(), 1), 3),
                fc_span_5_95_h=span, fc_wind_counties_by_day=daily, first_peak_day=first_peak,
                second_wind_wave=second_wind, wetcold_counties_after_peak=wetcold_after,
                fc_top_types=",".join(fc.EVENT_TYPE.value_counts().head(4).index))


def reason(r: dict) -> str:
    fails = []
    if not r["F5_data"]:
        fails.append("EAGLE-I record not on disk for the window")
    if not r["F1_gust_driven"]:
        fails.append(f"not gust-driven with convection (wind share {r['fc_wind_county_share']:.2f}, "
                     f"convective counties {r['fc_convective_counties']})")
    if not r["F2_scale"]:
        fails.append("too small")
    if not r["F3_calm_prefix"]:
        fails.append(f"active prefix (prefix/forecast wind counties {r['prefix_to_fc_ratio']:.2f})")
    if not r["F4_multiday"]:
        fails.append(f"single burst (wind span {r['fc_span_5_95_h']:.0f} h)")
    if not fails and not r["R1_second_wave"]:
        fails.append("no second wave")
    return "; ".join(fails)


def main():
    se = pd.read_parquet(INTERIM / "storm_events_county.parquet")
    rows = []
    for c in candidates().itertuples():
        m = metrics(se, c.day)
        s, e = pd.Timestamp(m["window_start_utc"]), pd.Timestamp(m["window_end_utc"])
        r = dict(day=str(c.day.date()), pool=c.pool, **m)
        r["F5_data"] = bool(c.eaglei_on_disk) and e <= EAGLEI_END.get(c.day.year, e)
        r["F1_gust_driven"] = m["fc_wind_county_share"] >= 0.5 and m["fc_convective_counties"] >= 100
        r["F2_scale"] = m["fc_wind_states"] >= 5 and m["fc_wind_counties"] >= 100
        r["F3_calm_prefix"] = m["prefix_to_fc_ratio"] <= 0.25
        r["F4_multiday"] = m["fc_span_5_95_h"] >= 24
        r["R1_second_wave"] = m["second_wind_wave"] or m["wetcold_counties_after_peak"] >= 50
        r["passes"] = all(r[k] for k in ("F1_gust_driven", "F2_scale", "F3_calm_prefix", "F4_multiday",
                                         "F5_data", "R1_second_wave"))
        r["reason"] = reason(r)
        rows.append(r)
    tab = pd.DataFrame(rows)
    accepted, windows = [], []
    for r in tab[tab.passes].sort_values(["fc_wind_counties", "day"], ascending=[False, True]).itertuples():
        s, e = pd.Timestamp(r.window_start_utc), pd.Timestamp(r.window_end_utc)
        clash = [a for a, (s2, e2) in zip(accepted, windows) if s <= e2 and s2 <= e]
        if clash:
            tab.loc[tab.day == r.day, "reason"] = f"window overlaps accepted event {clash[0]}"
            continue
        if len(accepted) < MAX_EVENTS:
            accepted.append(r.day); windows.append((s, e))
    tab["selected"] = tab.day.isin(accepted)
    tab.loc[tab.selected, "reason"] = "selected: passes F1-F5 and R1"
    tab.to_csv(HERE / "event_selection.csv", index=False)
    sel = []
    for d in sorted(accepted):
        r = tab[tab.day == d].iloc[0]
        s = pd.Timestamp(r.window_start_utc)
        fc = se[(se.t_begin_utc >= s + pd.Timedelta(hours=PREFIX_H)) & (se.t_begin_utc < s + pd.Timedelta(hours=WINDOW_H))
                & se.EVENT_TYPE.isin(WIND)]
        sel.append(dict(event=d, window_start_utc=r.window_start_utc, window_end_utc=r.window_end_utc,
                        footprint_fips=sorted(fc.fips.unique().tolist()), n_footprint=int(fc.fips.nunique()),
                        states=sorted(fc.STATE.unique().tolist())))
    (HERE / "selected_events.json").write_text(json.dumps(dict(rule="see select_events.py docstring",
                                                               events=sel), indent=1) + "\n")
    cols = ["day", "pool", "fc_wind_counties", "fc_convective_counties", "fc_wind_states", "fc_wind_county_share",
            "prefix_to_fc_ratio", "fc_span_5_95_h", "fc_wind_counties_by_day", "wetcold_counties_after_peak",
            "passes", "selected", "reason"]
    with pd.option_context("display.width", 250, "display.max_colwidth", 90, "display.max_rows", 100):
        print(tab[cols].to_string(index=False))
    print("\nselected:", [(e["event"], e["n_footprint"], len(e["states"])) for e in sel])


if __name__ == "__main__":
    main()
