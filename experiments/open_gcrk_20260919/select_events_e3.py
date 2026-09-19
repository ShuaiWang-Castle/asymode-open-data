"""E3 event set (PREREG Amendment 2): twelve events from the pre-registered candidate pool.

Rules F2 (scale), F4 (multi-day), F5 (data) and R1 (second wave) are kept from `event_selection.csv`;
F1 (convective gusts) and F3 (calm prefix) are dropped. The five pre-registered events are kept and
the other candidates are added in decreasing order of forecast-window wind-report counties (ties by
date), skipping any window that overlaps one already taken, until there are twelve. Footprints are
defined exactly as in `select_events.py` (wind reports beginning in forecast hours 72-215). Public
Storm Events metadata only. Output: selected_events_e3.json.
"""
from __future__ import annotations

import json

import pandas as pd

import select_events as SE

N_EVENTS = 12
# Candidates that pass the metadata rules but leave no county after the data gates of
# build_panel216.py; they fail F5 (data) in substance and the next candidate takes their place.
DATA_GATE_FAILED = {"2024-05-26": "gate G4 leaves no county: an EAGLE-I collection gap of about 9.5 h on "
                                  "2024-05-24 puts every county's prefix below 90% observed hours"}


def main():
    tab = pd.read_csv(SE.HERE / "event_selection.csv")
    keep = tab.F2_scale & tab.F4_multiday & tab.F5_data & tab.R1_second_wave
    first = [e["event"] for e in json.loads((SE.HERE / "selected_events.json").read_text())["events"]]
    win = {r.day: (pd.Timestamp(r.window_start_utc), pd.Timestamp(r.window_end_utc)) for r in tab.itertuples()}
    taken = list(first)
    for r in tab[keep].sort_values(["fc_wind_counties", "day"], ascending=[False, True]).itertuples():
        if len(taken) >= N_EVENTS:
            break
        if r.day in taken or r.day in DATA_GATE_FAILED:
            continue
        s, e = win[r.day]
        if any(s <= win[a][1] and win[a][0] <= e for a in taken):
            continue
        taken.append(r.day)
    se = pd.read_parquet(SE.INTERIM / "storm_events_county.parquet")
    out = []
    for d in sorted(taken):
        s, e = win[d]
        fc = se[(se.t_begin_utc >= s + pd.Timedelta(hours=SE.PREFIX_H)) & (se.t_begin_utc < s + pd.Timedelta(hours=SE.WINDOW_H))
                & se.EVENT_TYPE.isin(SE.WIND)]
        out.append(dict(event=d, window_start_utc=str(s), window_end_utc=str(e), pre_registered=d in first,
                        footprint_fips=sorted(fc.fips.unique().tolist()), n_footprint=int(fc.fips.nunique()),
                        states=sorted(fc.STATE.unique().tolist())))
    (SE.HERE / "selected_events_e3.json").write_text(json.dumps(dict(rule="see select_events_e3.py docstring",
                                                                    replaced=DATA_GATE_FAILED, events=out), indent=1) + "\n")
    for o in out:
        print(o["event"], "pre-registered" if o["pre_registered"] else "added", o["n_footprint"], len(o["states"]))


if __name__ == "__main__":
    main()
