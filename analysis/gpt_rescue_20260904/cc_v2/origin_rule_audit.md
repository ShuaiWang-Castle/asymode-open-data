# Origin rule audit

## Rule as specified

Three outcome-blind anchors per panel: first NOAA event time − 6 h; the midpoint of
the NOAA event interval; last NOAA event time + 6 h. Each is rounded to the nearest
hourly origin having at least 24 h of past context and 24 h of future target.
Duplicates are dropped and never replaced by inspecting the outage curve.

## Metadata source, and one that could not be used

Anchors come from `data/interim/storm_events_county.parquet` (`t_begin_utc`,
`t_end_utc`), restricted to each panel's own counties and its own time window.
All 26/26 panels have NOAA rows there.

`data/interim/county_event_days.parquet` could **not** be used: it carries only
`fips, day, n_events, n_types, states`. A calendar day cannot place an anchor on an
hour.

The NOAA interval is clipped to the panel window. Long-duration rows — floods and
winter advisories — carry end times weeks after their begin; without clipping this
produced reported intervals up to 893 h inside a 168 h panel and put the midpoint
outside the panel entirely.

## Legal origin range

Every panel is 169 hourly steps (168 h). With 24 h of past context and 24 h of
future target, the legal origin index range is **[24, 143]** for all 26 panels.

## Result: executable, but degenerate on this cohort

| quantity | value |
|---|---|
| panels retaining all three anchors | 26/26 |
| anchors clipped into the legal range | **48/78** (62%) |
| panels whose `pre` anchor is exactly index 24 | **24/26** |
| panels whose `post` anchor is exactly index 143 | **25/26** |
| distinct midpoint indices | 13 (range 58–102) |
| median clipped NOAA interval | 162 h of a 168 h panel |

**Read this before the main run.** Panels are a storm day plus roughly three days
either side, and across a large multi-state county footprint the NOAA rows span
nearly the whole window. "First event − 6 h" therefore falls before the legal range
and clips to 24; "last event + 6 h" falls after it and clips to 143, for almost every
panel. Only the midpoint carries event-specific information.

The rule thus produces a nearly fixed grid `[24, ~85, 143]` rather than event-centred
origins, and 9 of 26 panels share the identical triple `(24, 84, 143)`. The stated
motivation for replacing the legacy full-window grid — that it "overweights quiet
pre-storm and late-recovery periods" — is **not achieved** on this cohort: index 24 is
24 h into the pre-storm period and index 143 sits deep in the recovery tail.

The rule was executed exactly as written. No replacement was improvised. This is
recorded as a design finding requiring adjudication before the 26-event main run.

## Per-panel audit

| event | noaa_begin | noaa_end | legal_lo | legal_hi | pre | pre_clip | mid | mid_clip | post | post_clip | kept | dropped |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2018-01-16 | 2018-01-14 00:00 | 2018-01-18 19:45 | 24 | 143 | 24 | True | 58 | False | 122 | False | [24, 58, 122] | - |
| 2018-10-11 | 2018-10-10 06:00 | 2018-10-15 23:00 | 24 | 143 | 24 | False | 98 | False | 143 | True | [24, 98, 143] | - |
| 2019-02-20 | 2019-02-18 01:00 | 2019-02-24 23:00 | 24 | 143 | 24 | True | 84 | False | 143 | True | [24, 84, 143] | - |
| 2019-02-24 | 2019-02-22 02:00 | 2019-02-28 23:00 | 24 | 143 | 24 | True | 84 | False | 143 | True | [24, 84, 143] | - |
| 2019-11-27 | 2019-11-25 17:56 | 2019-12-01 23:00 | 24 | 143 | 24 | True | 92 | False | 143 | True | [24, 92, 143] | - |
| 2020-02-06 | 2020-02-04 20:00 | 2020-02-10 23:00 | 24 | 143 | 24 | True | 93 | False | 143 | True | [24, 93, 143] | - |
| 2020-08-04 | 2020-08-02 09:00 | 2020-08-08 16:06 | 24 | 143 | 24 | True | 85 | False | 143 | True | [24, 85, 143] | - |
| 2020-10-29 | 2020-10-27 05:00 | 2020-11-02 23:00 | 24 | 143 | 24 | True | 86 | False | 143 | True | [24, 86, 143] | - |
| 2021-02-15 | 2021-02-13 02:00 | 2021-02-19 23:00 | 24 | 143 | 24 | True | 84 | False | 143 | True | [24, 84, 143] | - |
| 2021-05-04 | 2021-05-02 20:45 | 2021-05-08 23:00 | 24 | 143 | 24 | True | 94 | False | 143 | True | [24, 94, 143] | - |
| 2021-06-21 | 2021-06-19 03:36 | 2021-06-25 23:00 | 24 | 143 | 24 | True | 85 | False | 143 | True | [24, 85, 143] | - |
| 2021-08-11 | 2021-08-09 00:50 | 2021-08-15 23:00 | 24 | 143 | 24 | True | 84 | False | 143 | True | [24, 84, 143] | - |
| 2021-12-11 | 2021-12-09 06:00 | 2021-12-15 23:00 | 24 | 143 | 24 | True | 86 | False | 143 | True | [24, 86, 143] | - |
| 2021-12-15 | 2021-12-13 05:00 | 2021-12-19 23:00 | 24 | 143 | 24 | True | 86 | False | 143 | True | [24, 86, 143] | - |
| 2022-01-16 | 2022-01-15 12:30 | 2022-01-20 23:00 | 24 | 143 | 30 | False | 102 | False | 143 | True | [30, 102, 143] | - |
| 2022-03-12 | 2022-03-10 07:00 | 2022-03-16 23:00 | 24 | 143 | 24 | True | 87 | False | 143 | True | [24, 87, 143] | - |
| 2022-04-13 | 2022-04-11 08:05 | 2022-04-17 23:00 | 24 | 143 | 24 | True | 88 | False | 143 | True | [24, 88, 143] | - |
| 2022-06-08 | 2022-06-06 00:38 | 2022-06-12 23:00 | 24 | 143 | 24 | True | 84 | False | 143 | True | [24, 84, 143] | - |
| 2022-06-17 | 2022-06-15 17:00 | 2022-06-21 23:00 | 24 | 143 | 24 | True | 92 | False | 143 | True | [24, 92, 143] | - |
| 2022-07-23 | 2022-07-21 00:43 | 2022-07-27 23:00 | 24 | 143 | 24 | True | 84 | False | 143 | True | [24, 84, 143] | - |
| 2024-01-09 | 2024-01-08 12:00 | 2024-01-13 23:00 | 24 | 143 | 30 | False | 101 | False | 143 | True | [30, 101, 143] | - |
| 2024-01-12 | 2024-01-10 00:00 | 2024-01-16 23:00 | 24 | 143 | 24 | True | 83 | False | 143 | True | [24, 83, 143] | - |
| 2024-05-08 | 2024-05-06 01:00 | 2024-05-12 23:00 | 24 | 143 | 24 | True | 84 | False | 143 | True | [24, 84, 143] | - |
| 2024-05-26 | 2024-05-24 00:15 | 2024-05-30 23:00 | 24 | 143 | 24 | True | 84 | False | 143 | True | [24, 84, 143] | - |
| 2024-06-26 | 2024-06-24 07:00 | 2024-06-30 23:00 | 24 | 143 | 24 | True | 87 | False | 143 | True | [24, 87, 143] | - |
| 2024-09-27 | 2024-09-25 00:01 | 2024-10-01 23:00 | 24 | 143 | 24 | True | 84 | False | 143 | True | [24, 84, 143] | - |
