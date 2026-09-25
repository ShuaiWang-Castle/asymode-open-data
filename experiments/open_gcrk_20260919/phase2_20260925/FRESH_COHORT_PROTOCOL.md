# Prospective fresh-cohort gate

**Status:** no confirmatory cohort has been selected. This protocol defines
what must be frozen before any new outage target, observation mask, panel gate,
checkpoint or model result is opened.

## Why the earlier event table is not enough

The historical event pipeline is reproducible but not strictly label-blind.
`select_events.py` discovers eligible years from the presence of
`eaglei_outages_*.parquet` and uses an EAGLE-I record-end rule as F5.
`select_events_e3.py` then excludes 2024-05-26 because an EAGLE-I collection
gap caused its G4 observation gate to leave no county. The existing
preprocessing protocol already records this limitation. Those decisions are
valid descriptions of the historical screen, but they cannot certify a new
confirmatory cohort.

The tracked G1/G2/G3 manifests and the two open-GCRK selected-event files have
a conservative union of **29 event anchors** whose outcome panels or results
may have been inspected. Their normalized newline-terminated list has SHA-256
`209cade895e8d51f32758b772dc82b4f57c4993f8ff8ad043899cdb29c2cd037`.
The old weather-selection table contains 40 dates. Twenty-two are not in the
29-event outcome union, but they were already screened and rejected under the
old rule. They may support a separately labelled outcome-fresh sensitivity;
they are not the selection-fresh confirmation required here.

The original strict F1--F5 plus R1 rule leaves **zero** outcome-fresh dates.
Under the weather-only parts of the already documented wider E3 rule
(F2 scale, F4 duration and R1 second-wave/wet-cold structure, with the public
release-range check), 11 dates are outcome-fresh and do not overlap a
historical 216-hour window: 2018-03-02, 2020-06-09, 2020-11-15, 2020-12-25,
2021-01-14, 2022-01-04, 2022-02-18, 2022-11-05, 2024-04-06, 2024-11-20 and
2024-12-19. The deterministic top six by forecast-window wind-county count
are 2020-06-09, 2018-03-02, 2022-11-05, 2020-11-15, 2024-04-06 and
2021-01-14. This is an availability audit, not a frozen cohort: all 11 were
already present in the 40-row metadata screen, and the underlying NOAA/ERA5
payload is absent here. No EAGLE-I label or county mask was used to derive
this list.

## Locked selection boundary

A confirmatory cohort must contain three to six nonoverlapping 216-hour
windows selected only from a newly hashed NOAA Storm Events and/or ERA5
weather snapshot. Each window begins 72 hours before its event anchor. It must
not overlap the conservative 216-hour window around any of the 29 historical
anchors, and its anchor must not appear in the 40-date historical weather
screen.

Selection uses an allow-list, not merely a deny-list. NOAA event type, UTC
times, county/state identity, meteorological magnitude and weather-derived
footprint, duration, prefix-calmness and wave summaries are allowed. NOAA
property/crop damage, deaths and injuries are forbidden even though the raw
Storm Events details file carries them. Any EAGLE-I availability, outage,
coverage, customer-out, label, model-error or checkpoint field is forbidden.
Whether the eventual target files exist cannot enter event ranking or
replacement. Source acquisition failure after the lock is reported as missing
data; it does not trigger replacement with a more convenient event.

Before lock, save a candidate manifest with:

- the exact event anchors and UTC windows;
- the selection rule and the names of every field it used;
- SHA-256, public URL and source role for every NOAA/ERA5 input;
- an empty `outcome_access_before_lock` list;
- new panel, split and weather-information-set identifiers;
- seeds `0..4`.

Run the gate before accessing EAGLE-I rows or constructing targets:

```bash
PYTHONDONTWRITEBYTECODE=1 python \
  experiments/open_gcrk_20260919/phase2_20260925/fresh_cohort_gate.py \
  --candidate runs/fresh_cohort_candidate.json --verify-tracked-sources
```

Passing is necessary but not sufficient. The script verifies internal
metadata, candidate source bytes, tracked source hashes and exclusions; it
cannot prove that a human
did not previously view an unrecorded outcome. The frozen manifest hash and
the signed-off access boundary must be appended to `ATTEMPTS.md` before data
construction. The later model bundle must separately pass
`comparator_protocol_gate.py`.

## Current blocker

This checkout lacks `storm_events_county.parquet`,
`event_days_stratified.parquet`, the raw NOAA yearly detail files, the new ERA5
weather snapshot, and the raw EAGLE-I/216-hour artifacts. Consequently no real
selection-fresh candidate list is emitted here. The 11-date audit above can be
retained as an explicitly metadata-screened sensitivity only; promoting it to
the stronger selection-fresh claim would defeat the gate.
