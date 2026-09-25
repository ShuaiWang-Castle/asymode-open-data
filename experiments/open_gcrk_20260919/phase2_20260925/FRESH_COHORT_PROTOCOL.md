# Prospective fresh-cohort gate

**Status:** a six-event NOAA-only candidate cohort was locked on 2026-09-25
before any new outage target, observation mask, panel gate, checkpoint or
model result was opened. Passing the identity gate does not establish target
availability or model performance.

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
2021-01-14. This remains a historical availability audit, not the newly
locked cohort: all 11 were already present in the 40-row metadata screen.
No EAGLE-I label or county mask was used to derive this list.

## Locked selection boundary

A confirmatory cohort must contain three to six nonoverlapping 216-hour
windows selected only from a newly hashed NOAA Storm Events and/or ERA5
weather snapshot. Each window begins 72 hours before its event anchor. It must
not overlap the conservative 216-hour window around any of the 29 historical
anchors **or any of the 40 prior weather-screen anchors**. Excluding only the
exact weather-screen date is insufficient because shifting an anchor by one
day can reuse the same storm window.

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
  --candidate \
    experiments/open_gcrk_20260919/phase2_20260925/FRESH_COHORT_CANDIDATE.json \
  --verify-tracked-sources
```

Passing is necessary but not sufficient. The script verifies internal
metadata, candidate source bytes, tracked source hashes and exclusions; it
cannot prove that a human
did not previously view an unrecorded outcome. The frozen manifest hash and
the signed-off access boundary must be appended to `ATTEMPTS.md` before data
construction. The later model bundle must separately pass
`comparator_protocol_gate.py`.

## Locked NOAA cohort and remaining blocker

The public-derived catalogs at `main` commit
`8dd47c5ccd829611f27b69a3d64c274a0a24c400` were restored without importing
its old 168-hour panels. Their SHA-256 values matched the release ledger. The
published anchor catalog's `dominant` field is not used because its generating
script is absent from the release; the new screen reads only `day` from that
file and recomputes each hazard family from allow-listed Storm Events fields.

The locked anchors are 2018-02-19 (wet), 2019-01-28 (winter), 2019-06-16
(convective), 2020-01-11 (wind), 2020-04-08 (convective), and 2022-01-29
(winter). No tropical anchor passed the unchanged county, state, prefix and
duration filters. The raw EAGLE-I rows, ERA5 inputs, source masks and 216-hour
panels remain unavailable, so these are **weather-selected event identities,
not usable training examples or empirical results**. If a locked event lacks
outcome support, record attrition and do not replace it after labels are
opened.
