#!/usr/bin/env python3
"""Write intake/SCHEMA_DRIFT.md and intake/MISSING_DATA_REQUEST.md from measured intake files."""
import argparse, datetime, json
from pathlib import Path
import pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument('--work', required=True); a = ap.parse_args()
A = Path(a.work)
now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
aud = json.loads((A / 'intake/US_DATA_AUDIT.json').read_text())['summary']
spl = json.loads((A / 'locks/SPLITS.json').read_text())
cal = json.loads((A / 'results/synthetic/EXACT_LAW_CALIBRATION.json').read_text())['calibration']
cnt = pd.read_csv(A / 'intake/US_WINDOW_COUNTS_BY_EVENT.csv')
miss = pd.read_csv(A / 'intake/US_MISSING_WEATHER_COUNTIES.csv', dtype={'fips': str})
by = cnt.groupby('split')[['legal_states_only', 'excluded_by_missing_weather_only', 'final_windows']].sum()
split_rows = '\n'.join(
    f"| {s} | {int(r.legal_states_only):,} | {int(r.excluded_by_missing_weather_only):,} | "
    f"{100 * r.excluded_by_missing_weather_only / r.legal_states_only:.1f}% | {int(r.final_windows):,} |"
    for s, r in by.loc[['train', 'validation', 'evaluation']].iterrows())
cnt['share'] = cnt.excluded_by_missing_weather_only / cnt.legal_states_only
worst_rows = '\n'.join(
    f"| {r.event} | {r.split} | {int(r.counties_weather_legal)}/{int(r.counties)} | {100 * r.share:.1f}% |"
    for r in cnt.sort_values('share', ascending=False).head(8).itertuples())
law0 = cal['laws'][0]

drift = f"""# Schema and semantic drift recorded at intake

Written {now}, before any model was fitted. Every item was measured on the opened
binaries (`intake/US_DATA_AUDIT.json`), not read from a README.

## Task A: EAGLE-I panels and ERA5 drivers

1. **Stored weather has 12 raw channels in the expected order**
   (`cape, cloud, gust, precip, pressure, rh, snowfall, soil_moisture, t2m_c, u10, v10,
   wind_speed`) in all 26 driver files. The manifest `channels` list has 14 entries:
   `clock_sin` and `clock_cos` are generated features, not stored columns. The clock
   used here is computed from the driver timestamps.

2. **Missing county weather is encoded as zero.** {aud['counties_all_zero_whenever_present']:,} of
   {aud['unique_panel_counties']:,} unique panel counties carry a driver tensor that is exactly zero in all 12
   channels at all 169 hours. Status never switches across events
   ({aud['counties_switching_status']} switches), and this set is exactly the set whose 2-m temperature and
   surface pressure are both identically zero. In code, `src/asymode/weather.py::apply_weights`
   initialises with `np.zeros` and writes only the FIPS present in
   `era5_county_weights.parquet`. The same defect was documented on
   `open-audit-20260904@b21c4b4`; this round re-verified it independently.

   **Rule adopted (inputs only, identical for both models):** such a county's weather is
   illegal, its windows are excluded, and missing weather is never fed as zero. No
   outcome, peak, second wave or damage level enters the rule. It still changes the
   population, so the removed volume is reported:

   | split | windows with 49 legal states | removed by missing weather | share | final after dedup |
   |---|---:|---:|---:|---:|
{split_rows}

   Most affected events:

   | event | split | counties with weather | state-legal windows removed |
   |---|---|---:|---:|
{worst_rows}

   The exclusion is systematic by county mapping rather than random. Task A therefore
   speaks only for counties the published drivers actually cover. The full list is
   `intake/US_MISSING_WEATHER_COUNTIES.csv`.

3. **Observed-but-non-finite targets.** {aud['total_observed_nonfinite_cells']:,} 15-minute cells have
   `observed=True` with a non-finite `y`. They are illegal states; nothing is filled.

4. **Bidirectional gap filling at build time.** `scripts/build_drivers.py` reindexes to
   the panel hours and calls `interpolate(limit_direction="both")`. The inputs are
   reanalysis with two-sided gap filling. They are used only under
   RETROSPECTIVE_KNOWN_REANALYSIS_CONDITIONAL_RESPONSE and are never described as a
   forecast available at the origin.

5. **States are exact snapshots.** This round uses `y[:, 4*h]`, the stock at the UTC top
   of the hour, as protocol 4.3 requires. The R1 run and the conservation preflight used
   hourly means of observed sub-steps, so their numbers are not directly comparable.

6. **Densified zeros.** Some `observed=True` zeros come from the repository densification
   rule rather than an original zero report.

7. **Fixed customer denominator.** The 2024 modelled customer count is applied to earlier
   events. Customer drift is not corrected, and no evaluation-period `y` re-estimates it.

8. **Time and key alignment.** All 26 panels start at 00:00 UTC on a uniform 15-minute grid.
   Every driver timestamp equals the panel's hourly snapshot, and FIPS order is identical in
   all 26 pairs. The loader still aligns by key and fails closed.

9. **Origin grid.** Legal origin hours are 24..144 at UTC 00/06/12/18, giving 21 candidates
   per county-event. Past weather covers hours t-23..t, the given future weather covers
   t+1..t+24, and step k reads the weather of hour t+k+1.

## Task B: ANEEL ledger

No drift. The ledger passes all 192 array fingerprints. The repository `DATA_MANIFEST.json`
carries `$W` path placeholders, so the Task B loader writes a runtime manifest that points at
the verified ledger inside this checkout.

## Task C: synthetic

No external data. The 0.011 target SD sits {100 * (0.011 / cal['min_achievable_sd_8h'] - 1):.1f}% above the 1/L floor
(minimum achievable SD {cal['min_achievable_sd_8h']:.5f}). Its rho = {law0['rho']:.5f} lies inside [0, 1] and was not
truncated.
"""
(A / 'intake/SCHEMA_DRIFT.md').write_text(drift)

req = f"""# Missing data request

Written {now}. Tasks B and C are not blocked. Task A runs, restricted to counties with real
weather.

## What is missing

ERA5 county weather for **{len(miss):,} counties** that appear in the 26-event panel set: the
field `X` (12 channels x 169 hours) in `data/interim/drivers_<event>.npz` for those FIPS. Their
published tensors are all-zero placeholders. The underlying gap is the missing rows for those
FIPS in `data/interim/era5_county_weights.parquet`. The FIPS and their events are listed in
`intake/US_MISSING_WEATHER_COUNTIES.csv`.

## Sources tried

1. **Other branches.** `main@8dd47c5`, `aneel-open-data@de406e16`,
   `open-audit-20260904@8c4322f` and `aistats-current@fc03d36` were checked. The open-audit
   branch documents the defect, but none carries a repaired driver release, a new driver
   digest or a coverage mask. No Git LFS objects are configured.
2. **Local project data.** `data/raw/era5/*.nc` (26 files, about 2.9 GB) exists in the PI's
   local checkout. Filling the missing counties needs a regenerated county-to-grid weight table
   and a new immutable driver release, with a new digest and fail-closed coverage tests, as the
   defect audit specifies. That is a data release change, not a model run, and this round did
   not perform it.
3. **Official source.** ERA5 on the Copernicus CDS requires an account; per protocol 4.2 none
   was requested.

## What it blocks

- A full-coverage US comparison. Task A covers {spl['splits']['train']['county_event_pairs']:,} train,
  {spl['splits']['validation']['county_event_pairs']:,} validation and {spl['splits']['evaluation']['county_event_pairs']:,} evaluation county-event pairs after removal. The
  removed volume is tabulated in `intake/SCHEMA_DRIFT.md`.
- It does not block Task B (ANEEL) or Task C (synthetic).

## What would unblock it

A repaired driver release built from the full panel-county union, with a per-county coverage
mask and a new digest. Task A could then be rerun on the same frozen splits as a repaired
replication, not a fresh confirmation.
"""
(A / 'intake/MISSING_DATA_REQUEST.md').write_text(req)
print('wrote SCHEMA_DRIFT.md and MISSING_DATA_REQUEST.md')
