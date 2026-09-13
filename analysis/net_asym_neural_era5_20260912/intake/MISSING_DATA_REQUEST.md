# Missing data request

Written 2026-09-13T00:20:23+00:00. Tasks B and C are not blocked. Task A runs, restricted to counties with real
weather.

## What is missing

ERA5 county weather for **1,265 counties** that appear in the 26-event panel set: the
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

- A full-coverage US comparison. Task A covers 3,427 train,
  1,710 validation and 1,240 evaluation county-event pairs after removal. The
  removed volume is tabulated in `intake/SCHEMA_DRIFT.md`.
- It does not block Task B (ANEEL) or Task C (synthetic).

## What would unblock it

A repaired driver release built from the full panel-county union, with a per-county coverage
mask and a new digest. Task A could then be rerun on the same frozen splits as a repaired
replication, not a fresh confirmation.
