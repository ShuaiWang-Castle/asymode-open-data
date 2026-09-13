# Schema and semantic drift recorded at intake

Written 2026-09-13T00:20:23+00:00, before any model was fitted. Every item was measured on the opened
binaries (`intake/US_DATA_AUDIT.json`), not read from a README.

## Task A: EAGLE-I panels and ERA5 drivers

1. **Stored weather has 12 raw channels in the expected order**
   (`cape, cloud, gust, precip, pressure, rh, snowfall, soil_moisture, t2m_c, u10, v10,
   wind_speed`) in all 26 driver files. The manifest `channels` list has 14 entries:
   `clock_sin` and `clock_cos` are generated features, not stored columns. The clock
   used here is computed from the driver timestamps.

2. **Missing county weather is encoded as zero.** 1,265 of
   2,625 unique panel counties carry a driver tensor that is exactly zero in all 12
   channels at all 169 hours. Status never switches across events
   (0 switches), and this set is exactly the set whose 2-m temperature and
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
| train | 92,077 | 30,479 | 33.1% | 60,707 |
| validation | 38,138 | 5,231 | 13.7% | 32,907 |
| evaluation | 42,311 | 16,800 | 39.7% | 25,430 |

   Most affected events:

   | event | split | counties with weather | state-legal windows removed |
   |---|---|---:|---:|
| 2021-12-15 | train | 70/361 | 79.8% |
| 2020-08-04 | train | 89/213 | 59.2% |
| 2024-01-09 | evaluation | 121/240 | 49.8% |
| 2020-02-06 | train | 163/326 | 49.1% |
| 2024-01-12 | evaluation | 290/578 | 48.7% |
| 2018-10-11 | train | 126/235 | 45.3% |
| 2020-10-29 | train | 140/240 | 43.2% |
| 2019-11-27 | train | 284/480 | 41.5% |

   The exclusion is systematic by county mapping rather than random. Task A therefore
   speaks only for counties the published drivers actually cover. The full list is
   `intake/US_MISSING_WEATHER_COUNTIES.csv`.

3. **Observed-but-non-finite targets.** 41,297 15-minute cells have
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

No external data. The 0.011 target SD sits 2.2% above the 1/L floor
(minimum achievable SD 0.01076). Its rho = 0.00144 lies inside [0, 1] and was not
truncated.
