# Measurement-first extension: pre-analysis protocol (2026-09-25)

This is a prospective extension to the completed results and amendments in
`../PREREG.md` and `../RESULTS.md`. The previous results are retained as results
under their original observation rule. No new model-selection result was
inspected before defining this protocol. Raw 15-minute source tables and the
current 216-hour panels are not available in this checkout; this file records
decisions to make *before* repeating the experiments on those files.

## Question and information set

An absent county/quarter-hour EAGLE-I row is **not an observed zero**. The
published 2014–2022 EAGLE-I release documents suppression of zero-outage
rows; inspect each 2024 export before asserting identical source behavior.
Collection/parser failures can also remove rows. `src/asymode/panel.py`
currently labels an absent cell zero when a
national collection timestamp has at least five reports and that county has
reported within a centered seven-day window. Both are proxies: a national run
does not certify that a particular utility was observed, and the centered
window uses reports that can occur after the forecast origin. The 90% panel
coverage gate also depends on those constructed labels. We will quantify
these decisions without pretending that the released rows disclose parser
health. Existing experiments condition the 144-hour outage rollout on
*realized future ERA5 reanalysis*; this is a weather-conditioned hindcast,
not an operational 144-hour weather forecast.

## Fixed sample, split, and labels

1. Keep the twelve episodes and county footprints in
   `../selected_events_e3.json` fixed for the first audit. The historical
   selection **replaced one event after an EAGLE-I collection gap failed its
   coverage gate**, so it must not be described as entirely independent of
   observation availability. Select any new event set using weather metadata
   before inspecting outage labels or trained-model errors. Use
   the same UTC window, hour-71 origin, source hashes, 2024 EAGLE-I customer
   denominator, and archived county/ERA5 spatial operators as the existing
   216-hour data. Do not select episodes, counties, or new thresholds by
   comparing model residuals or forecasting-window outage severity. Record
   excluded counties per original eligibility gate; keep a second, explicitly
   labeled sensitivity set whose membership is based solely on pre-origin
   source availability and static metadata.
2. Materialize four separate indicators at native 15-minute resolution:
   `source_row_present`, `assumed_zero`, `unresolved_absence`, and
   `national_collection_proxy`. Preserve observed positive counts, the raw
   denominator, original timestamps, duplicate/conflict flags, and a digest
   of each input file. Source-row presence is **not** a utility-uptime flag.
   Quarantine conflicting duplicate counts and invalid numerators or
   denominators; report the affected fraction rather than silently clipping
   them to valid values. Treat exact duplicate rows deterministically.
3. Freeze hourly target A to the existing mean of the four UTC quarter-hour
   slots after explicit zero-fill where the old `observed` mask allowed it.
   Also form target E, an actual recorded positive at the final available
   quarter-hour in the hour, **only as an observed-positive diagnostic**.
   Target B: retain each explicit positive row; carry its last value for at
   most four immediately following quarter-hours without a source row, and
   mark subsequent absent slots unknown (even when a national run exists).
   Do not carry through a national-collection outage, the forecast origin, or
   an event boundary. The zero after recovery is not source-certified by B.
   Retain explicit zero/nonpositive source rows for quality review as well;
   a valid explicit zero resets carry. B's source-row inclusion is independent
   of A's inferred run/service mask: a positive row outside that mask remains
   a B observation, while a no-run proxy slot prevents extending its value
   into later slots. Count B-only county-hours separately and compare A with
   B only on their predeclared common valid support. Carry never crosses the
   hour-71/72 forecast origin, even when the source cadence is continuous.
   Target C conditions on actually recorded positive source rows only and
   must never be called a full outage-trajectory ground truth. Keep a fixed
   UTC hourly clock across counties: no hour of model maximum or future
   outage-derived alignment.
4. Pre-specify two service-window diagnostics on A: centered service days
   1, 3, 7 (original), and 14; and a past-only variant whose service evidence
   ends at the target time, with its cold-start limitations reported. These
   are sensitivity checks, not parser uptime estimates. Preserve each
   15-minute row-presence mask before making hourly averages. Document any
   2024 denominator mismatch, numerator above denominator, or synchronized
   abrupt county drops as uncertain observations requiring a separate audit.

## Analysis and decision gates

For each event, county, and the 1–6, 7–24, 25–48, and 49–144 hour bins after
the origin, report raw-positive, assumed-zero, unknown, and proxy-run slot
counts; the fraction of hourly labels that include assumed zeros; contrasts
between A and B on their **common valid county-hours**; the number of eligible
counties under the original and pre-origin-only gate; and changes to onset,
peak time/height, and recovery time. The common support and aggregate
weights must be held fixed in any paired model comparison. Report the
all-zero baseline and error restricted to observed-active and ambiguous-zero
hours as separate descriptive estimands; selection on positivity makes the
former conditional.

Do **not** spend compute training new architectures until the row-level audit
and sample counts are recorded and reviewed. B leaves most long unreported
gaps **unknown** and may have highly selective positive-only support. First
report its common county-hour support and how many hour-71 initial conditions
it independently supplies. If that support is too sparse for matched training,
stop at a model-free sensitivity and seek independent feed-availability
evidence; do not zero-fill B's unknown cells simply to force a second fit.
If adequate, register the origin-treatment and common training/evaluation
mask **before** fitting W and W+Cin under both assumptions. Training B with an
A-imputed hour-71 origin is a conditional sensitivity, not an independent
validation of A. A small sensitivity permits training on A while retaining
its ambiguity flag. This is a sequencing rule, not a retrospective claim
that one label is correct. Run the already registered five-seed W+Cin confirmation
before declaring a new geographic mechanism superior. Preserve all seeds and
complete outer folds; report held-out events and event-level uncertainty,
especially when the number of independent storms is small. Every comparison
must use the same realized-weather information set, county/event support,
normalization fitted on training units, and pre-origin availability filter.
Existing event folds and their original OUTER outcomes have already been
inspected during the earlier research campaign. Reusing them for design
selection is **exploratory**, even if they are event-disjoint within a new
fit. A confirmatory transfer claim needs additional weather-selected events
locked before viewing their outcome data, or a clearly stated external cohort.
The prospective [`FRESH_COHORT_PROTOCOL.md`](FRESH_COHORT_PROTOCOL.md) now
implements the stronger boundary: exclude the conservative union of 29 event
anchors from G1/G2/G3 and both open-GCRK selected-event files, exclude any new
216-hour window that overlaps their conservative 216-hour windows, and use a
field allow-list over newly hashed NOAA/ERA5 inputs. It also rejects all 40
dates in the historical weather screen for a *selection-fresh* claim. The old
strict rule leaves zero outcome-fresh candidates; the wider pre-existing E3
weather rule leaves 11 nonoverlapping, outcome-fresh but already
metadata-screened dates. Those 11 are a separately labelled sensitivity, not
the stronger selection-fresh confirmation. EAGLE-I source support is assessed
only after the cohort manifest is frozen; failure after unlock produces
attrition, not event replacement.

ERA5 areal means and county-grid high-quantile/max gust already exist in the
round-2 inputs. Retain these as strong baselines; test any new high-resolution
weather or antecedent-rain feature only after logging its source, valid-time
semantics, footprint, and train-only scaling. No 4-hour smoothing is applied
to the hourly stock target, because it delays a return to zero by construction.

## Falsification and reporting

Do not use positive source-row density as a completeness threshold: true
recovery removes positive rows. If a county-feature gain vanishes under
pre-origin-only eligibility, whole-event holdout, an alternative reasonable
label, or the W+Cin baseline, report that negative result. If raw parser
telemetry cannot be obtained, identify true-zero and collection-failure
possibilities as **unresolved**, even when A and B give similar metrics.
Detailed audit commands, hashes, population counts, and any deviations are
appended to `ATTEMPTS.md` before training.
