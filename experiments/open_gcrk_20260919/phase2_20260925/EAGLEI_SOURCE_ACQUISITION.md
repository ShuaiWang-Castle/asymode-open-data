# EAGLE-I source acquisition gate

Snapshot: 2026-09-25 UTC. This note records source availability and rebuild
requirements. It does not contain outage observations or model results.

## Cohort-to-release mapping

The locked fresh weather cohort
(`FRESH_COHORT_CANDIDATE.json`, canonical SHA-256
`28563c0dc9affaa3b7ba5cbcbf51600b637ae519a9a481bec6fea70cfa51df97`)
uses anchors in 2018, 2019, 2020 and 2022. Every event therefore requires the
historic EAGLE-I release below; the 2024 release is not a valid substitute.

- Official dataset DOI: `10.13139/ORNLNCCS/1975202`
- OSTI record: https://www.osti.gov/biblio/1975202
- Resolved ORNL catalog record:
  https://doi.ccs.ornl.gov/dataset/ccec86f0-e144-5de8-aee0-fb26028b26e1
- ORNL delivery: Globus collection
  `57618e0a-2c99-45ff-9694-24141b92fa17`, path
  `/gen101/world-shared/doi-data/ORNLNCCS/202305/10.13139_ORNLNCCS_1975202/`
- Published Figshare record: https://doi.org/10.6084/m9.figshare.24237376

The OSTI page states that the release contains county-level observations at
15-minute intervals from 2014 through 2022. Its download button resolves to
the ORNL Globus collection, not to a versioned HTTP file. The public Figshare
record currently displays version 4 (posted 2026-02-25) and now contains
additional years through 2025. Therefore, a DOI alone is not sufficient data
identity for this study.

## Required raw files

Only these annual positive-record tables are required by the locked cohort:

- `eaglei_outages_2018.csv`
- `eaglei_outages_2019.csv`
- `eaglei_outages_2020.csv`
- `eaglei_outages_2022.csv`

The Scientific Data descriptor reports, respectively, 21,776,807;
24,074,123; 25,545,518; and 25,796,466 rows. These are published metadata,
not acceptance hashes. It also states that timestamps are UTC, explicit zero
outages are omitted, and an absent county/timestamp can mean either zero or a
collection gap. Consequently, no downloader or panel builder may convert all
absent rows to observed zeros.

## Acceptance requirements

Before reading outcomes or building a 216-hour target, record for every annual
file:

1. immutable source URL or Globus collection/path and retrieval date;
2. upstream version or file ID, if the provider exposes one;
3. byte length and locally computed SHA-256;
4. header/schema, timestamp timezone, minimum/maximum timestamp and row count;
5. duplicate-key, off-grid timestamp, invalid FIPS, nonfinite/negative count
   and within-year monotonicity checks.

The four annual files must come from one pinned provider snapshot. Do not mix a
current Figshare file with an older Globus copy merely because the basenames
match. If the provider exposes no checksum, two independent downloads may be
compared byte-for-byte, but the resulting hash is a locally established
identity and must be labelled as such.

After the raw files pass, run `measurement_audit.py` on each locked window.
Keep source-positive, inferred-zero, unknown and quarantined support distinct.
A locked event with insufficient support is attrition without replacement.
Copy the raw-file hashes, cohort canonical hash, observation-support hash and
panel hash unchanged into every comparator bundle.

## Current negative result

The research branch and the independent public `main` release do not contain
the four raw annual tables. The public release contains a small coverage-history
table and previously derived 168-hour panels, but those panels cannot establish
15-minute source support and cannot substitute for the new 216-hour cohort.

During this audit the OSTI and ORNL catalog records were reachable. The
Figshare web record was discoverable, but direct API/downloader requests from
this runtime returned HTTP 403, and the subsequent interactive inspection was
interrupted when the execution environment disconnected. No raw EAGLE-I byte
was acquired, no file checksum was computed, and no source mask or real-data
model comparison was produced.

Next gate: obtain the four pinned annual files through ORNL Globus or a
version-pinned Figshare download, compute the acceptance metadata above, and
only then audit the six locked windows. Do not replace an unavailable event or
reuse an old panel.
