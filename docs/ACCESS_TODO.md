# Actions that need the PI personally

I cannot create accounts or accept terms of service. These are blocking or will
block shortly.

## 1. Globus -- IN PROGRESS, archive located

EAGLE-I 2014-2022 is distributed only through Globus. The DOI folder resolves to

    Collection: OLCF DOI-DOWNLOADS
    Path: /gen101/world-shared/doi-data/ORNLNCCS/202305/10.13139_ORNLNCCS_1975202/

and contains exactly two files:

| file | size |
|---|---|
| `eaglei_outages.zip` | **1.11 GB** |
| `READMEdata.txt` | 8.54 KB |

**1.11 GB, not tens of GB.** No subset selection is needed; take the whole archive.

Order of operations:

1. Select `READMEdata.txt` (click the row, not the filename) and use the right
   panel's **Download**. It is 8 KB and it settles the open question in the data
   card: whether the modeled county-customer file (the denominator) and the
   state-by-year coverage file are inside the archive or shipped separately.
2. Then download `eaglei_outages.zip` the same way, into `data/raw/eaglei/`.
3. If **Download** is greyed out, the collection has no HTTPS access enabled and
   Globus Connect Personal is required after all; use **Transfer or Sync to...**
   with a local endpoint instead.

Once the archive is in place:

    ./.venv/bin/python scripts/ingest_eaglei.py inspect   # table of contents, no unpack
    ./.venv/bin/python scripts/ingest_eaglei.py build     # parquet panel

`inspect` reads the archive's index and the first lines of each member without
extracting, so the layout is checked before anything is unpacked. `build` refuses
to proceed if it cannot find a denominator file, rather than borrowing another
year's customer counts silently.

Disk: 41 GB free, against ~1.1 GB compressed and an unpacked size to be confirmed
by `inspect`.

## 2. Copernicus CDS account -- blocks weather drivers

ERA5 needs a free CDS account and an API key.
1. Register at <https://cds.climate.copernicus.eu>.
2. Accept the ERA5 licence in the account page (a licence acceptance -- yours to
   make, not mine).
3. Put the key in `~/.cdsapirc`. Do not paste the key into chat; I only need to
   know it exists.

Fallback if CDS is slow to approve: NOAA URMA/RTMA via NCEI is anonymous HTTPS
and I can start on that instead. Say the word and I will.

## 3. Decisions I should not make alone

* **County aggregation weighting** for weather: area-weighted or
  population-weighted. I recommend population-weighted, since the target counts
  customers rather than area. Needs your sign-off because it is stated in the paper.
* **Repository visibility and timing.** The repo is local and private. I will not
  publish it.
* **Venue.** See the deadline note: abstract 2026-09-29, full paper with all
  supplementary 2026-10-06. That is your call, not mine.
