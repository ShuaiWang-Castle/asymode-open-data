# Actions that need the PI personally

I cannot create accounts or accept terms of service. These are blocking or will
block shortly.

## 1. Globus account -- BLOCKING the outage panel

EAGLE-I 2014-2022 is distributed **only through Globus** from the ORNL landing
page. There is no anonymous HTTPS path; the OSTI and OpenEnergyHub records are
metadata that point back to the same Globus endpoint.

Steps:
1. Sign in at <https://app.globus.org> (institutional login -- Duke is a Globus
   subscriber, so the NetID should work; no separate password needed).
2. Install Globus Connect Personal on this Mac and name the local endpoint.
3. From <https://doi.ccs.ornl.gov/dataset/ccec86f0-e144-5de8-aee0-fb26028b26e1>
   use "Download Dataset on Globus", set the destination to
   the repository's `data/raw/eaglei/`.
4. Tell me when the transfer is queued and I will take it from there.

**Before transferring everything**, please report what the file listing shows --
per-year file sizes and whether a modeled-county-customer file and a coverage
file are present. A full-nation, full-history pull is likely tens of GB and we
do not need it: the event-day selection is already built, so I can name the exact
county-year subset once I see the layout.

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
