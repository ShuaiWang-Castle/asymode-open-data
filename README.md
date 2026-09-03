# County power-outage panels with an observation mask, and matched weather drivers

An open, analysis-ready dataset for modelling how outages start and how they are
restored: **26 United States storm events**, county-level outage trajectories at
15-minute resolution with an **explicit observation mask**, hourly ERA5 weather
aggregated onto the same counties, county static covariates, and the storm-event
catalogue used to pick the events.

Everything here is derived from public sources by the scripts in `scripts/`, and
every file is checksummed. See **[`data/README.md`](data/README.md)** for the
file-by-file description, the densification rule, and the licence and attribution
requirements of each source.

## Quick start

```bash
pip install numpy pandas pyarrow
python scripts/verify_open_data.py     # checksums + reproduces the archived onset audit
```

```python
import numpy as np
z = np.load("data/interim/panel_2021-05-04.npz", allow_pickle=True)
y, observed = z["y"], z["observed"]        # (counties x 15-min steps), fraction out / mask
fips, ts = z["fips"], z["ts"]              # county codes, UTC timestamps
d = np.load("data/interim/drivers_2021-05-04.npz", allow_pickle=True)
X, channels = d["X"], d["channels"]        # (counties x hours x 12), channel names
```

## The one thing to know before using it

**Zeros are omitted at the source, so the mask is not optional.** The publisher
does not record a county-timestamp with zero customers out, so an absent record
is either a true zero or an unobserved cell and the file cannot tell you which.
These panels are densified under a stated rule and `observed` marks the
difference; **excluding unobserved cells from every loss and metric, rather than
imputing them, is the intended use**. On these windows 95.3–99.8% of cells are
marked observed. `data/README.md` gives the rule in full.

## What is in the box

| | |
|---|---|
| events | 26 storm days, 2018–2024, windowed 2 days before to 5 days after |
| families | convective (11), winter (7), wind (4), tropical (3), flood (1) |
| target | `customers_out / total_customers` per county per 15 minutes, plus the mask |
| drivers | 12 hourly channels per county: `cape, cloud, gust, precip, pressure, rh, snowfall, soil_moisture, t2m_c, u10, v10, wind_speed` |
| statics | area, centroid, rural–urban continuum, neighbour degree, utility reliability indices, customers |
| catalogue | NOAA storm events, county-coded with zone records expanded to counties |
| size | about 60 MB; the ~4 GB of raw inputs are not redistributed but are all freely downloadable, and `scripts/` rebuilds every file from them |

## Rebuilding from the raw sources

```bash
python scripts/ingest_eaglei.py --all-archives      # outage records (Globus or Figshare)
python scripts/build_event_catalog.py               # NOAA storm events -> county-days
python scripts/fetch_era5.py                        # ERA5 (needs a free Copernicus CDS account)
python scripts/build_county_weights.py              # ERA5 grid -> county area weights
python scripts/build_panel.py --event-day <YYYY-MM-DD>
python scripts/build_drivers.py --event-day <YYYY-MM-DD>
python scripts/build_county_statics.py
```

## Licence and attribution

The derived files here are released for reuse, but **each underlying source
carries its own terms and its own required attribution** — EAGLE-I (CC BY 4.0),
Copernicus/ERA5 (Copernicus licence, derived products permitted with
attribution), and several U.S. Government public-domain sources. The table in
[`data/README.md`](data/README.md) lists them; please carry the attributions
through. `poweroutage.us` / `poweroutage.com` are not a source for any file here.
