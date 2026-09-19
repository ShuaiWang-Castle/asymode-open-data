"""Source URL, acquisition date and SHA-256 of every input file this study reads.

Files downloaded for this study log themselves (data_provenance/era5_fetch_log.jsonl,
data_provenance/geography_log.jsonl); this script covers the project's earlier public
downloads and the derived tables built from them, and writes
data_provenance/SOURCES.md and data_provenance/sources.json. Acquisition date = the
file's modification time on this machine (UTC).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RAW, INTERIM = ROOT / "data" / "raw", ROOT / "data" / "interim"
EVENTS = [e["event"] for e in json.loads((HERE / "selected_events.json").read_text())["events"]]

SOURCES = [
    ("EAGLE-I outage records 2014-2022 (ORNL/DOE), DOI 10.13139/ORNLNCCS/1975202",
     "https://doi.ccs.ornl.gov/dataset/ccec86f0-e144-5de8-aee0-fb26028b26e1", [RAW / "eaglei/eaglei_outages.zip"]),
    ("EAGLE-I outage records 2024 with modelled county customers, DOI 10.13139/OLCF/2500278",
     "https://doi.org/10.13139/OLCF/2500278", [RAW / "eaglei/eaglei_outages_2024.zip"]),
    ("NOAA NCEI Storm Events details, bulk CSV", "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/",
     sorted((RAW / "storm_events").glob("StormEvents_details-ftp_v1.0_d20*.csv.gz"))),
    ("NWS forecast zone to county correlation", "https://www.weather.gov/gis/ZoneCounty", [RAW / "nws/zone_county.txt"]),
    ("Census 2023 Gazetteer, counties",
     "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2023_Gazetteer/2023_Gaz_counties_national.zip",
     [RAW / "census/2023_Gaz_counties_national.zip", RAW / "census/2023_Gaz_counties_national.txt"]),
    ("Census 2023 cartographic boundary counties 1:500k",
     "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_500k.zip", [RAW / "census/cb_2023_us_county_500k.zip"]),
    ("Census 2023 county adjacency", "https://www2.census.gov/geo/docs/reference/county_adjacency/county_adjacency2023.txt",
     [RAW / "census/county_adjacency2023.txt"]),
    ("USDA ERS Rural-Urban Continuum Codes 2023", "https://www.ers.usda.gov/media/5768/2023-rural-urban-continuum-codes.csv",
     [RAW / "census/rucc2023.csv"]),
    ("EIA Form 861, 2023", "https://www.eia.gov/electricity/data/eia861/zip/f8612023.zip", [RAW / "eia/f8612023.zip"]),
    ("ERA5 single levels, Copernicus Climate Data Store (event windows)",
     "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels",
     sorted(p for e in EVENTS for p in (RAW / "era5").glob(f"era5_{e}*.nc"))),
]
DERIVED = [
    ("EAGLE-I records by year (scripts/ingest_eaglei.py)", [INTERIM / f"eaglei_outages_{y}.parquet" for y in (2019, 2021, 2022, 2024)]),
    ("EAGLE-I 2024 modelled county customers (scripts/ingest_eaglei.py)", [INTERIM / "eaglei_county_customers_2024.parquet"]),
    ("EAGLE-I state coverage history (scripts/ingest_eaglei.py)", [INTERIM / "eaglei_coverage_history.parquet"]),
    ("County-resolved Storm Events and event-day catalogue (scripts/build_event_catalog.py)",
     [INTERIM / "storm_events_county.parquet", INTERIM / "event_days_stratified.parquet"]),
    ("County statics: Census, RUCC, EIA-861, customers (scripts/build_county_statics.py)", [INTERIM / "county_statics.parquet"]),
    ("216-hour panels, drivers, weights (build_panel216.py)", sorted((INTERIM / "open_gcrk").glob("panel216_*.npz"))
     + [INTERIM / "open_gcrk/era5_county_weights_conus.parquet"]),
    ("Geography descriptors (build_geography.py)", [INTERIM / "open_gcrk/geography.parquet"]),
    ("Model inputs (build_features.py)", [INTERIM / "open_gcrk/features.npz"]),
]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def rec(p: Path) -> dict:
    return dict(file=str(p.relative_to(ROOT)), bytes=p.stat().st_size, sha256=sha256(p),
                date_utc=dt.datetime.fromtimestamp(p.stat().st_mtime, dt.timezone.utc).strftime("%Y-%m-%d"))


def main():
    out = dict(raw=[], derived=[])
    md = ["# Data sources for experiments/open_gcrk_20260919", "",
          "Acquisition date = file modification time on the analysis machine (UTC). Files downloaded",
          "during this study also log URL, request and time individually: `era5_fetch_log.jsonl`,",
          "`geography_log.jsonl` (one line per raster and per Soil Data Access query).", "",
          "## Public raw inputs", "", "| source | URL | file | date | bytes | SHA-256 |", "|---|---|---|---|---:|---|"]
    for name, url, files in SOURCES:
        for p in files:
            if p.exists():
                r = rec(p); out["raw"].append(dict(source=name, url=url, **r))
                md.append(f"| {name} | {url} | `{r['file']}` | {r['date_utc']} | {r['bytes']} | `{r['sha256']}` |")
    geo = HERE / "data_provenance" / "geography_log.jsonl"
    if geo.exists():
        lines = [json.loads(x) for x in geo.read_text().splitlines() if x.strip()]
        rasters = [x for x in lines if "file" in x]
        by = {}
        for x in rasters:
            by.setdefault(x["source"], []).append(x)
        for src, xs in by.items():
            md.append(f"| {len(xs)} county rasters | {src} | `data/raw/geography/` | "
                      f"{min(x['downloaded_utc'][:10] for x in xs)} | {sum(x['bytes'] for x in xs)} | per file in `geography_log.jsonl` |")
        q = [x for x in lines if "query_sha256" in x]
        if q:
            md.append(f"| {len(q)} Soil Data Access queries | https://sdmdataaccess.sc.egov.usda.gov/Tabular/post.rest | "
                      f"(tabular responses, not stored) | {min(x['downloaded_utc'][:10] for x in q)} | | per query in `geography_log.jsonl` |")
    md += ["", "## Derived tables read by the study", "", "| content | file | date | bytes | SHA-256 |", "|---|---|---|---:|---|"]
    for name, files in DERIVED:
        for p in files:
            if p.exists():
                r = rec(p); out["derived"].append(dict(content=name, **r))
                md.append(f"| {name} | `{r['file']}` | {r['date_utc']} | {r['bytes']} | `{r['sha256']}` |")
    (HERE / "data_provenance" / "sources.json").write_text(json.dumps(out, indent=1) + "\n")
    (HERE / "data_provenance" / "SOURCES.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:12]))


if __name__ == "__main__":
    main()
