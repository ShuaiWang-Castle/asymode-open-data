# Data sources for experiments/open_gcrk_20260919

Acquisition date = file modification time on the analysis machine (UTC). Files downloaded
during this study also log URL, request and time individually: `era5_fetch_log.jsonl`,
`geography_log.jsonl` (one line per raster and per Soil Data Access query).

## Public raw inputs

| source | URL | file | date | bytes | SHA-256 |
|---|---|---|---|---:|---|
| EAGLE-I outage records 2014-2022 (ORNL/DOE), DOI 10.13139/ORNLNCCS/1975202 | https://doi.ccs.ornl.gov/dataset/ccec86f0-e144-5de8-aee0-fb26028b26e1 | `data/raw/eaglei/eaglei_outages.zip` | 2026-09-01 | 1118012101 | `89b8a7b68cf6a0b0e7f7b47c9246e27b94e81c1b3f3923c985f6d43ab014fda2` |
| EAGLE-I outage records 2024 with modelled county customers, DOI 10.13139/OLCF/2500278 | https://doi.org/10.13139/OLCF/2500278 | `data/raw/eaglei/eaglei_outages_2024.zip` | 2026-09-01 | 91464719 | `2e87ddcbb4b0f877bd8eda4702277e68a2ca2a43780976274a0752f89458cf09` |
| NOAA NCEI Storm Events details, bulk CSV | https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/ | `data/raw/storm_events/StormEvents_details-ftp_v1.0_d2018_c20260323.csv.gz` | 2026-09-01 | 9585982 | `19702480f50cd2a8dacce5e57891feba156afb87b1ea5cbb4eaeda69d2b964a4` |
| NOAA NCEI Storm Events details, bulk CSV | https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/ | `data/raw/storm_events/StormEvents_details-ftp_v1.0_d2019_c20260323.csv.gz` | 2026-09-01 | 11602648 | `f5043ecc776818b0feffdf1e0fcbaf63329db0635eab1057c7b8ba00ef332e3b` |
| NOAA NCEI Storm Events details, bulk CSV | https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/ | `data/raw/storm_events/StormEvents_details-ftp_v1.0_d2020_c20260323.csv.gz` | 2026-09-01 | 10444606 | `895c56fd46991c4d9a135d67558dc4b447a02a2314efac0ace645135b98f9c9d` |
| NOAA NCEI Storm Events details, bulk CSV | https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/ | `data/raw/storm_events/StormEvents_details-ftp_v1.0_d2021_c20260323.csv.gz` | 2026-09-01 | 10563953 | `60c1daf96dbe8eafd48c80df5b70df51a0125ab95d467ee7009a646761d57715` |
| NOAA NCEI Storm Events details, bulk CSV | https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/ | `data/raw/storm_events/StormEvents_details-ftp_v1.0_d2022_c20260625.csv.gz` | 2026-09-01 | 11817849 | `7d6b79d0049a6edec96b061738289daddba90be26fc9afe7b0a6fd617f21452e` |
| NOAA NCEI Storm Events details, bulk CSV | https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/ | `data/raw/storm_events/StormEvents_details-ftp_v1.0_d2023_c20260323.csv.gz` | 2026-09-01 | 12888092 | `713784bed40d9e5a95b1d6240a654f865f3ef97703713105c7b35437270da134` |
| NOAA NCEI Storm Events details, bulk CSV | https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/ | `data/raw/storm_events/StormEvents_details-ftp_v1.0_d2024_c20260728.csv.gz` | 2026-09-01 | 12693243 | `2070b83eccab041b36360ab73645b9a249c3eefc5b92b5b3fc0cbba4d9fcc09c` |
| NOAA NCEI Storm Events details, bulk CSV | https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/ | `data/raw/storm_events/StormEvents_details-ftp_v1.0_d2025_c20260819.csv.gz` | 2026-09-01 | 12585209 | `d9b46b4c6aae554723cadbb9691f3d5258371c02e530ba815d6fe41e4550149f` |
| NWS forecast zone to county correlation | https://www.weather.gov/gis/ZoneCounty | `data/raw/nws/zone_county.txt` | 2026-09-01 | 339683 | `ca0b04f84273f95748d1dd0c501c1a806235ac4e9ba170c5e95bcac2e2eb468e` |
| Census 2023 Gazetteer, counties | https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2023_Gazetteer/2023_Gaz_counties_national.zip | `data/raw/census/2023_Gaz_counties_national.zip` | 2026-09-01 | 141650 | `919df59ba90759cce85468c0337e898e4b39c08eaffce86ddd88fa41f1f7f0c8` |
| Census 2023 Gazetteer, counties | https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2023_Gazetteer/2023_Gaz_counties_national.zip | `data/raw/census/2023_Gaz_counties_national.txt` | 2023-09-21 | 647830 | `b0aecde6748947236e931faeb6a619b89aea0346e54a454239aac76abcd2ef82` |
| Census 2023 cartographic boundary counties 1:500k | https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_500k.zip | `data/raw/census/cb_2023_us_county_500k.zip` | 2026-09-01 | 11630077 | `99d6597b1fc7767deef62e01d28d8b5dcbd578e151855f7dc0d173cbf5bf0868` |
| Census 2023 county adjacency | https://www2.census.gov/geo/docs/reference/county_adjacency/county_adjacency2023.txt | `data/raw/census/county_adjacency2023.txt` | 2026-09-01 | 1116106 | `df62fedd7d27528af7fd870c7ec92232fb2d6bea3ec6cbc9928946b78e128f64` |
| USDA ERS Rural-Urban Continuum Codes 2023 | https://www.ers.usda.gov/media/5768/2023-rural-urban-continuum-codes.csv | `data/raw/census/rucc2023.csv` | 2026-09-01 | 629322 | `ec455ee2a8bc5fc8e070575ea5bee7dce46fc6037f8c3449cbf56e8b45331fa7` |
| EIA Form 861, 2023 | https://www.eia.gov/electricity/data/eia861/zip/f8612023.zip | `data/raw/eia/f8612023.zip` | 2026-09-01 | 4604583 | `0acd67ab84cdaef09fd47b4449f2c944f8731c2b3c695ec3a97bb6c0690a4d61` |
| ERA5 single levels, Copernicus Climate Data Store (event windows) | https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels | `data/raw/era5/era5_2019-03-13.nc` | 2026-09-19 | 85966996 | `c628bb8f98627b286b70b3412ef4efff182ac5dd3657ed55ccbdc824a1ce6ad8` |
| ERA5 single levels, Copernicus Climate Data Store (event windows) | https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels | `data/raw/era5/era5_2021-03-26.nc` | 2026-09-19 | 87013876 | `3c0df5c1149264f325306ae71361271323058104f9564d98634e0bf607300272` |
| ERA5 single levels, Copernicus Climate Data Store (event windows) | https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels | `data/raw/era5/era5_2021-12-11.nc` | 2026-09-01 | 97897109 | `b52d209e1ac93d006c700f407e63d71f459ce273f813b2b9e9a7aafc8b71c6ea` |
| ERA5 single levels, Copernicus Climate Data Store (event windows) | https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels | `data/raw/era5/era5_2022-06-08.nc` | 2026-09-01 | 100010810 | `b423371a678eeb04a99321966bf47b5e34f5d83bed221af88374a86dd3bd96bb` |
| ERA5 single levels, Copernicus Climate Data Store (event windows) | https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels | `data/raw/era5/era5_2024-02-27.m0.nc` | 2026-09-19 | 56537674 | `66facfad2622cbbcf8097f9a30ec96381742fdc5284194bb3b0986c2c57aa4ea` |
| ERA5 single levels, Copernicus Climate Data Store (event windows) | https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels | `data/raw/era5/era5_2024-02-27.m1.nc` | 2026-09-19 | 29125545 | `ed6db8c52d9eba764d83a3869ac1daae7b318701fd7d5167eb45fa2c85172469` |
| 2510 county rasters | https://imagery.geoplatform.gov/iipp/rest/services/Vegetation/USFS_EDW_NLCD_Landcover_CONUS/ImageServer/exportImage | `data/raw/geography/` | 2026-09-19 | 87734820 | per file in `geography_log.jsonl` |
| 2510 county rasters | https://imagery.geoplatform.gov/iipp/rest/services/Vegetation/USFS_EDW_NLCD_TCC_CONUS/ImageServer/exportImage | `data/raw/geography/` | 2026-09-19 | 181386618 | per file in `geography_log.jsonl` |
| 2510 county rasters | https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage | `data/raw/geography/` | 2026-09-19 | 1720923413 | per file in `geography_log.jsonl` |
| 126 Soil Data Access queries | https://sdmdataaccess.sc.egov.usda.gov/Tabular/post.rest | (tabular responses, not stored) | 2026-09-19 | | per query in `geography_log.jsonl` |

## Derived tables read by the study

| content | file | date | bytes | SHA-256 |
|---|---|---|---:|---|
| EAGLE-I records by year (scripts/ingest_eaglei.py) | `data/interim/eaglei_outages_2019.parquet` | 2026-09-01 | 66811321 | `6737019e3c258b0f6c43df6277807282bf5443e21dc22132061f14b8e12dd695` |
| EAGLE-I records by year (scripts/ingest_eaglei.py) | `data/interim/eaglei_outages_2021.parquet` | 2026-09-01 | 70062640 | `da81b306796c30543ee0ab4a6af302d2e023868c6e960c5d0cf9e797ddf58ddc` |
| EAGLE-I records by year (scripts/ingest_eaglei.py) | `data/interim/eaglei_outages_2022.parquet` | 2026-09-01 | 63797872 | `facc40c5fd7699a05a5ac96e3257f6d7307e6d7188beb63aec6b13acedf5df08` |
| EAGLE-I records by year (scripts/ingest_eaglei.py) | `data/interim/eaglei_outages_2024.parquet` | 2026-09-01 | 75951752 | `9acf567bb7c53d249b4c3722c8c9dfe04c40e0d51d7a589779f3c7da738c84a8` |
| EAGLE-I 2024 modelled county customers (scripts/ingest_eaglei.py) | `data/interim/eaglei_county_customers_2024.parquet` | 2026-09-19 | 36974 | `107216f8548c301c31fe448aa65610938980feae9b7f0145cd9b445c29ca0531` |
| EAGLE-I state coverage history (scripts/ingest_eaglei.py) | `data/interim/eaglei_coverage_history.parquet` | 2026-09-19 | 10932 | `d1950c98aeb59e18b1d2c1a1c9a58db2e94fb75a7e98f901c54aad05c1f557fb` |
| County-resolved Storm Events and event-day catalogue (scripts/build_event_catalog.py) | `data/interim/storm_events_county.parquet` | 2026-09-19 | 11633999 | `89c953fc25256e87c5d00961f13cb952a1cb251eff6813d7c81b43f75275e85f` |
| County-resolved Storm Events and event-day catalogue (scripts/build_event_catalog.py) | `data/interim/event_days_stratified.parquet` | 2026-09-19 | 10972 | `09c7f3277723c2fa9591f67902905e53091b913eaeb585e014d6cd47c221c876` |
| County statics: Census, RUCC, EIA-861, customers (scripts/build_county_statics.py) | `data/interim/county_statics.parquet` | 2026-09-19 | 266782 | `c6ca6b2acfb9fb58b9ba6b5d53d2c51eab719a17ff501bc3e92bf8ad142e6003` |
| 216-hour panels, drivers, weights (build_panel216.py) | `data/interim/open_gcrk/panel216_2019-03-13.npz` | 2026-09-19 | 8769305 | `d2e5f818a00c9b04da93dfcb286762a964caa397d7329173d36bcb29be94050c` |
| 216-hour panels, drivers, weights (build_panel216.py) | `data/interim/open_gcrk/panel216_2021-03-26.npz` | 2026-09-19 | 15625717 | `7b1830824300fc8c42b8b355b0ea0efc66391708c4391424c612c44155e4c37d` |
| 216-hour panels, drivers, weights (build_panel216.py) | `data/interim/open_gcrk/panel216_2021-12-11.npz` | 2026-09-19 | 23572289 | `962c23c361c0d8536f03c585d9f140e7bf51853353fb214cd3bb0f3cd8b4379e` |
| 216-hour panels, drivers, weights (build_panel216.py) | `data/interim/open_gcrk/panel216_2022-06-08.npz` | 2026-09-19 | 13048595 | `89e4b98ced06c0cc41687b4428e21d6ce7a82da1660df3ad0c6727590ea11d6e` |
| 216-hour panels, drivers, weights (build_panel216.py) | `data/interim/open_gcrk/panel216_2024-02-27.npz` | 2026-09-19 | 12320561 | `fd97a035b62f51f9002090dc4d8a0f53545f95a3d628a39a934e4958e0018d3c` |
| 216-hour panels, drivers, weights (build_panel216.py) | `data/interim/open_gcrk/era5_county_weights_conus.parquet` | 2026-09-19 | 377651 | `2e771ca4d6ec721121f3efdd0e97643ce00b55f0db5a748c4ae0339c1f2c2a3e` |
| Geography descriptors (build_geography.py) | `data/interim/open_gcrk/geography.parquet` | 2026-09-19 | 529058 | `d3f25636e6d047e99fcf0ed7e3bfbd6e2c1ed9a8363c6f9b835ac58cdac43984` |
| Model inputs (build_features.py) | `data/interim/open_gcrk/features.npz` | 2026-09-19 | 97158561 | `1972d0589425d57fe9fd452bf9f4a9e8ba8e70b0596bf158ac1120b458efbfed` |
