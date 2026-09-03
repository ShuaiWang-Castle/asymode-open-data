# Integrity audit

manifest: configs/panel_manifest_g2-convective-11.json
panel_digest: 76a73ed794af
channel_digest: dec964873cb2
events (11): 2021-05-04, 2021-06-21, 2021-08-11, 2021-12-11, 2022-04-13, 2022-06-08, 2022-06-17, 2022-07-23, 2024-05-08, 2024-05-26, 2024-06-26

## Panel / driver alignment

| event | counties | 15-min steps | hours | fips match | ts UTC 15-min monotone | driver hours >= panel hours |
|---|---|---|---|---|---|---|
| 2021-05-04 | 344 | 673 | 168 | yes | yes | yes |
| 2021-06-21 | 256 | 673 | 168 | yes | yes | yes |
| 2021-08-11 | 253 | 673 | 168 | yes | yes | yes |
| 2021-12-11 | 256 | 673 | 168 | yes | yes | yes |
| 2022-04-13 | 207 | 673 | 168 | yes | yes | yes |
| 2022-06-08 | 183 | 673 | 168 | yes | yes | yes |
| 2022-06-17 | 356 | 673 | 168 | yes | yes | yes |
| 2022-07-23 | 186 | 673 | 168 | yes | yes | yes |
| 2024-05-08 | 236 | 673 | 168 | yes | yes | yes |
| 2024-05-26 | 339 | 673 | 168 | yes | yes | yes |
| 2024-06-26 | 230 | 673 | 168 | yes | yes | yes |

## Feature block

scored channels (14): `cape, cloud, gust, precip, pressure, rh, snowfall, soil_moisture, t2m_c, u10, v10, wind_speed, clock_sin, clock_cos`
matches the public 14-channel block: **yes**

## Clock provenance

clock rebuilt independently from panel timestamps matches the harness: **yes**
clock differs from the legacy lead-phase channel: **yes**
distinct clock phases across sampled origins: 2 (a lead-time clock would give 1)

## Observation mask

pooled samples: 22,768 · counties 1,566 · events 11
scored cells per horizon: 22,558 (h+1) .. 22,542 (h+48)
unobserved share excluded from every loss and metric: 0.95%
all masked-in targets finite: **yes**

## Verdict

**PASS — no integrity or alignment check failed.**
