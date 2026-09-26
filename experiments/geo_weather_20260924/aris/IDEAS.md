# Ideas ledger (ARIS style)

Status: **active**, **parked**, **abandoned** (with the reason and, where applicable, prior art), **absorbed** (merged
into another idea).

| id | idea | status | note |
|---|---|---|---|
| I01 | GCRK: geography-conditioned response kernel on the damage hidden state | parked | fails on public data (open_gcrk RESULTS 18-21); PI suggests testing it without the bound on its opening alpha (I10) |
| I02 | v0 local mechanisms with learned physical scalars | abandoned | zero gradient at step 0, clamping, weak identification (DESIGN section 7) |
| I03 | Exposure-integrated hazard (EIH): customer-weighted integral of local, weather-gated, fading-memory hazards; competing-hazard entry | active (framework) | DESIGN v1; audits F0/F1 built in |
| I04 | Sub-county geography via ERA5 elevation bands | abandoned on these data | C01, C02, C06 |
| I05 | HRRR as the weather source of the system | active | C06, C07; host inputs not yet on HRRR |
| I06 | Load x trigger slots (chain reactions) | active, untested at scale | W1 event-grouped single seed: -0.28% vs base |
| I07 | Within-county normalised susceptibility | parked | W1 single seed -1.42%; no fingerprinting by construction |
| I08 | Target cleaning of EAGLE-I artefacts | abandoned as a gain | C03 |
| I09 | Stratified, warning-based episode panel for the whole system | active (design) | DATASET_DESIGN v0, under review |
| I10 | Remove the bound on GCRK's opening (beta = alpha instead of tanh(alpha)); and the non-negativity clip of the EIH coefficients (signed hazard) | active (PI request 2026-09-26) | to test on the new panel with three seeds |
| I11 | Host peak magnitude (predicted peaks 0.18x observed above 20%) | active | the largest error budget (notes/DATA_PATTERNS.md) |
