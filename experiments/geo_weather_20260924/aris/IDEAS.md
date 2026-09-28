# Ideas ledger (ARIS style)

Status: **active**, **parked**, **abandoned** (with the reason and, where applicable, prior art), **absorbed** (merged
into another idea).

| id | idea | status | note |
|---|---|---|---|
| I01 | GCRK: geography-conditioned response kernel on the damage hidden state | parked | fails on public data (open_gcrk RESULTS 18-21); PI suggests testing it without the bound on its opening alpha (I10) |
| I02 | v0 local mechanisms with learned physical scalars | abandoned | zero gradient at step 0, clamping, weak identification (DESIGN section 7) |
| I03 | Exposure-integrated hazard (EIH): customer-weighted integral of local, weather-gated, fading-memory hazards; competing-hazard entry | abandoned (PI 2026-09-27) | geography enters only through GCRK or a kernel architecture in the hidden layer, never through hand-built hazard features; DESIGN v1; audits F0/F1 built in |
| I04 | Sub-county geography via ERA5 elevation bands | abandoned on these data | C01, C02, C06 |
| I05 | HRRR as the weather source of the system | active | C06, C07; host inputs not yet on HRRR |
| I06 | Load x trigger slots (chain reactions) | active, untested at scale | W1 event-grouped single seed: -0.28% vs base |
| I07 | Within-county normalised susceptibility | parked | W1 single seed -1.42%; no fingerprinting by construction |
| I08 | Target cleaning of EAGLE-I artefacts | abandoned as a gain | C03 |
| I09 | Designed panel of parent weather systems for the whole system | active (design registered) | DATASET_DESIGN v1: five regimes, sealed confirmation tranche, pre-window gates, near-miss controls, regime-balanced estimand |
| I10 | Remove the bound on GCRK's opening (beta = alpha instead of tanh(alpha)); and the non-negativity clip of the EIH coefficients (signed hazard) | active (PI request 2026-09-26) | to test on the new panel with three seeds |
| I11 | Host peak magnitude (predicted peaks 0.18x observed above 20%) | active, **next by DATASET_DESIGN 9.3** (Stage 0: two headline regimes) | the largest error budget (notes/DATA_PATTERNS.md); the host fails zero in synoptic wind and heavy rain on the designed panel |
| I12 | Hazard dictionary for every regime (local gust exceedance, phase-resolved precipitation, temperature-gated loads, node-level compound products, antecedent wetness, convective organisation) | active (design) | from the physical review of the panel design (contrib/REVIEW_dataset_physics.md section 5) |
| I13 | Diagnose the heavy-rain loss of the hazard pathway (+11% vs host at seed 0 with the full or the mechanical dictionary) before any new arm: which active terms fire in heavy-rain systems (candidate: poorly drained x wetness x gust exceedance, where wet soils meet thunderstorm gusts without outages), by stratum and hazard class | active (diagnostic, development only) | 2026-09-27 cycle; coefficients in runs/.../v1_H*_s0/fold0k/DONE.json |
| I14 | Spatio-temporal GCRK: the response states of neighbouring counties are coupled inside the kernel's recurrence, e_t = M_t^-1 (P_t e_{t-1} + f_t), P_t a Markov mixing (non-negative shares, row sums <= 1, so the unit bound holds): a static share kappa_s over the neighbours (exp(-d / 50 km) x exp(-gamma |code_i - code_j|^2), the geography code of the kernel), and a directional share kappa_a from upwind neighbours (ERA5 10 m wind at the pair's midpoint, scaled by speed / 10 m/s). Neighbours: the sampled county-events of the same system within 150 km, at most 8 (panel_v1/space_v1.py). All three scalars start at 0, so the arm equals GCRK at step 0 (paired init) | screened 2026-09-27, seed 0: ties the host (pooled -0.06%, regime-balanced +0.3%), beats temporal GCRK (-0.50% pooled, -0.4% balanced; tropical -3.2, winter -1.5, synoptic +2.5); coupling used in 2 of 5 folds, gamma ~ 0; not queued for more seeds | reason: GCRK is only a temporal kernel; each county's state sees its own weather alone. Single seed first: STGCRK+Cin seed 0 against GCRK+Cin and the host W+Cin seed 0, five event folds. Later (PI): a recovery-side kernel; a spatial null (neighbours replaced at matched distance) only for a survivor |

## Proposed after the frozen kernel review (2026-09-28; no training authorization)

| id | idea | status | note |
|---|---|---|---|
| I15 | Recovery hidden-state response kernel driven by the existing damage hidden departure and predicted damage rate; minimal version has no predicted-stock feedback | proposed, awaiting PI discussion | Kernel states and codes are active but forecast contributions remain small; recovery lacks dedicated impact history. Fixed-damage zero-recovery ceiling still misses most large peaks, so this tests history, not a claim that slower restoration cures magnitude. Formulas, boundedness, paired initialization and screening in notes/KERNEL_PROPOSALS_20260928.md. |
| I16 | Public service-county links inside the same recovery recurrence, with fixed context counties and bounded Markov mixing | proposed, after I15 and public-graph audit; awaiting PI discussion | The sampled spatial graph is not a service network; strong learned mixing has negligible frozen output sensitivity. Public EIA links are a proxy for shared service context, not observed crew flows; context coverage and sealing must be audited before implementation. |
| I17 | Population/geography-type response states within each county, combined after local nonlinear processing into the one shared damage hidden layer | proposed alternative, awaiting PI discussion | Targets exposure aggregation and magnitude. Existing geography already includes 50 km summaries and co-location fields; the change is preserving local weather-geography correspondence before aggregation, not adding another county-level feature list. |

No proposal is queued. Frozen constant-geography substitutions are dependence diagnostics, not trained NULLs.
The registered primary estimand after Stage 0 is tropical/winter balanced MSE; all-five balance and pooled RMSE
are reported separately. The prospective single-seed screen must be explicit and frozen before a PI-approved run.

## I18 geographic-kernel screen (2026-09-28; authorized before training)

| id | idea | status | note |
|---|---|---|---|
| I18 | Replace the county-specific geography norm in GCRK with one frozen training-fold RMS norm, retaining radial geography information; all rate/kernel parameters and recurrence otherwise unchanged | running since 14:55 ET, code/design 395350d; seed-0 five-fold screen authorized by PI in this chat | notes/I18_GEO_RMS_SCREEN_20260928.md; arm GCRK+Cin-georms, label v1_gcrk_georms_s0; compare host and original GCRK; no recovery work or extra seeds now |
