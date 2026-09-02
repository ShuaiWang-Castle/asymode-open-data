# Separating interruption from restoration: identifiability and phase separation in a two-rate model of county power-outage dynamics

**DRAFT v0.1 — 2026-09-02.** Working draft for AISTATS 2027 (abstract 2026-09-29, paper 2026-10-06; 8 pages + appendix; double-blind — no author information anywhere in this file). Every number carries its evidence grade from `RESULTS_LEDGER.md`: **[A]** provable or verifiable by re-running a script · **[B]** full protocol (county-held-out folds × 3 seeds, paired, sign gate) · **[B-synth]** full protocol on synthetic data with known truth · **[C]** preliminary, not to be quoted in the final paper without a rerun. Nothing in this draft is new; it assembles the ledger. Two decisions are still the PI's and are marked **⟦PI⟧** where they bite: (1) the framing path — this draft is written on the mechanism-led path; the short-horizon path would change the abstract and §1 only — and (2) whether any metric besides RMSE is reported as a headline (none was pre-registered; §6.3 says what is and is not available).

---

## Abstract

County-level power outages are the net of two concurrent processes — interruption and restoration — observed only through their difference: the share of customers without power. We study a two-rate compartmental model in which both rates are unknown functions of exogenous weather, ask when the two can be separated from data at all, and test the model on public data (EAGLE-I county outage records, ERA5 reanalysis, NOAA Storm Events) under a pre-registered, county-held-out protocol. Theory: the one-step map is a varying-coefficient regression with known basis (1−y, −y); the two rate functions are identifiable at a driver value if and only if that driver value is observed at more than one state, and the Fisher information determinant equals n² times the conditional state variance. Two consequences follow — restoration is invisible at zero outage, and separability is supplied mostly by the cross-section of counties under the same storm hour — and both are visible in the data (effective sample size for the two rates 93:1; 74–80% of interrupted counties start from exactly zero across 22 storm days). Empirically the two-rate model beats statistical baselines by 7–9% at 24–48 h and a gradient-boosting regressor on identical information by 21% at 1 h, but loses to it by 6–8% from 6 h on; a per-horizon refit shows the long-horizon loss is structural to the model form, not a cost of producing one trajectory. The mechanism is where the structure pays: the advantage of two rates over a parameter-matched single signed rate is ordered across event families by measured phase separation (tropical −3.5%, convective −2.0%, wind ≈ 0) and reverses on winter storms (+4.3%), where interruption and restoration coincide in time — the pre-registered negative control. We argue that the value of separating the two processes is a property of the data's phase structure, and give the condition under which the separation is learnable.

---

## 1 Introduction

Resilience curves of an electric distribution grid — customers out against time through a storm — are the sum of an outage process and a restore process that overlap in time; Carrington, Dobson and Wang (2021) show any such curve decomposes into the two, retrospectively, from utility outage-management records. Forecasting models nevertheless model the net process, on the stated ground that "separately modeling these two dynamics is not practical" because they are concurrent (Zhu et al. 2021). Whether the two rates *can* be separated from observations of their difference is an identifiability question, and it has a short exact answer.

We take the simplest model that carries the distinction. With y_t ∈ [0,1] the share of a county's customers without power and x_t the exogenous drivers,

    y_{t+1} = clip( y_t + U(x_t)(1 − y_t) − R(x_t) y_t , 0, 1 ),                    (1)

with U, R unknown functions parametrised by small networks. Interruption acts on the served pool (1 − y); restoration on the interrupted pool y. Four things are established in this paper.

1. **When the two rates are separable [A].** Rearranged, one step of (1) is a varying-coefficient regression, Δy = (1−y)U(x) − yR(x). U(x) and R(x) are identified iff x is observed at two different states; the Fisher information at x has determinant Σ_{i<j}(y_i − y_j)² = n²·Var(y | x). Hence restoration carries no information at y = 0, the precision ratio of the two rate estimates is Σ(1−y_i)²/Σy_i², and — since the same weather rarely recurs at different states within one county — the separability is supplied by pooling counties under the same storm hour. This is the forced-input case that the functional-identifiability framework of Loman, Browning and Baker (2026) does not cover: their observations are state variables without exogenous forcing, and exogenous forcing is exactly what makes our two functions separable.

2. **Onset from zero is the dominant regime [A].** Across 22 storm days in four event families, 57–93% of the counties a storm interrupts sit at exactly zero beforehand (mean 73.6% on convective days, 80.5% on others). Any inflow proportional to y — the epidemic/diffusion form used in outage-propagation models — is identically zero on all of them. A steel-manned epidemic form with a learnable seed reaches parity on public data only by inflating its seed until its inflow stops depending on the state (seed dominates the inflow on 77% of scored cells) [B].

3. **Where the model wins and loses, per horizon [B].** Under a county-held-out protocol on 11 convective storm panels, the two-rate model beats damped persistence and the epidemic form by 7–9% at 24 and 48 h on every fold, and beats histogram gradient boosting on identical inputs by 21% at 1 h. From 6 h on the trees win by 2–8%. Refitting the two-rate model once per horizon closes the 6 h gap entirely and none of the 24/48 h gap: the long-horizon loss is a property of the model form, not of producing a single rollout. A rank-ceiling diagnostic computed before the comparison predicted the sign of this split.

4. **The mechanism [B at 48 h].** Against a parameter-matched single signed-rate model — the same capacity, one rate with a sign — the two-rate model's advantage is ordered across event families by the median fall/rise duration ratio of county-events (tropical 4.0, wind 2.1, convective 1.7, winter 1.0): −3.5% on tropical, −2.0% on convective, ≈ 0 on wind, and **+4.3% on winter**, the pre-registered negative control, where damage and restoration coincide in time. The ordering survives matching on event severity; it does not resolve at the grain of individual county-events within one family (a registered null we report beside it).

The paper is organised as a methods-then-synthetic-then-real study. §2 places it in the literature; §3 gives the model; §4 the theory; §5 the data and the onset audit; §6 the protocol and results; §7 the mechanism; §8 what is and is not claimed.

---

## 2 Related work

**Functional identifiability.** Loman, Browning and Baker (2026) generalise structural identifiability to unknown functions in ODE models using differential algebra: an additive unknown function beside a known parametric term is non-identifiable, "fully augmented" systems are non-identifiable, and neural ODEs are identifiable iff the full state is observed. Their observations are state variables without exogenous inputs. Our unknowns multiply *different known functions of the state* and are driven by exogenous inputs that recur across counties; §4 gives the condition for that case. Norden et al. (2025) show that non-identifiability of a mechanistic model degrades a downstream learner, which is the practical reason to care. The classical statistical object is the varying-coefficient model (Hastie and Tibshirani 1993); §4 is its specialisation to basis (1−y, −y), and we do not claim the mathematics.

**Hybrid mechanistic–neural forecasting.** Ziarelli et al. (2024) learn transmission-rate dynamics inside a SEIR layer from exogenous variables and reach the identifiability point in prose (from a low initial rate the trajectory becomes independent of the rate dynamics and the parameter is "hardly identifiable"); our Prop. 1 states the same phenomenon as a condition. Su et al. (2026) catalogue failure modes of naive neural–mechanistic couplings under partial observability and shifting rates and jointly infer time-varying transmission, recovery and immunity-loss rates; our per-horizon result (§6.2) is a measured instance of a coupling that helps at one horizon and hurts at others.

**Outage and restoration processes.** Carrington, Dobson and Wang (2021) decompose utility resilience curves into overlapping outage and restore processes and derive resilience metrics from their statistics; Afsharinejad, Ji and Wilcox (2020) document heterogeneous, policy-shaped recovery; unsupervised clustering of resilience curves finds a recovery pivot — fast then slow — within recovery (2023). Zhu et al. (2021) model the net process with a non-homogeneous Poisson intensity driven by discounted weather accumulations, evaluated in-sample; Chen et al. (2025) put a compartmental neural ODE inside a decision-focused pipeline with decision regret as the headline. Pure-ML county-level predictors (graph attention for outage duration; weather-plus-socioeconomic regressors; SARIMAX) supply the baseline context. None of these evaluates county-held-out across storms, and none states when the two processes are separable.

---

## 3 Model

**State equation.** Eq. (1). The clip is a numerical guard: at y = 1 the inflow term vanishes and at y = 0 the outflow term vanishes, so the dynamics preserve [0,1] on their own.

**Rate networks.** U(x) = cap_U·σ(f_U(x)), R(x) = cap_R·σ(f_R(x)), with f_U, f_R two-layer networks (hidden width 32 in the main runs) on the driver block; the caps are fixed in advance. Rates are composed in logit space so that any additive pathway keeps its gradient; a clamp in rate space would pin a drifting pathway at zero. With hidden width 0 each rate is a logistic GLM with readable coefficients.

**Comparators, all trained with identical inputs, capacity, optimiser, seeds and initialisation rule.**
- *transmission*: inflow u·y·(1−y) — the epidemic/diffusion form, identically zero at y = 0.
- *transmission + seed*: inflow u·(y + ε)(1−y), ε ≥ 0 learned through a softplus — the steel-manned epidemic form that can ignite from zero.
- *net*: a single signed rate, y_{t+1} = clip(y_t + cap·tanh(f(x))), no state scaling and no concurrency.
- *net_scaled*: a single signed rate with state scaling — the parameter-matched comparator for the mechanism study (hidden width 48; 3,121 parameters against 3,138 for the two-rate model).
- Statistical baselines: all-zero, persistence, damped persistence (one fitted decay), hour-of-day climatology.
- Learned baselines on identical information: histogram gradient boosting and linear models, one model per horizon (§6.2).

**Initialisation.** Each arm's initial rates are calibrated so that its initial *flows* match the observed mean one-step rise and fall on the training folds (`calibrate_init`). A network initialised at cap/2 starts at 0.125 per hour against a base rate near 10⁻⁴ and saturates the state before it can learn its way down; the first real-data run produced exactly that false negative, and the rule was fixed before any comparison was graded.

---

## 4 Theory: when the two rates are separable

Statements marked [A] are proved and numerically verified in the repository; [P] are proved in Appendix A of this draft and still to be independently checked.

**Setting.** Unclipped cells of (1) give
    Δ_t := y_{t+1} − y_t = (1 − y_t) U(x_t) − y_t R(x_t),                         (2)
a varying-coefficient regression with known basis (1−y, −y) and unknown coefficient functions of x. The continuous-time analogue dy/dt = U(x)(1−y) − R(x)y has the same structure in (y, ẏ) and every statement below transfers.

**Proposition 1 (identifiability iff conditional state dispersion) [A for n = 2; P in general].** Fix x. The conditional mean m(x, y) = (1−y)U(x) − yR(x) is affine in y with intercept U(x) and slope −(U(x) + R(x)). If the conditional law of y given x has at least two support points, (U(x), R(x)) is determined; if it is a point mass at y*, only (1−y*)U(x) − y*R(x) is, and every pair on that line fits. Hence (U, R) is identifiable on the driver support iff, for almost every x, the state at which x is observed is non-degenerate. For two observations the system has determinant y₁ − y₂.

**Proposition 2 (information equals conditional state variance) [P].** With n unclipped observations at driver value x, states y₁..y_n and noise variance σ², the design A has rows (1 − y_i, −y_i) and
    det(AᵀA) = Σ_{i<j} (y_i − y_j)² = n²·Var_emp(y | x).
By Gauss–Markov the best linear unbiased rate estimates satisfy
    Var(R̂) ≥ σ² Σ_i (1−y_i)² / det(AᵀA),   Var(Û) ≥ σ² Σ_i y_i² / det(AᵀA),
so **Var(R̂)/Var(Û) = Σ(1−y_i)²/Σy_i²**; under Gaussian noise these are the Cramér–Rao bounds.

**Corollary 2a (zero-state blind spot).** If every observation at x has y = 0, U(x) is identified (Δ = U) and R(x) carries no information; at y = 1 the roles swap. On a target where most scored cells are near zero, the precision ratio is large: restoration is learned from a small effective sample. §5.4 measures it.

**Corollary 2b (where the dispersion comes from).** Var(y | x) = E[Var(y | x, county)] + Var(E[y | x, county]). The same weather recurs rarely at different states within one county; the same storm hour hits many counties at different states. If the second term dominates, separability is a property of the cross-section, and a single county's history cannot identify R. (Registered diagnostic D-6, not yet run.)

**Proposition 3 (functional identifiability under smoothness) [P].** If U and R are L-Lipschitz and two observations within a δ-ball of x have state gap |y₁ − y₂| ≥ s, the two-row solve recovers (U(x), R(x)) with error ≤ √2(√2 Lδ + ‖ε‖)/s. The *functions* are identified up to a bias–variance trade-off governed by L, δ and the state gap; the network *weights* never are.

**Proposition 4 (reachable interval) [A].** With U ∈ [u_min, u_max] and R ∈ [r_min, r_max], the map y ↦ y(1 − U − R) + U is a contraction wherever U + R < 1, and the state, once inside, stays in [u_min/(u_min + r_max), u_max/(u_max + r_min)] regardless of the forcing path. On the synthetic generator of §6.1 the band is [0.0742, 0.9438]; the realised maximum is 0.9288 at both a 50× and a 200× forcing scale. Consequence: attainable state dispersion, hence attainable information about R, is bounded by the driver ranges alone.

**Proposition 5 (rollout stability) [P].** The one-step map has slope λ = 1 − U − R ∈ [1 − cap_U − cap_R, 1). A per-step misspecification ε propagates as e_h ≤ ε·min(h, 1/(U + R)): at most linear, saturating, never compounding. The long-horizon loss to a per-horizon regressor (§6.2) is therefore not error accumulation; it is representational — a scalar-state, current-driver map cannot express restoration that depends on time since failure.

**What the theory predicts, and where it is checked.** Prop. 2/Cor. 2a → the effective-sample-size asymmetry between the two rates (§5.4) and the onset-from-zero dominance (§5.3). Prop. 1/Cor. 2b → the ridge in synthetic recovery when state spread is small (§6.1) and the registered pooling diagnostic D-6. Prop. 5 → the per-horizon control (§6.2). The candidate link from R-identifiability to the family ordering of §7 is a hypothesis, registered with a fixed interpretation, not a result.

---

## 5 Data

All sources are public and citable; no restricted dataset, severity index, feature list, county set, fold assignment or normalisation statistic from any other source is used, and every fit starts from random initialisation.

**5.1 Target.** y_{c,t} = customers_out_{c,t} / customers_c from EAGLE-I (ORNL/DOE): county-level customers out at 15-minute resolution, 2014–2022 (DOI 10.13139/ORNLNCCS/1975202; the release ends 2022-11-12) and 2024 (DOI 10.13139/OLCF/2500278). The denominator is the publisher's own modelled per-county customer total carried in the 2024 records (3,059 counties; median 16,995; range 5 to 3,799,750; modelled from LandScan, EIA-861 and HIFLD, described by the publisher as approximate) [A]. Applied to 2018–2022 it carries customer drift, which we state; a population-share stand-in used while building the pipeline was off by a typical 23% and by more than 2× in 8.3% of counties, and no graded number rests on it.

**5.2 Zeros and the observation mask.** "Entries with 0 customers without power were not included." In 2021 the records fill 23.3% of the county × 15-minute grid (24.8 M rows, 3,045 counties); the rest is a true zero or an unobserved cell, and the file cannot say which. We densify with an explicit mask: a timestamp is a collection run if any county reports at it; a county is in service on a day if it reports within ±7 days; a cell missing while both hold is a zero, everything else stays missing and is excluded from every loss and metric. On the first eight storm windows the mask marks 95.3–99.8% of cells observed. Scored hourly cells are exactly zero 42–46% of the time.

**5.3 Events and panels.** Days are ranked by county footprint in NOAA Storm Events (bulk CSVs 2018–2025; zone-coded hurricanes, winter storms and high-wind events are expanded to counties through the NWS zone–county correlation, which takes tropical-cyclone rows from 0 to 2,758). 436 days carry ≥ 150 counties: 194 convective, 185 winter, 43 wind, 10 flood, 4 tropical. Each panel is a storm day windowed −2 to +5 days, counties gated at ≥ 70% state coverage. **Panel sets** (each pinned by a manifest digest carried in every result file): *g2-convective-11* — 11 convective-season storm days 2021–2024, the primary study (22,768 pooled (county, storm, origin) samples, 1,566 counties); *g3-all-26* — 26 panels across tropical (3), wind (4), convective (12), winter (6), flood (1), the generalisation and negative-control set; *g1* — a superseded 12-panel convective set kept for one decisive rerun.

**Onset audit [A, denominator-free].** For every county a storm later interrupts (peak y ≥ 0.01), the *median* pre-storm state is exactly zero for 73.6% of counties on the 12 convective days (range 56.6–91.0%; 89.3% below 10⁻⁴; 97.9% below 10⁻³) and 80.5% on 14 further days across winter, wind, tropical and flood — 22 storm days agreeing in direction without exception. (A first pass scored the *maximum* over the lead-in and found 2.3%; that criterion demands not one outage record in two days and is the wrong statistic; it is recorded so it is not rediscovered.)

**5.4 Drivers.** ERA5 single-levels (Copernicus CDS), CONUS at 0.25°, hourly, area-weighted to counties: 10-m wind components u10/v10 and derived wind speed, 10-m gust, 2-m temperature (°C) and relative humidity from dew point, CAPE, surface pressure, total cloud, top-layer soil moisture, total precipitation (mm), snowfall, plus a diurnal clock (sin, cos): the **14-channel block** (channel digest carried in every result). County statics (Census gazetteer and adjacency, USDA RUCC 2023, EIA-861 2023 service territory, sales and SAIDI/SAIFI) and hazard composites are built and registered for the input-asymmetry hypotheses but are not inputs to any result in this draft.

**Identification leverage, measured [A].** Eq. (2) says a cell informs U in proportion to (1−y) and R in proportion to y. Over the training folds of the primary study the Kish effective sample size is 934,811 for interruption against 9,648 for restoration — **97:1** on the g1 panels, 93:1 on g2, and 88–97:1 across three panel subsets, while the raw leverage-mass ratio swings 388–656:1 and is not quoted. The asymmetry is not that restoration is rarely observed — P(y > 0 | observed) = 0.545 — but that E[y² | y > 0] = 3.4×10⁻³: magnitude and concentration, exactly as Cor. 2a says. Per-horizon leverage is flat (no trend from 1 h to 48 h): restoration evidence does not arrive late in the window.

---

## 6 Protocol and results

**6.0 Protocol (fixed before any real fit).** Hourly resolution; forecast origins every 12 h with ≥ 24 h of history; horizons 1, 6, 24, 48 h; observed cells only. Five county-held-out folds (deterministic hash of the county code and a seed) × 3 seeds = 15 (seed, fold) units per arm, with a county-held-out inner split for early stopping. Comparisons are paired within units (fold difficulty cancels); a result carries a claim only if the sign holds in ≥ 12 of 15 units. Every result file records the panel digest, channel digest and source commit; a checker refuses to compare mismatched digests. Hypotheses, kill conditions and interpretations were written before the corresponding runs (`docs/PREREGISTRATION_*.md`); results are graded and corrections are appended, never overwritten. Diagnostics D-1 to D-5 (oracle shrinkage, rank ceiling, level/within-origin decomposition, trajectory coherence, within-family ratio) were specified before the predictions they act on existed.

### 6.1 Synthetic identifiability and onset [B-synth]

*Recovery vs state spread (EXP01).* A generator with known rates; 6 forcing levels × 3 seeds, 384 trajectories of 96 steps, recovery scored on 4,000 driver points.

| forcing | state spread | trajectory RMSE | nRMSE(U) | nRMSE(R) | corr of rate errors |
|---|---|---|---|---|---|
| 0.15 | 0.053 | 0.0094 ± 0.0066 | 0.977 ± 0.608 | 0.805 ± 0.631 | +0.78 ± 0.05 |
| 0.30 | 0.086 | 0.0038 ± 0.0008 | 0.227 ± 0.064 | 0.296 ± 0.086 | +0.63 ± 0.06 |
| 0.60 | 0.151 | 0.0039 ± 0.0005 | 0.096 ± 0.022 | 0.198 ± 0.055 | +0.46 ± 0.07 |
| 1.20 | 0.217 | 0.0030 ± 0.0004 | 0.046 ± 0.007 | 0.107 ± 0.015 | +0.43 ± 0.05 |
| 2.40 | 0.251 | 0.0026 ± 0.0004 | 0.025 ± 0.004 | 0.082 ± 0.011 | +0.37 ± 0.04 |
| 4.80 | 0.259 | 0.0021 ± 0.0002 | 0.020 ± 0.001 | 0.055 ± 0.003 | +0.36 ± 0.04 |

The pre-registered ridge prediction holds: the two rate errors are positively correlated in 18/18 fits, +0.78 at the smallest state spread, declining monotonically — an inflated U is paid for by an inflated R while the trajectory still fits [B-synth]. From forcing 0.3 to 4.8 trajectory RMSE improves 1.8× while nRMSE(U) improves 11.5× and nRMSE(R) 5.4× [C, not pre-registered].

*Onset (EXP02).* Under a neutral generator (served-pool exponent κ = 1.5, implemented by no arm), half the trajectories starting at y = 0:

| arm | RMSE all | RMSE onset | RMSE started | fitted ε |
|---|---|---|---|---|
| two-rate (susceptible) | 0.0057 ± 0.0002 | **0.0062 ± 0.0002** | 0.0052 ± 0.0002 | — |
| transmission | 0.3022 ± 0.0077 | 0.4222 ± 0.0082 | 0.0609 ± 0.0025 | — |
| transmission + seed | 0.1105 ± 0.0035 | **0.1436 ± 0.0053** | 0.0608 ± 0.0025 | 0.0078 ± 0.0001 |

The seeded epidemic form loses onset by 23× (per-seed ranges disjoint, 3/3) and started trajectories by 12×; the fitted seed stays two orders of magnitude below the states it must explain. A pure-transmission arm rolled from y₀ = 0 returns exactly 0 [A].

### 6.2 Primary study: convective panels (g2), per horizon

*Statistical baselines and the structural ladder (EXP05-g2) [B per rung].* 135 fits, 0 at the epoch cap. Paired against the two-rate model, positive = arm worse:

| arm | h+1 | h+6 | h+24 | h+48 |
|---|---|---|---|---|
| net (no scaling, no concurrency) | +4.2% 14/15 | +5.3% 15/15 | +6.9% 15/15 | +7.3% 14/15 |
| net_scaled (parameter-matched single signed rate) | −1.7% 3/15 | −1.3% 4/15 | +0.9% 9/15 | **+2.0% 12/15, t = 3.4** |
| transmission | −1.6% 3/15 | +0.6% 10/15 | +6.3% 15/15 | +9.5% 15/15 |
| transmission + seed (ε = 0.0090) | +1.4% 10/15 | +1.8% 12/15 | +0.0% 8/15 | +0.3% 6/15 |
| damped persistence | −2.4% 3/15 | +0.8% 9/15 | +5.7% 14/15 | +8.3% 15/15 |

State scaling (net → net_scaled) is worth 4–6% at every horizon, 15/15. Concurrency (net_scaled → two rates, parameter-matched) is worth +2.0% at 48 h (12/15) and +0.9% at 24 h (not significant), and costs 1.3–1.7% at 1 and 6 h. The seeded epidemic arm reaches parity by degenerating: ε = 0.0090 is 0.93× the mean scored state and dominates the inflow on 76.8% of scored cells, so on three quarters of cells its inflow is a constant multiple of (1−y). The pure epidemic form loses 6–10% at long horizons on every fold. (On the earlier 8-window set, the two-rate model beat damped persistence and the epidemic form by 7.3–9.2% at 24/48 h, 15/15, and all-zero by 11.7–13.4%; nothing beat damped persistence at 1–6 h.)

*Learned baselines on identical information (EXP07) [B for the ordering].* Positive = baseline worse than the two-rate model:

| baseline | h+1 | h+6 | h+24 | h+48 |
|---|---|---|---|---|
| gradient boosting, same inputs | **+25.9% 15/15, t = 7.5** | −2.4% 2/15 | **−7.3% 0/15, t = −9.8** | **−6.0% 0/15, t = −6.3** |
| gradient boosting + extra lead-in history | +19.3% 13/15 | −4.4% 1/15 | −7.1% 0/15 | −6.9% 0/15 |
| linear, same inputs | +109.7% 15/15 | +17.4% 15/15 | +5.0% 13/15 | +4.2% 14/15 |
| linear, unbounded head | +150.7% | +64.5% | +29.5% | +34.0% (15/15 each) |

Three alternative explanations were excluded: collapse (pred_sd 0.025, max 0.84); under-training (a 400-epoch/patience-40 refit stops naturally at 63–102 epochs, 0/5 at cap, and moves results by ≤ 2% with no consistent sign; the trees still lead by 4.1/7.7/4.8% at 6/24/48 h); a cap on the trees (10/15 folds ran the full 400 rounds — the trees' advantage is understated). The trees fit one model per horizon; the dynamics fit one rollout — controlled next.

*Per-horizon control (EXP10) [B]; post-hoc, labelled.* The two-rate model trained once per horizon with the loss taken only there; 60 fits, 0 at cap. Against the single-rollout model: −1.1% (13/15), **−2.4% (13/15, t = −4.7)**, −0.15% (8/15), +0.5% (7/15) at 1/6/24/48 h. Against the trees: dynamics win 21.5% at 1 h (0/15 for the trees); dead even at 6 h (−0.01%, 7/15); trees ahead by 7.7% and 6.9% (15/15) at 24 and 48 h. **Releasing the single-rollout constraint closes the 6 h gap exactly and none of the 24/48 h gap** — the interpretation fixed before the run.

*Rank ceiling (D-2) [A], computed before EXP07 and predicting its sign.* Ranking counties by the truth one step ahead (an illegal predictor) and scoring against the truth at h gives a ceiling on ranking skill: median Spearman 0.54 at 6 h, 0.34 at 24 h, 0.29 at 48 h; at 24/48 h the ceiling is below 0.5 in 75%/95% of forecast origins. Where ordering is unpredictable the MSE-optimal answer is each county's level, and a per-horizon regressor of the covariate-to-level map wins — the direction on record before the trees ran.

### 6.3 What the trees' win is, and what the rollout is [A]

*Decomposition (D-3).* For every arm at long horizons the level term is ≤ 0.04 of MSE (0.11–0.14 for pure transmission); the differences live in the within-origin term. The trees' per-origin Spearman is *lower* than the two-rate model's (0.25/0.17 vs 0.29/0.22 at 24/48 h, ceiling 0.34/0.29): they rank counties worse and post a lower MSE. Their gain is per-county magnitude calibration inside an origin, not ordering.

*Shrinkage (D-1).* Oracle rescaling pred′ = a·pred^λ has headroom ≤ 1% for every arm; the oracle wants the dynamics compressed (λ* = 0.88 at 48 h) and the trees expanded (λ* = 1.10, a* = 1.27). No peak-weighting or rescaling recovers the gap.

*Trajectory coherence (D-4) [B].* A pseudo-trajectory of the four horizon forecasts shows a reversal the truth does not have in 60.5–60.9% of samples for the trees and 60–63% for the linear per-horizon regressors, against 23.2–24.1% for the two-rate rollout (county-block 95% interval of the difference, worst seed, [+0.355, +0.376]); monotone baselines sit at 0. Two sentences must accompany the number: it is a property of per-horizon direct regressors, not of trees; and the rollout is not coherent either — 23.6% is not zero. **⟦PI⟧** "Produces a coherent trajectory" is unavailable; "carries 2.6× fewer excess reversals than a per-point regressor of equal accuracy" is a measured, registered property that can be reported beside RMSE, but its interpretation was fixed after EXP07 landed and the paper must say so. No alternative headline metric was pre-registered.

### 6.4 Architecture [B/C]

Capacity is doing work: reducing *both* rates to GLMs costs 3.7% at 24 h and 4.8% at 48 h, 0/15 folds better, t = 4.7/5.8 [B]. The registered capacity-asymmetry hypothesis (a GLM restoration rate would be "no worse") fails on both panel sets: 12/15 folds worse at 48 h [C, dead]. Removing ambient meteorology (cloud, pressure, humidity, temperature) from the restoration rate alone costs 1.1–1.2% at 48 h, 13/15 on both panel sets (t = 3.0, 5.5); from the interruption rate alone 0.8–1.0%, 9/15 (below the bar) on both; the registered symmetric-control condition is unmet twice and the input-asymmetry result stays [C]. A decisive rerun holding the panel set fixed and changing only the channel block reproduced the second panel set's table cell for cell, so an earlier null on 10 channels was a property of the channel block (ambient's marginal value exists only with the wind components present — an interaction), not of the panels. Noisy-OR gates on the interruption rate train without collapsing and change RMSE by ≤ 0.3% [C].

---

## 7 Mechanism: the advantage follows phase separation

**Phase separation, measured [A].** For each county-event, rise = onset crossing 0.01 to peak; fall = peak back below 0.01. Median fall/rise ratio: tropical 4.0, wind 2.1, convective 1.7, winter 1.0. Within matched severity bands (county-event peak in [0.02,0.05), [0.05,0.15), [0.15,1]) the ordering tropical > (wind ≈ convective) > winter holds in every band and winter sits at 1.00–1.25 at every severity: **phase separation is not a size effect**, but wind and convective are not separable once severity is matched. The rank ceiling separates tropical from the rest within bands (0.40/0.66 vs 0.20–0.42); the onset share does not order the families at all (0.62–0.78 within bands). The honest count is one measurement that orders the families independently of severity.

**Pre-registration (H-E1, H-E2).** H-E1: the two-rate advantage over the parameter-matched single signed rate is larger on tropical than on wind and convective, and smaller on winter than on either. H-E2 (negative control): on winter, the two-rate model does not beat the single-rate model; kill condition a > 3% win with t < −3. Comparator, parameter matching and a degeneracy rule (an arm with > 90% exact-zero predictions is labelled degenerate) were amended before the run.

**Result (EXP06 on g3, folds drawn within family, 540 rows) [B at 48 h; C at 24 h].** Paired, negative = two-rate better:

| family | fall/rise p50 | h+24 | h+48 | h+48, width-matched | converged probe (400 ep, seed 0) h+24 / h+48 |
|---|---|---|---|---|---|
| tropical | 4.0 | **−3.9% 15/15, t = −3.9** | **−5.1% 14/15, t = −3.9** | −6.1% 14/15 | −1.7% 4/5 / **−3.5% 4/5** |
| convective | 1.7 | −0.9% 9/15 | −2.0% 12/15, t = −3.4 | −3.0% 14/15 | — |
| wind | 2.1 | +0.0% 8/15 | −0.1% 8/15 | −0.3% 11/15 | — |
| winter | 1.0 | **+2.5% 5/15, t = +2.5** | **+2.8% 5/15, t = +2.4** | +2.3% 5/15 | **+4.2% 0/5 / +4.3% 0/5** |

H-E1 passes at both horizons; H-E2 survives — on winter the two-rate model is *worse* than the single signed rate, and the negative control becomes more negative when fully trained (+2.8% → +4.3%, 0/5). The 60-epoch runs hit the cap often (tropical 53/90 units, winter 31/90), so magnitudes are read from the converged probe (tropical ≈ −3.5%, winter ≈ +4.3% at 48 h) and the 24 h row is graded [C]. Against the pure epidemic form the two-rate model wins 13.0–14.8% on tropical and 5.9–6.0% on wind (15/15, t ≤ −10); the seeded form is least degenerate on tropical (seed dominates 54.7% of cells; mean state 0.048) and still loses by 7.5–7.9% there.

**Within-family null (D-5) [B as a null].** Inside the convective family, binning county-events by their own fall/rise ratio (quartiles 0.58/1.67/4.67) gives a flat advantage profile (Δ MSE −2.8×10⁻⁶, +2.2×10⁻⁴, +6.7×10⁻⁵, +1.2×10⁻⁴; county-block intervals touching zero in three of four bins; B = 2,000, 3 seeds). The mechanism is a family-level result and does not resolve at county-event grain; the paper states it so.

**Reading.** The structure pays where damage and restoration separate in time and costs a little where they coincide. Prop. 2 offers a candidate reason — families with prolonged restoration under calm weather present more state dispersion at shared drivers, hence more information about R — registered as diagnostic D-6 with a fixed interpretation, and not claimed here.

---

## 8 Discussion: what is and is not claimed

*Claimed.* (i) An exact, elementary condition for separating two concurrent rate processes from their net, with its finite-sample information formula, in a setting the functional-identifiability literature does not cover. (ii) Onset from exactly zero as the dominant regime in public county outage data, and the consequence for state-proportional inflows. (iii) A county-held-out, pre-registered evaluation on which the two-rate model beats statistical baselines at long horizons and a learned regressor at 1 h, and loses to the regressor from 6 h on for reasons that are structural and were predicted. (iv) A family-level mechanism with a negative control that behaves as one.

*Not claimed.* Per-point accuracy beyond 1 h; a coherent trajectory (23.6% excess reversals); resolution of the mechanism within a family; any capacity or input asymmetry between the two rates (both registered, both failed or preliminary); that R-identifiability explains the family ordering (registered, not run).

*Limitations.* Denominator drift (2024 totals on 2018–2022 numerators). Equal-weight RMSE on a target that is exactly zero in ~45% of cells rewards predicting nothing; we report all-zero as a baseline and per-horizon rank ceilings for that reason, and we registered no alternative headline metric. Weather is reanalysis, so every forecast has perfect driver foresight; the horizon results measure the state equation, not a forecasting system. Prop. 5 implies the natural extension — an age-structured restoration compartment — which is theory-implied and not tested.

*Reproducibility.* Every number in this draft is in `RESULTS_LEDGER.md` with its script, result file, manifest digest and source commit; `scripts/paired_review.py` regenerates every paired table from the archived result files, and `scripts/make_figures.py` regenerates every figure.

---

## Appendix A — Proofs (draft; to be checked independently)

**Prop. 1.** For fixed x write a = U(x), b = R(x). E[Δ | x, y] = (1−y)a − yb = a − (a+b)y. Two support points y₁ ≠ y₂ give a − (a+b)y₁ and a − (a+b)y₂, whose difference (a+b)(y₂−y₁) determines a+b, hence a, hence b. A single support point y* determines only a − (a+b)y* = (1−y*)a − y*b; the solution set is a line in (a, b). ∎

**Prop. 2.** Rows a_i = (1−y_i, −y_i). Cauchy–Binet: det(AᵀA) = Σ_{i<j} det[a_i; a_j]² with det[a_i; a_j] = −(1−y_i)y_j + y_i(1−y_j) = y_i − y_j. And Σ_{i<j}(y_i−y_j)² = nΣy_i² − (Σy_i)² = n²Var_emp(y). Gauss–Markov: Var(β̂) = σ²(AᵀA)⁻¹, whose diagonal entries are σ²Σy_i²/det (for the U-coordinate) and σ²Σ(1−y_i)²/det (for the R-coordinate). ∎

**Cor. 2a.** If all y_i = 0, A has rows (1, 0): U is the sample mean of Δ; the R-column is zero and R does not enter the likelihood. Symmetric at y = 1. ∎

**Prop. 3.** Let (a, b) = (U(x), R(x)) and (a_i, b_i) = (U(x_i), R(x_i)) with ‖x_i − x‖ ≤ δ. Then Δ_i = (1−y_i)a − y_i b + η_i with |η_i| ≤ Lδ·((1−y_i) + y_i) + |ε_i| = Lδ + |ε_i|. The 2×2 solve has ‖A⁻¹‖₂ = 1/σ_min(A) and σ_min(A) ≥ |det A|/‖A‖₂ ≥ |y₁ − y₂|/√2 since ‖A‖₂ ≤ √2. Hence ‖(â,b̂) − (a,b)‖ ≤ √2·‖η‖/|y₁−y₂| ≤ √2(√2 Lδ + ‖ε‖)/s. ∎

**Prop. 4.** Already verified in the repository (EXP01 correction): the map is y ↦ y(1−U−R) + U, a contraction with fixed point U/(U+R) when U + R < 1; the fixed points over the rate box are bounded by the stated ratios. ∎

**Prop. 5.** e_{h+1} = |f̂(ŷ_h) − f(y_h)| ≤ |f̂(ŷ_h) − f(ŷ_h)| + |f(ŷ_h) − f(y_h)| ≤ ε + λ e_h with λ = 1 − U − R < 1, so e_h ≤ ε Σ_{k<h} λ^k ≤ ε·min(h, 1/(1−λ)). ∎

## Appendix B — Pre-registration and grades (pointer)

`docs/PREREGISTRATION_asymmetry.md` (H-A1/A2/A3/A3′/B/C/D), `docs/PREREGISTRATION_phase_separation.md` (H-E1/E2 with three amendments), `docs/PREREGISTRATION_external_priors.md` (externally suggested priors, each labelled directional/generic and re-tested here or unrun), `docs/PREREGISTRATION_exp01_h2.md`. Grades and every correction: `RESULTS_LEDGER.md`. Sixteen graded findings and the three framing paths: `docs/EVIDENCE_SUMMARY.md`.

## Appendix C — Figures

fig01 synthetic identifiability sweep · fig02 synthetic onset · fig03 onset audit on public data · fig04 two-rate advantage over the parameter-matched single rate by family, with converged-probe markers · fig05 per-horizon comparison on the primary study. All regenerated from archived result files by `scripts/make_figures.py`.

## Appendix D — Open items before submission

1. ⟦PI⟧ Framing path (this draft: mechanism-led) and headline-metric decision (§6.3).
2. Independent check of Appendix A; D-6 and S-2 (theory plan) run and graded.
3. Full read of Loman–Browning–Baker §3 and Su et al. for the related-work claims.
4. Convergence-budget rerun of EXP06 at 24 h if the [C] row is to be reported.
5. LaTeX, anonymisation check, 8-page fit.
