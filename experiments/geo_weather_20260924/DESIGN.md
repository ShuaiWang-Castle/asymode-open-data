# DESIGN v1 — exposure-integrated local hazards

Status 2026-09-25. v1 replaces v0 (section 7) after two reviews: `contrib/REVIEW_formal.md` (identification,
one principle, falsifiers) and `notes/LITERATURE_framework.md` (literature map, adversarial review); the
physical operators and constants follow `contrib/MECHANISMS_structure.md`.

## 1. Principle

Assumptions: (A1) the county outage fraction is a linear functional of local outage states over the county's
exposure measure; (A2) the hazard at a location depends only on the local forcing history and local attributes;
(A3) one map with shared parameters for every location; (A4) causal, fading memory; (A5) zero hazard in calm
weather for every geography; (A6) independent failure mechanisms add their hazards (competing risks).

Then the county damage input is

    Lambda_i(t) = integral  sum_m lambda_m[ xi_<=t(x), g(x) ]  K_i(dx)

with K_i the county's normalised exposure measure, xi the local forcing, g the local attributes. The county-level
model is the special case in which K_i is a point mass. By Boyd & Chua (1985) a causal fading-memory operator is
approximated by a bank of exponential filters followed by a static readout, so a fixed filter bank loses nothing
in principle.

**What the identity does and does not say.** The gap between Lambda_i and lambda(county means) splits into a
*marginal term* (the county's static exposure-weighted distribution of geography, pushed through the event's
weather) and a *coupling term* (which part of the geography lies under which part of the weather field). A
county-level vector that encodes the distribution (for example quantiles) can supply the marginal term; only the
coupling term needs sub-county fields, and it vanishes where the resolved weather is uniform over the county.
The defensible claim is therefore the *interface* — damage is linear in the exposure measure, so a shared local
response integrated against that measure has the right inductive bias and no capacity to memorise a county —
not that county-level information is insufficient. (v0 claimed the latter; that was wrong.)

Recoverability ladder: L0 county means; L1w hazards per weather cell, area-weighted; L1w' the same,
population-weighted; L1g the customer-weighted distribution of downscaling-relevant geography (elevation offset
from the model orography, wetness, canopy) under the event's weather; L2 co-location inside the county; L3
sub-grid weather not predictable from geography, network topology, restoration, reporting (not recoverable here).

## 2. Minimal form: exposure-integrated hazard (EIH)

* **Nodes** k = (ERA5 cell c, band of dz), dz = elevation - ERA5 model orography (surface geopotential / g), fixed
  band edges -300, -150, -50, +50, +150, +300 m; weights = WorldPop 2020 population; attributes = population-weighted
  means inside the node (`build_nodes_cs.py`).
* **Fixed downscaling D** (no learned constant): T_k = T_c - Gamma(month) dz_k with the monthly near-surface lapse
  rates of Kunkel (1989) as tabulated by Liston & Elder (2006); dew point from the mean of a monthly dew-point lapse
  and constant relative humidity; wet-bulb temperature (Stull 2011); snow fraction sigmoid((1.0 - Tw)/0.7)
  (T50 from Jennings et al. 2018, scale from Dai 2008).
* **Dictionary psi_b** (non-negative, zero in calm, fixed knots instead of learned thresholds): gust ramps
  [G - 10]+, [G - 15]+, [G - 20]+; liquid precipitation; precipitation x hat functions of node wet-bulb temperature
  at -3, -1.5, 0, +1.5, +3 C; convective log1p(CAPE/1000 x precipitation).
* **Modulators m_a(g_k)**: 1, canopy, canopy x leaf-on (May-Oct), poorly drained share. Each multiplies a gated
  psi, so the county-specific part changes with the weather by construction. Variant `quadn` normalises each
  modulator to mean one over the county (geography then only redistributes hazard inside the county).
* **Memory bank F_tau**: identity and unit-gain exponential filters, tau = 3, 12, 48 h, run from the first prefix
  hour. Linear and time-invariant, so applied after the exposure integral.
* **Features** Phi_i,j(t) = F_tau * sum_k w_ik m_a(g_k) psi_b(D(W_c(k)(t), g_k)): 10 x 4 x 4 = 160 per county-hour,
  precomputed once per data version (`build_eih.py`); each scaled by its fitting-set 99th percentile.
* **Entry into the host** as a competing hazard: u = u_host + (cap - u_host)(1 - exp(-beta . Phi_t)), beta >= 0
  (projected), beta = 0 at the start (the arm equals W+Cin exactly at step 0, and d u / d beta = (cap - u_host) Phi
  there: no saddle, no clamp, no stale standardisation), beta on a slow clock (0.1 x the host step), fixed from the
  first run. Mechanism features stay out of the recovery network.

Parameters: 160 non-negative coefficients; no threshold, band edge, lapse rate or time constant is learned.

## 3. Arms and falsifiers (run in this order; each can end the line)

* **F0 variance audit (no training).** The same dictionary computed on the ladder: `area` (cells, area weights),
  `pop` (cells, population), `quad` (cell x band), `mean` (one node), `pooled` (bands under county-mean weather),
  `other` (another county's bands from the same relief stratum under this county's weather). Report the share of
  county-hours, outage-weighted, where quad - pop and pop - area are material, by event. Kill the quadrature if
  quad - pop is negligible where outages happen.
* **F1 residual structure (no training).** Regress the out-of-fold W+Cin residuals on those contrasts against a
  permutation null within event x state.
* **F2 fixed-feature screen** (program.md quick screen): W+Cin vs W+Cin+H with Phi from `area`, `pop`, `quad`,
  `mean`, `pooled`, `other`, `quadn`. Predictions if the framework is right: quad <= pooled < mean, quad < other,
  gains located in the county-hours F0 flags. Two numbers for every gain: against its placebo and against the base.
* **F3 planted recovery** at the amplitude F0 allows, with proxy exposure weights and compound noise.
* **F4 event transfer**: the event-grouped design; per-event gains must rank with each event's F0 contrast.

## 4. Extensions, only after F2 shows signal

Low-rank load x trigger slots (latent mechanisms: a geography profile x a weather response); node outage stocks
(depletion of the served pool; frailty selection); a learnable population/road exposure mix; the local 98th
percentile of daily-maximum gust (Klawa & Ulbrich 2003) in place of fixed gust knots (a monthly-mean proxy is
being fetched; it assumes one tail-to-mean ratio everywhere); ERA5 precipitation type and pressure-level
downscaling (TopoSCALE, Fiddes & Gruber 2014); HRRR as an oracle downscaler.

## 5. Known and new

Known: the identity (ecological bias, disaggregation regression), quadrature plus downscaling (TopoSUB/TopoSCALE,
elevation bands, MicroMet), parameter learning through a process model (dPL, UDE, MPR), fragility and accretion
physics, local-then-aggregate outage models. Plausibly new: disaggregation regression through a two-rate
population balance, with nested placebos that separate information from interface. Main risk: the gain comes
from exposure weighting or a better weather product, which is data engineering and will be reported as such.

## 6. Data limits known before any run

The twelve-event panel was selected on wind-report footprints: no event lists ice storm among its main report
types and six of twelve are warm-season convective events, so winter-phase mechanisms can barely be tested on it.
The population-weighted spread of node elevation offsets inside a county has a median of about 20 m (about one
county in fifteen exceeds 100 m). Both are to be measured, not assumed, in F0.

## 7. History

v0 (2026-09-24): learned mechanism scalars (lapse rate, thresholds, band edges, time constants) in bounded
sigmoids, county-wide strata mixing cells, intensities standardised once and read into the damage network by a
zero-initialised matrix. Retired for the reasons in the two reviews: zero gradient at step 0, clamping of the
most intense hours, loss of monotonicity, a static canopy offset in calm air, weak identification of thresholds
and time constants, and strata that do not change the weather. The module stays in `src/asymode/geo_mech.py`
for reference.

## 8. Status after the first night (2026-09-25; RESULTS.md)

* The audits are part of the framework, not an afterthought: because every county enters only through its
  exposure measure, what sub-county geography *can* change is computable before training (F0), and whether it
  aligns with what the host misses is testable before training (F1). On the twelve-event wind panel both say no
  (sub-grid content < 1% of outage-weighted county-hours; effects worth 2% of pooled RMSE would have been detected),
  and the trained arm agrees (+0.73%, interval +-4%). On ice storms F0 finds content exactly where the phase
  physics puts it (near-freezing precipitation), and the trained arms are too noisy on eight events to decide;
  the single strongest feature is pre-registered for 18 independent ice-storm episodes (`PREREG_W2.md`).
* v1.1 of the dictionary: plain rain is a load (antecedent wetness), not a trigger; the pathway with rain triggers
  added diffuse hazard (mean bias up, peaks not sharper).
* The largest gain of the night is not geographic: training on EAGLE-I targets without their collection artefacts
  (-2.66% vs base, -3.40% vs a matched placebo).
