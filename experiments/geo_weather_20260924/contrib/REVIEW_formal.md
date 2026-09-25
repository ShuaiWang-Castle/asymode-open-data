# Formal review — DESIGN v0 "local geo-weather mechanisms integrated over where the customers are"

Requested 2026-09-24 by the open-data line. Read: `DESIGN.md`, `program.md`,
`experiments/open_gcrk_20260919/RESULTS.md` §§11–21, and `docs/CONSULT_IDENTIFIABILITY.md` §2 for
the settled propositions cited below. Methods only. Every number quoted here comes from those files
(section given); nothing was run for this review, and the physical constants were not re-checked
against the literature.

## 0. Verdict

1. **The identity is right; the conclusion drawn from it is not.** DESIGN §1 is correct accounting
   under locality and additivity over customers. But "no county-level geography vector, memory or
   conditioning can supply it" is false in general and false for the document's own example (§1.3).
   Most of the aggregation gap is a known functional of the event's weather and the county's
   *static*, customer-weighted distribution of local conditions, and a county-level vector that
   encodes that distribution recovers it. Only a co-location term needs sub-county fields; it
   vanishes wherever the resolved weather is uniform over the county, and it is probably small at
   0.25° (F1 measures it).
2. **The same over-reach sits in RESULTS §18.2** ("No static county descriptor, of any kind and
   through any interface, can explain more than that share"). An intraclass correlation bounds
   *additive* county effects. A static descriptor that interacts with event weather varies from
   event to event and is not bounded by it. RESULTS §21 is an instance: W+Cin −2.4% against −0.8% for
   the constant level term W+C, and in that section's own words "the context does more through the
   network's non-linearity with the weather than as a shift". Both sentences should be corrected
   before anything is written up.
3. **The defensible contribution is therefore the interface, not the information.** County damage
   is *linear in the customer measure*. A model that integrates a shared local response against that
   measure has the right inductive bias; county-level interfaces that push means through a
   non-linear network must learn the integral from data and gain capacity to fingerprint in
   exchange. The placebos have to separate the two (§4, F2).
4. **As specified, the mechanism parameters are weakly identified** (§2). Gains are absorbed by the
   host's first layer; thresholds, band edges and time constants are identified only by
   within-county spread and event shape; the lapse rate and the band width are nearly degenerate;
   sigmoid-bounded scalars stall at their bounds; full-batch Adam moves rarely active parameters as
   fast as dense ones.
5. **There is one clean principle** (§3): the county damage input is an exposure-weighted integral
   of local, weather-gated, fading-memory hazards with shared parameters. A minimal form with good
   gradients has no learned threshold, edge or time constant (fixed physical downscaling, fixed
   feature basis, fixed log-spaced memory bank, R unlabeled load × trigger slots, about 130
   parameters). With a *normalised* within-county susceptibility, geography cannot single out a
   county by construction, and its coefficients are identified by within-county variation only.
6. **Three experiments can falsify it** (§4), the first without training: an information audit; a
   nested information-versus-interface test with hard placebos and localisation of the gain; event
   transfer with a planted control.

## 1. The identity and what is recoverable (Q1)

### 1.1 Exact content

Write K_i for the county's exposure measure (DESIGN uses customer density ρ_i; §1.6 argues the right
measure is broader), K̃_i for it normalised to mass one, g(x) for local static attributes, ξ_t(x) for
the local forcing history up to t, and λ(ξ, g) for the local hazard. Let

```
μ_{i,t} = law of (ξ_t(X), g(X)) when X ~ K̃_i
Λ_i(t)  = ∫ λ dμ_{i,t}                                   (DESIGN §1)
```

Two exact consequences:

* **Sufficiency.** Under locality nothing about county i matters beyond μ_{i,t}. A model that is a
  functional of μ has no county-specific degree of freedom; county identity cannot help it.
* **Aggregation gap.** Λ_i(t) − λ(mean ξ, mean g) = ½ tr(∇²λ · Σ_{i,t}) + higher moments, with
  Σ_{i,t} the covariance of (ξ, g) under μ_{i,t}. Its blocks are gg (static spread of geography times
  the event's curvature), ξξ (within-county weather spread) and ξg (co-location). This is aggregation
  bias in the sense of Theil (1954), and the reason land-surface models carry sub-grid tiles
  (Avissar & Pielke 1989; Giorgi 1997).

### 1.2 Marginal term and coupling term

Let ν_i be the K̃_i-weighted distribution of g (static) and π_{i,t} the K̃_i-weighted distribution of
ξ_t (event-specific). Decompose λ by functional ANOVA with respect to ν_i ⊗ π_{i,t}
(λ = λ_0 + λ_g + λ_ξ + λ_gξ, each non-constant part centred in each of its arguments). Because
μ_{i,t} has marginals ν_i and π_{i,t},

```
Λ_i(t) = ∫ λ d(ν_i ⊗ π_{i,t})  +  ∫ λ_gξ d(μ_{i,t} − ν_i ⊗ π_{i,t})
          marginal term              coupling term
```

**Lemma.** ∫ λ dμ takes the same value for every μ with marginals (ν, π) near the independent
coupling iff λ is additively separable, λ(ξ, g) = a(ξ) + b(g). *Sketch:* "if" is immediate. Otherwise
there are support points with λ(ξ,g) + λ(ξ′,g′) ≠ λ(ξ,g′) + λ(ξ′,g); moving mass ε from the pairs
(ξ,g′), (ξ′,g) to (ξ,g), (ξ′,g′) keeps both marginals and changes the integral by ε times that
non-zero difference (small neighbourhoods and continuity for continuous marginals).

What follows:

* The **marginal term** needs the *distributions* ν_i and π_{i,t}, not their means, and it changes
  from event to event through π_{i,t} although ν_i is static. A county-level vector that encodes ν_i
  (for example K quantiles), entering through a function of the event's weather, can supply it.
* The **coupling term** needs co-location: which part of the county's geography lies under which
  part of the event's weather field. No county-level vector supplies it. It is zero whenever the
  resolved weather is uniform over the county (μ = ν ⊗ δ) or λ is additive.

### 1.3 The DESIGN §1 example is a marginal term

Take a county inside one weather cell with T_k = T_c + Γ(z_k − z_c). Then

```
Λ_i = p · E_{Δz ~ ν_i}[ band(T_c + Γ·Δz) ]
```

a county-specific smoothing of the band, evaluated at the event's cell temperature. It moves with the
freezing level through T_c, which the host already reads; the county-specific ingredient is the
static distribution of customer-weighted elevation offsets. A county-level quantile vector of Δz plus
a band-shaped interaction with T_c reproduces it up to quadrature error. The quadrature is the right
*way* to compute it (linear in ν_i, few parameters, no capacity to memorise), but it is not
information that a county-level vector cannot hold.

In set-learning terms the quadrature is a Deep-Sets / kernel-mean-embedding model whose outer map is
the identity (Zaheer et al. 2017; Muandet et al. 2017; distribution regression: Szabó et al. 2016).
Additivity over customers is what justifies the identity outer map. That is the claim worth making.

### 1.4 The recoverability ladder (a natural order for the experiments)

| level | what the model sees | what it recovers | status here |
|---|---|---|---|
| L0 | county means of weather and geography | λ(mean ξ, mean g) | original host and descriptors |
| L1w | non-linear hazards per weather cell, then averaged | ξξ block | done with area weights in round 2 (RESULTS §14; its −2.9% bundles this with other input changes) |
| L1w′ | the same, weighted by population instead of area | where the hazard meets the customers | not run; zero new parameters |
| L1g | + customer-weighted static distribution of downscaling-relevant geography (Δz from model orography, exposure, wetness), integrated | gg block and the forcing spread that downscaling creates; the DESIGN example | the new idea |
| L2 | + co-location of nodes with their own cell's weather inside the county | ξg coupling between cells | small unless the county's cells carry different weather in the same hour |
| L3 | — | sub-grid weather not predictable from geography (convective cells), network topology, restoration operations, reporting | unrecoverable from any of these inputs |

The ICC of 0.06 (RESULTS §18.2) says the residual is not a county constant; it does not say how it
splits over L1–L3. The ceiling of this design is the L1g + L2 share, and that can be measured before
any training (F1).

**Resolution.** A 0.25° cell is about 28 × 21 km at 40° N (≈ 600 km²); a typical eastern-US county
overlaps a handful of cells. With ERA5 alone, L2 comes only from differences between those few cells,
and inside a cell the only sub-grid forcing variation is the one downscaling derives from geography,
which is L1g. If L2 is the claim, a km-scale analysis is needed; HRRR analyses exist for all twelve
events of the current set (2019–2024).

### 1.5 Present versus learnable

"Recoverable" has a second meaning: whether λ can be *learned* from county aggregates even when μ is
known. That is ecological inference (Robinson 1950; King 1997). With λ = Σ_j θ_j ψ_j the outcome sees
only the integrals ∫ ψ_j dμ_{i,t}, and θ is identified iff those integrals are not collinear over the
observed (i, t). For interaction terms the integrals are nearly collinear with their marginal-product
versions whenever the coupling term is small, so the design can be correct and still unlearnable at
this noise; §2.1 turns this into a computable quantity.

### 1.6 Two corrections to the identity itself

* **Exposure is not customer density.** In a radial distribution network a customer loses power when
  an element on its upstream path fails. The measure that multiplies local hazard is
  K_i(dx) ∝ (line density at x) × (customers interrupted by a fault at x, i.e. those downstream of the
  device that clears it): supported on the network, broader than customer density, and heavier on
  rural, forested line corridors between settlements than their population suggests. For small
  hazards the identity holds with K_i in place of ρ_i (for large ones the path saturates,
  1 − exp(−∫_path λ)). Topology is not public; use population smoothed with a feeder-scale length ℓ,
  or road density as a proxy, and pre-register a short sensitivity set (ℓ = 0, a few km, 10–15 km).
  Everything in §§1.1–1.5 holds with K_i.
* **Precedents.** Population-weighted local exceedances normalised by local climatology are an
  established storm-loss index (Klawa & Ulbrich 2003: Σ population × (v/v₉₈ − 1)³ for v > v₉₈); a
  static sub-grid distribution plus one dynamic store is TOPMODEL's saturated-area construction
  (Beven & Kirkby 1979). The novelty cannot be the integral; it can be the integral inside a learned
  population balance with identification guarantees. The local-quantile threshold is itself a
  parameter-free geography × weather interaction (assets adapted to the local climate), and it
  answers RESULTS §18.1 (geography is already in the climatology) directly.

## 2. Identifiability, gradients and stability (Q2)

### 2.1 Where the information on geography coefficients comes from

For a log-linear modulation λ = exp(α·g(x))·h(ξ),

```
∂Λ_i/∂α = ∫ g λ dK̃_i = ḡ_i Λ_i  +  Cov_{K̃_i}(g, λ)
                        between      within
```

(the Mundlak 1978 split of g into county mean and deviation). The *between* part is confounded with
every county-level factor correlated with g; it is what fingerprinted in RESULTS §§18–19. The *within*
part is non-zero only where the event's hazard varies inside the county. The regionalised design buys
the within part and nothing else; where within-county hazard contrast is small, α is again a
county-level regression coefficient with all the problems already measured. Two consequences:

* separate α_b (on ḡ_i) from α_w (on deviations): the claim is α_w ≠ 0 and transferable, and α_b is
  where fingerprinting lives (shrink it hard or fix it at zero) — or use the normalised
  susceptibility of §3.3, which keeps only the within part by construction;
* the within information depends on weather, geography and K_i alone, so it can be computed before
  training and converted into a minimum detectable effect (F1d).

### 2.2 Mechanism channels as host inputs: absorbed directions

DESIGN §2(d) feeds the M intensities into the damage network's first layer, z = W x + b. Any
parameter direction δθ with ∂Λ_m/∂θ ≈ cΛ_m + d over all county-hours is absorbed by that layer:

* **Gains** (prefactors such as b and c in §2(c)): ∂Λ/∂gain = Λ/gain, exactly absorbed, never
  identified. Fix them.
* **Fragility threshold** in softplus(a·log(I/I₀)): for I ≪ I₀ the channel is ≈ (I/I₀)^a and only
  w·I₀^(−a) is identified; for I ≫ I₀ it is a·log I − a·log I₀ and the offset goes to the bias. I₀ is
  identified only by county-hours near I₀.
* **Wind threshold** in [W_k − θ_w]₊²: ∂Λ/∂θ_w = −2 Σ_k ρ_k [W_k − θ_w]₊, which escapes absorption
  only because the within-county exceedance distribution changes shape. θ_w is identified by exactly
  the within-county spread the identity relies on.
* The host's occurrence gate and the (1 − P) factor add two multiplicative paths that can switch a
  channel off, and the damage sigmoid sits at 0.022–0.032 on large rises (RESULTS §11), so
  σ′ ≈ 0.02–0.03 multiplies every gradient that reaches a mechanism through it.

Recommendation: **enter mechanisms as an additive hazard** (independent failure modes; competing
risks):

```
u = 1 − (1 − u_host) · exp(−Δt Σ_m Λ_m)        ∂u/∂Λ_m = Δt (1 − u_host) exp(−Δt Σ Λ) ≈ Δt
```

No absorption, no σ′ attenuation, and no extra host capacity reading county-varying channels. The slot
weights stay only partly separable from the host's own weather response, so the claims should rest on
the geography coefficients and on nested comparisons, not on reading slot weights as physical
constants.

### 2.3 Thresholds and band functions

* **Aggregation smooths a threshold only where the within-county distribution has mass at it:**
  d/dθ ∫ 1[x > θ] dμ = −(density of μ at θ). With a few dozen nodes a hard threshold gives a
  piecewise-constant loss (zero gradient almost everywhere). Use each node's internal spread as the
  softness: if node k stands for a stratum with forcing mean x_k and spread s_k, then
  ∫ 1[x > θ] dN(x_k, s_k²) = Φ((x_k − θ)/s_k). The temperature is then the quadrature's own error model,
  not a tuning knob.
* **Sigmoid-bounded scalars** θ = lo + (hi − lo)·σ(φ) have ∂θ/∂φ → 0 at the bounds, so a parameter
  pushed to a bound stays there. Prefer unbounded parameters with a quadratic penalty toward the
  physical value; log-parameterise positive quantities.
* **Dead thresholds.** [W − θ_w]₊² has zero gradient wherever every node is below θ_w. A 31-km
  reanalysis gust is smoother than a station gust, so a station-derived θ_w starts dead in most
  county-hours. Exceedance over the local climatological quantile (Klawa & Ulbrich) removes the
  threshold parameter altogether.
* **Lapse rate versus band edges.** band(T_c + Γ·Δz) depends on (T_lo − T_c)/Γ and (T_hi − T_c)/Γ,
  i.e. on the band in elevation units. Γ and the band width trade off unless T_c varies a lot while
  precipitation falls, and near-freezing precipitation is exactly when it does not. Fix Γ, or better,
  decide phase per node from ERA5's 0 °C-isotherm height (check its convention when the surface is
  already below freezing and when there are two warm layers), and learn the band's shape, not Γ.
* **Band shape.** Represent band(·) as a non-negative combination of fixed B-spline bumps on a
  wet-bulb grid around 0 °C (for monotone responses, I-splines with non-negative weights: Ramsay
  1988). Parameters enter linearly and there is no edge pair that can collapse (T_lo ≥ T_hi is an
  absorbing zero-gradient state for learned edges).
* **Literature constants.** Please cite a source for each literature-initialised constant (θ_w, band
  edges, τ ranges). Where no source can be given, initialise from the physical definition (wet-bulb
  near 0 °C) with the spline representation and let the data move it.

### 2.4 Time constants

Estimating the time constants of a sum of exponentials is ill-conditioned (Lanczos 1956; Istratov &
Vyvenko 1999), and within a 144-hour forecast window a time constant of several days is barely
distinguishable from infinity. Making τ_S and τ_L functions of geography stacks two ill-posed
problems, and the kernel record shows the symptom: learned memory lengths varied 9% across counties
(RESULTS §12) and planted ones were not recovered (§18.4). Replace learned τ by a fixed log-spaced bank
with non-negative mixing weights and let geography move the mixture (§3.3). Where the physical state
is observed, use it: ERA5 volumetric soil water replaces the saturation store S_k and its τ_S.

### 2.5 Through the population balance

* **Heterogeneity depletion.** With node hazards h_k the correct inflow is Σ_k ρ_k (1 − P_k) h_k,
  not (1 − P)·Σ_k ρ_k h_k; the difference is −Cov_ρ(h, P). The most exposed nodes go out first, so the
  county-level inflow for the same weather falls as the event proceeds: frailty selection (Vaupel,
  Manton & Stallard 1979). A county-level balance fed with aggregated hazard will read this as fatigue
  or saturation. Either carry node states P_k (cheap: K per county) or register the bias. Separating
  true state dependence from heterogeneity is the mixed-proportional-hazard problem (Elbers & Ridder
  1982; Heckman & Singer 1984); it needs covariate variation, which the node attributes provide.
* **Initialising node states** at the forecast origin, where only the county fraction y is observed:
  the minimum weighted-KL projection of a prior vector q onto Σ_k ρ_k P_k = y is a common logit shift,
  logit P_k = logit q_k + δ, with δ from a monotone one-dimensional root find (differentiable through
  the implicit-function theorem).
* **Damage versus restoration.** Prop 1 of `docs/CONSULT_IDENTIFIABILITY.md` applies unchanged: U and
  R separate only through state dispersion. A mechanism tied to a condition that also slows restoration
  (heavy wet snow) can be traded against slower recovery. Keep mechanism channels out of the recovery
  network; timing then separates them (damage acts on onsets, restoration on decays).

### 2.6 Stability of node-level states

* **Stores.** For dS/dt = f − S/τ with piecewise-constant input, use
  S_{t+1} = e^{−Δt/τ}·S_t + τ(1 − e^{−Δt/τ})·f_t: unconditionally stable and positivity-preserving.
  Explicit Euler is unstable when Δt/τ > 2 (τ under 30 minutes at hourly steps), which a sigmoid range
  may admit.
* **Outage state.** With hazards λ_u, λ_r ≥ 0 and Λ = λ_u + λ_r,

  ```
  P_{t+1} = P_t + (λ_u (1 − P_t) − λ_r P_t) · Δt · φ(Λ Δt),     φ(x) = −expm1(−x) / x
  ```

  is exact for piecewise-constant rates, stays in [0, 1] without a clamp, and has
  ∂P_{t+1}/∂P_t = e^{−ΛΔt} ∈ (0, 1]: a contraction (the exact-integrator counterpart of Prop 4 in the
  consultation brief), so rollout gradients decay rather than explode.
* **Products of stores** (wind × S, precipitation × band × canopy) stay bounded if every factor is
  bounded: normalise S by field capacity and cap accretion loads at a physical maximum.
* **Spin-up.** Run every store from the first prefix hour; RESULTS §11 found the analogous origin
  artifact in the kernel's path summaries.

### 2.7 Optimisation

Full-batch Adam normalises each parameter's step, so a parameter active in a handful of county-events
moves as fast as a dense one and can be fitted to those events. RESULTS §20 shows the same effect from
the other side: conditioning maps on the host's clock overfit, and the same maps at a tenth of the step
were harmless. Put mechanism and geography parameters on a slower clock, or on penalties toward
physical values, as a fixed design choice from the first run.

## 3. One principle (Q3)

### 3.1 Statement

Assumptions:

* **A1 Additivity over customers.** The county outage fraction is a linear functional of local outage
  states, Y_i(t) = ∫ P_t dK̃_i. (Counting; exact.)
* **A2 Locality.** The hazard at x depends only on the local forcing history and local attributes;
  network non-locality is carried by K_i.
* **A3 Shared parameters.** One map for every location; heterogeneity only through ξ(x), g(x), K_i.
* **A4 Causality and fading memory.**
* **A5 Weather gating.** The hazard is zero (or a shared background) in calm conditions for every g.
* **A6 Superposition.** Independent failure mechanisms have additive hazards (competing risks).

**Principle.** The county damage input is

```
Λ_i(t) = ∫ Σ_m λ_m[ ξ_{≤t}(x), g(x) ] K̃_i(dx)
```

with each λ_m a local, causal, fading-memory, weather-gated functional with location-independent
parameters. The county-level model is the special case in which K̃_i is a point mass at one
representative location.

**Representation.** By Boyd & Chua (1985), a causal time-invariant operator with fading memory can be
approximated uniformly on bounded input sets by a finite bank of linear exponential filters followed
by a static polynomial readout. Placing a static feature map before the filters (Hammerstein–Wiener)
makes accretion-type memory ("integrate a non-linear function of the input") cheap. A low-rank,
sign-constrained, second-order readout of that structure is the minimal form below; as rank and basis
grow, the class becomes dense in local fading-memory operators, so nothing is given up in principle.

### 3.2 Minimal parameterisation

```
node forcing (fixed physics)   f_k(t) = F(D(w_cell(t), g_k))          non-negative, zero in calm
memory bank (fixed τ_j)        m_{k,j}(t) = (h_{τ_j} * f_k)(t)         τ_j log-spaced, hours to days
slot r = 1..R (unlabeled)      λ_{r,k}(t) = s_{r,k} · (a_r · f_k(t)) · (b_r · [1, m_k(t)])
county input                   Λ_i(t) = Σ_k ρ_k Σ_r λ_{r,k}(t)
entry into the host            u = 1 − (1 − u_host) · exp(−Δt Λ_i(t))
```

* D is physical downscaling with fixed constants (lapse rate or 0 °C-isotherm height for phase;
  terrain exposure for wind). F is a fixed set of non-negative features that vanish in calm: gust
  exceedance over the local climatological quantile, precipitation, precipitation × fixed spline bumps
  of wet-bulb temperature around 0 °C, observed soil wetness × precipitation, leaf-on × canopy. Nothing
  else is hand-coded: physics where the physics is settled (the forcing), learning where it is not (the
  damage response).
* a_r, b_r ≥ 0 (softplus-parameterised); the constant 1 in the load vector allows purely instantaneous
  slots.
* s_{r,k}: normalised susceptibility (§3.3).
* Size: with about 6 features, 3 time scales, R = 4 and 8 attributes, about 4 × (6 + 19 + 8) ≈ 130
  parameters. No learned threshold, edge or time constant; each parameter enters linearly inside exp
  or inside a product of non-negative linear forms. Initialise a_r and b_r small and positive, not at
  zero: a product of zero-initialised factors is a saddle with zero gradient (and RESULTS §18.4 shows
  how slowly even non-degenerate zero-initialised maps travel at this step size).

How the named mechanisms appear without being coded:

* **Memory:** b_r selects filtered channels. Geography-conditioned dissipation becomes
  geography-dependent weights over the fixed bank (slots with different τ and different α), which is
  well conditioned where learning τ(g) is not.
* **Rain → saturation → windthrow:** one slot with trigger a = gust exceedance, load b = filtered
  precipitation or observed soil wetness, susceptibility on drainage/wetness index and canopy.
* **Elevation → temperature → wet snow:** downscaled wet-bulb temperature at the node (D), feature =
  precipitation × near-0 °C bumps, load = short-τ filter (accretion with shedding), trigger = 1 or
  wind, susceptibility on canopy.
* **New mechanisms:** more slots with free non-negative a, b over the whole feature set, named after
  the fact from (a_r, b_r) and held to the same tests.

Symmetries to fix: scale inside a product (normalise ‖a_r‖ = ‖b_r‖ = 1 and carry the scale in s),
permutation of slots, and degeneracy when a_r and b_r read the same features (give each slot physically
distinct instantaneous and filtered sets). Non-negativity helps uniqueness, as in non-negative matrix
factorisation. Choose R from a pre-registered short list (e.g. 2 or 4), not by a sweep.

### 3.3 Normalised susceptibility: no fingerprinting, within-county identification

Let s_{r,k} = exp(α_r·(g_k − ḡ_i)) / Σ_j ρ_j exp(α_r·(g_j − ḡ_i)), so that Σ_k ρ_k s_{r,k} = 1. Then

```
Λ_{i,r}(t) = E_ρ[λ_r] + Cov_ρ(s_r, λ_r)          ∂Λ_{i,r}/∂α_r = Cov_{ρ·s_r}(g, λ_r)
```

* **Fingerprinting is impossible by construction.** If the local hazard is uniform over the county
  (calm, or uniform forcing without downscaling contrast), geography has no effect at all; in general
  |effect| ≤ sd_ρ(s_r)·sd_ρ(λ_r). The effect is spanned by R·q shared coefficients, and nothing is
  indexed by county.
* **Identification is within-county by construction:** α_r moves only with the within-county,
  event-specific co-variation of hazard and geography (the within part of §2.1). The only unnormalised
  route for geography is D, whose constants are fixed.
* **Division of labour:** county-level differences in vulnerability go to the county context (which
  already helps, RESULTS §21); geography redistributes hazard inside the county. The cost is at most the
  between-county geography effect, which RESULTS §12 (E0), §16 (D3) and §18 show to be small and
  non-transferable on these data. A shrunk between term can be added later as a nested test.
* **Nesting:** α = 0 is the geography-blind shared operator (the analogue of the shared kernel, the best
  kernel arm on the five-event panel, RESULTS §18.3), and one node per county is the county-mean model.
  Both are exact special cases in parameter space; arms that differ only in whether α is trained are
  identical at step 0 and can be bit-checked there.

## 4. What would falsify it (Q4)

**F1 — information audit (no training; run first).** Mechanism parameters fixed at physical values.

* (a) *Contrast.* Δ_{i,m}(t) = Λ_quad − Λ_mean (one node at the county mean) and
  C_{i,m}(t) = Λ_quad − Λ_pooled (all of the county's nodes under county-mean weather: L1g without
  co-location). Report the share of county-hours and the number of distinct events in which |Δ| and |C|
  exceed a registered fraction of the host's typical damage rate. A mechanism with contrast in fewer
  than about three events cannot be tested for transfer.
* (b) *Redundancy.* County-grouped regression of the new sub-grid summaries (customer-weighted quantiles
  of Δz from model orography, exposed-ridge share, wetness-index distribution) on the host's county-mean
  ERA5 channels, exactly as in RESULTS §18.1. High R² means the host already has them.
* (c) *Alignment.* Correlation of Δ and C with the base's out-of-fold residuals at the county-event
  level, sign pre-registered, against a null that recomputes Δ with the node sets of another county from
  the same relief stratum; calibrate the null's false-positive rate on synthetic residuals before
  reading it.
* (d) *Within information.* The Fisher information for the within coefficient (§2.1) under the base's
  residual noise, converted into a minimum detectable effect in the evaluation metric and set against
  the evaluation's resolution of about 2% (RESULTS §§12, 18).
* (e) *Ceiling.* Add the F1 intensities and distribution summaries to the unit-level gradient-boosted
  regressor of RESULTS §16 (D3), same folds and targets.

Kill: negligible contrast (a), high redundancy (b), no alignment beyond the null (c), or a minimum
detectable effect above anything (a) and (e) make plausible. The identity's extra information is then
not present at this resolution, and the remedy is data (population weighting, km-scale forcing, more
events of the relevant type), not model.

**F2 — information versus interface (trained; screen protocol, paired initialisation).** Arms on the
current base, with mechanisms entering as in §2.2:

* +M_mean (one node at the county mean);
* +M_pooled (the county's node distribution under county-mean weather: integral interface, no
  co-location);
* +M_quad (nodes under their own cell's weather);
* +M_other (nodes of another county *from the same relief stratum*: a hard placebo — a flat county's
  nodes in a mountain county make an easy one);
* optionally +DS (the same node set as a free set embedding into the host's first layer: M_pooled's
  information through a non-linear interface).

Predictions if the framework is right: M_quad ≤ M_pooled < M_mean, M_quad < M_other, M_pooled < DS, and
the improvement sits in the county-hours F1 flagged. Kill: M_quad ≈ M_other (capacity, not
information); M_quad ≈ M_mean (no distributional content); or a pooled gain carried by county-hours
where Δ ≈ 0 (not the identity's mechanism). The F1-flagged subset is defined without outcomes, so
registering it as a secondary evaluation set adds power legitimately. Report against the base and
against the placebo, with event × state intervals as well as county intervals: the claim is about
events.

**F3 — event transfer, with a planted control.**

* (a) Plant one local mechanism on the real fields, folds and noise model of RESULTS §18.4, at the
  amplitude F1 says could exist. The protocol must recover the within coefficient (sign in at least four
  of five folds, interval coverage) and produce a gain above the evaluation's resolution. If it cannot,
  a null on real data is uninformative and should not be run as a claim.
* (b) Real data, event-grouped design: M_quad − M_mean must be negative on held-out events (event ×
  state interval below zero), and per-event gains must rank with each event's F1 contrast
  (pre-registered Spearman > 0). A gain that appears only in the county-grouped design, where the events
  were seen in training, falsifies "moves with the event". For memory slots, a common circular time
  shift of the forcing (the same shift at every location, which keeps spatial and temporal dependence)
  must remove the gain.

## 5. Smaller points

* **Run L1w′ first:** recompute the round-2 cell-level hazards with population instead of area weights.
  No parameters; it isolates the "where the customers are" half of the identity.
* **Nodes:** stratify the joint customer-weighted distribution per county-cell (weighted k-means, or
  quantiles of a composite index), not the product of marginal strata, which grows as K^q and loses the
  correlation between elevation, canopy and population.
* **Reference height:** use ERA5's own surface geopotential as z_c (2-m temperature refers to model
  terrain); the DEM enters only through z_k.
* **Mechanisms without sub-grid forcing** (CAPE, cell precipitation) gain nothing from the quadrature
  beyond the susceptibility redistribution; keep them in the host.
* **program.md rule 2:** for a claim about events, decide on the event × state interval; the county
  interval understates uncertainty when counties share an event's weather.
* **Two sentences to correct** (§0, items 1–2): RESULTS §18.2's "any interface" and DESIGN §1's "no
  county-level geography vector, memory or conditioning can supply it".

## References

* Avissar, R., & Pielke, R. A. (1989). A parameterization of heterogeneous land surfaces for
  atmospheric numerical models and its impact on regional meteorology. *Monthly Weather Review*,
  117(10), 2113–2136.
* Beven, K. J., & Kirkby, M. J. (1979). A physically based, variable contributing area model of basin
  hydrology. *Hydrological Sciences Bulletin*, 24(1), 43–69.
* Boyd, S., & Chua, L. O. (1985). Fading memory and the problem of approximating nonlinear operators
  with Volterra series. *IEEE Transactions on Circuits and Systems*, 32(11), 1150–1161.
* Elbers, C., & Ridder, G. (1982). True and spurious duration dependence: the identifiability of the
  proportional hazard model. *Review of Economic Studies*, 49(3), 403–409.
* Giorgi, F. (1997). An approach for the representation of surface heterogeneity in land surface
  models. Part I: Theoretical framework. *Monthly Weather Review*, 125(8), 1885–1899.
* Heckman, J., & Singer, B. (1984). A method for minimizing the impact of distributional assumptions
  in econometric models for duration data. *Econometrica*, 52(2), 271–320.
* Istratov, A. A., & Vyvenko, O. F. (1999). Exponential analysis in physical phenomena. *Review of
  Scientific Instruments*, 70(2), 1233–1257.
* King, G. (1997). *A Solution to the Ecological Inference Problem*. Princeton University Press.
* Klawa, M., & Ulbrich, U. (2003). A model for the estimation of storm losses and the identification
  of severe winter storms in Germany. *Natural Hazards and Earth System Sciences*, 3(6), 725–732.
* Lanczos, C. (1956). *Applied Analysis*. Prentice Hall.
* Muandet, K., Fukumizu, K., Sriperumbudur, B., & Schölkopf, B. (2017). Kernel mean embedding of
  distributions: a review and beyond. *Foundations and Trends in Machine Learning*, 10(1–2), 1–141.
* Mundlak, Y. (1978). On the pooling of time series and cross section data. *Econometrica*, 46(1),
  69–85.
* Ramsay, J. O. (1988). Monotone regression splines in action. *Statistical Science*, 3(4), 425–441.
* Robinson, W. S. (1950). Ecological correlations and the behavior of individuals. *American
  Sociological Review*, 15(3), 351–357.
* Szabó, Z., Sriperumbudur, B. K., Póczos, B., & Gretton, A. (2016). Learning theory for distribution
  regression. *Journal of Machine Learning Research*, 17(152), 1–40.
* Theil, H. (1954). *Linear Aggregation of Economic Relations*. North-Holland.
* Vaupel, J. W., Manton, K. G., & Stallard, E. (1979). The impact of heterogeneity in individual
  frailty on the dynamics of mortality. *Demography*, 16(3), 439–454.
* Zaheer, M., Kottur, S., Ravanbakhsh, S., Póczos, B., Salakhutdinov, R., & Smola, A. (2017). Deep
  Sets. *Advances in Neural Information Processing Systems 30*.
