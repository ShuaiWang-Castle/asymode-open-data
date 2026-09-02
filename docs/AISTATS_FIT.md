# Does this paper fit AISTATS? Venue evidence and the literature to cite or argue with

Written 2026-09-02 from the conference's own paper lists (`virtual.aistats.org`,
JSON endpoints `aistats-{2024,2025,2026}-orals-posters.json`) and from the abstracts
of the external papers named below. The 2026 file carries titles but no
abstracts, so cross-year counts use **titles only**; 2024/2025 abstracts were
read for the papers quoted. Keyword screens are indicative, not a census.

## 1. What AISTATS accepts in our neighbourhood

Title-only screen, identical regular expressions for every year:

| year | unique titles | ODE / dynamical-system / state-space / mechanistic | identifiability | time series / forecasting | named science or infrastructure application |
|---|---|---|---|---|---|
| 2024 | 555 | 7 | 3 | 14 | 1 |
| 2025 | 591 | 13 | 6 | 15 | 2 |
| 2026 | 576 | 9 | 6 | 21 | 3 |

Three readings, all of which matter for how the paper is written:

1. **AISTATS is not an applications venue.** Titles naming a domain (climate,
   grid, epidemiology, ecology, physics) are under 1% every year. A paper whose
   title and abstract say "power outage forecasting" is competing in a slot
   that barely exists.
2. **Dynamical systems, identifiability and time series are stable, small
   tracks** (about 5–6% of the programme combined, flat to slightly rising).
   The identifiability cluster in 2026 alone: ε-identifiability of causal
   quantities; affine identifiability of nonlinear CCA; identifiability of
   degenerate Gaussian mixtures; identifiability of tensor ranks;
   parameter-identifiable physical reasoning. A proved identifiability
   condition is a native AISTATS object.
3. **What the accepted dynamics papers look like** (abstracts read): ODE
   parameter estimation with probabilistic solvers and multi-modal likelihoods
   (2024); equation discovery with sparsity priors (2024); GP-ODEs (2025);
   Koopman-equivariant GPs with forecasting uncertainty (2025); adjoint
   sensitivity analysis for dynamical systems (2025); state-space parameter
   estimation by SMC (2025); cost-aware simulation-based inference (2025);
   structured temporal inference in state-space models (2026); **fundamental
   limits of active learning for linear dynamical systems (2026)** — a
   limits/negative paper; and one plainly applied hybrid paper, residual
   mechanistic-plus-ML models for biopharmaceutical production (2025). Methods
   with a statistical argument, occasionally an applied hybrid when the
   modelling contribution is crisp.

**Verdict.** The project fits AISTATS under one condition: the paper is a
statistical-ML paper whose *vehicle* is outages, not an outage paper that uses
ML. Concretely, the contributions have to be ordered as

* the identifiability condition [A] — two unknown rate functions of shared
  exogenous inputs acting on one observed state, separable exactly when the
  state differs across steps that share drivers (the determinant argument), with
  the synthetic recovery experiment as its numerical check;
* the evaluation methodology — county-held-out folds across storms, paired
  comparisons with a sign gate, masked scoring on densified data, and the
  fixed-in-advance pathology screens (oracle shrinkage, rank ceiling,
  level/within-origin decomposition, trajectory coherence);
* the empirical findings *as findings about structure*: the coupling wins where
  the state carries the information (1 h) and loses where covariates dominate
  (≥ 6 h), the per-horizon refits show this is structural, and the family-level
  phase-separation result with its winter reversal is the mechanism.

The framing decision still open (narrow to short horizons / change the headline
metric / lead with mechanism and accept the per-point negative) interacts with
venue: the first and third are AISTATS-shaped; a decision-relevant headline
metric (as in Chen et al. 2025) pulls the paper toward the applications shape
that AISTATS does not take.

## 2. Literature to cite, grouped by the argument it serves

Three of the closest papers were already read in full — see
`docs/RELATED_WORK_PRECHECK.md` (Chen et al. 2025; Ziarelli et al. 2024; Zhu et
al. 2021). The rest below were read at abstract level unless stated.

### A. Functional identifiability — the theory we sit next to

* **Loman, Browning, Baker (2026), "Structural functional identifiability and
  model discovery in differential equation models", arXiv 2606.30289.**
  Generalises structural parameter identifiability to *unknown functions*, via
  differential algebra. Negative results: an additive unknown function beside a
  known parametric term is non-identifiable (their Prop. 3.1); "fully augmented"
  systems with an additive unknown function in every equation are
  non-identifiable (Prop. 3.3); neural ODEs are identifiable iff the full state
  is observed; non-identifiable hybrids "reduce to neural ODEs". **Where we
  differ, and it is a real gap:** they assume observations are state variables
  and treat no exogenous, time-varying forcing; they do not treat one observed
  scalar state carrying *two* unknown functions of the same inputs. That is
  precisely our setting, and exogenous forcing is what makes our two functions
  separable. Cite as the general framework; state our condition as the
  forced-input analogue it does not cover.
* **Norden, Oostwal, Chappell, Tino, Bunte (2025), "On the importance of
  structural identifiability for machine learning with partially observed
  dynamical systems", arXiv 2502.04131.** Non-identifiability of a mechanistic
  model degrades a downstream learner; making the equivalence classes explicit
  repairs it. Supports the claim that identifiability is a prerequisite for the
  learned rates to mean anything.
* Review: **"Structural identifiability of compartmental models: recent progress
  and future directions" (2025), arXiv 2507.04496.** Background for the
  compartmental framing.
* **"Identifiability challenges in sparse linear ODEs" (2025), arXiv
  2506.09816.** Identifiability failures even in linear ODEs — useful to argue
  that the question is not trivial.

### B. Hybrid mechanistic + neural forecasting — the method family

* **Su, Lee, Cui, Ramakrishnan (2026), "How (Not) to Hybridize Neural and
  Mechanistic Models for Epidemiological Forecasting", arXiv 2602.06323.**
  Catalogues failure modes of "seemingly straightforward couplings" under
  partial observability and shifting rates; their fix makes non-stationarity
  explicit (trend/seasonal/residual control signals driving a controlled neural
  ODE) and jointly infers time-varying transmission, recovery and immunity-loss
  rates. Closest in spirit to our learned-rate design. **Discussion point:** our
  per-horizon result is a clean, measured instance of a coupling that helps at
  one horizon and hurts at others, on a target they do not study; and our
  ambient ablation shows a covariate whose value exists only in the presence
  of other covariates (interaction), which is the kind of thing their
  "make it explicit" prescription would have to handle.
* **Ziarelli et al. (2024)**, arXiv 2410.11545 — learned transmission-rate
  dynamics inside SEIR; reaches the identifiability point in prose. The
  structural template for our paper (already in the pre-check).
* **"Forecasting seasonal influenza epidemics with physics-informed neural
  networks" (2025), arXiv 2506.03897** (SIR-INN) — SIR structure inside a
  network for national flu forecasting; a data point that the compartment-plus-
  network shape is current.
* **"Structured hybrid mechanistic models for robust estimation of
  time-dependent intervention outcomes" (2026), arXiv 2602.11350** and
  **"Automatic and structure-aware sparsification of hybrid neural ODEs"
  (2025), arXiv 2505.18996** — hybrid-model methodology; cite in related work,
  not load-bearing.
* **AISTATS 2025, "Improving N-glycosylation and biopharmaceutical production
  predictions using AutoML-built residual hybrid models"** — venue precedent
  for an applied hybrid paper.

### C. The outage-process literature — the application's own prior art

* **Carrington, Dobson, Wang (2021), "Extracting resilience metrics from
  distribution utility data using outage and restore process statistics", IEEE
  Trans. Power Systems, arXiv 2011.00693.** "A resilience curve generated from
  utility data can always be decomposed into an outage process and a restore
  process, and these processes generally overlap in time." **This is the direct
  support for the two-process view** — and it is retrospective and event-level,
  from utility outage-management records. We do the decomposition
  prospectively, from public county aggregates, with learned rates, and we say
  exactly when the two processes are separable. Follow-up: Dobson group,
  "Quantifying distribution system resilience from utility data" (2024), arXiv
  2407.10773.
* **Zhu et al. (2021)**, arXiv 2109.09711 — the foil: states that because
  outage and restoration are concurrent, "separately modeling these two
  dynamics is not practical", and models the net process. Carrington–Dobson–Wang
  say the decomposition always exists; Zhu et al. say it is not practical to
  model; our identifiability condition adjudicates between them.
* **Afsharinejad, Ji, Wilcox (2020), "Heterogeneous recovery from large scale
  power failures", arXiv 2012.15420** and the same group's Joule (2021) paper
  on recovery services — recovery has its own scaling and heterogeneity,
  distinct from failure; supports treating restoration as a process with its
  own drivers.
* **"Unraveling fundamental properties of power system resilience curves using
  unsupervised machine learning" (2023), arXiv 2310.10030** — 200+ resilience
  curves cluster into triangular and trapezoidal archetypes; recovery has a
  *pivot* (fast, then slow). Complementary to our fall/rise asymmetry: they
  find structure *within* recovery, we find the asymmetry *between* onset and
  recovery and its ordering across event families.
* **Chen, Fioretto, Qiu, Zhu (2025)**, arXiv 2502.18321 — compartmental neural
  ODEs for outages with decision regret as headline (pre-check). Pure-ML
  outage predictors for baseline context: graph attention for outage duration
  (2025, arXiv 2511.10898); weather plus socio-economic predictive modelling
  (2025, arXiv 2512.22699); SARIMAX outage prediction (2025, arXiv 2511.01017).

### D. AISTATS-native anchors (cite sparingly, to signal venue fit)

2026: "High effort, low gain: fundamental limits of active learning for linear
dynamical systems" (limits precedent); "Structured temporal inference in
state-space models"; the identifiability cluster listed in §1. 2025:
"Koopman-equivariant Gaussian processes" (forecasting uncertainty for dynamical
systems); "Local stochastic sensitivity analysis for dynamical systems". 2024:
"Data-adaptive probabilistic likelihood approximation for ODEs" (parameter
estimation with deep local maxima — adjacent to our initialisation/calibration
findings); "Equation discovery with Bayesian spike-and-slab priors".

## 3. The argument the introduction can make, in four moves

1. Resilience curves decompose into an outage and a restore process that
   overlap in time (Carrington–Dobson–Wang), yet the modelling literature treats
   the net process because separating concurrent processes is "not practical"
   (Zhu et al.).
2. Whether they *can* be separated is an identifiability question about two
   unknown functions on one observed state — a case outside the functional
   identifiability framework of Loman–Browning–Baker, which assumes state
   observation without exogenous forcing. We give the condition, prove it, and
   verify it on a synthetic generator that none of the compared arms can
   represent.
3. On public data, under a county-held-out protocol with pre-registered
   hypotheses and pathology screens, the separated model wins where state
   information dominates and loses where covariates dominate; per-horizon
   refits show the loss is structural, in line with the failure modes
   catalogued by Su et al.
4. The mechanism: the two-rate advantage over a parameter-matched single rate
   is ordered by measured phase separation across event families and reverses
   on the family with none (winter), which is the negative control.

## 4. What was not done

No full text of the external 2025–2026 papers was read beyond the three in the
pre-check; the 2026 AISTATS list has no abstracts; keyword screens can miss
papers whose titles are oblique. Before submission, the writing session should
read Loman–Browning–Baker §3 and Su et al. in full — both could either sharpen
or complicate the identifiability and hybrid-failure claims above.
