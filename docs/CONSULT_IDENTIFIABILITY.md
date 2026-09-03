# Consultation brief: when can two concurrent rate processes be separated from their net?

You are being asked to consult on the design of the decisive experiment for one
question in an ongoing study. Read this brief, then answer the questions in the
last section. **Do not start implementing.** Argue about design first.

---

## 1. The object

A county's power-outage state is `y ∈ [0,1]`, the fraction of tracked customers
without power, observed every 15 minutes. It moves under two processes that never
appear separately in the data:

    y_{t+1} = clip( y_t + U(x_t)(1 − y_t) − R(x_t) y_t , 0, 1 )

`U` is the interruption rate acting on the still-served pool `(1−y)`; `R` is the
restoration rate acting on the interrupted pool `y`. Both are unknown functions of
exogenous weather `x_t` (a 14-channel block: CAPE, cloud, gust, precipitation,
pressure, RH, snowfall, soil moisture, 2-m temperature, u10, v10, wind speed, and
sin/cos of the UTC hour). They are parameterised as `cap·σ(f(x))` with small
two-layer networks and fitted by rollout MSE over a 48-hour horizon.

Only `y` is observed. `U` and `R` never are.

The application literature disagrees about whether this separation is even worth
attempting. One line shows any resilience curve decomposes into an outage process
and a restore process that overlap in time (Carrington, Dobson & Wang 2021, IEEE
Trans. Power Systems). Another models the net process on the stated ground that
because the two are concurrent, "separately modeling these two dynamics is not
practical" (Zhu et al. 2021).

## 2. What is already settled, with its grade

Grades: **[A]** provable or directly verifiable · **[B]** full protocol
(held-out folds, ≥3 seeds, paired, sign gate) · **[C]** preliminary.

**Prop 1 — identifiability iff conditional state dispersion [A].** One unclipped
step is a varying-coefficient regression with a known basis:

    Δy = (1−y)·U(x) − y·R(x) + ε,        φ(y) = (1−y, −y)

The conditional mean is affine in `y` with intercept `U(x)` and slope
`−(U(x)+R(x))`. So `(U(x), R(x))` is identified from the conditional mean **iff**
the conditional law of `y` given `x` is not a point mass. Two observations sharing
`x` at states `y₁ ≠ y₂` give a 2×2 system with determinant `y₁ − y₂`.

**Prop 2 — information equals conditional state variance [A].** With
`Q(x) = E[φ(Y)φ(Y)ᵀ | X=x]`:

    det Q(x) = Var(Y | X=x) =: v(x),     ½ ≤ λ_max(Q) ≤ 1,     v ≤ λ_min(Q) ≤ 2v

so `v(x)` is, within a factor of two, the curvature in the weakest identifiable
direction — not merely a convenient determinant. Finite sample at fixed `x`:
`det(AᵀA) = Σ_{i<j}(y_i−y_j)² = N²·v̂`, and the efficient least-squares variances
are `Var(Û) = σ²B̂/(N v̂)`, `Var(R̂) = σ²Â/(N v̂)` with `Â = mean (1−y)²`,
`B̂ = mean y²`. **`Â/B̂` is a variance ratio, not a precision ratio**; the
precision ratio is its reciprocal. Verified by 1,726 randomised property tests
including endpoints, point masses and heavily imbalanced designs.

**Orthogonal coordinates [A].** With `μ(x) = E[Y|X=x]`, put
`α = (1−μ)U − μR` (the drift at the mean state) and `τ = U + R` (total turnover).
Then `E[Δ|x,y] = α − τ(y − μ)` and the Gram matrix is exactly `diag(1, v)`. So
`α` is always identified; **all of the identification difficulty lives in `τ`, and
its information is exactly `v(x)`.** The unidentified direction at a point mass is
`n_μ = (μ, 1−μ)`, and adding `c·n_μ` changes the drift by `c(μ − Y)`, with
conditional signal energy `c²v`.

**Theorem 5 — the exact one-rate approximation gap [A].** Against the best
state-scaled *single signed rate* (which can be a positive inflow on `(1−y)` or a
negative outflow on `y`, but not both at once), the conditional excess risk is

    v(x) · min( R(x)²/A_x , U(x)²/B_x )

with `A_x = E[(1−Y)²|x]`, `B_x = E[Y²|x]`. It is positive **iff** `v(x) > 0` **and
both rates are strictly positive**. So state dispersion controls both whether the
second degree of freedom is estimable and whether it can help at all — but
dispersion alone is not enough; concurrent activity of both rates is the second
factor. Verified on 10,000 random draws to a maximum relative error of 7.4e−13,
with boundary cases (`U=0`, `R=0`, `v_Q=0`, `P=Q`) and an independent check that
each closed-form projection really is the risk minimiser.

**Transfer decomposition [A].** Fitting the one-rate class on environment `P` and
evaluating on `Q` gives exactly

    E_Q[(m(Y) − a*_P(1−Y))²] = R²·v_Q/A_Q + R²·A_Q·(C_P/A_P − C_Q/A_Q)²

with `C = E[Y(1−Y)]` — an irreducible term plus a **projection-shift** term. On
public data, partitioning driver space by k-means fitted on training rows only,
the projection shift between a held-out storm and the remaining storms is an order
of magnitude larger than between a random split with the same event mixture, at
K = 16 and K = 32, for both one-rate branches.

**Prop 4 — reachable interval [A].** `y ↦ y(1−U−R) + U` is a contraction wherever
`U+R < 1`, so once inside, the state stays in
`[u_min/(u_min+r_max), u_max/(u_max+r_min)]` regardless of the forcing *path*.
Forcing amplitude moves the state within the walls; only the rate constants move
the walls.

**Synthetic ridge [B-synth].** Across a forcing sweep, the two rate errors are
positively correlated in 18/18 fits: +0.78 at the smallest state spread, declining
monotonically to +0.36 at the largest. From forcing 0.3 to 4.8 the trajectory RMSE
improves 1.8× while the recovery error of `U` improves 11.5× and of `R` 5.4×.
**Fit quality and rate recovery are different things, and they separate exactly
where Prop 2 says they should.**

## 3. What was measured on real data, and what it refuted

Public data: EAGLE-I county outage records (CC BY 4.0), ERA5 weather aggregated to
counties, NOAA storm events; 26 storm panels 2018–2024 across convective, winter,
wind, tropical and flood; an explicit observation mask (unobserved cells are
excluded, never imputed); 52.9–54.3% of scored targets are exactly zero at every
horizon.

**Local information geometry (D-6), zero training.** k-nearest-neighbour
neighbourhoods in a training-fitted PCA of the driver block, k ∈ {50, 200, 800},
plus an 8×8 quantile grid; cells are forecast origins; five folds.

* The theorem identities hold on the data: `λ_min(Q̂)` is within 1–2% of `v̂`.
* Local information is very small: median `v(x)` is 3e−4 to 2e−3 on convective
  panels and 1e−6 to 4e−4 on other families; `N·v` at k = 200 is 0.26 on
  convective and 0.003–0.012 elsewhere.
* The **local variance ratio `Â/B̂` is ~700 on convective and 1.6e4–6e4 on winter,
  tropical, flood and wind**. This is the correct local statement about how much
  harder `R` is to estimate than `U`.
* **A statistic that must NOT be used**: an earlier draft quoted a Kish effective
  sample size of 93:1 (or 97:1) as "the precision ratio of the two rates". That is
  wrong. Kish ESS measures how concentrated each scalar leverage weight is across
  cells; it is not the joint conditional information ratio for two functions. It
  may be reported as a leverage-concentration diagnostic under that name only.
* **Where the dispersion comes from** is only partly estimable: at k = 50 nearly
  every neighbourhood row is a different county (rows per county ≈ 1), so the
  within-county term is zero by construction. At k = 200 the cross-county share of
  `v` is 0.78 over 26 events (event-cluster 95% CI [0.75, 0.82]); at k = 800 it is
  0.44–0.70. The honest statement is that repeats of similar drivers *within* one
  county are rare in these panels, so the dispersion that identifies the two rates
  comes overwhelmingly from different counties under similar weather — a fact
  about the data's design, and it requires a common-rate-function assumption
  across counties (or sufficient county modifiers in `x`) before it licenses any
  pooling claim.
* **A refutation to respect.** Ranking event families by local information gives
  convective ≫ winter > tropical ≈ flood > wind. The empirical advantage of the
  two-rate model over a parameter-matched single signed rate orders them almost in
  reverse (tropical > convective > wind ≈ 0 > winter). **Identifiability does not
  explain the family ordering**, and the paper must not claim it does.

**A second refutation.** An earlier proposition claimed rollout error saturates at
`ε·min(h, 1/(U+R))`. That is false without a uniform positive lower bound on
`U+R`: a counterexample with total intensity → 0 gives nearly linear accumulation.
The correct statement is the time-varying product-sum bound. "Long-horizon error
is not accumulation" must not be claimed.

**A confirmatory campaign that failed.** Under true leave-one-event-out (11 storm
events, 3 seeds, parameter-matched comparator within 0.54%), the two-rate model's
advantage over the single signed rate is +4.55% at 24 h (8/11 events, randomization
p = 0.041) and +3.71% at 48 h (6/11, interval includes zero). Both fall short of
the preregistered gates. The two-rate model also does **not** beat damped
persistence on the equal-event mean at any horizon under this protocol.

## 4. The experiment that is unfinished, and the actual question

Everything above establishes when separation is possible **in principle** and how
much information the data carries **as a design property**. What is not
established is whether the *estimator we actually use* — small networks fitted by
rollout MSE — approaches that information limit, or whether its error floor is set
somewhere else entirely (optimisation, function approximation, the single-rollout
objective).

A synthetic experiment was designed for this and halted part-way. Its design holds
fixed the true rate functions, the driver paths, the noise variance, the total
sample size, the relaxation scale, and the optimiser and its budget, and
manipulates **only** the conditional state dispersion at shared drivers, by
replicating each driver path `r` times with initial states drawn from `U(0, s)` and
sweeping `s`. It has two layers: an *oracle* local least-squares estimator on the
exact fixed-design model, and the *neural* rollout estimator. A negative control
changes the rate magnitudes at fixed dispersion, so that a result driven by signal
scale rather than by dispersion is detectable.

Partial results before it was halted, not graded:

* the oracle layer's realised MSE divided by the exact Gauss–Markov / Cramér–Rao
  value sits in 0.82–1.13 across cells, i.e. the covariance formula is numerically
  exact on this design;
* at small dispersion the two rate errors are positively correlated at +0.85 to
  +0.86 — the Prop 1 ridge, now seen under a design that holds forcing, noise and
  sample size fixed;
* the neural layer's recovery error had not yet moved across the sweep when the
  run stopped.

## 5. The repository

`ShuaiWang-Castle/asymode-open-data` (**private**; ask the owner for access, or
ask them to paste specific files).

* `main` — the open dataset: 26 panels with observation masks, matched hourly
  county-aggregated ERA5 drivers, county statics, the storm-event catalogue,
  provenance and per-source licences, checksums, and a verifier that reproduces
  the archived onset audit from the published files.
* branch `cc-event-transfer-confirmation-20260903` — the working tree: model code
  (`src/asymode/`), experiments, the results ledger, preregistrations, and the
  confirmatory package under `results/event_transfer_confirmatory_20260903/`.

Files worth reading first, in order: `docs/THEORY_PLAN.md` (the propositions and
the registered checks), `src/asymode/information.py` (the identification geometry,
with 1,726 property tests in `tests/test_theory.py`),
`experiments/d6_information_geometry.py` and
`results/d6_information_geometry_{g2,g3}.json` (the real-data measurement),
`experiments/s2_crlb_tracking.py` (the halted synthetic experiment),
`experiments/cc_theory_projection.py` (Theorem 5 and the transfer decomposition),
`docs/CC_SOUNDNESS_FINDINGS.md` and
`results/event_transfer_confirmatory_20260903/17_FINAL_REPORT.md` (what was
refuted and why).

## 6. Ground rules

* This is a **statistics and experiment-design** question, not a leaderboard task.
  A negative result is a result; do not propose searching for a split, metric,
  subset or architecture that recovers a desired sign.
* The event (storm) is the statistical unit on real data. Seeds measure
  optimisation variability only and are never counted as independent events.
* No claim may rest on a statistic that was chosen after seeing the outcome.
* Every proposed diagnostic must come with its interpretation fixed in advance,
  including what would kill it.

## 7. What I want from you

1. **Is the halted synthetic design right?** Manipulating dispersion by replicating
   driver paths with spread initial states holds the drivers fixed but changes the
   *marginal* state distribution, and therefore the operating point of the rate
   functions. Is that a confound, and if so what is the cleaner manipulation?
2. **What is the right endpoint for "the estimator reaches the limit"?** The oracle
   layer has a closed-form CRLB. The neural estimator has no such bound. Is
   comparing neural rate-recovery MSE against `σ²·tr(Q⁻¹)/N` on matched cells
   meaningful, or does function approximation make that comparison vacuous? If it
   is vacuous, what would you compare instead?
3. **Rollout versus one-step.** The theory is about the one-step conditional
   transition; training is by rollout MSE. Does that gap explain why the estimator
   might not track the information bound, and how would you *measure* that rather
   than assert it? (An auxiliary teacher-forced one-step loss with a small fixed
   grid is available and has not been run.)
4. **The real-data counterpart.** Given that within-county repeats of similar
   drivers are rare and the county decomposition is only partly estimable at small
   neighbourhoods, is there an identification-strength statistic on real data that
   would be *decision-relevant* — i.e. one that predicts, per event or per driver
   region, where the two-rate model actually beats the one-rate projection? Note
   that Theorem 5 says the predictor should be `v·min(R²/A, U²/B)`, which needs
   cross-fitted rate estimates; that has not been computed.
5. **What should be dropped.** Given the two refutations in §3, which parts of the
   identifiability story are still worth a paper section, and which should be
   demoted to an appendix lemma or cut?

Answer 1–5 directly. Where you disagree with a design above, say so and give the
replacement. Where you think a question is unanswerable as posed, say that instead
of answering a different one.
