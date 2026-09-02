# Theory plan: what to prove, why it is worth proving, and how it connects to the data

Written 2026-09-02. Status of each statement is marked: [A] already proved and
verified in this repo; [P] provable with elementary arguments, not yet written;
[H] hypothesis the theory generates, to be tested with a fixed interpretation.

## 0. What the theory is for

The paper's spine is one question the application literature disagrees on:
outage and restoration are concurrent processes (Carrington–Dobson–Wang 2021
show any resilience curve decomposes into the two), and the modelling literature
says separating them "is not practical" (Zhu et al. 2021). Our theory should
answer **when the two rate functions can be separated from data at all, and
what governs how precisely** — stated exactly, proved in a few lines, and then
used to explain three things we have already measured: the information
asymmetry between U and R (Kish ESS 93:1), the zero-state blind spot, and the
family ordering of the two-rate advantage (H-E). The mathematics is elementary
and partly classical (varying-coefficient regression); the contribution is the
specialisation to a dynamical model with exogenous forcing and the consequences
for what data can teach it. That has to be said plainly in the paper.

## 1. Setting

State y_t ∈ [0,1] (share of customers out), drivers x_t ∈ X ⊂ R^d, one step
    y_{t+1} = clip( y_t + U(x_t)(1 − y_t) − R(x_t) y_t , 0, 1 ),
U, R : X → [0, cap] unknown. Fully observed: (x_t, y_t). Unclipped cells give
    Δ_t := y_{t+1} − y_t = (1 − y_t) U(x_t) − y_t R(x_t)          (1)
which is a **varying-coefficient regression** with known basis (1 − y, −y) and
unknown coefficient functions of x. Everything below follows from that view.
Continuous-time analogue dy/dt = U(x)(1−y) − R(x) y has the same structure with
(y, ẏ); the discrete statements transfer verbatim.

## 2. Statements

**Prop 1 — identifiability iff conditional state dispersion. [P]**
Fix x. The conditional mean m(x, y) = (1−y)U(x) − yR(x) is affine in y with
intercept U(x) and slope −(U(x)+R(x)). If the conditional law of y_t given
x_t = x has at least two support points, (U(x), R(x)) is determined; if it is a
point mass at y*, only one linear combination is, and every (U′, R′) on the line
(1−y*)U′ − y*R′ = m(x, y*) fits. Hence (U, R) is identifiable on the driver
support iff, for a.e. x, the state at which x is observed is not degenerate.
The two-step determinant already in the ledger [A] (det = y₁ − y₂) is the n = 2
case.

**Prop 2 — information equals conditional state variance. [P]**
With n unclipped observations at driver value x, states y₁..y_n, noise variance
σ², the design A has rows (1 − y_i, −y_i) and, by Cauchy–Binet,
    det(AᵀA) = Σ_{i<j} (y_i − y_j)² = n² · Var_emp(y | x).
Gauss–Markov then gives the variance of the best linear unbiased rate estimates:
    Var(r̂) ≥ σ² Σ_i (1−y_i)² / det(AᵀA),   Var(û) ≥ σ² Σ_i y_i² / det(AᵀA),
so **Var(r̂)/Var(û) = Σ(1−y_i)² / Σ y_i²**. Under Gaussian noise this is the
Cramér–Rao bound. Conditioning of the two-row case is 1/|y₁ − y₂|.

**Cor 2a — the zero-state blind spot. [P]** If every observation at x has
y = 0, U(x) is identified (Δ = U) and R(x) carries no information; at y = 1 the
roles swap. On panels where most scored cells are near zero, the precision ratio
Σ(1−y)²/Σy² is large: R is learned from a small effective sample. This is the
theoretical counterpart of the measured Kish ESS ratio U:R = 93:1, which the
ledger records as a property of the panel and mask alone.

**Cor 2b — where the dispersion comes from: pooling. [P]** By the law of total
variance, Var(y | x) = E[Var(y | x, county)] + Var(E[y | x, county]). Within one
county the same weather recurs rarely at different states; across counties the
same storm hour hits many counties at different states. If the second term
dominates, the separability of U and R is supplied by the cross-section, and a
single county's history cannot identify R. (Testable; see D-6.)

**Prop 3 — functional identifiability under smoothness. [P]** If U, R are
L-Lipschitz and the observations within a δ-ball of x have state gap
|y₁ − y₂| ≥ s, the two-row solve has error ≤ √2 (√2 Lδ + ‖ε‖)/s. This is the
statement the neural parametrisation actually relies on: the *functions* are
identified up to a bias-variance trade-off governed by L, δ and the state gap;
the network *weights* never are (permutation symmetry), and the paper should
say so.

**Prop 4 — reachable interval. [A]** Under driver ranges [u_min,u_max],
[r_min,r_max], the state stays in [u_min/(u_min+r_max), u_max/(u_max+r_min)]
once inside, independent of the forcing path. Consequence: the attainable state
dispersion, hence the attainable information about R, is bounded by the driver
ranges — a link between Prop 2 and the physics of the event.

**Prop 5 — rollout stability. [P]** The one-step map y ↦ U + (1−U−R) y has
slope λ = 1 − U − R ∈ [1 − cap_u − cap_r, 1). A per-step misspecification ε
propagates as e_h ≤ ε min(h, 1/(U+R)): linear at worst, saturating, never
compounding. So the long-horizon loss to gradient boosting (EXP07/EXP10) is not
error accumulation; it is representational — a scalar-state, current-driver map
cannot express restoration that depends on time since failure (crew logistics),
which an age-structured two-compartment model can. That extension is
theory-implied, not yet registered.

## 3. Relation to the literature (to be stated, not hidden)

* Varying-coefficient models (Hastie & Tibshirani 1993; Fan & Zhang) — Prop 1–2
  are the specialisation to basis (1−y, −y). Cite; do not claim the mathematics.
* Loman, Browning, Baker 2026 (functional identifiability of ODE models) — they
  observe state variables without exogenous forcing and prove additive unknown
  functions are non-identifiable. Our two unknowns multiply *different known
  functions of the state*, and exogenous forcing supplies repeated driver values
  at different states: identifiability comes from exactly the ingredient their
  setting lacks. Position ours as the forced-input analogue.
* Zhu et al. 2021 ("not practical") vs Carrington–Dobson–Wang 2021 ("always
  decomposable"): Prop 1 is the condition that adjudicates.

## 4. Tests the theory generates — registered here, interpretation fixed

**D-6 information geometry of the panels** (diagnostic; no training). Bin the
standardised driver block (k-NN or coarse bins on the leading components) and
compute per bin: n, Var_emp(y | bin), Σ(1−y)²/Σy² (precision ratio), and the
within-county vs cross-county split of Var(y | bin). Aggregate per family
(g3-all-26, families as in EXP06).
* Fixed interpretation. (i) If the cross-county term carries more than half the
  conditional state variance in every family, Cor 2b is supported and the paper
  may say pooling is the source of separability; if not, that sentence is not
  written. (ii) [H] If per-family R-information (median over bins of
  n·Var(y|bin), normalised by cell count) orders tropical > {wind, convective}
  > winter, the identifiability account is a candidate explanation of the H-E
  ordering and goes into the paper as a *consistency* statement; if it does not
  order, the theory does not explain H-E and the paper says the mechanism is
  phase separation alone. Either outcome is reportable.

**S-2 CRLB tracking on synthetic data** (cheap training). With the existing
generator (κ = 1.5), manipulate the state dispersion at shared drivers (number
of pooled counties, or initial-state spread) at fixed sample size, fit the
two-rate model, and plot the recovery error of R against the Gauss–Markov
bound of Prop 2. Fixed interpretation: if the empirical error tracks the bound
within a constant factor across the sweep, Prop 2 is the predictive statement
for the paper; if the error floor is set elsewhere (optimisation, init), the
theory is descriptive only and the paper's claim is limited to Prop 1.

**Onset check** [A already]: U identified from onset cells alone — the
onset-from-zero dominance result is the empirical face of Cor 2a and is already
graded.

## 5. Sequence and ownership

0. Theory note with full proofs of Prop 1–3, 5 and the pooling decomposition —
   this session, one day. Nothing here needs the framing decision.
1. D-6 on g3 — this session's data-mining lane, half a day of compute, after
   the PI's go-ahead (it is a diagnostic, but it is registered and graded like
   everything else).
2. S-2 — experiment lane if one exists, otherwise this session; one day.
3. Paper theory section (one page): setting, Prop 1, Prop 2 with Cor 2a/2b,
   one sentence each for Prop 3–5, the literature positioning of §3, and the
   D-6/S-2 checks as the evidence. Writing session, after 0–2.
4. Age-structured extension (Prop 5's implication) — only if the framing
   decision is path 3 (mechanism-led) and there is time before 2026-10-06.

Deadline arithmetic: abstract 2026-09-29, paper 2026-10-06. Theory must be
frozen by 2026-09-15 for the writing session to build on it.
