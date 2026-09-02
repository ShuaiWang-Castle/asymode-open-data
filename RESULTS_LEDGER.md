# Results ledger

Every number that may enter a paper lives here, with the file it came from, the
protocol it passed, and a grade. **A number without an archive path does not get
quoted, by me or by the writing session.**

## Grades

* **[A]** provable, or directly verifiable by inspection
* **[B]** full protocol passed — county-held-out folds, >= 3 seeds, sign gate — may enter the paper
* **[B-synth]** >= 3 seeds and sign gate passed **on synthetic data with known ground
  truth**. There are no county folds on synthetic data, so this cannot be [B].
  It may enter the paper *labelled as a synthetic study*, never as evidence about
  real outages. *(New tier — flagged to the PI for approval.)*
* **[C]** preliminary — internal discussion only, must not enter the paper
* **[degenerate]** — an arm whose predictions are exactly zero on more than 90%
  of scored cells under the full protocol (`frac_pred_zero > 0.9`, per arm per
  fold in the result JSON). Applied to every arm by the same rule, fixed before
  the run. Reported as a mechanism fact; **excluded from any quantitative
  comparison of what structure buys.** Beating a collapsed arm is not evidence.
* **[void]** — a gate arm that closed (`frac_gate_closed` above threshold), or a
  test whose kill condition was applied to a regime the design could not reach.
  Not "worse"; not informative.

Status as of 2026-09-01: the model has now been fitted to public observations
under the full protocol (EXP05). Its **pre-registered criteria failed 3 of 4**, and
the surviving evidence is a long-horizon result that has not yet been
pre-registered. Nothing from EXP05 may enter a paper until it is rerun under
criteria written for the paired design.

---

## EXP01 — identifiability of the two rates

Source: `results/exp01_identifiability.json` · script `experiments/exp01_identifiability.py`
Protocol: 6 forcing levels x 3 seeds = 18 fits; n=384 trajectories, T=96, 250 epochs max.
Ground truth known in closed form; recovery scored on 4,000 driver points drawn
from the observed driver distribution.

### The determinant argument — **[A]**

At one step the data give `dy = u(x)(1-y) - r(x)y`: one equation, two unknowns.
Two observations sharing driver `x` at states `y1 != y2` form a 2x2 system with
determinant `(y1 - y2)`. The interruption/restoration split is identified **only**
through variation of the state under comparable drivers, and conditioning
degrades as `|y1 - y2| -> 0`. Provable; no experiment required.

### Recovery vs state spread — **[B-synth]**

| forcing | state spread | traj RMSE | nRMSE(u) | nRMSE(r) | err corr |
|---|---|---|---|---|---|
| 0.15 | 0.053 | 0.0094 ± 0.0066 | 0.977 ± 0.608 | 0.805 ± 0.631 | +0.78 ± 0.05 |
| 0.30 | 0.086 | 0.0038 ± 0.0008 | 0.227 ± 0.064 | 0.296 ± 0.086 | +0.63 ± 0.06 |
| 0.60 | 0.151 | 0.0039 ± 0.0005 | 0.096 ± 0.022 | 0.198 ± 0.055 | +0.46 ± 0.07 |
| 1.20 | 0.217 | 0.0030 ± 0.0004 | 0.046 ± 0.007 | 0.107 ± 0.015 | +0.43 ± 0.05 |
| 2.40 | 0.251 | 0.0026 ± 0.0004 | 0.025 ± 0.004 | 0.082 ± 0.011 | +0.37 ± 0.04 |
| 4.80 | 0.259 | 0.0021 ± 0.0002 | 0.020 ± 0.001 | 0.055 ± 0.003 | +0.36 ± 0.04 |

### Pre-registered hypotheses, adjudicated against the kill conditions as written

**H1 — "trajectory fit stays good across the sweep" — FAILED.**
Kill condition was >3x variation in trajectory RMSE. Observed 0.0183 / 0.0019 =
**9.6x**, driven by one badly-fit seed at the weakest forcing. H1 is dead as
stated and does not enter the paper.

A weaker claim survives the same data and is **[C]** because it was not
pre-registered: from forcing 0.3 to 4.8, trajectory RMSE improves 1.8x while
nRMSE(u) improves **11.5x** and nRMSE(r) **5.4x**. Recovery degrades far faster
than fit. Needs its own pre-registration and a rerun before it may be quoted.

**H2 — "recovery is non-monotone; strong forcing pins y near 1 and loses u" — VOID, not failed.**
Best recovery is at the top of the sweep for all 3 seeds, and
`frac(y > 0.99) = 0.0000` at **every** forcing level: the sweep never entered the
regime it was built to test. The kill condition was applied to a sweep that could
not reach the hypothesis, so **the verdict carries no information and H2 must not
be reported as a negative result.**

> **Correction, superseding the diagnosis this entry first carried — [A].** The
> original note attributed the failure to pulsed forcing letting restoration pull
> the state back, and prescribed sustained forcing plus a larger `cap_u`/`cap_r`
> ratio. The first half of that is wrong and would have produced a second failed
> run. The state's reachable band does not depend on the forcing *pattern* at all.
>
> `y <- y(1 - u - r) + u` is a contraction wherever `u + r < 1`, which holds
> throughout, so `y` approaches `u/(u + r)` from below and cannot overshoot it.
> The band is therefore closed-form in the rate constants alone:
>
>     floor   = u_min/(u_min + r_max) = 0.0742
>     ceiling = u_max/(u_max + r_min) = 0.9438
>
> Checked against the generator at `pulse_scale` 50 and 200: realised maximum
> 0.9288, identical at both, i.e. a 4x increase in forcing moves it not at all.
> **Both** regimes H2 names lie outside the band — the low end too, which the
> original diagnosis missed: `frac(y < 0.01)` never exceeds 0.004 at any forcing
> level, because at `b_u = -3.5` no non-negative hazard can drive `u` below
> `r/99`. Forcing amplitude moves the state *within* the walls; only the rate
> constants move the walls.
>
> Second design registered in `docs/PREREGISTRATION_exp01_h2.md`, implemented in
> `experiments/exp09_identifiability_state.py`, sweeping the equilibrium directly.
> Its design-validity check reaches `frac(y < 0.01) = 0.659` and
> `frac(y > 0.99) = 0.309` at the two ends. **The criteria in the EXP01 docstring
> are void for that design; nothing in this section may be quoted for H2.**

**H3 — "the two rate errors are positively correlated when badly identified" — PASSED.**
Error correlation is positive in **all 18 runs**, is +0.78 ± 0.05 at the weakest
forcing, and declines monotonically to +0.36 ± 0.04 at the strongest. Sign gate
passed. This is the ridge the determinant argument predicts: an inflated
interruption rate is paid for by an inflated restoration rate, while the
trajectory still looks fine. **[B-synth]**

---

## EXP02 — onset, and what the epidemic form costs

Source: `results/exp02_onset.json` · script `experiments/exp02_onset.py`
Protocol: 3 arms x 2 generators x 3 seeds = 18 fits. Identical drivers, loss,
capacity, and seeds across arms. Half of all trajectories start at exactly y = 0.

### The structural statement — **[A]**

An inflow proportional to `y` is identically zero at `y = 0`. A model in that
family cannot start an outage in a county that does not already have one.
Verified numerically: the pure transmission arm rolled from `y0 = 0` returns
`y_T = 0.00000` exactly. Provable, not measured.

### Cost under a neutral generator (kappa = 1.5, which **no** arm implements) — **[B-synth]**

| arm | RMSE all | RMSE onset | RMSE started | fitted seed eps |
|---|---|---|---|---|
| susceptible *(this work)* | 0.0057 ± 0.0002 | **0.0062 ± 0.0002** | 0.0052 ± 0.0002 | — |
| transmission | 0.3022 ± 0.0077 | 0.4222 ± 0.0082 | 0.0609 ± 0.0025 | — |
| transmission + learnable seed | 0.1105 ± 0.0035 | **0.1436 ± 0.0053** | 0.0608 ± 0.0025 | 0.0078 ± 0.0001 |

Well-specified reference (kappa = 1.0) gives the same ordering with a wider gap:
0.0027 / 0.4851 / 0.1545 on onset.

Sign gate: susceptible beats the steelmanned epidemic arm on onset in **3/3
seeds**, and the two per-seed ranges are **disjoint** (0.0060–0.0064 vs
0.1362–0.1479). Kill condition passed.

### Two things worth stating plainly

**The seed does not rescue the epidemic form, and not for the reason I predicted.**
I pre-registered that the fitted `eps` would be "pushed far above its
initialisation". It moves from 0.0010 to 0.0078 — **7.7x init, but still two
orders of magnitude below the states it must explain**, and extremely stable
across seeds (± 0.0001). The arm does not crank the seed up; it settles on a
compromise. The reading — **[C]**, since it was not pre-registered — is that a
seed large enough to ignite from zero would swamp the `y`-dependence and damage
the fit on already-started trajectories, so the optimiser refuses. That tension
is the interesting result and deserves its own experiment.

**The epidemic form is not only bad at onset.** On trajectories that *already*
have outages it still scores 0.061 against 0.0052, roughly **12x worse**. The
onset gap is larger (23x) but the failure is not confined to onset. Unregistered,
therefore **[C]**.

### What this does NOT establish

That onset-from-zero is common in real county outage data. That is an empirical
claim about the world and cannot be settled on synthetic trajectories. It is the
first thing to test once the panel exists.

---

## EXP03 — does onset-from-zero actually happen? Public observations say yes

Source: `results/panel_onset_audit.json` · scripts `scripts/build_panel.py`,
`src/asymode/panel.py` · panels archived at `data/interim/panel_*.npz`

Twelve storm days with the largest county footprints in 2021, 2022 and 2024, each
windowed two days before to five days after, counties from the public storm
catalog gated at >= 70% state coverage. For every county the storm later
interrupts (peak `y >= 0.01`), what was its *typical* state beforehand?

| storm day | counties | interrupted | median `y_pre` = 0 | < 1e-4 | < 1e-3 |
|---|---|---|---|---|---|
| 2021-05-04 | 344 | 263 | 82.9% | 94.3% | 99.2% |
| 2021-06-21 | 256 | 197 | 70.6% | 87.8% | 98.0% |
| 2021-08-11 | 253 | 198 | 56.6% | 84.3% | 97.0% |
| 2021-12-11 | 256 | 188 | 91.0% | 99.5% | 99.5% |
| 2022-04-13 | 207 | 138 | 77.5% | 86.2% | 97.1% |
| 2022-06-08 | 183 | 115 | 73.9% | 89.6% | 98.3% |
| 2022-06-17 | 356 | 304 | 66.2% | 83.4% | 93.7% |
| 2022-07-23 | 186 | 139 | 67.6% | 87.8% | 100.0% |
| 2024-01-09 | 240 | 205 | 79.0% | 96.1% | 99.0% |
| 2024-05-08 | 236 | 176 | 75.6% | 94.3% | 100.0% |
| 2024-05-26 | 339 | 270 | 77.0% | 86.7% | 97.4% |
| 2024-06-26 | 230 | 178 | 65.7% | 82.0% | 96.1% |
| **mean** | | | **73.6%** | **89.3%** | **97.9%** |

Denominator: `eaglei_2024_modelled` — the publisher's own per-county customer
totals, modelled from LandScan population, EIA-861 and HIFLD service territories,
and described by them as approximate.

**[A] for the headline, and it is denominator-free.** `median y_pre = 0` holds
exactly when `median customers_out = 0`, so the 73.6% column is a property of the
published records, not of any normalisation. Confirmed empirically: run under the
provisional denominator the same column read 73.7%, and under the publisher's
modelled one it reads 73.6%. The 1e-4 and 1e-3 columns do depend on the
denominator and moved by less than half a point.

**Onset is the dominant regime, not an edge case.** Across twelve independent
storm systems spanning three years, between 57% and 91% of the counties a storm
interrupts were at exactly zero beforehand. Twelve days out of twelve agree in
direction. An inflow proportional to `y` is identically zero on all of them.

Among the minority carrying a nonzero baseline the epidemic form is suppressed
rather than dead, by a factor `1/y_pre`: hundreds to tens of thousands, depending
on the storm. The rate is bounded, so it cannot be inflated to compensate. **[C]**
— denominator-dependent.

### A measurement correction worth recording

A first pass scored the pre-storm state by its *maximum* over the lead-in window
and reported that only 2.3% of counties were at zero. That criterion demands a
county not log a single outage record in two days, which almost no county
satisfies — utilities report handfuls of customers out continually. The typical
state, not the extreme, is what the dynamics see. Scored by the median the figure
is 73.6%. The maximum-based number is wrong for this question and is recorded here
only so it is not rediscovered and believed.

## EXP04 — statistical baselines under the county-held-out protocol

Source: `results/exp04_baselines.json` · script `experiments/exp04_baselines.py` ·
protocol `src/asymode/evalproto.py`

12 storm panels x 5 county-held-out folds x 3 fold seeds = 180 evaluations per
baseline. Hourly resolution, forecast origins every 6 h with >= 24 h of history,
horizons t+1 / t+6 / t+24 / t+48. Metrics computed over observed cells only;
unobserved cells are excluded, never imputed. Fold membership is a deterministic
hash of the county code, fixed before any model was fitted.

| baseline | RMSE h+1 | RMSE h+6 | RMSE h+24 | RMSE h+48 |
|---|---|---|---|---|
| all-zero | 0.0337 ± 0.0207 | 0.0341 ± 0.0218 | 0.0345 ± 0.0223 | 0.0280 ± 0.0211 |
| persistence | 0.0107 ± 0.0047 | 0.0268 ± 0.0127 | 0.0399 ± 0.0214 | 0.0403 ± 0.0248 |
| damped persistence | **0.0105** ± 0.0045 | **0.0246** ± 0.0122 | **0.0320** ± 0.0190 | **0.0270** ± 0.0198 |
| hour-of-day climatology | 0.0328 ± 0.0199 | 0.0331 ± 0.0210 | 0.0335 ± 0.0214 | 0.0273 ± 0.0203 |

12 panels x 5 folds x 3 seeds = 180 evaluations per baseline, on the publisher's
modelled denominator. The ordering is identical to the run on the provisional
denominator, which is the robustness check that matters here.

**The bar to beat is damped persistence**, and it is not a soft one: a single
fitted decay constant with no covariates at all.

**Predicting zero everywhere beats persistence at 24 and 48 hours.** That is a
direct consequence of the onset audit -- the target is dominated by counties at
exactly zero, so a constant zero is a strong RMSE baseline while persistence
carries a storm's peak forward into a recovery it cannot see. Any dynamics claim
must clear the all-zero line, and reporting it is what keeps the later comparison
honest. It also means **RMSE alone is a poor headline metric for this target**;
the paper needs a metric that does not reward predicting nothing, and choosing it
is an open decision.

The dispersion is large (± 0.021 on all-zero) because it pools eight storms of very
different severity. Per-panel reporting, or normalising by storm severity, is
needed before these numbers go in a table.

**Grade: [B]** — county-held-out folds, 3 seeds, consistent ranking across all of
them, on the publisher's own denominator. The absolute values still carry one
stated caveat: the denominator is a 2024 snapshot applied to 2021 and 2022 as
well, so county customer drift is folded into the target.

## Denominator — resolved

Source: `data/interim/eaglei_county_customers_2024.parquet` · `scripts/ingest_eaglei.py`

The 2024 release carries `total_customers` in the records: **3,059 counties**, and
it is constant within the year for 3,059 of 3,061 (99.9%). The publisher documents
the method — LandScan high-resolution population, EIA-861 utility data, HIFLD
service territories — and calls the totals approximate. Median county 16,995
customers, range 5 to 3,799,750. **[A]**, verifiable by re-running the ingest.

How wrong the provisional stand-in was, over 3,051 shared counties: median ratio
1.17, typical absolute error 23%, log-correlation 0.965, and **8.3% of counties off
by more than 2x**. Good enough to have built the pipeline on, not good enough to
have published. **[A]**

Note the release also lists a Figshare mirror, which is an anonymous HTTPS route
to the same data for anyone reproducing this without a Globus account.

## EXP05 — the model on public observations, against baselines

Source: `results/exp05_real_dynamics.json` · script `experiments/exp05_real_dynamics.py`

8 storm windows with ERA5 drivers, 16,328 pooled (county, storm, origin) samples,
1,360 counties, 12 driver channels, 48-hour rollout. 5 county-held-out folds x 3
seeds = 15 units per arm. All three arms receive identical inputs, capacity,
optimiser, seeds and initialisation rule, so only the inflow form differs.
Baselines are recomputed on exactly these samples, not carried over from EXP04.

| method | RMSE h+1 | h+6 | h+24 | h+48 |
|---|---|---|---|---|
| all-zero | 0.03847 | 0.03694 | 0.04089 | 0.03628 |
| persistence | 0.01134 | 0.02789 | 0.04705 | 0.04916 |
| damped persistence | 0.01102 | **0.02497** | 0.03826 | 0.03528 |
| **susceptible** *(this work)* | 0.01127 | 0.02533 | **0.03542** | **0.03204** |
| transmission | **0.01109** | 0.02497 | 0.03821 | 0.03528 |
| transmission + learnable seed | 0.01134 | 0.02481 | 0.03580 | 0.03259 |

### Pre-registered criteria, adjudicated as written — 3 of 4 FAIL

1. *"beats both epidemic arms at every horizon, in every unit"* — **FAILS**. At
   h+1 and h+6 the susceptible arm is level with or slightly behind both.
2. *"beats damped persistence at h+6 and beyond"* — **FAILS at h+6** (+1.4%,
   7/15). Passes at h+24 and h+48.
3. *"dies if the seeded epidemic arm matches within one standard deviation at any
   horizon"* — **DIES**. It matches within one SD at every horizon.
4. *"no arm beats all-zero at h+24 and h+48"* — **PASSES**; the susceptible arm
   beats it by 13.4% and 11.7%, 15/15.

Criterion 3 was badly specified and I am recording that rather than quietly
replacing it: it compares a paired quantity against a *marginal* standard
deviation, which is dominated by how hard each fold is rather than by the
difference between arms. The right test on this design is paired. That is a flaw
in how I wrote the criterion, not grounds for ignoring its verdict — the literal
verdict stands, and the paired analysis below is **unregistered** and needs its own
pre-registration before it may be quoted as confirmatory.

### Paired analysis — [C], unregistered

Differences within each (fold, seed) unit, so fold difficulty cancels.

| comparison | h+1 | h+6 | h+24 | h+48 |
|---|---|---|---|---|
| vs transmission | +1.7% (3/15) | +1.4% (5/15) | **−7.3% (15/15, t=−20.0)** | **−9.2% (15/15, t=−10.9)** |
| vs seeded transmission | −0.6% (9/15) | +2.1% (3/15) | −1.1% (11/15, t=−3.0) | −1.7% (10/15, t=−2.9) |
| vs damped persistence | +2.3% (3/15) | +1.4% (7/15) | **−7.4% (15/15, t=−9.1)** | **−9.2% (15/15, t=−9.5)** |
| vs all-zero | −70.7% (15/15) | −31.4% (15/15) | −13.4% (15/15) | −11.7% (15/15) |

Read plainly:

* **At 24 and 48 hours the dynamical form matters**, and by roughly the same
  margin against the epidemic form and against the best statistical baseline:
  7–9%, every fold, every seed.
* **At 1 and 6 hours it does not.** Nothing here beats damped persistence at short
  range, and the susceptible arm is marginally worse. A paper that claims a
  general accuracy win would be overclaiming; the win is at long horizons.
* **The steelmanned epidemic arm nearly matches it** — 1–2%, and only 10–11 folds
  out of 15.

### Why the steelman closes the gap — the actual finding

On synthetic data the seed left a 23x gap on onset. Here it closes to 1–2%. The
reason is visible in the fitted parameter. Measured on **exactly the population
the model was scored on** — hourly forecast-target cells over the eight panels
EXP05 actually used, 774,142 cells:

* fitted `eps` = **0.00721 ± 0.00088**, 7.2x its initialisation
* mean scored state `y` = **0.00852**, so `eps` is **0.85x the mean state** —
  comparable to it, not larger
* **46.4%** of scored cells sit at `y = 0` exactly
* on **76.0%** of scored cells, `eps` supplies more than 90% of `(y + eps)`;
  on 87.4% it supplies more than half

So on three quarters of the cells the arm is scored on, its inflow
`u (y + eps)(1 - y)` has degenerated to `u · eps · (1 - y)`, a constant multiple
of `(1 - y)`. **The epidemic form reaches parity only by inflating its seed until
its inflow stops depending on the state.** The arm that cannot do that — pure
transmission — loses by 7–9% at long horizons on every fold.

Needs its own pre-registration and a rerun before it is quoted. **[C]**

#### Correction to an earlier version of this entry

An earlier draft of this section reported 57.4% exact zeros, an `eps` *larger*
than the typical state, and 90%-dominance on 83.1% of observations. Those figures
were computed over **15-minute cells across the whole panel window**, which is not
what the model sees: the model is scored on hourly cells after the forecast
origin. Hourly averaging removes 4.3 points of exact zeros (an hour containing any
non-zero sub-step is non-zero), and dropping the quiet pre-storm lead-in removes
about 6.7 more. On the correct population the direction of the finding holds but
it is weaker, and the claim that the seed exceeds the typical state was simply
wrong — it is 0.85x. Recorded rather than silently replaced, because the wrong
version had already been reported.

### Initialisation, recorded because it nearly produced a false negative

The first real-data run had the susceptible arm losing catastrophically (h+48 RMSE
0.13 against 0.06). Cause: a rate network initialised at `cap/2` starts at 0.125
per hour against a base rate near 1e-4, and the susceptible arm saturates the state
before it can learn its way down, while the epidemic arm is shielded because its
inflow is multiplied by `y ≈ 0.01`. The fix in `calibrate_init` sets each arm's
initial rates so its *initial flows* match the observed mean one-step rise and
fall — one rule, applied identically, calibrated on training folds only. Giving
every arm the same initial *rate* instead would favour whichever arm's multiplier
happens to be near one, which is an artefact of parameterisation.

## EXP03b — onset across event types, not just convective storms

Source: `results/panel_onset_stratified.json` · `data/interim/event_days_stratified.parquet`

The first onset audit used eight storm days, and every one of them was convective
(thunderstorm wind dominant in all eight). That is the fastest-onset,
fastest-recovery category in the record, and sampling only it is a selection
choice that was never justified. Fourteen further panels, stratified by dominant
event family and spanning 2018-2024:

| family | storm day | counties | interrupted | median `y_pre` = 0 |
|---|---|---|---|---|
| winter | 2018-01-16 | 355 | 97 | 92.8% |
| winter | 2019-02-20 | 474 | 239 | 91.0% |
| winter | 2022-01-16 | 556 | 273 | 86.8% |
| winter | 2022-03-12 | 536 | 262 | 80.2% |
| winter | 2021-02-15 *(Uri)* | 810 | 405 | 74.9% |
| winter | 2024-01-12 | 578 | 322 | 62.1% |
| wind | 2019-02-24 | 688 | 420 | 91.2% |
| wind | 2019-11-27 | 480 | 272 | 86.0% |
| wind | 2021-12-15 | 361 | 236 | 85.6% |
| wind | 2024-09-27 *(Helene)* | 445 | 405 | 64.7% |
| tropical | 2020-10-29 *(Zeta)* | 240 | 208 | 85.6% |
| tropical | 2018-10-11 *(Michael)* | 235 | 198 | 81.3% |
| tropical | 2020-08-04 *(Isaias)* | 213 | 191 | 58.1% |
| flood | 2020-02-06 | 326 | 183 | 86.9% |
| **mean** | | | | **80.5%** |

**The onset finding is stronger away from convective storms**, not weaker: 80.5%
against 73.6% on the convective set, across 14 more storms, four event families
and six years. Twenty-two storm days now agree in direction without exception.
**[A]** — denominator-free, same argument as EXP03.

## EXP08 — the input and capacity axes, one at a time

Source: `results/exp08_architecture.json` · script `experiments/exp08_architecture.py`
Protocol: 9 arms x 5 county-held-out folds x 3 seeds = 135 fits, horizon 48 h,
stride 12, 24,688 pooled samples over 12 panels, 1,672 counties.
Panel set `1c2bc7bfdfa6` (generation `g1-convective-2021-2024`); driver build
carried 10 raw meteorological channels plus the diurnal clock.
**Superseded by the rerun on `g2-convective-11` / channels `dec964873cb2`.**

> **Scope caveat, applies to every number in this section.** The 12 panels are
> convective-season events in 2021–2022 and 2024, and the driver block was raw
> meteorology only — no hazard composites, no pre-origin outage history, no
> neighbouring-county aggregates, no county statics. The drivers have since been
> rebuilt with two extra channels, so **this run's inputs no longer exist on
> disk** and it cannot be regenerated. Everything here is superseded by the rerun
> on the final panel and channel set. Graded accordingly.

### Capacity, H-D — **the registered criterion is not met**

Deltas are paired within each (seed, fold) against the control, which is the
susceptible arm at full capacity on all channels. Negative favours the variant.

| arm | phi_U / phi_R params | h+1 | h+6 | h+24 | h+48 | folds better, h+48 |
|---|---|---|---|---|---|---|
| `cap_r_glm` restoration -> GLM | 1505 / 13 | +0.4% | +0.4% | +1.0% | +0.7% | 3/15 |
| `cap_u_glm` interruption -> GLM | 13 / 1505 | +1.4% | +0.6% | +1.6% | +1.8% | 3/15 |
| `cap_both_glm` both -> GLM | 13 / 13 | −0.1% | +0.6% | +2.9% | +3.2% | 0/15 |

The registered condition was that the reduced-capacity **restoration** rate be no
worse than the full-capacity one, and that equal performance confirms it while
worse performance kills it. `cap_r_glm` is worse at every horizon, on 12 of 15
folds at h+48. **H-D fails as registered — [C].**

Two things survive the failure and are worth keeping.

* **Capacity is doing real work — [B].** Removing the hidden layer from *both*
  rates costs 2.9% at h+24 and 3.2% at h+48, and loses on **15 of 15** folds at
  both horizons. The model is not over-parameterised, and a fully-GLM version of
  it is a worse model.
* **The two capacities contribute close to independently — [C].** At h+24 the
  single-sided costs sum to 2.6% against 2.9% measured; at h+48, 2.5% against
  3.2%. Near-additive, so the hidden layers are not substituting for each other.

The mirror arm is what makes the failure readable, and is why it was run. Shrinking
the interruption rate costs about twice what shrinking the restoration rate costs
(1.6% vs 1.0% at h+24, 1.8% vs 0.7% at h+48). That ordering is the direction H-D
predicts. **It is a magnitude difference, not a sign flip, and this project does
not accept magnitude differences as evidence of asymmetry** — that is the standard
the pre-registration sets for H-A1, H-A2 and H-B, and applying a weaker one here
because the result is congenial would be exactly the failure the pre-registration
exists to prevent. Reportable as "restoration tolerates capacity reduction better
than interruption does, on this panel set"; **not** reportable as H-D confirmed.

### Ambient meteorology, H-A3 — **null, and the control failed to be a control — [B]**

| removal | h+1 | h+6 | h+24 | h+48 |
|---|---|---|---|---|
| from restoration | −0.5% | −0.3% | +0.6% | +0.7% |
| from interruption | +0.2% | −0.0% | +0.2% | +0.1% |
| from both | −0.2% | −0.0% | +0.6% | +0.4% |

Every delta is under 0.8% and win counts sit near 8/15 throughout. Removing
ambient meteorology from either rate, or from both at once, does not measurably
change anything.

H-A3 was registered as a **negative** case: ambient fields should help *both*
rates, demonstrating that asymmetry is selective rather than universal. The
registration anticipated two outcomes — symmetric help, which confirms it, and
asymmetry, which weakens H-A. The observed outcome is a third one it did not
enumerate: the family carries no signal at all on this panel set.

**A family with no signal is trivially symmetric, so it cannot serve as the
control H-A3 was written to be.** The negative case has to be re-run on a family
that demonstrably helps at least one rate, or the selectivity claim has no
control behind it. This is a hole in the pre-registration, not a result.

Scope matters here: ambient is `cloud, pressure, rh, t2m_c` after soil moisture
was reassigned to the hazard family, and these are convective warm-season events.
Temperature and humidity are more plausibly load-bearing in winter and tropical
events, none of which is in this panel set.

### Gate machinery, H-C pilot — **[C], machine check only**

Not the registered test: H-C predicts the gate wants county identity and hazard
composites and not raw weather, and none of those families existed in this driver
build. Two endpoint widths were run to exercise the mechanism.

| arm | gate width | mean gate | frac closed | frac open | void |
|---|---|---|---|---|---|
| `gate_clock` | 2 | 0.510 | 0.00000 | 0.00000 | 0/15 |
| `gate_all` | 12 | 0.512 | 0.00006 | 0.00095 | 0/15 |

The gate trains without collapsing, which was the thing to establish: the
composition has a known absorbing state at `g = 0`, since the pulse network's
gradient is proportional to the gate and the gate's to the pulse. Adding the gate
changes RMSE by at most 0.3% in either direction.

**The mean is not evidence of an inert gate, and reporting it alone was a
measurement gap.** The rate depends on the product `g * sigmoid(pulse_logit)`, so
the pulse bias can absorb any constant factor in the gate: the gate's *level* is
unidentified and a gate sitting at its initial 0.5 is consistent with both an
inert gate and an active one. Only the spread across inputs separates them, and
it was not recorded. `gate_sd` has been added for the registered run.

### How much of this could be initialisation — **[A]**

Every arm is calibrated so its *bias* reproduces the training-fold mean flow, but
the rest of each network is drawn independently and the architectures differ, so
the realised initial rate does not match across arms. Measured over 2,048 training
samples before any training, `init_u_mean` ranges from 6.95e-4 to 1.12e-3 — a
**1.62x spread**.

The effects above are 0.4%–3.2% of RMSE. These are different quantities and the
spread does not translate into an RMSE bound, but it does mean the arms are not
starting from a common point, and the smaller differences in this section should
not be read as structure without that being said. The 15/15 result for
`cap_both_glm` is the one least exposed to it.

### Identification leverage on each rate — **[A] derivation, numbers tied to `1c2bc7bfdfa6`**

The state equation exposes the rates only through `dy = u(1-y) - r*y`, so a cell
carries information about `u` in proportion to `(1-y)` and about `r` in proportion
to `y`. A cell at `y = 0` carries **no** information about restoration — not
little, none — and that is a property of the dynamics, not of the units. Two
summaries per side, over the training folds:

| | leverage mass | Kish ESS |
|---|---|---|
| interruption | 923,304 | 934,811 |
| restoration | 1,737 | 9,648 |
| ratio | 531 : 1 | **97 : 1** |

**The Kish effective sample size is the quotable one.** Across three different
panel subsets it stays at 88–97:1 while the leverage-mass ratio swings between
388:1 and 656:1; a statistic that moves by a factor of two with panel composition
should not carry a claim.

**The asymmetry is not that restoration is rarely observed.** Decomposing
`lev_r = P(y>0) * E[y^2 | y>0] * n_cells` gives `P(y > 0 | observed) = 0.545` —
restoration is identifiable in more than half of all observed cells. The
asymmetry is entirely magnitude and concentration: `E[y^2 | y>0] = 3.4e-3`, and
the 9,648 effective cells are concentrated out of roughly 640,000 identifiable
ones, while the interruption side is nearly uniform (934,811 of 1,174,270).

Per-horizon leverage is flat (`lev_r/n` between 1.78e-3 and 2.06e-3 from h+1 to
h+48, no trend). **Recorded as a negative result so the story is not proposed
again: restoration evidence does not arrive late in the forecast window.**

This measures the leverage the *data* offers. It explains why a capacity
asymmetry might be warranted; it does not test H-D, which is decided by the kill
condition alone.

## EXP08 on g2 — the designated run. Review.

Source: `results/exp08_architecture.json` · manifest `g2-convective-11`
(panels `76a73ed794af`, channels `dec964873cb2`: 12 drivers incl. wind components
+ 2 clock) · 9 arms x 5 county-held-out folds x 3 seeds = 135 fits · county-held-out
inner split for early stopping · 22,768 samples, 1,566 counties.

g1 (12 panels, 10 channels, row-split early stopping) is superseded. Where the two
runs disagree, **g2 is the one that counts** — that was the stated purpose of the
manifest mechanism — and the disagreement is recorded rather than smoothed over.

Paired against `control`, negative = variant better:

| arm | h+1 | h+6 | h+24 | h+48 |
|---|---|---|---|---|
| cap_both_glm | −1.4% (11/15) | +0.3% (7/15) | **+3.7% (0/15, t=+4.7)** | **+4.8% (0/15, t=+5.8)** |
| cap_r_glm | +0.3% (7/15) | +0.8% (6/15) | +0.8% (6/15) | +1.0% (3/15, t=+2.3) |
| cap_u_glm | +2.0% (3/15) | +1.0% (3/15) | +1.6% (2/15, t=+4.3) | +1.6% (4/15, t=+3.6) |
| in_ambient_u_only | −0.2% (9/15) | +0.5% (7/15) | +0.9% (3/15) | +1.1% (2/15) |
| in_ambient_r_only | +0.3% (6/15) | +1.1% (4/15) | +1.1% (2/15) | +1.1% (4/15) |
| in_ambient_none | −0.8% (10/15) | +0.8% (5/15) | **+2.3% (1/15, t=+5.1)** | **+2.4% (2/15, t=+4.2)** |
| gate_clock | −0.2% (11/15) | +0.7% (4/15) | −0.1% (9/15) | −0.1% (10/15) |
| gate_all | +0.5% (7/15) | +1.1% (4/15) | +0.0% (8/15) | +0.1% (8/15) |

### H-D — DEAD on g2 as well. Consistent with g1.

`cap_r_glm` is worse at h+48 (+1.0%, 12/15 folds worse). The registered condition
was "no worse". Dead twice, on two sample sets. The mirror shows cutting U costs
more than cutting R (1.6% vs 1.0% at h+48) — a magnitude difference, not a sign
flip, and it does not count.

**Capacity is doing work — [B], stronger than on g1.** Both sides reduced to GLMs:
+3.7% at h+24 and +4.8% at h+48, **0/15 folds better**. The model is not
over-parameterised. This is the most initialisation-robust result in the run.

### H-A3 on g2 — the [B] confirmation is retracted; the void is NOT reinstated. [C]

Sources: `results/exp08_architecture.json` (g2, ambient arms mis-scoped) and
`results/exp08_ha3_retest.json` (same panels `76a73ed794af`, same channels
`dec964873cb2`, same 15 units; `check_comparable` exit 0; family map verified to
partition all 14 channels with `ambient = {cloud, pressure, rh, t2m_c}`).

**What was wrong with the first g2 entry.** The ablation arms are built by naming
the families to keep; the two wind components I had added were in no family and
were dropped along with ambient, so "remove ambient" removed six channels. The
retest removes exactly four. Paired against the same `control`
(positive = variant worse):

| removal | h | mis-scoped (6 ch) | clean (4 ch) |
|---|---|---|---|
| from restoration | 24 | +0.89% 12/15 t=2.20 | +0.60% 10/15 t=1.78 |
| from restoration | 48 | +1.14% 13/15 t=2.37 | **+1.12% 13/15 t=3.03** |
| from interruption | 24 | +1.05% 13/15 t=2.99 | +0.47% 10/15 t=1.03 |
| from interruption | 48 | +1.09% 11/15 t=2.49 | +0.84% 9/15 t=1.80 |
| from both | 24 | +2.34% 14/15 t=5.12 | **+1.17% 12/15 t=2.73** |
| from both | 48 | +2.38% 13/15 t=4.15 | **+1.19% 10/15 t=2.20** |

**Half of the long-horizon effect was the wind components.** The [B] grade
rested on the inflated numbers and stays retracted.

**But the void is withdrawn too.** I wrote above that "the clean test is g1, its
null stands, the void is reinstated." That inferred the retest's outcome before
it ran — the same over-reach, in the opposite direction, as adjudicating on a
superseded run. On the clean g2 ablation ambient **has signal**: both-removal
hurts at h+24 and h+48 with t = 2.7 and 2.2.

**Adjudication against the registered condition ("hurts both sides").**
Removing from the restoration side clears the fold bar (13/15, t = 3.03 at
h+48). Removing from the interruption side does not (9/15 and 10/15,
t ≤ 1.80). Same sign on both sides; additive at h+24 (1.06% vs 1.17%), not at
h+48 (1.96% vs 1.19% — the two sides use ambient redundantly). **Grade: [C].**
Not confirmed as a symmetric control; not void. H-A3' (precipitation + wind
speed) remains the registered second control.

**An unexplained discrepancy, stated as such.** The g1 run (12 panels, 10
drivers + clock) was *also* a clean four-channel ablation — its family map had
no orphans — and it was null (|Δ| < 0.8%). The explanation I accepted for the
g1→g2 reversal (contamination) is therefore wrong. Two candidates remain:
the panel set (g1 includes `2024-01-09`), or the presence of the wind components
in *both* arms changing ambient's marginal value. Neither is established.

*Registered, not run:* rerun the three ambient arms on g1's 12-panel set with
the 14-channel driver block. Null → the difference is the panel set; signal →
it is the wind components. Interpretation fixed now so the outcome cannot be
narrated after the fact.

*Side evidence:* the leverage/ESS figures were **bit-reproduced in the retest**
(Kish ESS U:R 93:1, `P(y>0 | observed) = 0.5408`), confirming they depend on the
panel and mask only, not on any arm's configuration. [A]

*Note:* the retest's config carries `source: None`; the fingerprint had not
propagated to that entry point. Digests match, so comparability is established
by the manifest; the rerun above must record the fingerprint.

### H-C pilot — machine validated, no effect at these widths. [C]

`gate_sd` = 0.160 across inputs: the gate **varies with input**, so it is active,
not inert (the level itself is unidentifiable, as recorded). `frac_gate_closed`
= 0.0002, no void. Effect on error |Δ| ≤ 1.1% at both widths (2 and 12). The
gate can be trained and does not collapse; whether width matters is untested
until the registered covariates (hazard composites, statics) are wired in.

### Initialisation residual — [A]

`init_u_mean` spans 7.00e-4 to 1.29e-3 across arms, **1.84x** (was 1.62x on g1).
Effects of about 1% sit against this and should not be read as structural. The
0/15 and 1/15 results above are the ones robust to it.

## EXP07 — learned baselines on identical information. [B] for the ordering.

Source: `results/exp07_learned_baselines.json` · manifest `g2-convective-11`
(panels `76a73ed794af`, channels `dec964873cb2`) · `check_comparable` with the
g2 two-rate `control` exits 0 · 15 shared units · reproduced with
`scripts/paired_review.py`. Positive = baseline worse than the two-rate model.

| baseline | h+1 | h+6 | h+24 | h+48 |
|---|---|---|---|---|
| trees_matched (same information set) | +25.9% 15/15 t=+7.5 | **−2.4% 2/15 t=−4.1** | **−7.3% 0/15 t=−9.8** | **−6.0% 0/15 t=−6.3** |
| trees_lookback (extra lead-in history) | +19.3% 13/15 | −4.4% 1/15 | −7.1% 0/15 | −6.9% 0/15 |
| linear_matched | +109.7% 15/15 | +17.4% 15/15 | +5.0% 13/15 | +4.2% 14/15 |
| linear_unbounded | +150.7% 15/15 | +64.5% 15/15 | +29.5% 15/15 | +34.0% 15/15 |
| linear_histonly | +115.0% 15/15 | +18.2% 15/15 | +7.5% 15/15 | +8.0% 15/15 |

**On identical inputs, histogram gradient boosting beats the two-rate dynamics
at h+6, h+24 and h+48 — 0/15 folds better for the dynamics at the two long
horizons. The dynamics win at h+1 by 20.6% (trees +25.9%, 15/15).** Every
linear baseline loses to the dynamics at every horizon.

Checks the experiment lane ran on itself, and I verified:

* The dynamics did not collapse: pred_sd 0.0249, pred_max 0.836.
* **The trees are cap-limited** — at h+24, 10/15 folds ran the full 400 rounds
  (mean 380). Their advantage is therefore *understated*.
* The trees fit one model per horizon; the dynamics fit one rollout. A real
  asymmetry favouring the trees; it cannot plausibly account for 7.9%, but it
  is not quantified. **Control registered below.**
* **The dynamics are not under-trained.** `results/exp08_convergence_check.json`
  — the first result file carrying a clean source fingerprint (`e217ea0`,
  dirty False) — refits the two-rate `control` on seed 0's five folds with the
  budget raised from 60 epochs / patience 8 to 400 / 40. It stopped naturally at
  63, 83, 81, 84 and 102 epochs, **0/5 at the cap**; val loss improved 3.4%.
  Against the 60-epoch fits: h+1 +1.9%, h+6 +0.7% (slightly worse — mild
  overfit), h+24 −0.3%, h+48 −1.0%. Net effect near zero and not even
  consistent in sign; the trees still lead at h+6 / h+24 / h+48 by 4.1% / 7.7% /
  4.8% after relaxing. **A 7.9% gap is not a training-budget artefact.** Scope:
  this probe is seed 0 only; `check_comparable` correctly refuses to treat it as
  a full-protocol peer of g2 (differ on: ['seeds']), and it is not graded as one.

**Grade: [B] for the ordering** — 5 county-held-out folds x 3 seeds, 0/15 at
the two long horizons, and three alternative explanations excluded: collapse
(no), under-training (no), a cap on the trees (yes, but it favours the trees).
The one unquantified asymmetry is the model count, controlled below.

**Reading.** The state equation buys a
lot where the state itself dominates the next value (h+1) and loses where the
covariate-to-target map dominates (long horizons) to a regressor that can fit
that map freely. It is consistent with EXP05: the dynamics beat *statistical*
baselines at long horizons; against a *learned* baseline that advantage is gone.

**A registered prediction that this result fulfils.** D-2 (registered 20:29,
run 20:32) states, from the sender's generic prior: *where the rank ceiling is
low, the MSE-optimal answer is each county's level, and a model with dynamics is
pushed toward a constant and loses to level estimators such as trees.* The
ceiling is low at h+24 and h+48 (0.34, 0.29) and moderate at h+6 (0.54); the
trees win at exactly those horizons and lose at h+1, where the ceiling is
trivially high and the state dominates. The direction of this result was on
record 44 minutes before the result existed. That does not soften it — it says
the loss is structural to the metric-and-horizon combination, not an accident —
but it means the paper can present the horizon split as *predicted*, not as
discovered after the fact.

*Correction to my own D-2 entry.* There I wrote that the dynamics winning at
long horizons against statistical baselines suggested "the bounded,
mean-reverting state equation is itself a good level estimator." EXP07 shows a
tree is a *better* one. The dynamics are a better level estimator than
persistence; not than a learned regressor. That reading is withdrawn.

**What was on record before this result landed (21:16):** per-horizon reporting
(registered 20:29–20:32, D-2) and the note from the first evening that RMSE is a
poor headline metric for this zero-dominated target. **No alternative metric was
ever registered.** A metric change now cannot be presented as pre-registered; it
would be a new registration made after seeing this result, and the paper would
have to say so.

### Control, not run — per-horizon two-rate fits. **Result-driven, not pre-registered.**

This control was proposed by the experiment lane *after* seeing EXP07 — it is a
response to the result, not a plan that preceded it. Quantifying a known
asymmetry is legitimate, and the paper must say the control was added post hoc.
The same standard applied to the metric question applies here.

Fit the two-rate model **once per scored horizon** (loss restricted to that
horizon; four models, same inputs, same folds, same seeds), matching the trees'
model count. Interpretation fixed now: if the per-horizon dynamics close most of
the gap, the loss was the single-rollout constraint; if they do not, the
structure itself loses on covariate-driven horizons. Runs after the convergence
check and the decisive H-A3 rerun.

### EXP10 — per-horizon two-rate fits: the constraint explains h+6, not h+24/48. [B]

Source: `results/exp10_per_horizon.json` · fingerprint `ef4f0cf`, dirty False ·
three-way comparable with EXP07 and EXP08 (`check_comparable` exit 0 for both) ·
60 fits (4 horizons x 15 units), **0/60 at the epoch cap**, pred_sd 0.029, no
collapse · reproduced with `scripts/paired_review.py` (now merges per-horizon rows
by unit; regression-checked against the H-A3 retest table).

The same two-rate model, trained once per scored horizon with the loss taken only
there — everything else identical to the EXP08 control. Paired, negative = the
per-horizon fit is better:

| | vs single-rollout control | vs trees (positive = trees better) |
|---|---|---|
| h+1 | −1.1% (13/15 better) | −21.5%, 0/15 for trees — dynamics win |
| h+6 | **−2.4% (13/15, t=−4.7)** | **−0.01%, 7/15 — dead even** |
| h+24 | −0.15% (8/15 better, t=−0.4) | **+7.7%, 15/15, t=+9.4** |
| h+48 | +0.5% (7/15 better, t=+0.9) | **+6.9%, 15/15, t=+7.7** |

Against the interpretation fixed before the run:

* **h+6 closes completely.** Releasing the single-rollout constraint is worth
  2.4% there — exactly the trees' whole h+6 advantage. At six hours the model
  pays for coherence and nothing else.
* **h+24 and h+48 do not close.** 8/15 and 7/15 — a coin flip — and the gap to
  the trees stays at 7.7% / 6.9%, 15/15. **The long-horizon gap is structural to
  the two-rate form on this target, not a cost of producing one trajectory.**

**Grade [B]** for the null at long horizons: 15 county-held-out units, the
registered kill condition ("does not close") met, three-way comparable, no
collapse, no cap. Post-hoc control, labelled as such.

*Consequence for the paper.* The path "we lose per point because we produce a
coherent trajectory" is closed at h+24/48 by this experiment, and **the premise
behind it is closed by D-4 on the real export**: across the four scored horizons
the two-rate rollout itself shows excess sign changes in 23.6% of samples, so
"we produce a coherent trajectory" is not a property the paper can claim at
this resolution. What remains is a comparison of *the trees'* coherence against
that 23.6% — D-4's re-registered measurement, with its interpretation fixed to
allow "the trees are no worse, and the framing is unavailable". The long-horizon
loss itself is the D-2 prediction fulfilled and is reported as such.

*Recorded because it recurs:* this is the third mechanistically plausible story
tonight (bounded output head; logit-state input; single-rollout constraint) whose
causal chain was correct and whose magnitude was not. Plausibility is not
evidence; a registered interpretation plus a measurement is.

## EXP06 — H-E: the two-rate advantage tracks phase separation. [B]

Source: `results/exp06_by_family.json` · manifest `g3-all-26` (`db286b4960a4`),
channels `dec964873cb2` · fingerprint `0f5d56a`, dirty False · 540 rows = 4
families x 9 arms x 15 county-held-out units (folds drawn *within* family) ·
no family skipped; flood not run (1 panel) · adjudicated against the
registration in `docs/PREREGISTRATION_phase_separation.md` as amended before
this run, comparator `net_scaled` (parameter-matched single signed rate).

Paired advantage of the two-rate model over `net_scaled`, negative = two-rate
better:

| family | fall/rise p50 | h+24 | h+48 | width-matched h+48 |
|---|---|---|---|---|
| **tropical** | 4.0 | **−3.9% 15/15 t=−3.9** | **−5.1% 14/15 t=−3.9** | −6.1% 14/15 |
| convective | 1.7 | −0.9% 9/15 t=−2.0 | −2.0% 12/15 t=−3.4 | −3.0% 14/15 |
| wind | 2.1 | +0.0% 8/15 t=+0.1 | −0.1% 8/15 t=−0.3 | −0.3% 11/15 |
| **winter** | 1.0 | **+2.5% 5/15 t=+2.5** | **+2.8% 5/15 t=+2.4** | +2.3% 5/15 |

**H-E1 — PASS at both horizons.** Tropical is above both wind and convective;
winter is below both. Wind and convective were left unordered by the amendment
and are not scored; they come out as convective > wind here.

**H-E2 — SURVIVES.** On winter the two-rate model does not beat the single
net-rate arm; it is *worse* by 2.5–2.8% (5/15). The kill condition (a > 3% win
with t < −3 on winter) is nowhere near. This is the load-bearing result: the
place where the mechanism says the structure should not help is the place
where it does not — and slightly hurts.

Read with the phase-separation ratio, the advantage is monotone across the
families the registration orders: 4.0 → −5.1%, 1.7 → −2.0%, 1.0 → +2.8%
(h+48). The structure pays where damage and restoration separate in time, and
costs a little where they coincide.

**exp06-H1 (susceptible vs pure epidemic on slow families) — PASS.** Tropical
−13.0% / −14.8%, wind −5.9% / −6.0%, all 15/15, t ≤ −10. **H3 (seed
dominance) — PASS on all four families**, weakest on tropical (54.7% of scored
cells), where the mean state is 0.048 and ε is 0.09x it: on tropical the seeded
arm is much less degenerate, and it still loses to the two-rate model by
7.5–7.9% at long horizons (15/15).

Damped persistence, reported as the lower bound: the two-rate model beats it on
every family at h+24/48 except at h+1, by 5–15% (13–15/15).

**Caveat that bounds the grade.** `hit_epoch_cap` is high: tropical 53/90
units, convective 47/90, winter 31/90, wind 19/90 stopped at the 60-epoch cap.
The earlier single-arm convergence probe on g2 moved results by ≤ 2% and not
consistently; the sign gates here (15/15 vs 5/15) are far from their margins.
Magnitudes may move; the ordering and the winter sign are not expected to. A
family-level convergence probe is registered below.

**Grade: [B].** 5 county-held-out folds x 3 seeds per family, both hypotheses
adjudicated as written before the run, kill condition not met, comparator
parameter-matched, no degenerate arm. Magnitudes to be re-read after the
convergence probe.

*Registered, not run — family convergence probe:* refit `susceptible` and
`net_scaled` on tropical and winter, seed 0, with the cap raised to 400 /
patience 40. Interpretation fixed: if the tropical advantage stays ≥ 3% and the
winter sign stays positive, the [B] stands with revised magnitudes; if either
flips, H-E is regraded [C] and the run repeated at the higher budget.

## EXP05 on g2 — the structural ladder with out-of-fold export. [B] per rung

Source: `results/exp05_g2_sixarm.json` + `results/oof_<arm>.npz` (9 arms) ·
manifest g2 (`panel_digest 76a73ed794af`, recomputed from the OOF panel list),
channels `dec964873cb2` · fingerprint `7b673b6`, dirty False · 135 fits, 0 at
cap · `check_comparable` vs EXP08 exit 0 · **export reconciles with the archive
on 36/36 (arm, horizon)** (D-3 audit), so the OOF predictions are the scored
predictions.

Paired against `susceptible` (positive = arm worse):

| rung / arm | h+1 | h+6 | h+24 | h+48 |
|---|---|---|---|---|
| `net` (no scaling, no concurrency) | +4.2% 14/15 | +5.3% 15/15 | +6.9% 15/15 | +7.3% 14/15 |
| `net_scaled` (param-matched; **H-E comparator**) | −1.7% 3/15 t=−3.1 | −1.3% 4/15 t=−2.6 | +0.9% 9/15 t=+2.0 | **+2.0% 12/15 t=+3.4** |
| `net_scaled_narrow` (width-matched) | −1.7% 5/15 | −1.2% 4/15 | +1.9% 12/15 | +3.1% 14/15 |
| `transmission` | −1.6% 3/15 | +0.6% 10/15 | +6.3% 15/15 | +9.5% 15/15 |
| `transmission_seed` (ε = 0.0090) | +1.4% 10/15 | +1.8% 12/15 | +0.0% 8/15 | +0.3% 6/15 |
| damped persistence | −2.4% 3/15 | +0.8% 9/15 | +5.7% 14/15 | +8.3% 15/15 |

**Degeneracy rule, applied as registered:** `net` has `frac_pred_zero` = 0.859,
**below** the 0.9 threshold, so it is *not* labelled degenerate. It is noted
that 0.859 is double the data's own share of exact zeros (0.42–0.45) and that
the arm is plainly part-collapsed; the rule is applied as written and the
number is reported beside it.

**Rung by rung.** State scaling (`net` → `net_scaled`) is worth 4–6% at every
horizon, 15/15. **Concurrency** (`net_scaled` → two rates, parameter-matched)
is worth **+2.0% at h+48 (12/15, t=3.4) and +0.9% at h+24 (9/15, not
significant), and costs 1.3–1.7% at h+1 and h+6 (3–4/15).** The two-rate
model's structural advantage over the best single signed rate is small, real
only at 48 hours, and reversed at short horizons. Width-matched gives the same
picture, slightly larger. **[B] for the h+48 concurrency rung (sign gate met);
[C] for h+24 (fails the fold bar).**

The epidemic arms behave as before: pure transmission loses 6–10% at long
horizons, 15/15; the seeded arm reaches parity by degenerating — ε = 0.0090,
0.93x the mean scored state, dominating the inflow on **76.8%** of scored cells
(consistent with the corrected figure on the earlier run).

## D-1, D-3, D-4 on the real export. [A] each, as diagnostics

**D-1 (oracle shrinkage).** Headroom ≤ 0.01 for every arm at every horizon.
For the dynamics λ* falls from 1.00 (h+1) to 0.88 (h+48): the oracle wants the
long-horizon predictions *more* compressed, not less. **The registered prior is
confirmed: shrinkage is MSE-optimal here, and the peak-weighting family has no
room.** Under the rule attached to D-1, none of those interventions is attempted.

**D-3 (level vs within-origin).** The level term is a small share of MSE for
every arm — 0.01–0.04 for the two-rate model at all horizons, 0.11–0.14 for pure
transmission, 0.13 for all-zero. **Long-horizon error is dominated by the
within-origin term for every arm**, and the differences between arms live there
too. *This corrects a sentence in the EXP07 and D-2 entries:* I wrote that a
long-horizon win or loss where the rank ceiling is low is "a level win/loss".
It is not — the ceiling being low means within-origin error is large and
largely irreducible for everyone; arms differ in how much of it they carry, not
in level. Per-origin Spearman for the two-rate model is 0.29 / 0.22 at h+24 /
h+48 against a ceiling of 0.34 / 0.29 — near the ceiling; `net_scaled` sits at
0.35 / 0.28.

**D-4 (trajectory coherence) — a result against this paper's own narrative.**
The registration took the rollout arms as the coherence floor, "coherent by
construction." Measured at the four scored horizons, they are not: the two-rate
rollout shows an excess sign change — a wiggle the truth does not have — in
**23.6%** of samples (`transmission_seed` 33.9%, `net_scaled` 12.0%); the true
floor is the monotone baselines at 0.0. Residual roughness is 0.021 for the
two-rate model against 0.019 for persistence. **Across 1, 6, 24, 48 hours the
rollout's pseudo-trajectory is not smoother than a constant.** The framing
"more accurate per point, but not a trajectory" therefore cannot be argued from
our own coherence; it could only be argued from the trees being *worse* than
this, and **the trees are unmeasured** — EXP07 was not exported. D-4's
registered interpretation is void as written (its floor was wrong); the
statistic stands and the comparison to trees is re-registered below.

*Re-registered:* export EXP07's per-horizon predictions in the same layout and
run D-4 on them. Interpretation, fixed now: the framing is available only if the
trees' S1 exceeds the two-rate rollout's 0.236 by a margin that survives
county-block resampling; if the trees are as coherent or more, the framing is
unavailable and the paper says the two-rate model's advantage is confined to
h+1 and a 2% concurrency rung at h+48.

### D-4 — trajectory coherence of per-horizon predictors. Registered before predictions exist.

If the per-horizon control closes most of the gap, the honest statement is that
the two-rate model pays for producing **one coherent trajectory** while the
trees produce four unrelated point forecasts. That is a claim about the trees'
output and must be measured, not asserted. Fixed now, on the agreed OOF layout
(which already carries every arm's predictions at h = 1, 6, 24, 48 per sample):

* For each sample, the four per-horizon predictions form a pseudo-trajectory.
  Statistic 1: the fraction of samples whose pseudo-trajectory has a sign
  change in its first difference that the **truth** at the same horizons does
  not have. Statistic 2: the mean absolute second difference across the four
  horizons, per arm. The rollout arms are coherent by construction and give the
  floor; the trees are compared to it.
* Interpretation fixed: if the trees' pseudo-trajectories are materially less
  coherent than the truth's, "more accurate per point, not a trajectory" is a
  measured property; if they are as coherent, that framing is unavailable and
  the paper does not use it.

## D-2 — rank ceiling per horizon. [A]

Source: `results/d2_rank_ceiling.json` · `experiments/d2_rank_ceiling.py` ·
manifest `g2-convective-11`, 176 forecast origins · zero training, deterministic.

For each origin and horizon, counties are ranked by an **illegal** predictor —
the truth one step ahead, `y[o+1]` — and that ranking is scored against the truth
at `y[o+h]` by Spearman ρ. No legal model can rank better than one that has seen
the future, so this is a ceiling on ranking skill.

| h | ceiling ρ, p50 | p25 | origins with ceiling < 0.3 | < 0.5 |
|---|---|---|---|---|
| 6 | 0.543 | 0.412 | 5% | 42% |
| 24 | 0.343 | 0.177 | **43%** | **75%** |
| 48 | 0.289 | 0.167 | **51%** | **95%** |

(h+1 is degenerate — the illegal predictor is the target — and is omitted.)

**At 24 and 48 hours, in three quarters to nineteen twentieths of forecast
origins, even a predictor that has already seen the future cannot rank the
counties.** Ranking is intrinsically unpredictable there. The MSE-optimal answer
in those cells is each county's *level*, not its ordering.

Two consequences, one expected and one not:

* *Expected:* model comparison must be reported per horizon, and the paper must
  say that h+24 and h+48 reward level estimation rather than ordering. The
  short-horizon null in EXP05 sits where the ceiling is moderate (h+6, 0.54) —
  there is ranking skill to be had and nothing beat damped persistence at it.
* *Not expected:* the registered prior (D-2's source) predicted that where the
  ceiling is low, a model with dynamics is pushed toward a constant and loses
  to level-estimators such as trees. **Our dynamics win exactly there** — 7–9%
  over the epidemic form and the best statistical baseline at h+24/h+48, 15/15
  folds (EXP05). A reading consistent with both: the bounded, mean-reverting
  state equation is itself a good level estimator, while persistence overshoots
  and the unbounded forms cannot shrink. **Testable:** decompose RMSE per
  horizon into a level term and a ranking term. Registered as D-2's follow-up,
  not run. *(Superseded in part by EXP07: against a learned baseline the
  dynamics lose at exactly these horizons — see the correction there.)*

Half of all targets are exactly zero at every horizon (p50 0.46–0.51), which is
the same fact seen from the target side.

## D-2 by event family, and a severity-matched control. [A]

Sources: `results/d2_rank_ceiling_by_family.json`,
`results/d2_severity_matched.json` · all 26 panels · zero training.

Per family, the rank ceiling (illegal predictor = truth at origin+1) at h+48:
tropical 0.59, wind 0.36, convective 0.29, winter 0.28. An earlier version of
this entry called this "a third independent measurement separating the families
in the same order" as the fall/rise ratio and the onset audit. **That sentence is
withdrawn.** The experiment lane pointed out that tropical events are also
larger, longer and more spatially uneven, so a higher ceiling could be nothing
but higher target variance — and that the other two measurements might track
size the same way. The control: recompute all three inside matched severity
bands (county-event peak `y` in [0.02, 0.05), [0.05, 0.15), [0.15, 1]).

| band | measure | tropical | wind | convective | winter |
|---|---|---|---|---|---|
| mid | fall/rise p50 | **2.94** | 2.00 | 1.86 | **1.00** |
| high | fall/rise p50 | **7.36** | 3.79 | 3.67 | **1.25** |
| mid | ceiling h+48 | **0.40** | 0.21 | 0.20 | 0.23 |
| high | ceiling h+48 | **0.66** | 0.42 | 0.34 | 0.40 |
| mid | onset share (pre-storm median y = 0) | 0.73 | 0.77 | 0.67 | 0.76 |
| high | onset share | 0.68 | 0.70 | 0.70 | 0.62 |

What survives matching, and what does not:

* **Phase separation is not a size effect.** The fall/rise ordering
  tropical > (wind, convective) > winter holds in every band, and winter sits at
  1.0–1.25 at every severity. This is the mechanism variable H-E rests on, and
  it is the one that survives. **But wind and convective are not separable once
  severity is matched** (2.00 vs 1.86; 3.79 vs 3.67): the cross-family
  wind > convective gap was size. H-E1's ordering is amended accordingly.
* **The rank ceiling is partly a size effect.** Tropical's ceiling stays highest
  within bands (0.40 mid, 0.66 high), so the tropical-versus-rest separation is
  real; the wind / convective / winter differences collapse (0.20–0.23 mid).
  The interpretive note in H-E — a tropical win can be a ranking win — stands on
  the matched numbers; nothing else about the ceiling ordering does.
* **The onset share does not order by family at all.** Within bands it runs
  0.62–0.78 with winter often highest. It was never evidence for phase
  separation and should not have been listed alongside the other two. The
  onset finding (EXP03) is untouched — it is a claim about *all* families, not
  an ordering among them.

The honest count is therefore **one** measurement that orders the families by
phase separation independently of severity, plus a rank ceiling that separates
tropical from the rest. Not three.

## Pre-registered but not yet run — asymmetry hypotheses

`docs/PREREGISTRATION_asymmetry.md`, written before any of the feature families
in it exist in this repository. Four hypotheses with explicit kill conditions:
directional input asymmetry with a required **sign flip**, level-versus-clearance
separation from the same source, gate input width, and capacity following
effective evidence. One family is registered as a negative case on purpose.

**Nothing from it may be quoted until it has been run here.** It is a plan, not a
result, and it is listed in this ledger only so that it cannot later be presented
as though it had been decided after seeing the data.

## A review failure worth recording

Two results were reported to me and I accepted them without asking whether they
had been screened for numerical pathology. Both turned out to be artefacts.

* A history-only linear baseline scoring worse than a constant-zero predictor. I
  called it "an informative result, not a bug" and said to report it unadjusted.
  It was a bug: three branches of the model kept a default `U(-1,1)` bias while
  one was set to the training mean, so the initial prediction sat two orders of
  magnitude above the target. Zero-initialising them moved the arm from 0.058 to
  0.0275 at h+6, better than the zero baseline throughout. The conclusion drawn
  from it -- that a baseline's skill comes almost entirely from the drivers -- is
  withdrawn.
* A comparison of bounded and unbounded output heads. I asked for the reasoning
  to be written into a docstring as a citable instance. The measurement behind it
  had never run: a string edit had silently failed to match and the bounded head
  was still in place. Measured properly, neither head dominates -- the unbounded
  one is better at h+1 and worse than predicting zero at h+48, where 38.5% of its
  predictions need clipping.

Neither reached this ledger, so no graded number was affected. The process
failure was mine: an anomalous result is a reason to ask what was checked, not a
finding to endorse. **Rule going forward: a result that contradicts a strong prior
gets a pathology screen -- initialisation, gradient flow, output range -- reported
alongside it, before it is discussed as evidence.**

## Public data acquired — **[A]** (verifiable by re-running the script)

Source: `scripts/build_event_catalog.py`, `data/interim/*.parquet`

* NOAA Storm Events, 2021 / 2022 / 2024 / 2025 bulk CSVs, 47 MB
* 146,475 county-coded event rows spanning 2021-01-01 to 2025-12-29
* 63,638 county-days carrying an outage-relevant event, over 3,267 counties
* Largest single UTC day: 369 counties across 28 states (2022-06-17)
* Census county adjacency 2023 and Gazetteer 2023

Two facts that change the data plan, both **[A]**:

1. `EPISODE_ID` never crosses a state line, capping the largest episode at 65
   counties in one state. Keying events by UTC day instead yields 260–369
   counties across 16–32 states. Event selection keys on the day.
2. Tropical cyclones are filed as **zone** records, not county records, so a
   county filter drops every hurricane. Ida, Ian and Beryl are currently
   invisible. Requires the NWS zone-to-county correlation file. **Open gap.**
