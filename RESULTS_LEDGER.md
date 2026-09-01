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

Status as of 2026-09-01: **nothing is [B] yet.** The panel now exists and public
observations have been *audited*, but no model has been fitted to them, and no
county-held-out folds have been run.

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

**H2 — "recovery is non-monotone; strong forcing pins y near 1 and loses u" — FAILED.**
Best recovery is at the top of the sweep for all 3 seeds. The reason is a design
fault, not a refutation: `frac(y > 0.99) = 0.0000` at **every** forcing level, so
the sweep never entered the saturation regime it was built to test. With
`cap_u = 0.30` and `cap_r = 0.15` under pulsed forcing, restoration always pulls
the state back before it pins. **Fix: sustained rather than pulsed forcing, and a
larger `cap_u`/`cap_r` ratio. Rerun before the claim is made at all.**

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
