# Review: DATASET_DESIGN v0 (adversarial, methods only)

Read: the draft, RESULTS §§0, 3-5, PREREG_W2, program.md, DESIGN, the last 40 log rows, and the inherited gate code
(`../open_gcrk_20260919/build_panel216.py`, `src/asymode/panel.py`). Nothing was trained. The effective-event numbers
come from sizes and outcome shares on disk (`panel_gates_w2{d,e}.csv`, `results/data_patterns/patterns.json`,
`results/w1_event_vs_zero.json`) and a Monte Carlo of the draft's allocation. No unread outcome was opened.

## 0. Verdict

1. **The draft fails its own criterion 4.** With 3 sealed episodes per regime, no per-regime claim can reach 8
   effective events. The pooled claim clears 8 easily only when events are weighted by county count (15). A pooled
   RMSE weights them by squared error, which leaves 3-11.
2. **Equal episode counts do not give regimes equal weight.** MSE weights events by squared outcome mass, so the pooled
   RMSE mostly measures tropical cyclones and big winter storms, in training and in evaluation.
3. **Warnings remove most outcome selection, not all.** The draft also brings selection back in through the
   observation gates and through controls that are weighted in the metric but not in the loss.
4. **The sealed set costs no training if the model is trained on development and evaluated once.** The real limits
   are the number of independent weather systems and the hypothesis budget, and both can be planned now.

## 1. Target, failure points, and what each split can support

As drafted, the target (one model, interactions shared across regimes) fails where:

* **Sharing cannot be tested.** A test of regime differences needs per-regime estimates, and each regime has 3.5-5
  effective development events (§3). The shared coefficients will be those of the regime with the most squared
  outcome mass.
* **Each fold trains on about 5 episodes per regime.** That is fewer than W1's six storms per fold, where no arm beat
  the all-zero forecast pooled over unseen ice storms (RESULTS §3). Where the host has no skill, "better" mostly means
  fewer false alarms.
* **The episode is not the unit of independence.** One cyclone issues blizzard, high-wind and tornado warnings on the
  same days. Typed as three episodes, they can straddle folds or the seal.

| split | supports | cannot support |
|---|---|---|
| event folds | new episodes from the same frame, pooled, at the MDEs of §3 | per-regime or future-season claims |
| county folds | new counties inside storms already seen (host residuals correlate over about 50 km, `results/F1_w2e_hrrr/H2a.json`) | new-weather claims: its two-fold keep was a false positive, and PREREG_W2 treats a county-fold-only gain as a falsifier. Discard only |
| leave-one-region-out | a description | a registered claim: about 7 episodes per region (3-5 effective), with region confounded with regime mix and with the coverage gate (NE, KS, ND, WY and MT fail it) |
| sealed 18 | one pooled, large-effect claim | anything per regime; a second claim without splitting alpha |
| *missing:* forward-time split | later seasons (coverage, HRRR versions and warning products all change) | - |

## 2. Selection on the outcome, controls, estimand

**Storm Events reports** are compiled from damage, so selecting on them selects on y. All four earlier panels were
chosen this way. Zone-based warnings, issued hours ahead from forecasts, remove most of this. Some selection remains:

(a) **Convective and flash-flood warnings are partly damage-driven.** They are issued minutes ahead and extended on
real-time reports, including upstream damage.

(b) **Warning criteria are local.** Each office sets them to local impact (snowfall thresholds differ several-fold;
wind-prone zones use higher wind criteria), so selection depends on geography x weather, the object of study.
Unweighted, the panel keeps weak southern snowfalls but not weak northern ones, and the model learns warning criteria
as vulnerability.

(c) **Origin timing differs by regime.** With the origin at the first warning, convective origins fall at damage onset
(early damage sits in the initial condition), while tropical origins fall a day or two ahead.

(d) **Storm Events is an outcome proxy even as a diagnostic.** Reading it opens a sealed episode.

**Controls.** If every county in the episode domain has a known inclusion probability π (1 if warned),
Horvitz-Thompson weights 1/π rebuild the domain population, whatever links warnings to damage. Hence:

* **Weight the loss as well as the metric.** At the same weather, warned counties have higher y, so unweighted
  training on a warned-enriched sample over-predicts across the population. That is the kind of false alarm behind
  W1's worst unseen-storm error (2018-11-13, threefold).
* **Sample near misses, not "a random 25% of the region".** Calm, distant counties add almost nothing to either SSE
  and teach nothing beyond A5. Sample damaging-level weather without a warning instead: by distance ring from the
  warned footprint and by a pre-declared hazard index (π near 1 close by, 0.05-0.1 far away). Record π per
  county-event and cap the weights.
* **Keep π high where missed events can occur.** Damage with no warning enters only through controls, at weight 1/π,
  so a small π there inflates the variance.

**Estimand.** Each regime's design-weighted MSE over all county-hours of a declared domain (warned counties plus a
buffer).

* **Primary:** the equal-weight average over regimes of MSE skill, against the all-zero forecast and against the host.
* **Secondary:** frequency-weighted, with episode weights N_h/n_h and N_h counted from the frame.
* **Descriptive only:** pooled relative RMSE, which weights regimes by squared outcome mass.

Train on the same objective (regime-normalised loss), or training undoes the equal allocation.

## 3. Allocation, size, power

**Equal vs frequency.** From 6 episodes per regime, frequency weights rest on a few convective episodes standing for
thousands. Set a per-regime floor from the claim requirement and allocate the rest by N_h^0.5 (Bankier 1988). The
draft's 60 cells hold 36 episodes, so most are empty; use balanced sampling (cube method, Deville & Tillé 2004).

**Effective events**, G_eff = (Σw)² / Σw² (Kish 1965), on the built panels:

| weight w | wind (12) | W1 (8) | W2d (12) | W2e (29) | W2e kept (23) |
|---|---|---|---|---|---|
| county-events | 10.9 | 5.9 | 9.3 | 22.7 | 18.4 |
| Σy² (all-zero SSE) | 5.0 | 4.9 | - | - | - |
| host SSE | 5.8 | 3.8 (event folds) | - | - | - |
| registered feature | - | 3.5 | 7.5 | 9.9 | 12.0 |

G_eff/n is 0.74-0.91 with county weights, 0.41-0.61 with squared-error weights and 0.34-0.62 for a single feature. On
the wind panel, 1% of county-events carry 47% of the SSE.

**Monte Carlo of the draft.** Sizes lognormal (CV 0.5, as in W2e); within-regime outcome mass lognormal (sd 0.7-1.0,
matched to W1 and the wind panel); regime mean mass equal, moderate (tropical 5x, heat 0.1x) or strong (20x, 0.05x).
Medians:

| | county | Σy² pooled (equal / moderate / strong) | Σy² per regime | regime-normalised |
|---|---|---|---|---|
| dev 36 | 29.5 | 15-20 / 9-11 / 5-6 | 3.5-5.2 | 21-31 |
| sealed 18 | 15.0 | 8-11 / 5-6 / 3 | 2.0-2.7 | 12-16 |

Every per-regime value is below 8; H2a needed 23 storms of one regime to reach 12. The regime-normalised estimand
recovers about R x (per-regime G_eff), a second reason to make it primary.

**Power for trained claims.** The per-event SD of the relative change in RMSE (σ_e) is 2.8-7% on county folds (wind)
and 8.8% (ERA5 arm) to 18-19% (HRRR arms) on event folds (W1, one seed). The minimum detectable effect (MDE), one-sided
α = 0.05, power 0.8, t on G_eff - 1 df, at σ_e = 9%:

| G_eff | 5 | 8 | 15 | 20 | 30 |
|---|---|---|---|---|---|
| MDE | 12% | 8.8% | 6.1% | 5.2% | 4.2% |

**Episodes each claim needs** (G_eff ≥ 8, or power 0.8 at σ_e = 9%):

* an alignment test with a feature spread across regimes: about 10;
* a regime-specific feature: 13-24 of that regime;
* any per-regime claim under squared-error weights: 15-20 per regime;
* a trained pooled 5% gain (G_eff ≈ 22): about 45 regime-normalised, 75-100 under pooled RMSE;
* a 3% gain (G_eff ≈ 58): about 100 even regime-normalised.

The sealed 18 can confirm only gains of about 6-12%; no trained gain here has survived replication (the best
seed-averaged one is 4.2%, with an interval across zero).
**The sealed set's realistic currency is information (alignment tests, as H2a), not small forecast gains.**

**Seeds and folds.** A two-fold screen produced a false positive that its placebo, on the same folds, missed; on W1 a
second seed moved a contrast from -14.8% to +20.1%. Rules:

* A single-seed, five-fold screen may discard a change, never keep it.
* A keep needs three seeds (seed-averaged predictions) on all five folds, with its twin.
* A confirmation uses five seeds.
* Measure the seed floor once, with an A/A run (the host against itself on disjoint seeds, event folds), and set the
  seed count from it. This amends program.md rule 3.

**Minimal design within budget.** A run takes about 40 minutes per 10-12k county-events; H2b ran 15 runs in about
3 h on three workers, so roughly 100 runs a day.

* **Regimes:** convective, synoptic wind, tropical, winter, heavy rain. Heat, and probably flood-only, become
  negative-control strata outside the headline: little mechanical damage, unstable skill against a near-zero
  reference, and large heat outages that are mostly load shedding.
* **Development:** 15 systems per regime (75), each capped at about 140 county-events by within-episode sampling
  proportional to size (PPS): highest-hazard counties with certainty, PPS on a weather hazard index for the rest,
  plus ring controls. About 10.5k county-events, one 40-minute run. Episodes buy power; counties only add cost.
* **Confirmation:** 20 systems per regime (100, inference only): per-regime G_eff about 9-12, regime-normalised
  46-60. Add a prospective tranche (all qualifying systems after registration). For tropical, 35 systems is most of
  the population, so count it from best track first.
* **Compute per candidate:** screen 5 runs, keep 30 (arm and twin x 3 seeds x 5 folds), confirm 15 (host, arm and twin
  x 5 seeds on all of development). About 50 runs, half a day. Stage 0 adds 30 runs (host with 3 seeds plus A/A) and
  tests whether 12 training systems per regime suffice.

## 4. Sealing and multiplicity

* **Used material.** Windows within ±7 days of a panel whose outcomes were used (wind, W1, W2d, W2e) cannot be
  sealed. Those panels hold most large winter storms (37), so the sealed winter stratum will skew small unless size is
  stratified. PREREG_W2 keeps H1a open for an enlarged panel containing W2e: declare whether the new winter stratum is
  that panel (its C1 alignment then stays unread) or close H1a. Finish or withdraw H2b by amendment first; a paused,
  unread registered test is a file drawer.
* **Independence.** Assign whole systems to one side, with a ±7-day buffer per county.
* **What counts as a read.** Any outcome-derived number on a sealed tranche outside a registered test (per-episode
  errors, outcome-weighted F0, Storm Events, gate counts from its records, y-weighted G_eff). Compute eligibility
  from features or the host's predicted Σŷ². Store sealed outcomes separately under a committed hash, and commit the
  draw code and seed before drawing.
* **Budget, registered now.** At most K = 4 confirmatory tests per tranche, in a fixed order or at α/K, revealing only
  the registered statistics. A spent tranche joins development and the prospective tranche takes over. Important
  claims must replicate there.
* **Reusable holdouts.** Thresholdout (Dwork et al. 2015) and the Ladder (Blum & Hardt 2015) need a holdout large
  relative to the number of queries. With tens of effective episodes their noise swamps the effects: use them at most
  for coarse model selection on a separate validation tranche.
* **Winner's curse.** The one confirmed effect came in at 0.30 of its exploratory size (0.070 against 0.234). Plan power
  at one third: 2.25 times the clusters of a one-half discount.
* **Big storms are in the news.** Numeric criteria registered in advance, and the prospective tranche, guard against
  that.

## 5. Data-quality gates and operator-initiated outages

* **G3/G4 depend on the outcome.** EAGLE-I stores no zero rows, and the in-service rule (`src/asymode/panel.py`) uses
  a centred ±7-day window that looks into the forecast window. A county quiet for a fortnight becomes "unobserved"
  and is dropped: for controls, selection on y. Define coverage from collection runs and from reporting outside the
  episode (say, the 30 days before the prefix), and compare warned and control drop rates on development.
* **Early years.** For 2015-2017 the coverage gate uses 2018 coverage. G3 then drops a median of 30% of the counties
  left after G2, against 7% in 2018-2019 and 1.3% from 2020. All six HRRR-gap exclusions in W2e were from 2015-2017. Make
  July 2018-2025 the primary frame and 2015-2018 secondary. If HRRR enters the host, align period strata with HRRR
  versions.
* **Artefact flags are computed from y**, so they stay a training mask only, and evaluation keeps every observed hour.
* **Operator-initiated outages.** Decide from the record's type, footprint and time, never its size or the EAGLE-I
  series. Sources: utility-commission PSPS post-event reports (county lists), DOE OE-417 load-shed events, NERC EEA3
  declarations, and the FERC-NERC reports on February 2021 and December 2022. Exclude by county where counties are
  listed, otherwise by state or balancing authority. Flag, but keep, pre-emptive flood or surge de-energisation and
  fire-damage outages. The committed script runs before any gate output is seen and applies to every tranche;
  sensitivity checks use development only.

## 6. Revisions

1. **Declare the §2 estimand and train on it.** *Reason:* pooled MSE is one regime's metric, and effective events
   collapse under it (§3).
2. **Make the parent weather system the unit of independence and splitting.** Type regimes per county-episode by a
   registered priority rule (tropical first, from best track); map zones and polygons to counties by customer overlap.
   *Reason:* compound systems leak across folds and the seal.
3. **Define membership from products issued before the hazard** (watches, outlooks, the first TC advisory), fix the
   origin's lead, and treat advisories alike in every regime. *Reason:* §2 (a) and (c); the draft includes winter
   advisories only.
4. **Use ring x hazard controls**, with π recorded and 1/π in both loss and metric. *Reason:* §2.
5. **Make the observation gates outcome-independent** and audit their drop rates. *Reason:* the ±7-day rule reads the
   forecast window.
6. **Primary frame July 2018-2025.** *Reason:* coverage is imputed and HRRR has gaps before then.
7. **Set a regime floor, allocate the rest by N_h^0.5 (counted first), sample with balance, and make heat and
   flood-only negative-control strata.** *Reason:* mostly empty strata; near-zero outcome mass.
8. **Use within-episode PPS (about 140 county-events) and 75 development systems.** *Reason:* episodes carry power,
   counties carry cost.
9. **Make confirmation inference-only, with 20 per regime plus a prospective tranche.** *Reason:* 3 per regime cannot
   reach 8.
10. **Register the sealing rules of §4 and settle H1a and H2b.** *Reason:* contamination; the file drawer.
11. **Register the hypothesis budget, the MDEs, and power at one third.** *Reason:* multiplicity; the H2a ratio of
    0.30.
12. **Assign split roles:** event folds (regime-stratified, system-grouped) primary; county folds discard-only;
    leave-one-region-out descriptive; add forward-time. *Reason:* the §1 table.
13. **Adopt the seed and fold protocol with an A/A run, and amend program.md rule 3.** *Reason:* the two-fold false
    positive and the seed reversal.
14. **Stage 0: require the host to beat zero in each regime.** *Reason:* on W1 it lost to zero on unseen storms.
15. **Decide operator-outage exclusions from type, footprint and time only.** *Reason:* mechanism purity without
    reading y.
16. **State the claim population:** the coverage-gated states (NE, KS, ND, WY, MT, TN and MS fail). *Reason:* whole
    regime x region cells are missing.

**References.** Bankier (1988) *Am. Stat.* 42:174. Blum & Hardt (2015) ICML. Deville & Tillé (2004) *Biometrika*
91:893. Dwork et al. (2015) *Science* 349:636. Horvitz & Thompson (1952) *JASA* 47:663. Kish (1965) *Survey Sampling*.
