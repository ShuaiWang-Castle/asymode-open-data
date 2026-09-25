# RESULTS — geo-weather line (exposure-integrated local hazards)

Every number points to a file under `results/` or `data_provenance/`; the attempt log is `RESEARCH_LOG.md`, the
design `DESIGN.md` (v1), the literature `LITERATURE.md`.

## 0. Summary of the first night (2026-09-24 23:30 to 2026-09-25 03:10 EDT; updated as the night continues)

* **Framework.** DESIGN v1: the county damage input as an exposure-weighted integral of local, weather-gated,
  fading-memory hazards with shared parameters, entering the host as a non-negative competing hazard, with two
  audits that need no training (F0: what sub-county geography can change; F1: whether that aligns with what the
  host misses, with its own power analysis).
* **Wind panel (12 events).** F0 and F1 close the sub-county geography channels (effects worth 2% of pooled RMSE
  would have been detected; none is there); the trained hazard arm agrees (+0.73%). Population weighting of the
  host inputs -0.25%; canopy as county context +1.58%, as hazard-gated inputs +1.17% (folds 1-2): no gain.
  Literature-guided target cleaning looked like a gain on the screen (-2.66% vs base, -3.40% vs its placebo) but
  is +0.01% over all five folds ([-1.81, +1.83]): a screen false positive, now a program.md rule.
* **Winter ice-storm panel W1 (8 events).** F0 finds sub-county content only where the phase physics predicts it
  (near-freezing precipitation, <= 1.9% of outage-weighted county-hours). County-grouped screens are inside the
  noise (two seeds); holding out whole storms, the hazard arm cuts the host's error by 8.5%, but on unseen ice
  storms neither beats the all-zero forecast.
* **Pre-registered test (PREREG_W2.md).** The strongest W1 feature (48-h near-freezing precipitation, elevation
  bands against cells) was registered for 12 independent ice storms (2015-2025) with frozen test code (239fd7d);
  the registered test ran and returned **not testable** (7.5 effective events, 8 required), decided from the design;
  the W2d residual alignment is unread and kept for an enlarged confirmatory panel.

Screen protocol (program.md): county-grouped outer folds 1-2 of the twelve-event wind panel (2,489 held-out
county-events), a fixed 900 training steps on all development units, seed 0, paired initialisation; pooled hourly
RMSE of the open-loop rollouts; intervals are cluster-bootstrap 95% (counties; event x state), 2,000 draws.
Relative change = RMSE(arm) / RMSE(base) - 1. The screen resolves about +-3%: a keep needs > 1% with the county
interval below zero and a win over its placebo.

## 1. F0: what the sub-grid structure can act on (no training)

`results/F0/f0_audit.{json,md}`. Exposure-integrated hazard features (DESIGN v1 section 2) on the recoverability
ladder; a difference is material when it exceeds 10% of the feature's 99th percentile.

| contrast | what it isolates | largest share of county-hours (outage-weighted) | features >= 1% |
|---|---|---|---|
| quad - pop | elevation bands inside cells (lapse-rate downscaling) | 0.41% (0.93%) | 0 |
| pooled - mean | the static band distribution under county weather | 0.09% (0.32%) | 0 |
| quad - pooled | co-location of bands with their own cell's weather | 0.17% (0.63%) | 0 |
| pop - area | exposure weighting (population instead of area) | 2.83% (12.17%) | 18 |

On this panel the sub-grid structure has almost nothing to act on: within-county elevation offsets are small
(population-weighted sd of dz: median 16 m, 90th percentile 92 m; `data_provenance/nodes_cs.json`) and the
panel's events are wind and convective storms, where elevation matters only through the rain-snow phase. By the
kill rule of DESIGN section 3 the quadrature is dropped for this panel. Exposure weighting is material, mostly
through the modulators: people live under less canopy than the county's area average (gust x canopy: 12% of the
outage-weighted county-hours differ materially).

**F1: residual structure** (`results/F1/F1_cd.md`, `audit_f1.py`, by the formal contributor). Cluster-robust score
tests of each ladder contrast against the W+Cin held-out residuals, max-T over the 160 features and Bonferroni
over contrasts, with the false-positive rate calibrated on synthetic residual fields: nothing passes (the
strongest, pop - area gust_x10 x canopy, has a final p of 0.145). The power analysis propagates each regressor
through the base's own damage and recovery rates: any of these effects worth 2% of pooled RMSE would have been
detected (power about 1). On this panel the sub-grid and exposure geography channels are absent at a size that
matters, not undetectable.

**F0 on the winter panel W1** (`results/F0_w1/`; section 3). The same audit on eight ice-storm events: the
elevation bands now change eight features by more than the materiality threshold, all near-freezing
precipitation (wet-bulb hats at -1.5, 0, +1.5 C), in up to 1.93% of the outage-weighted county-hours and two to
three events; pop - area reaches 7.2%. Small, but the quadrature passes F0 there, as the phase physics predicts.

## 2. Screens on the wind panel

| arm | change | RMSE | vs base | county interval | event x state interval | events better | file |
|---|---|---|---|---|---|---|---|
| base | W+Cin, round-2 inputs (area weights) | 0.023914 | - | - | - | - | results/screen_S1.json |
| pop | every host input population-weighted (data v3p) | 0.023854 | -0.25% | [-3.66, +2.96] | [-4.60, +3.74] | 5/12 | results/screen_S1.json |
| clean | training loss without EAGLE-I artefact hours (evaluation unchanged) | 0.023278 | -2.66% | [-5.74, -0.06] | [-6.14, +0.16] | 6/12 | results/screen_S2a.json |
| clean placebo | as many hours dropped at random, same event and outage-level bin | 0.024098 | +0.77% | [-0.53, +2.28] | [-0.46, +2.03] | 4/12 | results/screen_S2c.json |
| clean vs its placebo | (second number of the two-numbers rule) | - | -3.40% | [-6.55, -0.63] | [-7.00, -0.37] | 11/12 | results/screen_S2c_vs_placebo.json |
| hazard (area) | W+Cin+H on eih_area (competing hazard, 160 features) | 0.024089 | +0.73% | [-3.48, +4.41] | [-3.36, +4.06] | 6/12 | results/screen_S2b.json |

**Target cleaning** (`build_train_mask.py`; LITERATURE_preprocessing change 4). EAGLE-I stores no zero rows and its
scrapers time out in storms, so hourly series carry artefacts that no weather input explains: one-hour dips and
spikes, plateaus of an identical non-zero count for four days or more (stale maps), fractions at the denominator.
They are 0.40% of the observed forecast hours but 5.1% of the sum of squared targets. Dropping them from the
training loss only, with the evaluation targets and masks unchanged, is the first change of this line that clears
the screen. Its placebo drops as many hours in the same event and outage-level bin at random (3,410 of 3,531
matched; 3.8% of the sum of squared targets). The gain sits on the unflagged hours (-2.89%; the flagged hours
themselves get +0.94% worse, as they are no longer fitted) and in both phases (rise to the peak -3.74%, decay after
it -2.04%; `results/diag/clean_split.json`): the artefacts distorted the learned response everywhere.

**Not confirmed on the other folds.** Folds 3, 4 and 5 of the same design give +0.63% ([-1.00, +2.40]), +2.53%
([-2.77, +8.40]) and +1.43% ([-1.05, +3.86]); over all five folds the change is +0.01% (county [-1.81, +1.83],
event x state [-1.44, +1.35], 5/12 events; `results/screen_clean_5fold.json`). The screen's two folds produced a false positive that its own placebo did
not catch, because the placebo shares the same two folds. Lesson for program.md: a screen keep needs the other
folds (or another seed) before it is called a gain.

**Canopy.** The data-pattern note (`notes/DATA_PATTERNS.md`) finds canopy x wind the one robust geographic signal
in the base's residuals (same sign in 11 of 12 events). Giving the host canopy as a seventh county context did not
transfer: +1.58% against the cleaned base over folds 1-2 (county [-0.41, +3.51]); neither did the
hazard-gated version, gust ramps x canopy as three extra damage inputs with zero-initialised weights (+1.17%,
[-0.91, +3.14]; `results/screen_S4_canopy.json`). The association was found in-sample, so this is the expected failure mode of geography that the
host can use as a county signature.

## 3. The winter ice-storm panel W1

Eight ice-storm episodes selected from Storm Events metadata only (`select_events_winter.py`: at least 40 counties
with Ice Storm reports in three days; 2021-02-13 excluded for operator load shedding; 2022-12-13 dropped by the
EAGLE-I coverage gate), 3,098 county-events, 1,626 counties; ERA5 from the ARCO-ERA5 store (identical to the CDS
files to packing precision, `data_provenance/arco_verify.json`); EAGLE-I 2023 added from figshare. The W1 base has
modest skill (MSE skill 0.16 against the all-zero forecast; negative in four smaller events,
`results/w1_base_skill.json`).

Hazard arms (W+Cin+H; triggers gust, wet-bulb-hat precipitation and convective, plain rain excluded as a load),
seed 0, folds 1-2 (`results/screen_W1_seed0.json`):

| arm | nodes | vs W1 base | county interval | events better |
|---|---|---|---|---|
| Hq | own cell x elevation-band nodes | -0.77% | [-6.89, +5.55] | 5/8 |
| Hp | own cells, no bands | +4.11% | [-3.53, +13.40] | 4/8 |
| Ho (placebo) | another county's bands, same relief stratum | +0.74% | [-9.31, +14.14] | 4/8 |

Fold 1 alone gave Hq -9.61% (event x state [-16.91, -1.73], 8/8 events) and seed 1 repeated the sign (-5.37%),
but fold 2 reversed (+4.45%): a handful of county-events decide these numbers (65% of the Hq-Hp difference on
fold 1 sat in five county-events). The residual audit on W1 (`results/F1_w1/`) finds the sub-grid contrast not
testable (fewer than 20 effective event x state clusters for every feature), not absent; its strongest
descriptive feature is the 48-h memory of precipitation near a wet-bulb temperature of -1.5 C. That single
feature is pre-registered for an independent panel (`PREREG_W2.md`, committed before any W2 data): 18 ice-storm
episodes 2014-2025 disjoint from W1.

**Event-grouped W1, all eight storms held out in turn** (four folds of two storms; `results/screen_W1_event.json`,
`results/w1_event_vs_zero.json`): pooled RMSE all-zero 0.0415, base 0.0519, hazard arm 0.0475 (-8.54% against the
base, county [-15.94, +0.03], event x state [-18.28, +5.87], 5/8 storms). The arm beats the all-zero forecast on four
storms (2020-10-25, 2023-01-30, 2023-02-20, 2024-12-13) and loses on four; the base beats it on none by more than
the arm does. Neither transfers to unseen ice storms with skill over the null; the physics-gated pathway mainly
shrinks the host's false alarms.

First two folds (four unseen storms; `results/screen_W1_event_f12.json`), for the record: the hazard arm is -13.57% against the base (county [-23.64, -1.62], event x state
[-24.06, +9.04], 3/4 storms). But on unseen ice storms the host has no skill over the all-zero forecast: RMSE
zero / base / hazard arm = 0.0414 / 0.0884 / 0.0719 (2018-11-13, a threefold false alarm), 0.0465 / 0.0502 /
0.0428 (2023-01-30), 0.0270 / 0.0270 / 0.0283 (2019-02-05), 0.0431 / 0.0428 / 0.0420 (2023-02-20). The pathway
mostly shrinks the host's false alarm; it beats the null clearly on one storm. Transfer to unseen ice storms is the
open problem this panel exposes.

(Further rows are added as the screens finish.)

Three seeds on folds 1-2 (`results/w1_seeds3_f12.json`): Hq -0.77 / -5.23 / -5.93% (seed-averaged prediction
-2.79%, county [-9.96, +4.23]); Hp +4.11 / -6.22 / -3.11% (seed-averaged -0.92%, [-8.93, +7.12]). The bands are
ahead of the plain cells in two of three seeds, inside the noise.

## 4. The pre-registered test on independent ice storms (W2d)

`PREREG_W2.md` (header and amendment 1 committed at 02:02-02:03 EDT, test code frozen at 239fd7d and amendment 2 at
03:06, all before any W2d feature existed; the times typed inside the file were wrong, see its erratum). Panel: the
ice-storm episodes of 2014-2025 with at least 20 counties, disjoint from W1; after the gates 12 events, 4,049
county-events. The registered single test (the 48-h near-freezing precipitation feature, elevation bands against
cells, residual alignment with the W2d base) returned **not testable** (`results/F1_w2d/H1a.md`): effective clusters
event x state 30.5 and county 114.3 pass, but effective events 7.5 fall short of 8 because the feature's mass is
uneven across events. Decided from the design; no residual was read. By the registered decision table: more
independent near-freezing events are needed, no claim either way. The W2d residuals stay unread (the F0 audit of
W2d, which weights by the observed outages, was computed by the build chain and set aside unread) so that an
enlarged panel containing W2d can still be the confirmatory test.
