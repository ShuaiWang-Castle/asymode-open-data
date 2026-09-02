# Pre-registration: which covariates belong on which rate

Written before any of these features exist in this repository, and before any of
the hypotheses below has been fitted on the data. Every claim here is a
*hypothesis to be tested on public observations*. None of it is a finding, and
nothing in this file may be quoted as evidence.

The point of writing it down first is that these hypotheses did not arise from
looking at this data. Testing a pre-selected shortlist and reporting only the
survivors would be a garden of forking paths; registering the whole list before
running is the control for that.

## The structural hypotheses

**H-A — Input asymmetry is real and directional.**
The two rates read different evidence, and moving a family across the state
equation does not merely fail to help but actively hurts.

* H-A1  Pre-origin *outage level* (last value, max, mean, trend over the lead-in)
  helps the restoration rate and hurts the interruption rate.
* H-A2  Neighbouring-county weather helps the restoration rate and hurts the
  interruption rate.
* H-A3  Ambient meteorological fields beyond the core hazard variables help
  **both** sides. Registered explicitly as a *negative* case: the claim under test
  is that asymmetry is selective, not that every family is asymmetric. If
  everything turns out asymmetric, that is evidence the test is not discriminating.

**H-B — Two summaries of the same source separate onto opposite sides.**
From the same pre-origin outage history, the *level* belongs on restoration and
the *clearance rate* (how fast outages were being cleared before the window)
belongs on interruption. Mechanistic reading to be tested, not assumed: clearance
rate proxies county infrastructure fragility and modulates how damage occurs;
level is present backlog and modulates where restoration resumes.

**H-C — A gate on the interruption rate wants less input, not more.**
If the interruption rate is written as a gated pulse plus an always-on background
term rather than a single bounded sigmoid, the gate performs better when it reads
county identity and hazard composites but **not** raw weather and **not** the
diurnal clock. Two sigmoids multiplied on the same inputs is the suspected failure
mode; the gate is expected to be sensitive to input width in a way the rates are
not.

**H-D — Capacity asymmetry follows evidence, not symmetry.**
The restoration rate is identified from a much smaller effective sample than the
interruption rate, so it should carry less capacity — plausibly a plain logistic
GLM with directly readable coefficients — while the interruption rate keeps a
hidden layer. The effective sample sizes on each side are to be measured here, on
this panel, and reported whatever they say.

## Kill conditions

* Each of H-A1, H-A2, H-B requires a **sign flip** — beneficial on one side,
  harmful on the other — consistent across at least 3 seeds and 5 county-held-out
  folds. A family that merely helps less on one side than the other does **not**
  count; that is a magnitude difference, not asymmetry.
* H-A3 is confirmed only if the family helps on both sides. If it is asymmetric
  too, H-A is weakened, not strengthened, and that must be said.

  **H-A3 was voided on manifest g1 and is CONFIRMED on manifest g2** (see
  `RESULTS_LEDGER.md`, EXP08 on g2). The void below was recorded on a superseded
  run and stands only as history. The amendment that follows it — H-A3', a second
  control family — is retained because it was registered before the g2 result was
  seen, and two controls are stronger than one.

  *Original void entry, g1 only:* The family chosen for it -- ambient fields
  outside the core hazard set -- turned out to carry no signal on *either* rate on
  the convective panels: all three removals moved the error by less than 0.8%,
  with win counts near chance. A family with no signal is symmetric trivially, so
  it cannot serve as the control the hypothesis needs. The test was uninformative
  about symmetry, exactly as a kill condition applied to an unreachable regime is
  uninformative about identifiability.

  **H-A3' replaces it, registered here before being run.** The control family
  becomes precipitation and wind speed. The selection rule is stated so it cannot
  be tuned afterwards: a control family must have a *documented mechanism on both
  sides*. Rain and wind damage plant, and they also stop crews from working
  safely at height. Soil moisture fails this rule and stays with hazard -- it has
  a damage mechanism and no restoration mechanism, which is why moving it would
  rig the control. The kill conditions are unchanged: H-A3' is confirmed only if
  removing the family hurts both sides, and it is void again if it turns out to
  have no signal on either.
* H-C requires a monotone or single-peaked relationship between gate input width
  and the failure rate across initialisations. A flat profile kills it.
* H-D requires the reduced-capacity restoration rate to be no worse than the
  full-capacity one. Equal performance confirms it; worse performance kills it.
* **A null result is reportable and will be reported.** If none of these
  asymmetries appears on public data, the paper says the asymmetry is not
  supported at county scale on this target, and the contribution reduces to the
  dynamical form alone.

## Feature families to build, all from primary public sources

Every family below is derived in this repository from sources named in
`docs/DATA_CARD.md`. Nothing is imported.

| family | side under test | built from |
|---|---|---|
| hazard composites (wet wind, wind energy, snow/ice, near-freezing, cold precip, and their rolling sums) | interruption | ERA5 |
| hazard path (cumulative since forecast origin, hours since peak gust) | interruption | ERA5 |
| regional footprint (share of counties in the same state above a gust threshold) | interruption | ERA5 + panel |
| lagged weather aggregates (rolling max/sum over 6/12/24 h) | interruption | ERA5 |
| freeze–thaw cycling (hours below freezing, crossings) | interruption | ERA5 |
| wind direction shift (3 h and 6 h, wrapped) | interruption | ERA5 |
| pre-origin clearance rate | interruption *(H-B)* | panel |
| pre-origin outage level, max, mean, trend | restoration *(H-A1)* | panel |
| neighbouring-county weather (adjacency-based) | restoration *(H-A2)* | ERA5 + Census adjacency |
| ambient meteorology (dewpoint, pressure, boundary-layer height, cloud layers, radiation, VPD) | both *(H-A3)* | ERA5 |
| county statics (land area, rural–urban code, population, utility count, reliability indices, customer density, cooperative share, forest cover, housing age, elevation) | gate *(H-C)* | Census, USDA ERS, EIA-861, USDA FIA, ACS |
| diurnal clock | restoration | derived |

## What is deliberately excluded

* Any severity index that is not defined in `docs/DATA_CARD.md`, and any variable
  derived from one.
* Any fixed geographic scope, state indicator set, or county list carried from
  elsewhere. County selection here follows the public event catalog and the
  coverage gate, and nothing else.
* Any anticausal family — features reading the forecast window's future. The
  drivers over the forecast horizon are given as a forecast stand-in, which is
  stated in `experiments/exp05_real_dynamics.py`; features that aggregate *across*
  the horizon are a different and stronger assumption and are not used.
