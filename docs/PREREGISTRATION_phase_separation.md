# Pre-registration: does the advantage track phase separation?

Written before any family-stratified fit has been run, and before the primary
convective result exists. Nothing here is a finding.

## The measurement that motivates it

Damage and restoration are separate processes, so they have separate timescales.
For every county the storm interrupts, measure them separately: the rise from the
first crossing of `y = 0.01` to the peak, and the fall from the peak back below
it. Their ratio is how far apart the two phases sit.

Measured on the panels built so far, county-events with peak `y >= 0.02`:

| family | county-events | rise p50 | fall p50 | fall/rise p50 | share >= 5 |
|---|---|---|---|---|---|
| tropical | 522 | 5 h | 29 h | 4.0 | 45.0% |
| wind | 1,043 | 6 h | 17 h | 2.1 | 25.8% |
| convective | 1,619 | 3 h | 7 h | 1.7 | 24.5% |
| winter | 1,248 | 5 h | 4 h | **1.0** | 10.1% |

Winter storms in this catalogue fall as fast as they rise. Whatever else is true
of them, they are not a setting where separating the two rates can buy anything,
and that makes them useful.

## The hypothesis

**H-E — the advantage of two separately parameterised rates over one net rate
grows with the separation between the damage and restoration timescales.**

Operationally, per event family, take the paired advantage of the susceptible arm
over damped persistence at h+24 and h+48, and regress it on the family's median
fall/rise ratio.

* **H-E1** The ordering of the advantage matches the ordering of the ratio:
  tropical > wind > convective > winter.
* **H-E2** On winter, the advantage is not distinguishable from zero — the
  susceptible arm and the best single-rate baseline are level.

## Why H-E2 matters more than H-E1

A structural claim needs a place where the structure should *not* help, and does
not. Without it, "our model wins" is compatible with the model simply having more
capacity, better optimisation, or a luckier initialisation. Winter is that place:
the two phases coincide, so the mechanism the paper argues for is absent, and a
win there would be evidence *against* the stated mechanism rather than for it.

**A win on winter is therefore a bad outcome for this paper, not a good one.**
Recording that in advance is the point of writing this down.

## Kill conditions

* H-E1 dies if the ordering is violated at either horizon by more than one
  adjacent swap, across 3 seeds.
* H-E2 dies if the susceptible arm beats damped persistence on winter by more
  than 3% at either horizon with a paired t below −3. If that happens, the
  mechanism story is wrong and must be withdrawn, whatever the primary result
  says.
* The whole hypothesis is void if fewer than three families produce a fitted
  result under the full protocol — three points do not identify a trend, and two
  certainly do not.

## Known weaknesses, stated now rather than when a reviewer finds them

* **Family sizes are very unequal.** Tropical has 3 panels against convective's
  11. The trend, if it appears, is estimated from few points at the ends.
* **Family is a proxy.** The mechanism is phase separation, not meteorology. A
  better test would stratify county-events directly by their own fall/rise ratio,
  within family, and that is the follow-up if H-E survives. It is not the primary
  test here only because within-family stratification cuts the per-cell sample
  further.
* **The ratio is measured on the same panels the models are fitted to.** It is a
  property of the data, not of a fit, so this is not leakage in the usual sense —
  but the family assignment was chosen before any model was fitted, and must stay
  fixed.
* `freeze_cycle` is identically zero outside the cold-season panels, so it is
  excluded from the primary convective runs and used only in the generalisation
  check. A family of dead channels would otherwise dilute normalisation and could
  make an arm differ for a reason that is not structural.

## Scope, decided and fixed

**Primary study: convective, 11 panels, 1,566 distinct counties.** Chosen for
sample size and because it is the harder case — the phase ratio there is 1.7, not
4.0. A structure that pays off where separation is moderate and the events are
the most common is a stronger claim than one demonstrated only where separation
is obvious.

The other families are a secondary generalisation check under this hypothesis,
never a source of primary numbers.
