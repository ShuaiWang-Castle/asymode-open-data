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

Operationally, per event family, take the paired advantage of the two-rate arm
over a **single net-rate arm** at h+24 and h+48, and regress it on the family's
median fall/rise ratio.

* **H-E1** The ordering of the advantage matches the ordering of the ratio:
  tropical > wind > convective > winter.
* **H-E2** On winter, the advantage is not distinguishable from zero — the
  two-rate arm and the single net-rate arm are level.

### Amendment, 2026-09-01 evening, before any family-stratified fit

**The comparator was changed from damped persistence to a single net-rate arm.**
The original wording compared the susceptible arm to damped persistence. That
comparator's *difficulty varies by family*: winter events are symmetric, short
and shallow — exactly the regime in which damped persistence is close to optimal
for any model. "No advantage on winter" would then have two explanations, (a) the
phases do not separate so the structure has nothing to buy, or (b) winter is easy
for every method and the ceiling is low. Only (a) is the mechanism this
hypothesis is about, and the confound produces precisely the outcome H-E2 hopes
to see — so passing it would not have counted. Raised by the experiment lane on
reading this file; adopted.

The single net-rate arm is the two-rate model with its two rates collapsed into
one signed rate, everything else held equal:

    two rates (this work):   y <- clip(y + u(1-y) - r y, 0, 1),  u = cap_u sigma(f_U(x)), r = cap_r sigma(f_R(x))
    one net rate (control):  y <- clip(y + n, 0, 1),             n = cap_n tanh(f_N(x))

Same inputs, same width, same optimiser, same seeds, same initialisation rule
(`calibrate_init` sets the net rate to the observed mean one-step net change).
"Two rates versus one" becomes the only difference between the arms.

### Second amendment, same evening, still before any family fit

**Two further confounds in the net-rate comparator, both raised by the
experiment lane on implementing it, both adopted.**

*Parameter count.* One hidden-32 network has 1,569 parameters; the two-rate arm
has 3,138. "Two rates beat one" would then also read "twice the parameters beat
half" — the same class of confound as the one that prompted the first amendment,
and again in the direction that favours this paper. A single network of hidden
width 48 has 3,121 parameters, within 0.5%. Two alignments are possible and
neither is uniquely right, so **both are run and reported**:

* `net_scaled`, hidden 48 — **parameter-matched. This is the load-bearing
  comparator for H-E.**
* `net_scaled_narrow`, hidden 32 — width-matched; appendix, to show the
  conclusion does not hinge on the alignment chosen.
* `net`, hidden 48 — no state scaling either; the "no structure at all" end.

*Concurrency.* An earlier draft of this file said a single network with the
state scaling kept was "essentially" the input- and capacity-symmetric arms of
`PREREGISTRATION_asymmetry.md`. That was wrong. Those arms still carry **two
non-negative rates that are both active at every step**; a single signed rate
forces damage and restoration to be **mutually exclusive** within a step. That
is a distinct structural commitment — and it is the one closest to the paper's
central picture, that interruption and restoration run concurrently. It is
therefore tested on its own, and the ablation becomes a ladder in which each
rung adds exactly one structural element:

    net            no state scaling, no concurrency            (one signed rate)
    net_scaled     state scaling, no concurrency               (one signed rate x (1-y) or y)
    sym_in+sym_arch  scaling + concurrency, shared inputs and width (two non-negative rates)
    two-rate       scaling + concurrency + input and capacity asymmetry

All four rungs are reported for every family. The H-E comparator is
`net_scaled`.

*Boundary check, recorded because it matters:* `net_scaled` **can ignite from
`y = 0`** — a positive net flow times `(1 - y)` is the flow itself at zero — so
it is not disposed of by the algebra that disposes of the epidemic form. It is a
real opponent, which is the point.

Initialisation for the single-rate arms follows `calibrate_init` as far as it
can: the net rate is set to the observed mean one-step net change. A single
signed rate cannot reproduce both the mean rise and the mean fall at once; that
is the arm's defining property, not a defect of the rule.

Damped persistence is still reported for every family, **as a lower bound, not
as the comparator for the mechanism claim.**

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
* H-E2 dies if the two-rate arm beats `net_scaled` (parameter-matched) on winter
  by more than 3% at either horizon with a paired t below −3. If that happens, the
  mechanism story is wrong and must be withdrawn, whatever the primary result
  says.
* If the net-rate arm cannot be run under the full protocol, H-E falls back to
  the damped-persistence comparator **and the following weakness is carried into
  the paper verbatim:** a zero advantage on winter cannot then be distinguished
  from winter being easy for every method.
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
