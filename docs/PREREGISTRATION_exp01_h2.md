# Pre-registration: identifiability of the two rates (EXP01, second design)

Written before the new sweep has been run. Nothing below is a finding.

**The criteria in the docstring of `experiments/exp01_identifiability.py` are void
for this design.** They were written for a sweep that could not reach the regimes
they referred to, so the verdicts they produced -- for H2 in particular -- are not
evidence either way and must not be quoted as a negative result. Re-registering
here rather than editing that docstring keeps both versions on the record.

## Why the first design could not test H2

H2 predicted that rate recovery is non-monotone in forcing: near `y = 0` the
restoration term is multiplied by `y` and `r` is unidentified, near `y = 1` the
interruption term is multiplied by `(1 - y)` and `u` is unidentified. The sweep
moved `pulse_scale` over a 32-fold range and reported `frac(y > 0.99) = 0.0000`
at every level.

That was not a weak effect. It was unreachable, and provably so. Write the
generator's rates at their extremes:

    u in [cap_u * sigmoid(b_u), cap_u]          hazard >= 0, sigmoid saturates
    r in [cap_r * sigmoid(b_r - a_r), cap_r * sigmoid(b_r + a_r)]   daylight in [-1, 1]

The update `y <- y (1 - u - r) + u` is a contraction whenever `u + r < 1`, which
holds throughout, so `y` approaches `u / (u + r)` from below and cannot overshoot
it. The attainable band is therefore closed-form, and for the constants the first
sweep used (`cap_u = 0.30`, `cap_r = 0.15`, `b_u = -3.5`, `a_r = 1.5`,
`b_r = -0.5`):

    floor   = u_min / (u_min + r_max) = 0.0742
    ceiling = u_max / (u_max + r_min) = 0.9438

Confirmed against the generator at `pulse_scale` 50 and 200, where the realised
maximum is 0.9288 and does not move between them. **Both** of the regimes H2 names
lie outside that band. Reaching `y > 0.99` requires `r < u/99 <= 0.00303`, i.e.
`cap_r <= 0.0254`, and the sweep held `cap_r` six times larger; reaching `y < 0.01`
requires `u < r/99`, which no non-negative hazard can produce at `b_u = -3.5`.

The fault is that forcing amplitude was treated as the identifiability axis. It
is not: the walls of the attainable band are set by the rate constants, and
`pulse_scale` moves the state *within* the walls without moving the walls. A
sweep along it can only ever sample the interior.

The general lesson, which applies past this experiment: when a hypothesis names a
limiting regime, the design has to show the regime is reachable before it can
report that the regime was not informative.

## The new sweep

The axis is the equilibrium the generator parks at, and it is set directly rather
than approached. For a target `y*`, put

    u0 = S * y*        r0 = S * (1 - y*)        S = 0.10 fixed

so that `u0 / (u0 + r0) = y*` exactly, and `u0 + r0 = S` at every point -- the
relaxation rate is held constant across the sweep, so points differ in *where*
the state sits and not in how fast it gets there. Each cap is set to twice its
operating rate, which places each sigmoid at its midpoint where it is best
conditioned, and the intercepts then follow in closed form:

    cap_u = 2 * u0    b_u = -a_u * median(hazard)
    cap_r = 2 * r0    b_r = 0

Sweep: `y* in {0.01, 0.03, 0.08, 0.20, 0.40, 0.60, 0.80, 0.92, 0.97, 0.99}`,
three or more seeds, `pulse_scale = 1.0` throughout.

Trajectories start at `y = 0` and the first 64 steps are simulated but not
fitted. The relaxation time is `1/S`, so climbing from zero to `y* = 0.99` takes
about `ln(100)/S` steps; without a burn-in a large part of every window would be
the *approach* to the target rather than the target, and that approach is exactly
the well-identified interior the ends of the sweep are trying to leave. Fitting
through it would make the degenerate regimes look more identifiable than they
are, which is the direction of error that would manufacture a null.

**Design-validity check, run before registering, and a property of the generator
rather than a result.** No model was fitted. Building the data exactly as the
experiment builds it and rolling the known dynamics forward gives realised
`mean y` of 0.0096 at `y* = 0.01` and 0.9863 at `y* = 0.99`, with
`frac(y < 0.01) = 0.659` at the low end and `frac(y > 0.99) = 0.309` at the high
end. Both degenerate regimes are inside the swept range. This is the check the
first design lacked, and it is a precondition for the sweep, not evidence about
the hypotheses.

## Hypotheses

**H1 -- trajectory fit is not evidence of rate recovery.** Trajectory RMSE stays
low across the whole sweep while rate recovery does not.

**H2a -- recovery is non-monotone in the state.** Joint recovery error,
`max(nrmse_u, nrmse_r)`, is minimised in the interior of the sweep and rises at
both ends.

**H2b -- the degeneracy is side-specific, and directional.** At the low end the
restoration rate is the one that is lost (`nrmse_r > nrmse_u`); at the high end
the interruption rate is (`nrmse_u > nrmse_r`). This is the substantive claim;
H2a is its symptom and would also be produced by a symmetric loss of both rates.

**H3 -- the fit slides along a ridge.** In the worst-identified regimes the two
rate errors are positively correlated: an inflated interruption rate is paid for
by an inflated restoration rate.

**H2c -- control: the pattern is about the state, not the rate magnitudes.**
Along the sweep, `u0` and `r0` necessarily change size as `y*` moves, so recovery
could in principle track rate magnitude rather than state position. Control
condition: hold `y* = 0.5` and sweep `S` across the same range of values that
`u0` takes in the main sweep. Registered prediction: the U-shape does **not**
appear in the control.

## Kill conditions

* H1 dies if trajectory RMSE varies by more than 3x across the sweep.
* H2a dies if the argmin of joint recovery error is at either end of the sweep
  for a majority of seeds, or if joint error varies by less than 2x across the
  sweep -- a flat profile means the axis carries no identifiability structure and
  is a null result, not a weak confirmation.
* **H2b requires a sign flip in `nrmse_r - nrmse_u`** between the lowest and
  highest sweep points, consistent across at least 3 seeds. One side degrading
  more than the other at a single end is a magnitude difference and does not
  count. If H2a holds but H2b fails, the reportable claim is that recovery
  degrades at the edges without the asymmetry the mechanism predicts.
* H3 dies if the sign of the error correlation is not consistent across all seeds
  at the worst-identified sweep point.
* **H2c is a gate on the others.** If the control reproduces the U-shape, then
  H2a and H2b are confounded with rate magnitude and may not be reported as
  evidence about state position, whatever they show.
* A null result is reportable and will be reported. This is a synthetic
  experiment: it can establish that a degeneracy exists in a system whose truth
  is known, and it cannot establish anything about county data. Its grade is
  [B-synth] at best.

## What is deliberately held fixed

* `pulse_scale`, at 1.0. It is no longer the axis, and leaving it free would
  reintroduce the confound the first design died of.
* The burn-in length, at 64 steps, and the fitted window, at 96.
* The driver process, the seeds, the network capacity, the optimiser, and the
  fitting budget. Only the generator's rate constants move.
* Both rates keep all three driver channels as input. The model has to discover
  which channel drives which rate; handing it the answer would measure something
  else.
