# Model notes

## The state equation

    y_{t+1} = clip( y_t + u_t (1 - y_t) - r_t y_t , 0, 1 )

`y` is a fraction, so the dynamics must preserve `[0, 1]` without relying on the
clip to do it. Writing the inflow against the served pool `(1 - y)` and the
outflow against the interrupted pool `y` gives that for free: at `y = 1` the
inflow term vanishes, at `y = 0` the outflow term vanishes. The clip is a
numerical guard, not the mechanism.

## Why the rates are logits

`RateNet` returns `cap * sigmoid(logit)`. Two properties matter.

The bound is structural, so no clamp is needed on the rate. This is what makes
additive structural pathways safe: a variant that wants to add a memory term, a
context convolution, or a gate adds it to the **logit**. The alternative -- adding
in rate space and clamping to `[0, cap]` -- destroys the gradient wherever the
clamp binds. A pathway initialised near zero that drifts negative then pins the
rate at zero, receives no gradient, and never recovers.

`hidden = 0` collapses the network to a single linear layer, i.e. a logistic GLM
whose per-channel coefficients are directly readable. That is the low-capacity end
of the capacity-asymmetry probe and also the interpretability fallback.

## The three asymmetries, and why they are separable

They are separate switches because a reviewer will ask which one is doing the
work, and "all of it together helps" is not an answer.

1. **Dynamical.** `InflowForm.SUSCEPTIBLE` is `u(1-y)`; `InflowForm.TRANSMISSION`
   is `u*y*(1-y)`, the epidemic/diffusion form. The latter is identically zero at
   `y = 0`.
2. **Input.** `idx_u` and `idx_r` select driver channels per rate. Passing the
   same list to both switches the axis off without changing anything else.
3. **Capacity.** `hidden_u` and `hidden_r` size the two networks independently.

## The steelmanned epidemic arm

`InflowForm.TRANSMISSION_SEED` is `u * (y + eps) * (1 - y)` with `eps >= 0`
learnable through a softplus. It exists because the pure epidemic arm loses the
onset comparison by algebra, and a comparison decided by algebra convinces nobody.
With a seed the arm *can* ignite from zero, and the question becomes quantitative.

The arm is then caught between two requirements: `eps` must be large enough to
ignite from `y = 0`, but a large `eps` swamps the `y`-dependence that made the
form epidemic at all, pushing it toward the susceptible form it was meant to
contrast with. The fitted `eps` is reported as a headline number for that reason.

## Identifiability

At one step the data give `dy = u(x)(1-y) - r(x)y`: one equation, two unknowns.
Two observations sharing driver `x` at states `y1 != y2` give a 2x2 system with
determinant `(y1 - y2)`. The split of a net change into interruption and
restoration is therefore identified **only through variation of the state under
comparable drivers**, and conditioning degrades as `|y1 - y2| -> 0`. Two
saturation regimes follow directly: at `y = 0` the restoration term is multiplied
by zero and `r` is unidentified; at `y = 1` the interruption term is multiplied by
zero and `u` is unidentified.

This is why `exp01` sweeps forcing rather than reporting a single recovery number,
and why it reports the correlation of the two rate errors: in the badly
conditioned regime the fit slides along a ridge where an inflated `u` is paid for
by an inflated `r`, and the trajectory still looks fine.

## Deliberately not carried over

The target is defined in `docs/DATA_CARD.md` from public sources. No externally
derived severity index, feature list, county set, fold assignment, normalisation
statistic, or threshold constant is used anywhere in this project. All fits start
from random initialisation.
