# Adjudication of the conservation diagnosis

This note converts the supplied diagnosis into claims that can be checked from the public repository. It deliberately separates proved algebra from empirical extrapolation.

## Accepted as exact

### A1. Weighted constant-fit conservation identity

For positive weights `w_i`, consider the unconstrained weighted least-squares problem

\[
\min_{U,R}\sum_i w_i\{\Delta_i-U(1-y_i)+Ry_i\}^2.
\]

If the two-column design has full rank, its normal equations imply

\[
U(1-\mu_w)-R\mu_w=\bar\Delta_w,
\]

where `mu_w` and `bar_Delta_w` use the same normalized weights. This is an exact intercept-like moment condition.

### A2. Closed-window corollary

If, under the same rows and weights, `bar_Delta_w=0`, and the fitted rates are interior, then

\[
U/R=\mu_w/(1-\mu_w).
\]

This explains why a low-occupancy full-window constant fit can have a small interruption **rate** and a much larger restoration **rate** while the two mean flows remain balanced.

### A3. Closed-window selection bound

Let

\[
A=(1-\mu)^2+v,\qquad B=\mu^2+v.
\]

For `mu <= 1/2`, local closure gives `U=R mu/(1-mu)`, so the binding branch is `U^2/B` and

\[
G=\frac{vR^2\mu^2}{(1-\mu)^2(\mu^2+v)},
\]

hence

\[
\Gamma_n\leq
\frac{nR^2\min\{\mu^2,v\}}
{(1-\mu)^2\sigma_\varepsilon^2}.
\]

The factor `(1-mu)^{-2}` is retained in the exact audit, even though it is numerically close to one when `mu` is about `1e-3`.

### A4. Delivered difference of the output collapse

For proposals `U,R >= 0`, define the one-flow collapse by the signed rate `s=U-R`. Then

\[
m_2(y)-m_1(y)=\min\{U,R\}(1-2y).
\]

The experiment must therefore record both the common component `c=min(U,R)` and the delivered transition-space difference `c(1-2y)`.

## Accepted as a serious hypothesis, not yet a dataset-level conclusion

### H1. The broad seven-day panel can make the constant common component tiny

The pilot's fold-2 constants have approximately `U/R=1e-3`. That is consistent with a mean outage occupancy near `1e-3` under an approximately closed, interior constant fit. The exact source-window occupancy, mean drift, boundary status, and residual scale must be recomputed from the public bytes with the correct mask and weights.

### H2. A storm-conditioned window may materially increase the design index

When the full panel spends only a fraction `f` in an active regime, concentrating on a predeclared panel-level storm window can increase `mu` while reducing `n`. Under additional approximations, `Gamma` may scale roughly as `1/f`. This is a heuristic to measure, not a guaranteed scaling law.

### H3. County-level resolution may be too coarse for balanced flows

For a constant rate ratio `rho=U/R`, interruption and restoration flows are within a factor `K` only for

\[
\frac{\rho}{K+\rho}
\le y \le
\frac{K\rho}{1+K\rho}.
\]

At small `rho` this band has width of order `rho`. The audit compares the `K=2` band with each event's empirical state distribution and median one-customer fraction. The claim that the band is below reporting resolution is made only if those numbers support it.

## Not accepted without further evidence

### N1. “The 26-event main run must also have `c/R=0.1%`”

A global constant identity does not impose a pointwise identity on `U(x)` and `R(x)`. Under closure, the general moment is

\[
\mathbb E[U(X)(1-Y)]
=
\mathbb E[R(X)Y],
\]

not `U(x)/R(x)=mu/(1-mu)` for every driver state. Rare interruption contexts can carry large conditional `U(x)` while preserving the global balance.

### N2. “Any neural model containing constants satisfies the identity exactly”

That conclusion requires an admissible unsaturated constant-shift perturbation, no regularization contribution in that direction, and stationarity. Bounded sigmoid heads, active constraints, dead gradients, early stopping, and nonconvex optimization can all prevent the exact moment from holding. For neural models it is a convergence diagnostic, not an unconditional theorem.

### N3. “A correct comparator cannot produce a several-percent MSE difference”

The constant rate ratio alone does not bound relative MSE improvement. The denominator can be small, rollout can accumulate a small drift difference, and conditional rates can be heterogeneous. The earlier result remains reproduced legacy evidence until a properly measured structural comparison adjudicates it.

### N4. “The documented implementation defects are irrelevant”

The conservation preflight comes first because it is cheaper and may reveal a low-information design. The dead interruption head, incorrect transition mask, reset temporal state, degenerate starts, and missing treatment-dose traces remain real defects. They are deferred, not dismissed.

## Resulting project order

1. Verify public bytes and exact transition population.
2. Measure closure, residual scale, local `Gamma`, and state-resolution geometry before neural training.
3. Review the report.
4. Only then decide whether to repair and rerun the estimator, redesign the storm window, or treat low information as the empirical finding.
