# Theoretical implications of the audit

## 1. The algebraic theory survives; the empirical interpretation does not

The exact class-comparison theorem remains correct. For

\[
m_0(x,y)=U_0(x)(1-y)-R_0(x)y,
\]

and the state-scaled one-flow class

\[
\mathcal F_1(x)=\{a(1-y):a\ge0\}\cup\{-by:b\ge0\},
\]

the population approximation gap is

\[
G(x)=v(x)\min\left\{\frac{R_0(x)^2}{A(x)},
                         \frac{U_0(x)^2}{B(x)}\right\}.
\]

The identification identity `det Q(x)=v(x)` and the local fixed-design bias--variance benchmark are also mathematically valid within their stated assumptions.

What fails is the inference that the current real-data gain validates simultaneous physical interruption and restoration. The experiment does not isolate that claim.

## 2. The current gain occurs where the restoration multiplier is inactive

At `y=0`,

\[
m_0(x,0)=U_0(x).
\]

The restoration component does not enter the observed transition. For the signed one-flow class,

\[
m_1(x,0)=[s(x)]_+.
\]

If the signed head is negative, the one-flow prediction is exactly zero. The health audit reports that the two-flow advantage is positive at zero and near-zero origins but reverses for `y0>0.01`.

Therefore the existing result is evidence, at most, that **decoupling the onset direction from the recovery direction** can help a zero-heavy predictive task. It is not evidence that two simultaneously active state contributions improve prediction in the interior. A concurrency claim requires positive gains on observations where both state multipliers are non-negligible.

## 3. The one-flow boundary effect is partly structural and partly an implementation defect

Boundary silence is inherent in the chosen one-flow class: a negative signed rate multiplied by `y=0` cannot create an outage. But the confirmatory harness initializes the signed head near `-0.1` in every fold by subtracting two differently scaled rates. This makes the boundary behavior much more likely and prevents the current result from being interpreted as a clean class-comparison experiment.

A repaired experiment must use a nested parameterization and a common constrained-loss initialization. Only then can boundary behavior be attributed to the one-flow restriction itself.

## 4. Add an explicit omitted-context proposition

The paper currently writes rates as functions of observed drivers `X` alone. Suppose the true transition also depends on latent or omitted context `H`, such as storm age, accumulated exposure, utility characteristics, or crew deployment:

\[
\Delta Y=(1-Y)U(X,H)-YR(X,H)+\varepsilon.
\]

After marginalizing `H`,

\[
\mathbb E[\Delta Y\mid X=x,Y=y]
=(1-y)\mathbb E[U(x,H)\mid x,y]
-y\mathbb E[R(x,H)\mid x,y].
\]

This conditional mean is affine in `y` with coefficients that depend only on `x` only if the two conditional expectations on the right are invariant in `y` (or satisfy an equivalent cancellation). If latent storm/county context is associated with the state, the pooled two-flow model is itself misspecified.

This proposition should be stated before any physical interpretation of the learned heads. The current non-affinity audit finds no detectable curvature at the tested resolution, but it has weak power because the state is concentrated near zero. Absence of detected curvature is not evidence that the omitted-context condition holds.

## 5. Reinterpret the rates as conditional-mean components unless external labels are available

Without interruption and restoration event logs, utility crew records, or another direct process measurement, the two neural outputs should be described as:

> nonnegative conditional-mean interruption and restoration components under the model.

They should not be called recovered physical hazards. The theory identifies functions within the assumed conditional-mean class, not causal mechanisms.

## 6. The cross-county identification claim must be revised

The raw between-county variance share near 0.78 is almost reproduced by permuting county labels because local neighborhoods contain about one row per county. It is therefore a degrees-of-freedom statistic, not evidence that cross-county heterogeneity supplies 78% of useful identification.

The k=800 analysis gives a smaller positive excess over the permutation null. The defensible statement is that broader neighborhoods contain some cross-county state dispersion, while the magnitude and validity of that information depend on pooling homogeneity. The value 0.78 must not be presented as an identified scientific share.

## 7. The local plug-in Gamma selector is not supported by these data

The plug-in estimate

\[
\widehat\Gamma
=\frac{n\widehat G}{\widehat\sigma^2}
\]

is dominated by boundary estimates and noise. Nearly half of k=200 neighborhoods set the estimated coactivity factor to zero, yet most corresponding unconstrained estimates are statistically indistinguishable from zero rather than significantly negative. Noise-floor corrections push the median selector to zero at every tested bandwidth.

The finite-sample Gamma result may remain as a solvable benchmark. It should not be used as a real-data operational selector unless a future dataset has enough interior-state information to estimate both components reliably.

## 8. A nested parameterization aligns the algorithm with the theory

Define

\[
s(x)=U(x)-R(x),\qquad c(x)=\min\{U(x),R(x)\}.
\]

Then

\[
U(x)=[s(x)]_+ + c(x),\qquad
R(x)=[-s(x)]_+ + c(x).
\]

This gives a one-to-one representation of every nonnegative two-flow pair. The one-flow class is exactly the restriction `c(x)=0`. Thus the empirical ablation can free one concurrency function while keeping the signed direction function and all shared features fixed.

A useful corollary is

\[
G(x)=0
\quad\Longleftrightarrow\quad
v(x)=0\ \text{or}\ c(x)=0,
\]

because `c=0` is equivalent to at least one original component being zero. The magnitude of `G`, however, still depends on the state moments and on which one-flow branch gives the better projection; `c` alone is not a complete selection score.

## 9. The primary empirical theorem test must be one-step, not only rollout

The exact approximation gap is a one-step conditional-risk statement. A recursive horizon introduces

- optimization error;
- model misspecification;
- state-distribution shift along the predicted path;
- accumulation or cancellation of transition errors.

For predicted and true mean trajectories, the error obeys the product-sum recursion

\[
e_{t+1}=(1-\widehat U_t-\widehat R_t)e_t+\delta_t,
\]

where `delta_t` is the local transition-function error evaluated along the reference path. Therefore a two-flow one-step advantage need not survive at 24 or 48 hours.

The next paper-level experiment should test two linked hypotheses:

1. **transition hypothesis:** the nested two-flow model lowers held-out teacher-forced one-step MSE relative to its `c=0` submodel;
2. **rollout hypothesis:** the same fitted model lowers 24-hour recursive MSE.

The first is the direct theory test. The second is the application consequence.

## 10. Interpretation table for the next result

| Real-data result | Defensible conclusion |
|---|---|
| Two-flow wins overall and for interior states | evidence that the second conditional-mean component adds predictive value beyond onset boundary behavior |
| Two-flow wins only at zero/near-zero origins | separation helps onset/recovery decoupling; no evidence for concurrent interior flows |
| One-step wins but rollout does not | local representation gain is lost through estimation or recursive propagation |
| Rollout wins but one-step does not | forecast gain exists, but the oracle-gap mechanism is not established |
| Neither wins | the richer class is not worth its estimation/misspecification cost on this dataset |
| Both neural models lose to no-weather affine transition | weather representation or data quality is the limiting factor; do not interpret learned rates |

The theory is still useful precisely because it can accommodate each outcome. It should not be written as a guarantee that the richer neural model must win.
