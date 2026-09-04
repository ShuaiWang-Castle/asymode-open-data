# Conservation identity and its scope

## 1. Exact weighted constant-fit result

Let `p_i=1-y_i`, and fit

\[
\Delta_i=Up_i-Ry_i+\varepsilon_i
\]

by weighted least squares with normalized positive weights `w_i`. Write

\[
\mu_w=\sum_iw_i y_i,
\qquad
\bar\Delta_w=\sum_iw_i\Delta_i.
\]

### Proposition 1

If the design has rank two and the unconstrained optimum is used, then

\[
\widehat U(1-\mu_w)-\widehat R\mu_w=\bar\Delta_w.
\]

### Proof

At the optimum, the residual

\[
e_i=\Delta_i-\widehat U(1-y_i)+\widehat R y_i
\]

is orthogonal to both `(1-y_i)` and `y_i`. Since `(1-y_i)+y_i=1`, it is also orthogonal to the constant vector. Thus `sum_i w_i e_i=0`, which is the stated identity.

The same proof shows why the weights matter. Changing from row-pooled to equal-event weighting changes `mu_w`, `bar_Delta_w`, and the fitted rates together.

## 2. Closed-window relation is empirical, not automatic

For complete trajectories with common time weights,

\[
\frac1T\sum_{t=0}^{T-1}(Y_{t+1}-Y_t)
=
\frac{Y_T-Y_0}{T}.
\]

A window is approximately closed only when this boundary term is negligible under the exact analysis weights. Missing observation pairs, unequal county lengths, overlapping origins, event weighting, or selecting a non-closed active interval can make the weighted mean drift nonzero.

Only after measuring `bar_Delta_w` may one use the following as an exact identity under zero drift, or as a clearly labelled approximation under near closure:

\[
\widehat U/\widehat R
\approx
\mu_w/(1-\mu_w).
\]

The preflight reports a dimensionless closure ratio

\[
\frac{|\bar\Delta_w|}
{|\widehat U|(1-\mu_w)+|\widehat R|\mu_w}.
\]

The closed-window formulas are marked applicable only when this ratio is at most `0.05` and the unconstrained solution is inside the fixed rate box.

## 3. Boundary-constrained fits

The paper model imposes nonnegativity and rate caps. If a constant optimum lies on a boundary, the normal equations are replaced by KKT inequalities. The unconstrained identity still holds for the unconstrained fit, but the constrained fit may have

\[
\widehat U(1-\mu)-\widehat R\mu\ne\bar\Delta.
\]

The audit therefore reports unconstrained and exact box-constrained solutions separately, plus the active-boundary status. It never uses a boundary fit to claim the closed-window ratio identity.

## 4. Context-dependent models

For a fitted conditional model

\[
\widehat m(X,Y)=
\widehat U(X)(1-Y)-\widehat R(X)Y,
\]

global closure can at most imply the mean-flow relation

\[
\mathbb E[\widehat U(X)(1-Y)]
-
\mathbb E[\widehat R(X)Y]
=
\mathbb E[\Delta]
\]

when the fitted class and optimizer make the constant-shift residual moment valid. It does not imply a constant pointwise ratio between `U(x)` and `R(x)`.

A local ratio relation requires three additional conditions inside a driver neighborhood:

1. the local conditional rates are adequately approximated as constant;
2. the local fit is interior;
3. the local mean drift is approximately zero.

Those conditions are measured rather than presumed.

## 5. Consequence for the selection index

Let

\[
A=(1-\mu)^2+v,
\qquad
B=\mu^2+v,
\]

and let the one-flow oracle gap be

\[
G=v\min\left\{\frac{R^2}{A},\frac{U^2}{B}\right\}.
\]

The local fixed-design benchmark is

\[
\Gamma_n=\frac{nG}{\sigma_\varepsilon^2}.
\]

The preflight reports three different objects and does not conflate them:

1. **Plug-in Gamma:** uses the exact nonnegative box fit and its local residual scale. It is descriptive and can be upward or downward biased.
2. **Near-closure fitted-rate formula:** evaluates the exact zero-drift ceiling only in cells whose measured closure ratio is small and whose unconstrained fit is interior. Because the empirical mean drift need not equal zero exactly, this is labelled a diagnostic formula rather than an unconditional bound.
3. **General rate-cap bound:**
   \[
   \Gamma_n
   \le
   \frac{nv}{\sigma_\varepsilon^2}
   \min\left\{\frac{C_R^2}{A},\frac{C_U^2}{B}\right\},
   \]
   which requires no closure assumption but may be loose.

For `mu <= 1/2` under local closure,

\[
G=
\frac{vR^2\mu^2}
{(1-\mu)^2(\mu^2+v)}
\le
\frac{R^2\min\{\mu^2,v\}}
{(1-\mu)^2}.
\]

This is the exact qualified form of the proposed ceiling.

## 6. What the collapse changes

With

\[
s=U-R,
\]

the collapsed one-flow drift uses `[s]_+` on the served pool and `[-s]_+` on the interrupted pool. The difference from the two-flow drift is

\[
\Delta m(y)=c(1-2y),
\qquad
c=\min\{U,R\}.
\]

Consequently, any future trained comparison must save the distribution of `c` and `c(1-2Y)`. A model label alone does not establish that a nontrivial structural treatment was delivered.

## 7. Empirical interpretation rule

- A low full-window constant `U/R` is evidence that the **global constant task** is low-occupancy and nearly flow-balanced.
- A low local `Gamma` distribution after a predeclared exogenous storm-window restriction is evidence that this public design has little finite-sample room for the second flow under the benchmark.
- Neither statement alone proves that the two-process state equation is wrong, that conditional interruption is absent, or that an earlier forecast difference was caused by parameterization.
