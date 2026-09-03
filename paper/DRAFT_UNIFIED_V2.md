# Separating Opposing Flows for Event-Robust Neural Dynamics

## Abstract

Many bounded dynamical systems evolve through two opposing flows, yet forecasting models often collapse their net effect into one signed update. We study when that collapse is statistically consequential. For a state transition

\[
Y_{t+1}-Y_t=U(X_t)(1-Y_t)-R(X_t)Y_t+\varepsilon_t,
\]

we first characterize local identification of the two conditional rate functions. The conditional design determinant equals \(\operatorname{Var}(Y\mid X=x)\), and its smallest eigenvalue is within a factor of two of that variance. Our main result concerns transfer across environments. The best one-signed-rate approximation is an environment-dependent \(L_2\) projection. When a projection learned under source-event distribution \(P\) is applied to target event \(Q\), its risk decomposes exactly into the target oracle approximation error and a nonnegative projection-shift penalty determined by changes in conditional state moments. A minimax corollary shows that no fixed collapsed rate can be simultaneously oracle across two events whose projection ratios differ. These results motivate a deliberately simple two-rate neural dynamical model rather than a larger black-box forecaster. We evaluate it on eleven public county-level power-outage events under leave-one-event-out transfer. The corrected evidence shows a stable advantage over a parameter-matched signed-rate network at six hours and a positive but heterogeneous average advantage at twenty-four hours, while no universal long-horizon dominance is claimed. The paper thus separates three questions that are often conflated: whether opposing rates are identifiable, whether collapsing them creates approximation and transfer error, and whether that structural benefit exceeds finite-sample estimation and misspecification costs.

## 1. Introduction

A single observed state can rise and fall because two processes operate simultaneously. Customers lose service while other customers are restored; individuals enter and leave a compartment; inventory arrives while demand depletes stock. A common predictive simplification replaces those opposing flows by one signed update. Such a model is parsimonious, but its signed rate is not generally a structural object. It is the best projection of two nonnegative flows under the state distribution encountered during training.

This distinction matters under event transfer. Within one event, a flexible signed-rate model may learn the net change associated with that event's characteristic state distribution. When the entire event is unseen, the same driver values can occur at different state levels and phases. The source-optimal signed projection can then cease to be target optimal even if the underlying opposing-flow functions remain stable.

Power-outage evolution provides a demanding test bed. A county-level outage fraction is bounded, highly concentrated near zero, and driven by concurrent interruption and restoration. Public data contain many counties under related weather but relatively few independent storms. This setting makes both the value and the cost of structural decomposition visible: two rates can remove an event-dependent projection error, but the restoration rate is weakly identified in low-outage regimes and can be expensive to estimate.

We make three tightly connected contributions.

1. **Event-dependent projection theory.** We derive the exact source-to-target risk of collapsing two nonnegative rates into either one interruption-only or one restoration-only signed branch. The target risk equals an irreducible target approximation gap plus a nonnegative event-projection-shift term. A minimax corollary gives the unavoidable worst-event projection regret of any fixed collapsed coefficient.
2. **Identification geometry as a prerequisite, not a performance theorem.** We show that the determinant of the conditional Gram matrix equals conditional state variance and that its smallest eigenvalue is within a factor of two of that variance. This establishes when the two functions are recoverable and explains why accurate state prediction need not imply accurate rate recovery.
3. **A single theory-aligned neural evaluation.** We retain a simple two-rate neural state equation and compare it with a parameter-matched one-signed-rate network, gradient boosting on the same information, and damped persistence. The primary protocol holds out one complete event, balances the training objective across source events, uses two outcome-blind validation events, and trains on both rollout and teacher-forced conditional-transition error. The one-rate collapse and removal of the one-step term are the only model ablations.

The intended claim is not that mechanistic structure always beats flexible prediction. Our theory explicitly includes the opposing force: the structural approximation and shift benefit must exceed additional estimation error and any relative misspecification of the two-rate model.

## 2. Problem setup

Let \(Y_t\in[0,1]\) denote a bounded state and \(X_t\in\mathcal X\) an exogenous driver vector. The conditional transition mean is

\[
m(x,y)=\mathbb E[Y_{t+1}-Y_t\mid X_t=x,Y_t=y]
       =U_0(x)(1-y)-R_0(x)y,
\]

where \(U_0,R_0\ge0\). In the outage application, \(U_0\) is an interruption transition function and \(R_0\) a restoration transition function. We use “rate” as shorthand for a one-step conditional transition component; causal interpretation would require assumptions not made here.

Define

\[
\phi(y)=\begin{bmatrix}1-y\\-y\end{bmatrix},
\qquad
\beta_0(x)=\begin{bmatrix}U_0(x)\\R_0(x)\end{bmatrix},
\]

so \(m(x,y)=\phi(y)^\top\beta_0(x)\).

A collapsed signed-rate model retains the same state-dependent scaling but permits only one direction at a time. At fixed \(x\), its coefficient class is

\[
\mathcal B_x=\{(a,0):a\ge0\}\cup\{(0,b):b\ge0\}.
\]

This is the population analogue of the parameter-matched `net_scaled` neural comparator.

We index event environments by \(e\). Conditional expectations under event \(e\) are denoted \(\mathbb E_e[\cdot\mid X=x]\). Define

\[
A_e(x)=\mathbb E_e[(1-Y)^2\mid X=x],
\quad
B_e(x)=\mathbb E_e[Y^2\mid X=x],
\quad
C_e(x)=\mathbb E_e[Y(1-Y)\mid X=x],
\]

and \(v_e(x)=\operatorname{Var}_e(Y\mid X=x)\).

## 3. Identification geometry

### Theorem 1: local functional identification

Let

\[
Q(x)=\mathbb E[\phi(Y)\phi(Y)^\top\mid X=x].
\]

Then

\[
\det Q(x)=\operatorname{Var}(Y\mid X=x)=v(x),
\]

and

\[
v(x)\le\lambda_{\min}(Q(x))\le2v(x),
\qquad
\frac12\le\lambda_{\max}(Q(x))\le1.
\]

Consequently, \(U_0(x)\) and \(R_0(x)\) are identified from the conditional transition mean if and only if \(v(x)>0\). If \(Y\mid X=x\) is degenerate at \(\mu(x)\), only

\[
(1-\mu(x))U_0(x)-\mu(x)R_0(x)
\]

is identified.

#### Proof sketch

Writing \(\mu=\mathbb E[Y\mid X=x]\) and \(s=\mathbb E[Y^2\mid X=x]\),

\[
Q=
\begin{bmatrix}
1-2\mu+s & -(\mu-s)\\
-(\mu-s) & s
\end{bmatrix}.
\]

Direct expansion gives \(\det Q=s-\mu^2=v\). Moreover,

\[
\operatorname{tr}Q=1-2\mu+2s
                =\mu^2+(1-\mu)^2+2v\in[1/2,1].
\]

Since \(\lambda_{\min}=\det Q/\lambda_{\max}\), the stated bounds follow from \(\lambda_{\max}\in[1/2,1]\). Full boundary details are deferred to the appendix.

### Corollary 1: prediction can be accurate while rates are not

For any candidate \(\beta\), squared one-step excess risk satisfies

\[
\mathcal R(\beta)-\mathcal R(\beta_0)
=
\mathbb E[(\beta(X)-\beta_0(X))^\top Q(X)(\beta(X)-\beta_0(X))].
\]

Thus the weak rate direction receives weight of order \(v(X)\). When conditional state variance is small, substantially different rate pairs can induce nearly identical conditional predictions. Identification is therefore a prerequisite for interpreting the rates, not a guarantee of lower forecast error.

## 4. Collapsed rates are environment-dependent projections

### Theorem 2: source-to-target projection decomposition

Fix \(x\) and first consider the unconstrained interruption-only class \(a(1-y)\). Under source environment \(P\), its population minimizer is

\[
a_P^*(x)=U_0(x)-R_0(x)\frac{C_P(x)}{A_P(x)}.
\]

When this source projection is evaluated under target environment \(Q\),

\[
\boxed{
\begin{aligned}
&\mathbb E_Q[(m(x,Y)-a_P^*(x)(1-Y))^2\mid X=x]\\
&=R_0(x)^2\frac{v_Q(x)}{A_Q(x)}
+R_0(x)^2A_Q(x)
\left(
\frac{C_P(x)}{A_P(x)}-\frac{C_Q(x)}{A_Q(x)}
\right)^2.
\end{aligned}}
\]

The first term is the target oracle interruption-only approximation error. The second is the event-projection-shift penalty.

For the restoration-only class \(-b y\),

\[
b_P^*(x)=R_0(x)-U_0(x)\frac{C_P(x)}{B_P(x)},
\]

and

\[
\boxed{
\begin{aligned}
&\mathbb E_Q[(m(x,Y)+b_P^*(x)Y)^2\mid X=x]\\
&=U_0(x)^2\frac{v_Q(x)}{B_Q(x)}
+U_0(x)^2B_Q(x)
\left(
\frac{C_P(x)}{B_P(x)}-\frac{C_Q(x)}{B_Q(x)}
\right)^2.
\end{aligned}}
\]

Nonnegative clipping of \(a_P^*\) or \(b_P^*\) produces the corresponding piecewise boundary result.

#### Proof

For the interruption branch,

\[
m(x,Y)-a(1-Y)
=(U_0-a)(1-Y)-R_0Y.
\]

Differentiating its source risk gives

\[
A_P(a-U_0)+R_0C_P=0,
\]

which yields \(a_P^*\). Under \(Q\), complete the square around
\(a_Q^*=U_0-R_0C_Q/A_Q\):

\[
R_Q(a)=R_Q(a_Q^*)+A_Q(a-a_Q^*)^2.
\]

The identity \(A_QB_Q-C_Q^2=v_Q\) gives
\(R_Q(a_Q^*)=R_0^2v_Q/A_Q\), and substitution of \(a_P^*-a_Q^*\) gives the result. The restoration branch is symmetric.

### Corollary 2: minimax event regret of a fixed collapsed rate

For two event environments \(P,Q\), any fixed interruption-only coefficient has worst-event excess projection risk at least

\[
\boxed{
\inf_a\max_{e\in\{P,Q\}}
\{R_e(a)-R_e(a_e^*)\}
=
\frac{A_PA_Q}{(\sqrt{A_P}+\sqrt{A_Q})^2}
R_0^2
\left(
\frac{C_P}{A_P}-\frac{C_Q}{A_Q}
\right)^2.
}
\]

This quantity is strictly positive whenever \(R_0>0\) and the event projection ratios differ. The symmetric restoration expression replaces \((A,R_0,C/A)\) by \((B,U_0,C/B)\).

This corollary is the paper's central robustness statement. It does not assert that a finite-sample two-rate neural estimator must outperform every collapsed estimator. It states that collapse introduces an environment-dependent approximation target that cannot be simultaneously oracle across sufficiently different events.

## 5. Representation, estimation, and misspecification

Let \(\widehat m_2\) and \(\widehat m_1\) be fitted two-rate and one-rate predictors. Their target risk difference can be organized as

\[
\begin{aligned}
\mathcal R_Q(\widehat m_1)-\mathcal R_Q(\widehat m_2)
={}&G_Q+S_{P\to Q}+M_{P\to Q}\\
&-\{E_{2,Q}-E_{1,Q}\}
-\{B_{2,Q}-B_{1,Q}\},
\end{aligned}
\]

where \(G_Q\) is the target oracle collapse gap, \(S_{P\to Q}\) the projection-shift penalty, \(M_{P\to Q}\) a possible wrong-branch penalty, \(E\) finite-sample estimation error, and \(B\) relative model misspecification. This is a conceptual decomposition rather than an estimable equality without further assumptions.

The two-rate model is favored when its structural benefit exceeds the extra cost of estimating two functions. This explicitly permits null or negative events, especially where restoration is weakly identified or where event-specific omitted variables violate structural rate invariance.

## 6. Neural two-rate dynamics

The proposed model is intentionally unchanged and low capacity:

\[
U_\theta(x)=c_U\sigma(f_U(x;\theta_U)),
\qquad
R_\theta(x)=c_R\sigma(f_R(x;\theta_R)),
\]

\[
\widehat Y_{t+1}
=
\widehat Y_t+U_\theta(X_t)(1-\widehat Y_t)-R_\theta(X_t)\widehat Y_t.
\]

Both \(f_U\) and \(f_R\) are two-layer ReLU networks of width 32. The comparator `net_scaled` uses one width-48 signed network and is parameter matched within one percent. Positive signed flow acts on \(1-Y\) and negative signed flow on \(Y\); the only intended structural difference is whether interruption and restoration may be simultaneously active.

No semiparametric component, event embedding, recurrent state, damage gate, or recovery-memory module is used in the present paper version.

## 7. Theory-aligned training

The primary task is a 24-hour event-conditioned forecast. Six and twenty-four hours are the primary endpoints. The training objective is equal across source events:

\[
L_{\mathrm{train}}
=
\frac1{|\mathcal E_{\mathrm{train}}|}
\sum_e
\left[
\frac12L_{\mathrm{roll},e}^{1:24}
+
\frac12L_{\mathrm{step},e}^{1:24}
\right].
\]

The rollout term trains the deployed trajectory. The teacher-forced term directly estimates the conditional transition appearing in Theorems 1 and 2. Its weight is fixed rather than tuned.

Each outer fold holds out one complete event. Two validation events are selected using weather, footprint, and observation coverage only; the remaining eight events train the model. Checkpoints minimize

\[
L_{\mathrm{val}}
=
\frac1{2}\sum_{e\in\mathcal E_{\mathrm{val}}}
\left[
\frac12L_{\mathrm{path24},e}
+
\frac14L_{6,e}
+
\frac14L_{24,e}
\right].
\]

This replaces the previous unstable practice of early stopping on one chronological-neighbor event.

## 8. Data

The primary cohort consists of eleven public county-level severe-weather event panels from 2021, 2022, and 2024. Each panel contains county outage fractions, an explicit observation mask, and twelve hourly ERA5 weather channels; two UTC clock channels are added from timestamps. Unobserved cells are excluded from every loss and metric.

All eleven events remain in the final evaluation. A physical-feature audit identifies 2021-06-21 as the central event and 2021-05-04, 2021-12-11, and 2022-07-23 as three cluster medoids. These labels are used only for implementation smoke tests and outcome-blind validation design, never to remove unfavorable test events.

## 9. Experiments

### 9.1 Main experiment

The main table contains four methods:

- two-rate neural dynamics;
- parameter-matched `net_scaled`;
- histogram gradient boosting on the same available information;
- damped persistence.

The statistical unit is the held-out event. Neural seeds are averaged within each event. We report equal-event RMSE at h+6 and h+24, event-level paired gains, event bootstrap intervals, exact label-swap randomization, sign counts, and leave-one-event influence.

### 9.2 Ablations

There are only two ablations.

1. **Structural collapse:** two-rate versus `net_scaled`.
2. **Objective alignment:** the identical two-rate architecture trained without the teacher-forced one-step term.

No architecture sweep is part of the paper.

### 9.3 Corrected evidence preceding the v2 protocol

Under the corrected eleven-event leave-one-event-out run with a single validation event and pooled rollout loss, two-rate versus `net_scaled` had:

- h+6: +2.94% equal-event mean, 10/11 events, event-bootstrap interval above zero;
- h+24: +4.55% mean, 8/11 events, bootstrap interval [+0.74%, +9.25%], exact randomization p=0.041;
- h+48: +3.71% mean, 6/11 events, interval including zero.

The h+24 effect was positive after removing any one event, but it failed an earlier preregistered 9/11 sign threshold. We therefore describe it as a positive heterogeneous average effect rather than a universal event-level advantage. The claim that event holdout strengthens the advantage relative to county holdout was not confirmed by a direct paired test.

These corrected results motivate, but do not substitute for, the unified v2 experiment. The v2 protocol changes the forecast scope, event weighting, transition objective, and validation design as a single prespecified package and reports the resulting comparison without further selection.

## 10. Limitations

First, the rate functions are conditional predictive functions, not causal hazards. Event-specific infrastructure damage, utility operations, and crew deployment are not fully observed, so structural invariance can fail. Second, the event count is small even though county-hour observations are numerous; inference must remain event based. Third, the public source omits explicit zero records, making the observation mask essential. Fourth, the present model assumes perfect future weather inputs. Fifth, the 24-hour scope does not resolve long-tail restoration; the earlier 48-hour null result remains an important boundary of the method.

## 11. Conclusion

Separating opposing flows has two distinct statistical requirements. Conditional state variation is needed to identify both rate functions, while event variation can make a collapsed signed rate transfer poorly because its population target changes with the event-specific state distribution. A two-rate model removes that projection dependence under structural invariance, but its empirical benefit is neither automatic nor universal. The proposed evaluation is designed to measure exactly this tradeoff with one simple neural architecture, one complete event-held-out experiment, and two theory-matched ablations.
