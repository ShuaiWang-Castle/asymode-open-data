# Core theory v2: environment-dependent collapse of opposing flows

This note fixes the theory hierarchy for the AISTATS paper. Only Theorem 2 and its minimax corollary are the headline theoretical contribution. Identification geometry is a supporting condition. Rollout stability is an appendix result.

## 1. Model

At a fixed driver value `x`, let

\[
m_x(y)=U(x)(1-y)-R(x)y,
\qquad U(x),R(x)\ge0.
\]

For event environment \(e\), define

\[
A_e=\mathbb E_e[(1-Y)^2\mid X=x],\quad
B_e=\mathbb E_e[Y^2\mid X=x],\quad
C_e=\mathbb E_e[Y(1-Y)\mid X=x],
\]

\[
v_e=\operatorname{Var}_e(Y\mid X=x)=A_eB_e-C_e^2.
\]

## 2. Supporting identification theorem

Let

\[
\phi(Y)=(1-Y,-Y)^\top,
\qquad
Q_e=\mathbb E_e[\phi(Y)\phi(Y)^\top\mid X=x].
\]

Then

\[
\det Q_e=v_e,
\qquad
v_e\le\lambda_{\min}(Q_e)\le2v_e.
\]

Thus the two conditional functions are identified at \(x\) exactly when \(v_e>0\). This theorem does not predict which event yields a forecasting gain. Its experimental role is limited to:

- validating the algebra on synthetic data;
- explaining why rate recovery becomes unstable near a degenerate state distribution;
- justifying a teacher-forced conditional-transition term in the training objective.

## 3. Headline theorem: event-dependent one-rate projection

### 3.1 Interruption-only branch

The source-event population projection is

\[
a_P^*=U-R\frac{C_P}{A_P}.
\]

For a target event \(Q\),

\[
\mathcal R_Q^+(a_P^*)
=R^2\frac{v_Q}{A_Q}
+R^2A_Q
\left(\frac{C_P}{A_P}-\frac{C_Q}{A_Q}\right)^2.
\]

The two terms are:

\[
G_Q^+=R^2\frac{v_Q}{A_Q}
\]

(target-event oracle approximation gap) and

\[
S_{P\to Q}^+=R^2A_Q
\left(\frac{C_P}{A_P}-\frac{C_Q}{A_Q}\right)^2
\]

(event-projection-shift penalty).

### 3.2 Restoration-only branch

Similarly,

\[
b_P^*=R-U\frac{C_P}{B_P},
\]

\[
\mathcal R_Q^-(b_P^*)
=U^2\frac{v_Q}{B_Q}
+U^2B_Q
\left(\frac{C_P}{B_P}-\frac{C_Q}{B_Q}\right)^2.
\]

### 3.3 Nonnegative boundary

The constrained coefficients are

\[
a_P^+=\max(0,a_P^*),
\qquad
b_P^+=\max(0,b_P^*).
\]

When the unconstrained optimum is nonnegative, the identities above are exact. When it is negative, the constrained optimum is the boundary zero and the additional boundary term is reported explicitly. No theorem statement may silently ignore this case.

## 4. Minimax corollary

For two events \(P,Q\), the event-specific interruption-only risks satisfy

\[
\mathcal R_e^+(a)-\mathcal R_e^+(a_e^*)=A_e(a-a_e^*)^2.
\]

Balancing the two weighted squared distances gives

\[
\inf_a\max_{e\in\{P,Q\}}
\{\mathcal R_e^+(a)-\mathcal R_e^+(a_e^*)\}
=
\frac{A_PA_Q}{(\sqrt{A_P}+\sqrt{A_Q})^2}(a_P^*-a_Q^*)^2.
\]

Since

\[
a_P^*-a_Q^*=-R\left(\frac{C_P}{A_P}-\frac{C_Q}{A_Q}\right),
\]

we obtain

\[
\boxed{
\inf_a\max_e
\{\mathcal R_e^+(a)-\mathcal R_e^+(a_e^*)\}
=
\frac{A_PA_Q}{(\sqrt{A_P}+\sqrt{A_Q})^2}
R^2
\left(\frac{C_P}{A_P}-\frac{C_Q}{A_Q}\right)^2.
}
\]

A fixed collapsed interruption coefficient therefore has strictly positive worst-event projection regret whenever restoration is active and the two event projection ratios differ. The restoration-only expression is symmetric.

This is a representation/transfer statement, not a finite-sample forecasting dominance theorem.

## 5. Finite-sample predictive comparison

The empirical risk difference can be organized conceptually as

\[
\mathcal R_Q(\widehat m_1)-\mathcal R_Q(\widehat m_2)
=G_Q+S_{P\to Q}+M_{P\to Q}
-(E_{2,Q}-E_{1,Q})-(B_{2,Q}-B_{1,Q}),
\]

where:

- \(G_Q\): target oracle collapse gap;
- \(S_{P\to Q}\): event projection shift;
- \(M_{P\to Q}\): wrong signed-branch selection penalty;
- \(E_{k,Q}\): finite-sample estimation error;
- \(B_{k,Q}\): relative misspecification error.

The theorem predicts a structural cost of collapse. It deliberately permits the two-rate estimator to lose when its extra estimation or misspecification cost is larger.

## 6. Exact mapping from theory to experiments

There is one main experiment and two ablations.

### Main experiment

Eleven leave-one-event-out folds; proposed two-rate NN versus parameter-matched `net_scaled`, HGB on the same information, and damped persistence. Primary endpoints h+6 and h+24.

### Structural ablation

Two-rate versus `net_scaled` estimates the finite-sample consequence of removing concurrent opposing flows. This is the only ablation that tests the headline theorem's model class.

### Objective ablation

Joint rollout/teacher-forced training versus rollout-only training tests whether optimizing the conditional transition risk appearing in the theorems stabilizes the two-rate estimator.

### Synthetic theorem figure

A fixed two-point state distribution varies \(v\) and the source/target projection ratio. It verifies:

1. \(\det Q=v\);
2. rate MSE grows as the weak information direction collapses;
3. the exact source-to-target projection identity;
4. the minimax worst-event regret formula.

No family-ordering, phase-ratio, recovery-memory, damage-gate, or semiparametric experiment is needed to support the core theorem.

## 7. Claims explicitly excluded

The paper does not claim:

- local identifiability orders empirical event gains;
- event holdout necessarily increases the two-rate advantage;
- the two-rate model dominates HGB or persistence at every horizon;
- learned rates are causal hazards;
- long-horizon errors cannot accumulate;
- a second damage wave requires a second damage network.
