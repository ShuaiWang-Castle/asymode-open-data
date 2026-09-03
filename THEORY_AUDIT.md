# Theory audit: one signed flow versus two opposing flows

## Verdict

**Soundness: PASS, with explicitly stated scope.** The theory in `paper/aistats/main.tex` is mathematically correct after the boundary and scope corrections recorded below. It is suitable as the theoretical component of an AISTATS paper. It is not, by itself, a competitive pure-theory paper: the contribution is exact and useful but algebraically elementary, so the locked neural comparison and the single controlled phase diagram remain necessary for significance.

## Submission-grade results

1. **Exact constrained oracle gap.** For the union of the nonnegative damage-only and restoration-only rays,
   \[
   G(x)=v(x)\min\{R_0(x)^2/A(x),D_0(x)^2/B(x)\},
   \]
   with all nonnegative-boundary and degenerate-state cases handled. The gap is strictly positive exactly when conditional state variation is nonzero and both physical flows are active.
2. **Integrated population result.** The pointwise gap integrates to the optimal risk over all measurable signed one-flow predictors. This does not claim that a finite neural network attains the measurable oracle.
3. **Identification geometry.** The conditional Gram matrix obeys `det Q = Var(Y|X=x)`, `v <= lambda_min(Q) <= 2v`, and `1/2 <= lambda_max(Q) <= 1`. The statement is conditional-function identification, not neural-weight identification and not a nonparametric convergence theorem.
4. **Finite-sample information.** The Cauchy--Binet determinant identity, Gaussian Fisher determinant, and OLS covariance formulas are exact. The displayed ratio is correctly labeled a variance ratio rather than a precision ratio.
5. **Oracle-span bias--variance benchmark.** In the fixed-design Gaussian problem, the relaxed rank-one and rank-two fitted-mean risks and their chi-square risk-difference distribution are exact. The `Gamma_n=1` threshold is explicitly restricted to this oracle-selected, span-relaxed benchmark.
6. **Nonnegative-ray risk.** The appendix gives the exact truncated-normal risk for an arbitrary signal projected onto a fixed nonnegative ray, including active-boundary behavior.
7. **Event-shift extension.** The source-to-target projection identity is correct under a named structural-invariance assumption, and a boundary-safe version is included.
8. **State preservation.** The neural transition remains in `[0,1]` whenever the two per-step rate caps sum to at most one.

## Corrections relative to earlier drafts

- The old rollout statement claiming that long-horizon error cannot accumulate was removed; it is false without a uniform positive lower bound on total transition intensity.
- The Fisher determinant now includes the factor `sigma^{-4}`.
- The restoration/interruption uncertainty ratio is called a variance ratio, not a precision ratio.
- Conditional identifiability is no longer presented as an explanation of event-family forecasting rankings.
- Event shift is an extension, not the defining theorem and not a guarantee that event holdout improves two-flow performance.
- The `Gamma_n=1` crossover is no longer presented as an exact neural-network model-selection rule.
- The event-shift result now states the invariance assumption needed to hold the two structural functions fixed across environments.

## Remaining scientific dependency

The theory answers three distinct questions:

- **necessity:** does collapsing to one flow create oracle approximation error?
- **learnability:** can damage and restoration be separated from the conditional data distribution?
- **finite-sample usefulness:** does the representation benefit exceed the additional effective estimation and misspecification cost?

AISTATS-level significance therefore still depends on one main event-held-out experiment showing where the two-flow neural inductive bias helps or fails, without removing unfavorable events or hiding the strong black-box baseline.
