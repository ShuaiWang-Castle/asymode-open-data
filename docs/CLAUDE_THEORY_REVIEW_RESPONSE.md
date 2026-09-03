# Response to the independent theory review

## Verdict

The algebraic results are correct within their stated domains. The revision does not add more disconnected theorems. It closes the gap between:

1. pointwise population representation;
2. fixed-design one-step estimation; and
3. dynamic multi-step neural prediction.

The current paper is therefore framed around one question: **when is one signed, state-scaled flow enough, and when is it worth estimating separate damage and restoration flows?**

## Core hierarchy

### 1. Necessity: exact one-flow collapse gap

For

```text
m_x(y) = D(x)(1-y) - R(x)y,
```

the exact population error of the best nonnegative one-flow ray is

```text
G(x) = v(x) min{R(x)^2/A(x), D(x)^2/B(x)}.
```

This is the central theorem. It applies to the specific state-scaled signed comparator used in the paper, not to every possible scalar predictor `a(x)h(y)`.

### 2. Learnability: orthogonal net-drift/turnover coordinates

With

```text
alpha = (1-mu)D - mu R,
tau   = D + R,
```

we have

```text
m_x(y) = alpha(x) - tau(x)(y-mu(x)),
E[psi psi^T | x] = diag(1,v(x)).
```

The net-drift coordinate has unit information, while total turnover has information `v(x)`. This makes precise why good state prediction need not imply good recovery of damage and restoration separately.

In the interior fixed-design benchmark,

```text
Gamma_n = (n v_hat / sigma^2) * min{R^2/A_hat, D^2/B_hat}
        = turnover information * concurrency strength.
```

### 3. Finite-sample usefulness

The `Gamma=1` crossover is retained only as an oracle-selected, span-relaxed, fixed-design Gaussian benchmark. It is not an exact threshold for the bounded neural estimators or for the dynamic panel.

The user-provided balanced two-state solvable case is the sole synthetic experiment. It fixes `D`, `R`, `n`, and `sigma`, varies only state dispersion `delta`, and verifies the exact crossover.

### 4. One step to rollout

The false statement that long-horizon error cannot accumulate has been removed. The correct bridge is

```text
e_{t+1} = (1-tau_hat_t)e_t + delta_t,
e_H = sum_k delta_k product_{j>k}(1-tau_hat_j).
```

A one-step advantage may disappear, persist, or reverse under rollout because signed errors can cancel or reinforce along the realized path. The real experiment must therefore report both held-out teacher-forced transition error and held-out 24-hour rollout error.

### 5. Event transfer

Source-to-target projection shift and turnover-leverage inflation are retained only as appendix benchmarks. Event shift is not the paper's defining mechanism and does not guarantee that two flows win on every held-out event.

## Explicit scope limits

1. **Pooled affinity.** Unobserved county heterogeneity can make the pooled conditional mean nonlinear in the current state. Learned rates are predictive components, not automatically causal physical rates.
2. **Continuous covariates.** The fixed-`x` theory is a population conditional statement. Finite estimation shares information across nearby covariates through the neural parameterization.
3. **Dynamic design.** The current state is predetermined in the panel. Fixed-design OLS formulas are benchmarks rather than exact finite-sample formulas for recurrent data.
4. **Noise and state measurement.** EAGLE-I fractions are heteroskedastic and contain reporting, denominator, and state-measurement error. The real study therefore does not plug an estimated `Gamma` into the fixed-design threshold.
5. **One step versus rollout.** The population gap is a one-step conditional-risk statement. Multi-step performance is governed by the rollout recursion and must be tested separately.

## Consequence for the experiment

Both neural classes use the same event-balanced 24-hour rollout objective. Teacher-forced one-step MSE is evaluated after fitting and is not inserted into the training objective. This isolates the model class:

```text
one signed state-scaled flow
versus
two simultaneous nonnegative damage/restoration flows.
```

A predictive result is attributed to flow separation only when the two-flow model improves both the held-out transition endpoint and the held-out h+24 forecast endpoint.
