# Binding implementation amendment

This file resolves three details left open in `09_LOCKED_CC_PLAN_V2.md` and `10_CC_EXECUTION_PROMPT_V2.md`. Where wording conflicts, this amendment controls.

## 1. Fixed hourly caps

Do not choose caps after inspecting pilot accuracy. Use the existing paper-scale bounds:

```text
C_U_main = 0.25
C_U_background = 0.015
C_R = 0.25
```

Therefore

```text
U_tilde <= 0.265
R_tilde <= 0.25
U_tilde + R_tilde <= 0.515 < 1
```

and the affine state map preserves `[0,1]` without an active numerical clamp. These caps are shared across every event, fold, start, and arm. The competition value `0.5` is not transplanted because the open-data hourly units and target construction differ.

## 2. Mapping an exact constant fit into the modular interruption architecture

The constant-class solver produces a desired interruption flow `U0` and recovery flow `R0`. Initialize the modular network so its update-0 outputs reproduce these constants rather than merely initializing individual logits.

Set all output weights to zero. Fix the occurrence gate at

```text
g0 = 0.5
```

by zero weights and zero bias. Initialize the hold gate with zero weights and bias `-3.0`; because the raw logit is constant at update 0, the hold leaves that constant unchanged.

Split the desired interruption flow deterministically in proportion to the two pathway caps:

```text
b0 = U0 * C_U_background / (C_U_main + C_U_background)
p0 = U0 - b0
```

Then set:

```text
background_bias = logit(b0 / C_U_background)
raw_U_bias      = logit(p0 / (g0 * C_U_main))
```

and use the same `raw_U_bias` for both interruption MLP heads. Clip only the probability arguments to `[1e-8, 1-1e-8]` for numerical initialization; record the resulting approximation error.

Set:

```text
recovery_bias = logit(R0 / C_R)
```

with the same numerical clipping rule. An update-0 model passes only if its maximum absolute flow error relative to the constant-class solution is at most `1e-6`, or the smallest machine-precision error permitted by the clipped zero branch, whichever is larger and explicitly reported.

For the two fixed one-flow starts:

```text
interruption start: U0 = a0, R0 = 0
restoration start:  U0 = 0,  R0 = b0
```

where `a0` and `b0` are the exact ray optima. Do not initialize a signed quantity with `U0-R0` from the two-flow fit.

## 3. State-stratified sampling must not silently change the objective

Stage-A minibatches may oversample the active-state pool to avoid batches containing almost no recovery information, but the optimized loss must remain the natural equal-event transition risk.

Within each source event, let `pi_z` be the sampling probability for state stratum `z`. Weight each sampled transition by the inverse of its sampling probability and normalize within event. Verify on a fixed model that the Monte Carlo weighted minibatch loss agrees with the exact full-event loss within sampling error.

Report two evaluation quantities:

1. **natural event risk**—the prespecified primary risk under the observed event distribution;
2. **equal-stratum risk**—a diagnostic average over the available state strata.

The equal-stratum diagnostic may explain performance but may not replace the natural primary endpoint after results are seen.

## 4. No claim change from this amendment

These choices are implementation controls. They do not declare the earlier result correct or incorrect, and they do not alter the theory. Their purpose is to prevent cap choice, modular bias decomposition, or state oversampling from becoming alternative explanations for the new comparison.
