# Code and algorithm audit

## 1. The confirmatory comparator is not initialized by the comparator's own flow equation

The reusable calibration function already distinguishes the model classes. In `src/asymode/dynamics.py`, calling

```python
calibrate_init(y, mask, InflowForm.NET_SCALED)
```

matches the initial signed **flow** as far as the one-flow class permits. The earlier `exp05_real_dynamics.py` correctly calls `calibrate_init(..., arm.inflow)` separately for every arm.

The confirmatory harness regresses from that rule:

```python
u0, r0 = calibrate_init(ytr, mtr, InflowForm.SUSCEPTIBLE)
model = make_model(arm, ..., u0, r0)
```

and its factory initializes `net_scaled` with

```python
u_init = u0 - r0
```

This is invalid. `u0` multiplies the served pool `(1-y)`, while `r0` multiplies the interrupted pool `y`. They are rates on different exposures, not two directly subtractable flow magnitudes.

The committed result records make the scale of the defect visible. Across outer folds,

```text
u0 ≈ 0.00083 -- 0.00097
r0 ≈ 0.091   -- 0.107
```

so the signed head is actually initialized around

```text
n0 = u0-r0 ≈ -0.090 -- -0.107.
```

The observed transition scale that produced `u0` is approximately `1e-3`, so the signed comparator begins roughly two orders of magnitude away from a neutral net-flow initialization and on the restoration side in every fold.

### Consequence at the zero boundary

For `net_scaled`, a negative signed output updates

\[
y_{t+1}=y_t+n_t y_t.
\]

At `y_t=0`, both the state change and its gradient with respect to `n_t` are zero. This is true even without the numerical clamp. Therefore every zero-origin sample is locally silent until other samples move the signed head across zero. In contrast, the two-flow arm has a positive sigmoid interruption head and immediately produces positive inflow at `y=0`.

The observed 41--43% exact-zero predictions from `net_scaled` are therefore not surprising. The current contrast combines the scientific restriction with a severe initialization asymmetry.

### Required correction

Do not initialize either model by moment-matching positive and negative increments separately. Fit the exact constant versions of the two statistical classes on the training transitions:

\[
(\widehat U_0,\widehat R_0)
=\arg\min_{U,R\ge0}\sum_i
\{\Delta y_i-U(1-y_i)+Ry_i\}^2,
\]

and

\[
\widehat n_0
=\arg\min_{n\in\mathbb R}\sum_i
\{\Delta y_i-[n]_+(1-y_i)+[-n]_+y_i\}^2.
\]

The second problem is the minimum of two one-dimensional nonnegative least-squares fits. This gives a flow-scale initialization under the exact class used in the paper.

## 2. Parameter-count matching is not architecture matching

The current two-flow arm consists of two independent depth-two width-32 MLPs. The one-flow arm is one width-48 MLP. Their parameter counts are close, but their representation learning, optimization geometry, and implicit regularization are not. A result can therefore reflect independent feature extraction versus a shared function, not only one versus two flows.

### Required replacement: nested shared-backbone parameterization

Use one shared encoder `h_theta(context_t)` and decompose the rates as

\[
U_t=[s_t]_+ + c_t,\qquad
R_t=[-s_t]_+ + c_t,
\]

where `s_t` is a signed direction head and `c_t≥0` is a concurrency head. This is not an approximation: for every `U,R≥0`,

\[
s=U-R,\qquad c=\min\{U,R\}
\]

recovers the same rates. The one-flow model is the exact nested submodel `c_t≡0`; the two-flow model frees only the concurrency head. Both arms share the same encoder and signed head.

With a common cap `C`, one bounded implementation is

\[
s_t=C\tanh(a_s^\top h_t+b_s),
\]

\[
c_t=(C-|s_t|)\sigma(a_c^\top h_t+b_c).
\]

Then `0≤U_t,R_t≤C`, and `U_t+R_t≤2C`. Taking `C≤1/2` preserves the state interval without an active clamp. The one-flow arm sets `c_t=0` but otherwise uses the identical computation graph.

This parameterization directly tests the paper's question: what is the value of freeing the second, concurrent component?

## 3. The current state clamp is not the main boundary pathology

The code repeatedly attributes one-flow collapse to clipping. For the configured bounded rates, however, the two-flow map

\[
y'=U+(1-U-R)y
\]

already maps `[0,1]` into `[0,1]` whenever `U,R≥0` and `U+R≤1`. The scaled signed map also remains in `[0,1]` for a signed rate in `[-1,1]`. At `y=0`, a negative one-flow rate gives exactly zero before clipping.

The audit should therefore distinguish:

- **numerical clamp activation**, which can be measured directly;
- **structural boundary silence**, caused by multiplying a negative direction by `y=0`;
- **initialization bias**, which places the signed model on that silent branch.

Calling all three “clip collapse” obscures the mechanism.

## 4. Checkpoint selection is unstable and omits the strongest baseline: the initialization

The confirmatory harness validates on one cyclic chronological event. Several seeds select epochs 1--2. The alternative “event-balanced” implementation performs only one optimizer update per training event per epoch and also selects many epoch-1 checkpoints. Neither harness compares trained checkpoints with epoch 0.

### Required optimizer protocol

Define budget in optimizer updates, not epochs:

```text
pretrain_updates = 2,000
rollout_updates  = 3,000
validation_every = 250 updates
minimum_updates_before_stopping = 1,000
```

At every update:

1. sample a source event uniformly;
2. sample one minibatch of county-origin rows within that event;
3. take one optimizer step.

This estimates an equal-event objective without reducing an epoch to eight or nine total updates. Record gradient updates, examples processed, and wall time for every arm.

Checkpoint candidates must include:

- the calibrated initialization (`update=0`);
- every validation checkpoint;
- an average of the best three adjacent checkpoints as a prespecified stability control.

The validation score must average event-level losses over county-held-out validation rows from **all** source events. No single storm should decide when training stops.

## 5. Rollout-only training is inherited from the competition objective, not from the theory

The paper's exact results concern the conditional transition mean. The real model is trained only through 24--48 recursive steps. This makes gradients depend on the entire simulated path, amplifies state-distribution imbalance, and permits a good trajectory fit with poorly estimated rates. The committed audit already shows that a no-weather affine one-step map beats every neural arm on one-step RMSE.

Use a fixed two-stage objective rather than a weight sweep:

1. **teacher-forced transition pretraining** on observed adjacent states;
2. **24-hour rollout fine-tuning** from the forecast origin.

Both one-flow and two-flow arms receive the identical schedule. One-step MSE is the theorem-aligned endpoint; rollout MSE measures whether the local advantage propagates.

## 6. The model genealogy contains competition-specific complexity that should not enter the paper implementation

`src/asymode/dynamics.py` still exposes epidemic arms, seeded epidemic arms, independent channel subsets, unequal capacities, state-fed rates, a multiplicative gate, a background channel, and multiple initialization rules. The competition repository adds running maxima, handcrafted hazard banks, diurnal gates, rate-space/logit-space switches, and many warm-started pathways.

These options are useful historical records but dangerous as a paper code path. They make it easy to change several scientific axes at once and difficult to verify which implementation actually produced a table.

### Required code organization

Create a clean paper-only module with exactly:

```text
SharedContextEncoder
NestedOneFlowModel      # c == 0
NestedTwoFlowModel      # c learned
constant_class_fits     # exact initialization
train_transition_then_rollout
```

The existing variant zoo should remain untouched as historical code. The paper experiment should import only the clean module.

## 7. Missing temporal context is the leading model-form failure

The current rate heads see only the weather vector at the current forecast step. The error audit attributes most long-horizon SSE to near-zero origins that later cross a material outage threshold, and reports that the total onset amount is approximately right while timing is wrong. A memoryless map `U(x_t)` cannot distinguish a brief gust spike from accumulated exposure, nor a first pulse from a delayed second pulse.

After the code/protocol repair gate, the only authorized architecture extension should be one shared temporal encoder over exogenous weather:

\[
h_t=\operatorname{GRU}_\theta(x_{t-L:t}),\qquad L=24\text{ hours},
\]

followed by the same signed and concurrency heads. The encoder must not read the current state `y_t`; otherwise the conditional mean is no longer affine in the state and the main theory no longer applies without modification. Past weather, clock, forecast weather, and fixed county covariates may be included in the context.

This is one controlled response to the diagnosed onset-timing error, not another architecture sweep.
