# Unified AISTATS v2 — start here

This branch now has one scientific center:

> **When is one signed flow sufficient, and when should prediction separate damage/interruption and restoration into two nonnegative opposing flows?**

Read in order:

1. `docs/CORE_THEORY_V2.md`
2. `docs/UNIFIED_AISTATS_V2_PROTOCOL.md`
3. `paper/DRAFT_UNIFIED_V2.md`
4. `experiments/unified_aistats_v2.py`
5. `experiments/unified_aistats_v2_baselines.py`

## Locked hierarchy

1. **Core representation theorem:** the best one-flow model has exact oracle gap

   \[
   G(x)=v(x)\min\{R(x)^2/A(x),D(x)^2/B(x)\}.
   \]

   One flow is sufficient exactly when one physical flow is absent or the conditional state is degenerate.

2. **Core estimation theorem:** in the local fixed-design Gaussian benchmark, two flows are worth estimating when

   \[
   \Gamma_n=nG_n/\sigma^2>1.
   \]

3. **Supporting identification result:** conditional state variance controls whether the two functions can be separated.

4. **Event shift:** a secondary corollary that adds a nonnegative projection-shift penalty to one flow. It is not the headline.

## Empirical design

- Primary data: all eleven events in `g2-convective-11`.
- Evaluation: leave one complete event out.
- Forecast scope: first 24 hours; primary endpoints h+6 and h+24.
- Proposed model: the existing two-rate neural dynamics, unchanged in architecture.
- Theory comparator: parameter-matched `net_scaled` one-flow NN.
- External baselines: HGB on the same information and damped persistence.
- Main experiment: one performance table.
- Central ablation: two flows collapsed to one.
- Controlled theory ablation: one synthetic phase diagram varying `D`, `R`, `v`, `n`, and `sigma^2`.

No semiparametric model, memory state, damage gate, second damage head, family-ordering campaign, or architecture sweep belongs to the current study.
