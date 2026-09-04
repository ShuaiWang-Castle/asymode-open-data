# Evidence status and lessons from the final competition model

**Status:** authoritative scientific interpretation for the next implementation pass on `open-audit-20260904`.

This note supersedes any language in the earlier rescue audit that treated a plausible confound as a completed refutation. It does **not** overwrite the archived results or the diagnostic calculations. It changes how they are interpreted until a controlled adjudicating experiment is complete.

## 1. Evidence-preservation rule

The existing positive structural findings remain part of the evidence record:

- the archived event-held-out comparison is reproducible;
- the two-process state equation has repeatedly been useful in the data-challenge genealogy;
- the susceptible-pool inflow and the separation of interruption from restoration remain scientifically motivated;
- diagnostics concerning initialization, optimization, state occupancy, and cohort composition are alternative explanations or validity checks, not automatic negations of the earlier findings.

Accordingly, no file in `RESULTS_LEDGER.md` is to be relabeled as "refuted," "withdrawn," or "invalid" merely because a diagnostic identifies a possible confound. The undertrained unified-v2 output may be marked **diagnostic / not paper evidence**, but it is not to be deleted. The earlier confirmatory result remains **reproduced legacy evidence pending adjudication**.

The next experiment has one purpose: determine which parts of the earlier result survive after the comparison is made cleaner and the model is trained adequately. Until that run is complete, statements such as "the gain is only a boundary artifact," "concurrency is unsupported," or "the old conclusion is false" are prohibited.

## 2. Three different meanings of asymmetry

The project has accumulated several uses of the word *asymmetry*. They must be separated.

### A. Dynamical asymmetry

The state equation is

\[
Y_{t+1}=Y_t+U_t(1-Y_t)-R_tY_t.
\]

The known state multipliers allocate gradient and influence asymmetrically: near zero, interruption is visible and restoration is weakly informed; near one, the roles reverse. This is the basic compartmental structure.

### B. Flow separation

A one-flow model permits only one signed direction at a given context. A two-flow model permits two nonnegative conditional-mean components to coexist. The paper's representation-gap and identification results concern this distinction.

### C. Representation asymmetry

The interruption and restoration functions need not have the same inputs, temporal treatment, or functional capacity. This is an algorithmic inductive bias, not a consequence of the one-step oracle-gap theorem. The final competition model provides unusually strong evidence that this third form of asymmetry matters in practice.

The next main comparison must test **B while holding A and C fixed**. Otherwise a gain could be caused by different feature extractors or capacities rather than by the availability of the second flow.

## 3. What the final competition model teaches us

The PI-provided final data-challenge model retained the same two-process state equation but used a deliberately asymmetric function class:

- interruption side: two parallel width-32 MLPs whose logits are averaged, a distinct narrow occurrence gate, an independent low-capacity background pathway, and a learned first-order hold;
- restoration side: one linear model, recomputed every eight forecast steps and held between updates;
- process-specific inputs: a rich interruption block, a narrower and genuinely different occurrence block, and a recovery block without clock or current-state input;
- capacity allocation: approximately 6,156 parameters on interruption versus 44 on restoration;
- open-loop prediction: after the cutoff, the model receives weather and county information but no future observed outage state.

The important empirical lesson is not the exact parameter count. It is the set of architectural invariants:

1. sparse interruption occurrence and interruption magnitude should not be forced through the same representation;
2. the occurrence gate must read a genuinely different input block from the magnitude network;
3. the low-capacity background path must remain independent rather than becoming another output of the dominant trunk;
4. interruption benefits from substantially more nonlinear capacity than restoration;
5. a temporal hold mechanism materially broadens the predicted interruption pulse;
6. the second interruption network behaves as an ensemble component and became *more*, not less, valuable under sufficient optimization;
7. optimization budget is a scientific variable: increasing the full-batch budget from 400 to 1,600 steps improved every reported horizon without changing the architecture.

The repository genealogy is consistent with these principles: it contains two interruption heads, a distinct occurrence head and background head, and a learned logit hold. Those mechanisms are not to be rediscovered by a broad sweep; they are prior design information supplied by the completed competition study.

## 4. Consequence for the proposed paper model

The earlier rescue proposal forced both processes through one small shared GRU/backbone. That is no longer the selected design. It risks removing precisely the process-specific inductive bias that the competition work found to be load-bearing.

The selected host is instead a **clean, competition-informed asymmetric two-process scaffold**. It keeps only the invariants above and omits the rest of the historical model zoo. Both empirical arms use the same scaffold, the same input blocks, the same optimization schedule, and the same initialization procedure.

Let the scaffold produce two nonnegative proposals

\[
\widetilde U_t=F_U(x^U_{1:t},x^{\mathrm{occ}}_t),
\qquad
\widetilde R_t=F_R(x^R_t).
\]

Define their signed difference

\[
s_t=\widetilde U_t-\widetilde R_t.
\]

The two arms are then:

\[
\begin{aligned}
\text{two-flow:}\quad &U_t=\widetilde U_t,\qquad R_t=\widetilde R_t,\\
\text{one-flow collapse:}\quad &U_t=[s_t]_+,\qquad R_t=[-s_t]_+.
\end{aligned}
\]

Thus both arms have the same process-specific feature maps and parameter budget. The only functional difference is whether their common component

\[
c_t=\min\{\widetilde U_t,\widetilde R_t\}
\]

is retained or removed. At the level of the conditional transition function, the one-flow arm is exactly the state-scaled signed class used in the theory. Its internal parameterization is deliberately generous and therefore gives the restricted class a conservative comparison.

This output-level nesting replaces the previous "two independent MLPs versus one width-48 MLP" comparison. It also avoids forcing restoration to inherit the interruption representation merely for architectural symmetry.

## 5. What remains open

The following are not settled by the existing artifacts:

- whether the earlier two-flow gain survives class-correct initialization and sufficient optimization;
- whether the gain is primarily directional decoupling at onset, concurrent interior-flow value, or both;
- whether the process-specific architecture transfers from the competition setting to the open multi-storm data;
- whether a broader and more event-balanced cohort changes the result;
- whether current weather aggregation and reconstructed zeros materially affect the comparison;
- whether the neural estimator reaches the local information regime described by the theory.

The next run must be allowed to answer these questions in either direction. Its design may increase statistical relevance—by using heterogeneous events, event-centered origins, equal-event weighting, and explicit state strata—but it may not select events, origins, or metrics using prior model gains.

## 6. Paper policy before adjudication

Do not replace the manuscript's result placeholders and do not rewrite the conclusion as a negative result. The paper may continue to state the mathematical claims and the intended positive empirical hypothesis. Any older empirical number remains clearly labeled by its original protocol. The new result will determine the final wording after it is reviewed at the event level.
