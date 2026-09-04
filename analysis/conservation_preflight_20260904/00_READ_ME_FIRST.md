# Conservation preflight: read this first

**Branch:** `open-audit-20260904`  
**Immediate status:** zero-training design audit only.  
**Evidence policy:** neither the reproduced legacy result nor the paper conclusion is relabelled by this task.

## Decision

The conservation observation supplied after the V2 pilot is important enough to change the order of work. Before repairing the interruption heads or running another pilot, the project must measure whether the public county-level design contains enough information for the one-flow/two-flow contrast to be statistically resolvable.

The exact starting point is the weighted constant least-squares identity

\[
\widehat U(1-\widehat\mu)-\widehat R\widehat\mu=\overline{\Delta Y}.
\]

When the unconstrained constant fit is interior and the sampled transition window is empirically closed, this reduces to

\[
\frac{\widehat U}{\widehat R}=\frac{\widehat\mu}{1-\widehat\mu}.
\]

The fold-2 pilot constants therefore reveal a genuine design warning: their fitted rate ratio is approximately `1e-3`, and the structural treatment begins very small. The project should have computed the residual scale and the implied selection geometry before spending a neural-training budget.

## What is not yet established

The stronger statement that the null was universally or structurally guaranteed is not accepted without qualification.

1. The ratio identity requires an empirically closed window; `mean(Delta)=0` must be measured under the exact mask and weights rather than inferred from the nominal panel length.
2. The exact normal-equation identity applies automatically to an unconstrained or interior constant least-squares fit. A nonnegative or capped fit on a boundary obeys KKT conditions, not the same equality.
3. Global closure constrains **mean flows**, not pointwise conditional rates. A context-dependent interruption rate can be large on rare weather states while its full-window average is small.
4. A neural fit satisfies the analogous mean-residual moment only if the relevant constant-shift direction is admissible, non-saturated, unpenalized, and optimized to stationarity.
5. A small constant `min(U,R)` does not, by itself, prove that the final conditional common component is pointwise negligible or that an earlier percentage MSE difference must be a parameterization artifact.

These qualifications are not a rejection of the conservation insight. They define the audit that can determine how much of it applies to this dataset.

## Authorized next task

Run the GitHub-only preflight specified in this directory. The executing agent must implement and test the audit only inside the authorized `implementation/` subdirectory before producing results. It uses two isolated clean clones:

- code and analyses from `open-audit-20260904`;
- public data from pinned `main` commit `8dd47c5ccd829611f27b69a3d64c274a0a24c400`.

It verifies the published SHA-256 manifest, constructs the correct adjacent-observation transitions, and reports event-, source-fold-, and local-neighbourhood conservation identities, closure, residual scale, the descriptive plug-in selection index, an assumption-sensitive near-closure formula, state-resolution diagnostics, and a fixed full-panel versus exogenous 48-hour storm-window comparison. It performs no model fitting beyond two-variable constant least squares and no neural optimization.

## Hard stop

Do not repair the neural model, rerun the three-event pilot, launch the five-fold campaign, change horizons, select events by outcomes, or edit the manuscript conclusion before the preflight report is reviewed by Shuai.

## Read order

1. `00_READ_ME_FIRST.md`
2. `01_CLAUDE_CLAIM_ADJUDICATION.md`
3. `02_CONSERVATION_THEORY_NOTE.md`
4. `03_GITHUB_ONLY_PROTOCOL.md`
5. `05_OUTPUT_SCHEMA.md`
6. `04_CODEX_GPT_WORK_PROMPT.md`
