# Claude Code: start here

## Canonical branch

Use **only** this branch for the current AISTATS project:

```text
aistats-current
```

`main` is retained as the immutable public-data/source baseline. It is **not** the current paper-development branch. All other historical working branches are obsolete and must not be used as evidence, implementation guidance, or experiment specifications.

## Required read order

1. `CC_START_HERE.md`
2. `docs/AISTATS_MANUSCRIPT_AND_EXPERIMENT_LOCK_V3.md`
3. `paper/aistats/main.tex`
4. `paper/aistats/sections/` in numerical order
5. `paper/aistats/MANUSCRIPT_PREFLIGHT.md`
6. `FIREWALL.md`
7. `RESULTS_LEDGER.md` only when a claim or archived result needs source verification

## Current scientific state

The canonical manuscript asks when a single signed flow is sufficient and when interruption and restoration should be modeled as two nonnegative flows. The theory is organized around the exact one-flow representation gap, the identification geometry, and the finite-sample selection index. The exactly solvable case belongs in the main paper. The real-storm experiment section contains the locked design and result placeholders for the formal run.

## Hard instructions

- Do not recover protocols, prose, claims, or experimental targets from deleted historical branches.
- Do not replace the canonical manuscript with an older `main.tex`, `DRAFT.md`, or prior experiment prompt.
- Do not add memory, semi-parametric terms, gates, secondary-damage heads, new event subsets, or broad sweeps unless Shuai explicitly changes the locked study.
- Treat storm event as the inferential unit; neural seeds measure optimization variability only.
- Preserve the one-flow versus two-flow structural comparison as the main empirical contrast.
- Before changing the paper or running experiments, report the checked-out branch and current commit SHA.

## Canonical paper entry point

```text
paper/aistats/main.tex
```

Any future handoff should cite `aistats-current` rather than a dated GPT/CC branch name.
