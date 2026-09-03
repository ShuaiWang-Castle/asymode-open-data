# AISTATS manuscript: one flow versus two opposing flows

This directory contains the canonical LaTeX source for the Gamma-centered paper.

## Template status

AISTATS 2027 had not released its author kit when this version was compiled. The source therefore uses the latest official AISTATS style, `aistats2026.sty`, without modifying the style file. The package name and checklist should be replaced only after the 2027 author kit is released and diffed against the current instructions.

## Build

```bash
./build.sh
```

The build does four things in order:

1. regenerates `figures/flow_selection_solvable_case.pdf` from the supplied Monte Carlo CSV;
2. downloads the official AISTATS 2026 paper pack when the style file is absent;
3. runs `latexmk` with BibTeX and fails on unresolved references or citations;
4. audits page size and rejects Type 3 fonts.

Python requirements for the figure are `numpy`, `pandas`, and `matplotlib`.

## Paper structure

- `main.tex`: canonical driver;
- `sections/01_introduction.tex`: one question and three linked contributions;
- `sections/02_setup.tex`: pooled model and orthogonal identification geometry;
- `sections/03_value.tex`: exact one-flow gap, the unique boxed Gamma criterion, and the exactly solvable case;
- `sections/04_experiments.tex`: completed 24/48-hour leave-one-event-out experiment and local information audit;
- `sections/05_discussion.tex`: limitations, rollout bridge, and conclusion;
- `sections/appendix.tex`: complete proofs and event-level results.

## Scientific status

The exact population gap and orthogonal information results are submission-grade under their stated pooled conditional-mean assumption. `Gamma_n=1` is an oracle-selected, span-relaxed fixed-design Gaussian benchmark, not a universal neural-network threshold. The completed event-held-out experiment is reported without changing its endpoints or objective: the 24-hour average structural gain is positive but fails the original strict confirmation gate, the 48-hour result is uncertain, and HGB and damped persistence remain stronger in mean long-horizon RMSE.
