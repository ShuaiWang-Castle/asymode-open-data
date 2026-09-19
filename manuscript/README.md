# AsymODE manuscript

Current working manuscript for the NET versus Asym conditional-response study.

- [Manuscript PDF](AsymODE.pdf)
- [Complete LaTeX source](AsymODE.tex)
- [All figures, combined](AsymODE_Figures.pdf)
- [Individual figure PDFs](figures/)

## Compile

Run from this directory:

```bash
pdflatex -interaction=nonstopmode -halt-on-error AsymODE.tex
pdflatex -interaction=nonstopmode -halt-on-error AsymODE.tex
```

The figure paths in the source are `figures/xxxx.pdf`. The A/B filename suffixes have been removed. The conference style is embedded in the source and materializes as `aistats2026.sty` when compiling.

This publication copies the current working draft; no experiment has been rerun and no experimental branch has been overwritten. Figure 1 is a conceptual illustration, not a new fitted-model experiment.
