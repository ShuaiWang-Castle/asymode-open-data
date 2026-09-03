# AISTATS theory-first manuscript

This directory contains the paper-level LaTeX source for the one-flow-versus-two-flow study.

## Template status

AISTATS 2027 has not released its author kit as of 2026-09-03. The source therefore uses the latest official AISTATS style, `aistats2026.sty`, without modifying the style file. The build script downloads the official 2026 paper pack at compile time. When the 2027 kit is released, replace the package name and checklist only after diffing the new official instructions.

## Build

```bash
./build.sh
```

The script downloads the official AISTATS 2026 paper pack when necessary, runs `latexmk`, rejects unresolved citations/references and Type 3 fonts, and writes `main.pdf`.

## Scientific status

The population oracle-gap theorem and identification theorem are submission-grade with the assumptions and boundary cases retained in `main.tex`. The finite-sample `Gamma_n=1` result is intentionally an oracle-selected, span-relaxed fixed-design benchmark; it is not the exact decision boundary for the constrained neural networks. The event-shift result additionally requires structural invariance of the two conditional flow functions.

Real-data result cells remain `TBD` until the locked event-held-out experiment is complete. The compiled manuscript is therefore a rigorous theory-first working paper, not yet a final empirical submission.
