# AISTATS manuscript workspace

Canonical source: `main.tex`.

The ordinary source tree is reconstructed deterministically from the checksum-locked payload by running from the repository root:

```bash
python tools/hydrate_aistats_exemplar_rewrite.py
```

## Build the writing draft

```bash
cd paper/aistats_exemplar_rewrite
./build.sh
```

The build regenerates the exactly solvable case, compiles the paper, checks references and citations, rejects Type 3 fonts, verifies US-Letter output, and checks that the eight-page main-paper limit is respected before references. The current PDF has an eight-page main paper followed by references, the reproducibility checklist, and a fifteen-page supplement.

## Submission gate

```bash
./pre_submission_check.sh
```

This stricter command intentionally fails while positive-form result placeholders remain. `results_placeholders.tex` is the only file in which provisional empirical values are permitted. It must be replaced from the immutable output of the locked experiment before submission.

The source uses the official `aistats2026.sty` during this working revision because the AISTATS 2027 author kit was not available when the revision was assembled. Replace it only with the official 2027 kit and rerun the complete preflight.
