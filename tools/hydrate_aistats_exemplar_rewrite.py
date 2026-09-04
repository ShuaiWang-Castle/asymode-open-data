#!/usr/bin/env python3
"""Materialize the AISTATS exemplar rewrite from its staged source archive.

The payload is split only because the GitHub connector writes UTF-8 text files.
The hydrated directory contains ordinary LaTeX, BibTeX, Markdown, CSV, Python,
and shell sources. Generated PDFs and the conference style file are not stored
in the payload; the build script recreates them deterministically.
"""
from __future__ import annotations

import base64
import hashlib
import io
import shutil
import tarfile
from pathlib import Path

PARTS = Path("tools/aistats_exemplar_payload")
TARGET = Path("paper/aistats_exemplar_rewrite")
EXPECTED_SHA256 = "58f9ed6d08e2815834dffd512ec66989f7cc2fa8aa9e0cbbccafe697960e9751"


def main() -> None:
    files = sorted(PARTS.glob("part-*.b64"))
    if not files:
        raise SystemExit(f"No payload parts found under {PARTS}")
    encoded = "".join(path.read_text(encoding="utf-8").strip() for path in files)
    raw = base64.b64decode(encoded, validate=True)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_SHA256:
        raise SystemExit(
            f"Payload checksum mismatch: expected {EXPECTED_SHA256}, observed {digest}"
        )

    if TARGET.exists():
        shutil.rmtree(TARGET)
    TARGET.mkdir(parents=True)
    root = TARGET.resolve()
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        for member in archive.getmembers():
            destination = (TARGET / member.name).resolve()
            if destination != root and root not in destination.parents:
                raise SystemExit(f"Unsafe archive member: {member.name}")
        archive.extractall(TARGET)

    required = [
        TARGET / "main.tex",
        TARGET / "references.bib",
        TARGET / "results_placeholders.tex",
        TARGET / "sections/02_related_work.tex",
        TARGET / "sections/04_theory.tex",
        TARGET / "sections/05_experiments.tex",
        TARGET / "appendices/A_proofs.tex",
        TARGET / "appendices/D_experimental_protocol.tex",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"Hydration incomplete; missing: {missing}")
    print(
        f"Hydrated {TARGET} from {len(files)} parts; "
        f"archive sha256={digest}; files={sum(p.is_file() for p in TARGET.rglob('*'))}"
    )


if __name__ == "__main__":
    main()
