"""Which panels a run pooled, recorded so comparability is checkable.

Two results are comparable only if they were fitted and scored on the same
samples. That is easy to state and easy to violate: the panel directory grows as
covariate downloads land, so a script that globs it silently pools a different
set every time it is run, and two result files written a few hours apart can
disagree for that reason alone with nothing in either file to say so.

So the panel set is named, not discovered, and its digest travels inside every
result file. A digest mismatch is then a fact a script can check rather than
something a reader has to remember.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

MANIFEST_NAME = "PANEL_MANIFEST.json"


def digest(panels: list[str]) -> str:
    """Order-independent digest of a panel set."""
    return hashlib.sha256("|".join(sorted(panels)).encode()).hexdigest()[:12]


def available(interim: Path) -> list[str]:
    """Panel days that have both a panel and a driver file."""
    return sorted(p.stem.replace("panel_", "") for p in interim.glob("panel_*.npz")
                  if (interim / f"drivers_{p.stem.replace('panel_', '')}.npz").exists())


def read_manifest(interim: Path) -> dict | None:
    f = interim / MANIFEST_NAME
    return json.loads(f.read_text()) if f.exists() else None


def resolve(interim: Path, spec: str | None) -> tuple[list[str], str]:
    """Resolve a `--panels` argument to an explicit list and its digest.

    `None` or "manifest" uses the manifest and fails if it is absent, because a
    silent fallback to globbing is the failure this module exists to prevent.
    "auto" globs deliberately and is for exploration, not for archived runs.
    Anything else is read as a path to a JSON file with a "panels" key.
    """
    have = available(interim)
    if spec == "auto":
        return have, digest(have)
    if spec is None or spec == "manifest":
        m = read_manifest(interim)
        if m is None:
            raise SystemExit(
                f"no {MANIFEST_NAME} in {interim}. Write one with "
                f"scripts/build_panel_manifest.py, or pass --panels auto to pool "
                f"whatever is on disk (exploration only -- such a run is not "
                f"comparable to anything).")
        want = list(m["panels"])
    else:
        want = list(json.loads(Path(spec).read_text())["panels"])
    missing = sorted(set(want) - set(have))
    if missing:
        raise SystemExit(f"panels named but not built: {missing}")
    return sorted(want), digest(want)
