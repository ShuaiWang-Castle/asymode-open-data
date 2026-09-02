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


def channel_names(interim: Path) -> list[str]:
    """Driver channel names, plus the two the context step appends.

    Read from the files rather than hard-coded: the driver builder is free to
    reorder or extend, and a positional list would silently mean something else
    the next time it does.
    """
    for df in sorted(interim.glob("drivers_*.npz")):
        import numpy as np
        z = np.load(df, allow_pickle=True)
        return [str(c) for c in z["channels"]] + ["clock_sin", "clock_cos"]
    raise FileNotFoundError(f"no driver files in {interim}")


def channel_digest(names: list[str]) -> str:
    """Digest of the channel set.

    Recorded alongside the panel digest because the same panels rebuilt with a
    different set of covariates produce a different experiment, and that is the
    more insidious of the two: the panel list is visible in a directory listing,
    the channel list is inside the files.
    """
    return hashlib.sha256("|".join(names).encode()).hexdigest()[:12]


def source_version(root: Path) -> dict:
    """The commit a run was produced from, and whether the tree was dirty.

    A result file names its panels and its channels; this names the code. Without
    it, reproducing an archived number means guessing which revision produced it,
    and `dirty` is the honest part -- a run from an uncommitted tree cannot be
    reproduced from the history at all, and should say so rather than imply a
    commit that does not contain it.
    """
    import subprocess
    def run(*a):
        try:
            return subprocess.run(a, cwd=root, capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except Exception:
            return ""
    head = run("git", "rev-parse", "--short", "HEAD")
    if not head:
        return {"commit": None, "dirty": None}
    return {"commit": head, "dirty": bool(run("git", "status", "--porcelain"))}
