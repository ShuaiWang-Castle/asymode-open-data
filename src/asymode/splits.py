"""Outer split protocols, pinned and digested.

Three seeds are three different things and are never overloaded:
* `outer_split_seed` decides which units are held out (and nothing else);
* `inner_split_seed` decides the early-stopping split inside each training set;
* `model_seed` decides initialisation and data order.

Two outer protocols exist and are named in every result:
* `split_unit = "event"`  -- PRIMARY. Whole storm panels are held out; no event
  contributes any county or origin to both sides. Folds are balanced on sample
  counts only (pre-outcome metadata), by a deterministic greedy rule.
* `split_unit = "county"` -- SECONDARY. Counties are held out across all events
  (unseen-county generalisation *within observed event families*). One fixed
  county -> fold map is shared by every arm and every model seed.

A split is a plain dict {unit_id: fold}. Its digest is the sha256 of the sorted
JSON, and it is written to `configs/splits/` so that a result file can name the
exact partition it was scored on.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

__all__ = ["county_folds", "event_folds", "split_digest", "save_split", "load_split",
           "assign_rows", "check_disjoint"]


def _hash_int(s: str) -> int:
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], "big")


def county_folds(fips: list[str], k: int = 5, outer_split_seed: int = 0) -> dict[str, int]:
    """Deterministic county -> fold map: sha256(seed:fips) mod k. Same rule as the
    legacy `evalproto.make_folds`, now keyed on the outer split seed only."""
    return {f: _hash_int(f"{outer_split_seed}:{f}") % k for f in sorted(set(fips))}


def event_folds(sizes: dict[str, int], k: int = 5, outer_split_seed: int = 0) -> dict[str, int]:
    """Deterministic balanced event -> fold map.

    Events are visited from largest to smallest sample count (ties broken by a
    seeded hash of the event id) and each goes to the fold with the smallest
    running total. Uses sample counts only -- no outcome enters the assignment.
    With fewer events than folds, some folds are empty and the caller must refuse.
    """
    if k < 2:
        raise ValueError("k must be >= 2")
    order = sorted(sizes, key=lambda e: (-int(sizes[e]), _hash_int(f"{outer_split_seed}:{e}")))
    load = [0] * k
    out: dict[str, int] = {}
    for e in order:
        f = int(np.argmin(load))
        out[e] = f
        load[f] += int(sizes[e])
    return out


def split_digest(mapping: dict[str, int]) -> str:
    return hashlib.sha256(json.dumps(dict(sorted(mapping.items())), sort_keys=True).encode()).hexdigest()[:12]


def save_split(mapping: dict[str, int], unit: str, k: int, outer_split_seed: int, root: Path) -> Path:
    d = split_digest(mapping)
    p = root / "configs" / "splits" / f"{unit}_k{k}_s{outer_split_seed}_{d}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text(json.dumps({"split_unit": unit, "k": k, "outer_split_seed": outer_split_seed,
                                 "digest": d, "mapping": dict(sorted(mapping.items()))}, indent=1))
    return p


def load_split(path: Path) -> dict[str, int]:
    return json.loads(Path(path).read_text())["mapping"]


def assign_rows(unit_ids: np.ndarray, mapping: dict[str, int]) -> np.ndarray:
    """Fold of every row from its unit id; refuses rows whose unit is unmapped."""
    missing = sorted(set(unit_ids.tolist()) - set(mapping))
    if missing:
        raise KeyError(f"{len(missing)} unit ids have no fold, e.g. {missing[:3]}")
    return np.array([mapping[u] for u in unit_ids.tolist()], dtype=np.int64)


def check_disjoint(unit_ids: np.ndarray, assign: np.ndarray, fold: int) -> None:
    """Train/test membership must be disjoint at the unit level."""
    te = set(unit_ids[assign == fold].tolist()); tr = set(unit_ids[assign != fold].tolist())
    if te & tr:
        raise AssertionError(f"fold {fold}: {len(te & tr)} units on both sides")
