"""Result-file header (schema v2) and the digests that make results comparable.

A result is only comparable to another if they agree on what was scored: which
samples (panels, split), which inputs (channels, clock), which cells (mask), and
which number (metric). Every one of those is a definition string with a digest,
and `scripts/check_comparable.py` refuses to compare files whose digests differ.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json

__all__ = ["digest_of", "CLOCKS", "MASK_DEFINITION", "METRIC_DEFINITION", "result_header",
           "REQUIRED_KEYS"]

CLOCKS = {
    "utc_hour":       "sin/cos of the UTC hour of day of each forecast step, from the panel timestamps",
    "none":           "no clock channels",
    "lead_phase_old": "DIAGNOSTIC ONLY: sin/cos of the lead time modulo 24 h (legacy; not hour of day)",
}
MASK_DEFINITION = ("cell observed iff a collection run exists at its timestamp and the county reports "
                   "within +/-7 days; hourly = mean of observed 15-min sub-steps; unobserved cells excluded "
                   "from every loss and metric")
METRIC_DEFINITION = "RMSE over observed hourly cells at horizons 1/6/24/48 pooled over test samples"

REQUIRED_KEYS = ["experiment_id", "created_utc", "source_commit_at_launch", "dirty_at_launch",
                 "panel_ids", "panel_digest", "channel_names", "channel_digest",
                 "clock_definition", "clock_digest", "split_unit", "outer_split_digest",
                 "outer_split_seed", "inner_split_seed", "model_seeds", "mask_definition",
                 "mask_digest", "metric_definition", "metric_digest", "hyperparameters",
                 "wall_time_s", "convergence", "schema_version"]


def digest_of(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:12]


def result_header(*, experiment_id: str, source: dict, panel_ids, panel_digest: str,
                  channel_names, channel_digest: str, clock: str, split_unit: str,
                  outer_split_digest: str, outer_split_seed: int, inner_split_seed: int,
                  model_seeds, hyperparameters: dict) -> dict:
    if clock not in CLOCKS:
        raise ValueError(f"unknown clock {clock!r}; allowed {sorted(CLOCKS)}")
    if split_unit not in ("event", "county"):
        raise ValueError("split_unit must be 'event' or 'county'")
    return {
        "schema_version": 2,
        "experiment_id": experiment_id,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_commit_at_launch": source.get("commit"),
        "dirty_at_launch": source.get("dirty"),
        "panel_ids": sorted(panel_ids),
        "panel_digest": panel_digest,
        "channel_names": list(channel_names),
        "channel_digest": channel_digest,
        "clock_definition": CLOCKS[clock],
        "clock_digest": digest_of(clock + "|" + CLOCKS[clock]),
        "split_unit": split_unit,
        "outer_split_digest": outer_split_digest,
        "outer_split_seed": int(outer_split_seed),
        "inner_split_seed": int(inner_split_seed),
        "model_seeds": [int(s) for s in model_seeds],
        "mask_definition": MASK_DEFINITION,
        "mask_digest": digest_of(MASK_DEFINITION),
        "metric_definition": METRIC_DEFINITION,
        "metric_digest": digest_of(METRIC_DEFINITION),
        "hyperparameters": hyperparameters,
        "wall_time_s": None,
        "convergence": None,
    }
