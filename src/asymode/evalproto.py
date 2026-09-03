"""Evaluation protocol: county-held-out folds, masked metrics, forecast origins.

Fixed before any model is fitted, and versioned, so a fold assignment can never be
quietly reshaped around a result. Fold membership is a deterministic function of
the county code and a seed -- not a shuffle -- so it is reproducible from the
identifiers alone and does not depend on the order counties happen to arrive in.

Two rules that the masking exists to enforce:

  * A cell the publisher never observed is never scored. Roughly three quarters
    of the raw grid is absent and the densification only fills what it can defend;
    scoring a model against a filled-in guess measures the filling, not the model.
  * A test county is unseen in training. Held-out *counties*, not held-out time,
    because the question is whether the dynamics transfer to a county the model
    has never met -- which is what a forecasting system actually faces.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


def make_folds(fips: list[str], k: int = 5, seed: int = 0) -> np.ndarray:
    """Deterministic county -> fold assignment in [0, k)."""
    out = np.empty(len(fips), dtype=np.int64)
    for i, f in enumerate(fips):
        h = hashlib.sha256(f"{seed}:{f}".encode()).digest()
        out[i] = int.from_bytes(h[:8], "big") % k
    return out


def inner_split(fips_rows: "np.ndarray", seed: int = 0, fold: int = 0,
                frac: float = 0.15):
    """Split training rows into a fit set and an early-stopping set, by county.

    The outer folds hold out counties, because generalising to a county the model
    has never seen is the thing being measured. An inner split that partitions
    *rows* instead leaves the same counties on both sides of it, so the validation
    curve it produces cannot see county-level overfitting -- and a model selected
    on that curve is selected for performance on counties it has already read.
    The stopping rule has to hold out what the evaluation holds out.

    Returns (fit_rows, val_rows) as positions into `fips_rows`. The offset keeps
    the inner split from reproducing the outer partition it sits inside, and
    varying it with the fold matters for a second reason: `make_folds` hashes the
    county id, so a fold-independent offset would hold out the *same* counties in
    every fold and those counties would then never be trained on at all.
    """
    uniq = sorted(set(fips_rows.tolist()))
    k = max(2, round(1.0 / frac))
    assign = make_folds(uniq, k=k, seed=1000 + 10 * seed + fold)
    val_counties = {c for c, f in zip(uniq, assign) if f == 0}
    is_val = np.array([f in val_counties for f in fips_rows])
    if is_val.all() or not is_val.any():        # degenerate on a tiny county set
        n = max(1, int(frac * len(fips_rows)))
        return np.arange(n, len(fips_rows)), np.arange(n)
    return np.where(~is_val)[0], np.where(is_val)[0]


@dataclass
class Task:
    """A forecasting task over one panel.

    y and observed are (C, T) hourly arrays. An origin `o` means: everything up to
    and including o is available; predict o + h for each horizon h.
    """
    y: np.ndarray
    observed: np.ndarray
    fips: list[str]
    horizons: tuple[int, ...] = (1, 6, 24, 48)
    min_history: int = 24

    def origins(self, stride: int = 6) -> np.ndarray:
        hi = self.y.shape[1] - max(self.horizons) - 1
        return np.arange(self.min_history, hi + 1, stride)


def score(pred: np.ndarray, task: Task, rows: np.ndarray, origins: np.ndarray) -> dict:
    """pred is (n_rows, n_origins, n_horizons). Metrics over observed cells only."""
    y, obs = task.y[rows], task.observed[rows]
    out = {}
    for hi, h in enumerate(task.horizons):
        tgt_t = origins + h
        tgt = y[:, tgt_t]
        m = obs[:, tgt_t]
        e = (pred[:, :, hi] - tgt)[m]
        out[f"rmse_h{h}"] = float(np.sqrt(np.mean(e ** 2))) if e.size else float("nan")
        out[f"mae_h{h}"] = float(np.mean(np.abs(e))) if e.size else float("nan")
        out[f"n_h{h}"] = int(e.size)
    ok = [out[f"rmse_h{h}"] for h in task.horizons if np.isfinite(out[f"rmse_h{h}"])]
    out["rmse_mean"] = float(np.mean(ok)) if ok else float("nan")
    return out


def to_hourly(y15: np.ndarray, obs15: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Collapse 15-minute steps to hourly.

    The hourly value is the mean of the observed sub-steps, and an hour counts as
    observed if any of its sub-steps was. Averaging rather than sampling on the
    hour keeps a spike that lasted forty minutes from vanishing because it missed
    the sampling instant.
    """
    C, T = y15.shape
    n = T // 4
    y = y15[:, :n * 4].reshape(C, n, 4)
    o = obs15[:, :n * 4].reshape(C, n, 4)
    cnt = o.sum(axis=2)
    tot = np.where(o, np.nan_to_num(y), 0.0).sum(axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        yh = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    return yh.astype(np.float32), cnt > 0
