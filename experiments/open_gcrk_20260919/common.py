"""Shared loading and metric code for the open-data GCRK evaluation."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# OPEN_GCRK_ROUND selects a later round (PREREG Amendment 2); unset = round 1 as pre-registered
ROUND = os.environ.get("OPEN_GCRK_ROUND", "r1")
FEATURES = ROOT / "data" / "interim" / "open_gcrk" / ("features.npz" if ROUND == "r1" else f"features_{ROUND}.npz")
RUNS = ROOT / "runs" / "open_gcrk_20260919" / ("" if ROUND == "r1" else ROUND)
RESULTS = HERE / "results" / ("" if ROUND == "r1" else ROUND)
SPLITS_FILE = HERE / ("splits.json" if ROUND == "r1" else f"splits_{ROUND}.json")
SEEDS = (0, 1, 2, 3, 4)
SEGMENTS = {"1-6 h": (1, 6), "7-24 h": (7, 24), "25-48 h": (25, 48), "49-144 h": (49, 144)}
PEAK_MIN = 0.01


def load_features() -> dict:
    z = np.load(FEATURES, allow_pickle=False)
    return {k: z[k] for k in z.files}


def load_splits() -> dict:
    return json.loads(SPLITS_FILE.read_text())


def cell(design, seed, fold, arm) -> Path:
    return RUNS / design / f"seed{seed}" / f"fold{int(fold):02d}" / arm


def collect(design: str, arm: str, seed: int, n: int, key: str = "P") -> np.ndarray | None:
    """OUTER predictions of one (design, arm, seed) assembled over all folds; None if incomplete."""
    sp = load_splits()[design]
    P = np.full((n, 144), np.nan, np.float64)
    for f in sp:
        p = cell(design, seed, f, arm) / "outer.npz"
        if not p.exists():
            return None
        z = np.load(p)
        assert np.array_equal(np.sort(z["idx"]), np.sort(np.array(sp[f]["outer"])))
        P[z["idx"]] = z[key]
    assert np.isfinite(P).all()
    return P


def timesfm(F: dict, arm: str = "WEATHER") -> np.ndarray | None:
    p = RUNS / "timesfm" / f"timesfm_{arm}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    pos = {(e, f): i for i, (e, f) in enumerate(zip(z["event"], z["fips"]))}
    return np.stack([z["P"][pos[(e, f)]] for e, f in zip(F["event"], F["fips"])]).astype(np.float64)


def metrics(P: np.ndarray, F: dict, units: np.ndarray | None = None) -> dict:
    """Pooled metrics over observed cells of the given units (all units if None)."""
    u = np.arange(len(F["y"])) if units is None else np.asarray(units)
    y, m, p = F["y"][u].astype(np.float64), F["m"][u].astype(bool), P[u]
    e = (p - y)
    out = dict(rmse=float(np.sqrt((e[m] ** 2).mean())), mae=float(np.abs(e[m]).mean()), n_cells=int(m.sum()))
    lead = np.arange(1, 145)
    for name, (a, b) in SEGMENTS.items():
        s = m & ((lead >= a) & (lead <= b))[None, :]
        out[f"rmse {name}"] = float(np.sqrt((e[s] ** 2).mean()))
    ym = np.where(m, y, -np.inf)
    peak_obs = ym.max(1)
    elig = peak_obs >= PEAK_MIN
    t_obs = ym.argmax(1)
    pm = np.where(m, p, -np.inf)
    mag = pm.max(1) - peak_obs
    tim = pm.argmax(1) - t_obs
    out.update(n_peak_units=int(elig.sum()),
               peak_mag_abs_mean=float(np.abs(mag[elig]).mean()), peak_mag_abs_median=float(np.median(np.abs(mag[elig]))),
               peak_mag_signed_mean=float(mag[elig].mean()),
               peak_time_abs_mean=float(np.abs(tim[elig]).mean()), peak_time_abs_median=float(np.median(np.abs(tim[elig]))),
               peak_time_signed_mean=float(tim[elig].mean()))
    zero = m & (y == 0)
    act = m & (y > 0.01)
    out["false_activity_share"] = float((p[zero] > 0.001).mean())
    out["under_half_share"] = float((p[act] < 0.5 * y[act]).mean())
    return out


def unit_sse(P: np.ndarray, F: dict) -> np.ndarray:
    m = F["m"].astype(bool)
    return np.where(m, (P - F["y"]) ** 2, 0.0).sum(1)


def persistence(F: dict) -> np.ndarray:
    return np.repeat(F["y0"].astype(np.float64)[:, None], 144, 1)
