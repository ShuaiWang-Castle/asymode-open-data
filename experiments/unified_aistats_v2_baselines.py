#!/usr/bin/env python3
"""External baselines for the unified AISTATS v2 main experiment.

The script evaluates exactly two non-neural baselines under the same eleven
leave-one-event-out folds and the same h+6/h+24 information sets:

1. histogram gradient boosting, with tree count chosen on the two fixed
   outcome-blind validation events;
2. damped persistence, with a training-only event-balanced damping coefficient.

No additional baseline family or hyperparameter sweep is authorized here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from asymode import panels as panelset, schema  # noqa: E402
import exp05_real_dynamics as exp05  # noqa: E402
import unified_aistats_v2 as v2  # noqa: E402

INTERIM = ROOT / "data/interim"
HORIZONS = (6, 24)
TREE_COUNTS = (50, 150, 300)


def features(y0: np.ndarray, X: np.ndarray, h: int) -> np.ndarray:
    """Use exactly the weather path available up to the requested horizon."""
    return np.concatenate([y0[:, None], X[:, :h].reshape(len(X), -1)], axis=1).astype(np.float32)


def event_mse(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray, ix: np.ndarray) -> float:
    keep = mask[ix]
    if not np.any(keep):
        return float("nan")
    err = pred[ix][keep] - truth[ix][keep]
    return float(np.mean(err**2))


def fit_hgb_fold(
    fold: dict,
    event_rows: dict[str, np.ndarray],
    y0: np.ndarray,
    X: np.ndarray,
    yt: np.ndarray,
    mask: np.ndarray,
) -> list[dict]:
    tr = np.concatenate([event_rows[e] for e in fold["train"]])
    te = event_rows[fold["test"][0]]
    rows = []
    for h in HORIZONS:
        F = features(y0, X, h)
        target = yt[:, h-1]
        observed = mask[:, h-1]
        trh = tr[observed[tr]]
        if len(trh) == 0:
            raise RuntimeError("no observed HGB training targets")
        best = None
        for n_iter in TREE_COUNTS:
            model = HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.05,
                max_iter=n_iter,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                l2_regularization=1e-4,
                early_stopping=False,
                random_state=0,
            )
            model.fit(F[trh], target[trh])
            pred = np.clip(model.predict(F), 0.0, 1.0)
            val_losses = []
            for e in fold["validation"]:
                val_losses.append(event_mse(pred, target, observed, event_rows[e]))
            score = float(np.nanmean(val_losses))
            rec = (score, n_iter, model, pred, val_losses)
            if best is None or score < best[0]:
                best = rec
        assert best is not None
        score, n_iter, model, pred, val_losses = best
        mse = event_mse(pred, target, observed, te)
        rows.append({
            "arm": "hgb_same_information",
            "fold": fold["fold"],
            "test_event": fold["test"][0],
            "validation_events": fold["validation"],
            "horizon": h,
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "selected_max_iter": n_iter,
            "validation_equal_event_mse": score,
            "validation_event_mse": dict(zip(fold["validation"], val_losses)),
            "n_train": int(len(trh)),
            "n_test": int(np.sum(observed[te])),
        })
    return rows


def fit_damped_fold(
    fold: dict,
    event_rows: dict[str, np.ndarray],
    y0: np.ndarray,
    yt: np.ndarray,
    mask: np.ndarray,
) -> list[dict]:
    """Event-balanced weighted least-squares damping, clipped to [0,1]."""
    te = event_rows[fold["test"][0]]
    rows = []
    for h in HORIZONS:
        target = yt[:, h-1]
        observed = mask[:, h-1]
        numerator = 0.0
        denominator = 0.0
        for e in fold["train"]:
            ix = event_rows[e]
            ix = ix[observed[ix]]
            if len(ix) == 0:
                continue
            w = 1.0 / len(ix)
            numerator += w * float(np.sum(y0[ix] * target[ix]))
            denominator += w * float(np.sum(y0[ix] ** 2))
        alpha = float(np.clip(numerator / max(denominator, 1e-12), 0.0, 1.0))
        pred = alpha * y0
        mse = event_mse(pred, target, observed, te)
        rows.append({
            "arm": "damped_persistence",
            "fold": fold["fold"],
            "test_event": fold["test"][0],
            "validation_events": fold["validation"],
            "horizon": h,
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "alpha": alpha,
            "n_test": int(np.sum(observed[te])),
        })
    return rows


def summarize(rows: list[dict]) -> dict:
    out = {}
    for arm in sorted({r["arm"] for r in rows}):
        out[arm] = {}
        for h in HORIZONS:
            vals = [r["rmse"] for r in rows if r["arm"] == arm and r["horizon"] == h]
            out[arm][f"h{h}"] = {
                "equal_event_mean_rmse": float(np.mean(vals)),
                "median_rmse": float(np.median(vals)),
                "n_events": len(vals),
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--test-events", nargs="*", default=None)
    args = ap.parse_args()

    source = panelset.source_version(ROOT)
    wanted, panel_digest = panelset.resolve(INTERIM, str(v2.MANIFEST))
    y0, X, yt, mask, fips, panel, origin, t0h = exp05.load_pooled(
        v2.HORIZON, 12, panels=wanted, with_time=True
    )
    X = exp05.add_context(X, y0, v2.HORIZON, t0_hour=t0h, clock="utc_hour")
    events = sorted(set(panel.tolist()))
    smap = v2.load_split_map(events)
    folds = smap["folds"]
    if args.test_events:
        keep = set(args.test_events)
        folds = [f for f in folds if f["test"][0] in keep]
    event_rows = {e: np.where(panel == e)[0] for e in events}

    rows = []
    for fold in folds:
        rows.extend(fit_hgb_fold(fold, event_rows, y0, X, yt, mask))
        rows.extend(fit_damped_fold(fold, event_rows, y0, yt, mask))
        print(f"completed baselines for {fold['test'][0]}", flush=True)

    channels = schema.channel_list(panelset.channel_names(INTERIM), "utc_hour")
    payload = {
        "config": {
            **schema.result_header(
                experiment_id=Path(args.out).stem,
                source=source,
                panel_ids=sorted(wanted),
                panel_digest=panel_digest,
                channel_names=channels,
                channel_digest=panelset.channel_digest(panelset.channel_names(INTERIM)),
                clock="utc_hour",
                split_unit="event",
                outer_split_digest=smap["digest"],
                outer_split_seed=0,
                inner_split_seed=0,
                model_seeds=[0],
                hyperparameters={
                    "horizons": HORIZONS,
                    "tree_counts": TREE_COUNTS,
                    "hgb_learning_rate": 0.05,
                    "damped_fit": "event-balanced training-only weighted least squares",
                },
            ),
            "protocol": "docs/UNIFIED_AISTATS_V2_PROTOCOL.md",
            "split_map": str(v2.SPLIT_MAP.relative_to(ROOT)),
            "information_rule": "y0 plus weather/clock path only through the evaluated horizon",
        },
        "rows": rows,
        "summary": summarize(rows),
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    print(json.dumps(payload["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
