#!/usr/bin/env python3
"""Assemble the single theorem-aligned AISTATS v2 result table.

Primary inference is on paired event-level MSE differences. Equal-event RMSE is
reported only for readability. Neural seeds are averaged inside each event.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

HORIZONS = (6, 24)
NEURAL_ARMS = ("two_rate_v2", "net_scaled_v2", "two_rate_rollout_only")
BASELINE_ARMS = ("hgb_same_information", "damped_persistence")


def exact_sign_flip_p(d: np.ndarray) -> float:
    observed = abs(float(np.mean(d)))
    vals = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(d)):
        vals.append(abs(float(np.mean(d * np.asarray(signs)))))
    vals = np.asarray(vals)
    return float(np.mean(vals >= observed - 1e-15))


def bootstrap_ci(d: np.ndarray, B: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(B, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def neural_event_table(rows: list[dict]) -> dict[tuple[str, str, int], float]:
    out = {}
    events = sorted({r["test_event"] for r in rows})
    for arm in NEURAL_ARMS:
        for e in events:
            for h in HORIZONS:
                vals = [r[f"mse_h{h}"] for r in rows if r["arm"] == arm and r["test_event"] == e]
                if vals:
                    out[(arm, e, h)] = float(np.mean(vals))
    return out


def baseline_event_table(rows: list[dict]) -> dict[tuple[str, str, int], float]:
    out = {}
    for r in rows:
        out[(r["arm"], r["test_event"], int(r["horizon"]))] = float(r["mse"])
    return out


def comparison(
    table: dict[tuple[str, str, int], float],
    proposed: str,
    reference: str,
    events: list[str],
    h: int,
    B: int,
    seed: int,
) -> dict:
    d = np.array([table[(reference, e, h)] - table[(proposed, e, h)] for e in events], dtype=float)
    ref = np.array([table[(reference, e, h)] for e in events], dtype=float)
    pro = np.array([table[(proposed, e, h)] for e in events], dtype=float)
    lo, hi = bootstrap_ci(d, B=B, seed=seed+h)
    loeo = {e: float(np.mean(np.delete(d, i))) for i, e in enumerate(events)}
    return {
        "proposed": proposed,
        "reference": reference,
        "horizon": h,
        "events": events,
        "mse_difference_by_event": {e: float(x) for e, x in zip(events, d)},
        "mean_mse_difference": float(np.mean(d)),
        "median_mse_difference": float(np.median(d)),
        "relative_mean_pct_of_reference": float(100*np.mean(d)/np.mean(ref)),
        "n_positive": int(np.sum(d > 0)),
        "n_events": len(events),
        "bootstrap_ci95": [lo, hi],
        "bootstrap_B": B,
        "exact_two_sided_sign_flip_p": exact_sign_flip_p(d),
        "leave_one_event_out_means": loeo,
        "loeo_min": float(min(loeo.values())),
        "supported": bool(lo > 0 and exact_sign_flip_p(d) < 0.05),
        "equal_event_rmse_proposed": float(np.mean(np.sqrt(pro))),
        "equal_event_rmse_reference": float(np.mean(np.sqrt(ref))),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--neural", required=True)
    ap.add_argument("--baselines", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--bootstrap", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=20260903)
    args = ap.parse_args()

    neural = json.loads(Path(args.neural).read_text())
    baselines = json.loads(Path(args.baselines).read_text())
    table = neural_event_table(neural["rows"])
    table.update(baseline_event_table(baselines["rows"]))
    events = sorted({r["test_event"] for r in neural["rows"]})
    if len(events) != 11:
        raise RuntimeError(f"expected eleven test events, got {len(events)}")

    comps = []
    for h in HORIZONS:
        comps.append(comparison(table, "two_rate_v2", "net_scaled_v2", events, h, args.bootstrap, args.seed))
        comps.append(comparison(table, "two_rate_v2", "hgb_same_information", events, h, args.bootstrap, args.seed+100))
        comps.append(comparison(table, "two_rate_v2", "damped_persistence", events, h, args.bootstrap, args.seed+200))
        comps.append(comparison(table, "two_rate_v2", "two_rate_rollout_only", events, h, args.bootstrap, args.seed+300))

    payload = {
        "metric_lock": "docs/UNIFIED_AISTATS_V2_METRIC_LOCK.md",
        "neural_result": args.neural,
        "baseline_result": args.baselines,
        "events": events,
        "comparisons": comps,
    }
    Path(args.out_json).write_text(json.dumps(payload, indent=1))

    lines = [
        "# Unified AISTATS v2 main report",
        "",
        "Primary inference uses paired event-level MSE differences; positive means the proposed two-rate model has lower risk. Equal-event RMSE is shown for readability.",
        "",
        "| comparison | horizon | RMSE proposed | RMSE reference | mean MSE difference | relative % | CI 95% | exact p | positive events | supported |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|---|",
    ]
    for c in comps:
        lines.append(
            f"| {c['proposed']} vs {c['reference']} | h+{c['horizon']} | "
            f"{c['equal_event_rmse_proposed']:.6f} | {c['equal_event_rmse_reference']:.6f} | "
            f"{c['mean_mse_difference']:+.8f} | {c['relative_mean_pct_of_reference']:+.2f}% | "
            f"[{c['bootstrap_ci95'][0]:+.8f}, {c['bootstrap_ci95'][1]:+.8f}] | "
            f"{c['exact_two_sided_sign_flip_p']:.4f} | {c['n_positive']}/{c['n_events']} | "
            f"{'yes' if c['supported'] else 'no'} |"
        )
    lines += ["", "## Event-level structural comparison", ""]
    for h in HORIZONS:
        c = next(x for x in comps if x["reference"] == "net_scaled_v2" and x["horizon"] == h)
        lines += [f"### h+{h}", "", "| event | MSE(net-scaled) − MSE(two-rate) |", "|---|---:|"]
        for e, x in c["mse_difference_by_event"].items():
            lines.append(f"| {e} | {x:+.8f} |")
        lines.append("")
    Path(args.out_md).write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
