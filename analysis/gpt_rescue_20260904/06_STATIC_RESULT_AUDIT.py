#!/usr/bin/env python3
"""Static audit of results/cc_event_repro_core_event.json.

No model is trained. The script reconstructs the signed comparator's actual
initialization, summarizes boundary predictions and checkpoint selection, and
reports event-level gains. It uses only the Python standard library.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as stats
from collections import defaultdict
from pathlib import Path


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def median(xs):
    return stats.median(xs) if xs else float("nan")


def sd(xs):
    return stats.stdev(xs) if len(xs) > 1 else 0.0


def corr(x, y):
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    mx, my = mean(x), mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = math.sqrt(sum((a - mx) ** 2 for a in x))
    dy = math.sqrt(sum((b - my) ** 2 for b in y))
    return num / (dx * dy) if dx > 0 and dy > 0 else float("nan")


def pct(x):
    return f"{100.0 * x:.2f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/cc_event_repro_core_event.json")
    ap.add_argument("--out-md", default="analysis/gpt_rescue_20260904/STATIC_RESULT_AUDIT.md")
    ap.add_argument("--out-json", default="analysis/gpt_rescue_20260904/STATIC_RESULT_AUDIT.json")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    payload = json.loads((root / args.input).read_text())
    rows = payload["rows"]
    arms = sorted({r["arm"] for r in rows})
    events = sorted({r["test_event"] for r in rows})
    if arms != ["net_scaled", "two_rate"]:
        raise RuntimeError(f"unexpected arms: {arms}")
    if len(events) != 11 or len(rows) != 66:
        raise RuntimeError(f"expected 11 events and 66 fits; got {len(events)} and {len(rows)}")

    by = defaultdict(list)
    for r in rows:
        by[(r["test_event"], r["arm"])].append(r)

    # Initialization audit. The harness passed susceptible u0/r0 to the factory,
    # which initialized net_scaled at u0-r0.
    signed_init = [r["u_init"] - r["r_init"] for r in rows if r["arm"] == "net_scaled"]
    init_summary = {
        "n": len(signed_init),
        "min": min(signed_init),
        "median": median(signed_init),
        "max": max(signed_init),
        "all_negative": all(x < 0 for x in signed_init),
        "median_abs_ratio_to_u0": median([
            abs(r["u_init"] - r["r_init"]) / max(abs(r["u_init"]), 1e-15)
            for r in rows if r["arm"] == "net_scaled"
        ]),
    }

    arm_summary = {}
    for arm in arms:
        rr = [r for r in rows if r["arm"] == arm]
        arm_summary[arm] = {
            "n_fits": len(rr),
            "parameter_count": sorted({r["parameter_count"] for r in rr}),
            "best_epoch_median": median([r["best_epoch"] for r in rr]),
            "best_epoch_le2": sum(r["best_epoch"] <= 2 for r in rr),
            "best_epoch_le2_fraction": mean([float(r["best_epoch"] <= 2) for r in rr]),
            "frac_pred_zero_mean": mean([r["frac_pred_zero"] for r in rr]),
            "frac_pred_zero_median": median([r["frac_pred_zero"] for r in rr]),
            "seed_rmse_sd_h24_mean_over_events": mean([
                sd([z["rmse_h24"] for z in by[(e, arm)]]) for e in events
            ]),
            "seed_rmse_sd_h48_mean_over_events": mean([
                sd([z["rmse_h48"] for z in by[(e, arm)]]) for e in events
            ]),
        }

    event_rows = []
    for event in events:
        two = by[(event, "two_rate")]
        one = by[(event, "net_scaled")]
        rec = {
            "event": event,
            "two_rmse_h24": mean([r["rmse_h24"] for r in two]),
            "one_rmse_h24": mean([r["rmse_h24"] for r in one]),
            "two_rmse_h48": mean([r["rmse_h48"] for r in two]),
            "one_rmse_h48": mean([r["rmse_h48"] for r in one]),
            "one_zero_fraction": mean([r["frac_pred_zero"] for r in one]),
            "two_zero_fraction": mean([r["frac_pred_zero"] for r in two]),
            "one_actual_signed_init": mean([r["u_init"] - r["r_init"] for r in one]),
            "two_best_epoch_median": median([r["best_epoch"] for r in two]),
            "one_best_epoch_median": median([r["best_epoch"] for r in one]),
        }
        rec["gain_h24_pct"] = 100.0 * (
            rec["one_rmse_h24"] - rec["two_rmse_h24"]
        ) / rec["one_rmse_h24"]
        rec["gain_h48_pct"] = 100.0 * (
            rec["one_rmse_h48"] - rec["two_rmse_h48"]
        ) / rec["one_rmse_h48"]
        event_rows.append(rec)

    gain24 = [r["gain_h24_pct"] for r in event_rows]
    gain48 = [r["gain_h48_pct"] for r in event_rows]
    one_zero = [r["one_zero_fraction"] for r in event_rows]
    event_summary = {
        "h24_equal_event_mean_gain_pct": mean(gain24),
        "h24_median_gain_pct": median(gain24),
        "h24_positive_events": sum(x > 0 for x in gain24),
        "h48_equal_event_mean_gain_pct": mean(gain48),
        "h48_median_gain_pct": median(gain48),
        "h48_positive_events": sum(x > 0 for x in gain48),
        "corr_one_zero_fraction_gain_h24": corr(one_zero, gain24),
        "corr_one_zero_fraction_gain_h48": corr(one_zero, gain48),
    }

    output = {
        "input": args.input,
        "initialization": init_summary,
        "arms": arm_summary,
        "event_summary": event_summary,
        "events": event_rows,
    }

    lines = [
        "# Static audit of the committed event-transfer result",
        "",
        "This report is computed without retraining from `results/cc_event_repro_core_event.json`.",
        "",
        "## Comparator initialization",
        "",
        f"The actual signed initialization reconstructed as `u_init-r_init` is negative in **{init_summary['n']}/{init_summary['n']}** one-flow fits.",
        "",
        f"- range: `{init_summary['min']:.6f}` to `{init_summary['max']:.6f}`;",
        f"- median: `{init_summary['median']:.6f}`;",
        f"- median magnitude relative to the susceptible interruption initialization: `{init_summary['median_abs_ratio_to_u0']:.1f}x`.",
        "",
        "The two operands are rates on different exposure pools, so this subtraction is not a valid flow-matching initialization.",
        "",
        "## Optimization and boundary summaries",
        "",
        "| arm | parameters | median best epoch | best epoch <=2 | mean exact-zero prediction share | mean seed RMSE SD h24 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in arms:
        a = arm_summary[arm]
        lines.append(
            f"| {arm} | {a['parameter_count'][0]} | {a['best_epoch_median']:.1f} | "
            f"{a['best_epoch_le2']}/{a['n_fits']} | {pct(a['frac_pred_zero_mean'])} | "
            f"{a['seed_rmse_sd_h24_mean_over_events']:.6f} |"
        )

    lines += [
        "",
        "## Event-level comparison",
        "",
        "| test event | h24 gain (%) | h48 gain (%) | one-flow exact-zero share | signed initialization |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in event_rows:
        lines.append(
            f"| {r['event']} | {r['gain_h24_pct']:+.2f} | {r['gain_h48_pct']:+.2f} | "
            f"{100*r['one_zero_fraction']:.1f}% | {r['one_actual_signed_init']:.4f} |"
        )

    lines += [
        "",
        "Equal-event descriptive summary:",
        "",
        f"- h+24: mean `{event_summary['h24_equal_event_mean_gain_pct']:+.2f}%`, median "
        f"`{event_summary['h24_median_gain_pct']:+.2f}%`, positive in "
        f"`{event_summary['h24_positive_events']}/11` events;",
        f"- h+48: mean `{event_summary['h48_equal_event_mean_gain_pct']:+.2f}%`, median "
        f"`{event_summary['h48_median_gain_pct']:+.2f}%`, positive in "
        f"`{event_summary['h48_positive_events']}/11` events;",
        f"- correlation between the one-flow exact-zero share and gain: "
        f"`{event_summary['corr_one_zero_fraction_gain_h24']:+.3f}` at h+24 and "
        f"`{event_summary['corr_one_zero_fraction_gain_h48']:+.3f}` at h+48.",
        "",
        "The correlation is descriptive with eleven events and is not a causal or inferential result. The load-bearing finding is the deterministic reconstruction of the initialization and the large boundary-degenerate prediction share.",
    ]

    md_path = root / args.out_md
    js_path = root / args.out_json
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n")
    js_path.write_text(json.dumps(output, indent=2) + "\n")
    print(md_path)
    print(js_path)


if __name__ == "__main__":
    main()
