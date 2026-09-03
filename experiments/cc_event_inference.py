"""Sections 7-9: event-level estimands and inference for the concurrency gain.

The event is the statistical unit. Seeds are averaged inside each event and never
counted as extra events. Reads the per-fit records written by
`cc_event_transfer.py` and reports, per horizon:

* g_{e,h} = 100 * (Rbar_net - Rbar_two) / Rbar_net, positive when net_scaled is worse;
* equal-event mean and median, and the pooled-cell RMSE gain as a secondary estimand;
* exact two-sided sign test;
* exact paired randomization over all 2^11 model-label swaps;
* paired event bootstrap (>= 50,000 resamples), percentile 95% interval;
* leave-one-event-out means and the largest single-event contribution;
* if a county-held-out run is given, d_{e,h} = g_event - g_county with the same tests.

    python experiments/cc_event_inference.py --event <core_event.json> \
        --county <core_county.json> --out-dir results/event_transfer_confirmatory_20260903
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = (1, 6, 24, 48)
PRIMARY = (24, 48)


def event_rmse(rows, arm, event, h, seeds):
    """Mean over seeds of the per-event RMSE at horizon h."""
    v = [r[f"rmse_h{h}"] for r in rows if r["arm"] == arm and r["test_event"] == event and r["seed"] in seeds]
    return float(np.mean(v)) if v else float("nan")


def per_event_from_predictions(pred_dir: Path, split: str, arm: str, seeds) -> dict:
    """Per-event, seed-averaged RMSE straight from the archived OOF predictions.

    Used for the county protocol, where a test event is spread over several folds,
    so the per-fit records are not per-event.
    """
    out = {}
    acc = None
    for s in seeds:
        f = pred_dir / f"pred_{split}_{arm}_seed{s}.npz"
        if not f.exists():
            return {}
        z = np.load(f, allow_pickle=True)
        p, y, m, ev = z["pred"], z["y"], z["mask"], z["panel"]
        if acc is None:
            acc = {e: {h: [] for h in HORIZONS} for e in sorted(set(ev.tolist()))}
        for e in acc:
            sel = ev == e
            for hi, h in enumerate(HORIZONS):
                k = sel & m[:, hi] & np.isfinite(p[:, hi])
                if k.any():
                    acc[e][h].append(float(np.sqrt(np.mean((p[k, hi] - y[k, hi]) ** 2))))
    for e, d in acc.items():
        out[e] = {h: float(np.mean(v)) if v else float("nan") for h, v in d.items()}
    return out


def gains(rmse_two: dict, rmse_net: dict, h: int) -> dict:
    return {e: 100.0 * (rmse_net[e][h] - rmse_two[e][h]) / rmse_net[e][h]
            for e in rmse_two if np.isfinite(rmse_two[e][h]) and np.isfinite(rmse_net[e][h])}


def sign_test(v) -> float:
    """Exact two-sided sign test, zeros dropped."""
    v = [x for x in v if x != 0]
    n, k = len(v), sum(1 for x in v if x > 0)
    if n == 0:
        return float("nan")
    c = lambda i: math.comb(n, i)
    p_le = sum(c(i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return float(min(1.0, 2 * p_le))


def randomization_test(v) -> float:
    """Exact paired randomization: every model-label swap flips one event's sign."""
    v = np.asarray(v, float)
    n = len(v)
    if n > 22:
        raise ValueError("exact enumeration is only sensible for small n")
    obs = abs(v.mean())
    signs = np.array(list(itertools.product([1, -1], repeat=n)))
    means = np.abs(signs @ v) / n
    return float((means >= obs - 1e-12).mean())


def bootstrap_ci(v, B=50000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(v, float)
    n = len(v)
    m = v[rng.integers(0, n, (B, n))].mean(1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975)), float(m.mean())


def loeo(v):
    v = np.asarray(v, float)
    return [float((v.sum() - x) / (len(v) - 1)) for x in v]


def analyse(g: dict, label: str, B: int, seed: int) -> dict:
    ev = sorted(g)
    v = [g[e] for e in ev]
    lo, hi, bm = bootstrap_ci(v, B, seed)
    lo_means = loeo(v)
    return {
        "label": label, "events": ev, "gains": {e: round(g[e], 4) for e in ev},
        "n_events": len(v), "n_positive": int(sum(1 for x in v if x > 0)),
        "mean": float(np.mean(v)), "median": float(np.median(v)),
        "bootstrap_mean": bm, "ci95": [lo, hi], "bootstrap_B": B,
        "sign_test_p": sign_test(v), "randomization_p": randomization_test(v),
        "loeo_means": {e: round(m, 4) for e, m in zip(ev, lo_means)},
        "loeo_min": float(min(lo_means)), "loeo_all_positive": bool(min(lo_means) > 0),
        "largest_contributor": ev[int(np.argmax(np.abs(v)))],
        "sign_changes_without_largest": bool(
            (np.mean(v) > 0) != (lo_means[int(np.argmax(np.abs(v)))] > 0)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", required=True)
    ap.add_argument("--county", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--bootstrap", type=int, default=50000)
    ap.add_argument("--bootstrap-seed", type=int, default=20260903)
    a = ap.parse_args()
    outd = ROOT / a.out_dir
    outd.mkdir(parents=True, exist_ok=True)
    pred_dir = outd / "predictions"

    ed = json.loads((ROOT / a.event).read_text())
    seeds = ed["config"]["model_seeds"]
    rows = ed["rows"]
    events = sorted({r["test_event"] for r in rows})
    two = {e: {h: event_rmse(rows, "two_rate", e, h, seeds) for h in HORIZONS} for e in events}
    net = {e: {h: event_rmse(rows, "net_scaled", e, h, seeds) for h in HORIZONS} for e in events}

    result = {"config": {"event_run": a.event, "county_run": a.county, "seeds": seeds,
                         "bootstrap": a.bootstrap, "bootstrap_seed": a.bootstrap_seed,
                         "event_split_digest": ed["config"]["outer_split_digest"],
                         "panel_digest": ed["config"]["panel_digest"]},
              "event_held_out": {}, "county_held_out": {}, "difference": {}}

    print("=" * 96)
    print("EVENT-HELD-OUT concurrency gain g_e,h = 100*(RMSE_net - RMSE_two)/RMSE_net, positive = net_scaled worse")
    for h in HORIZONS:
        g = gains(two, net, h)
        r = analyse(g, f"event h+{h}", a.bootstrap, a.bootstrap_seed)
        result["event_held_out"][h] = r
        star = "PRIMARY" if h in PRIMARY else "descriptive"
        print(f"\nh+{h} [{star}]  mean {r['mean']:+.2f}%  median {r['median']:+.2f}%  "
              f"{r['n_positive']}/{r['n_events']} positive  CI95 [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}]  "
              f"sign p={r['sign_test_p']:.4f}  rand p={r['randomization_p']:.4f}  "
              f"LOEO min {r['loeo_min']:+.2f}%")
        for e in r["events"]:
            print(f"      {e}: {r['gains'][e]:+7.2f}%")

    # pooled-cell secondary estimand
    pooled = {}
    for arm in ("two_rate", "net_scaled"):
        for h in HORIZONS:
            num = sum(r[f"mse_h{h}"] * r[f"n_h{h}"] for r in rows if r["arm"] == arm)
            den = sum(r[f"n_h{h}"] for r in rows if r["arm"] == arm)
            pooled.setdefault(arm, {})[h] = math.sqrt(num / den)
    result["pooled_cell_rmse"] = {arm: {h: pooled[arm][h] for h in HORIZONS} for arm in pooled}
    result["pooled_gain_pct"] = {h: 100 * (pooled["net_scaled"][h] - pooled["two_rate"][h]) / pooled["net_scaled"][h]
                                 for h in HORIZONS}
    print("\npooled-cell (size-weighted, secondary): " +
          "  ".join(f"h+{h} {result['pooled_gain_pct'][h]:+.2f}%" for h in HORIZONS))

    # ---- county control and the direct difference ----------------------------
    if a.county:
        cd = json.loads((ROOT / a.county).read_text())
        cseeds = cd["config"]["model_seeds"]
        ctwo = per_event_from_predictions(pred_dir, "county", "two_rate", cseeds)
        cnet = per_event_from_predictions(pred_dir, "county", "net_scaled", cseeds)
        if not ctwo or not cnet:
            print("\ncounty predictions not found; skipping the difference test")
        else:
            print("\n" + "=" * 96)
            print("COUNTY-HELD-OUT (secondary control) and the direct difference d_e,h = g_event - g_county")
            for h in HORIZONS:
                gc = gains(ctwo, cnet, h)
                rc = analyse(gc, f"county h+{h}", a.bootstrap, a.bootstrap_seed)
                result["county_held_out"][h] = rc
                ge = result["event_held_out"][h]["gains"]
                common = sorted(set(gc) & set(ge))
                d = {e: ge[e] - gc[e] for e in common}
                rd = analyse(d, f"difference h+{h}", a.bootstrap, a.bootstrap_seed)
                result["difference"][h] = rd
                print(f"\nh+{h}  county mean {rc['mean']:+.2f}% ({rc['n_positive']}/{rc['n_events']})  "
                      f"CI95 [{rc['ci95'][0]:+.2f}, {rc['ci95'][1]:+.2f}]")
                print(f"      d mean {rd['mean']:+.2f}%  median {rd['median']:+.2f}%  "
                      f"{rd['n_positive']}/{rd['n_events']} positive  "
                      f"CI95 [{rd['ci95'][0]:+.2f}, {rd['ci95'][1]:+.2f}]  rand p={rd['randomization_p']:.4f}  "
                      f"LOEO min {rd['loeo_min']:+.2f}")

    # ---- gates ---------------------------------------------------------------
    gate = {}
    for h in PRIMARY:
        r = result["event_held_out"][h]
        gate[f"core_h{h}"] = {
            "mean_gain_positive": bool(r["mean"] > 0),
            "bootstrap_lower_bound_positive": bool(r["ci95"][0] > 0),
            "randomization_p_lt_0.05": bool(r["randomization_p"] < 0.05),
            "at_least_9_of_11_positive": bool(r["n_positive"] >= 9),
            "every_loeo_mean_positive": bool(r["loeo_all_positive"]),
        }
        gate[f"core_h{h}"]["CONFIRMED"] = all(gate[f"core_h{h}"].values())
        if h in result["difference"]:
            d = result["difference"][h]
            gate[f"strengthening_h{h}"] = {
                "mean_positive": bool(d["mean"] > 0),
                "bootstrap_lower_bound_positive": bool(d["ci95"][0] > 0),
                "at_least_8_of_11_positive": bool(d["n_positive"] >= 8),
                "loeo_mean_positive": bool(d["loeo_all_positive"]),
            }
            gate[f"strengthening_h{h}"]["CONFIRMED"] = all(gate[f"strengthening_h{h}"].values())
    result["gates"] = gate
    print("\n" + "=" * 96)
    print(json.dumps(gate, indent=1))

    (outd / "08_EVENT_INFERENCE.json").write_text(json.dumps(result, indent=1))
    # CSV deliverables
    with (outd / "04_CORE_EVENT_RESULTS.csv").open("w") as f:
        f.write("event,horizon,rmse_two_rate,rmse_net_scaled,gain_pct\n")
        for e in events:
            for h in HORIZONS:
                f.write(f"{e},{h},{two[e][h]:.8f},{net[e][h]:.8f},"
                        f"{100 * (net[e][h] - two[e][h]) / net[e][h]:.6f}\n")
    with (outd / "05_CORE_SEED_RESULTS.csv").open("w") as f:
        keys = ["test_event", "validation_event", "seed", "arm", "parameter_count", "best_epoch",
                "stopped_at_epoch", "hit_epoch_cap", "training_time_s", "final_validation",
                "u_init", "r_init", "pred_sd", "frac_pred_zero"] + [f"rmse_h{h}" for h in HORIZONS]
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
    with (outd / "09_LOEO_INFLUENCE.csv").open("w") as f:
        f.write("scope,horizon,excluded_event,mean_without_it\n")
        for scope in ("event_held_out", "difference"):
            for h, r in result.get(scope, {}).items():
                for e, m in r["loeo_means"].items():
                    f.write(f"{scope},{h},{e},{m:.6f}\n")
    if result["county_held_out"]:
        with (outd / "06_COUNTY_SPLIT_RESULTS.csv").open("w") as f:
            f.write("event,horizon,rmse_two_rate,rmse_net_scaled,gain_pct\n")
            for h in HORIZONS:
                for e, gv in result["county_held_out"][h]["gains"].items():
                    f.write(f"{e},{h},{ctwo[e][h]:.8f},{cnet[e][h]:.8f},{gv:.6f}\n")
        with (outd / "07_SPLIT_DIFFERENCE_RESULTS.csv").open("w") as f:
            f.write("event,horizon,g_event,g_county,d\n")
            for h in HORIZONS:
                for e, dv in result["difference"][h]["gains"].items():
                    f.write(f"{e},{h},{result['event_held_out'][h]['gains'][e]:.6f},"
                            f"{result['county_held_out'][h]['gains'][e]:.6f},{dv:.6f}\n")
    print(f"\nwritten: {a.out_dir}/08_EVENT_INFERENCE.json and the CSV deliverables")


if __name__ == "__main__":
    main()
