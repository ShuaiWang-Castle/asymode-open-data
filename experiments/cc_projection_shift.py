"""Section 10.2: real-data projection-shift diagnostic at K = 16 and K = 32.

Theorem 10.1 says the transfer penalty of a one-rate model fitted on environment P
and evaluated on environment Q is, per unit rate squared,

    interruption branch:  S_U(P,Q) = A_Q (C_P/A_P - C_Q/A_Q)^2
    restoration branch:   S_R(P,Q) = B_Q (C_P/B_P - C_Q/B_Q)^2

with A = E[(1-Y)^2], B = E[Y^2], C = E[Y(1-Y)] inside a driver-space cell. The true
rates cancel out of the comparison because the same functional is applied to both
splits, so this is a property of the state distributions alone; it is reported per
unit rate squared and never multiplied by an estimated rate.

The driver space is partitioned by k-means fitted on TRAINING rows only, after
standardising on training rows only. Two environments are compared:

* leave-one-event-out: Q is one whole event, P is every other event;
* pooled random control: Q is a random 20% of rows drawn stratified by event, so it
  has the same event mixture, P is the rest.

The diagnostic passes only if, for both K and both branches, the median
event-held-out shift exceeds the random-split shift.

    python experiments/cc_projection_shift.py --K 16 --out <dir>/11_PROJECTION_SHIFT_K16.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from asymode import panels as panelset  # noqa: E402
import exp05_real_dynamics as exp05  # noqa: E402

INTERIM = ROOT / "data/interim"
MIN_CELL = 30          # cells with fewer rows on either side are not evaluated


def moments(y):
    A = float(np.mean((1 - y) ** 2))
    B = float(np.mean(y ** 2))
    C = float(np.mean(y * (1 - y)))
    return A, B, C


def shift_terms(yP, yQ):
    AP, BP, CP = moments(yP)
    AQ, BQ, CQ = moments(yQ)
    su = AQ * (CP / AP - CQ / AQ) ** 2 if AP > 0 and AQ > 0 else np.nan
    sr = BQ * (CP / BP - CQ / BQ) ** 2 if BP > 0 and BQ > 0 else np.nan
    return su, sr


def cells_for(x, y, tr_idx, te_idx, K, seed):
    """k-means on training rows only; every row assigned to its nearest centroid."""
    from sklearn.cluster import KMeans
    mu, sd = x[tr_idx].mean(0), x[tr_idx].std(0) + 1e-9
    z = (x - mu) / sd
    km = KMeans(n_clusters=K, n_init=10, random_state=seed).fit(z[tr_idx])
    lab = km.predict(z)
    out = []
    for c in range(K):
        p = tr_idx[lab[tr_idx] == c]
        q = te_idx[lab[te_idx] == c]
        if len(p) >= MIN_CELL and len(q) >= MIN_CELL:
            su, sr = shift_terms(y[p], y[q])
            if np.isfinite(su) and np.isfinite(sr):
                out.append({"cell": c, "n_P": len(p), "n_Q": len(q), "S_U": su, "S_R": sr})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", default="configs/panel_manifest_g2-convective-11.json")
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--kmeans-seed", type=int, default=0)
    ap.add_argument("--control-seed", type=int, default=20260903)
    ap.add_argument("--n-control", type=int, default=11, help="random control splits, matched to the event count")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()

    want, panel_digest = panelset.resolve(INTERIM, str(ROOT / a.panels))
    y0, X, yt, m, fips, panel, origin, t0h = exp05.load_pooled(48, 12, panels=want, with_time=True)
    X = exp05.add_context(X, y0, 48, t0_hour=t0h, clock="utc_hour")
    x = X[:, 0, :].astype(np.float64)          # drivers at the first forecast step
    y = y0.astype(np.float64)                   # state at the origin
    events = sorted(set(panel.tolist()))
    print(f"K={a.K} · {len(y):,} origin rows · {len(events)} events · panels {panel_digest}")

    ev_rows, ctl_rows = [], []
    for e in events:
        te = np.where(panel == e)[0]
        tr = np.where(panel != e)[0]
        cs = cells_for(x, y, tr, te, a.K, a.kmeans_seed)
        for c in cs:
            ev_rows.append(dict(c, split="event", held_out=e))
        print(f"  event {e}: {len(cs)} usable cells", flush=True)

    rng = np.random.default_rng(a.control_seed)
    for rep in range(a.n_control):
        te = np.concatenate([rng.choice(np.where(panel == e)[0],
                                        max(1, int(0.2 * (panel == e).sum())), replace=False)
                             for e in events])            # same event mixture as the pool
        tr = np.setdiff1d(np.arange(len(y)), te)
        cs = cells_for(x, y, tr, te, a.K, a.kmeans_seed)
        for c in cs:
            ctl_rows.append(dict(c, split="random", held_out=f"rep{rep}"))
    print(f"  random control: {len(ctl_rows)} usable cells over {a.n_control} reps")

    def summarise(rows, key):
        v = np.array([r[key] for r in rows])
        return {"n_cells": len(v), "median": float(np.median(v)), "mean": float(np.mean(v)),
                "p25": float(np.quantile(v, 0.25)), "p75": float(np.quantile(v, 0.75))}

    res = {"config": {"K": a.K, "kmeans_seed": a.kmeans_seed, "control_seed": a.control_seed,
                      "n_control_reps": a.n_control, "min_cell": MIN_CELL,
                      "panel_digest": panel_digest, "events": events,
                      "statistic": "per unit rate squared: S_U = A_Q (C_P/A_P - C_Q/A_Q)^2, "
                                   "S_R = B_Q (C_P/B_P - C_Q/B_Q)^2",
                      "wall_time_s": round(time.time() - t0, 1)},
           "event_held_out": {b: summarise(ev_rows, b) for b in ("S_U", "S_R")},
           "random_split": {b: summarise(ctl_rows, b) for b in ("S_U", "S_R")},
           "per_event_median": {e: {b: float(np.median([r[b] for r in ev_rows if r["held_out"] == e]))
                                    for b in ("S_U", "S_R")} for e in events},
           "cells": ev_rows + ctl_rows}
    res["pass"] = {b: bool(res["event_held_out"][b]["median"] > res["random_split"][b]["median"])
                   for b in ("S_U", "S_R")}
    res["verdict"] = "PASS" if all(res["pass"].values()) else "FAIL"

    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    for b, name in (("S_U", "interruption-only"), ("S_R", "restoration-only")):
        e, c = res["event_held_out"][b], res["random_split"][b]
        print(f"\n{name}: event-held-out median {e['median']:.3e} (n={e['n_cells']}) vs "
              f"random-split median {c['median']:.3e} (n={c['n_cells']})  ->  "
              f"{'event LARGER (pass)' if res['pass'][b] else 'event NOT larger (fail)'}")
    print(f"\nverdict: {res['verdict']}\nwritten: {a.out}")


if __name__ == "__main__":
    main()
