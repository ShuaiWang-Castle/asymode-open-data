"""Event-level paired comparison from out-of-fold predictions (E3).

The inferential unit is the storm event (panel). For a pair of arms scored on
identical samples, this reports per horizon: every event's paired MSE difference,
the mean and median event effect, an event-cluster bootstrap interval, the
leave-one-event-out range of the mean, and -- separately, as an optimisation
diagnostic only -- the seed sign consistency. No fold x seed t-test is computed.

    python scripts/event_level_review.py results/oof_trees_matched.npz results/oof_susceptible.npz

Positive delta = arm A worse (higher MSE) than arm B. Both exports must share
samples exactly (fips, panel, origin_id, mask), which is asserted, and their
split/clock digests are printed so a legacy export cannot pass as a corrected one.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np


def load(p):
    z = np.load(p, allow_pickle=True)
    d = {k: z[k] for k in z.files}
    return d


def same_samples(a, b):
    for k in ("fips", "panel", "origin_id"):
        if not np.array_equal(a[k], b[k]):
            sys.exit(f"exports differ in {k}: not the same samples")
    if not np.array_equal(a["mask"], b["mask"]):
        sys.exit("exports differ in mask")


def event_effects(a, b, h_idx, seed_idx):
    """Per-event paired MSE difference at one horizon, one seed (or seed-mean)."""
    m = a["mask"][:, h_idx].astype(bool)
    y = a["y"][:, h_idx]
    pa = a["pred"][seed_idx, :, h_idx] if seed_idx is not None else a["pred"][:, :, h_idx].mean(0)
    pb = b["pred"][seed_idx, :, h_idx] if seed_idx is not None else b["pred"][:, :, h_idx].mean(0)
    d = (pa - y) ** 2 - (pb - y) ** 2
    ev = a["panel"]
    out = {}
    for e in np.unique(ev):
        sel = (ev == e) & m
        if sel.sum():
            out[str(e)] = float(d[sel].mean())
    return out


def cluster_bootstrap(vals, B=2000, seed=0):
    rng = np.random.default_rng(seed); v = np.asarray(vals); n = len(v)
    bs = np.array([v[rng.integers(0, n, n)].mean() for _ in range(B)])
    return float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--B", type=int, default=2000); ap.add_argument("--bootstrap-seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    x = ap.parse_args()
    A, Bx = load(x.a), load(x.b)
    same_samples(A, Bx)
    hs = A["horizons"].tolist(); seeds = A["seeds"].tolist()
    meta = {k: str(A[k]) for k in ("panel_digest", "channel_digest") if k in A}
    meta.update({k: str(A[k]) for k in ("split_unit", "outer_split_digest", "clock_digest") if k in A})
    if "outer_split_digest" not in A or not str(A.get("outer_split_digest", "")):
        meta["protocol"] = "LEGACY export (county folds keyed on the model seed; no clock/split digest) -- descriptive only"
    else:
        for k in ("split_unit", "outer_split_digest", "clock_digest"):
            if str(A[k]) != str(Bx[k]):
                sys.exit(f"the two exports disagree on {k}: {A[k]} vs {Bx[k]}")
        meta["protocol"] = f"protocol v{int(A.get('schema_version', 2))}, split_unit={A['split_unit']}"
    print(f"{Path(x.a).name} vs {Path(x.b).name} · events {len(np.unique(A['panel']))} · seeds {seeds} · {meta}")
    print("positive = first arm worse\n")
    table = {"meta": meta, "horizons": {}}
    for hi, h in enumerate(hs):
        eff = event_effects(A, Bx, hi, None)            # seed-averaged predictions
        v = np.array(list(eff.values())); names = list(eff)
        lo, hi_ = cluster_bootstrap(v, x.B, x.bootstrap_seed)
        loo = [np.delete(v, i).mean() for i in range(len(v))]
        # seed sign consistency: fraction of (seed, event) cells with positive delta
        signs = []
        for si in range(len(seeds)):
            es = event_effects(A, Bx, hi, si)
            signs += [es[e] > 0 for e in names]
        base = float(((Bx["pred"][:, :, hi].mean(0) - A["y"][:, hi]) ** 2)[A["mask"][:, hi].astype(bool)].mean())
        rel = 100 * v.mean() / base
        print(f"h+{h}: mean event dMSE {v.mean():+.3e} ({rel:+.1f}% of B's MSE) · median {np.median(v):+.3e} · "
              f"events worse {int((v > 0).sum())}/{len(v)} · event-cluster 95% [{lo:+.3e}, {hi_:+.3e}] · "
              f"LOO mean range [{min(loo):+.3e}, {max(loo):+.3e}] · seed-cells positive {np.mean(signs)*100:.0f}%")
        for e, val in sorted(eff.items(), key=lambda kv: kv[1]):
            print(f"      {e}: {val:+.3e}")
        table["horizons"][h] = {"per_event": eff, "mean": float(v.mean()), "median": float(np.median(v)),
                                "rel_pct": rel, "ci95": [lo, hi_], "loo_range": [float(min(loo)), float(max(loo))],
                                "events_worse": int((v > 0).sum()), "n_events": len(v),
                                "seed_cells_positive_frac": float(np.mean(signs))}
    if x.json:
        Path(x.json).write_text(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
