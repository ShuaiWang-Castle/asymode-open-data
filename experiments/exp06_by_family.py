"""EXP06 -- does the dynamical form matter more when the dynamics are slower?

EXP05 fitted the model on eight storm days that were, without exception,
convective. Convective events have the fastest onset and the fastest restoration
in the record, and on them the susceptible form beat the epidemic form and the
best statistical baseline at 24 and 48 hours but not at 1 or 6. The obvious
question is whether that horizon pattern is a property of the model or of the
event type that was sampled.

This experiment answers it by fitting the identical protocol separately on each
event family and reporting the horizon profile per family.

PRE-REGISTERED, written before any family-stratified fit was run. The previous
criteria were written for marginal comparisons and were adjudicated on a paired
design, which was a mistake; these are written for the paired design that the
data actually has.

  H1  On slower families (winter, tropical, wind), the susceptible arm beats the
      pure epidemic arm at h+24 and h+48 in at least 12 of 15 (fold, seed) units,
      with a paired t below -3.
  H2  The horizon at which the susceptible arm first beats damped persistence is
      EARLIER on slow families than on convective. Operationalised: the paired
      difference at h+6 is negative on winter and tropical, having been positive
      on convective.
  H3  The seeded epidemic arm's fitted eps again exceeds the mean observed state,
      on every family, confirming that it competes only by degenerating.

  Kill conditions. H1 dies if any slow family fails the 12/15 bar at either
  horizon. H2 dies if the h+6 paired difference is positive on winter. H3 dies if
  eps falls below the mean state on any family.

  A null result is a result: if the horizon profile is the same on every family,
  then the short-horizon weakness is structural, not an artefact of sampling
  convective storms, and the paper must say so and change its metric rather than
  its data.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from asymode.dynamics import InflowForm                              # noqa: E402
from asymode.evalproto import make_folds                             # noqa: E402

sys.path.insert(0, str(ROOT / "experiments"))
from exp05_real_dynamics import (load_pooled, add_context, run_arm,   # noqa: E402
                                 run_baseline, BASELINES, ARMS)

INTERIM = ROOT / "data" / "interim"


def family_of_day() -> dict:
    """Dominant event family per storm day, from the public catalog."""
    f = INTERIM / "event_days_stratified.parquet"
    if not f.exists():
        return {}
    d = pd.read_parquet(f)
    return {str(k.date()): v for k, v in zip(d["day"], d["dominant"])}


def paired(rows, a, b, h):
    key = lambda r: (r["seed"], r["fold"])
    A = {key(r): r for r in rows if r["arm"] == a}
    B = {key(r): r for r in rows if r["arm"] == b}
    ks = sorted(set(A) & set(B))
    if not ks:
        return None
    d = np.array([A[k][f"rmse_h{h}"] - B[k][f"rmse_h{h}"] for k in ks])
    sd = d.std(ddof=1)
    t = d.mean() / (sd / np.sqrt(len(d))) if sd > 0 else np.nan
    rel = 100 * d.mean() / np.mean([B[k][f"rmse_h{h}"] for k in ks])
    return {"delta": float(d.mean()), "rel_pct": float(rel),
            "wins": int((d < 0).sum()), "n": len(d), "t": float(t)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=48)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 6, 24, 48])
    ap.add_argument("--stride", type=int, default=12)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--cap-u", type=float, default=0.25)
    ap.add_argument("--cap-r", type=float, default=0.25)
    ap.add_argument("--families", nargs="+",
                    default=["convective", "winter", "tropical", "wind"])
    ap.add_argument("--out", default="results/exp06_by_family.json")
    a = ap.parse_args()

    fam = family_of_day()
    y0, X, yt, m, fips, panel = load_pooled(a.horizon, a.stride)
    X = add_context(X, y0, a.horizon)
    # Days absent from the stratified catalog are the original convective set.
    famv = np.array([fam.get(p, "convective") for p in panel])
    print(f"pooled {len(y0):,} samples over {len(set(panel))} storms")
    for f in sorted(set(famv)):
        print(f"  {f:<12} {int((famv==f).sum()):>7,} samples  "
              f"{len(set(panel[famv==f])):>2} storms  "
              f"{len(set(fips[famv==f])):>5} counties")

    out_rows = []
    for f in a.families:
        sel = np.where(famv == f)[0]
        if len(sel) < 500:
            print(f"\n{f}: only {len(sel)} samples, skipped"); continue
        print(f"\n=== {f}: {len(sel):,} samples ===", flush=True)
        sy0, sX, syt, sm = y0[sel], X[sel], yt[sel], m[sel]
        sf = fips[sel]
        uniq = sorted(set(sf))
        for seed in a.seeds:
            fold = make_folds(uniq, k=a.k, seed=seed)
            fmap = dict(zip(uniq, fold))
            assign = np.array([fmap[x] for x in sf])
            for k in range(a.k):
                te = np.where(assign == k)[0]; tr = np.where(assign != k)[0]
                if len(te) < 20 or len(tr) < 100:
                    continue
                for b in BASELINES:
                    r = run_baseline(b, tr, te, (sy0, sX, syt, sm), a)
                    out_rows.append({"family": f, "arm": b, "seed": seed,
                                     "fold": k, "n_test": len(te), **r})
                for arm in ARMS:
                    t0 = time.time()
                    r = run_arm(arm, tr, te, (sy0, sX, syt, sm), a, seed)
                    out_rows.append({"family": f, "arm": arm.value, "seed": seed,
                                     "fold": k, "n_test": len(te),
                                     "wall_s": round(time.time() - t0, 1), **r})
                print(f"  seed {seed} fold {k} done", flush=True)

    dst = ROOT / a.out; dst.parent.mkdir(parents=True, exist_ok=True)
    cfg = dict(vars(a)); cfg["out"] = a.out
    dst.write_text(json.dumps({"config": cfg, "rows": out_rows}, indent=2))

    print("\n" + "=" * 78)
    print("PAIRED: susceptible vs comparator, by family. negative = ours better")
    for f in a.families:
        fr = [r for r in out_rows if r["family"] == f]
        if not fr:
            continue
        print(f"\n--- {f} ---")
        for comp in ["transmission", "transmission_seed", "damped_persistence"]:
            line = f"  vs {comp:<20}"
            for h in a.horizons:
                p = paired(fr, "susceptible", comp, h)
                line += f" h{h}: {p['rel_pct']:+5.1f}% {p['wins']}/{p['n']} t={p['t']:+5.1f} |" if p else ""
            print(line)
        e = [r.get("fitted_seed_eps") for r in fr
             if r["arm"] == "transmission_seed" and r.get("fitted_seed_eps")]
        ys = sy0 if False else None
        if e:
            print(f"  fitted eps {np.mean(e):.5f}   mean state y0 "
                  f"{y0[famv == f].mean():.5f}   "
                  f"{'eps > state (degenerate)' if np.mean(e) > y0[famv==f].mean() else 'eps < state'}")
    print(f"\nwritten: {a.out}")


if __name__ == "__main__":
    main()
