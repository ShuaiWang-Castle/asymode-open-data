"""EXP06 -- does the dynamical form matter more when the dynamics are slower?

The primary study (EXP05, manifest g2-convective-11) fitted the model on eleven
convective storm days. Convective events have fast onset and fast restoration,
and on them the susceptible form beat the epidemic form and the best statistical
baseline at 24 and 48 hours but not at 1 or 6. The question is whether that
horizon pattern is a property of the model or of the event type sampled.

This experiment fits the identical protocol separately on each event family and
reports the horizon profile per family.

REGISTRATION -- consolidated 2026-09-01, before any family-stratified fit.

Two registrations covered this experiment with different comparators. That is a
forking path: whichever passed would be the one reported. They are merged here
and the merge is recorded, not hidden.

  * The structural claim -- two rates versus one, and whether the advantage
    tracks phase separation -- is registered ONLY in
    docs/PREREGISTRATION_phase_separation.md (H-E1, H-E2), against the
    parameter-matched single net-rate arm. Nothing below duplicates it.
  * What remains below concerns the EPIDEMIC family specifically: a different
    opponent and a different question, so no conflict.

  H1  On the slow families -- tropical and wind, whose measured fall/rise ratios
      are 4.0 and 2.1 -- the susceptible arm beats the pure epidemic arm at h+24
      and h+48 in at least 12 of 15 (fold, seed) units, with a paired t below -3.
      (An earlier draft listed winter as slow. The phase-separation measurement
      shows winter is the FASTEST family, ratio 1.0; it is the negative-control
      end of H-E, not a slow family. Corrected before running.)
  H2  VOID. It compared the susceptible arm to damped persistence across families
      -- the same question as H-E1/H-E2 with a comparator whose difficulty drifts
      by family in the direction that fakes the result. Superseded by H-E.
  H3  The seeded epidemic arm's fitted eps, per family, relative to that family's
      mean observed state. Registered prediction: eps dominates the inflow term on
      most scored cells (eps / (y + eps) > 0.9) on every family. NOTE that on the
      primary convective run eps was 0.85x the mean state, not above it, while
      still dominating 76% of scored cells -- so the operative test is the
      dominance share, not eps > mean. Kill: H3 dies on any family where eps
      dominates fewer than half of scored cells.

  Kill conditions. H1 dies if either slow family fails the 12/15 bar at either
  horizon. H3 as above.

  A null result is a result: if the horizon profile is the same on every family,
  the short-horizon weakness is structural, not an artefact of sampling
  convective storms, and the paper must say so and change its metric rather than
  its data.

Family skips (a fold with too few test or train counties) are recorded in the
result JSON, never silent; H-E's void condition -- fewer than three families with
a full-protocol result -- must be decidable from the JSON alone.
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
from asymode import panels as panelset                              # noqa: E402

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
    ap.add_argument("--panels", default="auto",
                    help="defaults to every built panel: this experiment needs the "
                         "families the primary manifest deliberately excludes. Pass "
                         "a manifest path once the generalisation set is pinned.")
    ap.add_argument("--out", default="results/exp06_by_family.json")
    a = ap.parse_args()

    fam = family_of_day()
    want, panel_digest = panelset.resolve(INTERIM, a.panels)
    y0, X, yt, m, fips, panel = load_pooled(a.horizon, a.stride, panels=want)
    X = add_context(X, y0, a.horizon)
    # Days absent from the stratified catalog are the original convective set.
    famv = np.array([fam.get(p, "convective") for p in panel])
    print(f"pooled {len(y0):,} samples over {len(set(panel))} storms")
    for f in sorted(set(famv)):
        print(f"  {f:<12} {int((famv==f).sum()):>7,} samples  "
              f"{len(set(panel[famv==f])):>2} storms  "
              f"{len(set(fips[famv==f])):>5} counties")

    # Every family and fold that does not produce a fitted result is recorded.
    # H-E is void below three families with a complete protocol result, and that
    # has to be decidable from this file rather than by counting rows and hoping
    # the absences were all deliberate.
    out_rows, skipped = [], []
    for f in a.families:
        sel = np.where(famv == f)[0]
        if len(sel) < 500:
            print(f"\n{f}: only {len(sel)} samples, skipped")
            skipped.append({"family": f, "seed": None, "fold": None,
                            "n_samples": int(len(sel)),
                            "reason": "family below 500 pooled samples"})
            continue
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
                    skipped.append({"family": f, "seed": seed, "fold": k,
                                    "n_test": int(len(te)), "n_train": int(len(tr)),
                                    "reason": "fold below the 20 test / 100 train floor"})
                    continue
                for b in BASELINES:
                    r = run_baseline(b, tr, te, (sy0, sX, syt, sm), a)
                    out_rows.append({"family": f, "arm": b, "seed": seed,
                                     "fold": k, "n_test": len(te), **r})
                for arm in ARMS:
                    t0 = time.time()
                    r = run_arm(arm, tr, te, (sy0, sX, syt, sm), a, seed, sf, k)
                    out_rows.append({"family": f, "arm": arm.name, "seed": seed,
                                     "inflow": arm.inflow.value,
                                     "fold": k, "n_test": len(te),
                                     "wall_s": round(time.time() - t0, 1), **r})
                print(f"  seed {seed} fold {k} done", flush=True)

    dst = ROOT / a.out; dst.parent.mkdir(parents=True, exist_ok=True)
    cfg = dict(vars(a)); cfg["out"] = a.out
    cfg["panels"] = sorted(set(panel.tolist()))
    cfg["panel_digest"] = panel_digest
    cfg["channels"] = panelset.channel_names(INTERIM)
    cfg["channel_digest"] = panelset.channel_digest(cfg["channels"])
    cfg["source"] = panelset.source_version(ROOT)
    dst.write_text(json.dumps({"config": cfg, "rows": out_rows,
                               "skipped": skipped}, indent=2))
    if skipped:
        print(f"\n{len(skipped)} family/fold combinations produced no fit:")
        for sk in skipped:
            print(f"  {sk['family']:<12} seed {sk['seed']} fold {sk['fold']}: {sk['reason']}")

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
