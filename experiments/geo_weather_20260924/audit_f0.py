"""F0 variance audit (DESIGN v1 section 3; no training): how much the exposure-integrated hazard features change
along the recoverability ladder, and whether they change where the outages are.

For each contrast (quad - pop: sub-grid bands; pop - area: exposure weighting; pooled - mean: the static band
distribution under county-mean weather; quad - pooled: co-location) and each instantaneous feature (psi x
modulator at tau = 0), over the forecast hours of all county-events:
  scale      the 99th percentile of the reference variant over its active county-hours
  material   |difference| > 0.1 scale
  share      share of county-hours with material difference (unweighted; and weighted by the observed outage
             fraction, i.e. where the outages are)
  events     events with material difference in at least 1% of their county-hours
  corr       correlation of the two variants over county-hours where either is active
Output: results/F0/f0_audit.json and results/F0/f0_audit.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GW = ROOT / "data" / "interim" / "geo_weather"
FEAT = ROOT / "data" / "interim" / "open_gcrk" / "features_e3r2.npz"
PAIRS = [("quad", "pop"), ("pop", "area"), ("pooled", "mean"), ("quad", "pooled"), ("quad", "mean")]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default=str(FEAT.relative_to(ROOT)))
    ap.add_argument("--prefix", default="", help="eih_<prefix><variant>.npz")
    ap.add_argument("--out", default="F0")
    a = ap.parse_args()
    F = np.load(ROOT / a.feat)
    y, m, ev = np.nan_to_num(F["y"]).astype(np.float64), F["m"].astype(bool), F["event"].astype(str)
    have = {v: GW / f"eih_{a.prefix}{v}.npz" for v in ("area", "pop", "quad", "mean", "pooled", "other", "quadn")}
    have = {k: p for k, p in have.items() if p.exists()}
    names = np.load(next(iter(have.values())))["names"].astype(str)
    inst = [i for i, n in enumerate(names) if n.endswith("@0")]
    Z = {k: np.load(p)["phi"][..., inst].astype(np.float32) for k, p in have.items()}
    w = np.where(m, y, 0.0)
    rows = []
    for va, vb in PAIRS:
        if va not in Z or vb not in Z:
            continue
        for j, i in enumerate(inst):
            A, B = Z[va][..., j], Z[vb][..., j]
            act = ((A > 1e-6) | (B > 1e-6)) & m
            if act.sum() < 100:
                continue
            scale = float(np.percentile(B[(B > 1e-6) & m], 99)) if ((B > 1e-6) & m).any() else float(np.percentile(A[act], 99))
            d = np.abs(A - B) / max(scale, 1e-9)
            mat = (d > 0.1) & m
            per_ev = {e: float(mat[ev == e].sum() / max(m[ev == e].sum(), 1)) for e in sorted(set(ev))}
            rows.append(dict(contrast=f"{va}-{vb}", feature=names[i], scale=scale,
                             share=float(mat.sum() / m.sum()), share_outage_weighted=float((w * mat).sum() / w.sum()),
                             events_ge_1pct=int(sum(v >= 0.01 for v in per_ev.values())),
                             corr=float(np.corrcoef(A[act], B[act])[0, 1]),
                             mean_abs_rel_diff_active=float(d[act].mean())))
    out = HERE / "results" / a.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "f0_audit.json").write_text(json.dumps(rows, indent=1) + "\n")
    lines = ["| contrast | feature | share of county-hours | outage-weighted share | events >= 1% | corr |",
             "|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["contrast"], -r["share_outage_weighted"])):
        lines.append(f"| {r['contrast']} | {r['feature']} | {100 * r['share']:.2f}% | {100 * r['share_outage_weighted']:.2f}% | "
                     f"{r['events_ge_1pct']} | {r['corr']:.3f} |")
    (out / "f0_audit.md").write_text("\n".join(lines) + "\n")
    summ = {}
    for r in rows:
        s = summ.setdefault(r["contrast"], dict(max_share=0, max_weighted=0, features_material=0))
        s["max_share"] = max(s["max_share"], r["share"]); s["max_weighted"] = max(s["max_weighted"], r["share_outage_weighted"])
        s["features_material"] += int(r["share_outage_weighted"] > 0.01)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
