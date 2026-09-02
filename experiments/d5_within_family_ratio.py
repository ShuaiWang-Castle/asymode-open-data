"""D-5 -- within one family, does the two-rate advantage grow with the county-
event's OWN phase separation?

H-E ordered families by their median fall/rise ratio and found the advantage
follows it. Family is a proxy; this is the registered follow-up at the finest
grain: inside the primary (convective) study, bin county-events by their own
fall/rise ratio and ask whether the paired advantage of the two-rate model over
the parameter-matched single rate grows across bins.

Rule, fixed before the numbers are computed:
  * bins = quartiles of the county-event fall/rise ratio among interrupted
    county-events (peak y >= 0.02), computed on the panels, independent of any
    model;
  * statistic = mean over samples in the bin of (SE_netscaled - SE_susceptible)
    at h+48, per seed, then averaged; positive = two-rate better;
  * PASS if the statistic is monotone non-decreasing from the lowest to the
    highest ratio quartile AND the top quartile's county-block 95% interval is
    above zero. KILL if the profile is non-monotone (any decrease between
    adjacent quartiles larger than the top quartile's interval half-width) or
    flat (top quartile interval touches zero).
Also reported, not scored: the same profile against the trees, as contrast.

Samples are (county, storm, origin); a county-event is (county, storm). Every
sample of a county-event inherits that event's ratio. County-block bootstrap
(B = 2000) for intervals, as in D-4. Overlapping forecasts: no forecast-level
significance is attached (see D-3).
"""
import argparse, json, sys
from pathlib import Path

import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from asymode.evalproto import to_hourly   # noqa: E402

INTERIM = ROOT / "data" / "interim"
THR, PEAK = 0.01, 0.02


def county_event_ratio(panels):
    """(panel, fips) -> fall/rise ratio for interrupted county-events; NaN otherwise."""
    out = {}
    for day in panels:
        z = np.load(INTERIM / f"panel_{day}.npz", allow_pickle=True)
        yh, _ = to_hourly(z["y"], z["observed"]); yh = np.nan_to_num(yh)
        fips = z["fips"].astype(str)
        for i in range(yh.shape[0]):
            y = yh[i]; pk = int(y.argmax()); peak = float(y[pk])
            if peak < PEAK:
                continue
            on = np.where(y[:pk + 1] >= THR)[0]
            if not len(on):
                continue
            rise = pk - on[0] + 1
            off = np.where(y[pk:] < THR)[0]; fall = int(off[0]) if len(off) else int(len(y) - pk)
            out[(day, fips[i])] = max(fall, 1) / rise
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof-dir", default=str(ROOT / "results"))
    ap.add_argument("--a", default="susceptible"); ap.add_argument("--b", default="net_scaled")
    ap.add_argument("--contrast", default="trees_matched")
    ap.add_argument("--h", type=int, default=48); ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--out", default="results/d5_within_family_ratio.json")
    a = ap.parse_args()
    za = np.load(Path(a.oof_dir) / f"oof_{a.a}.npz", allow_pickle=True)
    zb = np.load(Path(a.oof_dir) / f"oof_{a.b}.npz", allow_pickle=True)
    for k in ("fips", "panel", "origin_step", "mask"):
        if not np.array_equal(za[k], zb[k]):
            sys.exit(f"exports differ on {k}")
    H = [int(h) for h in za["horizons"]]; hi = H.index(a.h)
    fips, panel = za["fips"].astype(str), za["panel"].astype(str)
    ratio = county_event_ratio(sorted(set(panel)))
    r = np.array([ratio.get((p_, f_), np.nan) for p_, f_ in zip(panel, fips)])
    keep = np.isfinite(r) & za["mask"][:, hi].astype(bool)
    q = np.nanpercentile(r[keep], [25, 50, 75]); bins = np.digitize(r, q)   # 0..3
    y = za["y"][:, hi]
    res = {"a": a.a, "b": a.b, "h": a.h, "quartile_edges": [float(x) for x in q], "bins": []}
    rng = np.random.default_rng(0)

    def bin_stat(zx, zy, sel):
        vals, cis = [], []
        for si in range(len(za["seeds"])):
            se = (zy["pred"][si][:, hi] - y) ** 2 - (zx["pred"][si][:, hi] - y) ** 2   # + => x better
            d = se[sel]; f = fips[sel]
            by = {}
            for c, v in zip(f, d):
                acc = by.setdefault(c, [0.0, 0]); acc[0] += v; acc[1] += 1
            keys = list(by); sums = np.array([by[k][0] for k in keys]); cnt = np.array([by[k][1] for k in keys], float)
            boot = np.array([sums[idx].sum() / max(cnt[idx].sum(), 1) for idx in (rng.integers(0, len(keys), len(keys)) for _ in range(a.B))])
            vals.append(float(d.mean())); cis.append((float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))))
        return float(np.mean(vals)), (float(np.mean([c[0] for c in cis])), float(np.mean([c[1] for c in cis])))

    zc = None
    cp = Path(a.oof_dir) / f"oof_{a.contrast}.npz"
    if cp.exists():
        zc = np.load(cp, allow_pickle=True)
        if not np.array_equal(zc["fips"], za["fips"]): zc = None
    print(f"{a.a} vs {a.b} at h+{a.h}, {int(keep.sum()):,} scored samples with a county-event ratio; quartile edges {np.round(q,2)}")
    print(f"{'quartile':>9}{'ratio range':>16}{'n':>8}{'Δ MSE (b−a)':>14}{'95% CI (county-block)':>26}" + (f"{'vs '+a.contrast:>16}" if zc is not None else ""))
    prof = []
    for qi in range(4):
        sel = keep & (bins == qi)
        lo = q[qi - 1] if qi else float(np.nanmin(r[keep])); hi_ = q[qi] if qi < 3 else float(np.nanmax(r[keep]))
        m, ci = bin_stat(za, zb, sel); prof.append((m, ci))
        line = f"{qi+1:>9}{f'[{lo:.2f}, {hi_:.2f}]':>16}{int(sel.sum()):>8}{m:>+14.3e}{f'[{ci[0]:+.2e}, {ci[1]:+.2e}]':>26}"
        rec = {"q": qi + 1, "ratio_lo": float(lo), "ratio_hi": float(hi_), "n": int(sel.sum()), "delta_mse": m, "ci": ci}
        if zc is not None:
            mc, cic = bin_stat(za, zc, sel); line += f"{mc:>+16.3e}"; rec["delta_mse_vs_contrast"] = mc; rec["ci_vs_contrast"] = cic
        print(line); res["bins"].append(rec)
    top_ci = prof[3][1]; half = (top_ci[1] - top_ci[0]) / 2
    drops = [prof[i][0] - prof[i + 1][0] for i in range(3)]
    monotone = all(dr <= half for dr in drops); top_pos = top_ci[0] > 0
    verdict = "PASS" if (monotone and top_pos) else "KILL"
    res.update({"monotone_within_halfwidth": bool(monotone), "top_quartile_above_zero": bool(top_pos), "verdict": verdict})
    print(f"\nprofile monotone (drops within top-quartile half-width {half:.2e}): {monotone} · top quartile above zero: {top_pos} · DECISION: {verdict}")
    Path(ROOT / a.out).write_text(json.dumps(res, indent=1)); print(f"written: {a.out}")


if __name__ == "__main__":
    main()
