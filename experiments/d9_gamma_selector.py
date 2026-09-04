"""D-9: does Gamma's own threshold identify the cells where the second flow pays?

Zero training. Reading fixed in `docs/PREREGISTRATION_gamma_explains_gains.md`
(H-G3) before any number here was computed.

The theory's claim is a threshold, not a ranking: two flows are favoured exactly
when Gamma_n > 1. This evaluates that threshold as a cross-fitted selector --
the geometry is fitted on training rows of each outer fold, Gamma is evaluated at
the driver location of each HELD-OUT row, and the verdict is compared against the
archived held-out errors of the two parameter-matched arms.

Nothing is retrained. Inputs are the public panels and
`results/event_transfer_confirmatory_20260903/predictions/`.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
from asymode import panels as panelset, information as I                   # noqa: E402
import exp05_real_dynamics as exp05                                        # noqa: E402

INTERIM = ROOT / "data/interim"
PRED = ROOT / "results/event_transfer_confirmatory_20260903/predictions"
K_GRID = (50, 200, 800)
MIN_N, MIN_VAR = 30, 1e-8


def load_arm(arm: str, seeds=(0, 1, 2)):
    """Seed-averaged squared error per row and horizon, plus the row keys."""
    se, keys = None, None
    for s in seeds:
        f = PRED / f"pred_event_{arm}_seed{s}.npz"
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=True)
        e = (z["pred"].astype(np.float64) - z["y"].astype(np.float64)) ** 2
        se = e if se is None else se + e
        n = sum(1 for t in seeds if (PRED / f"pred_event_{arm}_seed{t}.npz").exists())
        keys = (z["panel"], z["fips"], z["origin_step"], z["mask"], z["horizons"])
    return se / n, keys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", default="configs/panel_manifest_g2-convective-11.json")
    ap.add_argument("--horizon", type=int, default=48)
    ap.add_argument("--stride", type=int, default=12)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--outer-split-seed", type=int, default=0)
    ap.add_argument("--out", default="results/d9_gamma_selector.json")
    a = ap.parse_args()
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors

    t0 = time.time(); source = panelset.source_version(ROOT)
    want, panel_digest = panelset.resolve(INTERIM, a.panels)
    y0, X, yt, m, fips, panel, origin, t0h = exp05.load_pooled(
        a.horizon, a.stride, panels=want, with_time=True)
    x = X[:, 0, :].astype(np.float64); y = y0.astype(np.float64)
    zz = (yt[:, 0] - y0).astype(np.float64); obs0 = m[:, 0].astype(bool)
    assign, _, split_digest, _ = exp05.outer_assignment(
        fips, panel, "event", a.k, a.outer_split_seed)

    se_two, keys = load_arm("two_rate")
    se_one, _ = load_arm("net_scaled")
    p_pan, p_fips, p_org, p_mask, horizons = keys
    # the prediction archive must be the same rows, in the same order
    assert len(p_pan) == len(panel), (len(p_pan), len(panel))
    assert (p_pan.astype(str) == panel.astype(str)).all(), "panel order differs"
    assert (p_fips.astype(str) == fips.astype(str)).all(), "county order differs"
    assert (p_org.astype(int) == origin.astype(int)).all(), "origin order differs"
    horizons = [int(h) for h in horizons]
    print(f"{len(y):,} rows aligned to the archive · horizons {horizons} · "
          f"panels [{panel_digest}] · split {split_digest}", flush=True)

    rows = []
    for f in range(a.k):
        tr = np.where(assign != f)[0]; te = np.where(assign == f)[0]
        mu_x, sd_x = x[tr].mean(0), x[tr].std(0) + 1e-9
        pca = PCA(n_components=5, random_state=0).fit((x[tr] - mu_x) / sd_x)
        ztr = pca.transform((x[tr] - mu_x) / sd_x)
        zte = pca.transform((x[te] - mu_x) / sd_x)
        for k in K_GRID:
            nn = NearestNeighbors(n_neighbors=min(k, len(tr))).fit(ztr)
            _, idx = nn.kneighbors(zte)                    # test row -> training neighbours
            for j, row in zip(te, idx):
                r = tr[row]; sel = obs0[r]
                yy, zvals = y[r][sel], zz[r][sel]
                n = len(yy)
                if n < MIN_N:
                    continue
                s = I.gram_stats(yy)
                A, B, v = s["A"], s["B"], s["var"]
                if v < MIN_VAR or A <= 0 or B <= 0:
                    continue
                D = np.column_stack([1.0 - yy, -yy])
                coef, _ = nnls(D, zvals)
                U, R = float(coef[0]), float(coef[1])
                res = zvals - D @ coef
                sig2 = float(res @ res) / max(n - 2, 1)
                if not np.isfinite(sig2) or sig2 <= 0:
                    continue
                kap = min(R * R / A, U * U / B)
                gam = n * v * kap / sig2
                rec = dict(rule=f"knn{k}", fold=f, row=int(j), event=str(panel[j]),
                           gamma=gam, kappa=kap, U=U, R=R, both_active=bool(U > 0 and R > 0))
                for hi, h in enumerate(horizons):
                    if p_mask[j, hi]:
                        rec[f"d_h{h}"] = float(se_one[j, hi] - se_two[j, hi])
                rows.append(rec)
        print(f"  fold {f}: {len(te):,} held-out rows x {len(K_GRID)} rules", flush=True)

    def group(rs, h):
        d = np.array([r[f"d_h{h}"] for r in rs if f"d_h{h}" in r])
        if not len(d):
            return None
        return dict(n=int(len(d)), two_flow_win_rate=float(np.mean(d > 0)),
                    mean_diff=float(d.mean()), median_diff=float(np.median(d)))

    summary = {}
    for rule in [f"knn{k}" for k in K_GRID]:
        c = [r for r in rows if r["rule"] == rule]
        hi = [r for r in c if r["gamma"] > 1.0]; lo = [r for r in c if r["gamma"] <= 1.0]
        entry = dict(n_cells=len(c), n_gamma_gt_1=len(hi), n_gamma_le_1=len(lo),
                     frac_gamma_gt_1=len(hi) / max(len(c), 1),
                     underpowered=bool(len(hi) < 200))
        for h in (24, 48):
            entry[f"h{h}"] = dict(gamma_gt_1=group(hi, h), gamma_le_1=group(lo, h))
            g, l = entry[f"h{h}"]["gamma_gt_1"], entry[f"h{h}"]["gamma_le_1"]
            entry[f"h{h}"]["win_rate_gap"] = (
                None if not g or not l else g["two_flow_win_rate"] - l["two_flow_win_rate"])
        summary[rule] = entry

    out = ROOT / a.out; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(
        config=dict(vars(a), panel_digest=panel_digest, split_digest=split_digest,
                    source=source, horizons=horizons,
                    preregistration="docs/PREREGISTRATION_gamma_explains_gains.md (H-G3)",
                    wall_time_s=round(time.time() - t0, 1)),
        summary=summary), indent=1))
    print(f"\nwritten: {a.out}\n")
    for rule, s in summary.items():
        print(f"[{rule}] cells {s['n_cells']:,} | Gamma>1 in {s['n_gamma_gt_1']:,} "
              f"({s['frac_gamma_gt_1']*100:.1f}%){'  UNDERPOWERED' if s['underpowered'] else ''}")
        for h in (24, 48):
            g, l, gap = s[f"h{h}"]["gamma_gt_1"], s[f"h{h}"]["gamma_le_1"], s[f"h{h}"]["win_rate_gap"]
            if g and l:
                print(f"    h+{h}: two-flow win rate  Gamma>1 {g['two_flow_win_rate']:.4f} (n={g['n']:,})"
                      f"   Gamma<=1 {l['two_flow_win_rate']:.4f} (n={l['n']:,})   gap {gap:+.4f}")


if __name__ == "__main__":
    main()
