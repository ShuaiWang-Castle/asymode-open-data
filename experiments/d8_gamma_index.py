"""D-8: does the paper's own selection index explain the weak empirical result?

Zero training. Reading fixed in `docs/PREREGISTRATION_gamma_explains_gains.md`
before any Gamma was computed on a real panel.

The manuscript's discussion claims that Gamma_n organises the observed outcome.
The repository never evaluated it: D-6 reports v, A, B, lambda_min and N*v, but
neither kappa nor G_n nor Gamma_n. This script computes the plug-in of Theorem 1
and Proposition 2 on exactly the D-6 neighbourhood geometry, so the explanatory
sentence can be checked instead of asserted.

Per neighbourhood cell, with Y the origin state and Z the observed one-step
change at the origin, the two-flow conditional mean is fitted by NONNEGATIVE
least squares -- the model class is nonnegative, so an unconstrained fit could
return a negative rate that the class cannot express and would understate kappa:

    Z ~ U (1-Y) + R (-Y),   U, R >= 0
    kappa = min(R^2/A, U^2/B),  G = v * kappa,  Gamma = n * G / sigma2

Scope is inherited from the propositions: a local fixed-design benchmark, not a
statement about neural generalisation.
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
from exp06_by_family import family_of_day                                  # noqa: E402

INTERIM = ROOT / "data/interim"
K_GRID = (50, 200, 800)
MIN_N, MIN_VAR = 30, 1e-8          # inherited from D-6, not introduced here


def cell_gamma(y, z, obs):
    """Plug-in kappa / G / Gamma for one neighbourhood. None if unusable."""
    y, z = y[obs], z[obs]
    n = len(y)
    if n < MIN_N:
        return None
    s = I.gram_stats(y)
    A, B, v = s["A"], s["B"], s["var"]
    if v < MIN_VAR or A <= 0 or B <= 0:
        return None
    D = np.column_stack([1.0 - y, -y])
    coef, _ = nnls(D, z)                       # U, R >= 0 by construction
    U, R = float(coef[0]), float(coef[1])
    resid = z - D @ coef
    dof = max(n - 2, 1)
    sigma2 = float(resid @ resid) / dof
    if not np.isfinite(sigma2) or sigma2 <= 0:
        return None
    kappa = min(R * R / A, U * U / B)
    G = v * kappa
    return dict(n=n, mu=s["mu"], var=v, A=A, B=B, U_hat=U, R_hat=R,
                sigma2=sigma2, kappa=kappa, G=G, gamma=n * G / sigma2,
                # both rates strictly positive is the coactivity precondition of
                # Theorem 1; a cell failing it has G = 0 for a structural reason
                both_active=bool(U > 0 and R > 0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", default="configs/panel_manifest_g2-convective-11.json")
    ap.add_argument("--horizon", type=int, default=48)
    ap.add_argument("--stride", type=int, default=12)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--outer-split-seed", type=int, default=0)
    ap.add_argument("--split-unit", choices=["event", "county"], default="event")
    ap.add_argument("--max-cells", type=int, default=1500)
    ap.add_argument("--out", default="results/d8_gamma_index.json")
    a = ap.parse_args()
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors

    t0 = time.time()
    source = panelset.source_version(ROOT)
    want, panel_digest = panelset.resolve(INTERIM, a.panels)
    y0, X, yt, m, fips, panel, origin, t0h = exp05.load_pooled(
        a.horizon, a.stride, panels=want, with_time=True)
    x = X[:, 0, :].astype(np.float64)
    y = y0.astype(np.float64)
    z = (yt[:, 0] - y0).astype(np.float64)      # observed one-step change
    obs = m[:, 0].astype(bool)
    fam = family_of_day(); famv = np.array([fam.get(p, "convective") for p in panel])
    assign, _, split_digest, _ = exp05.outer_assignment(
        fips, panel, a.split_unit, a.k, a.outer_split_seed)
    print(f"{len(y):,} origin cells · {len(set(panel))} events [{panel_digest}] · "
          f"split {split_digest} · observed one-step {obs.mean()*100:.1f}%", flush=True)

    cells = []
    rng = np.random.default_rng(a.outer_split_seed)
    for f in range(a.k):
        tr = np.where(assign != f)[0]
        mu_x, sd_x = x[tr].mean(0), x[tr].std(0) + 1e-9
        pca = PCA(n_components=5, random_state=0).fit((x[tr] - mu_x) / sd_x)
        zz = pca.transform((x[tr] - mu_x) / sd_x)
        sub = rng.choice(len(tr), size=min(a.max_cells, len(tr)), replace=False)
        for k in K_GRID:
            nn = NearestNeighbors(n_neighbors=min(k, len(tr))).fit(zz)
            _, idx = nn.kneighbors(zz[sub])
            for i, row in zip(sub, idx):
                r = tr[row]
                st = cell_gamma(y[r], z[r], obs[r])
                if st is None:
                    continue
                st.update(rule=f"knn{k}", fold=f, event=str(panel[tr[i]]),
                          family=str(famv[tr[i]]))
                cells.append(st)
        print(f"  fold {f}: {len(sub)} cells x {len(K_GRID)} rules", flush=True)

    def q(v, p): return float(np.quantile(v, p)) if len(v) else float("nan")
    summary = {}
    for rule in [f"knn{k}" for k in K_GRID]:
        c = [r for r in cells if r["rule"] == rule]
        g = np.array([r["gamma"] for r in c])
        summary[rule] = dict(
            n_cells=len(c),
            median_gamma=q(g, .5), q25_gamma=q(g, .25), q75_gamma=q(g, .75),
            frac_gamma_gt_1=float(np.mean(g > 1)) if len(g) else float("nan"),
            median_kappa=q(np.array([r["kappa"] for r in c]), .5),
            median_G=q(np.array([r["G"] for r in c]), .5),
            median_sigma2=q(np.array([r["sigma2"] for r in c]), .5),
            median_U=q(np.array([r["U_hat"] for r in c]), .5),
            median_R=q(np.array([r["R_hat"] for r in c]), .5),
            frac_both_active=float(np.mean([r["both_active"] for r in c])) if c else float("nan"),
            by_event={e: dict(n_cells=int(sum(1 for r in c if r["event"] == e)),
                              median_gamma=q(np.array([r["gamma"] for r in c if r["event"] == e]), .5),
                              median_kappa=q(np.array([r["kappa"] for r in c if r["event"] == e]), .5),
                              median_G=q(np.array([r["G"] for r in c if r["event"] == e]), .5))
                      for e in sorted({r["event"] for r in c})})
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(
        config=dict(vars(a), panel_digest=panel_digest, split_digest=split_digest,
                    source=source, preregistration="docs/PREREGISTRATION_gamma_explains_gains.md",
                    wall_time_s=round(time.time() - t0, 1)),
        summary=summary, cells=cells), indent=1))
    print(f"\nwritten: {a.out}")
    for rule, s in summary.items():
        print(f"[{rule}] cells {s['n_cells']:,} | median Gamma {s['median_gamma']:.4g} "
              f"(IQR {s['q25_gamma']:.3g}-{s['q75_gamma']:.3g}) | frac>1 {s['frac_gamma_gt_1']:.3f} "
              f"| median kappa {s['median_kappa']:.3g} | both rates active {s['frac_both_active']:.3f}")


if __name__ == "__main__":
    main()
