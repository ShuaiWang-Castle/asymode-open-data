"""A6.2 timing probe (PREREG Amendment 6): does geography change the timing or persistence of the
outage response? Run with OPEN_GCRK_ROUND=e3r2.

Units: forecast-window peak >= 1%. Targets: lag from the gust peak to the outage peak (h), hours the
outage stays at or above half its peak, and window mean / peak. The Amendment-4 regressor (same fixed
settings, the E3 county-grouped outer folds, fitted on the eligible development units) predicts each
from the unit-level summaries, without and with the 40 descriptors, and with the descriptors permuted
within event x state blocks (20 permutations). Writes results/<round>/review3_timing.csv.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
import pandas as pd

import common as C
import review2_checks as R2

RNG = np.random.default_rng(20260921)
N_PERM, N_BOOT, PEAK_MIN = 20, 2000, 0.01


def targets(F):
    m = F["m"].astype(bool); y = F["y"].astype(np.float64)
    names = list(F["damage_features"].astype(str)); g = F["xu"][:, 72:, names.index("gust")].astype(np.float64)
    ym = np.where(m, y, -np.inf)
    peak = ym.max(1); t_out = ym.argmax(1); t_gust = g.argmax(1)
    half = (m & (y >= 0.5 * peak[:, None])).sum(1).astype(np.float64)
    mean = np.where(m, y, 0).sum(1) / np.maximum(m.sum(1), 1)
    return peak, {"lag gust peak -> outage peak (h)": (t_out - t_gust).astype(np.float64),
                  "hours at or above half peak": half, "window mean / peak": mean / np.maximum(peak, 1e-9)}


def main():
    from sklearn.ensemble import HistGradientBoostingRegressor
    F = C.load_features(); n = len(F["y"])
    peak, T = targets(F)
    elig = peak >= PEAK_MIN
    X0 = R2.unit_features(F); G = F["geo"].astype(np.float64)
    ev = F["event"].astype(str); fips = F["fips"].astype(str); st = np.array([f[:2] for f in fips])
    block = np.char.add(np.char.add(ev, "|"), st); blocks = [np.where(block == b)[0] for b in np.unique(block)]
    folds = C.load_splits()["main"]
    u_c, inv_c = np.unique(fips, return_inverse=True)

    def oof(X, t):
        pred = np.full(n, np.nan)
        for spec in folds.values():
            dev = np.array(spec["dev"]); out = np.array(spec["outer"]); dev, out = dev[elig[dev]], out[elig[out]]
            mdl = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
                                                l2_regularization=1.0, early_stopping=False, random_state=0)
            mdl.fit(X[dev], t[dev]); pred[out] = mdl.predict(X[out])
        return pred

    rows = []
    for name, t in T.items():
        se = {}
        for lab, X in (("no geography", X0), ("with geography", np.concatenate([X0, G], 1))):
            p = oof(X, t); se[lab] = np.where(elig, (p - t) ** 2, 0.0)
        tt = t[elig]; sst = ((tt - tt.mean()) ** 2).sum()
        null = []
        for _ in range(N_PERM):
            perm = np.arange(n)
            for bi in blocks:
                perm[bi] = RNG.permutation(bi)
            p = oof(np.concatenate([X0, G[perm]], 1), t)
            null.append(float(np.sqrt(((p - t)[elig] ** 2).mean())))
        a0, a1 = np.bincount(inv_c, se["no geography"]), np.bincount(inv_c, se["with geography"])
        w = RNG.multinomial(len(u_c), np.full(len(u_c), 1 / len(u_c)), size=N_BOOT)
        boot = np.sqrt((w @ a1) / (w @ a0)) - 1
        r0, r1 = np.sqrt(se["no geography"].sum() / elig.sum()), np.sqrt(se["with geography"].sum() / elig.sum())
        rows.append(dict(target=name, n_units=int(elig.sum()), target_sd=float(tt.std()), rmse_no_geo=float(r0),
                         r2_no_geo=float(1 - se["no geography"].sum() / sst), rmse_geo=float(r1),
                         r2_geo=float(1 - se["with geography"].sum() / sst), rel_change=float(r1 / r0 - 1),
                         ci_lo=float(np.quantile(boot, .025)), ci_hi=float(np.quantile(boot, .975)),
                         null_rmse_p05=float(np.quantile(null, .05)), null_rmse_median=float(np.median(null)),
                         beats_null=bool(r1 < np.quantile(null, .05))))
        print(name, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS / "review3_timing.csv", index=False)
    pd.set_option("display.width", 230)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
