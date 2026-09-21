"""Footprint probe (exploratory, 2026-09-20): inside ONE event, do static descriptors act as a proxy
for where that storm hit? Run with OPEN_GCRK_ROUND=e3r2.

For every event separately: five county-grouped folds inside the event (seeded), the Amendment-4
regressor predicts the unit's forecast-window mean outage from the unit-level weather summaries
(a) alone, (b) with the 40 descriptors, (c) with the county's latitude and longitude only. The same
three feature sets are evaluated on the pooled twelve-event data with the E3 county-grouped folds.
Writes results/<round>/review3_footprint.csv.
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "1")
import numpy as np, pandas as pd
import common as C, review2_checks as R2
from sklearn.ensemble import HistGradientBoostingRegressor

F = C.load_features(); n = len(F["y"]); fips = F["fips"].astype(str); ev = F["event"].astype(str)
G = F["geo"].astype(float); X0 = R2.unit_features(F); _, mean = R2.unit_targets(F)
gz = pd.read_csv(C.ROOT / "data/raw/census/2023_Gaz_counties_national.txt", sep="\t", dtype={"GEOID": str}, encoding="latin-1")
gz.columns = [c.strip() for c in gz.columns]; gz = gz.set_index(gz.GEOID.str.zfill(5))
LL = np.stack([gz.loc[fips, "INTPTLAT"].to_numpy(float), gz.loc[fips, "INTPTLONG"].to_numpy(float)], 1)
sets = {"weather": X0, "weather + geography": np.concatenate([X0, G], 1), "weather + lat/lon": np.concatenate([X0, LL], 1)}
rng = np.random.default_rng(20260920)

def cv(idx, folds):
    out = {}
    for name, X in sets.items():
        pred = np.full(n, np.nan)
        for k in range(5):
            te = idx[folds == k]; tr = idx[folds != k]
            m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0,
                                              early_stopping=False, random_state=0).fit(X[tr], mean[tr]); pred[te] = m.predict(X[te])
        out[name] = float(np.sqrt(((pred[idx] - mean[idx]) ** 2).mean()))
    return out

rows = []
for e in sorted(set(ev)):
    idx = np.where(ev == e)[0]
    r = cv(idx, rng.integers(0, 5, len(idx)))          # one unit per county inside an event, so unit folds are county folds
    rows.append(dict(scope=e, units=len(idx), **r))
    print(e, "done", flush=True)
sp = C.load_splits()["main"]; fold_of = np.zeros(n, int)
for k, spec in enumerate(sp.values()):
    fold_of[np.array(spec["outer"])] = k
rows.append(dict(scope="pooled twelve events", units=n, **cv(np.arange(n), fold_of)))
d = pd.DataFrame(rows)
d["geography vs weather"] = d["weather + geography"] / d["weather"] - 1
d["lat/lon vs weather"] = d["weather + lat/lon"] / d["weather"] - 1
d.to_csv(C.RESULTS / "review3_footprint.csv", index=False)
pd.set_option("display.width", 200)
print(d.round(5).to_string(index=False))
w = d[d.scope != "pooled twelve events"]
print(f"\nwithin single events: geography {100 * w['geography vs weather'].mean():+.2f}% on average (better in {(w['geography vs weather'] < 0).sum()}/12), "
      f"lat/lon {100 * w['lat/lon vs weather'].mean():+.2f}% (better in {(w['lat/lon vs weather'] < 0).sum()}/12)")
