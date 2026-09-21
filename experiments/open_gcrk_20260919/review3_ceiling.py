"""Static-conditioning ceiling (exploratory, 2026-09-20). Run with OPEN_GCRK_ROUND=e3r2.

(1) Share of each geographic descriptor's between-county variance explained by the state alone.
(2) How much of what weather does not explain is a repeatable county effect? The Amendment-4 regressor
    (weather summaries, neighbours and prefix outage; no county context, no geography; E3 county-grouped
    folds) predicts the unit's forecast-window mean outage; the out-of-fold residuals of counties seen in
    at least two events give a one-way ANOVA intraclass correlation. A static county descriptor of any kind
    can explain at most that share of the residual variance.
Writes results/<round>/review3_ceiling.json."""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "1")
import json
import numpy as np, pandas as pd
import common as C, review2_checks as R2
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

F = C.load_features(); n = len(F["y"]); fips = F["fips"].astype(str)
G = F["geo"].astype(float); G = np.where(np.isnan(G), np.nanmedian(G, 0), G)
first = pd.Series(np.arange(n)).groupby(fips).first().to_numpy()
st = pd.get_dummies(pd.Series([f[:2] for f in fips[first]])).to_numpy(float)
names = list(F["geo_features"].astype(str)); r2 = {}
for j, nm in enumerate(names):
    g = G[first, j]; p = LinearRegression().fit(st, g).predict(st); r2[nm] = float(1 - ((g - p) ** 2).sum() / ((g - g.mean()) ** 2).sum())
X = R2.unit_features(F); nd = 42
cols = list(range(0, 3 * nd + 8)) + list(range(3 * nd + 14, 3 * nd + 19))
_, mean = R2.unit_targets(F); sp = C.load_splits()["main"]
def oof(t):
    pred = np.full(n, np.nan)
    for spec in sp.values():
        dev, out = np.array(spec["dev"]), np.array(spec["outer"])
        mdl = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0,
                                            early_stopping=False, random_state=0).fit(X[dev][:, cols], t[dev]); pred[out] = mdl.predict(X[out][:, cols])
    return pred
icc = {}
for tname, t in (("log window mean", np.log(mean + 0.002)), ("window mean", mean)):
    d = pd.DataFrame(dict(fips=fips, res=t - oof(t))); d = d[d.groupby("fips").res.transform("size") >= 2]
    grp = d.groupby("fips").res; k = grp.size()
    n0 = (len(d) - (k ** 2).sum() / len(d)) / (grp.ngroups - 1)
    msb = (k * (grp.mean() - d.res.mean()) ** 2).sum() / (grp.ngroups - 1); msw = ((d.res - grp.transform("mean")) ** 2).sum() / (len(d) - grp.ngroups)
    icc[tname] = dict(counties=int(grp.ngroups), units=int(len(d)), icc=float((msb - msw) / (msb + (n0 - 1) * msw)))
out = dict(state_r2_median=float(np.median(list(r2.values()))), state_r2=r2, county_effect=icc)
(C.RESULTS / "review3_ceiling.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps({k: v for k, v in out.items() if k != "state_r2"}, indent=1))
