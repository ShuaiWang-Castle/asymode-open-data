"""Redundancy probe (exploratory, 2026-09-21): how much of the geography is already in the weather the host
reads? Run with OPEN_GCRK_ROUND=e3r2. For every descriptor, the Amendment-4 regressor (fixed settings, E3
county-grouped folds, so a county is never on both sides) predicts it from the unit's weather only: prefix
mean and forecast-window mean of the 14 ERA5 channels. Reported: out-of-fold R2 per descriptor, and R2 from
surface pressure alone for the elevation descriptors. Writes results/<round>/review3_redundancy.csv."""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "1")
import numpy as np, pandas as pd
import common as C
from sklearn.ensemble import HistGradientBoostingRegressor

F = C.load_features(); n = len(F["y"]); sp = C.load_splits()["main"]
ch = list(F["weather_channels"].astype(str)); xu = F["xu"][:, :, :len(ch)].astype(np.float64)
X = np.concatenate([xu[:, :72].mean(1), xu[:, 72:].mean(1)], 1)
G = F["geo"].astype(np.float64); names = list(F["geo_features"].astype(str))
def oof(Xm, t):
    ok = np.isfinite(t); pred = np.full(n, np.nan)
    for spec in sp.values():
        dev, out = np.array(spec["dev"]), np.array(spec["outer"]); dev = dev[ok[dev]]
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0,
                                          early_stopping=False, random_state=0).fit(Xm[dev], t[dev]); pred[out] = m.predict(Xm[out])
    e = ok; return float(1 - ((pred - t)[e] ** 2).sum() / ((t[e] - t[e].mean()) ** 2).sum())
p = ch.index("pressure"); Xp = np.stack([xu[:, :72, p].mean(1), xu[:, 72:, p].mean(1)], 1)
rows = []
for j, nm in enumerate(names):
    r = dict(descriptor=nm, r2_from_weather=oof(X, G[:, j]))
    if nm in ("elev_mean", "elev_mean5", "relief_p95_p5", "slope_mean_deg"):
        r["r2_from_pressure_alone"] = oof(Xp, G[:, j])
    rows.append(r); print(nm, round(r["r2_from_weather"], 3), flush=True)
d = pd.DataFrame(rows).sort_values("r2_from_weather", ascending=False)
d.to_csv(C.RESULTS / "review3_redundancy.csv", index=False)
print(d.round(3).to_string(index=False))
print(f"\nmedian R2 {d.r2_from_weather.median():.3f}; descriptors with R2 > 0.5: {(d.r2_from_weather > 0.5).sum()}/{len(d)}; > 0.8: {(d.r2_from_weather > 0.8).sum()}")
