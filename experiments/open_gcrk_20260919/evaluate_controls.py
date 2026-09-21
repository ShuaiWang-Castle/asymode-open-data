"""A6.1 evaluation: shared-kernel and permuted-geography controls on real data (PREREG Amendment 6).
Run with OPEN_GCRK_ROUND=r2. OUTER RMSE with county-cluster intervals against W and GCRK of the R2
screen, and the training / inner loss of every arm at matched steps. Writes results/r2/controls_*.csv."""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "1")
import json
import numpy as np, pandas as pd
import common as C

ARMS = ("W", "GCRK", "GCRK-S", "GCRK-P")
F = C.load_features(); n = len(F["y"]); m = F["m"].astype(bool); y = F["y"].astype(float); z = m & (y == 0)
sp = C.load_splits()["main"]; rng = np.random.default_rng(20260921)
preds = {}
for a in ARMS:
    P = np.full((n, 144), np.nan)
    for f in sp:
        p = C.cell("main", 0, f, a) / "outer.npz"
        if p.exists():
            d = np.load(p); P[d["idx"]] = d["P"]
    if np.isfinite(P).all():
        preds[a] = P
rows = []
for a, P in preds.items():
    met = C.metrics(P, F)
    t = [json.loads((C.cell("main", 0, f, a) / "DONE.json").read_text()) for f in sp]
    rows.append(dict(arm=a, rmse=met["rmse"], mae=met["mae"], **{k: met[k] for k in met if k.startswith("rmse ")},
                     false_activity=float((P[z] > 0.001).mean()), t_star_median=int(np.median([x["best_step"] for x in t])),
                     beta=np.round([x["kernel"].get("beta", np.nan) for x in t], 2).tolist() if a != "W" else ""))
df = pd.DataFrame(rows); df["vs W"] = df.rmse / df.loc[df.arm == "W", "rmse"].iloc[0] - 1
u, inv = np.unique(F["fips"].astype(str), return_inverse=True)
w = rng.multinomial(len(u), np.full(len(u), 1 / len(u)), size=5000)
bt = []
sse = {a: C.unit_sse(P, F) for a, P in preds.items()}
for a, b in (("GCRK", "W"), ("GCRK-S", "W"), ("GCRK-P", "W"), ("GCRK", "GCRK-S"), ("GCRK", "GCRK-P")):
    if a in sse and b in sse:
        r = np.sqrt((w @ np.bincount(inv, sse[a])) / (w @ np.bincount(inv, sse[b]))) - 1
        bt.append(dict(contrast=f"{a} vs {b}", change=float(np.sqrt(sse[a].sum() / sse[b].sum()) - 1), ci_lo=float(np.quantile(r, .025)),
                       ci_hi=float(np.quantile(r, .975))))
curves = []
for a in ARMS:
    for f in sp:
        p = C.cell("main", 0, f, a) / "selection_trace.csv"
        if p.exists():
            t = pd.read_csv(p); fit = t[[c for c in t.columns if c.endswith("_fit_loss")]].mean(1)
            for st in (200, 300, 400, 500):
                if st in set(t.step):
                    i = t.index[t.step == st][0]
                    curves.append(dict(arm=a, fold=f, step=st, train_loss=float(fit[i]), inner_mse=float(t.pooled_inner_mse[i])))
cv = pd.DataFrame(curves)
piv = cv.pivot_table(index="step", columns="arm", values=["train_loss", "inner_mse"])
rel = pd.DataFrame({(k, a): piv[k][a] / piv[k]["W"] - 1 for k in ("train_loss", "inner_mse") for a in ARMS if a != "W" and a in piv[k]})
out = C.RESULTS; out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / "controls_main.csv", index=False); pd.DataFrame(bt).to_csv(out / "controls_bootstrap.csv", index=False)
rel.to_csv(out / "controls_curves_vs_W.csv")
pd.set_option("display.width", 220)
print(df.round(5).to_string(index=False)); print(pd.DataFrame(bt).round(4).to_string(index=False))
print("\nloss relative to W at matched steps (mean over folds):\n", rel.round(4).to_string())
