"""Any set of arms of one round and design at seed 0: pooled metrics, county / event-by-state / event cluster
bootstrap of the relative RMSE change for chosen contrasts, per-fold RMSE and selected steps.
Usage: OPEN_GCRK_ROUND=e3r2 python evaluate_arms.py --design main --arms W GCRK GCRK-S W+C W+Cin --contrasts GCRK-S:W W+Cin:W
Writes results/<round>/arms_<tag>_{main,bootstrap,folds}.csv."""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "1")
import argparse, json
import numpy as np, pandas as pd
import common as C

ap = argparse.ArgumentParser()
ap.add_argument("--design", default="main"); ap.add_argument("--arms", nargs="+", required=True)
ap.add_argument("--contrasts", nargs="+", default=[]); ap.add_argument("--tag", default="set"); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
F = C.load_features(); n = len(F["y"]); m = F["m"].astype(bool); y = F["y"].astype(float); z = m & (y == 0)
sp = C.load_splits()[a.design]; rng = np.random.default_rng(20260921)
ev = F["event"].astype(str); st = np.array([f[:2] for f in F["fips"].astype(str)])
labels = {"county": F["fips"].astype(str), "event x state": np.char.add(np.char.add(ev, "|"), st), "event": ev}
preds, rows, folds = {}, [], []
for arm in a.arms:
    P = np.full((n, 144), np.nan)
    for f in sp:
        p = C.cell(a.design, a.seed, f, arm) / "outer.npz"
        if p.exists():
            d = np.load(p); P[d["idx"]] = d["P"]
    if not np.isfinite(P).all():
        print("incomplete:", arm); continue
    preds[arm] = P; met = C.metrics(P, F)
    info = [json.loads((C.cell(a.design, a.seed, f, arm) / "DONE.json").read_text()) for f in sp]
    rows.append(dict(arm=arm, rmse=met["rmse"], mae=met["mae"], **{k: met[k] for k in met if k.startswith("rmse ")},
                     rmse_event_equal=float(np.mean([C.metrics(P, F, np.where(ev == e)[0])["rmse"] for e in np.unique(ev)])),
                     false_activity=float((P[z] > 0.001).mean()), t_star_median=int(np.median([i["best_step"] for i in info])),
                     minutes=int(sum(i["seconds"] for i in info) / 60)))
    for f, i in zip(sp, info):
        folds.append(dict(arm=arm, fold=f, rmse=C.metrics(P, F, np.array(sp[f]["outer"]))["rmse"], t_star=i["best_step"]))
df = pd.DataFrame(rows)
if "W" in preds:
    df["vs W"] = df.rmse / df.loc[df.arm == "W", "rmse"].iloc[0] - 1
sse = {k: C.unit_sse(v, F) for k, v in preds.items()}; bt = []
for c in a.contrasts:
    x, b = c.split(":")
    if x in sse and b in sse:
        for lab, g in labels.items():
            u, inv = np.unique(g, return_inverse=True)
            w = rng.multinomial(len(u), np.full(len(u), 1 / len(u)), size=5000)
            r = np.sqrt((w @ np.bincount(inv, sse[x])) / (w @ np.bincount(inv, sse[b]))) - 1
            bt.append(dict(contrast=f"{x} vs {b}", cluster=lab, change=float(np.sqrt(sse[x].sum() / sse[b].sum()) - 1),
                           ci_lo=float(np.quantile(r, .025)), ci_hi=float(np.quantile(r, .975))))
out = C.RESULTS; out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / f"arms_{a.tag}_main.csv", index=False); pd.DataFrame(bt).to_csv(out / f"arms_{a.tag}_bootstrap.csv", index=False)
fd = pd.DataFrame(folds); fd.to_csv(out / f"arms_{a.tag}_folds.csv", index=False)
pd.set_option("display.width", 230)
print(df.round(5).to_string(index=False)); print(pd.DataFrame(bt).round(4).to_string(index=False))
print(fd.pivot(index="fold", columns="arm", values="rmse").round(5).to_string())
