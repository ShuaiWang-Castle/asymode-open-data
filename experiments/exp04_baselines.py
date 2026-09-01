"""EXP04 -- statistical baselines under the county-held-out protocol.

These are the numbers every later model has to beat, computed under exactly the
protocol the later models will use: the same folds, the same origins, the same
mask, the same horizons.

The all-zero predictor is included deliberately and is not a joke. Roughly three
quarters of counties sit at exactly zero before a storm, so a model that outputs
zero everywhere is a genuinely strong RMSE baseline on this target. Any claim
about dynamics has to clear it, and reporting it keeps the later comparison honest.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from asymode.evalproto import Task, make_folds, score, to_hourly   # noqa: E402


def predict(name: str, task: Task, rows, origins, train_rows) -> np.ndarray:
    H = task.horizons
    y, obs = task.y[rows], task.observed[rows]
    out = np.zeros((len(rows), len(origins), len(H)), dtype=np.float32)

    if name == "zero":
        return out

    if name == "persistence":
        # Last observed value at or before the origin.
        for oi, o in enumerate(origins):
            hist = np.where(obs[:, :o + 1], y[:, :o + 1], np.nan)
            last = _last_valid(hist)
            out[:, oi, :] = np.nan_to_num(last)[:, None]
        return out

    if name == "climatology_hour":
        # Mean over TRAINING counties by hour of day. Test counties are unseen,
        # so nothing county-specific may be used.
        ytr, otr = task.y[train_rows], task.observed[train_rows]
        hod = np.arange(task.y.shape[1]) % 24
        mu = np.zeros(24, dtype=np.float32)
        for h in range(24):
            sel = hod == h
            v = ytr[:, sel][otr[:, sel]]
            mu[h] = float(v.mean()) if v.size else 0.0
        for oi, o in enumerate(origins):
            for hi, h in enumerate(H):
                out[:, oi, hi] = mu[(o + h) % 24]
        return out

    if name == "damped_persistence":
        # Persistence decayed toward the training mean; a restoration process with
        # no covariates. The decay constant is fitted on training counties only.
        ytr, otr = task.y[train_rows], task.observed[train_rows]
        mu = float(ytr[otr].mean()) if otr.any() else 0.0
        rho = _fit_decay(ytr, otr)
        for oi, o in enumerate(origins):
            last = np.nan_to_num(_last_valid(np.where(obs[:, :o + 1], y[:, :o + 1], np.nan)))
            for hi, h in enumerate(H):
                out[:, oi, hi] = mu + (last - mu) * rho ** h
        return out

    raise ValueError(name)


def _last_valid(a: np.ndarray) -> np.ndarray:
    idx = np.where(np.isfinite(a), np.arange(a.shape[1])[None, :], -1).max(axis=1)
    r = np.full(a.shape[0], np.nan, dtype=np.float32)
    ok = idx >= 0
    r[ok] = a[np.arange(a.shape[0])[ok], idx[ok]]
    return r


def _fit_decay(y: np.ndarray, obs: np.ndarray) -> float:
    """One-step AR coefficient around the mean, on observed consecutive pairs."""
    m = obs[:, :-1] & obs[:, 1:]
    if not m.any():
        return 0.9
    mu = float(y[obs].mean())
    a = (y[:, :-1] - mu)[m]; b = (y[:, 1:] - mu)[m]
    d = float((a * a).sum())
    return float(np.clip((a * b).sum() / d, 0.0, 1.0)) if d > 0 else 0.9


BASELINES = ["zero", "persistence", "damped_persistence", "climatology_hour"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--out", default="results/exp04_baselines.json")
    a = ap.parse_args()

    panels = sorted((ROOT / "data/interim").glob("panel_*.npz"))
    print(f"{len(panels)} archived panels")
    rows = []
    for p in panels:
        z = np.load(p, allow_pickle=True)
        yh, oh = to_hourly(z["y"], z["observed"])
        fips = z["fips"].tolist()
        task = Task(y=np.nan_to_num(yh), observed=oh, fips=fips)
        origins = task.origins(stride=a.stride)
        if len(origins) == 0:
            print(f"  {p.stem}: too short, skipped"); continue
        for seed in a.seeds:
            fold = make_folds(fips, k=a.k, seed=seed)
            for f in range(a.k):
                te = np.where(fold == f)[0]; tr = np.where(fold != f)[0]
                if len(te) == 0 or len(tr) == 0:
                    continue
                for b in BASELINES:
                    s = score(predict(b, task, te, origins, tr), task, te, origins)
                    rows.append({"panel": p.stem.replace("panel_", ""), "baseline": b,
                                 "seed": seed, "fold": f, "n_test": len(te), **s})
        print(f"  {p.stem}: {len(fips)} counties, {yh.shape[1]} hours, "
              f"{len(origins)} origins")

    out = ROOT / a.out; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": vars(a), "rows": rows}, indent=2))

    print(f"\n=== pooled over {len(panels)} panels x {a.k} folds x {len(a.seeds)} seeds ===")
    hs = [1, 6, 24, 48]
    print(f"{'baseline':<22}" + "".join(f"{'RMSE h+'+str(h):>18}" for h in hs))
    for b in BASELINES:
        g = [r for r in rows if r["baseline"] == b]
        line = f"{b:<22}"
        for h in hs:
            v = [r[f"rmse_h{h}"] for r in g if np.isfinite(r[f"rmse_h{h}"])]
            line += f"{np.mean(v):>12.5f}±{np.std(v):<5.5f}"
        print(line)
    print(f"\nwritten: {a.out}")


if __name__ == "__main__":
    main()
