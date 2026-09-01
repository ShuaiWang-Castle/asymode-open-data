"""EXP05 -- the first fit of the proposed dynamics to public observations.

Everything before this was either synthetic or descriptive. This is the model
itself, on real counties, against the same baselines under the same protocol:
identical county-held-out folds, identical forecast origins, identical horizons,
identical observation mask, identical driver channels.

**Only the dynamical axis varies here.** All three arms receive exactly the same
inputs, the same capacity, the same optimiser and the same seeds, so a difference
between them is attributable to the inflow form and to nothing else. The input and
capacity asymmetries are separate ablations and are deliberately not entangled
with this one.

  susceptible        u (1 - y)              -- this work
  transmission       u y (1 - y)            -- epidemic form
  transmission_seed  u (y + eps) (1 - y)    -- steelman, eps >= 0 learnable

Forecast setting: the drivers over the forecast window are given. That is the
standard operational assumption -- numerical weather prediction exists -- but it is
an assumption, it favours every weather-driven model equally, and the paper must
state it rather than let a reader assume otherwise.

Pre-registered, before the run. The dynamical claim survives only if the
susceptible arm beats **both** epidemic arms at every horizon, in every seed, and
also beats damped persistence at h+6 and beyond. It dies if the seeded epidemic
arm matches it within one standard deviation at any horizon, or if no arm beats
the all-zero predictor at h+24 and h+48.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from asymode.dynamics import (TwoRateODE, TwoRateConfig, InflowForm,   # noqa: E402
                              calibrate_init)
from asymode.evalproto import Task, make_folds, to_hourly            # noqa: E402

INTERIM = ROOT / "data" / "interim"
ARMS = [InflowForm.SUSCEPTIBLE, InflowForm.TRANSMISSION, InflowForm.TRANSMISSION_SEED]


def load_pooled(horizon: int, stride: int, min_history: int = 24):
    """Every (county, storm, origin) with drivers, pooled across storms.

    Pooling is the point: a county is the unit of held-out evaluation, and a model
    that only works on the storm it was fitted to is not a model of the dynamics.
    """
    Y0, X, YT, M, FIPS, PANEL = [], [], [], [], [], []
    for pf in sorted(INTERIM.glob("panel_*.npz")):
        day = pf.stem.replace("panel_", "")
        df = INTERIM / f"drivers_{day}.npz"
        if not df.exists():
            continue
        pz, dz = np.load(pf, allow_pickle=True), np.load(df, allow_pickle=True)
        assert pz["fips"].tolist() == dz["fips"].tolist()
        yh, oh = to_hourly(pz["y"], pz["observed"])
        Xh, fips = dz["X"], pz["fips"].tolist()
        n = min(yh.shape[1], Xh.shape[1])
        yh, oh, Xh = yh[:, :n], oh[:, :n], Xh[:, :n]
        yh = np.nan_to_num(yh)
        for o in range(min_history, n - horizon, stride):
            Y0.append(yh[:, o]); X.append(Xh[:, o + 1:o + 1 + horizon])
            YT.append(yh[:, o + 1:o + 1 + horizon]); M.append(oh[:, o + 1:o + 1 + horizon])
            FIPS.append(np.array(fips)); PANEL.append(np.full(len(fips), day))
    return (np.concatenate(Y0), np.concatenate(X), np.concatenate(YT),
            np.concatenate(M), np.concatenate(FIPS), np.concatenate(PANEL))


def add_context(X: np.ndarray, y0: np.ndarray, hours: int) -> np.ndarray:
    """Append a diurnal clock. Restoration follows crews, and crews follow the sun."""
    n, T, F = X.shape
    t = np.arange(T)[None, :].repeat(n, 0)
    out = np.concatenate([X,
                          np.sin(2 * np.pi * t / 24)[..., None],
                          np.cos(2 * np.pi * t / 24)[..., None]], axis=-1)
    return out.astype(np.float32)


def run_baseline(name, tr, te, data, args):
    """Baselines scored on exactly the samples the model is scored on.

    Recomputed here rather than read from the baseline experiment: that one pools
    twelve storms at a different origin stride, so its numbers are not comparable
    to these sample for sample. A comparison across different sample sets is not a
    comparison.
    """
    y0, X, yt, m = data
    out = {}
    if name == "zero":
        pred = np.zeros_like(yt[te])
    elif name == "persistence":
        pred = np.repeat(y0[te][:, None], yt.shape[1], axis=1)
    elif name == "damped_persistence":
        ytr = np.concatenate([y0[tr][:, None], yt[tr]], axis=1)
        mtr = np.concatenate([np.ones((len(tr), 1), bool), m[tr]], axis=1)
        mm = mtr[:, :-1] & mtr[:, 1:]
        mu = float(ytr[mtr].mean())
        aa = (ytr[:, :-1] - mu)[mm]; bb = (ytr[:, 1:] - mu)[mm]
        rho = float(np.clip((aa * bb).sum() / max((aa * aa).sum(), 1e-12), 0.0, 1.0))
        hh = np.arange(1, yt.shape[1] + 1)[None, :]
        pred = mu + (y0[te][:, None] - mu) * rho ** hh
        out["rho"] = rho
    else:
        raise ValueError(name)
    for h in args.horizons:
        e = (pred[:, h - 1] - yt[te][:, h - 1])[m[te][:, h - 1]]
        out[f"rmse_h{h}"] = float(np.sqrt(np.mean(e ** 2))) if e.size else float("nan")
        out[f"n_h{h}"] = int(e.size)
    out["fitted_seed_eps"] = None
    return out


BASELINES = ["zero", "persistence", "damped_persistence"]


def run_arm(arm, tr, te, data, args, seed):
    y0, X, yt, m = data
    torch.manual_seed(seed); np.random.seed(seed)
    mu = X[tr].reshape(-1, X.shape[-1]).mean(0)
    sd = X[tr].reshape(-1, X.shape[-1]).std(0) + 1e-6      # training folds only
    Xn = ((X - mu) / sd).astype(np.float32)

    t = lambda a: torch.tensor(a)
    # Calibrated on training folds only, by the same rule for every arm.
    ytr = np.concatenate([y0[tr][:, None], yt[tr]], axis=1)
    mtr = np.concatenate([np.ones((len(tr), 1), bool), m[tr]], axis=1)
    u0, r0 = calibrate_init(ytr, mtr, arm)
    cfg = TwoRateConfig(d_in=Xn.shape[-1], cap_u=args.cap_u, cap_r=args.cap_r,
                        hidden_u=args.hidden, hidden_r=args.hidden, inflow=arm,
                        u_init=u0, r_init=r0)
    model = TwoRateODE(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    idx = np.array(tr)
    n_val = max(1, int(0.15 * len(idx)))
    rng = np.random.default_rng(seed); rng.shuffle(idx)
    va, fit = idx[:n_val], idx[n_val:]
    Y0, XX, YT, MM = t(y0), t(Xn), t(yt), t(m.astype(np.float32))

    def loss_on(ix):
        pred = model(Y0[ix], XX[ix])
        se = (pred - YT[ix]) ** 2 * MM[ix]
        return se.sum() / MM[ix].sum().clamp_min(1.0)

    best, best_state, bad = float("inf"), None, 0
    for ep in range(args.epochs):
        model.train()
        perm = np.random.permutation(fit)
        for s in range(0, len(perm), args.batch):
            b = torch.tensor(perm[s:s + args.batch], dtype=torch.long)
            opt.zero_grad(); l = loss_on(b); l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(loss_on(torch.tensor(va, dtype=torch.long)))
        if vl < best - 1e-10:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        ti = torch.tensor(te, dtype=torch.long)
        pred = model(Y0[ti], XX[ti]).numpy()
    out = {}
    for h in args.horizons:
        e = (pred[:, h - 1] - yt[te][:, h - 1])[m[te][:, h - 1]]
        out[f"rmse_h{h}"] = float(np.sqrt(np.mean(e ** 2))) if e.size else float("nan")
        out[f"n_h{h}"] = int(e.size)
    s = model.seed
    out["fitted_seed_eps"] = float(s.detach()) if s is not None else None
    out["val_loss"] = best
    out["u_init"] = u0
    out["r_init"] = r0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=48)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 6, 24, 48])
    ap.add_argument("--stride", type=int, default=12)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--cap-u", type=float, default=0.25)
    ap.add_argument("--cap-r", type=float, default=0.25)
    ap.add_argument("--out", default="results/exp05_real_dynamics.json")
    a = ap.parse_args()

    y0, X, yt, m, fips, panel = load_pooled(a.horizon, a.stride)
    X = add_context(X, y0, a.horizon)
    print(f"pooled samples {len(y0):,} over {len(set(panel))} storms, "
          f"{len(set(fips)):,} counties, {X.shape[-1]} driver channels, "
          f"horizon {a.horizon} h")
    print(f"observed targets: {m.mean()*100:.1f}%   mean y0 {y0.mean():.5f}")

    rows = []
    for seed in a.seeds:
        fold = make_folds(sorted(set(fips)), k=a.k, seed=seed)
        fmap = {f: fo for f, fo in zip(sorted(set(fips)), fold)}
        assign = np.array([fmap[f] for f in fips])
        for f in range(a.k):
            te = np.where(assign == f)[0]; tr = np.where(assign != f)[0]
            for b in BASELINES:
                r = run_baseline(b, tr, te, (y0, X, yt, m), a)
                rows.append({"arm": b, "seed": seed, "fold": f,
                             "n_test": len(te), "wall_s": 0.0, **r})
            for arm in ARMS:
                t0 = time.time()
                r = run_arm(arm, tr, te, (y0, X, yt, m), a, seed)
                wall = round(time.time() - t0, 1)
                rows.append({"arm": arm.value, "seed": seed, "fold": f,
                             "n_test": len(te), "wall_s": wall, **r})
                print(f"  seed {seed} fold {f} {arm.value:<18} "
                      + " ".join(f"h{h}={r[f'rmse_h{h}']:.5f}" for h in a.horizons)
                      + f"  {wall}s", flush=True)

    out = ROOT / a.out; out.parent.mkdir(parents=True, exist_ok=True)
    cfg = dict(vars(a)); cfg["out"] = a.out
    out.write_text(json.dumps({"config": cfg, "rows": rows}, indent=2))

    print(f"\n=== pooled over {a.k} folds x {len(a.seeds)} seeds ===")
    print(f"{'arm':<22}" + "".join(f"{'RMSE h+'+str(h):>19}" for h in a.horizons))
    for name in BASELINES + [x.value for x in ARMS]:
        g = [r for r in rows if r["arm"] == name]
        line = f"{name:<22}"
        for h in a.horizons:
            v = [r[f"rmse_h{h}"] for r in g if np.isfinite(r[f"rmse_h{h}"])]
            line += f"{np.mean(v):>13.5f}±{np.std(v):<5.5f}"
        e = [r.get("fitted_seed_eps") for r in g if r.get("fitted_seed_eps") is not None]
        if e:
            line += f"   eps={np.mean(e):.4f}"
        print(line)
    print(f"\nwritten: {a.out}")


if __name__ == "__main__":
    main()
