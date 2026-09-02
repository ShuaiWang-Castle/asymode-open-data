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
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from asymode.dynamics import (TwoRateODE, TwoRateConfig, InflowForm,   # noqa: E402
                              calibrate_init)
from asymode.evalproto import (Task, make_folds, to_hourly,           # noqa: E402
                               inner_split)
from asymode import panels as panelset                              # noqa: E402

INTERIM = ROOT / "data" / "interim"
@dataclass(frozen=True)
class Arm:
    """One dynamical form, with the capacity it is given.

    Capacity is per-arm because the single-rate forms use one network where the
    two-rate forms use two. Left at a common width they would carry half the
    parameters, and "two rates beat one" would be readable as "3,138 parameters
    beat 1,569" -- a difference in capacity wearing the costume of a difference in
    structure. `hidden=48` puts one network at 3,121 parameters against the
    two-rate arms' 3,138. Neither matching is uniquely right, so the width-matched
    variant is run alongside and both are reported.
    """
    name: str
    inflow: InflowForm
    hidden: int | None = None      # None -> the shared --hidden


ARMS = [
    Arm("susceptible", InflowForm.SUSCEPTIBLE),
    Arm("transmission", InflowForm.TRANSMISSION),
    Arm("transmission_seed", InflowForm.TRANSMISSION_SEED),
    # The single-rate ladder. Each step adds exactly one structure:
    #   net -> net_scaled          adds state-dependent scaling
    #   net_scaled -> two rates    adds concurrency: two non-negative rates are
    #                              both live every step, so a county can lose and
    #                              regain customers at once, where one signed rate
    #                              forces the two directions to exclude each other.
    Arm("net", InflowForm.NET, hidden=48),
    Arm("net_scaled", InflowForm.NET_SCALED, hidden=48),
    Arm("net_scaled_narrow", InflowForm.NET_SCALED, hidden=32),
]


def origin_range(n: int, horizon: int, stride: int, min_history: int) -> range:
    """The forecast origins a panel of length `n` contributes.

    Factored out because a second loader that needs to line up with `load_pooled`
    sample for sample must not restate this: two copies of a loop are two chances
    to drift, and a misalignment here would silently pair one sample's features
    with another sample's target.
    """
    return range(min_history, n - horizon, stride)


def load_pooled(horizon: int, stride: int, min_history: int = 24,
                panels: list[str] | None = None):
    """Every (county, storm, origin) with drivers, pooled across storms.

    Pooling is the point: a county is the unit of held-out evaluation, and a model
    that only works on the storm it was fitted to is not a model of the dynamics.

    `panels` names the set explicitly. Passing `None` pools whatever is on disk,
    which is only safe when nothing is still downloading -- see `asymode.panels`.
    """
    keep = set(panels) if panels is not None else None
    Y0, X, YT, M, FIPS, PANEL = [], [], [], [], [], []
    for pf in sorted(INTERIM.glob("panel_*.npz")):
        day = pf.stem.replace("panel_", "")
        df = INTERIM / f"drivers_{day}.npz"
        if not df.exists() or (keep is not None and day not in keep):
            continue
        pz, dz = np.load(pf, allow_pickle=True), np.load(df, allow_pickle=True)
        assert pz["fips"].tolist() == dz["fips"].tolist()
        yh, oh = to_hourly(pz["y"], pz["observed"])
        Xh, fips = dz["X"], pz["fips"].tolist()
        n = min(yh.shape[1], Xh.shape[1])
        yh, oh, Xh = yh[:, :n], oh[:, :n], Xh[:, :n]
        yh = np.nan_to_num(yh)
        for o in origin_range(n, horizon, stride, min_history):
            Y0.append(yh[:, o]); X.append(Xh[:, o + 1:o + 1 + horizon])
            YT.append(yh[:, o + 1:o + 1 + horizon]); M.append(oh[:, o + 1:o + 1 + horizon])
            FIPS.append(np.array(fips)); PANEL.append(np.full(len(fips), day))
    return (np.concatenate(Y0), np.concatenate(X), np.concatenate(YT),
            np.concatenate(M), np.concatenate(FIPS), np.concatenate(PANEL))


def load_history(horizon: int, stride: int, lookback: int, min_history: int = 24,
                 panels: list[str] | None = None) -> np.ndarray:
    """The `lookback` hours of observed state before each origin, in sample order.

    Walks the panels in the same order, with the same driver requirement and the
    same origins as `load_pooled`, so row i here is row i there. The caller is
    expected to assert that, and does.

    This is *more* information than the dynamics receive: they are given the
    origin state and the drivers, and nothing before the origin. A baseline that
    reads it is not being handicapped, it is being advantaged, and any arm built
    on it has to be labelled that way.
    """
    keep = set(panels) if panels is not None else None
    H = []
    for pf in sorted(INTERIM.glob("panel_*.npz")):
        day = pf.stem.replace("panel_", "")
        df = INTERIM / f"drivers_{day}.npz"
        if not df.exists() or (keep is not None and day not in keep):
            continue
        pz, dz = np.load(pf, allow_pickle=True), np.load(df, allow_pickle=True)
        yh, _ = to_hourly(pz["y"], pz["observed"])
        n = min(yh.shape[1], dz["X"].shape[1])
        yh = np.nan_to_num(yh[:, :n])
        for o in origin_range(n, horizon, stride, min_history):
            H.append(yh[:, o - lookback + 1:o + 1])
    return np.concatenate(H)


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


def run_arm(arm, tr, te, data, args, seed, fips, fold_id):
    y0, X, yt, m = data
    torch.manual_seed(seed); np.random.seed(seed)
    mu = X[tr].reshape(-1, X.shape[-1]).mean(0)
    sd = X[tr].reshape(-1, X.shape[-1]).std(0) + 1e-6      # training folds only
    Xn = ((X - mu) / sd).astype(np.float32)

    t = lambda a: torch.tensor(a)
    # Calibrated on training folds only, by the same rule for every arm.
    ytr = np.concatenate([y0[tr][:, None], yt[tr]], axis=1)
    mtr = np.concatenate([np.ones((len(tr), 1), bool), m[tr]], axis=1)
    u0, r0 = calibrate_init(ytr, mtr, arm.inflow)
    hid = args.hidden if arm.hidden is None else arm.hidden
    cfg = TwoRateConfig(d_in=Xn.shape[-1], cap_u=args.cap_u, cap_r=args.cap_r,
                        hidden_u=hid, hidden_r=hid, inflow=arm.inflow,
                        u_init=u0, r_init=r0)
    model = TwoRateODE(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Early stopping holds out counties, not rows: see `inner_split`. A row-random
    # split leaves the same counties on both sides and cannot see the failure the
    # outer folds exist to measure.
    fi, vi = inner_split(fips[tr], seed=seed, fold=fold_id)
    tr_arr = np.asarray(tr)
    fit, va = tr_arr[fi], tr_arr[vi]
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
    ap.add_argument("--panels", default=None,
                    help="panel set: omit for the manifest, 'auto' to pool what "
                         "is on disk (exploration only), or a path to a JSON file")
    ap.add_argument("--out", default="results/exp05_real_dynamics.json")
    a = ap.parse_args()

    want, panel_digest = panelset.resolve(INTERIM, a.panels)
    y0, X, yt, m, fips, panel = load_pooled(a.horizon, a.stride, panels=want)
    X = add_context(X, y0, a.horizon)
    print(f"pooled samples {len(y0):,} over {len(set(panel))} storms [{panel_digest}], "
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
                r = run_arm(arm, tr, te, (y0, X, yt, m), a, seed, fips, f)
                wall = round(time.time() - t0, 1)
                rows.append({"arm": arm.name, "seed": seed, "fold": f,
                             "inflow": arm.inflow.value,
                             "n_param": sum(p.numel() for p in
                                            __import__("torch").nn.ModuleList(
                                                [m for m in (r.pop("_model", None),)
                                                 if m is not None]).parameters()) or r.get("n_param"),
                             "n_test": len(te), "wall_s": wall, **r})
                print(f"  seed {seed} fold {f} {arm.name:<18} "
                      + " ".join(f"h{h}={r[f'rmse_h{h}']:.5f}" for h in a.horizons)
                      + f"  {wall}s", flush=True)

    out = ROOT / a.out; out.parent.mkdir(parents=True, exist_ok=True)
    cfg = dict(vars(a)); cfg["out"] = a.out
    cfg["panels"] = sorted(set(panel.tolist()))
    cfg["panel_digest"] = panel_digest
    cfg["channels"] = panelset.channel_names(INTERIM)
    cfg["channel_digest"] = panelset.channel_digest(cfg["channels"])
    out.write_text(json.dumps({"config": cfg, "rows": rows}, indent=2))

    print(f"\n=== pooled over {a.k} folds x {len(a.seeds)} seeds ===")
    print(f"{'arm':<22}" + "".join(f"{'RMSE h+'+str(h):>19}" for h in a.horizons))
    for name in BASELINES + [x.name for x in ARMS]:
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
