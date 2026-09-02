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
from asymode import splits, schema                                   # noqa: E402
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
                panels: list[str] | None = None, with_time: bool = False):
    """Every (county, storm, origin) with drivers, pooled across storms.

    Pooling is the point: a county is the unit of held-out evaluation, and a model
    that only works on the storm it was fitted to is not a model of the dynamics.

    `panels` names the set explicitly. Passing `None` pools whatever is on disk,
    which is only safe when nothing is still downloading -- see `asymode.panels`.

    With `with_time=True` an eighth array is returned: the hour of day (0-23, on
    the panel timestamp clock, which the driver builder aligns to ERA5 `valid_time`,
    i.e. UTC) of the *first forecast step* of every sample. The clock channels are
    built from it, so they follow the calendar and do not restart at every origin.

    Returns `(y0, X, y, mask, fips, panel, origin)`. The last is the step index
    each sample's forecast starts from; together with `panel` it names the
    forecast, which is the grouping any within-forecast statistic needs and which
    `fips` and `panel` alone cannot express -- one storm contributes many origins.
    """
    keep = set(panels) if panels is not None else None
    Y0, X, YT, M, FIPS, PANEL, ORIGIN, T0H = [], [], [], [], [], [], [], []
    for pf in sorted(INTERIM.glob("panel_*.npz")):
        day = pf.stem.replace("panel_", "")
        df = INTERIM / f"drivers_{day}.npz"
        if not df.exists() or (keep is not None and day not in keep):
            continue
        pz, dz = np.load(pf, allow_pickle=True), np.load(df, allow_pickle=True)
        assert pz["fips"].tolist() == dz["fips"].tolist()
        yh, oh = to_hourly(pz["y"], pz["observed"])
        # hour of day of every hourly step: hourly step k covers 15-min stamps 4k..4k+3
        ts15 = pz["ts"]; hours_of_day = np.array([int(str(t)[11:13]) for t in ts15[::4]], dtype=np.int64)
        Xh, fips = dz["X"], pz["fips"].tolist()
        n = min(yh.shape[1], Xh.shape[1])
        yh, oh, Xh = yh[:, :n], oh[:, :n], Xh[:, :n]
        yh = np.nan_to_num(yh)
        for o in origin_range(n, horizon, stride, min_history):
            Y0.append(yh[:, o]); X.append(Xh[:, o + 1:o + 1 + horizon])
            YT.append(yh[:, o + 1:o + 1 + horizon]); M.append(oh[:, o + 1:o + 1 + horizon])
            FIPS.append(np.array(fips)); PANEL.append(np.full(len(fips), day))
            # The step this forecast starts from. (panel, origin) identifies one
            # forecast: every county in it shares a driver window and a clock, and
            # the level/ranking decomposition is only meaningful within one.
            ORIGIN.append(np.full(len(fips), o, dtype=np.int32))
            T0H.append(np.full(len(fips), hours_of_day[o + 1], dtype=np.int64))
    out = (np.concatenate(Y0), np.concatenate(X), np.concatenate(YT),
           np.concatenate(M), np.concatenate(FIPS), np.concatenate(PANEL),
           np.concatenate(ORIGIN))
    return out + (np.concatenate(T0H),) if with_time else out


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


def add_context(X: np.ndarray, y0: np.ndarray, hours: int, t0_hour=None,
                clock: str = "utc_hour") -> np.ndarray:
    """Append clock channels.

    `clock="utc_hour"` (default): sin/cos of the hour of day of each forecast step,
    from `t0_hour` (the hour of the first step, per sample, from the panel
    timestamps). The phase follows the calendar and differs between two origins
    of the same storm by their offset.

    `clock="lead_phase_old"`: the legacy channel -- sin/cos of the lead time modulo
    24 h, identical for every sample. It is *not* an hour of day; it is kept only
    as a diagnostic arm and is refused by the comparability checker against any
    other clock. `clock="none"`: no clock channels.
    """
    n, T, F = X.shape
    if clock == "none":
        return X.astype(np.float32)
    if clock == "lead_phase_old":
        t = np.arange(T)[None, :].repeat(n, 0).astype(np.float64)
    elif clock == "utc_hour":
        if t0_hour is None:
            raise ValueError("clock='utc_hour' needs t0_hour (use load_pooled(with_time=True))")
        t = (np.asarray(t0_hour, dtype=np.int64)[:, None] + np.arange(T)[None, :]) % 24
        t = t.astype(np.float64)
    else:
        raise ValueError(f"unknown clock {clock!r}")
    out = np.concatenate([X,
                          np.sin(2 * np.pi * t / 24)[..., None],
                          np.cos(2 * np.pi * t / 24)[..., None]], axis=-1)
    return out.astype(np.float32)


def outer_assignment(fips: np.ndarray, panel: np.ndarray, split_unit: str, k: int,
                     outer_split_seed: int):
    """Fold of every row, from ONE pinned map that depends on the outer split seed only.

    Returns (assign, mapping, digest, unit_ids). `split_unit="event"` holds out whole
    storm panels (balanced on sample counts); `"county"` holds out counties across
    all events. Neither sees the model seed.
    """
    if split_unit == "event":
        sizes = {p_: int(c) for p_, c in zip(*np.unique(panel, return_counts=True))}
        if len(sizes) < k:
            raise SystemExit(f"event-held-out needs >= {k} panels, have {len(sizes)}")
        mapping = splits.event_folds(sizes, k=k, outer_split_seed=outer_split_seed)
        unit_ids = panel
    elif split_unit == "county":
        mapping = splits.county_folds(sorted(set(fips.tolist())), k=k, outer_split_seed=outer_split_seed)
        unit_ids = fips
    else:
        raise ValueError(split_unit)
    assign = splits.assign_rows(unit_ids, mapping)
    for f in range(k):
        splits.check_disjoint(unit_ids, assign, f)
    return assign, mapping, splits.split_digest(mapping), unit_ids


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
    out["_test_pred"] = np.asarray(pred)[:, [h - 1 for h in args.horizons]].astype(np.float32)
    return out


BASELINES = ["zero", "persistence", "damped_persistence"]


def run_arm(arm, tr, te, data, args, seed, fips, fold_id, inner_units=None,
            inner_split_seed=None):
    """`seed` is the MODEL seed (initialisation and data order). The early-stopping
    holdout is drawn by `inner_split_seed` (default: the model seed, legacy) over
    `inner_units` (default: county codes), never over rows."""
    y0, X, yt, m = data
    torch.manual_seed(seed); np.random.seed(seed)
    inner_split_seed = seed if inner_split_seed is None else inner_split_seed
    inner_units = fips if inner_units is None else inner_units
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
    fi, vi = inner_split(inner_units[tr], seed=inner_split_seed, fold=fold_id)
    tr_arr = np.asarray(tr)
    fit, va = tr_arr[fi], tr_arr[vi]
    Y0, XX, YT, MM = t(y0), t(Xn), t(yt), t(m.astype(np.float32))

    def loss_on(ix):
        pred = model(Y0[ix], XX[ix])
        se = (pred - YT[ix]) ** 2 * MM[ix]
        return se.sum() / MM[ix].sum().clamp_min(1.0)

    best, best_state, bad, ran = float("inf"), None, 0, 0
    for ep in range(args.epochs):
        ran = ep + 1
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
    out = {"_test_pred": pred[:, [h - 1 for h in args.horizons]].astype(np.float32)}   # scored as-is
    for h in args.horizons:
        e = (pred[:, h - 1] - yt[te][:, h - 1])[m[te][:, h - 1]]
        out[f"rmse_h{h}"] = float(np.sqrt(np.mean(e ** 2))) if e.size else float("nan")
        out[f"n_h{h}"] = int(e.size)
    s = model.seed
    out["fitted_seed_eps"] = float(s.detach()) if s is not None else None
    # Recorded per arm because capacity varies across the single- and two-rate
    # forms; a structural comparison has to be readable alongside it.
    out["n_param"] = sum(p.numel() for p in model.parameters())
    out["hidden"] = hid
    # Collapse screen. The single-rate forms have an absorbing state the two-rate
    # forms do not: without the (1 - y) factor a persistently negative net rate
    # drives the state onto the clip floor and nothing pushes it back, so the arm
    # can degenerate into the all-zero predictor while still posting a plausible
    # RMSE against a target that is zero about half the time. An arm that loses
    # because it degenerated has not lost a structural argument, and the two must
    # be told apart from the file rather than by noticing a coincidence.
    out["pred_sd"] = float(pred.std())
    out["pred_max"] = float(pred.max())
    out["frac_pred_zero"] = float((pred <= 0.0).mean())
    out["val_loss"] = best
    # Whether the fit stopped because it converged or because it ran out of
    # budget. Without it, a model that lost cannot be distinguished from a
    # model that was not finished training -- and the standard applied to a
    # baseline has to be applied to the model under test.
    out["epochs_run"] = ran
    out["hit_epoch_cap"] = bool(ran >= args.epochs)
    out["u_init"] = u0
    out["r_init"] = r0
    out["frac_pred_one"] = float((pred >= 1.0).mean())      # clamp activity at the top
    return out


def _stash(oof, name, seed, fold, te, pred_te, n, seeds, n_h):
    """Place one fold's held-out predictions into the per-arm OOF arrays."""
    if name not in oof:
        oof[name] = {"pred": np.full((len(seeds), n, n_h), np.nan, np.float32),
                     "fold_of": np.full((len(seeds), n), -1, np.int8)}
    si = list(seeds).index(seed)
    oof[name]["pred"][si, te] = pred_te
    oof[name]["fold_of"][si, te] = fold


def _write_oof(oof, out_dir, yt, m, fips, panel, origin, origin_id, a, cfg):
    """One npz per arm in the audited layout. Refuses a store with holes."""
    H = [int(h) for h in a.horizons]
    y = yt[:, [h - 1 for h in H]].astype(np.float32)
    mask = m[:, [h - 1 for h in H]].astype(bool)
    for name, st in oof.items():
        if (st["fold_of"] < 0).any():
            raise SystemExit(f"OOF store for '{name}' has samples never held out; refusing to write")
        # A per-horizon regressor never predicts unscored cells, so NaN is allowed
        # exactly where the mask is False and nowhere else.
        if np.isnan(st["pred"][:, mask]).any():
            raise SystemExit(f"OOF store for '{name}' has NaN inside the scored mask; refusing to write")
        np.savez_compressed(out_dir / f"oof_{name}.npz", pred=st["pred"], fold_of=st["fold_of"],
                            y=y, mask=mask, origin_id=origin_id.astype(np.int32),
                            origin_step=np.asarray(origin, np.int32), fips=np.asarray(fips, str),
                            panel=np.asarray(panel, str), horizons=np.array(H, np.int32),
                            seeds=np.array(list(a.seeds), np.int32),
                            panel_digest=str(cfg.get("digest") or cfg.get("panel_digest") or ""),
                            channel_digest=str(cfg.get("channel_digest") or ""),
                            source=str(cfg.get("source") or ""))
        print(f"  oof_{name}.npz written", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=48)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 6, 24, 48])
    ap.add_argument("--stride", type=int, default=12)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="MODEL seeds (initialisation and data order); alias --model-seeds")
    ap.add_argument("--model-seeds", dest="seeds", type=int, nargs="+")
    ap.add_argument("--outer-split-seed", type=int, default=0, help="decides the held-out units; nothing else")
    ap.add_argument("--inner-split-seed", type=int, default=0, help="decides the early-stopping holdout")
    ap.add_argument("--split-unit", choices=["event", "county"], default="event",
                    help="event = PRIMARY (whole storms held out); county = secondary (unseen counties within observed families)")
    ap.add_argument("--clock", choices=sorted(schema.CLOCKS), default="utc_hour")
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
    ap.add_argument("--save-oof", action="store_true",
                    help="also write oof_<arm>.npz next to --out, in the audited out-of-fold layout")
    ap.add_argument("--out", default="results/exp05_real_dynamics.json")
    a = ap.parse_args()

    t_launch = time.time()
    source_at_launch = panelset.source_version(ROOT)
    want, panel_digest = panelset.resolve(INTERIM, a.panels)
    y0, X, yt, m, fips, panel, origin, t0h = load_pooled(
        a.horizon, a.stride, panels=want, with_time=True)
    X = add_context(X, y0, a.horizon, t0_hour=t0h, clock=a.clock)
    assign, split_map, split_digest, unit_ids = outer_assignment(
        fips, panel, a.split_unit, a.k, a.outer_split_seed)
    split_path = splits.save_split(split_map, a.split_unit, a.k, a.outer_split_seed, ROOT)
    print(f"split_unit {a.split_unit} · outer_split_seed {a.outer_split_seed} · digest {split_digest} · "
          f"{split_path.relative_to(ROOT)} · clock {a.clock}")
    print(f"pooled samples {len(y0):,} over {len(set(panel))} storms [{panel_digest}], "
          f"{len(set(fips)):,} counties, {X.shape[-1]} driver channels, "
          f"horizon {a.horizon} h")
    print(f"observed targets: {m.mean()*100:.1f}%   mean y0 {y0.mean():.5f}")

    rows = []
    # Out-of-fold store: each sample predicted once per seed by the fold that
    # held it out. Written only with --save-oof, in the layout the diagnostics
    # audit -- origin_id is a dense code of (panel, origin_step), never the step.
    oof = {}
    _, origin_id = np.unique(np.array([f"{p_}|{o_}" for p_, o_ in zip(panel, origin)]), return_inverse=True)
    for seed in a.seeds:                      # model seeds; the split does not move
        for f in range(a.k):
            te = np.where(assign == f)[0]; tr = np.where(assign != f)[0]
            for b in BASELINES:
                r = run_baseline(b, tr, te, (y0, X, yt, m), a)
                _stash(oof, b, seed, f, te, r.pop("_test_pred"), len(y0), a.seeds, len(a.horizons))
                rows.append({"arm": b, "seed": seed, "fold": f,
                             "n_test": len(te), "wall_s": 0.0, **r})
            for arm in ARMS:
                t0 = time.time()
                r = run_arm(arm, tr, te, (y0, X, yt, m), a, seed, fips, f,
                            inner_units=unit_ids, inner_split_seed=a.inner_split_seed)
                _stash(oof, arm.name, seed, f, te, r.pop("_test_pred"), len(y0), a.seeds, len(a.horizons))
                wall = round(time.time() - t0, 1)
                rows.append({"arm": arm.name, "seed": seed, "fold": f,
                             "inflow": arm.inflow.value,
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
    cfg["source"] = source_at_launch
    hp = {k: cfg[k] for k in ("epochs", "patience", "batch", "lr", "hidden", "cap_u", "cap_r",
                              "horizon", "stride", "k", "horizons") if k in cfg}
    cfg.update(schema.result_header(
        experiment_id=Path(a.out).stem, source=source_at_launch, panel_ids=cfg["panels"],
        panel_digest=panel_digest, channel_names=schema.channel_list(cfg["channels"], a.clock),
        channel_digest=cfg["channel_digest"], clock=a.clock, split_unit=a.split_unit,
        outer_split_digest=split_digest, outer_split_seed=a.outer_split_seed,
        inner_split_seed=a.inner_split_seed, model_seeds=a.seeds, hyperparameters=hp))
    cfg["split_file"] = str(split_path.relative_to(ROOT))
    cfg["wall_time_s"] = round(time.time() - t_launch, 1)
    fits = [r for r in rows if "epochs_run" in r]
    cfg["convergence"] = {"n_fits": len(fits), "n_at_epoch_cap": int(sum(r["hit_epoch_cap"] for r in fits)),
                          "max_frac_pred_zero": max((r["frac_pred_zero"] for r in fits), default=None),
                          "max_frac_pred_one": max((r.get("frac_pred_one", 0.0) for r in fits), default=None)}
    for r in rows:
        r.pop("_test_pred", None)
    out.write_text(json.dumps({"config": cfg, "rows": rows}, indent=2))
    if a.save_oof:
        _write_oof(oof, out.parent, yt, m, fips, panel, origin, origin_id, a, cfg)

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
