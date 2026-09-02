"""EXP07 -- learned baselines: gradient-boosted trees and a linear sequence model.

The ablation arms in EXP05 and EXP08 answer "does this structure help against
other structures". They do not answer "does any of this beat a competent
off-the-shelf regressor given the same inputs", and a reader is entitled to ask.
Two baselines, chosen to be the strongest cheap representatives of their family
rather than the most numerous:

  trees   histogram gradient boosting, one model per scored horizon
  linear  a direct multi-step linear map with a series decomposition

Both are scored on exactly the samples, folds, mask and horizons the dynamics are
scored on. None of these numbers is read from another experiment: a comparison
across different sample sets is not a comparison, and the panel digest in the
config block is what lets that be checked rather than assumed.

**The sequence baseline departs from the published DLinear in one way, on
purpose.** DLinear is channel-independent and consumes only a lookback window of
the series. Here the forcing *is* the storm: a model given the outage history and
no weather is a dressed-up persistence, which EXP04 already covers, and beating
it would establish nothing. So an exogenous pathway is added. That is a deviation
from the reference formulation and it favours the baseline.

Three variants separate where the skill comes from, since with an exogenous
pathway bolted on it is otherwise impossible to say:

  linear_matched   origin state + drivers -- exactly what the dynamics receive
  linear_lookback  the above plus a pre-origin window of the state
  linear_histonly  the pre-origin window alone, no drivers

`linear_lookback` and `linear_histonly` read history the dynamics never see, so
they are *advantaged*, not handicapped. If the dynamics still win, the claim is
stronger; if they lose, the pre-origin state matters and this file is where that
was found out. Either way the arms are labelled rather than pooled.

Environment note, measured here rather than assumed. LightGBM and torch each
ship their own OpenMP runtime, and on this machine loading both into one process
segfaults during a multi-threaded LightGBM fit -- reproducibly, in either import
order, with no Python-level error. Two things were checked rather than guessed:
the crash disappears with `n_jobs=1`, and scikit-learn's histogram gradient
boosting coexists with torch at full threading while returning results identical
to the same fit in a torch-free process.

The tree baseline therefore uses `HistGradientBoostingRegressor` by default. It
is the same algorithm -- histogram-binned gradient-boosted trees -- and it costs
no dependency and no fragile coincidence. `--tree-lib lightgbm` selects the named
library instead and forces single-threaded fitting, which is the configuration
that was verified not to crash; LightGBM is an optional install and is not
required to run this file.

Every arm here stops on the same rule as the dynamics: a validation set carved
out of the training folds *by county*, and eight evaluations without improvement.
Holding out rows instead would leave the same counties on both sides of the
stopping criterion, which cannot see the generalisation failure the outer folds
exist to measure -- a model selected that way is selected for its performance on
counties it has already read. The tree library's own early stopping splits by
row, so it is switched off and the stopping loop is run here instead.
"""
from __future__ import annotations

import argparse, json, importlib, importlib.util, sys, time
from pathlib import Path

import numpy as np
import torch                      # MUST precede any lightgbm import; see above
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from asymode.evalproto import make_folds, inner_split                 # noqa: E402
from asymode import splits, schema                                   # noqa: E402
from asymode import panels as panelset                                # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "exp05", Path(__file__).resolve().parent / "exp05_real_dynamics.py")
exp05 = importlib.util.module_from_spec(_spec)
# Register before executing. `dataclasses` resolves a class's module through
# `sys.modules` while it processes the decorator, so a module loaded from a spec
# and never registered raises as soon as it defines a dataclass -- which exp05
# now does. Loading it this way is deliberate: it guarantees these experiments
# pool their samples with exactly the code that built exp05's, rather than a copy.
sys.modules[_spec.name] = exp05
_spec.loader.exec_module(exp05)
load_pooled, load_history, add_context = exp05.load_pooled, exp05.load_history, exp05.add_context

INTERIM = ROOT / "data" / "interim"


# --------------------------------------------------------------------------
# tree ensemble
# --------------------------------------------------------------------------
def tree_features(y0, X, hist, h, use_hist):
    """The information set for horizon h, matched to what the dynamics get.

    The dynamics roll forward from the origin state and read the drivers one step
    at a time, so by step h they have seen x_1..x_h and nothing beyond. The path
    aggregates below therefore run from the origin to step h and stop. Aggregating
    across the whole forecast window would hand the trees a strictly stronger
    assumption than the model they are being compared against -- and is the
    anticausal family the pre-registration excludes.
    """
    win = X[:, :h]                                   # (N, h, C)
    f = [y0[:, None], X[:, h - 1], win.mean(1), win.max(1), win.min(1),
         win.sum(1), win[:, -1] - win[:, 0]]
    if use_hist:
        f += [hist, hist.mean(1)[:, None], hist.max(1)[:, None],
              (hist[:, -1] - hist[:, 0])[:, None]]
    return np.concatenate(f, axis=1).astype(np.float32)


def _fit_trees(args, seed, Xf, yf, Xv, yv):
    """Fit with early stopping on a county-held-out set.

    Both libraries can stop themselves, and neither is used to: they carve their
    validation set out by row, which is the split this protocol specifically must
    not use. The caller supplies the held-out counties and the loop below decides
    when to stop.
    """
    rmse = lambda mdl: float(np.sqrt(np.mean((mdl.predict(Xv) - yv) ** 2)))
    if args.tree_lib == "lightgbm":
        lgb = importlib.import_module("lightgbm")
        mdl = lgb.LGBMRegressor(
            n_estimators=args.trees, learning_rate=args.tree_lr,
            num_leaves=args.leaves, min_child_samples=50, random_state=seed,
            # Single-threaded on purpose: multi-threaded LightGBM in a process
            # that has imported torch segfaults here. See the module docstring.
            n_jobs=1, verbose=-1)
        mdl.fit(Xf, yf, eval_set=[(Xv, yv)],
                callbacks=[lgb.early_stopping(args.patience, verbose=False)])
        return mdl, int(mdl.best_iteration_ or args.trees), rmse(mdl)

    from sklearn.ensemble import HistGradientBoostingRegressor
    kw = dict(learning_rate=args.tree_lr, max_leaf_nodes=args.leaves,
              min_samples_leaf=50, l2_regularization=1.0, random_state=seed,
              early_stopping=False)
    probe = HistGradientBoostingRegressor(warm_start=True, max_iter=args.tree_step, **kw)
    best, best_iter, bad = float("inf"), args.tree_step, 0
    for it in range(args.tree_step, args.trees + 1, args.tree_step):
        probe.set_params(max_iter=it)
        probe.fit(Xf, yf)
        v = rmse(probe)
        if v < best - 1e-12:
            best, best_iter, bad = v, it, 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    # Refit at the chosen size. Boosting is additive and deterministic under a
    # fixed seed, so this reproduces the probe's state at `best_iter` exactly,
    # which the warm-started object has already grown past.
    final = HistGradientBoostingRegressor(max_iter=best_iter, **kw)
    final.fit(Xf, yf)
    return final, best_iter, best


def run_trees(tr, te, data, hist, args, seed, fips, fold_id, use_hist=False, inner_units=None,
              inner_split_seed=None):
    y0, X, yt, m = data
    inner_units = fips if inner_units is None else inner_units
    inner_split_seed = seed if inner_split_seed is None else inner_split_seed
    out = {}
    for h in args.horizons:
        # Unobserved targets are dropped, never imputed -- from training and from
        # scoring alike, so the row counts here match every other experiment's.
        f_tr = tr[m[tr, h - 1]]
        f_te = te[m[te, h - 1]]
        Xtr = tree_features(y0[f_tr], X[f_tr], hist[f_tr] if hist is not None else None, h, use_hist)
        Xte = tree_features(y0[f_te], X[f_te], hist[f_te] if hist is not None else None, h, use_hist)
        fi, vi = inner_split(inner_units[f_tr], seed=inner_split_seed, fold=fold_id)
        mdl, n_iter, val = _fit_trees(args, seed, Xtr[fi], yt[f_tr, h - 1][fi],
                                      Xtr[vi], yt[f_tr, h - 1][vi])
        out[f"n_iter_h{h}"], out[f"val_h{h}"] = n_iter, val
        pred = mdl.predict(Xte)
        # keep the scored predictions for the out-of-fold export: rows of `te`
        # inside this horizon's mask get the prediction, the rest stay NaN
        if "_test_pred" not in out:
            out["_test_pred"] = np.full((len(te), len(args.horizons)), np.nan, np.float32)
        out["_test_pred"][np.flatnonzero(m[te, h - 1]), list(args.horizons).index(h)] = pred.astype(np.float32)
        e = pred - yt[f_te, h - 1]
        out[f"rmse_h{h}"] = float(np.sqrt(np.mean(e ** 2))) if e.size else float("nan")
        out[f"n_h{h}"] = int(e.size)
        out[f"pred_sd_h{h}"] = float(np.std(pred))
        out[f"frac_neg_h{h}"] = float(np.mean(pred < 0))
    return out


# --------------------------------------------------------------------------
# linear direct multi-step
# --------------------------------------------------------------------------
class DecompLinear(nn.Module):
    """DLinear's decomposition head, plus a driver pathway.

    The series half is the published model: a moving-average trend, the remainder,
    and one linear map from the lookback to the horizon for each. The driver half
    is the addition, and it is a single linear map over the flattened forecast
    window, which is the most flexible thing a linear model can do with it.

    `bounded` selects the output head, and neither choice dominates, which is why
    both are run rather than argued about.

    An unbounded (published) head can express persistence: at h+1 the right answer
    is close to the origin state, and a linear map reaches it directly. A squashed
    head cannot, because no linear function of y0 equals logit(y0). Measured on
    identical folds, the unbounded head is much better at h+1 (0.018 vs 0.027) --
    and much worse at long horizons (h+48 0.051 vs 0.043), where it is beaten even
    by predicting zero, because nothing stops it running away.

    Feeding the state in logit space should in principle give both properties at
    once, since the identity map becomes expressible under a sigmoid. It was tried
    and it is worse at every horizon: roughly half the origin states are exactly
    zero, so the transform puts a large point mass at the clipping floor and the
    linear fit is dominated by it.

    So both heads run as separate arms. Where these baselines are compared against
    the dynamics, the stronger of the two at each horizon is the one that counts:
    the point of a baseline is to be hard to beat, and choosing the weaker variant
    would flatter the model under test.

    Predictions are clipped to [0, 1] at scoring time, which is ordinary
    post-processing and can only help the baseline.
    """

    def __init__(self, lookback: int, horizon: int, n_ch: int,
                 use_hist: bool, use_drivers: bool, bounded: bool = True,
                 kernel: int = 25):
        super().__init__()
        self.use_hist, self.use_drivers, self.bounded = use_hist, use_drivers, bounded
        self.kernel = kernel
        if use_hist:
            self.trend = nn.Linear(lookback, horizon)
            self.resid = nn.Linear(lookback, horizon)
        self.origin = nn.Linear(1, horizon)
        if use_drivers:
            self.drv = nn.Linear(horizon * n_ch, horizon)
        self.bias0 = nn.Parameter(torch.zeros(()))
        # Every branch starts at zero so the model's initial prediction really is
        # `bias0`, the training-fold mean. Torch's default initialisation draws
        # each branch bias from U(-1, 1), which against a target whose mean is
        # near 1e-2 puts the starting prediction two orders of magnitude too high
        # and an order of magnitude too spread out; the fit then has to climb back
        # down before it can begin, and on the history-only arm it does not manage
        # it. A baseline hobbled by its initialisation is not evidence about the
        # model it is compared against, which is the same reason the output layer
        # is linear rather than squashed.
        for lin in (b for b in (getattr(self, "trend", None), getattr(self, "resid", None),
                                self.origin, getattr(self, "drv", None)) if b is not None):
            nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)

    def _decompose(self, h):
        pad = self.kernel // 2
        t = torch.nn.functional.avg_pool1d(
            torch.nn.functional.pad(h.unsqueeze(1), (pad, pad), mode="replicate"),
            self.kernel, stride=1).squeeze(1)
        return t, h - t

    def forward(self, y0, X, hist):
        z = self.origin(y0.unsqueeze(-1)) + self.bias0
        if self.use_hist:
            t, r = self._decompose(hist)
            z = z + self.trend(t) + self.resid(r)
        if self.use_drivers:
            z = z + self.drv(X.reshape(X.shape[0], -1))
        return torch.sigmoid(z) if self.bounded else z


def run_linear(tr, te, data, hist, args, seed, fips, fold_id, use_hist,
               use_drivers, bounded=True, inner_units=None, inner_split_seed=None):
    y0, X, yt, m = data
    inner_units = fips if inner_units is None else inner_units
    inner_split_seed = seed if inner_split_seed is None else inner_split_seed
    torch.manual_seed(seed); np.random.seed(seed)
    mu = X[tr].reshape(-1, X.shape[-1]).mean(0)
    sd = X[tr].reshape(-1, X.shape[-1]).std(0) + 1e-6       # training folds only
    Xn = ((X - mu) / sd).astype(np.float32)
    # The state gets one scalar scale, not a per-lag one: the lookback is a single
    # series and standardising each lag against its own moments would distort the
    # temporal structure the decomposition is there to read. Training folds only,
    # like every other statistic here. Leaving the state unnormalised while the
    # drivers are normalised was measured to make the history-only arm diverge --
    # it needs large weights to reach a target of order 1e-2 from inputs of the
    # same order, and a handful of counties near y = 1 then dominate.
    y_mu = float(y0[tr].mean())
    y_sd = float(y0[tr].std()) + 1e-6
    y0n = ((y0 - y_mu) / y_sd).astype(np.float32)
    Hn = (((hist - y_mu) / y_sd).astype(np.float32) if hist is not None
          else np.zeros((len(y0), 1), np.float32))

    model = DecompLinear(Hn.shape[1], yt.shape[1], X.shape[-1], use_hist,
                         use_drivers, bounded=bounded)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    # Start from the constant predictor -- the training-fold mean -- rather than
    # from zero. The target's mean is near 1e-2 and its scale is what makes the
    # neural arms in this project fragile; the baseline gets the same courtesy.
    base = float(yt[tr][m[tr]].mean())
    with torch.no_grad():
        model.bias0.fill_(float(np.log(base / (1 - base))) if bounded else base)

    fi, vi = inner_split(inner_units[tr], seed=inner_split_seed, fold=fold_id)
    tr_arr = np.asarray(tr)
    fit, va = tr_arr[fi], tr_arr[vi]
    Y0, XX, YT, MM, HH = (torch.tensor(y0n), torch.tensor(Xn), torch.tensor(yt),
                          torch.tensor(m.astype(np.float32)), torch.tensor(Hn))

    def loss_on(ix):
        pred = model(Y0[ix], XX[ix], HH[ix])
        se = (pred - YT[ix]) ** 2 * MM[ix]
        return se.sum() / MM[ix].sum().clamp_min(1.0)

    best, best_state, bad = float("inf"), None, 0
    for ep in range(args.epochs):
        model.train()
        perm = np.random.permutation(fit)
        for s in range(0, len(perm), args.batch):
            b = torch.tensor(perm[s:s + args.batch], dtype=torch.long)
            opt.zero_grad(); loss_on(b).backward()
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
        pred = model(Y0[ti], XX[ti], HH[ti]).numpy()
    raw_lo, raw_hi = float(pred.min()), float(pred.max())
    pred = np.clip(pred, 0.0, 1.0)
    out = {"_test_pred": pred[:, [h - 1 for h in args.horizons]].astype(np.float32)}   # scored as-is
    for h in args.horizons:
        e = (pred[:, h - 1] - yt[te][:, h - 1])[m[te][:, h - 1]]
        out[f"rmse_h{h}"] = float(np.sqrt(np.mean(e ** 2))) if e.size else float("nan")
        out[f"n_h{h}"] = int(e.size)
    # Collapse screen. Against a target that is zero in roughly half its cells, a
    # model that has learned nothing but the mean posts a respectable RMSE. The
    # spread of its predictions is what separates the two, so it is recorded for
    # every arm rather than inspected when a number looks surprising.
    out["pred_sd"] = float(pred.std())
    out["pred_range"] = [float(pred.min()), float(pred.max())]
    out["raw_range"] = [raw_lo, raw_hi]        # before the [0,1] clip
    out["frac_clipped"] = float(np.mean((pred == 0.0) | (pred == 1.0)))
    out["frac_pred_const"] = float(np.mean(np.abs(pred - pred.mean()) < 1e-9))
    out["val_loss"] = best
    out["n_param"] = sum(p.numel() for p in model.parameters())
    return out


ARMS = [
    ("trees_matched",    "trees",  dict(use_hist=False)),
    ("trees_lookback",   "trees",  dict(use_hist=True)),
    # The three information variants share the bounded head so that the
    # decomposition across them is not also a decomposition across output layers.
    ("linear_matched",   "linear", dict(use_hist=False, use_drivers=True)),
    ("linear_lookback",  "linear", dict(use_hist=True,  use_drivers=True)),
    ("linear_histonly",  "linear", dict(use_hist=True,  use_drivers=False)),
    # The head question, asked once, on the matched information set.
    ("linear_unbounded", "linear", dict(use_hist=False, use_drivers=True, bounded=False)),
]
# Arms that read the pre-origin window see more than the dynamics do. Kept
# explicit so a summary table cannot quietly present them as like-for-like.
ADVANTAGED = {"trees_lookback", "linear_lookback", "linear_histonly"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=48)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 6, 24, 48])
    ap.add_argument("--stride", type=int, default=12)
    ap.add_argument("--lookback", type=int, default=24)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--model-seeds", dest="seeds", type=int, nargs="+")
    ap.add_argument("--outer-split-seed", type=int, default=0)
    ap.add_argument("--inner-split-seed", type=int, default=0)
    ap.add_argument("--split-unit", choices=["event", "county"], default="event")
    ap.add_argument("--clock", choices=sorted(schema.CLOCKS), default="utc_hour")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--trees", type=int, default=400)
    ap.add_argument("--tree-lr", type=float, default=0.05)
    ap.add_argument("--leaves", type=int, default=31)
    ap.add_argument("--tree-step", type=int, default=25,
                    help="boosting rounds between early-stopping evaluations")
    ap.add_argument("--tree-lib", choices=["sklearn", "lightgbm"], default="sklearn",
                    help="sklearn is the default for a measured reason; see the "
                         "module docstring before changing it")
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--panels", default=None,
                    help="panel set: omit for the manifest, 'auto' to pool what "
                         "is on disk (exploration only), or a path to a JSON file")
    ap.add_argument("--save-oof", action="store_true",
                    help="also write oof_<arm>.npz next to --out, in the audited out-of-fold layout")
    ap.add_argument("--out", default="results/exp07_learned_baselines.json")
    a = ap.parse_args()

    want, digest = panelset.resolve(INTERIM, a.panels)
    t_launch = time.time(); source_at_launch = panelset.source_version(ROOT)
    y0, X, yt, m, fips, panel, origin, t0h = load_pooled(
        a.horizon, a.stride, panels=want, with_time=True)
    X = add_context(X, y0, a.horizon, t0_hour=t0h, clock=a.clock)
    assign, split_map, split_digest, unit_ids = exp05.outer_assignment(fips, panel, a.split_unit, a.k, a.outer_split_seed)
    split_path = splits.save_split(split_map, a.split_unit, a.k, a.outer_split_seed, ROOT)
    print(f"split_unit {a.split_unit} · outer_split_seed {a.outer_split_seed} · digest {split_digest} · clock {a.clock}")
    hist = load_history(a.horizon, a.stride, a.lookback, panels=want)
    # Alignment is the whole risk in loading the pre-origin window separately, so
    # it is checked rather than trusted: the last hour of each lookback is the
    # origin state that `load_pooled` returned for the same row.
    assert len(hist) == len(y0), (len(hist), len(y0))
    assert np.allclose(hist[:, -1], y0), "history window is not aligned to the origins"

    print(f"pooled samples {len(y0):,} over {len(set(panel))} panels [{digest}], "
          f"{len(set(fips)):,} counties, {X.shape[-1]} channels, "
          f"lookback {a.lookback} h, horizon {a.horizon} h")
    print(f"observed targets: {m.mean()*100:.1f}%   mean y0 {y0.mean():.5f}   "
          f"target zeros {np.mean(yt[m] == 0)*100:.1f}%")

    arms = [x for x in ARMS if a.arms is None or x[0] in a.arms]
    rows = []
    oof = {}
    _, origin_id = np.unique(np.array([f"{p_}|{o_}" for p_, o_ in zip(panel, origin)]), return_inverse=True)
    for seed in a.seeds:                      # model seeds; the split does not move
        for f in range(a.k):
            te = np.where(assign == f)[0]; tr = np.where(assign != f)[0]
            for name, kind, kw in arms:
                kw = dict(kw, inner_units=unit_ids, inner_split_seed=a.inner_split_seed)
                t0 = time.time()
                if kind == "trees":
                    r = run_trees(tr, te, (y0, X, yt, m), hist, a, seed, fips, f, **kw)
                else:
                    r = run_linear(tr, te, (y0, X, yt, m), hist, a, seed, fips, f, **kw)
                wall = round(time.time() - t0, 1)
                exp05._stash(oof, name, seed, f, te, r.pop("_test_pred"), len(y0), a.seeds, len(a.horizons))
                rows.append({"arm": name, "kind": kind, "seed": seed, "fold": f,
                             "advantaged": name in ADVANTAGED,
                             "n_test": len(te), "wall_s": wall, **r})
                print(f"  seed {seed} fold {f} {name:<17} "
                      + " ".join(f"h{h}={r[f'rmse_h{h}']:.5f}" for h in a.horizons)
                      + f"  {wall}s", flush=True)

    out = ROOT / a.out; out.parent.mkdir(parents=True, exist_ok=True)
    cfg = dict(vars(a)); cfg["out"] = a.out
    cfg["panels"] = sorted(set(panel.tolist()))
    cfg["panel_digest"] = digest
    cfg["channels"] = panelset.channel_names(INTERIM)
    cfg["channel_digest"] = panelset.channel_digest(cfg["channels"])
    cfg["source"] = source_at_launch
    hp = {k: cfg[k] for k in ("epochs", "patience", "batch", "lr", "hidden", "cap_u", "cap_r", "horizon",
                              "stride", "k", "horizons", "lookback", "rounds") if k in cfg}
    cfg.update(schema.result_header(
        experiment_id=Path(a.out).stem, source=source_at_launch, panel_ids=cfg["panels"],
        panel_digest=digest, channel_names=schema.channel_list(cfg["channels"], a.clock),
        channel_digest=cfg["channel_digest"], clock=a.clock, split_unit=a.split_unit,
        outer_split_digest=split_digest, outer_split_seed=a.outer_split_seed,
        inner_split_seed=a.inner_split_seed, model_seeds=a.seeds, hyperparameters=hp))
    cfg["split_file"] = str(split_path.relative_to(ROOT))
    cfg["wall_time_s"] = round(time.time() - t_launch, 1)
    out.write_text(json.dumps({"config": cfg, "rows": rows}, indent=2))
    if a.save_oof:
        exp05._write_oof(oof, out.parent, yt, m, fips, panel, origin, origin_id, a, cfg)

    print(f"\n=== pooled over {a.k} folds x {len(a.seeds)} seeds ===")
    print(f"{'arm':<18}{'info':<12}" + "".join(f"{'RMSE h+'+str(h):>19}" for h in a.horizons))
    for name, _, _ in arms:
        g = [r for r in rows if r["arm"] == name]
        tag = "advantaged" if name in ADVANTAGED else "matched"
        line = f"{name:<18}{tag:<12}"
        for h in a.horizons:
            v = [r[f"rmse_h{h}"] for r in g if np.isfinite(r[f"rmse_h{h}"])]
            line += f"{np.mean(v):>13.5f}±{np.std(v):<5.5f}"
        print(line)
    print(f"\nwritten: {a.out}")


if __name__ == "__main__":
    main()
