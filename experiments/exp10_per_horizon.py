"""EXP10 -- how much of the tree advantage is the single-rollout constraint?

Result-driven, not pre-registered. It exists because EXP07 found that gradient
boosted trees beat the two-rate dynamics at every horizon past h+1, and one
asymmetry in that comparison had not been quantified: the trees fit **one model
per scored horizon**, each free to optimise for that horizon alone, while the
dynamics fit **one model that must roll a single trajectory through all of them**.
That is a real difference in effective capacity and in what the model is asked to
do, and it cannot explain a 7.9% gap by itself -- but unquantified is unquantified.

The control removes the asymmetry from the dynamics' side: the same two-rate
model, same inputs, same folds, same seeds, same recipe, trained separately for
each scored horizon with the loss taken **only at that horizon**. The rollout is
still a rollout -- the state is still integrated step by step -- but the fit is no
longer asked to be simultaneously right at h+1 and h+48.

Reading fixed before the run, so that neither outcome can be narrated afterwards:

  * If the gap to the trees mostly closes, the dynamics were paying for the
    single-rollout constraint, not for their structure. That is a statement about
    what the model is asked to produce, and it makes the trajectory itself the
    thing the paper has to argue is worth having.
  * If the gap does not close, the constraint is not the explanation and the
    structure is simply worse than a flexible regressor at covariate-driven
    horizons. The paper says so.

Neither reading rescues a claim that the two-rate form is more accurate pointwise
at long horizons. That claim is already dead; this only establishes why.
"""
from __future__ import annotations

import argparse, importlib.util, json, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from asymode.dynamics import (TwoRateODE, TwoRateConfig, InflowForm,   # noqa: E402
                              calibrate_init)
from asymode.evalproto import make_folds, inner_split                  # noqa: E402
from asymode import panels as panelset                                 # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "exp05", Path(__file__).resolve().parent / "exp05_real_dynamics.py")
exp05 = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = exp05
_spec.loader.exec_module(exp05)
load_pooled, add_context = exp05.load_pooled, exp05.add_context

INTERIM = ROOT / "data" / "interim"


def fit_one(target_h, tr, te, data, args, seed, fips, fold_id):
    """Train the two-rate model to be right at one horizon only.

    The rollout still integrates the state step by step to reach `target_h`; what
    changes is that the loss reads only the final step. Every other element of the
    recipe -- normalisation from training folds, calibrated initial flows,
    county-held-out early stopping, optimiser, capacity -- is the one EXP08 used,
    so the difference between this and EXP08's control is the objective and
    nothing else.
    """
    y0, X, yt, m = data
    torch.manual_seed(seed); np.random.seed(seed)
    mu = X[tr].reshape(-1, X.shape[-1]).mean(0)
    sd = X[tr].reshape(-1, X.shape[-1]).std(0) + 1e-6
    Xn = ((X - mu) / sd).astype(np.float32)

    ytr = np.concatenate([y0[tr][:, None], yt[tr]], axis=1)
    mtr = np.concatenate([np.ones((len(tr), 1), bool), m[tr]], axis=1)
    u0, r0 = calibrate_init(ytr, mtr, InflowForm.SUSCEPTIBLE)
    cfg = TwoRateConfig(d_in=Xn.shape[-1], cap_u=args.cap_u, cap_r=args.cap_r,
                        hidden_u=args.hidden, hidden_r=args.hidden,
                        inflow=InflowForm.SUSCEPTIBLE, u_init=u0, r_init=r0)
    model = TwoRateODE(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    fi, vi = inner_split(fips[tr], seed=seed, fold=fold_id)
    tr_arr = np.asarray(tr)
    fit, va = tr_arr[fi], tr_arr[vi]
    j = target_h - 1
    Y0 = torch.tensor(y0)
    XX = torch.tensor(Xn[:, :target_h])          # only the steps it has to walk
    YT = torch.tensor(yt[:, j])
    MM = torch.tensor(m[:, j].astype(np.float32))

    def loss_on(ix):
        pred = model(Y0[ix], XX[ix])[:, -1]
        se = (pred - YT[ix]) ** 2 * MM[ix]
        return se.sum() / MM[ix].sum().clamp_min(1.0)

    best, best_state, bad, ran = float("inf"), None, 0, 0
    for ep in range(args.epochs):
        ran = ep + 1
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
        pred = model(Y0[ti], XX[ti])[:, -1].numpy()
    e = (pred - yt[te][:, j])[m[te][:, j]]
    return {"horizon": target_h,
            f"rmse_h{target_h}": float(np.sqrt(np.mean(e ** 2))) if e.size else float("nan"),
            f"n_h{target_h}": int(e.size),
            "val_loss": best, "epochs_run": ran,
            "hit_epoch_cap": bool(ran >= args.epochs),
            "pred_sd": float(pred.std()), "pred_max": float(pred.max()),
            "frac_pred_below_1e6": float((pred < 1e-6).mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=48)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 6, 24, 48])
    ap.add_argument("--stride", type=int, default=12)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--cap-u", type=float, default=0.25)
    ap.add_argument("--cap-r", type=float, default=0.25)
    ap.add_argument("--panels", default=None)
    ap.add_argument("--out", default="results/exp10_per_horizon.json")
    a = ap.parse_args()

    want, digest = panelset.resolve(INTERIM, a.panels)
    y0, X, yt, m, fips, panel, origin = load_pooled(
        a.horizon, a.stride, panels=want)
    X = add_context(X, y0, a.horizon)
    names = panelset.channel_names(INTERIM)
    print(f"pooled {len(y0):,} samples over {len(set(panel))} panels [{digest}], "
          f"{len(set(fips)):,} counties, {X.shape[-1]} channels")

    rows = []
    for seed in a.seeds:
        fold = make_folds(sorted(set(fips)), k=a.k, seed=seed)
        fmap = {f: fo for f, fo in zip(sorted(set(fips)), fold)}
        assign = np.array([fmap[f] for f in fips])
        for f in range(a.k):
            te = np.where(assign == f)[0]; tr = np.where(assign != f)[0]
            for h in a.horizons:
                t0 = time.time()
                r = fit_one(h, tr, te, (y0, X, yt, m), a, seed, fips, f)
                rows.append({"arm": "per_horizon", "seed": seed, "fold": f,
                             "n_test": len(te),
                             "wall_s": round(time.time() - t0, 1), **r})
                print(f"  seed {seed} fold {f} h+{h:<3} "
                      f"rmse={r[f'rmse_h{h}']:.5f}  {r['epochs_run']}ep  "
                      f"{rows[-1]['wall_s']}s", flush=True)

    out = ROOT / a.out; out.parent.mkdir(parents=True, exist_ok=True)
    cfg = dict(vars(a)); cfg["out"] = a.out
    cfg["panels"] = sorted(set(panel.tolist()))
    cfg["panel_digest"] = digest
    cfg["channels"] = names
    cfg["channel_digest"] = panelset.channel_digest(names)
    cfg["source"] = panelset.source_version(ROOT)
    out.write_text(json.dumps({"config": cfg, "rows": rows}, indent=2))
    print(f"\nwritten: {a.out}")


if __name__ == "__main__":
    main()
