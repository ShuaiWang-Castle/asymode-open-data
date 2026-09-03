"""Section 5.3: histogram gradient boosting on the same information, under the same
leave-one-event-out folds. One direct model per horizon.

This is a strong statistical reference for long-horizon accuracy, not the comparator
that establishes the structural two-rate claim. The tree cap is lifted to 2,000
rounds with the same held-out early stopping, because the archived run recorded the
trees as cap-limited and a reference that is still improving understates the gap.

Damped persistence is computed on the same test rows at no cost.

    python experiments/cc_hgb_reference.py --model-seeds 0 1 2 \
        --out results/event_transfer_confirmatory_20260903/hgb_event.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from asymode import panels as panelset, splits, schema  # noqa: E402
import exp05_real_dynamics as exp05  # noqa: E402
import exp07_learned_baselines as exp07  # noqa: E402
import cc_event_transfer as CC  # noqa: E402

INTERIM = ROOT / "data/interim"
HORIZONS = (1, 6, 24, 48)


def damped_persistence(y0, yt, m, tr, te, horizons):
    """One fitted decay constant, estimated on training rows only."""
    ytr = np.concatenate([y0[tr][:, None], yt[tr]], axis=1)
    mtr = np.concatenate([np.ones((len(tr), 1), bool), m[tr]], axis=1)
    mm = mtr[:, :-1] & mtr[:, 1:]
    num = float((ytr[:, :-1][mm] * ytr[:, 1:][mm]).sum())
    den = float((ytr[:, :-1][mm] ** 2).sum()) + 1e-12
    rho = float(np.clip(num / den, 0.0, 1.0))
    out = {}
    pred = np.stack([y0[te] * rho ** h for h in range(1, yt.shape[1] + 1)], axis=1)
    for h in horizons:
        e = (pred[:, h - 1] - yt[te][:, h - 1])[m[te][:, h - 1]]
        out[f"mse_h{h}"] = float(np.mean(e ** 2))
        out[f"rmse_h{h}"] = float(np.sqrt(out[f"mse_h{h}"]))
        out[f"n_h{h}"] = int(e.size)
    out["rho"] = rho
    return out, pred[:, [h - 1 for h in horizons]].astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", default="configs/panel_manifest_g2-convective-11.json")
    ap.add_argument("--model-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--trees", type=int, default=2000)
    ap.add_argument("--lookback", type=int, default=24)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    source = panelset.source_version(ROOT)
    want, panel_digest = panelset.resolve(INTERIM, str(ROOT / a.panels))
    y0, X, yt, m, fips, panel, origin, t0h = exp05.load_pooled(48, 12, panels=want, with_time=True)
    X = exp05.add_context(X, y0, 48, t0_hour=t0h, clock="utc_hour")
    hist = exp05.load_history(48, 12, a.lookback, panels=want)
    smap = json.loads((ROOT / "configs/event_split_map_g2.json").read_text())
    split_digest = splits.split_digest({f["test"][0]: f["fold"] for f in smap["folds"]})

    # every field exp07._fit_trees reads; `patience` counts early-stopping evaluations,
    # not epochs, so 8 evaluations at a 25-round step is 200 rounds without improvement
    targs = types.SimpleNamespace(horizons=list(HORIZONS), trees=a.trees, tree_lr=0.05,
                                  leaves=31, tree_step=25, tree_lib="sklearn", patience=8)
    rows, preds = [], {}
    n_h = len(HORIZONS)
    for f in smap["folds"]:
        e = f["test"][0]
        te = np.where(panel == e)[0]
        va = np.where(np.isin(panel, f["validation"]))[0]
        tr = np.where(np.isin(panel, f["train"]))[0]
        assert not (set(panel[te]) & set(panel[tr])), "event leak"
        # the trees early-stop on the validation EVENT, exactly as the neural arms do
        pool = np.concatenate([tr, va])
        inner = np.concatenate([np.zeros(len(tr), bool), np.ones(len(va), bool)])
        for seed in a.model_seeds:
            t1 = time.time()
            out = {}
            pred = np.zeros((len(te), n_h), np.float32)
            for hi, h in enumerate(HORIZONS):
                f_tr = pool[inner == 0]
                f_tr = f_tr[m[f_tr, h - 1]]
                f_va = pool[inner == 1]
                f_va = f_va[m[f_va, h - 1]]
                f_te = te[m[te, h - 1]]
                Xtr = exp07.tree_features(y0[f_tr], X[f_tr], None, h, False)
                Xva = exp07.tree_features(y0[f_va], X[f_va], None, h, False)
                Xte = exp07.tree_features(y0[f_te], X[f_te], None, h, False)
                mdl, n_iter, val = exp07._fit_trees(targs, seed, Xtr, yt[f_tr, h - 1], Xva, yt[f_va, h - 1])
                p = mdl.predict(Xte)
                err = p - yt[f_te, h - 1]
                out[f"mse_h{h}"] = float(np.mean(err ** 2))
                out[f"rmse_h{h}"] = float(np.sqrt(out[f"mse_h{h}"]))
                out[f"n_h{h}"] = int(err.size)
                out[f"n_iter_h{h}"] = int(n_iter)
                out[f"hit_cap_h{h}"] = bool(n_iter >= a.trees)
                idx = np.searchsorted(te, f_te)
                pred[idx, hi] = p.astype(np.float32)
            out.update(arm="hgb_same_information", seed=seed, fold=f["fold"], test_event=e,
                       validation_event=f["validation"][0], split_unit="event",
                       training_time_s=round(time.time() - t1, 1))
            rows.append(out)
            print(f"  {e} seed {seed} " + " ".join(f"h{h}={out[f'rmse_h{h}']:.5f}" for h in HORIZONS)
                  + f"  iters {[out[f'n_iter_h{h}'] for h in HORIZONS]}  {out['training_time_s']}s", flush=True)
            preds.setdefault(("hgb_same_information", seed), np.full((len(y0), n_h), np.nan, np.float32))[te] = pred
        dp, dpred = damped_persistence(y0, yt, m, tr, te, HORIZONS)
        dp.update(arm="damped_persistence", seed=0, fold=f["fold"], test_event=e,
                  validation_event=f["validation"][0], split_unit="event", training_time_s=0.0)
        rows.append(dp)
        preds.setdefault(("damped_persistence", 0), np.full((len(y0), n_h), np.nan, np.float32))[te] = dpred
        print(f"  {e} damped_persistence rho={dp['rho']:.4f} "
              + " ".join(f"h{h}={dp[f'rmse_h{h}']:.5f}" for h in HORIZONS), flush=True)

    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = dict(vars(a))
    cfg.update(schema.result_header(
        experiment_id=Path(a.out).stem, source=source, panel_ids=sorted(want), panel_digest=panel_digest,
        channel_names=schema.channel_list(panelset.channel_names(INTERIM), "utc_hour"),
        channel_digest=panelset.channel_digest(panelset.channel_names(INTERIM)), clock="utc_hour",
        split_unit="event", outer_split_digest=split_digest, outer_split_seed=0, inner_split_seed=0,
        model_seeds=a.model_seeds, hyperparameters={"trees": a.trees, "tree_lr": 0.05, "leaves": 31,
                                                    "horizons": list(HORIZONS)}))
    cfg["wall_time_s"] = round(time.time() - t0, 1)
    out.write_text(json.dumps({"config": cfg, "rows": rows}, indent=1))
    pdir = out.parent / "predictions"
    pdir.mkdir(parents=True, exist_ok=True)
    for (arm, seed), p in preds.items():
        np.savez_compressed(pdir / f"pred_event_{arm}_seed{seed}.npz", pred=p,
                            y=yt[:, [h - 1 for h in HORIZONS]].astype(np.float32),
                            mask=m[:, [h - 1 for h in HORIZONS]], panel=panel.astype(str),
                            fips=fips.astype(str), horizons=np.array(HORIZONS, np.int32),
                            split_unit="event", outer_split_digest=str(split_digest),
                            panel_digest=str(panel_digest))
    print(f"\nwritten: {a.out}")


if __name__ == "__main__":
    main()
