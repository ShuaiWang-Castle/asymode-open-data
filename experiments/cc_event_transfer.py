"""Confirmatory event-transfer experiment: two_rate vs parameter-matched net_scaled.

Primary protocol is **leave one complete event out**: for each of the 11 events of
`g2-convective-11`, that event is absent from training and validation, one
validation event is named by a mapping fixed before any fit
(`configs/event_split_map_g2.json`, cyclic chronological), and the remaining nine
events are training data. The secondary control is county-held-out over the same
11 events with one fixed county->fold map that does not move with the model seed.

Every neural arm goes through the SAME `fit_arm` function: same optimiser, learning
rate, batch size, epoch cap, early-stopping rule, seeds, rows, loss, mask and
initialisation principle. Only the model factory differs. `cc_fairness_audit.py`
asserts that and fails if it is violated.

Normalisation is estimated on training-event rows only and applied unchanged to
validation and test. The observation mask excludes unobserved targets from the
loss, from early stopping and from every reported metric.

    python experiments/cc_event_transfer.py --split-unit event --arms two_rate net_scaled \
        --model-seeds 0 1 2 --out results/event_transfer_confirmatory_20260903/core_event.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from asymode import panels as panelset, splits, schema  # noqa: E402
from asymode.burden import BurdenConfig, RecoveryBurdenODE  # noqa: E402
from asymode.dynamics import InflowForm, TwoRateConfig, TwoRateODE, calibrate_init  # noqa: E402
import exp05_real_dynamics as exp05  # noqa: E402

INTERIM = ROOT / "data/interim"
SPLIT_MAP = ROOT / "configs/event_split_map_g2.json"

# One place where the shared budget lives, so no arm can be given a different one.
BUDGET = dict(epochs=60, patience=12, batch=512, lr=3e-3, cap_u=0.25, cap_r=0.25)
HORIZONS = (1, 6, 24, 48)


# --------------------------------------------------------------------------- splits
def build_event_split_map(events: list[str]) -> dict:
    """Cyclic chronological leave-one-event-out: test e_i, validate e_{i+1 mod n}."""
    ev = sorted(events)
    folds = []
    for i, e in enumerate(ev):
        v = ev[(i + 1) % len(ev)]
        folds.append({"fold": i, "test": [e], "validation": [v],
                      "train": [x for x in ev if x not in (e, v)]})
    return {"protocol": "leave-one-event-out",
            "rule": "events sorted chronologically; validation is the next event cyclically; "
                    "the remaining events train. Fixed before any fit; never chosen from outcomes.",
            "manifest": "g2-convective-11", "n_events": len(ev), "events": ev, "folds": folds}


def load_or_write_split_map(events: list[str]) -> dict:
    if SPLIT_MAP.exists():
        m = json.loads(SPLIT_MAP.read_text())
        if sorted(m["events"]) != sorted(events):
            raise SystemExit(f"{SPLIT_MAP} names different events; refusing to proceed")
        return m
    m = build_event_split_map(events)
    SPLIT_MAP.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_MAP.write_text(json.dumps(m, indent=1))
    print(f"wrote {SPLIT_MAP.relative_to(ROOT)} before any fit")
    return m


# --------------------------------------------------------------------------- models
def make_model(arm: str, d_in: int, u0: float, r0: float):
    if arm == "two_rate":
        return TwoRateODE(TwoRateConfig(d_in=d_in, cap_u=BUDGET["cap_u"], cap_r=BUDGET["cap_r"],
                                        hidden_u=32, hidden_r=32, inflow=InflowForm.SUSCEPTIBLE,
                                        u_init=u0, r_init=r0))
    if arm == "net_scaled":
        return TwoRateODE(TwoRateConfig(d_in=d_in, cap_u=BUDGET["cap_u"], cap_r=BUDGET["cap_r"],
                                        hidden_u=48, hidden_r=48, inflow=InflowForm.NET_SCALED,
                                        u_init=u0 - r0, r_init=None))
    if arm == "recovery_burden":
        return RecoveryBurdenODE(BurdenConfig(d_in=d_in, cap_u=BUDGET["cap_u"], cap_r=BUDGET["cap_r"],
                                              hidden_u=32, hidden_r=32, u_init=u0, r_init=r0))
    raise ValueError(arm)


def fit_arm(arm, tr, va, te, data, hist, seed, log):
    """Train one arm. Identical code path for every arm; only `make_model` differs."""
    y0, X, yt, m = data
    torch.manual_seed(seed)
    np.random.seed(seed)
    mu = X[tr].reshape(-1, X.shape[-1]).mean(0)
    sd = X[tr].reshape(-1, X.shape[-1]).std(0) + 1e-6          # training events only
    Xn = ((X - mu) / sd).astype(np.float32)

    ytr = np.concatenate([y0[tr][:, None], yt[tr]], axis=1)
    mtr = np.concatenate([np.ones((len(tr), 1), bool), m[tr]], axis=1)
    u0, r0 = calibrate_init(ytr, mtr, InflowForm.SUSCEPTIBLE)   # same rule for every arm
    model = make_model(arm, Xn.shape[-1], u0, r0)
    opt = torch.optim.Adam(model.parameters(), lr=BUDGET["lr"])

    T = torch.tensor
    Y0, XX, YT, MM = T(y0), T(Xn), T(yt), T(m.astype(np.float32))
    H = T(hist.astype(np.float32)) if hist is not None else None
    needs_b = isinstance(model, RecoveryBurdenODE)

    def predict(ix):
        if needs_b:
            return model(Y0[ix], XX[ix], model.burden_from_history(H[ix]))
        return model(Y0[ix], XX[ix])

    def loss_on(ix):
        se = (predict(ix) - YT[ix]) ** 2 * MM[ix]
        return se.sum() / MM[ix].sum().clamp_min(1.0)

    best, best_state, bad, ran, curve = float("inf"), None, 0, 0, []
    va_t = torch.tensor(va, dtype=torch.long)
    t0 = time.time()
    for ep in range(BUDGET["epochs"]):
        ran = ep + 1
        model.train()
        perm = np.random.permutation(tr)
        for s in range(0, len(perm), BUDGET["batch"]):
            b = torch.tensor(perm[s:s + BUDGET["batch"]], dtype=torch.long)
            opt.zero_grad()
            l = loss_on(b)
            l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(loss_on(va_t))
        curve.append(vl)
        if vl < best - 1e-10:
            best, bad, best_ep = vl, 0, ran
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= BUDGET["patience"]:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = predict(torch.tensor(te, dtype=torch.long)).numpy()

    rec = {"arm": arm, "seed": seed, "parameter_count": sum(p.numel() for p in model.parameters()),
           "best_epoch": int(best_ep), "stopped_at_epoch": int(ran),
           "hit_epoch_cap": bool(ran >= BUDGET["epochs"]),
           "training_time_s": round(time.time() - t0, 1), "final_validation": best,
           "u_init": float(u0), "r_init": float(r0),
           "val_curve_tail": [round(v, 10) for v in curve[-10:]],
           "n_train": len(tr), "n_val": len(va), "n_test": len(te)}
    if needs_b:
        rec["rho"] = float(model.rho.detach())
    for h in HORIZONS:
        e = (pred[:, h - 1] - yt[te][:, h - 1])[m[te][:, h - 1]]
        rec[f"mse_h{h}"] = float(np.mean(e ** 2)) if e.size else float("nan")
        rec[f"rmse_h{h}"] = float(np.sqrt(rec[f"mse_h{h}"])) if e.size else float("nan")
        rec[f"n_h{h}"] = int(e.size)
    rec["pred_sd"] = float(pred.std())
    rec["frac_pred_zero"] = float((pred <= 0).mean())
    log(f"    {arm:<16} seed {seed} ep {ran:>3} (best {best_ep:>3}) "
        + " ".join(f"h{h}={rec[f'rmse_h{h}']:.5f}" for h in HORIZONS)
        + f"  {rec['training_time_s']}s")
    return rec, pred[:, [h - 1 for h in HORIZONS]].astype(np.float32)


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", default="configs/panel_manifest_g2-convective-11.json")
    ap.add_argument("--split-unit", choices=["event", "county"], default="event")
    ap.add_argument("--arms", nargs="+", default=["two_rate", "net_scaled"])
    ap.add_argument("--model-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--events", nargs="*", default=None, help="restrict the test events (screen/smoke)")
    ap.add_argument("--k-county", type=int, default=5)
    ap.add_argument("--outer-split-seed", type=int, default=0)
    ap.add_argument("--inner-split-seed", type=int, default=0)
    ap.add_argument("--lookback", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=None, help="override the shared cap (convergence probe only)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pred-dir", default=None)
    a = ap.parse_args()
    if a.epochs:
        BUDGET["epochs"] = a.epochs
        BUDGET["patience"] = max(BUDGET["patience"], a.epochs // 5)

    t_launch = time.time()
    source = panelset.source_version(ROOT)
    want, panel_digest = panelset.resolve(INTERIM, str(ROOT / a.panels))
    y0, X, yt, m, fips, panel, origin, t0h = exp05.load_pooled(48, 12, panels=want, with_time=True)
    X = exp05.add_context(X, y0, 48, t0_hour=t0h, clock="utc_hour")
    hist = exp05.load_history(48, 12, a.lookback, panels=want)
    assert len(hist) == len(y0) and np.allclose(hist[:, -1], y0), "history window not aligned to origins"
    chan = schema.channel_list(panelset.channel_names(INTERIM), "utc_hour")
    assert len(chan) == X.shape[-1] == 14, (len(chan), X.shape[-1])

    events = sorted(set(panel.tolist()))
    lines: list[str] = []

    def log(s: str) -> None:
        print(s, flush=True)
        lines.append(s)

    if a.split_unit == "event":
        smap = load_or_write_split_map(events)
        folds = [f for f in smap["folds"] if a.events is None or f["test"][0] in a.events]
        split_digest = splits.split_digest({f["test"][0]: f["fold"] for f in smap["folds"]})
    else:
        cmap = splits.county_folds(sorted(set(fips.tolist())), k=a.k_county,
                                   outer_split_seed=a.outer_split_seed)
        split_digest = splits.split_digest(cmap)
        assign = splits.assign_rows(fips, cmap)
        folds = [{"fold": f, "test": None, "validation": None, "train": None} for f in range(a.k_county)]
    log(f"protocol {a.split_unit} · {len(folds)} folds · split digest {split_digest} · "
        f"{len(y0):,} samples · {len(events)} events · panels {panel_digest}")

    rows, preds = [], {}
    n_h = len(HORIZONS)
    for f in folds:
        if a.split_unit == "event":
            te = np.where(panel == f["test"][0])[0]
            va = np.where(np.isin(panel, f["validation"]))[0]
            tr = np.where(np.isin(panel, f["train"]))[0]
            tag = f["test"][0]
            assert not (set(panel[te]) & set(panel[tr])) and not (set(panel[te]) & set(panel[va])), "event leak"
        else:
            te = np.where(assign == f["fold"])[0]
            pool = np.where(assign != f["fold"])[0]
            fi, vi = exp05.inner_split(fips[pool], seed=a.inner_split_seed, fold=f["fold"])
            tr, va = pool[fi], pool[vi]
            tag = f"county-fold-{f['fold']}"
            assert not (set(fips[te]) & set(fips[tr])), "county leak"
        log(f"  fold {f['fold']} test={tag} n_tr={len(tr):,} n_va={len(va):,} n_te={len(te):,}")
        for arm in a.arms:
            for seed in a.model_seeds:
                rec, p = fit_arm(arm, tr, va, te, (y0, X, yt, m), hist, seed, log)
                rec.update(fold=f["fold"], test_event=tag,
                           validation_event=(f["validation"][0] if a.split_unit == "event" else "county-inner"),
                           split_unit=a.split_unit)
                rows.append(rec)
                key = (arm, seed)
                if key not in preds:
                    preds[key] = np.full((len(y0), n_h), np.nan, np.float32)
                preds[key][te] = p

    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = dict(vars(a))
    cfg.update(schema.result_header(
        experiment_id=Path(a.out).stem, source=source, panel_ids=sorted(want), panel_digest=panel_digest,
        channel_names=chan, channel_digest=panelset.channel_digest(panelset.channel_names(INTERIM)),
        clock="utc_hour", split_unit=a.split_unit, outer_split_digest=split_digest,
        outer_split_seed=a.outer_split_seed, inner_split_seed=a.inner_split_seed,
        model_seeds=a.model_seeds, hyperparameters=dict(BUDGET, horizons=list(HORIZONS), lookback=a.lookback)))
    cfg["wall_time_s"] = round(time.time() - t_launch, 1)
    fits = [r for r in rows if "stopped_at_epoch" in r]
    cfg["convergence"] = {
        "n_fits": len(fits), "n_at_epoch_cap": int(sum(r["hit_epoch_cap"] for r in fits)),
        "frac_at_cap": float(np.mean([r["hit_epoch_cap"] for r in fits])) if fits else None,
        "max_tail_improvement": max(
            ((r["val_curve_tail"][0] - min(r["val_curve_tail"])) / max(r["val_curve_tail"][0], 1e-12)
             for r in fits if len(r["val_curve_tail"]) >= 2), default=0.0)}
    out.write_text(json.dumps({"config": cfg, "rows": rows}, indent=1))

    pd_dir = ROOT / (a.pred_dir or (out.parent / "predictions"))
    pd_dir.mkdir(parents=True, exist_ok=True)
    for (arm, seed), p in preds.items():
        np.savez_compressed(pd_dir / f"pred_{a.split_unit}_{arm}_seed{seed}.npz",
                            pred=p, y=yt[:, [h - 1 for h in HORIZONS]].astype(np.float32),
                            mask=m[:, [h - 1 for h in HORIZONS]], panel=panel.astype(str),
                            fips=fips.astype(str), origin_step=origin.astype(np.int32),
                            horizons=np.array(HORIZONS, np.int32), split_unit=str(a.split_unit),
                            outer_split_digest=str(split_digest), panel_digest=str(panel_digest))
    (out.parent / "logs").mkdir(parents=True, exist_ok=True)
    (out.parent / "logs" / f"{Path(a.out).stem}.log").write_text("\n".join(lines) + "\n")
    log(f"\nwritten: {a.out}  ({len(rows)} fits, {cfg['convergence']['n_at_epoch_cap']} at the epoch cap)")


if __name__ == "__main__":
    main()
