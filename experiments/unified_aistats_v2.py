#!/usr/bin/env python3
"""Unified AISTATS v2 neural experiment.

This is the only neural experiment authorized by docs/UNIFIED_AISTATS_V2_PROTOCOL.md.
It uses all eleven g2 events, a 24-hour forecast horizon, two outcome-blind
validation events per outer fold, event-balanced optimization, and a fixed
half-rollout/half-teacher-forced training objective.

Primary comparison: two_rate_v2 versus parameter-matched net_scaled_v2.
Single ablation: the identical two-rate network under rollout-only training.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from asymode import panels as panelset, schema, splits  # noqa: E402
from asymode.dynamics import InflowForm, TwoRateConfig, TwoRateODE, calibrate_init  # noqa: E402
import exp05_real_dynamics as exp05  # noqa: E402

INTERIM = ROOT / "data/interim"
MANIFEST = ROOT / "configs/panel_manifest_g2-convective-11.json"
SPLIT_MAP = ROOT / "configs/event_split_map_g2_two_validation.json"
HORIZON = 24
PRIMARY_HORIZONS = (6, 24)
REPORT_HORIZONS = (1, 6, 24)
BUDGET = {
    "epochs": 60,
    "patience": 12,
    "batch": 512,
    "lr": 3e-3,
    "cap_u": 0.25,
    "cap_r": 0.25,
    "rollout_weight": 0.5,
    "one_step_weight": 0.5,
}


def _sha12(obj: object) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()[:12]


def load_split_map(events: list[str]) -> dict:
    smap = json.loads(SPLIT_MAP.read_text())
    if sorted(smap["events"]) != sorted(events):
        raise RuntimeError("two-validation split map does not match the manifest events")
    folds = []
    for f in smap["folds"]:
        test = list(f["test"])
        val = list(f["validation"])
        if len(test) != 1 or len(val) != 2:
            raise RuntimeError(f"invalid fold: {f}")
        if set(test) & set(val):
            raise RuntimeError(f"test/validation overlap: {f}")
        train = [e for e in events if e not in set(test + val)]
        if len(train) != 8:
            raise RuntimeError(f"expected eight training events: {f}")
        folds.append({"fold": int(f["fold"]), "test": test, "validation": val, "train": train})
    return {**smap, "folds": folds, "digest": _sha12(folds)}


def make_model(arm: str, d_in: int, u0: float, r0: float) -> TwoRateODE:
    if arm in {"two_rate_v2", "two_rate_rollout_only"}:
        cfg = TwoRateConfig(
            d_in=d_in,
            cap_u=BUDGET["cap_u"],
            cap_r=BUDGET["cap_r"],
            hidden_u=32,
            hidden_r=32,
            inflow=InflowForm.SUSCEPTIBLE,
            u_init=u0,
            r_init=r0,
            tags={"protocol": "unified-aistats-v2", "arm": arm},
        )
        return TwoRateODE(cfg)
    if arm == "net_scaled_v2":
        cfg = TwoRateConfig(
            d_in=d_in,
            cap_u=BUDGET["cap_u"],
            cap_r=BUDGET["cap_r"],
            hidden_u=48,
            hidden_r=48,
            inflow=InflowForm.NET_SCALED,
            u_init=u0 - r0,
            r_init=None,
            tags={"protocol": "unified-aistats-v2", "arm": arm},
        )
        return TwoRateODE(cfg)
    raise ValueError(f"unknown arm {arm}")


def teacher_forced_prediction(
    model: TwoRateODE,
    y0: torch.Tensor,
    drivers: torch.Tensor,
    y_true: torch.Tensor,
) -> torch.Tensor:
    """One-step states using the observed current state rather than a rollout state."""
    current = torch.cat([y0[:, None], y_true[:, :-1]], dim=1)
    out = []
    lo, hi = model.cfg.clip_state
    for t in range(drivers.shape[1]):
        y = current[:, t]
        u, r = model.rates(drivers[:, t], y)
        if model.is_net:
            if model.cfg.inflow is not InflowForm.NET_SCALED:
                raise RuntimeError("v2 supports only the scaled signed-rate comparator")
            delta = torch.where(u > 0, u * (1.0 - y), u * y)
            nxt = torch.clamp(y + delta, lo, hi)
        else:
            nxt = torch.clamp(y + u * (1.0 - y) - r * y, lo, hi)
        out.append(nxt)
    return torch.stack(out, dim=1)


def step_mask(mask: torch.Tensor) -> torch.Tensor:
    """A transition is scored only when both current and next states are observed."""
    current_observed = torch.cat([torch.ones_like(mask[:, :1]), mask[:, :-1]], dim=1)
    return mask * current_observed


def chunks(ix: np.ndarray, size: int) -> Iterable[np.ndarray]:
    for start in range(0, len(ix), size):
        yield ix[start : start + size]


def event_objective(
    model: TwoRateODE,
    ix: np.ndarray,
    tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    rollout_weight: float,
    one_step_weight: float,
    backward: bool,
) -> tuple[float, float, float]:
    """Exact masked event loss, evaluated in chunks.

    Denominators are event-level constants. During training, backward is called
    once per chunk and the accumulated gradient equals the full event objective.
    """
    Y0, XX, YT, MM = tensors
    den_roll = float(MM[ix].sum().item())
    SM_all = step_mask(MM[ix])
    den_step = float(SM_all.sum().item())
    if den_roll <= 0 or (one_step_weight > 0 and den_step <= 0):
        raise RuntimeError("event has no observed cells for the requested objective")

    total_roll = 0.0
    total_step = 0.0
    for b_np in chunks(ix, BUDGET["batch"]):
        b = torch.tensor(b_np, dtype=torch.long)
        pred = model(Y0[b], XX[b])
        roll_num = (((pred - YT[b]) ** 2) * MM[b]).sum()
        loss = rollout_weight * roll_num / den_roll
        total_roll += float(roll_num.detach())
        if one_step_weight > 0:
            tf = teacher_forced_prediction(model, Y0[b], XX[b], YT[b])
            sm = step_mask(MM[b])
            step_num = (((tf - YT[b]) ** 2) * sm).sum()
            loss = loss + one_step_weight * step_num / den_step
            total_step += float(step_num.detach())
        if backward:
            loss.backward()
    roll = total_roll / den_roll
    one = total_step / den_step if one_step_weight > 0 else float("nan")
    combined = rollout_weight * roll + (one_step_weight * one if one_step_weight > 0 else 0.0)
    return combined, roll, one


def validation_score(
    model: TwoRateODE,
    validation_events: list[str],
    event_rows: dict[str, np.ndarray],
    tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[float, dict[str, dict[str, float]]]:
    """Equal-event predictive checkpoint criterion fixed in the protocol."""
    Y0, XX, YT, MM = tensors
    detail: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        for event in validation_events:
            ix = event_rows[event]
            pred_parts = []
            for b_np in chunks(ix, BUDGET["batch"]):
                b = torch.tensor(b_np, dtype=torch.long)
                pred_parts.append(model(Y0[b], XX[b]))
            pred = torch.cat(pred_parts, dim=0)
            yy = YT[ix]
            mm = MM[ix]
            path_num = float((((pred - yy) ** 2) * mm).sum())
            path_den = float(mm.sum())
            path = path_num / max(path_den, 1.0)
            endpoints = {}
            for h in PRIMARY_HORIZONS:
                mh = mm[:, h - 1]
                num = float((((pred[:, h - 1] - yy[:, h - 1]) ** 2) * mh).sum())
                den = float(mh.sum())
                endpoints[h] = num / max(den, 1.0)
            score = 0.5 * path + 0.25 * endpoints[6] + 0.25 * endpoints[24]
            detail[event] = {
                "score": score,
                "path24_mse": path,
                "mse_h6": endpoints[6],
                "mse_h24": endpoints[24],
            }
    return float(np.mean([detail[e]["score"] for e in validation_events])), detail


def test_metrics(
    model: TwoRateODE,
    ix: np.ndarray,
    tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[dict[str, float], np.ndarray]:
    Y0, XX, YT, MM = tensors
    with torch.no_grad():
        parts = []
        for b_np in chunks(ix, BUDGET["batch"]):
            b = torch.tensor(b_np, dtype=torch.long)
            parts.append(model(Y0[b], XX[b]))
        pred = torch.cat(parts, dim=0)
    yy = YT[ix]
    mm = MM[ix]
    out: dict[str, float] = {}
    path_num = float((((pred - yy) ** 2) * mm).sum())
    path_den = float(mm.sum())
    out["mse_path24"] = path_num / max(path_den, 1.0)
    out["rmse_path24"] = math.sqrt(out["mse_path24"])
    for h in REPORT_HORIZONS:
        mh = mm[:, h - 1]
        num = float((((pred[:, h - 1] - yy[:, h - 1]) ** 2) * mh).sum())
        den = float(mh.sum())
        out[f"mse_h{h}"] = num / max(den, 1.0)
        out[f"rmse_h{h}"] = math.sqrt(out[f"mse_h{h}"])
        out[f"n_h{h}"] = int(den)
    out["pred_sd"] = float(pred.std())
    out["frac_pred_zero"] = float((pred <= 0).float().mean())
    return out, pred.cpu().numpy().astype(np.float32)


def fit_one(
    arm: str,
    seed: int,
    fold: dict,
    event_rows: dict[str, np.ndarray],
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    log,
) -> tuple[dict, np.ndarray]:
    y0, X, yt, mask = arrays
    tr = np.concatenate([event_rows[e] for e in fold["train"]])
    te = event_rows[fold["test"][0]]

    torch.manual_seed(seed)
    np.random.seed(seed)
    mu = X[tr].reshape(-1, X.shape[-1]).mean(axis=0)
    sd = X[tr].reshape(-1, X.shape[-1]).std(axis=0) + 1e-6
    Xn = ((X - mu) / sd).astype(np.float32)

    ytr = np.concatenate([y0[tr, None], yt[tr]], axis=1)
    mtr = np.concatenate([np.ones((len(tr), 1), dtype=bool), mask[tr]], axis=1)
    u0, r0 = calibrate_init(ytr, mtr, InflowForm.SUSCEPTIBLE)
    model = make_model(arm, Xn.shape[-1], u0, r0)
    opt = torch.optim.Adam(model.parameters(), lr=BUDGET["lr"])

    Y0 = torch.tensor(y0, dtype=torch.float32)
    XX = torch.tensor(Xn, dtype=torch.float32)
    YT = torch.tensor(yt, dtype=torch.float32)
    MM = torch.tensor(mask.astype(np.float32), dtype=torch.float32)
    tensors = (Y0, XX, YT, MM)

    one_step_weight = 0.0 if arm == "two_rate_rollout_only" else BUDGET["one_step_weight"]
    rollout_weight = 1.0 if one_step_weight == 0.0 else BUDGET["rollout_weight"]

    best = float("inf")
    best_state = None
    best_epoch = 0
    bad = 0
    history = []
    started = time.time()

    for epoch in range(1, BUDGET["epochs"] + 1):
        model.train()
        order = list(fold["train"])
        np.random.shuffle(order)
        train_detail = {}
        for event in order:
            opt.zero_grad(set_to_none=True)
            combined, roll, one = event_objective(
                model,
                event_rows[event],
                tensors,
                rollout_weight=rollout_weight,
                one_step_weight=one_step_weight,
                backward=True,
            )
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            train_detail[event] = {"combined": combined, "rollout": roll, "one_step": one}

        model.eval()
        score, val_detail = validation_score(model, fold["validation"], event_rows, tensors)
        history.append({"epoch": epoch, "validation": score})
        if score < best - 1e-10:
            best = score
            best_epoch = epoch
            bad = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= BUDGET["patience"]:
                break

    if best_state is None:
        raise RuntimeError("no checkpoint was selected")
    model.load_state_dict(best_state)
    model.eval()
    metrics, pred = test_metrics(model, te, tensors)
    rec = {
        "arm": arm,
        "seed": seed,
        "fold": fold["fold"],
        "test_event": fold["test"][0],
        "validation_events": fold["validation"],
        "training_events": fold["train"],
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "best_epoch": best_epoch,
        "stopped_at_epoch": history[-1]["epoch"],
        "best_validation_score": best,
        "validation_detail": val_detail,
        "u_init": float(u0),
        "r_init": float(r0),
        "rollout_weight": rollout_weight,
        "one_step_weight": one_step_weight,
        "training_time_s": round(time.time() - started, 2),
        **metrics,
    }
    log(
        f"{fold['test'][0]} {arm} seed={seed} best={best_epoch} stop={history[-1]['epoch']} "
        + " ".join(f"h{h}={metrics[f'rmse_h{h}']:.6f}" for h in REPORT_HORIZONS)
    )
    return rec, pred


def summarize(rows: list[dict]) -> dict:
    events = sorted({r["test_event"] for r in rows})
    arms = sorted({r["arm"] for r in rows})
    by_arm = {}
    for arm in arms:
        by_arm[arm] = {}
        for h in REPORT_HORIZONS:
            vals = []
            for e in events:
                x = [r[f"rmse_h{h}"] for r in rows if r["arm"] == arm and r["test_event"] == e]
                if x:
                    vals.append(float(np.mean(x)))
            by_arm[arm][f"rmse_h{h}"] = {
                "equal_event_mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "n_events": len(vals),
            }
        eps = [r["best_epoch"] for r in rows if r["arm"] == arm]
        by_arm[arm]["best_epoch"] = {
            "median": float(np.median(eps)),
            "min": int(np.min(eps)),
            "max": int(np.max(eps)),
        }

    comparisons = {}
    pairs = [
        ("net_scaled_v2", "two_rate_v2", "two_rate_vs_net_scaled"),
        ("two_rate_rollout_only", "two_rate_v2", "joint_vs_rollout_only"),
    ]
    for reference, proposed, label in pairs:
        if reference not in arms or proposed not in arms:
            continue
        comparisons[label] = {}
        for h in REPORT_HORIZONS:
            gains = {}
            for e in events:
                ref = [r[f"rmse_h{h}"] for r in rows if r["arm"] == reference and r["test_event"] == e]
                pro = [r[f"rmse_h{h}"] for r in rows if r["arm"] == proposed and r["test_event"] == e]
                if ref and pro:
                    rr = float(np.mean(ref))
                    pp = float(np.mean(pro))
                    gains[e] = 100.0 * (rr - pp) / rr
            g = np.array(list(gains.values()), dtype=float)
            comparisons[label][f"h{h}"] = {
                "event_gains_pct": gains,
                "equal_event_mean_pct": float(np.mean(g)),
                "median_pct": float(np.median(g)),
                "n_positive": int(np.sum(g > 0)),
                "n_events": int(len(g)),
            }
    return {"by_arm": by_arm, "comparisons": comparisons}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument(
        "--arms",
        nargs="+",
        default=["two_rate_v2", "net_scaled_v2", "two_rate_rollout_only"],
    )
    ap.add_argument("--test-events", nargs="*", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pred-dir", default=None)
    args = ap.parse_args()

    source = panelset.source_version(ROOT)
    wanted, panel_digest = panelset.resolve(INTERIM, str(MANIFEST))
    y0, X, yt, mask, fips, panel, origin, t0h = exp05.load_pooled(
        HORIZON, 12, panels=wanted, with_time=True
    )
    X = exp05.add_context(X, y0, HORIZON, t0_hour=t0h, clock="utc_hour")
    channels = schema.channel_list(panelset.channel_names(INTERIM), "utc_hour")
    if X.shape[-1] != 14 or len(channels) != 14:
        raise RuntimeError(f"expected 14 channels, got X={X.shape[-1]}, names={len(channels)}")

    events = sorted(set(panel.tolist()))
    smap = load_split_map(events)
    folds = smap["folds"]
    if args.test_events:
        keep = set(args.test_events)
        unknown = keep - set(events)
        if unknown:
            raise ValueError(f"unknown test events: {sorted(unknown)}")
        folds = [f for f in folds if f["test"][0] in keep]
    event_rows = {e: np.where(panel == e)[0] for e in events}

    lines = []
    def log(msg: str) -> None:
        print(msg, flush=True)
        lines.append(msg)

    log(
        f"unified-aistats-v2 events={len(events)} folds={len(folds)} samples={len(y0)} "
        f"panel_digest={panel_digest} split_digest={smap['digest']}"
    )
    rows = []
    pred_dir = ROOT / args.pred_dir if args.pred_dir else None
    if pred_dir:
        pred_dir.mkdir(parents=True, exist_ok=True)

    for fold in folds:
        test = fold["test"][0]
        if set(fold["train"]) & set(fold["validation"] + fold["test"]):
            raise RuntimeError("event leakage in split map")
        log(
            f"fold={fold['fold']} test={test} val={','.join(fold['validation'])} "
            f"train={len(fold['train'])}"
        )
        for arm in args.arms:
            for seed in args.seeds:
                rec, pred = fit_one(arm, seed, fold, event_rows, (y0, X, yt, mask), log)
                rows.append(rec)
                if pred_dir:
                    np.savez_compressed(
                        pred_dir / f"{arm}__{test}__seed{seed}.npz",
                        pred=pred,
                        test_rows=event_rows[test],
                        test_event=test,
                        arm=arm,
                        seed=seed,
                    )

    summary = summarize(rows)
    config = {
        **schema.result_header(
            experiment_id=Path(args.out).stem,
            source=source,
            panel_ids=sorted(wanted),
            panel_digest=panel_digest,
            channel_names=channels,
            channel_digest=panelset.channel_digest(panelset.channel_names(INTERIM)),
            clock="utc_hour",
            split_unit="event",
            outer_split_digest=smap["digest"],
            outer_split_seed=0,
            inner_split_seed=0,
            model_seeds=args.seeds,
            hyperparameters={**BUDGET, "horizon": HORIZON, "primary_horizons": PRIMARY_HORIZONS},
        ),
        "protocol": "docs/UNIFIED_AISTATS_V2_PROTOCOL.md",
        "split_map": str(SPLIT_MAP.relative_to(ROOT)),
        "arms": args.arms,
        "test_events": [f["test"][0] for f in folds],
        "statistical_unit": "event; seeds averaged within event",
    }
    payload = {"config": config, "rows": rows, "summary": summary, "log": lines}
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    log(f"wrote {out.relative_to(ROOT)}")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
