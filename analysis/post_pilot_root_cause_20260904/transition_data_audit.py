#!/usr/bin/env python3
"""Data-path audit for the V2 pilot, with no model fitting.

Reconstructs the exact Stage-A transition inclusion rule from
`experiments/paper_v2_pilot.py` and compares it with the intended adjacent-observed
transition rule. It also compares row-pooled and equal-event constant fits and
quantifies how the anchor windows sample positive, negative and quiet transitions.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from asymode.evalproto import to_hourly  # noqa: E402
from asymode_paper.initialization import fit_r_ray, fit_two_flow, fit_u_ray, modular_init  # noqa: E402
from asymode_paper.asymmetric_flows import CAP_R, CAP_U_BKG, CAP_U_MAIN  # noqa: E402
from paper_v2_pilot import county_group  # noqa: E402

INTERIM = ROOT / "data/interim"
CC = ROOT / "analysis/gpt_rescue_20260904/cc_v2"
HERE = ROOT / "analysis/post_pilot_root_cause_20260904"
EPS_DY = 1e-8


def load_hourly(event: str):
    z = np.load(INTERIM / f"panel_{event}.npz", allow_pickle=True)
    y, obs = to_hourly(z["y"], z["observed"])
    return np.nan_to_num(y), obs.astype(bool), np.asarray(z["fips"], dtype=str)


def selected_transitions(event: str, anchors: list[int], which: str = "fit") -> dict:
    y, obs, fips = load_hourly(event)
    idx = np.where(np.array([county_group(f, event) for f in fips]) == which)[0]
    seen = set()
    rows = []
    for k in anchors:
        origin_obs = obs[:, k]
        for t in range(24):
            current = k + t
            nxt = current + 1
            if nxt >= y.shape[1]:
                continue
            for ci in idx:
                key = (int(ci), nxt)
                if key in seen:
                    continue
                # Literal pilot inclusion: observed origin and observed next target.
                pilot_include = bool(origin_obs[ci] and obs[ci, nxt])
                if not pilot_include:
                    continue
                seen.add(key)
                correct_pair = bool(obs[ci, current] and obs[ci, nxt])
                rows.append((event, ci, current, y[ci, current], y[ci, nxt] - y[ci, current], correct_pair))
    if not rows:
        return dict(event=event, y=np.array([]), dy=np.array([]), correct=np.array([], bool))
    a = np.array(rows, dtype=object)
    return dict(event=event, y=a[:, 3].astype(float), dy=a[:, 4].astype(float), correct=a[:, 5].astype(bool))


def full_transitions(event: str, which: str = "fit") -> dict:
    y, obs, fips = load_hourly(event)
    idx = np.where(np.array([county_group(f, event) for f in fips]) == which)[0]
    yy = y[idx, :-1]
    dy = y[idx, 1:] - yy
    m = obs[idx, :-1] & obs[idx, 1:]
    return dict(event=event, y=yy[m], dy=dy[m])


def summary(y: np.ndarray, dy: np.ndarray) -> dict:
    if len(y) == 0:
        return {}
    return dict(
        n=int(len(y)),
        y_zero=float(np.mean(y == 0)),
        y_active=float(np.mean(y > 0.01)),
        dy_pos=float(np.mean(dy > EPS_DY)),
        dy_neg=float(np.mean(dy < -EPS_DY)),
        dy_quiet=float(np.mean(np.abs(dy) <= EPS_DY)),
        mean_dy=float(np.mean(dy)),
        mean_pos_dy=float(np.mean(dy[dy > EPS_DY])) if np.any(dy > EPS_DY) else 0.0,
        mean_neg_dy=float(np.mean(dy[dy < -EPS_DY])) if np.any(dy < -EPS_DY) else 0.0,
        state_var=float(np.var(y)),
        A=float(np.mean((1.0 - y) ** 2)),
        B=float(np.mean(y ** 2)),
        U_ray_numerator=float(np.sum(dy * (1.0 - y))),
        R_ray_numerator=float(np.sum((-dy) * y)),
    )


def fit_with_weights(per_event: dict[str, dict], equal_event: bool) -> dict:
    ys, dys, ws = [], [], []
    for e, d in per_event.items():
        y, dy = d["y"], d["dy"]
        if len(y) == 0:
            continue
        ys.append(y); dys.append(dy)
        ws.append(np.full(len(y), 1.0 / len(y) if equal_event else 1.0))
    y = np.concatenate(ys); dy = np.concatenate(dys); w = np.concatenate(ws)
    U, R = fit_two_flow(y, dy, w, CAP_U_MAIN + CAP_U_BKG, CAP_R)
    a = fit_u_ray(y, dy, w, CAP_U_MAIN + CAP_U_BKG)
    b = fit_r_ray(y, dy, w, CAP_R)
    pred_u = a * (1.0 - y)
    pred_r = -b * y
    sse_u = float(np.sum(w * (dy - pred_u) ** 2))
    sse_r = float(np.sum(w * (dy - pred_r) ** 2))
    init = modular_init(U, R, CAP_U_MAIN, CAP_U_BKG, CAP_R)
    raw_prob = 1.0 / (1.0 + np.exp(-init["raw_u_bias"]))
    bkg_prob = 1.0 / (1.0 + np.exp(-init["background_bias"]))
    return dict(
        U=U, R=R, U_ray=a, R_ray=b,
        correct_one_flow_branch="interruption" if sse_u <= sse_r else "restoration",
        sse_u=sse_u, sse_r=sse_r,
        raw_u_bias=init["raw_u_bias"], raw_sigmoid_derivative=raw_prob * (1.0 - raw_prob),
        background_bias=init["background_bias"],
        background_sigmoid_derivative=bkg_prob * (1.0 - bkg_prob),
    )


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((CC / "pilot_run_config.json").read_text())
    folds = json.loads((CC / "event_folds_v2.json").read_text())
    anchors = {e: list(v) for e, v in folds["anchors"].items()}
    fold_of = {e: int(f) for f, es in folds["folds"].items() for e in es}
    pilot = list(cfg["pilot_events"])

    records = []
    fold_payload = {}
    for test in pilot:
        f = fold_of[test]
        if str(f) in fold_payload:
            continue
        src = [e for e in fold_of if fold_of[e] != f]
        sel = {e: selected_transitions(e, anchors[e]) for e in src}
        full = {e: full_transitions(e) for e in src}
        for e in src:
            ss = summary(sel[e]["y"], sel[e]["dy"])
            fs = summary(full[e]["y"], full[e]["dy"])
            records.append(dict(
                heldout_fold=f, event=e,
                selected_n=ss.get("n", 0), full_n=fs.get("n", 0),
                selected_fraction_of_full=ss.get("n", 0) / max(fs.get("n", 1), 1),
                bogus_current_unobserved_share=float(np.mean(~sel[e]["correct"])) if len(sel[e]["correct"]) else 0.0,
                **{f"selected_{k}": v for k, v in ss.items()},
                **{f"full_{k}": v for k, v in fs.items()},
            ))
        fold_payload[str(f)] = dict(
            source_events=src,
            selected_row_pooled=fit_with_weights(sel, equal_event=False),
            selected_equal_event=fit_with_weights(sel, equal_event=True),
            full_row_pooled=fit_with_weights(full, equal_event=False),
            full_equal_event=fit_with_weights(full, equal_event=True),
            selected_pooled_summary=summary(
                np.concatenate([d["y"] for d in sel.values()]),
                np.concatenate([d["dy"] for d in sel.values()]),
            ),
            full_pooled_summary=summary(
                np.concatenate([d["y"] for d in full.values()]),
                np.concatenate([d["dy"] for d in full.values()]),
            ),
            pilot_mask_bogus_share=float(
                np.mean(np.concatenate([~d["correct"] for d in sel.values()]))
            ),
        )

    df = pd.DataFrame(records)
    df.to_csv(HERE / "TRANSITION_EVENT_AUDIT.csv", index=False)
    (HERE / "TRANSITION_DATA_AUDIT.json").write_text(json.dumps(fold_payload, indent=2))

    lines = [
        "# Generated transition/data-path audit",
        "",
        "This audit reconstructs the pilot's Stage-A sampling and constant fits without training a neural model.",
        "",
    ]
    for f, z in fold_payload.items():
        rp = z["selected_row_pooled"]
        ee = z["selected_equal_event"]
        fr = z["full_row_pooled"]
        ss = z["selected_pooled_summary"]
        fs = z["full_pooled_summary"]
        lines += [
            f"## Held-out fold {f}", "",
            "### Transition composition", "",
            "| sample | n | share y>0.01 | positive dy | negative dy | quiet dy | state variance |",
            "|---|---:|---:|---:|---:|---:|---:|",
            f"| anchor windows | {ss['n']} | {ss['y_active']:.4f} | {ss['dy_pos']:.4f} | {ss['dy_neg']:.4f} | {ss['dy_quiet']:.4f} | {ss['state_var']:.3e} |",
            f"| all adjacent observed pairs | {fs['n']} | {fs['y_active']:.4f} | {fs['dy_pos']:.4f} | {fs['dy_neg']:.4f} | {fs['dy_quiet']:.4f} | {fs['state_var']:.3e} |",
            "",
            f"The anchor-window sampler retains `{ss['n']/fs['n']:.1%}` of available adjacent transitions. Under the literal pilot mask, `{z['pilot_mask_bogus_share']:.3%}` of included Stage-A rows do not have an observed current state and are formed from the zero-filled state array.",
            "",
            "### Constant-class sensitivity", "",
            "| weighting/sample | U | R | U-ray | R-ray | correct ray | raw-U sigmoid derivative |",
            "|---|---:|---:|---:|---:|---|---:|",
            f"| selected, row pooled (pilot) | {rp['U']:.3e} | {rp['R']:.3e} | {rp['U_ray']:.3e} | {rp['R_ray']:.3e} | {rp['correct_one_flow_branch']} | {rp['raw_sigmoid_derivative']:.3e} |",
            f"| selected, equal event | {ee['U']:.3e} | {ee['R']:.3e} | {ee['U_ray']:.3e} | {ee['R_ray']:.3e} | {ee['correct_one_flow_branch']} | {ee['raw_sigmoid_derivative']:.3e} |",
            f"| full panel, row pooled | {fr['U']:.3e} | {fr['R']:.3e} | {fr['U_ray']:.3e} | {fr['R_ray']:.3e} | {fr['correct_one_flow_branch']} | {fr['raw_sigmoid_derivative']:.3e} |",
            "",
            "The exact global constant may be a valid baseline, but its very negative interruption logit creates a poor trainable start for a rare, context-dependent onset model: the sigmoid derivative shown in the final column is the maximum scale available to all upstream interruption gradients before the additional gate and state multipliers.",
            "",
        ]
    lines += [
        "## Mask defect", "",
        "The intended teacher-forced mask is `observed(current) AND observed(next)`. The pilot instead uses `observed(origin) AND observed(next)` for every step in an anchor window. Since the hourly state array is zero-filled before packing, an unobserved intermediate current state is treated as zero. Even a small contaminated share is disproportionately relevant to the zero-state interruption signal and must be removed before another one-step comparison.",
        "",
        "## Weighting mismatch", "",
        "The training and validation objectives are described as equal-event risks, but the class initialization and deterministic constant baselines concatenate all source transitions with unit row weights. The initial point is therefore optimized for a different estimand. The table above reports the corresponding equal-event constants; the main implementation should use the objective-matched version or clearly separate baseline scoring from trainable initialization.",
    ]
    (HERE / "TRANSITION_DATA_AUDIT_GENERATED.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
