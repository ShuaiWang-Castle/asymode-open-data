#!/usr/bin/env python3
"""Formal coarse two-flow temporal-transfer experiment.

The confirmation task is frozen by
analysis/coarse_flow_formal_20260904/00_FORMAL_LOCK.md.  This script supports a
reproduction-only development mode and a one-shot confirmation mode.  It never
selects K or a feature after seeing confirmation events.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import flow_data as F

K_FIXED = 8
FIT_YEARS_DEVELOPMENT = {2018, 2019, 2020}
VALIDATION_YEAR = 2021
FINAL_SOURCE_YEARS = {2018, 2019, 2020, 2021}
CONFIRMATION_YEARS = {2022, 2024}
SEED = 0
BOOTSTRAP_B = 50_000


def year(event: str) -> int:
    return int(event[:4])


def exact_signflip(diff: np.ndarray) -> float:
    diff = np.asarray(diff, float)
    obs = abs(float(diff.mean()))
    vals = [abs(float(np.mean(diff * np.asarray(s))))
            for s in itertools.product((-1.0, 1.0), repeat=len(diff))]
    return float(np.mean(np.asarray(vals) >= obs - 1e-18))


def bootstrap_ci(diff: np.ndarray, seed: int = 20260904) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    d = np.asarray(diff, float)
    draws = d[rng.integers(0, len(d), size=(BOOTSTRAP_B, len(d)))].mean(axis=1)
    return float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def active_rows(events: dict, names: list[str]) -> pd.DataFrame:
    parts = []
    for e in names:
        ev = events[e]
        if not ev.active_available:
            continue
        parts.append(F.transitions(ev, ev.active_start, ev.active_end))
    if not parts:
        raise RuntimeError("no available active-48 source events")
    return pd.concat(parts, ignore_index=True)


def fit_damped(source: pd.DataFrame, weights: np.ndarray) -> float:
    y = source["y"].to_numpy(float)
    yn = y + source["delta"].to_numpy(float)
    den = float(np.sum(weights * y * y))
    return float(np.clip(np.sum(weights * y * yn) / den if den > 0 else 1.0, 0.0, 1.0))


def path_generic(ev, origins, predict_step, horizon=24):
    records = []
    for o in origins:
        if o < 0 or o + horizon >= ev.y.shape[1]:
            continue
        idx = np.where(ev.obs[:, o] & np.isfinite(ev.y[:, o]))[0]
        if not len(idx):
            continue
        yhat = ev.y[idx, o].copy()
        sse = 0.0; n = 0; h_mse = np.nan
        for step in range(1, horizon + 1):
            t = o + step
            yhat = np.clip(predict_step(idx, t, yhat), 0.0, 1.0)
            mask = ev.obs[idx, t] & np.isfinite(ev.y[idx, t])
            if mask.any():
                err = yhat[mask] - ev.y[idx[mask], t]
                sse += float(np.sum(err * err)); n += int(mask.sum())
                if step == horizon:
                    h_mse = float(np.mean(err * err))
        records.append((sse / max(n, 1), h_mse, n))
    if not records:
        return np.nan, np.nan, 0
    return (float(np.mean([r[0] for r in records])),
            float(np.nanmean([r[1] for r in records])),
            int(sum(r[2] for r in records)))


def direct_examples(ev, origins, horizon=24):
    xs, ys = [], []
    for o in origins:
        if o < 0 or o + horizon >= ev.y.shape[1]:
            continue
        mask = (ev.obs[:, o] & ev.obs[:, o + horizon]
                & np.isfinite(ev.y[:, o]) & np.isfinite(ev.y[:, o + horizon]))
        idx = np.where(mask)[0]
        if not len(idx):
            continue
        future = ev.X[idx, o + 1:o + horizon + 1, :].astype(float)
        # Frozen direct baseline: y0, current 24-feature vector, future mean and
        # maximum for the 12 raw weather channels, and phase at origin.
        X = np.c_[ev.y[idx, o], ev.features[idx, o, :],
                  future.mean(axis=1), future.max(axis=1),
                  np.full((len(idx), 1), (o - ev.peak) / 24.0)]
        xs.append(X); ys.append(ev.y[idx, o + horizon])
    return np.vstack(xs), np.concatenate(ys)


def cluster_diagnostics(model: dict, source: pd.DataFrame) -> pd.DataFrame:
    X = source[F.EVENT_FEATURES].to_numpy(np.float32)
    Z = model["scaler"].transform(X).astype(model["km"].cluster_centers_.dtype)
    labels = model["km"].predict(Z)
    w = F.equal_event_weights(source)
    rows = []
    for k in range(K_FIXED):
        z = labels == k
        wk = w[z] / w[z].sum()
        y = source.loc[z, "y"].to_numpy(float)
        d = source.loc[z, "delta"].to_numpy(float)
        U, R = model["two"][k][:2]
        pred = U * (1 - y) - R * y
        sigma2 = float(np.sum(wk * (d - pred) ** 2))
        mu = float(np.sum(wk * y)); v = float(np.sum(wk * (y - mu) ** 2))
        A = float(np.sum(wk * (1-y)**2)); B = float(np.sum(wk * y*y))
        G = v * min(R*R/A if A > 0 else np.inf, U*U/B if B > 0 else np.inf)
        n_eff = float(1.0 / np.sum(wk*wk))
        one = model["one"][k]
        rows.append(dict(cluster=k, n=int(z.sum()), n_eff=n_eff, mu=mu, v=v,
                         U=U, R=R, c=min(U,R), one_branch=one[2],
                         one_U=one[0], one_R=one[1], sigma=math.sqrt(sigma2),
                         G=G, Gamma_neff=n_eff*G/sigma2 if sigma2 > 0 else np.nan,
                         treatment_over_sigma=(min(U,R)*math.sqrt(float(np.sum(wk*(1-2*y)**2)))
                                               / math.sqrt(sigma2) if sigma2 > 0 else np.nan)))
    return pd.DataFrame(rows)


def fit_models(source: pd.DataFrame, fit_direct_events: list, events: dict, skip_baselines: bool = False):
    F.EVENT_FEATURES = next(iter(events.values())).feature_names
    sieve = F.fit_clusters(source, K_FIXED, seed=SEED, cap=None)
    weights = F.equal_event_weights(source)
    alpha = fit_damped(source, weights)
    y = source["y"].to_numpy(np.float32)
    target = (source["y"] + source["delta"]).to_numpy(np.float32)
    X = source[F.EVENT_FEATURES].to_numpy(np.float32)
    recursive_hgb = direct_hgb = None
    if not skip_baselines:
        recursive_hgb = HistGradientBoostingRegressor(
            max_iter=200, learning_rate=.05, max_leaf_nodes=31,
            min_samples_leaf=100, l2_regularization=1e-4,
            early_stopping=False, random_state=SEED,
        ).fit(np.c_[X, y], target, sample_weight=weights * len(weights))
        dx, dy = [], []
        for e in fit_direct_events:
            ev = events[e]
            x, yy = direct_examples(ev, range(ev.peak-24, ev.peak+1))
            dx.append(x); dy.append(yy)
        direct_hgb = HistGradientBoostingRegressor(
            max_iter=250, learning_rate=.05, max_leaf_nodes=31,
            min_samples_leaf=100, l2_regularization=1e-4,
            early_stopping=True, random_state=SEED,
        ).fit(np.vstack(dx), np.concatenate(dy))
    return sieve, alpha, recursive_hgb, direct_hgb


def evaluate_event(ev, sieve, alpha, recursive_hgb, direct_hgb):
    origins = list(range(ev.peak-24, ev.peak+1))
    rows = []
    # Sieve one/two paths.
    paths = F.path_event(sieve, ev, origins, horizon=24)
    for arm in ("one", "two"):
        rows.append(dict(event=ev.event, family=ev.family, arm=arm,
                         path24_mse=float(paths[f"{arm}_path_mse"].mean()),
                         h24_mse=float(paths[f"{arm}_h24_mse"].mean())))
    # Baselines.
    predictors = {
        "persistence": lambda idx, t, y: y,
        "damped_persistence": lambda idx, t, y: alpha * y,
    }
    if recursive_hgb is not None:
        predictors["recursive_hgb"] = lambda idx, t, y: recursive_hgb.predict(
            np.c_[ev.features[idx, t, :], y])
    for name, fn in predictors.items():
        pm, hm, _ = path_generic(ev, origins, fn)
        rows.append(dict(event=ev.event, family=ev.family, arm=name,
                         path24_mse=pm, h24_mse=hm))
    if direct_hgb is not None:
        xd, yd = direct_examples(ev, origins)
        pdirect = np.clip(direct_hgb.predict(xd), 0.0, 1.0)
        rows.append(dict(event=ev.event, family=ev.family, arm="direct_hgb_h24",
                         path24_mse=np.nan, h24_mse=float(np.mean((pdirect-yd)**2))))
    # Theory-aligned one-step endpoint over active-48 only.
    if ev.active_available:
        _, step = F.one_step_event(sieve, ev, ev.active_start, ev.active_end)
        for arm in ("one", "two"):
            for r in rows:
                if r["arm"] == arm:
                    r["active48_step_mse"] = step[arm]["mse"]
                    r["active48_n"] = step[arm]["n"]
    return rows


def paired_table(metrics: pd.DataFrame, endpoint: str) -> pd.DataFrame:
    p = metrics.pivot_table(index="event", columns="arm", values=endpoint)
    rows = []
    for ref in [c for c in p.columns if c != "two"]:
        q = p[[ref, "two"]].dropna()
        d = (q[ref] - q["two"]).to_numpy(float)
        lo, hi = bootstrap_ci(d)
        rows.append(dict(endpoint=endpoint, reference=ref, n_events=len(d),
                         mean_reference=float(q[ref].mean()),
                         mean_two=float(q["two"].mean()),
                         mean_difference=float(d.mean()),
                         mean_gain_pct=float(np.mean(100*d/np.maximum(q[ref],1e-18))),
                         median_gain_pct=float(np.median(100*d/np.maximum(q[ref],1e-18))),
                         positive_events=int(np.sum(d>0)),
                         signflip_p=exact_signflip(d),
                         bootstrap_lo=lo, bootstrap_hi=hi,
                         loo_min_mean=float(min(np.delete(d,i).mean() for i in range(len(d))))
                         if len(d)>1 else np.nan))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", choices=["development", "confirmation"], required=True)
    ap.add_argument("--skip-baselines", action="store_true")
    args = ap.parse_args()
    root = Path(args.data_root).resolve(); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    events, manifest = F.load_all(root)
    F.EVENT_FEATURES = next(iter(events.values())).feature_names
    if args.stage == "development":
        source_names = [e for e in events if year(e) in FIT_YEARS_DEVELOPMENT]
        target_names = [e for e in events if year(e) == VALIDATION_YEAR]
    else:
        source_names = [e for e in events if year(e) in FINAL_SOURCE_YEARS]
        target_names = [e for e in events if year(e) in CONFIRMATION_YEARS]
    source = active_rows(events, source_names)
    sieve, alpha, recursive_hgb, direct_hgb = fit_models(source, source_names, events, args.skip_baselines)
    metrics = pd.DataFrame([r for e in target_names for r in
                            evaluate_event(events[e], sieve, alpha, recursive_hgb, direct_hgb)])
    metrics.to_csv(out/"EVENT_METRICS.csv", index=False)
    cluster_diagnostics(sieve, source).to_csv(out/"SOURCE_CLUSTER_DIAGNOSTICS.csv", index=False)
    paired = pd.concat([paired_table(metrics, ep) for ep in
                        ["active48_step_mse", "path24_mse", "h24_mse"]], ignore_index=True)
    paired.to_csv(out/"PAIRED_RESULTS.csv", index=False)
    provenance = {
        "stage": args.stage, "manifest_digest": manifest["digest"], "K": K_FIXED,
        "source_events": source_names, "target_events": target_names,
        "feature_names": F.EVENT_FEATURES, "random_state": SEED,
        "runtime_seconds": round(time.time()-started,2),
        "data_root_provenance": (root/"EXPORT_PROVENANCE.txt").read_text().strip()
            if (root/"EXPORT_PROVENANCE.txt").exists() else "not embedded",
    }
    (out/"RUN_PROVENANCE.json").write_text(json.dumps(provenance, indent=2))
    # Programmatic report: no hand-selected event or metric.
    one_step = paired[(paired.endpoint == "active48_step_mse") & (paired.reference == "one")]
    path = paired[(paired.endpoint == "path24_mse") & (paired.reference == "one")]
    def row_or_none(x):
        return None if x.empty else x.iloc[0]
    rs, rp = row_or_none(one_step), row_or_none(path)
    structural_basic = bool(rs is not None and rp is not None
                            and rs.mean_difference > 0 and rp.mean_difference > 0
                            and rs.loo_min_mean > 0 and rp.loo_min_mean > 0)
    structural_stat = bool(structural_basic and rs.signflip_p < 0.05 and rp.signflip_p < 0.05)
    report = [
        f"# Coarse two-flow {args.stage} report", "",
        f"Source years: `{sorted(set(year(e) for e in source_names))}`.  ",
        f"Target years: `{sorted(set(year(e) for e in target_names))}`.  ",
        f"K: `{K_FIXED}`; event is the inferential unit.", "",
        "## Paired results", "", paired.to_markdown(index=False), "",
        "## Predeclared structural gate", "",
        f"- positive one-step and path means with positive leave-one-event minima: **{structural_basic}**",
        f"- both exact sign-flip p-values below 0.05: **{structural_stat}**", "",
        "The statistical gate is descriptive in development mode. In confirmation mode it is the frozen paper-evidence decision.", "",
        "## Cluster diagnostics", "", cluster_diagnostics(sieve, source).to_markdown(index=False), "",
    ]
    (out/"REPORT.md").write_text("\n".join(report))
    print(paired.to_string(index=False))

if __name__ == "__main__":
    main()
