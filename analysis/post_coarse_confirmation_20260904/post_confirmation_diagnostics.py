#!/usr/bin/env python3
"""Post-hoc diagnostics for the frozen coarse-flow confirmation.

This script is not a model-selection or confirmation runner. It uses already
revealed 2022/2024 outcomes to decompose the frozen active-48 one-step result and
to audit semantic weather-driver coverage. Outputs are explanatory only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="Pinned public-data root")
    ap.add_argument(
        "--formal-dir",
        default=str(Path(__file__).resolve().parents[1] / "coarse_flow_formal_20260904"),
        help="Directory containing implementation/flow_data.py",
    )
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent))
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    formal_dir = Path(args.formal_dir).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(formal_dir / "implementation"))
    import flow_data as F  # noqa: E402

    events, manifest = F.load_all(data_root)
    if manifest["digest"] != "db286b4960a4":
        raise RuntimeError("unexpected manifest digest")
    F.EVENT_FEATURES = next(iter(events.values())).feature_names

    # Semantic driver-coverage audit.
    coverage = []
    county_status: dict[str, set[bool]] = {}
    for event, ev in sorted(events.items()):
        zero_county = np.all(np.isclose(ev.X, 0.0, atol=1e-12), axis=(1, 2))
        for fips, is_zero in zip(ev.fips, zero_county):
            county_status.setdefault(str(fips).zfill(5), set()).add(bool(is_zero))
        if ev.active_available:
            d = F.transitions(ev, ev.active_start, ev.active_end)
            zero_transition = zero_county[d["ci"].to_numpy(int)]
            active_n = len(d)
            active_frac = float(zero_transition.mean())
        else:
            active_n = 0
            active_frac = np.nan
        coverage.append(
            {
                "event": event,
                "year": int(event[:4]),
                "family": ev.family,
                "n_counties": len(ev.fips),
                "n_zero_weather_counties": int(zero_county.sum()),
                "frac_zero_weather_counties": float(zero_county.mean()),
                "active_n": active_n,
                "active_frac_zero_weather_transitions": active_frac,
            }
        )
    cov = pd.DataFrame(coverage)
    cov.to_csv(out / "weather_coverage_by_event.csv", index=False)
    mixed = sum(len(v) > 1 for v in county_status.values())
    always_zero = sum(v == {True} for v in county_status.values())
    always_nonzero = sum(v == {False} for v in county_status.values())

    # Refit the exact frozen source estimator.
    source_names = [e for e in events if int(e[:4]) in {2018, 2019, 2020, 2021}]
    source = pd.concat(
        [
            F.transitions(events[e], events[e].active_start, events[e].active_end)
            for e in source_names
            if events[e].active_available
        ],
        ignore_index=True,
    )
    model = F.fit_clusters(source, 8, seed=0, cap=None)

    Xs = source[F.EVENT_FEATURES].to_numpy(np.float32)
    Zs = model["scaler"].transform(Xs).astype(model["km"].cluster_centers_.dtype)
    source_labels = model["km"].predict(Zs)
    source_weights = F.equal_event_weights(source)
    centers = pd.DataFrame(
        model["scaler"].inverse_transform(model["km"].cluster_centers_),
        columns=F.EVENT_FEATURES,
    )
    for k in range(8):
        centers.loc[k, "U"] = model["two"][k][0]
        centers.loc[k, "R"] = model["two"][k][1]
        centers.loc[k, "one_branch"] = model["one"][k][2]
        centers.loc[k, "source_transition_share"] = float(np.mean(source_labels == k))
        centers.loc[k, "source_equal_event_weight"] = float(source_weights[source_labels == k].sum())
    centers.index.name = "cluster"
    centers.to_csv(out / "source_cluster_centers.csv")

    # Exact target-oracle decomposition in the fixed source cells.
    event_rows: list[dict] = []
    cluster_rows: list[dict] = []
    target_names = [e for e in events if int(e[:4]) in {2022, 2024}]
    for event in target_names:
        ev = events[event]
        if not ev.active_available:
            event_rows.append({"event": event, "family": ev.family, "available": False})
            continue
        df = F.transitions(ev, ev.active_start, ev.active_end)
        X = df[F.EVENT_FEATURES].to_numpy(np.float32)
        y = df["y"].to_numpy(float)
        delta = df["delta"].to_numpy(float)
        labels = model["km"].predict(
            model["scaler"].transform(X).astype(model["km"].cluster_centers_.dtype)
        )
        pred_one, _, _, _ = F.predict_rate(model, X, y, "one")
        pred_two, _, _, _ = F.predict_rate(model, X, y, "two")
        oracle_one = np.empty_like(delta)
        oracle_two = np.empty_like(delta)
        branch_mismatch = 0
        target_two_boundary = 0
        active_clusters = 0
        for k in range(8):
            z = labels == k
            if not z.any():
                continue
            active_clusters += 1
            w = np.ones(int(z.sum()), dtype=float) / int(z.sum())
            one = F.fit_one(y[z], delta[z], w)
            two = F.fit_two(y[z], delta[z], w)
            oracle_one[z] = one[0] * (1.0 - y[z]) - one[1] * y[z]
            oracle_two[z] = two[0] * (1.0 - y[z]) - two[1] * y[z]
            branch_mismatch += int(one[2] != model["one"][k][2])
            target_two_boundary += int(min(two[0], two[1]) < 1e-10)
            src_one_mse = float(np.mean((delta[z] - pred_one[z]) ** 2))
            src_two_mse = float(np.mean((delta[z] - pred_two[z]) ** 2))
            oracle_one_mse = float(np.mean((delta[z] - oracle_one[z]) ** 2))
            oracle_two_mse = float(np.mean((delta[z] - oracle_two[z]) ** 2))
            cluster_rows.append(
                {
                    "event": event,
                    "family": ev.family,
                    "cluster": k,
                    "n": int(z.sum()),
                    "share": float(z.mean()),
                    "src_U": model["two"][k][0],
                    "src_R": model["two"][k][1],
                    "src_one_branch": model["one"][k][2],
                    "tgt_U": two[0],
                    "tgt_R": two[1],
                    "tgt_one_branch": one[2],
                    "src_one_mse": src_one_mse,
                    "src_two_mse": src_two_mse,
                    "oracle_one_mse": oracle_one_mse,
                    "oracle_two_mse": oracle_two_mse,
                    "oracle_gap": oracle_one_mse - oracle_two_mse,
                    "transfer_one": src_one_mse - oracle_one_mse,
                    "transfer_two": src_two_mse - oracle_two_mse,
                    "observed_diff": src_one_mse - src_two_mse,
                }
            )
        source_one_mse = float(np.mean((delta - pred_one) ** 2))
        source_two_mse = float(np.mean((delta - pred_two) ** 2))
        oracle_one_mse = float(np.mean((delta - oracle_one) ** 2))
        oracle_two_mse = float(np.mean((delta - oracle_two) ** 2))
        event_rows.append(
            {
                "event": event,
                "family": ev.family,
                "available": True,
                "n": len(df),
                "source_one_mse": source_one_mse,
                "source_two_mse": source_two_mse,
                "observed_diff": source_one_mse - source_two_mse,
                "target_oracle_one_mse": oracle_one_mse,
                "target_oracle_two_mse": oracle_two_mse,
                "oracle_gap": oracle_one_mse - oracle_two_mse,
                "transfer_one": source_one_mse - oracle_one_mse,
                "transfer_two": source_two_mse - oracle_two_mse,
                "transfer_penalty_diff": (source_one_mse - oracle_one_mse)
                - (source_two_mse - oracle_two_mse),
                "decomposition_error": (source_one_mse - source_two_mse)
                - (
                    (oracle_one_mse - oracle_two_mse)
                    + (source_one_mse - oracle_one_mse)
                    - (source_two_mse - oracle_two_mse)
                ),
                "active_clusters": active_clusters,
                "branch_mismatch": branch_mismatch,
                "target_two_boundary": target_two_boundary,
            }
        )

    event_out = pd.DataFrame(event_rows)
    cluster_out = pd.DataFrame(cluster_rows)
    event_out.to_csv(out / "target_oracle_decomposition.csv", index=False)
    cluster_out.to_csv(out / "target_cluster_decomposition.csv", index=False)

    available = event_out[event_out["available"] == True]  # noqa: E712
    summary = {
        "manifest_digest": manifest["digest"],
        "unique_counties": len(county_status),
        "always_zero_weather_counties": always_zero,
        "always_nonzero_weather_counties": always_nonzero,
        "mixed_weather_status_counties": mixed,
        "mean_observed_difference": float(available["observed_diff"].mean()),
        "mean_target_oracle_gap": float(available["oracle_gap"].mean()),
        "mean_transfer_penalty_difference": float(available["transfer_penalty_diff"].mean()),
        "positive_target_oracle_gap_events": int((available["oracle_gap"] > 0).sum()),
        "n_available_events": len(available),
        "max_decomposition_error": float(available["decomposition_error"].abs().max()),
    }
    pd.Series(summary).to_json(out / "POSTHOC_SUMMARY.json", indent=2)
    print(pd.Series(summary).to_string())


if __name__ == "__main__":
    main()
