#!/usr/bin/env python3
"""Audit the observation assumptions behind a 216-hour EAGLE-I panel.

The documented EAGLE-I release omits zero-outage rows. A missing county-quarter is therefore unknown,
not an observed zero. This script distinguishes source-positive records from the
zeros inferred by the existing panel's collection-run and county-service rules.
It also computes a *sensitivity scenario*, carrying a positive source value for
up to ``--locf-quarters`` later missing quarters and leaving longer absences
unknown. LOCF is not ground truth.

Use the original national, year-level EAGLE-I parquet, not a county-only extract:
the existing panel defines a collection-run proxy using at least five rows across
all counties. The script reads a bounded window from that parquet and reconstructs
the panel's mask before comparing values. A mismatch is shown explicitly.

Example:
    python experiments/open_gcrk_20260919/phase2_20260925/measurement_audit.py \
        --raw data/interim/eaglei_outages_2021.parquet \
        --panel data/interim/open_gcrk/panel216_2021-12-11.npz \
        --out runs/measurement_audit_2021-12-11.json

    python experiments/open_gcrk_20260919/phase2_20260925/measurement_audit.py --self-test

The installed Python needs a parquet engine (pyarrow or fastparquet) for real
inputs. ``--self-test`` needs only numpy and pandas.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


ORIGIN = 72
HOURS = 216
QUARTERS_PER_HOUR = 4
LEADS = {
    "full_1_144h": (72, 216),
    "1_6h": (72, 78),
    "7_24h": (78, 96),
    "25_48h": (96, 120),
    "49_144h": (120, 216),
}


def _normalize_time(values: pd.Series | pd.Index) -> pd.DatetimeIndex:
    # The publisher states GMT. Do not silently localize naive timestamps to the
    # machine's timezone, and reject missing or unparsable timestamps.
    parsed = pd.to_datetime(values, errors="raise", utc=True)
    out = pd.DatetimeIndex(parsed).tz_convert("UTC").tz_localize(None)
    if out.hasnans:
        raise ValueError("Source contains null timestamps")
    return out


def _read_panel(path: Path) -> dict:
    # build_panel216.py stores timestamp strings in an object-dtype NumPy array.
    # Loading it requires pickle; only pass trusted project-generated panels.
    with np.load(path, allow_pickle=True) as z:
        needed = {"fips", "ts", "y", "observed", "denominator"}
        if not needed.issubset(z.files):
            raise ValueError(f"panel missing keys: {sorted(needed - set(z.files))}")
        a = {k: z[k].copy() for k in needed}
        a["event"] = str(z["event"].item()) if "event" in z.files else path.stem
    a["fips"] = np.asarray([str(x).zfill(5) for x in a["fips"]])
    if len(set(a["fips"])) != len(a["fips"]):
        raise ValueError("Panel repeats a FIPS code")
    a["ts"] = _normalize_time(a["ts"])
    expect = pd.date_range(a["ts"][0], periods=HOURS, freq="h")
    if len(a["ts"]) != HOURS or not np.array_equal(a["ts"], expect):
        raise ValueError("Panel must contain 216 consecutive UTC hours")
    n = len(a["fips"])
    if a["y"].shape != (n, HOURS) or a["observed"].shape != (n, HOURS):
        raise ValueError("Panel y/observed must have shape (counties, 216)")
    a["observed"] = a["observed"].astype(bool)
    a["denominator"] = np.asarray(a["denominator"], dtype=float)
    if a["denominator"].shape != (n,) or not np.isfinite(a["denominator"]).all() or (a["denominator"] <= 0).any():
        raise ValueError("Every panel denominator must be finite and positive")
    if np.isnan(a["y"][a["observed"]]).any():
        raise ValueError("Panel has missing target values where observed=True")
    return a


def _read_raw(paths: list[Path], t0: pd.Timestamp, tlast: pd.Timestamp,
              service_days: int = 7) -> pd.DataFrame:
    # The original build reads +/-9 days for its +/-7-day centered service
    # rule. A longer sensitivity must request enough extra source history.
    extra_days = max(9, service_days + 2)
    first = t0 - pd.Timedelta(days=extra_days)
    last = tlast + pd.Timedelta(days=extra_days)
    annual = [re.fullmatch(r"eaglei_outages_(\d{4})\.parquet", p.name) for p in paths]
    if all(annual):
        supplied_years = {int(m.group(1)) for m in annual if m}
        needed_years = set(range(first.year, last.year + 1))
        if not needed_years.issubset(supplied_years):
            raise ValueError(f"Source window crosses annual files; add years {sorted(needed_years - supplied_years)} to --raw")
    try:
        chunks = [pd.read_parquet(
            path,
            columns=["fips", "ts", "customers_out"],
            filters=[("ts", ">=", first), ("ts", "<=", last)],
        ) for path in paths]
    except ImportError as exc:
        raise RuntimeError("Reading the real parquet requires pyarrow or fastparquet") from exc
    return pd.concat(chunks, ignore_index=True)


def _ratio_mean(q_values: np.ndarray, mask15: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    p15 = np.clip(q_values / denominator[:, None], 0, 1)
    p15 = np.where(mask15, p15, 0.0).reshape(len(denominator), HOURS, QUARTERS_PER_HOUR)
    n = mask15.reshape(len(denominator), HOURS, QUARTERS_PER_HOUR).sum(axis=-1)
    out = np.full((len(denominator), HOURS), np.nan, dtype=float)
    np.divide(p15.sum(axis=-1), n, out=out, where=n > 0)
    return out


def _finite_mean(a: np.ndarray) -> float | None:
    return float(np.mean(a)) if a.size else None


def _rmse_zero(a: np.ndarray) -> float | None:
    return float(np.sqrt(np.mean(np.square(a)))) if a.size else None


def audit(raw: pd.DataFrame, panel: dict, locf_quarters: int = 4,
          min_global_rows: int = 5, service_days: int = 7,
          service_mode: str = "centered") -> dict:
    """Compare source-row evidence with one immutable exported panel.

    Reconstructed masks are *assumptions*, not measurement of scraper health.
    Label contrasts use only county-hours valid under A, B, and the saved mask.
    """
    if locf_quarters < 0 or min_global_rows < 1 or service_days < 0:
        raise ValueError("locf_quarters/service_days must be nonnegative; min_global_rows >= 1")
    if service_mode not in {"centered", "past_only"}:
        raise ValueError("service_mode must be centered or past_only")
    required = {"fips", "ts", "customers_out"}
    if not required.issubset(raw.columns):
        raise ValueError(f"raw parquet missing columns: {sorted(required - set(raw.columns))}")
    df = raw.loc[:, ["fips", "ts", "customers_out"]].copy()
    df["ts"] = _normalize_time(df["ts"])
    df["fips"] = df["fips"].astype("string").str.strip().str.zfill(5)
    df["customers_out"] = pd.to_numeric(df["customers_out"], errors="coerce")
    n = len(panel["fips"])
    t0 = panel["ts"][0]
    tlast = panel["ts"][-1] + pd.Timedelta(minutes=45)
    grid = pd.date_range(t0, periods=HOURS * QUARTERS_PER_HOUR, freq="15min")
    sample = df[df["fips"].isin(set(panel["fips"]))].copy()
    in_window = sample[(sample.ts >= t0) & (sample.ts <= tlast)].copy()
    duplicated = in_window.duplicated(["fips", "ts"], keep=False)
    conflicting_pairs = int(in_window.loc[duplicated].groupby(["fips", "ts"], dropna=False)
                            .customers_out.nunique(dropna=False).gt(1).sum())
    conflict_keys = in_window.loc[duplicated].groupby(["fips", "ts"], dropna=False) \
        .customers_out.nunique(dropna=False).loc[lambda x: x > 1].index
    duplicate_extra_rows = int(in_window.duplicated(["fips", "ts"]).sum())
    bad_count_rows = int((in_window.customers_out.isna() | (in_window.customers_out < 0)).sum())
    den_by_fips = dict(zip(panel["fips"], panel["denominator"]))
    raw_den = in_window.fips.map(den_by_fips).astype(float)
    above_denominator = int((in_window.customers_out > raw_den).sum())
    outside_grid_rows = int((~in_window.ts.isin(grid)).sum())

    # Match the source panel's last-row assignment for duplicate county/times.
    unique = in_window[in_window.ts.isin(grid)].drop_duplicates(["fips", "ts"], keep="last")
    ci = pd.Index(panel["fips"]).get_indexer(unique["fips"])
    qi = grid.get_indexer(unique["ts"])
    raw_present = np.zeros((n, len(grid)), dtype=bool)
    counts = np.zeros((n, len(grid)), dtype=float)
    raw_present[ci, qi] = True
    counts[ci, qi] = unique.customers_out.to_numpy(float)
    raw_positive = raw_present & np.isfinite(counts) & (counts > 0)
    explicit_nonpositive = raw_present & ~raw_positive
    quarantined = raw_present & (~np.isfinite(counts) | (counts < 0) |
                                 (counts > panel["denominator"][:, None]))
    for fips, ts in conflict_keys:
        c = pd.Index(panel["fips"]).get_indexer([fips])[0]
        q = grid.get_indexer([ts])[0]
        if c >= 0 and q >= 0:
            quarantined[c, q] = True
    valid_source = raw_present & ~quarantined
    valid_positive = raw_positive & valid_source
    counts = np.nan_to_num(counts, nan=0.0)

    # The existing code groups *national source rows* by timestamp (duplicates
    # included). This is a proxy for a collection run, not verified coverage.
    around = df[(df.ts >= t0 - pd.Timedelta(days=1)) &
                (df.ts <= tlast + pd.Timedelta(days=1))]
    rows_at = around.groupby("ts").size().reindex(grid, fill_value=0)
    global_run_proxy = (rows_at.to_numpy() >= min_global_rows)

    # The existing county-service proxy is any row within +/- service_days of
    # the day. It uses future records, and cannot establish actual observation.
    days = pd.date_range(t0.floor("D") - pd.Timedelta(days=service_days + 1),
                         tlast.floor("D") + pd.Timedelta(days=service_days + 1), freq="D")
    daily = np.zeros((n, len(days)), dtype=bool)
    svc_source = sample[sample.ts.dt.floor("D").isin(days)]
    if len(svc_source):
        dc = pd.Index(panel["fips"]).get_indexer(svc_source["fips"])
        dd = days.get_indexer(svc_source.ts.dt.floor("D"))
        daily[dc, dd] = True
    day_pos = days.get_indexer(grid.floor("D"))
    centered_service = np.empty((n, len(grid)), dtype=bool)
    for d in np.unique(day_pos):
        available = daily[:, max(0, d - service_days):min(len(days), d + service_days + 1)].any(axis=1)
        centered_service[:, day_pos == d] = available[:, None]
    service = centered_service.copy()
    if service_mode == "past_only":
        target_times = grid.to_numpy(dtype="datetime64[ns]")
        for i, fips in enumerate(panel["fips"]):
            past = np.sort(sample.loc[sample.fips == fips, "ts"].to_numpy(dtype="datetime64[ns]"))
            if not len(past):
                service[i] = False
                continue
            prev = np.searchsorted(past, target_times, side="right") - 1
            elapsed = target_times - past[np.maximum(prev, 0)]
            service[i] = (prev >= 0) & (elapsed >= np.timedelta64(0, "ns")) & (
                elapsed <= np.timedelta64(service_days, "D"))
    inferred_mask15 = service & global_run_proxy[None, :]
    inferred_mask_hour = inferred_mask15.reshape(n, HOURS, QUARTERS_PER_HOUR).any(axis=-1)
    saved_mask_hour = panel["observed"]
    matched = inferred_mask_hour & saved_mask_hour
    observed_disagreement = int(np.count_nonzero(inferred_mask_hour != saved_mask_hour))

    panel_rebuilt_y = _ratio_mean(counts, inferred_mask15, panel["denominator"])
    deviations = np.abs(panel_rebuilt_y[matched] - panel["y"][matched])
    value_mismatch = int(np.count_nonzero(deviations > 1e-5))
    max_value_diff = float(deviations.max()) if deviations.size else None

    # Positive-only support is a selected set, not a missingness correction.
    used_positive = raw_positive & inferred_mask15
    positive_hour = used_positive.reshape(n, HOURS, QUARTERS_PER_HOUR).any(axis=-1)
    any_source_row_on_mask = (raw_present & inferred_mask15).reshape(
        n, HOURS, QUARTERS_PER_HOUR).any(axis=-1)
    all_assumed_zero_hour = matched & ~any_source_row_on_mask
    no_source_positive_hour = matched & ~positive_hour
    some_assumed_zero_hour = matched & (inferred_mask15 & ~raw_present).reshape(
        n, HOURS, QUARTERS_PER_HOUR).any(axis=-1)
    positive_q = np.where(used_positive, np.clip(counts / panel["denominator"][:, None], 0, 1), 0)
    positive_q = positive_q.reshape(n, HOURS, QUARTERS_PER_HOUR)
    pos_n = used_positive.reshape(n, HOURS, QUARTERS_PER_HOUR).sum(axis=-1)
    positive_only_y = np.full((n, HOURS), np.nan)
    np.divide(positive_q.sum(axis=-1), pos_n, out=positive_only_y, where=pos_n > 0)

    # B retains every valid on-grid source row, independently of A's inferred
    # run/service mask. Conflicting, invalid and above-denominator rows are
    # quarantined. A proxy-no-run slot interrupts carry but cannot erase a
    # valid source row itself. Carry is a separate assumption: it requires a
    # national run, stops at the forecast origin, and never overwrites a row.
    carried_counts = np.zeros_like(counts)
    for i in range(n):
        last_positive_q = -locf_quarters - 1
        last_positive_value = 0.0
        for q in range(len(grid)):
            if q == ORIGIN * QUARTERS_PER_HOUR:
                last_positive_q = -locf_quarters - 1
                last_positive_value = 0.0
            if not global_run_proxy[q]:
                last_positive_q = -locf_quarters - 1
                last_positive_value = 0.0
            elif valid_positive[i, q]:
                last_positive_q = q
                last_positive_value = counts[i, q]
            elif raw_present[i, q]:
                last_positive_q = -locf_quarters - 1
                last_positive_value = 0.0
            elif q - last_positive_q <= locf_quarters:
                carried_counts[i, q] = last_positive_value
    # A supplied raw row or a short carried positive is evidence under B;
    # otherwise the quarter is unknown, including long recovery gaps.
    locf_mask15 = valid_source | (carried_counts > 0)
    locf_y = _ratio_mean(counts + carried_counts, locf_mask15, panel["denominator"])
    locf_valid_hour = locf_mask15.reshape(n, HOURS, QUARTERS_PER_HOUR).any(axis=-1)
    locf_changed_hour = (carried_counts > 0).reshape(n, HOURS, QUARTERS_PER_HOUR).any(axis=-1)

    def segment(start: int, stop: int) -> dict:
        s = np.s_[:, start:stop]
        q = np.s_[:, start * QUARTERS_PER_HOUR:stop * QUARTERS_PER_HOUR]
        a_support, orig, alt = matched[s], panel["y"][s], locf_y[s]
        m = a_support & locf_valid_hour[s]
        positive_support = a_support & positive_hour[s]
        change = (alt - orig)[m]
        return {
            "candidate_county_hours": int(n * (stop - start)),
            "saved_observed_hours": int(saved_mask_hour[s].sum()),
            "reconstructed_proxy_observed_hours": int(inferred_mask_hour[s].sum()),
            "saved_and_reconstructed_A_hours": int(a_support.sum()),
            "B_valid_hours": int(locf_valid_hour[s].sum()),
            "B_valid_hours_outside_saved_A": int((locf_valid_hour[s] & ~saved_mask_hour[s]).sum()),
            "matched_support_hours_A_and_B": int(m.sum()),
            "mask_disagreement_hours": int((saved_mask_hour[s] != inferred_mask_hour[s]).sum()),
            "source_positive_hours_on_matched_support": int(positive_support.sum()),
            "source_positive_quarters_on_proxy_mask": int(used_positive[q].sum()),
            "quarantined_source_quarters": int(quarantined[q].sum()),
            "hours_with_only_assumed_zero_quarters": int(all_assumed_zero_hour[s].sum()),
            "hours_without_source_positive_row": int(no_source_positive_hour[s].sum()),
            "hours_with_at_least_one_assumed_zero_quarter": int(some_assumed_zero_hour[s].sum()),
            "assumed_zero_quarters_on_proxy_mask": int((inferred_mask15[q] & ~raw_present[q]).sum()),
            "locf_changed_quarters": int((carried_counts[q] > 0).sum()),
            "locf_changed_hours_on_matched_support": int((locf_changed_hour[s] & m).sum()),
            "mean_panel_p_on_matched_support": _finite_mean(orig[m]),
            "mean_locf_scenario_p_on_matched_support": _finite_mean(alt[m]),
            "mean_locf_minus_panel_p_on_matched_support": _finite_mean(change),
            "mean_absolute_locf_minus_panel_p_on_matched_support": _finite_mean(np.abs(change)),
            "zero_forecast_rmse_on_panel_label": _rmse_zero(orig[m]),
            "zero_forecast_rmse_on_all_reconstructed_A_hours": _rmse_zero(orig[a_support]),
            "zero_forecast_rmse_on_locf_scenario": _rmse_zero(alt[m]),
            "positive_only_support_hours": int(positive_support.sum()),
            "positive_only_mean_p_selected_support": _finite_mean(positive_only_y[s][positive_support]),
            "zero_forecast_rmse_positive_only_selected_support": _rmse_zero(
                positive_only_y[s][positive_support]),
        }

    return {
        "event": panel["event"],
        "county_count": n,
        "start_utc": str(t0),
        "end_utc": str(panel["ts"][-1]),
        "definitions": {
            "source_positive": "positive customers_out row at a county and 15-minute timestamp",
            "assumed_zero": "no source row where the existing run/service proxy marks a quarter observed",
            "global_run_proxy": f"at least {min_global_rows} national source rows at a timestamp; not a verified collection log",
            "county_service_proxy": (
                f"any source row within +/-{service_days} calendar days, including later rows"
                if service_mode == "centered" else
                f"any source row in the preceding {service_days} days, ending at this slot"
            ) + "; not verified county coverage",
            "locf_scenario": f"B uses valid source rows regardless of A's proxy mask and positive values carried at most {locf_quarters} subsequent 15-minute slots without bridging a global no-run proxy or the forecast origin; conflicting/invalid rows are quarantined; longer gaps are unknown, not zero; B's observed hours are selected toward positive reports",
            "positive_only": "selects hours with positive source records; biased evaluation subset, not an alternative truth",
            "sensitivity_support": "A/B numerical contrasts use only county-hours valid under both target masks and the saved panel; A proxy counts use the saved-and-reconstructed A mask",
        },
        "input_checks": {
            "raw_rows_in_filtered_input": int(len(df)),
            "distinct_raw_fips_in_filtered_input": int(df.fips.nunique()),
            "source_positive_quarters_within_panel_counties_window": int(raw_positive.sum()),
            "source_positive_quarters_excluded_by_proxy_mask": int((raw_positive & ~inferred_mask15).sum()),
            "explicit_nonpositive_source_quarters": int(explicit_nonpositive.sum()),
            "above_panel_denominator_source_rows": above_denominator,
            "centered_service_quarters_without_past_service_evidence": (
                int((centered_service & ~service).sum()) if service_mode == "past_only" else None
            ),
            "duplicate_extra_rows_within_panel_counties_window": duplicate_extra_rows,
            "duplicate_conflicting_count_pairs": conflicting_pairs,
            "invalid_source_counts_within_panel_counties_window": bad_count_rows,
            "off_grid_source_rows_within_panel_counties_window": outside_grid_rows,
            "requires_source_quality_review": bool(
                conflicting_pairs or bad_count_rows or above_denominator or outside_grid_rows
            ),
            "global_no_run_proxy_quarters": int((~global_run_proxy).sum()),
            "reconstructed_vs_saved_mask_disagreement_hours": observed_disagreement,
            "reconstructed_vs_saved_label_difference_gt_1e_5_hours": value_mismatch,
            "maximum_reconstructed_vs_saved_label_difference": max_value_diff,
            "label_reconciliation_ok": observed_disagreement == 0 and value_mismatch == 0,
            "warning": (
                "Input may be county-filtered or panel built with other source data/rules; "
                "sensitivity is restricted to matched hours and should not be used as a full-panel comparison"
                if observed_disagreement or value_mismatch else None
            ),
        },
        "prefix_0_71h": segment(0, ORIGIN),
        "lead_segments": {key: segment(*bounds) for key, bounds in LEADS.items()},
    }


def _self_test() -> None:
    t0 = pd.Timestamp("2024-01-01T00:00:00")
    grid = pd.date_range(t0, periods=HOURS * QUARTERS_PER_HOUR, freq="15min")
    hours = pd.date_range(t0, periods=HOURS, freq="h")
    rows = []
    # Five background counties let the test distinguish global-run proxy status
    # from evidence about either panel county. Two absent collection quarters:
    # one inside an otherwise observed hour, one full hour.
    absent = {150 * 4 + 1, *(200 * 4 + x for x in range(4))}
    for q, t in enumerate(grid):
        if q not in absent:
            rows.extend((f"9{i:04d}", t, 1) for i in range(5))
    rows.extend([
        ("00001", grid[72 * 4], 40),
        ("00001", grid[72 * 4 + 2], 20),
        ("00001", grid[73 * 4], 0),  # explicit zero resets forward carry
        ("00002", grid[100 * 4], 250),  # quarter-level ratio is capped at one
        ("00002", grid[ORIGIN * 4 - 1], 20),  # carry must not cross forecast origin
        ("00002", grid[200 * 4], 50),  # actual row, but below global-run proxy threshold
        ("00002", grid[72 * 4] + pd.Timedelta(minutes=7), 5),  # off-grid row
    ])
    raw = pd.DataFrame(rows, columns=["fips", "ts", "customers_out"])
    y = np.zeros((2, HOURS), dtype=np.float32)
    y[0, 72] = (0.4 + 0.2) / 4
    y[1, ORIGIN - 1] = (20 / 200) / 4
    y[1, 100] = 1 / 4
    obs = np.ones_like(y, dtype=bool)
    obs[:, 200] = False
    y[:, 200] = np.nan
    with tempfile.TemporaryDirectory(prefix="measurement-audit-") as d:
        f = Path(d) / "panel216_synthetic.npz"
        np.savez_compressed(f, fips=np.array(["00001", "00002"]), ts=hours.astype(str),
                            y=y, observed=obs, denominator=np.array([100, 200]), event="synthetic")
        panel = _read_panel(f)
        result = audit(raw, panel, locf_quarters=1)
        past_only = audit(raw, panel, locf_quarters=1,
                          service_mode="past_only", service_days=1)
        no_carry = audit(raw, panel, locf_quarters=0)
        raw_duplicate = pd.concat([pd.DataFrame(
            [("00001", grid[72 * 4], 41)], columns=raw.columns), raw], ignore_index=True)
        duplicate_check = audit(raw_duplicate, panel, locf_quarters=1)
        raw_at_gap = pd.concat([raw, pd.DataFrame(
            [("00001", grid[150 * 4], 40)], columns=raw.columns)], ignore_index=True)
        panel_at_gap = {**panel, "y": panel["y"].copy()}
        panel_at_gap["y"][0, 150] = (40 / 100) / 3
        gap_with_carry = audit(raw_at_gap, panel_at_gap, locf_quarters=4)
        edited_mask = {**panel, "observed": panel["observed"].copy()}
        edited_mask["observed"][0, 150] = False
        mismatch = audit(raw, edited_mask, locf_quarters=1)
    checks = result["input_checks"]
    assert checks["global_no_run_proxy_quarters"] == 5, checks
    assert checks["label_reconciliation_ok"], checks
    assert checks["source_positive_quarters_within_panel_counties_window"] == 5, checks
    assert checks["source_positive_quarters_excluded_by_proxy_mask"] == 1, checks
    assert checks["above_panel_denominator_source_rows"] == 1, checks
    assert result["lead_segments"]["25_48h"]["quarantined_source_quarters"] == 1
    assert checks["off_grid_source_rows_within_panel_counties_window"] == 1, checks
    assert no_carry["lead_segments"]["full_1_144h"]["locf_changed_quarters"] == 0
    assert duplicate_check["input_checks"]["duplicate_conflicting_count_pairs"] == 1
    assert duplicate_check["lead_segments"]["1_6h"]["quarantined_source_quarters"] == 1
    assert duplicate_check["input_checks"]["label_reconciliation_ok"]
    assert past_only["input_checks"]["centered_service_quarters_without_past_service_evidence"] > 0
    assert past_only["prefix_0_71h"]["reconstructed_proxy_observed_hours"] == 1
    assert gap_with_carry["input_checks"]["label_reconciliation_ok"]
    # Source at q0, a missing national run at q1: q2/q3 must not inherit q0.
    assert gap_with_carry["lead_segments"]["49_144h"]["locf_changed_quarters"] == 0
    try:
        _read_raw([Path("eaglei_outages_2021.parquet")],
                  pd.Timestamp("2021-12-08"), pd.Timestamp("2021-12-19"), service_days=14)
    except ValueError as exc:
        assert "2022" in str(exc)
    else:
        raise AssertionError("A cross-year service window must not silently truncate the raw source")
    assert mismatch["input_checks"]["reconstructed_vs_saved_mask_disagreement_hours"] == 1
    assert not mismatch["input_checks"]["label_reconciliation_ok"]
    first = result["lead_segments"]["1_6h"]
    assert first["saved_and_reconstructed_A_hours"] == 12, first
    assert first["matched_support_hours_A_and_B"] == 2, first
    assert result["prefix_0_71h"]["B_valid_hours"] > 0
    assert result["lead_segments"]["49_144h"]["B_valid_hours_outside_saved_A"] == 1
    assert first["source_positive_hours_on_matched_support"] == 1, first
    assert first["hours_with_only_assumed_zero_quarters"] == 10, first
    assert first["hours_without_source_positive_row"] == 11, first
    assert first["locf_changed_quarters"] == 2, first
    assert abs(first["mean_locf_minus_panel_p_on_matched_support"] - 0.15 / 2) < 1e-7, first
    full = result["lead_segments"]["full_1_144h"]
    assert full["saved_observed_hours"] == 2 * 143, full
    print("self-test passed: source positives, assumed zeros, partial/no-run hours, explicit zero, cap, LOCF and lead support")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, nargs="+",
                    help="National EAGLE-I yearly parquet(s), including adjoining years when service window crosses Jan 1")
    ap.add_argument("--panel", type=Path, help="Existing 216-hour panel216_<event>.npz")
    ap.add_argument("--out", type=Path, help="Output JSON path; otherwise print to stdout")
    ap.add_argument("--locf-quarters", type=int, default=4,
                    help="Maximum carry length in 15-minute slots (default 4 = one hour)")
    ap.add_argument("--service-days", type=int, default=7,
                    help="Source-proxy service window in days (original=7)")
    ap.add_argument("--service-mode", choices=["centered", "past_only"], default="centered",
                    help="Original centered daily service proxy or strictly past-only alternative")
    ap.add_argument("--hash-inputs", action="store_true",
                    help="SHA-256 both input files (may stream many GB for the national parquet)")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        _self_test()
        return
    if not a.raw or not a.panel:
        ap.error("--raw and --panel are required unless --self-test is used")
    panel = _read_panel(a.panel)
    raw = _read_raw(a.raw, panel["ts"][0],
                    panel["ts"][-1] + pd.Timedelta(minutes=45), a.service_days)
    out = audit(raw, panel, locf_quarters=a.locf_quarters,
                service_days=a.service_days, service_mode=a.service_mode)
    out["inputs"] = {"raw_parquet": [str(p) for p in a.raw], "panel_npz": str(a.panel),
                     "sha256_status": "computed" if a.hash_inputs else "not computed; consult source manifests"}
    if a.hash_inputs:
        raw_hashes = {}
        for path in a.raw:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            raw_hashes[str(path)] = digest.hexdigest()
        out["inputs"]["raw_sha256"] = raw_hashes
        digest = hashlib.sha256()
        with a.panel.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        out["inputs"]["panel_sha256"] = digest.hexdigest()
    encoded = json.dumps(out, indent=2, allow_nan=False) + "\n"
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(encoded)
        print(f"wrote {a.out}")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
