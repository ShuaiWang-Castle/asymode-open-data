"""D05 descriptive same-system/time matching; no predictive model is fitted.

Entry point: analyze_matches(panel, reference, anchors, bootstrap_draws=999).
The outcome-independent full risk set is the fixed clock 72,84,...,204. Extra
named (unit,time) anchors are explicitly outcome-selected descriptive cases.
Weather/context select controls; outcomes never select controls, except that
the separately reported state-adjusted design conditions on previous stock
within 0.02. Both designs require observed finite adjacent outcome hours.

Matching uses 108 strict-past weather-z mean/min/max features and six context
coordinates standardized on all D county-events without outcomes. These are
descriptive all-D reference distributions, not OOF prediction validation.
For each of three fixed calipers, three different counties are chosen among
eligible same-system/absolute-time candidates by combined RMS distance. Fewer
than three eligible candidates produces no match. Geography does not enter
selection. Missing geography only excludes the affected coordinate from its
reported contrast/slope, requiring that coordinate in focal and all controls.

The reported through-origin slopes are one-coordinate descriptive summaries
of geography/outage differences, not multivariate effects, fitted forecasts or
causal effects. Geography coordinates are correlated. Bootstrap resamples
merged event groups conditional on the already fixed matches and reference;
it excludes matching/reference selection uncertainty and cross-event county
dependence. It is not simultaneous inference for the 200 reported coordinates.
All five regimes and all 40 coordinates are retained regardless of results.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json

import numpy as np

from d05_weather_features import flatten_summaries, summarize_anchors

REGIMES = ("tropical", "winter", "synoptic_wind", "convective", "heavy_rain")
CLOCK = np.arange(72, 205, 12, dtype=np.int16)
CALIPERS = (0.5, 1.0, 2.0)
VERSIONS = ("unadjusted", "state_adjusted")
NEIGHBORS = 3
PRIMARY_CALIPER = 1.0
STATE_CALIPER = 0.02


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _valid_hours(panel, unit, time):
    y, observed = panel["y_full"], panel["obs_full"]
    time = np.asarray(time)
    inside = (time >= 1) & (time < y.shape[1])
    good = np.zeros(np.broadcast_shapes(np.shape(unit), time.shape), dtype=bool)
    uu, tt = np.broadcast_arrays(unit, time)
    valid = np.broadcast_to(inside, good.shape)
    good[valid] = (observed[uu[valid], tt[valid]] & observed[uu[valid], tt[valid] - 1] &
                   np.isfinite(y[uu[valid], tt[valid]]) & np.isfinite(y[uu[valid], tt[valid] - 1]))
    return good


def _validate(panel, anchors, bootstrap_draws):
    weather, geo = np.asarray(panel["weather"]), np.asarray(panel["geo"])
    y, obs, context = np.asarray(panel["y_full"]), np.asarray(panel["obs_full"]), np.asarray(panel["context"])
    n, hours = weather.shape[:2]
    if geo.shape != (n, 40) or context.shape != (n, 6) or y.shape != (n, hours) or obs.shape != y.shape:
        raise ValueError("D05 panel dimensions disagree")
    if hours <= CLOCK[-1] or obs.dtype != bool:
        raise ValueError("The full risk clock needs hours through 204 and a boolean observation mask")
    if not isinstance(bootstrap_draws, int) or bootstrap_draws < 0:
        raise ValueError("bootstrap_draws must be a nonnegative integer")
    meta = panel["meta"]
    for key in ("county", "system", "origin", "regime", "w"):
        if np.asarray(meta[key]).shape != (n,):
            raise ValueError(f"Invalid metadata {key}")
    if np.asarray(panel["merged_group"]).shape != (n,):
        raise ValueError("Missing merged event groups")
    if np.any(~np.isfinite(meta["w"])) or np.any(np.asarray(meta["w"]) <= 0):
        raise ValueError("Positive finite design weights are required")
    origin = np.asarray(meta["origin"]).astype("datetime64[s]")
    if np.isnat(origin).any() or np.any(origin != origin.astype("datetime64[h]")):
        raise ValueError("Origins must be finite whole-hour timestamps")
    system = np.asarray(meta["system"]).astype(str)
    county = np.asarray(meta["county"]).astype(str)
    regime = np.asarray(meta["regime"]).astype(str)
    if np.any(~np.isin(regime, REGIMES)):
        raise ValueError("Unexpected weather regime")
    for s in np.unique(system):
        rows = system == s
        if len(np.unique(origin[rows])) != 1:
            raise ValueError("Same-system origins differ: panel-hour matching would not be absolute-time matching")
        if len(np.unique(regime[rows])) != 1 or len(np.unique(np.asarray(panel["merged_group"])[rows])) != 1:
            raise ValueError("Regime or merged group varies within a system")
        if len(np.unique(county[rows])) != rows.sum():
            raise ValueError("A county occurs more than once in one system")
    if not isinstance(anchors, dict) or "full_risk_clock" in anchors:
        raise ValueError("anchors must be named cases, excluding reserved full_risk_clock")
    cases = {}
    for name, pair in anchors.items():
        if not isinstance(name, str) or not name or len(pair) != 2:
            raise ValueError("Each named anchor is (unit, time)")
        u, t = map(np.asarray, pair)
        if u.ndim != 1 or t.shape != u.shape or not np.issubdtype(u.dtype, np.integer) or not np.issubdtype(t.dtype, np.integer):
            raise ValueError("Case units/times must be one-dimensional integer vectors")
        if np.any((u < 0) | (u >= n)) or np.any((t < 0) | (t >= hours)):
            raise ValueError("Case anchors outside panel")
        if len(np.unique(u)) != len(u):
            raise ValueError("Each case definition can select at most one anchor per county-event unit")
        cases[name] = (u.astype(np.int32), t.astype(np.int16))
    return system, county, regime, origin, cases


def _rms_matrix(left, right):
    """Squared Euclidean identities avoid [focal,candidate,feature] tensors."""
    a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    square = (a * a).sum(axis=1)[:, None] + (b * b).sum(axis=1)[None, :] - 2 * (a @ b.T)
    return np.sqrt(np.maximum(square, 0) / a.shape[1])


class _GroupCache:
    """Bounded LRU of one system/hour's compact 108-coordinate summaries."""
    def __init__(self, panel, reference, systems, context_z, max_bytes=32 * 1024 * 1024):
        self.panel, self.reference, self.systems, self.context_z = panel, reference, systems, context_z
        self.system_units = {s: np.flatnonzero(systems == s) for s in np.unique(systems)}
        self.items = OrderedDict(); self.size = 0; self.max_bytes = max_bytes
        self.hits = 0; self.misses = 0

    def get(self, system, time):
        key = (str(system), int(time))
        if key in self.items:
            self.hits += 1
            self.items.move_to_end(key)
            return self.items[key]
        self.misses += 1
        u = self.system_units[system]
        t = np.full(len(u), time, dtype=np.int16)
        summary = summarize_anchors(self.panel, u, t, self.reference, include_pairs=False, chunk=256)
        x = flatten_summaries(summary, blocks=("standardized",), bands=(0, 1, 2))["feature_matrix"]
        c = self.context_z[u]
        valid = _valid_hours(self.panel, u, t) & np.isfinite(x).all(axis=1) & np.isfinite(c).all(axis=1)
        entry = {"unit": u, "weather": x, "context": c, "valid": valid,
                 "bytes": x.nbytes + c.nbytes + valid.nbytes + u.nbytes}
        while self.items and self.size + entry["bytes"] > self.max_bytes:
            _, old = self.items.popitem(last=False); self.size -= old["bytes"]
        # One exceptionally large group may exceed the cache allowance; do not retain it.
        if entry["bytes"] <= self.max_bytes:
            self.items[key] = entry; self.size += entry["bytes"]
        return entry


def _empty_record():
    return {"unit": [], "time": [], "weight": [], "neighbors": [],
            "weather_rms": [], "context_rms": [], "combined_rms": []}


def _join_record(record):
    result = {}
    for key, values in record.items():
        shape = (0, 3) if key in ("neighbors", "weather_rms", "context_rms", "combined_rms") else (0,)
        dtype = np.int32 if key in ("unit", "time", "neighbors") else np.float64
        result[key] = np.concatenate(values, axis=0) if values else np.empty(shape, dtype=dtype)
    return result


def _quantiles(x):
    return np.quantile(x, [0, .25, .5, .75, .9, .95, 1]) if len(x) else np.full(7, np.nan)


def _weight_stats(w):
    w = np.asarray(w, dtype=np.float64)
    total = w.sum()
    return {"weight_sum": total, "kish": total * total / (w @ w) if total > 0 else 0,
            "max_share": w.max() / total if total > 0 else None}


def _support(record, panel, focal_unit, focal_weights, candidate_counts, candidate_count_weights):
    u, t, neighbors, w = record["unit"], record["time"], record["neighbors"], record["weight"]
    meta = panel["meta"]
    selected, focal_weight = len(focal_unit), np.sum(focal_weights)
    observation_ids = neighbors.astype(np.int64).ravel() * panel["y_full"].shape[1] + np.repeat(t, 3)
    _, reuse = np.unique(observation_ids, return_counts=True)
    neighbor_units, unit_reuse = np.unique(neighbors, return_counts=True)
    neighbor_counties, county_reuse = np.unique(np.asarray(meta["county"])[neighbors].ravel(), return_counts=True)
    # Directed edge identities are hashed for reproducibility without serializing a giant graph.
    edge = np.column_stack([np.repeat(u, 3), np.repeat(t, 3), neighbors.ravel()]).astype("<i8")
    digest = hashlib.sha256(edge.tobytes()).hexdigest()
    out = {
        "requested_focals": int(selected), "matched_focals": len(u),
        "requested_weight": float(focal_weight), "matched_weight": float(w.sum()),
        "matched_weight_fraction": w.sum() / focal_weight if focal_weight > 0 else None,
        "matched_county_events": len(np.unique(u)), "matched_counties": len(np.unique(np.asarray(meta["county"])[u])),
        "systems": len(np.unique(np.asarray(meta["system"])[u])),
        "merged_groups": len(np.unique(np.asarray(panel["merged_group"])[u])),
        "focal_weight_concentration": _weight_stats(w),
        "candidate_count_quantiles_before_three_neighbor_requirement": _quantiles(candidate_counts),
        "candidate_count_observed_focals": len(candidate_counts),
        "candidate_count_observed_weight": sum(candidate_count_weights),
        "distance_quantile_probabilities": [0, .25, .5, .75, .9, .95, 1],
        "distance_quantiles": {key: _quantiles(record[key].ravel()) for key in ("weather_rms", "context_rms", "combined_rms")},
        "reuse": {"directed_edges": int(neighbors.size), "unique_control_observations": len(reuse),
                 "control_observation_use_quantiles": _quantiles(reuse),
                 "max_control_observation_uses": int(reuse.max()) if len(reuse) else 0,
                 "unique_control_county_events": len(neighbor_units),
                 "max_control_county_event_uses": int(unit_reuse.max()) if len(unit_reuse) else 0,
                 "unique_control_counties": len(neighbor_counties),
                 "max_control_county_uses": int(county_reuse.max()) if len(county_reuse) else 0,
                 "control_observation_kish_from_use_counts": _weight_stats(reuse)["kish"]},
        "topology": {"directed": True, "out_degree_each_matched_focal": 3,
                     "same_county_edges": int((np.asarray(meta["county"])[u, None] == np.asarray(meta["county"])[neighbors]).sum()),
                     "same_system_same_absolute_time": True, "matched_edge_sha256": digest,
                     "repeated_counties_across_systems_preserved": True},
    }
    out["by_regime"] = {}
    focal_regime = np.asarray(meta["regime"])[u]
    for regime in REGIMES:
        mask = focal_regime == regime
        requested = np.asarray(meta["regime"])[focal_unit] == regime
        requested_weight = focal_weights[requested].sum()
        out["by_regime"][regime] = {"requested_focals": int(requested.sum()),
            "requested_weight": requested_weight,
            "matched_focals": int(mask.sum()), "matched_weight": w[mask].sum(),
            "matched_weight_fraction": w[mask].sum() / requested_weight if requested_weight > 0 else None,
            "systems": len(np.unique(np.asarray(meta["system"])[u[mask]])),
            "merged_groups": len(np.unique(np.asarray(panel["merged_group"])[u[mask]]))}
    return out


def _contrasts(record, panel, reference, draws, seed):
    u, t, nn, w = record["unit"], record["time"], record["neighbors"], record["weight"]
    y, meta = panel["y_full"], panel["meta"]
    delta = y[u, t].astype(np.float64) - y[u, t - 1]
    control_delta = y[nn, t[:, None]].astype(np.float64) - y[nn, t[:, None] - 1]
    dy = delta - control_delta.mean(axis=1)
    ds = y[u, t] - y[nn, t[:, None]].mean(axis=1)
    dp = y[u, t - 1] - y[nn, t[:, None] - 1].mean(axis=1)
    geo = np.asarray(panel["geo"], dtype=np.float64)
    gz = (geo - reference["geo_mean"]) / reference["geo_scale"]
    good = np.isfinite(gz[u]) & np.isfinite(gz[nn]).all(axis=1)
    dx = gz[u] - gz[nn].mean(axis=1)
    dxraw = geo[u] - geo[nn].mean(axis=1)
    dx[~good] = np.nan; dxraw[~good] = np.nan
    regimes = np.asarray(meta["regime"])[u]
    rng = np.random.default_rng(seed)
    output = {}
    for regime in REGIMES:
        take = regimes == regime
        rr, ww, yy = u[take], w[take], dy[take]
        xx, xr, vv = dx[take], dxraw[take], good[take]
        groups, gi = np.unique(np.asarray(panel["merged_group"])[rr], return_inverse=True)
        systems = np.asarray(meta["system"])[rr]
        counties = np.asarray(meta["county"])[rr]
        group_w = np.zeros((len(groups), 40)); group_num = np.zeros_like(group_w); group_den = np.zeros_like(group_w)
        valid_w = ww[:, None] * vv
        xzero = np.where(vv, xx, 0)
        np.add.at(group_w, gi, valid_w)
        np.add.at(group_num, gi, valid_w * xzero * yy[:, None])
        np.add.at(group_den, gi, valid_w * xzero * xzero)
        numerator, denominator = group_num.sum(axis=0), group_den.sum(axis=0)
        slopes = np.divide(numerator, denominator, out=np.full(40, np.nan), where=denominator > 0)
        intervals = np.full((40, 2), np.nan); valid_draws = np.zeros(40, dtype=np.int32)
        if draws and len(groups) >= 2:
            multiplicities = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=draws)
            bn, bd = multiplicities @ group_num, multiplicities @ group_den
            boot = np.divide(bn, bd, out=np.full_like(bn, np.nan), where=bd > 0)
            for j in range(40):
                valid_boot = np.isfinite(boot[:, j]); valid_draws[j] = valid_boot.sum()
                # CI requires at least two observed event groups for this coordinate.
                if np.count_nonzero(group_w[:, j]) >= 2 and valid_boot.any():
                    intervals[j] = np.quantile(boot[valid_boot, j], [.025, .975])
        total = ww.sum()
        means = {name: float(ww @ values[take] / total) if total > 0 else None
                 for name, values in (("delta_fraction_difference", dy), ("stock_fraction_difference", ds),
                                      ("previous_stock_fraction_difference", dp))}
        rows = []
        for j, name in enumerate(panel["feature_names"]["geo"]):
            valid = vv[:, j]; vweight = ww[valid]; vtotal = vweight.sum()
            gw = group_w[:, j]
            _, ci = np.unique(counties[valid], return_inverse=True)
            county_weights = np.bincount(ci, weights=vweight)
            rows.append({"geography": name, "slope_delta_fraction_per_geo_reference_sd": slopes[j],
                         "bootstrap_ci95": intervals[j], "bootstrap_valid_draws": int(valid_draws[j]),
                         "matched_focals_with_coordinate": int(valid.sum()), "coordinate_weight": vtotal,
                         "systems": len(np.unique(systems[valid])), "merged_groups": int(np.count_nonzero(gw)),
                         "focal_weight_concentration": _weight_stats(vweight),
                         "merged_group_weight_concentration": _weight_stats(gw),
                         "county_weight_concentration": _weight_stats(county_weights),
                         "merged_group_slope_information_concentration": _weight_stats(group_den[:, j]),
                         "weighted_x_squared": denominator[j],
                         "mean_geo_z_difference": float(vweight @ xx[valid, j] / vtotal) if vtotal > 0 else None,
                         "mean_geo_raw_difference": float(vweight @ xr[valid, j] / vtotal) if vtotal > 0 else None,
                         "mean_delta_fraction_difference": float(vweight @ yy[valid] / vtotal) if vtotal > 0 else None})
        output[regime] = {"matched_focals": int(take.sum()), "matched_weight": total,
                          "systems": len(np.unique(systems)), "merged_groups": len(groups),
                          "mean_outcome_contrasts": means, "all_40_geography": rows}
    return output


def analyze_matches(panel, reference, anchors, bootstrap_draws=999):
    """Return JSON-safe support for all calipers and full results at caliper 1.

    ``anchors`` maps names such as Jmax/Smax to (unit,time) integer vectors with
    at most one anchor per county-event for each name. Full risk focals have
    unit design weight divided by their observation-valid clock count; case
    focals have that unit's full design weight. Failed matching does not
    redistribute excluded mass. Neighbors are equally weighted within a set.
    """
    systems, counties, regimes, origins, cases = _validate(panel, anchors, bootstrap_draws)
    n, hours = panel["y_full"].shape
    context = np.asarray(panel["context"], dtype=np.float64)
    finite = np.isfinite(context); count = finite.sum(axis=0)
    if np.any(count == 0):
        raise ValueError("Every context column requires a finite reference observation")
    cm = np.where(finite, context, 0).sum(axis=0) / count
    cs = np.sqrt(np.square(np.where(finite, context - cm, 0)).sum(axis=0) / count)
    cs[cs <= 1e-7] = 1
    cz = (context - cm) / cs
    cache = _GroupCache(panel, reference, systems, cz)
    clock_valid = _valid_hours(panel, np.arange(n)[:, None], CLOCK[None, :])
    clock_counts = clock_valid.sum(axis=1)
    risk_unit, clock_column = np.nonzero(clock_valid)
    risk_time = CLOCK[clock_column]
    design_w = np.asarray(panel["meta"]["w"], dtype=np.float64)
    sets = {"full_risk_clock": (risk_unit, risk_time), **cases}
    result = {"meta": {
        "analysis": "D05 fixed descriptive weather/context matching and geography contrasts",
        "reference_scope": reference["scope"], "prediction_validation": False,
        "weather_matching_coordinates": 108, "context_matching_coordinates": 6,
        "weather_features": "strict-past bands -48:-25,-24:-7,-6:-1, all 12 channel mean/min/max",
        "context_reference": "equal full-D county-event rows, outcome-blind mean/SD, no imputation",
        "context_names": list(panel["feature_names"].get("context", [f"context_{j}" for j in range(6)])),
        "context_mean": cm, "context_scale": cs, "context_count": count,
        "neighbor_rule": "three eligible distinct other counties, combined RMS ascending, deterministic unit-index tie break",
        "combined_rms": "sqrt((weather_RMS_squared + context_RMS_squared)/2)",
        "calipers": list(CALIPERS), "context_rms_caliper": 1, "primary_caliper": PRIMARY_CALIPER,
        "state_adjusted_previous_stock_caliper": STATE_CALIPER,
        "absolute_time": "panel.meta.origin[unit] + (time-72) hours UTC; same-system origins asserted identical",
        "full_risk_clock": CLOCK, "case_definitions": list(cases),
        "risk_units_without_observation_valid_clock": int((clock_counts == 0).sum()),
        "geography_missing_rule": "coordinate is valid only when focal and all three controls have finite standardized geography; other coordinates retained",
        "bootstrap": {"draws": bootstrap_draws, "unit": "merged event group within regime", "base_seed": 20260929,
                      "seed_rule": "base_seed + uint32_little_endian(SHA256(anchor_set_name:version) first 4 bytes)",
                      "conditional_on_fixed_matches_reference_and_case_selection": True,
                      "cross_event_county_dependence_resampled": False, "simultaneous_intervals": False},
        "interpretation": "all 40 marginal through-origin weighted slopes; correlated geography, descriptive association, no independent or causal effects",
    }, "anchor_sets": {}}
    for name, (unit, time) in sets.items():
        unit, time = np.asarray(unit, dtype=np.int32), np.asarray(time, dtype=np.int16)
        outcome_selected = name != "full_risk_clock"
        weight = design_w[unit] if outcome_selected else design_w[unit] / clock_counts[unit]
        eligible_outcome = _valid_hours(panel, unit, time)
        records = {(v, cal): _empty_record() for v in VERSIONS for cal in CALIPERS}
        candidate_counts = {(v, cal): [] for v in VERSIONS for cal in CALIPERS}
        candidate_weights = {(v, cal): [] for v in VERSIONS for cal in CALIPERS}
        weather_invalid_focals = 0
        grouped = {}
        for i in np.flatnonzero(eligible_outcome):
            grouped.setdefault((systems[unit[i]], int(time[i])), []).append(i)
        for (system, hour), positions in sorted(grouped.items()):
            entry = cache.get(system, hour)
            cu = entry["unit"][entry["valid"]]
            if not len(cu):
                weather_invalid_focals += len(positions)
                continue
            full_pos = np.searchsorted(entry["unit"], unit[positions])
            good_focal = entry["valid"][full_pos]
            weather_invalid_focals += int((~good_focal).sum())
            positions = np.asarray(positions)[good_focal]
            full_pos = full_pos[good_focal]
            cw, cc = entry["weather"][entry["valid"]], entry["context"][entry["valid"]]
            for start in range(0, len(positions), 128):
                rows, fp = positions[start:start + 128], full_pos[start:start + 128]
                fu = unit[rows]
                wd = _rms_matrix(entry["weather"][fp], cw)
                cd = _rms_matrix(entry["context"][fp], cc)
                score = np.sqrt((wd * wd + cd * cd) / 2)
                other_county = counties[fu, None] != counties[cu][None, :]
                # Compare stored stock values in float64 so float32 subtraction
                # cannot round a just-outside difference into the fixed 0.02 band.
                pprev = np.asarray(panel["y_full"])[fu, hour - 1].astype(np.float64)
                cp = np.asarray(panel["y_full"])[cu, hour - 1].astype(np.float64)
                for version in VERSIONS:
                    admissible = other_county & (cd <= 1)
                    if version == "state_adjusted":
                        admissible &= np.abs(pprev[:, None] - cp[None, :]) <= STATE_CALIPER
                    for caliper in CALIPERS:
                        allowed = admissible & (wd <= caliper)
                        counts_here = allowed.sum(axis=1)
                        key = (version, caliper)
                        candidate_counts[key].extend(counts_here.tolist())
                        candidate_weights[key].extend(weight[rows].tolist())
                        selected_rows, selected_control = [], []
                        for q in np.flatnonzero(counts_here >= NEIGHBORS):
                            choices = np.flatnonzero(allowed[q])
                            order = np.lexsort((cu[choices], score[q, choices]))[:NEIGHBORS]
                            selected_rows.append(q); selected_control.append(choices[order])
                        if not selected_rows:
                            continue
                        qi, ni = np.asarray(selected_rows), np.asarray(selected_control)
                        record = records[key]
                        record["unit"].append(fu[qi]); record["time"].append(np.full(len(qi), hour, dtype=np.int16))
                        record["weight"].append(weight[rows[qi]]); record["neighbors"].append(cu[ni])
                        for field, distances in (("weather_rms", wd), ("context_rms", cd), ("combined_rms", score)):
                            record[field].append(distances[qi[:, None], ni].astype(np.float32))
        set_result = {"outcome_selected": outcome_selected, "requested_focals": len(unit),
                      "adjacent_outcome_invalid_focals": int((~eligible_outcome).sum()),
                      "weather_or_context_invalid_focals": weather_invalid_focals,
                      "focal_weight_rule": "unit w" if outcome_selected else "unit w / observation-valid fixed clock count",
                      "versions": {}}
        for version in VERSIONS:
            versions = {"caliper_support": {}}
            for caliper in CALIPERS:
                key = (version, caliper); record = _join_record(records.pop(key))
                versions["caliper_support"][str(caliper)] = _support(record, panel, unit, weight,
                    candidate_counts[key], candidate_weights[key])
                if caliper == PRIMARY_CALIPER:
                    # Matching is fully fixed before reading current outcome differences.
                    label_seed = int.from_bytes(hashlib.sha256(f"{name}:{version}".encode()).digest()[:4], "little")
                    versions["bootstrap_seed"] = 20260929 + label_seed
                    versions["primary_caliper_regime_geography"] = _contrasts(record, panel, reference, bootstrap_draws, 20260929 + label_seed)
            set_result["versions"][version] = versions
        result["anchor_sets"][name] = set_result
    result["meta"]["cache"] = {"max_bytes": cache.max_bytes, "hits": cache.hits, "misses": cache.misses,
                                  "distance_focal_block": 128, "all_pair_feature_tensor_materialized": False}
    return _clean(result)


def _self_test():
    from d05_weather_features import WEATHER_NAMES, fit_reference
    groups, per_group, hours = 10, 4, 216
    n = groups * per_group
    x = np.tile(np.arange(per_group), groups).astype(np.float32)
    geo = np.repeat(x[:, None], 40, axis=1)
    geo[0, 1] = np.nan
    panel = {"weather": np.zeros((n, hours, 12), dtype=np.float32), "geo": geo,
             "context": np.zeros((n, 6), dtype=np.float32),
             "y_full": .2 + x[:, None].astype(np.float64) * np.arange(hours)[None, :] * .00001,
             "obs_full": np.ones((n, hours), dtype=bool),
             "merged_group": np.repeat([f"g{i}" for i in range(groups)], per_group),
             "meta": {"county": np.array([f"c{i}" for i in range(n)]),
                      "system": np.repeat([f"s{i:02d}" for i in range(groups)], per_group),
                      "origin": np.repeat([f"2020-01-{i+1:02d}T00:00:00" for i in range(groups)], per_group),
                      "regime": np.repeat(np.tile(REGIMES, 2), per_group), "w": np.ones(n)},
             "feature_names": {"weather": list(WEATHER_NAMES), "geo": [f"g{i}" for i in range(40)]}}
    ref = fit_reference(panel)
    check_rng = np.random.default_rng(47)
    aa, bb = check_rng.normal(size=(4, 6)), check_rng.normal(size=(7, 6))
    np.testing.assert_allclose(_rms_matrix(aa, bb), np.sqrt(np.square(aa[:, None] - bb).mean(axis=2)), atol=1e-14)
    anchors = {"Jmax": (np.arange(n), np.full(n, 120, dtype=int))}
    result = analyze_matches(panel, ref, anchors, bootstrap_draws=19)
    json.dumps(result, allow_nan=False)
    risk = result["anchor_sets"]["full_risk_clock"]["versions"]
    for version in VERSIONS:
        for caliper in CALIPERS:
            support = risk[version]["caliper_support"][str(caliper)]
            assert support["matched_focals"] == n * len(CLOCK)
            assert support["topology"]["same_county_edges"] == 0
            np.testing.assert_allclose(support["matched_weight"], n)
        rows = risk[version]["primary_caliper_regime_geography"]["tropical"]["all_40_geography"]
        np.testing.assert_allclose(rows[0]["slope_delta_fraction_per_geo_reference_sd"], .00001 * ref["geo_scale"][0], atol=1e-13)
        assert rows[0]["merged_groups"] == 2 and rows[1]["matched_focals_with_coordinate"] < rows[0]["matched_focals_with_coordinate"]
        assert len(rows) == 40
    # Changing current outcomes cannot change unadjusted matching.
    altered = {**panel, "y_full": .3 + np.random.default_rng(4).uniform(-.1, .1, size=(n, hours))}
    changed = analyze_matches(altered, ref, anchors, bootstrap_draws=0)
    assert changed["anchor_sets"]["full_risk_clock"]["versions"]["state_adjusted"]["caliper_support"]["1.0"]["matched_focals"] < n * len(CLOCK)
    for name in result["anchor_sets"]:
        for caliper in CALIPERS:
            a = result["anchor_sets"][name]["versions"]["unadjusted"]["caliper_support"][str(caliper)]
            b = changed["anchor_sets"][name]["versions"]["unadjusted"]["caliper_support"][str(caliper)]
            assert a["topology"]["matched_edge_sha256"] == b["topology"]["matched_edge_sha256"]
    # Unobserved target or previous hour cannot be controls; valid-clock mass stays unit w.
    missing = {**panel, "obs_full": panel["obs_full"].copy()}
    missing["obs_full"][0, 72] = False
    excluded = analyze_matches(missing, ref, {}, bootstrap_draws=0)
    support = excluded["anchor_sets"]["full_risk_clock"]["versions"]["unadjusted"]["caliper_support"]["1.0"]
    assert support["requested_focals"] == n * len(CLOCK) - 1
    assert support["matched_focals"] == n * len(CLOCK) - per_group
    np.testing.assert_allclose(support["requested_weight"], n)
    invalid = {**panel, "meta": {**panel["meta"], "origin": panel["meta"]["origin"].copy()}}
    invalid["meta"]["origin"][0] = "2020-03-01T00:00:00"
    try:
        analyze_matches(invalid, ref, {}, bootstrap_draws=0)
    except ValueError as exc:
        assert "origins differ" in str(exc)
    else:
        raise AssertionError("Unequal absolute times accepted")
    print("D05 matching synthetic checks passed: full-risk weights, no same-county control, outcome-blind neighbors, coordinatewise missingness, known slope, absolute-time guard, finite JSON.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
    else:
        parser.error("Library only: --self-test performs synthetic checks")
