"""Outcome-blind weather/geography summaries for descriptive D05 event anchors.

This module neither loads a panel nor reads outage outcomes. The caller supplies
the public-D panel and the county-event ``unit`` and panel-hour ``time`` anchors.
Offset zero is weather at ``time``; the strict past is offsets -48 through -1.
Offsets zero and above are descriptive information, not forecast predictors.
The retained ERA5 precip/snowfall/gust value indexed T covers (T-1,T]; the
other retained channels are instantaneous at T. Outage stock indexed T is the
mean of quarter-hour observations at T,T+.25,T+.5,T+.75. Thus +1 accumulated
weather/gust covers the outage observation hour retrospectively; it is not
necessarily weather that physically begins after an outage. No index-only
summary identifies the ordering of subhourly damage and weather.

``fit_reference`` uses equal county-event-hour mass for weather. CAPE, precip,
and snowfall are log1p(max(value, 0)) transformed before z scaling, as in D04;
raw output is unchanged. Geography uses one row per unique county, with no
imputation, projection, or outcome-based selection. A full-D reference is valid
for the registered descriptive matching, but is not a prediction-validation
reference. Predictive use requires an explicit training reference and caller-
controlled training units/hours.

Incomplete windows produce NaN, with counts and masks; there is no boundary
padding or partial-window substitution. A band's min/max are numeric extrema,
not a common hazard direction: cold, low pressure and signed wind components
must not be interpreted as high-is-harmful. Only hourly precipitation/snowfall
are summed as accumulations. Pair summaries are standardized second moments,
not causal interactions, correlations, or independent observations. Cross-band
terms preserve coarse ordering; fixed delays additionally retain within-band
order. They do not exhaust higher-order or subhourly mechanisms. The weather
reference can count overlapping county-event windows more than once.

Public API: fit_reference, extract_geography, extract_curves,
summarize_anchors, flatten_summaries. All arrays are structured until explicitly
flattened. For large anchor sets, call on batches or omit pairs. No real-data
work runs on import or through the --self-test command.
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np

WEATHER_NAMES = (
    "cape", "cloud", "gust", "precip", "pressure", "rh", "snowfall",
    "soil_moisture", "t2m_c", "u10", "v10", "wind_speed",
)
LOG_CHANNELS = ("cape", "precip", "snowfall")
OFFSETS = np.arange(-48, 25, dtype=np.int16)
BANDS = ((-48, -25), (-24, -7), (-6, -1), (0, 0), (1, 6), (7, 24))
BAND_NAMES = tuple(f"offset_{lo}_{hi}" for lo, hi in BANDS)
SYNC_PAIRS = np.array(list(itertools.combinations_with_replacement(range(12), 2)), dtype=np.int16)
CROSS_PAIRS = np.array(list(itertools.combinations(range(12), 2)), dtype=np.int16)
BAND_PAIRS = np.array(list(itertools.combinations(range(6), 2)), dtype=np.int16)
DIRECTED_PAIRS = np.array(list(itertools.product(range(12), repeat=2)), dtype=np.int16)
DELAYS = (1, 6, 24)
ACCUMULATION_CHANNELS = ("precip", "snowfall")
NEAR_OFFSETS = np.array([-1, 0, 1], dtype=np.int16)
WEATHER_TIME_SUPPORT = {
    name: ("preceding_hour_(T-1,T]" if name in ("precip", "snowfall", "gust")
           else "instantaneous_at_T") for name in WEATHER_NAMES
}
ANCHOR_TIME_SUPPORT = "outage stock at T is mean of quarter-hour observations T,T+0.25,T+0.5,T+0.75"


def _schema(panel):
    wx = np.asarray(panel["weather"])
    geo = np.asarray(panel["geo"])
    names = panel["feature_names"]
    if wx.ndim != 3 or wx.shape[2] != 12 or wx.shape[1] < 1:
        raise ValueError("weather must have shape [unit, hour, 12]")
    if tuple(names["weather"]) != WEATHER_NAMES:
        raise ValueError("Expected the registered D04 weather channel order")
    if geo.shape != (wx.shape[0], 40) or len(names["geo"]) != 40:
        raise ValueError("All 40 original geography columns are required")
    if not np.issubdtype(wx.dtype, np.number) or not np.issubdtype(geo.dtype, np.number):
        raise ValueError("Weather and geography must be numeric")
    return wx, geo


def _integer_vector(values, name):
    out = np.asarray(values)
    if out.ndim != 1 or not np.issubdtype(out.dtype, np.integer):
        raise ValueError(f"{name} must be a one-dimensional integer array")
    return out.astype(np.int64, copy=False)


def _anchors(panel, unit, time):
    wx, _ = _schema(panel)
    u, t = _integer_vector(unit, "unit"), _integer_vector(time, "time")
    if u.shape != t.shape or np.any((u < 0) | (u >= len(wx))):
        raise ValueError("Invalid or unequal anchor units/times")
    if np.any((t < 0) | (t >= wx.shape[1])):
        raise ValueError("Anchor time itself must be inside the panel")
    return u, t


def _transform_weather(values):
    transformed = np.array(values, dtype=np.float64, copy=True)
    for name in LOG_CHANNELS:
        j = WEATHER_NAMES.index(name)
        transformed[..., j] = np.log1p(np.maximum(transformed[..., j], 0))
    # Inf and NaN are missing observations, never observations for scaling.
    transformed[~np.isfinite(values)] = np.nan
    return transformed


def _check_reference(reference, panel):
    if reference is None:
        raise ValueError("Supply fit_reference(...) output for standardized summaries")
    if tuple(reference["weather_names"]) != WEATHER_NAMES:
        raise ValueError("Reference weather schema differs")
    if list(reference["geo_names"]) != list(panel["feature_names"]["geo"]):
        raise ValueError("Reference geography schema differs")
    if tuple(reference["log_channels"]) != LOG_CHANNELS:
        raise ValueError("Reference weather transformation differs")
    for field, size in (("weather_mean", 12), ("weather_scale", 12),
                        ("geo_mean", 40), ("geo_scale", 40)):
        if np.asarray(reference[field]).shape != (size,):
            raise ValueError(f"Invalid reference {field}")
    for field in ("weather_scale", "geo_scale"):
        values = np.asarray(reference[field])
        if np.any(~np.isfinite(values) | (values <= 0)):
            raise ValueError(f"Invalid reference {field}")


def fit_reference(panel, units=None, hour_mask=None, *, scope=None, chunk=128):
    """Fit outcome-blind means/scales; no values are imputed or winsorized.

    ``units`` must be unique panel units. ``hour_mask`` is a bool [hour] or
    [all_panel_units, hour] array; it controls weather reference hours only.
    ``scope='training'`` requires explicit units. Geography deduplicates their
    county IDs, selecting the first supplied row for each county, as in D04.
    Scope and selected units/hours are recorded for provenance. Count zero geo
    coordinates remain unsupported/NaN; constant supported coordinates use
    scale one and retain all future differences from their reference mean.
    """
    wx, geo = _schema(panel)
    if not isinstance(chunk, int) or chunk < 1:
        raise ValueError("chunk must be positive")
    supplied_units = units is not None
    units = np.arange(len(wx), dtype=np.int64) if units is None else _integer_vector(units, "units")
    if not len(units) or len(np.unique(units)) != len(units) or np.any((units < 0) | (units >= len(wx))):
        raise ValueError("Reference units must be nonempty, unique and in range")
    if scope is None:
        scope = "descriptive_subset" if supplied_units or hour_mask is not None else "descriptive_all_D"
    if scope not in ("descriptive_all_D", "descriptive_subset", "training"):
        raise ValueError("Unknown reference scope")
    if scope == "training" and not supplied_units:
        raise ValueError("A training reference requires explicit training units")
    if scope == "descriptive_all_D" and (len(units) != len(wx) or hour_mask is not None):
        raise ValueError("descriptive_all_D must include every panel unit and hour")
    if hour_mask is not None:
        hour_mask = np.asarray(hour_mask)
        if hour_mask.dtype != bool or hour_mask.shape not in ((wx.shape[1],), wx.shape[:2]):
            raise ValueError("hour_mask must be boolean [hour] or [all units, hour]")

    def blocks():
        for start in range(0, len(units), chunk):
            ix = units[start:start + chunk]
            x = _transform_weather(wx[ix])
            allowed = np.ones(x.shape[:2], dtype=bool)
            if hour_mask is not None:
                allowed &= hour_mask if hour_mask.ndim == 1 else hour_mask[ix]
            valid = np.isfinite(x) & allowed[..., None]
            yield x, valid, allowed

    count = np.zeros(12, dtype=np.int64)
    total = np.zeros(12, dtype=np.float64)
    selected_hours = 0
    for x, valid, allowed in blocks():
        count += valid.sum(axis=(0, 1))
        total += np.where(valid, x, 0).sum(axis=(0, 1), dtype=np.float64)
        selected_hours += int(allowed.sum())
    if np.any(count == 0):
        raise ValueError("Every weather channel needs at least one finite reference observation")
    mean = total / count
    ss = np.zeros(12, dtype=np.float64)
    for x, valid, _ in blocks():
        ss += np.square(np.where(valid, x - mean, 0)).sum(axis=(0, 1), dtype=np.float64)
    scale = np.sqrt(ss / count)
    active = scale > 1e-7
    scale[~active] = 1

    county = np.asarray(panel["meta"]["county"])
    if county.shape != (len(wx),):
        raise ValueError("One county identity is required for every unit")
    _, first = np.unique(county[units], return_index=True)
    geo_units = units[first]
    g = geo[geo_units].astype(np.float64)
    good = np.isfinite(g)
    gc = good.sum(axis=0)
    gm = np.divide(np.where(good, g, 0).sum(axis=0), gc,
                   out=np.full(40, np.nan), where=gc > 0)
    gs = np.sqrt(np.divide(np.square(np.where(good, g - gm, 0)).sum(axis=0), gc,
                           out=np.zeros(40), where=gc > 0))
    ga = (gc > 0) & (gs > 1e-7)
    gs[~ga] = 1
    return {
        "weather_names": list(WEATHER_NAMES), "geo_names": list(panel["feature_names"]["geo"]),
        "log_channels": list(LOG_CHANNELS), "weather_mean": mean, "weather_scale": scale,
        "weather_count": count, "weather_active": active,
        "geo_mean": gm, "geo_scale": gs, "geo_count": gc, "geo_active": ga,
        "scope": scope, "reference_units": units.copy(), "reference_geo_units": geo_units.copy(),
        "selected_weather_hours": selected_hours, "hour_mask_supplied": hour_mask is not None,
        "weather_weighting": "equal selected county-event-hour; overlapping event windows may repeat calendar hours",
        "geography_weighting": "equal unique county; first supplied unit per county, no imputation",
        "prediction_validation_reference": scope == "training",
        "outcomes_used": False,
    }


def extract_geography(panel, unit, reference=None):
    """Return all 40 raw coordinates and finite masks, optionally reference z."""
    wx, geo = _schema(panel)
    unit = _integer_vector(unit, "unit")
    if np.any((unit < 0) | (unit >= len(wx))):
        raise ValueError("Invalid geography units")
    raw = geo[unit].copy()
    valid = np.isfinite(raw)
    z = None
    if reference is not None:
        _check_reference(reference, panel)
        z = ((raw - reference["geo_mean"]) / reference["geo_scale"]).astype(np.float32)
        z[~valid] = np.nan
    return {"geo_raw": raw, "geo_z": z, "geo_valid": valid,
            "geo_names": list(panel["feature_names"]["geo"])}


def extract_curves(panel, unit, time, offsets=None, reference=None):
    """Return [anchor, offset, channel] curves, absolute indices and masks.

    The default offsets are every hour from -48 to +24 inclusive. Indices beyond
    the panel remain explicit in absolute_index; their values are NaN, not a
    copied boundary observation. Custom offsets must be unique integers.
    """
    u, t = _anchors(panel, unit, time)
    wx = np.asarray(panel["weather"])
    offsets = OFFSETS.copy() if offsets is None else _integer_vector(offsets, "offsets")
    if len(np.unique(offsets)) != len(offsets):
        raise ValueError("Offsets must be unique")
    idx = t[:, None] + offsets[None, :]
    inside = (idx >= 0) & (idx < wx.shape[1])
    raw = np.full((len(u), len(offsets), 12), np.nan, dtype=np.result_type(wx.dtype, np.float32))
    row, column = np.nonzero(inside)
    raw[row, column] = wx[u[row], idx[row, column]]
    valid = inside[..., None] & np.isfinite(raw)
    z = None
    if reference is not None:
        _check_reference(reference, panel)
        z = ((_transform_weather(raw) - reference["weather_mean"]) /
             reference["weather_scale"]).astype(np.float32)
        z[~valid] = np.nan
    return {
        "unit": u.copy(), "time": t.copy(), "offsets": offsets.copy(),
        "absolute_index": idx, "in_bounds": inside, "valid": valid,
        "raw": raw, "standardized": z, "weather_names": list(WEATHER_NAMES),
        "strict_past_offset": offsets < 0,
        "weather_time_support": WEATHER_TIME_SUPPORT.copy(),
        "anchor_time_support": ANCHOR_TIME_SUPPORT,
        "plus_one_interval_role": "precip/snowfall/gust at T+1 cover (T,T+1]; retrospective observation-interval aligned, nonpredictive",
        "reference_scope": None if reference is None else reference["scope"],
        **extract_geography(panel, u, reference),
    }


def _complete_stats(x, valid):
    count = valid.sum(axis=1).astype(np.int16)
    complete = count == x.shape[1]
    mean = np.where(valid, x, 0).sum(axis=1, dtype=np.float64) / x.shape[1]
    minimum = np.where(valid, x, np.inf).min(axis=1)
    maximum = np.where(valid, x, -np.inf).max(axis=1)
    stats = {}
    for name, values in (("mean", mean), ("min", minimum), ("max", maximum)):
        values = values.astype(np.float32)
        values[~complete] = np.nan
        stats[name] = values
    return stats, count, complete


def _pair_mean(earlier, later, pairs):
    """Mean pointwise products, never a product of pointwise means."""
    a, b = pairs.T
    left, right = earlier[:, :, a], later[:, :, b]
    valid = np.isfinite(left) & np.isfinite(right)
    count = valid.sum(axis=1).astype(np.int16)
    complete = count == left.shape[1]
    product = np.where(valid, left, 0).astype(np.float64) * np.where(valid, right, 0)
    value = (product.sum(axis=1) / left.shape[1]).astype(np.float32)
    value[~complete] = np.nan
    return value, count, complete


def _summarize_chunk(panel, u, t, reference, include_pairs):
    curves = extract_curves(panel, u, t, reference=reference)
    raw, z, valid = curves["raw"], curves["standardized"], curves["valid"]
    raw_stats, z_stats, counts, complete = [], [], [], []
    accum = []
    ai = [WEATHER_NAMES.index(name) for name in ACCUMULATION_CHANNELS]
    sync_value, sync_count, sync_complete = [], [], []
    for lo, hi in BANDS:
        select = (OFFSETS >= lo) & (OFFSETS <= hi)
        r, ct, co = _complete_stats(raw[:, select], valid[:, select])
        s, _, _ = _complete_stats(z[:, select], valid[:, select])
        raw_stats.append(r); z_stats.append(s); counts.append(ct); complete.append(co)
        total = np.where(valid[:, select], raw[:, select], 0).sum(axis=1, dtype=np.float64)[:, ai]
        total[~co[:, ai]] = np.nan
        accum.append(total.astype(np.float32))
        if include_pairs:
            value, count, good = _pair_mean(z[:, select], z[:, select], SYNC_PAIRS)
            sync_value.append(value); sync_count.append(count); sync_complete.append(good)
    result = {
        "raw": {key: np.stack([v[key] for v in raw_stats], axis=1) for key in ("mean", "min", "max")},
        "standardized": {key: np.stack([v[key] for v in z_stats], axis=1) for key in ("mean", "min", "max")},
        "support": {"count": np.stack(counts, axis=1), "complete": np.stack(complete, axis=1)},
        "accumulation": {"sum": np.stack(accum, axis=1)},
        "near_hours": {"raw": raw[:, [47, 48, 49]].copy(),
                       "standardized": z[:, [47, 48, 49]].copy(),
                       "valid": valid[:, [47, 48, 49]].copy(),
                       "in_bounds": curves["in_bounds"][:, [47, 48, 49]].copy(),
                       "absolute_index": curves["absolute_index"][:, [47, 48, 49]].copy()},
    }
    if include_pairs:
        result["synchronous"] = {"value": np.stack(sync_value, axis=1),
                                  "count": np.stack(sync_count, axis=1),
                                  "complete": np.stack(sync_complete, axis=1)}
        means = result["standardized"]["mean"]
        sym, anti, ec, lc = [], [], [], []
        sa, sb = SYNC_PAIRS.T
        aa, ab = CROSS_PAIRS.T
        for early, late in BAND_PAIRS:
            e, l = means[:, early], means[:, late]
            sym.append((e[:, sa] * l[:, sb] + e[:, sb] * l[:, sa]) * .5)
            anti.append((e[:, aa] * l[:, ab] - e[:, ab] * l[:, aa]) * .5)
            ec.append(result["support"]["count"][:, early])
            lc.append(result["support"]["count"][:, late])
        result["cross_band"] = {
            "symmetric": np.stack(sym, axis=1), "antisymmetric": np.stack(anti, axis=1),
            "early_count": np.stack(ec, axis=1), "late_count": np.stack(lc, axis=1),
        }
        result["cross_band"]["symmetric_complete"] = np.isfinite(result["cross_band"]["symmetric"])
        result["cross_band"]["antisymmetric_complete"] = np.isfinite(result["cross_band"]["antisymmetric"])
        # Strict past only: all s and s-delay are in offsets [-48,-1].
        past = z[:, :48]
        dv, dc, dg = [], [], []
        for delay in DELAYS:
            value, count, good = _pair_mean(past[:, :-delay], past[:, delay:], DIRECTED_PAIRS)
            dv.append(value); dc.append(count); dg.append(good)
        result["delayed"] = {"value": np.stack(dv, axis=1), "count": np.stack(dc, axis=1),
                             "complete": np.stack(dg, axis=1)}
    return result


def summarize_anchors(panel, unit, time, reference, *, include_pairs=True, chunk=512):
    """Compute fixed, complete-window summaries in anchor batches.

    Shapes: univariate [N,6,12]; accumulations [N,6,2]; synchronous
    [N,6,78]; cross-band symmetric [N,15,78] and antisymmetric [N,15,66];
    delayed [N,3,144]. For a<b, cross-band antisym is
    (mean_early[a]*mean_late[b] - mean_early[b]*mean_late[a])/2.
    Together with symmetric terms this retains both directed products;
    diagonal symmetric terms retain repeated same-channel exposure. Delayed
    entries are mean_s z_a(s-delay)*z_b(s), with both times strictly past.

    Geo output is [N,40]. ``include_pairs=False`` is sufficient for the
    108-coordinate strict-past mean/min/max descriptive matching vector.
    """
    u, t = _anchors(panel, unit, time)
    _check_reference(reference, panel)
    if not isinstance(chunk, int) or chunk < 1:
        raise ValueError("chunk must be positive")
    # Even an empty set has the same well-defined array schema.
    result = None
    for i in range(0, max(1, len(u)), chunk):
        piece = _summarize_chunk(panel, u[i:i + chunk], t[i:i + chunk], reference, include_pairs)
        if result is None:
            result = {block: {key: np.empty((len(u), *value.shape[1:]), dtype=value.dtype)
                              for key, value in fields.items()} for block, fields in piece.items()}
        for block, fields in piece.items():
            for key, value in fields.items():
                result[block][key][i:i + chunk] = value
    result.update({
        "unit": u.copy(), "time": t.copy(), "weather_names": list(WEATHER_NAMES),
        "bands": np.array(BANDS, dtype=np.int16), "band_names": list(BAND_NAMES),
        "band_expected_count": np.array([hi - lo + 1 for lo, hi in BANDS], dtype=np.int16),
        "band_strict_past": np.array([hi < 0 for _, hi in BANDS]),
        "reference_scope": reference["scope"], "reference_outcomes_used": False,
        "weather_time_support": WEATHER_TIME_SUPPORT.copy(),
        "anchor_time_support": ANCHOR_TIME_SUPPORT,
        **extract_geography(panel, u, reference),
    })
    result["accumulation"].update({"channel_names": list(ACCUMULATION_CHANNELS),
        "units": ["mm (hourly precip sum)", "mm water-equivalent (hourly snowfall sum)"],
        "count": result["support"]["count"][:, :, [3, 6]],
        "complete": result["support"]["complete"][:, :, [3, 6]]})
    result["near_hours"].update({"offsets": NEAR_OFFSETS.copy(),
        "strict_past_offset": NEAR_OFFSETS < 0,
        "weather_time_support": WEATHER_TIME_SUPPORT.copy(),
        "plus_one_interval_role": "precip/snowfall/gust at T+1 cover (T,T+1]; retrospective observation-interval aligned, nonpredictive"})
    if include_pairs:
        result["synchronous"]["channel_pairs"] = SYNC_PAIRS.copy()
        result["cross_band"].update({"band_pairs": BAND_PAIRS.copy(),
            "symmetric_channel_pairs": SYNC_PAIRS.copy(), "antisymmetric_channel_pairs": CROSS_PAIRS.copy()})
        result["delayed"].update({"delays": np.array(DELAYS, dtype=np.int16),
            "channel_pairs": DIRECTED_PAIRS.copy(),
            "expected_count": 48 - np.array(DELAYS, dtype=np.int16)})
    result["definitions"] = {
        "raw": "unchanged original weather; min/max are numeric extrema, not universal hazard severity",
        "standardized": "log1p nonnegative CAPE/precip/snowfall, then reference channel z; summaries after transform",
        "synchronous": "mean of within-hour z_a*z_b; a<=b includes squares",
        "cross_band": "products of band means; symmetric and signed antisymmetric coordinates retain both directions",
        "delayed": "mean z_a(s-delay)*z_b(s), all 144 ordered pairs, s and s-delay in strict past [-48,-1]",
        "complete_windows": "any required missing/out-of-panel observation makes that summary NaN; counts remain explicit",
        "time_alignment": "ERA5 precip/snowfall/gust indexed T cover preceding hour (T-1,T]; other retained weather instantaneous at T. Near-hour +1 interval quantities bracket the outage observation interval, not necessarily physically subsequent weather",
        "inference": "descriptive exposure summaries; no causal identification or independence claim",
    }
    return result


def flatten_summaries(summary, *, blocks=("standardized", "accumulation", "synchronous",
                                         "cross_band", "delayed", "geo_z"),
                      bands=None, statistics=("mean", "min", "max")):
    """Return feature_matrix/names/valid/strict_past with explicit coordinates.

    Registered descriptive matching uses
    ``blocks=('standardized',), bands=(0,1,2)``: exactly 108 columns. Its
    standardizer may use all D because this is descriptive matching, not OOF
    prediction validation. For actual prediction use a training reference.
    Selecting bands restricts cross-band pairs to pairs entirely among them;
    delayed and geo blocks do not depend on selected bands. No NaN is imputed.
    """
    selected = list(range(6)) if bands is None else list(_integer_vector(bands, "bands"))
    if len(set(selected)) != len(selected) or any(b < 0 or b >= 6 for b in selected):
        raise ValueError("bands must be unique indices in 0..5")
    if any(stat not in ("mean", "min", "max") for stat in statistics):
        raise ValueError("Unknown univariate statistic")
    if len(set(blocks)) != len(blocks):
        raise ValueError("Duplicate blocks")
    arrays, names, strict = [], [], []
    wxnames, bnames = summary["weather_names"], summary["band_names"]

    def add(value, labels, available):
        value = np.asarray(value)
        if value.ndim != 2 or value.shape[1] != len(labels):
            raise ValueError("Feature label/array mismatch")
        arrays.append(value); names.extend(labels); strict.extend([bool(available)] * len(labels))

    for block in blocks:
        if block in ("raw", "standardized"):
            for b in selected:
                for stat in statistics:
                    add(summary[block][stat][:, b],
                        [f"{block}:{stat}:{name}:{bnames[b]}" for name in wxnames], BANDS[b][1] < 0)
        elif block == "accumulation":
            for b in selected:
                add(summary[block]["sum"][:, b],
                    [f"raw:sum:{name}:{bnames[b]}" for name in ACCUMULATION_CHANNELS], BANDS[b][1] < 0)
        elif block == "synchronous":
            if block not in summary:
                raise ValueError("Pairs were not requested from summarize_anchors")
            for b in selected:
                add(summary[block]["value"][:, b],
                    [f"sync_z:{wxnames[a]}*{wxnames[c]}:{bnames[b]}" for a, c in SYNC_PAIRS], BANDS[b][1] < 0)
        elif block == "cross_band":
            if block not in summary:
                raise ValueError("Pairs were not requested from summarize_anchors")
            for p, (e, l) in enumerate(BAND_PAIRS):
                if e not in selected or l not in selected:
                    continue
                for kind, pairs in (("symmetric", SYNC_PAIRS), ("antisymmetric", CROSS_PAIRS)):
                    add(summary[block][kind][:, p],
                        [f"{kind}_z:{wxnames[a]}->{wxnames[b]}:{bnames[e]}_to_{bnames[l]}" for a, b in pairs], BANDS[l][1] < 0)
        elif block == "delayed":
            if block not in summary:
                raise ValueError("Pairs were not requested from summarize_anchors")
            for p, delay in enumerate(DELAYS):
                add(summary[block]["value"][:, p],
                    [f"delay_z:{wxnames[a]}->{wxnames[b]}:{delay}h:strict_past_48h" for a, b in DIRECTED_PAIRS], True)
        elif block in ("geo_raw", "geo_z"):
            add(summary[block], [f"{block}:{name}" for name in summary["geo_names"]], True)
        else:
            raise ValueError(f"Unknown feature block {block}")
    matrix = np.concatenate(arrays, axis=1).astype(np.float32) if arrays else np.empty((len(summary["unit"]), 0), dtype=np.float32)
    if len(names) != len(set(names)):
        raise ValueError("Duplicate flattened feature labels")
    return {"feature_matrix": matrix, "feature_names": names, "feature_valid": np.isfinite(matrix),
            "feature_strict_past": np.asarray(strict, dtype=bool),
            "reference_scope": summary["reference_scope"], "unit": summary["unit"].copy(),
            "time": summary["time"].copy()}


def _self_test():
    """Small synthetic-only checks; never loads or fits the real public panel."""
    rng = np.random.default_rng(17)
    weather = rng.normal(size=(4, 80, 12)).astype(np.float32)
    geo = rng.normal(size=(4, 40)).astype(np.float32)
    geo[1] = geo[0]; geo[0:2, 3] = np.nan; geo[:, 4] = np.nan
    panel = {"weather": weather, "geo": geo, "meta": {"county": np.array(["a", "a", "b", "c"])},
             "feature_names": {"weather": list(WEATHER_NAMES), "geo": [f"g{i}" for i in range(40)]}}
    ref = fit_reference(panel, chunk=2)
    assert np.all(ref["weather_count"] == 320) and len(ref["reference_geo_units"]) == 3
    transformed = _transform_weather(weather)
    np.testing.assert_allclose(ref["weather_mean"], transformed.mean(axis=(0, 1)), atol=1e-12)
    np.testing.assert_allclose(ref["weather_scale"], transformed.std(axis=(0, 1)), atol=1e-12)
    assert ref["geo_count"][3] == 2 and ref["geo_count"][4] == 0
    curves = extract_curves(panel, np.array([0, 2]), np.array([0, 79]), reference=ref)
    assert curves["absolute_index"][0, 0] == -48 and curves["absolute_index"][1, -1] == 103
    assert np.isnan(curves["raw"][0, :48]).all() and np.isnan(curves["raw"][1, 49:]).all()
    np.testing.assert_array_equal(curves["raw"][:, 48], weather[[0, 2], [0, 79]])
    assert np.isnan(curves["geo_z"][:, 4]).all()
    unit, time = np.array([0, 1, 2]), np.array([48, 55, 79])
    summary = summarize_anchors(panel, unit, time, ref, chunk=1)
    other = summarize_anchors(panel, unit, time, ref, chunk=4)
    flat = flatten_summaries(summary)
    np.testing.assert_allclose(flat["feature_matrix"], flatten_summaries(other)["feature_matrix"], equal_nan=True)
    matching = flatten_summaries(summary, blocks=("standardized",), bands=(0, 1, 2))
    assert matching["feature_matrix"].shape == (3, 108) and matching["feature_strict_past"].all()
    assert summary["support"]["count"][2, 4, 0] == 0
    assert np.isnan(summary["raw"]["mean"][2, 4:]).all()
    np.testing.assert_array_equal(summary["near_hours"]["absolute_index"], time[:, None] + NEAR_OFFSETS)
    np.testing.assert_array_equal(summary["near_hours"]["raw"][:2], weather[unit[:2, None], time[:2, None] + NEAR_OFFSETS])
    assert np.isnan(summary["near_hours"]["raw"][2, 2]).all()
    assert summary["weather_time_support"]["gust"] == "preceding_hour_(T-1,T]"
    assert summary["weather_time_support"]["soil_moisture"] == "instantaneous_at_T"
    manual = extract_curves(panel, unit[:1], time[:1], reference=ref)
    zz = manual["standardized"][0, :48]
    for q, (lo, hi) in enumerate(BANDS[:3]):
        sel = (OFFSETS >= lo) & (OFFSETS <= hi)
        np.testing.assert_allclose(summary["raw"]["mean"][0, q], manual["raw"][0, sel].mean(axis=0), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(summary["accumulation"]["sum"][0, 0, 0], weather[0, :24, 3].sum(), atol=1e-6)
    for q, delay in enumerate(DELAYS):
        for a, b in ((2, 3), (3, 2), (3, 3)):
            j = a * 12 + b
            expected = (zz[:-delay, a].astype(np.float64) * zz[delay:, b]).mean()
            np.testing.assert_allclose(summary["delayed"]["value"][0, q, j], expected, atol=1e-7)
    # Explicit earlier rain/later wind reverses the signed band-order statistic.
    directional = np.zeros((2, 80, 12), dtype=np.float32)
    directional[0, :24, 3] = 1; directional[0, 24:42, 2] = 1
    directional[1, :24, 2] = 1; directional[1, 24:42, 3] = 1
    p2 = {**panel, "weather": directional, "geo": geo[:2], "meta": {"county": np.array(["a", "b"])}}
    r2 = fit_reference(p2)
    s2 = summarize_anchors(p2, np.array([0, 1]), np.array([48, 48]), r2)
    pair = next(i for i, pair in enumerate(CROSS_PAIRS) if tuple(pair) == (2, 3))
    a0, a1 = s2["cross_band"]["antisymmetric"][:, 0, pair]
    assert a0 < 0 < a1
    # Sym + antisym reconstructs each a->b direction, including bandwise self.
    sym = next(i for i, pair in enumerate(SYNC_PAIRS) if tuple(pair) == (2, 3))
    e, l = s2["standardized"]["mean"][:, 0], s2["standardized"]["mean"][:, 1]
    np.testing.assert_allclose(s2["cross_band"]["symmetric"][:, 0, sym] + s2["cross_band"]["antisymmetric"][:, 0, pair], e[:, 2] * l[:, 3], atol=1e-7)
    selfpair = next(i for i, pair in enumerate(SYNC_PAIRS) if tuple(pair) == (3, 3))
    np.testing.assert_allclose(s2["cross_band"]["symmetric"][:, 0, selfpair], e[:, 3] * l[:, 3], atol=1e-7)
    # A single missing hour invalidates only summaries requiring that channel/hour.
    missing = {**panel, "weather": weather.copy()}
    missing["weather"][0, 0, 2] = np.nan
    ms = summarize_anchors(missing, np.array([0]), np.array([48]), ref)
    assert ms["support"]["count"][0, 0, 2] == 23 and np.isnan(ms["raw"]["mean"][0, 0, 2])
    assert np.isfinite(ms["raw"]["mean"][0, 0, 3])
    # Training reference excludes deliberately modified nontraining units/hours.
    hm = np.arange(80) < 40
    tr = fit_reference(panel, units=np.array([0]), hour_mask=hm, scope="training")
    changed = {**panel, "weather": weather.copy()}
    changed["weather"][1:] = 1e6; changed["weather"][0, 40:] = 1e6
    tr2 = fit_reference(changed, units=np.array([0]), hour_mask=hm, scope="training")
    np.testing.assert_array_equal(tr["weather_mean"], tr2["weather_mean"])
    np.testing.assert_array_equal(tr["weather_scale"], tr2["weather_scale"])
    empty = summarize_anchors(panel, np.array([], dtype=int), np.array([], dtype=int), ref)
    assert flatten_summaries(empty)["feature_matrix"].shape == (0, flat["feature_matrix"].shape[1])
    assert flat["feature_matrix"].shape[1] == 3328
    assert np.array_equal(panel["weather"], weather)  # All original inputs unchanged.
    print("D05 synthetic checks passed: boundaries, masks, complete windows, all-pair directions/self, references, chunks, flattening.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    arguments = parser.parse_args()
    if arguments.self_test:
        _self_test()
    else:
        parser.error("This is a library; only --self-test runs directly")
