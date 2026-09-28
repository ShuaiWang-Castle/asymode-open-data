"""D02 descriptive county-structure/weather dynamics; public D only, no fitting.

Weather alone selects anchors. Outcomes describe stock, adjacent-hour net change,
peak timing uncertainty, and matched-weather phase differences. None identifies
damage, recovery, propagation speed, or a causal weather-order effect.
"""
from __future__ import annotations

import os
for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                         "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_variable] = "2"

import argparse
import hashlib
import io
import itertools
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FEATURES = ROOT / "data/interim/panel_v1/features_v1D.npz"
LOOKUP = ROOT / "runs/geo_weather_20260924/county_structure_d02/county_structure_d02_lookup.npz"
OUTPUT = HERE / "results/v1/county_dynamics_d02.json"
CACHE = ROOT / "runs/geo_weather_20260924/county_dynamics_d02/weather_alignment.npz"
REGIMES = ["tropical", "winter", "synoptic_wind", "convective", "heavy_rain"]
ANCHORS = [("gust", "gust", 1.0), ("precip", "precip", 1.0),
           ("cold", "t2m_c", -1.0), ("snowfall", "snowfall", 1.0),
           ("soil", "soil_moisture", 1.0), ("cape", "cape", 1.0)]
OFFSETS = np.arange(-24, 25)
COARSE = np.flatnonzero(np.isin(OFFSETS, [-24, -12, -6, 0, 6, 12, 24]))
START, STOP = 72, 216
SEED = 20260929


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def merged_groups(system, family, origin, county):
    systems = np.unique(system)
    parent = {s: s for s in systems}
    def find(s):
        while parent[s] != s:
            parent[s] = parent[parent[s]]
            s = parent[s]
        return s
    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)
    info = {}
    for s in systems:
        ix = np.flatnonzero(system == s)
        info[s] = (family[ix[0]], np.datetime64(origin[ix[0]]), set(county[ix]))
    for a, b in itertools.combinations(systems, 2):
        fa, ta, ca = info[a]
        fb, tb, cb = info[b]
        if fa == fb or (abs(ta - tb) <= np.timedelta64(16, "D") and ca & cb):
            union(a, b)
    return np.array([find(s) for s in system])


def weighted_quantiles(values, weights, quantiles=(.1, .25, .5, .75, .9)):
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not ok.any():
        return [None] * len(quantiles)
    v, w = np.asarray(values)[ok], np.asarray(weights)[ok]
    order = np.argsort(v, kind="stable")
    v, w = v[order], w[order]
    positions = (np.cumsum(w) - .5 * w) / w.sum()
    return np.interp(quantiles, positions, v).tolist()


def weighted_mean(values, weights):
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    return float(np.dot(values[ok], weights[ok]) / weights[ok].sum()) if ok.any() else None


def support(d, ids):
    return dict(county_events=len(ids), counties=len(np.unique(d["county"][ids])),
                systems=len(np.unique(d["system"][ids])),
                merged_event_groups=len(np.unique(d["group"][ids])),
                design_weight=float(d["w"][ids].sum()))


def signed_distribution(values, weights):
    ok = np.isfinite(values) & (weights > 0)
    if not ok.any():
        return dict(n=0, mean=None, quantiles_10_25_50_75_90=[None]*5,
                    negative_fraction=None, zero_fraction=None, positive_fraction=None)
    v, w = values[ok], weights[ok]
    return dict(n=int(ok.sum()), mean=weighted_mean(v, w),
                quantiles_10_25_50_75_90=weighted_quantiles(v, w),
                negative_fraction=float(w[v < -1e-9].sum()/w.sum()),
                zero_fraction=float(w[np.abs(v) <= 1e-9].sum()/w.sum()),
                positive_fraction=float(w[v > 1e-9].sum()/w.sum()))


def grouped_signs(values, ids, d):
    good = np.isfinite(values)
    ids, values = ids[good], values[good]
    group, inv = np.unique(d["group"][ids], return_inverse=True)
    numerator, denominator = np.zeros(len(group)), np.zeros(len(group))
    np.add.at(numerator, inv, d["w"][ids] * values)
    np.add.at(denominator, inv, d["w"][ids])
    mean = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    return dict(groups=len(group), negative=int((mean < -1e-9).sum()),
                zero=int((np.abs(mean) <= 1e-9).sum()), positive=int((mean > 1e-9).sum()))


def curve(values, ids, d, draws=200):
    """Weighted observed-cell mean and a whole-event-group bootstrap, never zero filling."""
    x = values[ids].astype(np.float64)
    w = d["w"][ids]
    valid = np.isfinite(x)
    denominator = np.sum(w[:, None] * valid, 0)
    numerator = np.sum(w[:, None] * np.where(valid, x, 0), 0)
    mean = np.divide(numerator, denominator, out=np.full(x.shape[1], np.nan), where=denominator > 0)
    group, inv = np.unique(d["group"][ids], return_inverse=True)
    gn, gd = np.zeros((len(group), x.shape[1])), np.zeros((len(group), x.shape[1]))
    np.add.at(gn, inv, w[:, None] * np.where(valid, x, 0))
    np.add.at(gd, inv, w[:, None] * valid)
    ng = (gd > 0).sum(0)
    lo, hi = np.full(x.shape[1], np.nan), np.full(x.shape[1], np.nan)
    valid_draws = np.zeros(x.shape[1], int)
    if len(group) >= 4 and draws:
        rng = np.random.default_rng(SEED)
        counts = rng.multinomial(len(group), np.full(len(group), 1/len(group)), size=draws)
        bn, bd = counts @ gn, counts @ gd
        boot = np.divide(bn, bd, out=np.full_like(bn, np.nan), where=bd > 0)
        valid_draws = np.isfinite(boot).sum(0)
        for j in np.flatnonzero((ng >= 4) & (valid_draws >= .8 * draws)):
            lo[j], hi[j] = np.quantile(boot[np.isfinite(boot[:, j]), j], [.025, .975])
    return dict(mean=mean, ci95_lower=lo, ci95_upper=hi,
                observed_county_events=valid.sum(0), observed_event_groups=ng,
                observed_weight=denominator,
                observed_weight_fraction=np.divide(denominator, w.sum(),
                    out=np.zeros_like(denominator), where=w.sum() > 0), bootstrap_valid_draws=valid_draws)


def load_data():
    with np.load(FEATURES, allow_pickle=False) as f:
        names = f["weather_channels"].astype(str).tolist()[:12]
        weather = f["xu"][:, :, :12].astype(np.float32)
        y, obs = f["y_full"].astype(np.float32), f["obs_full"].astype(bool)
        d = {k: f[k].copy() for k in ["system", "family", "fips", "origin", "regime", "w", "w_raw"]}
    d["county"] = d.pop("fips").astype(str)
    d["w"] = d["w"].astype(float)
    d["group"] = merged_groups(d["system"], d["family"], d["origin"], d["county"])
    lookup_bytes = LOOKUP.read_bytes()
    d["lookup_sha256"] = hashlib.sha256(lookup_bytes).hexdigest()
    with np.load(io.BytesIO(lookup_bytes), allow_pickle=False) as f:
        county = f["county"].astype(str)
        lookup = dict(zip(county, f["type"].astype(int)))
        assert len(lookup) == len(county), "county structure lookup must be unique"
    missing = set(d["county"]) - set(lookup)
    assert not missing, f"Missing structure assignments for {len(missing)} counties"
    d["type"] = np.array([lookup[c] for c in d["county"]], dtype=np.int8)
    assert set(np.unique(d["type"])) <= set(range(6))
    assert y.shape == obs.shape == weather.shape[:2] and y.shape[1] == STOP
    assert len(names) == 12 and np.isfinite(weather).all(), "Weather requires a separate missingness audit"
    assert np.isfinite(y[obs]).all() and ((y[obs] >= 0) & (y[obs] <= 1)).all()
    d.update(weather=weather, names=names, y=y, obs=obs)
    valid = obs[:, START:STOP]
    yy = y[:, START:STOP]
    count = valid.sum(1)
    peak = np.where(valid, yy, -np.inf).max(1)
    low = np.where(valid, yy, np.inf).min(1)
    tie = valid & (yy == peak[:, None])
    first = tie.argmax(1) + START
    last = STOP - 1 - tie[:, ::-1].argmax(1)
    status = np.where(count == 0, 0, np.where(peak <= 0, 1, np.where(peak == low, 2, 3)))
    first[status != 3] = -1
    last[status != 3] = -1
    d.update(outage_status=status.astype(np.int8), outage_peak=peak,
             outage_peak_first=first, outage_peak_last=last, outage_peak_ties=tie.sum(1),
             forecast_coverage=count / (STOP-START),
             forecast_mean=np.divide(np.where(valid, yy, 0).sum(1), count,
                 out=np.full(len(y), np.nan), where=count > 0))
    return d


def align_weather(d, source, sign, name):
    index = sign * d["weather"][:, :, d["names"].index(source)]
    win = index[:, START:STOP]
    peak, floor = win.max(1), win.min(1)
    tied = win == peak[:, None]
    first = START + tied.argmax(1)
    last = STOP - 1 - tied[:, ::-1].argmax(1)
    flat = peak-floor <= np.maximum(1e-8, 1e-6*np.max(np.abs(win), 1))
    absent = (peak <= 1e-10) if name in ("precip", "snowfall", "cape") else np.zeros(len(win), bool)
    status = np.where(absent, 1, np.where(flat, 2, 0)).astype(np.int8)
    valid_peak = status == 0
    pos = first[:, None] + OFFSETS
    within = (pos >= 0) & (pos < STOP) & valid_peak[:, None]
    safe = np.clip(pos, 0, STOP-1)
    rows = np.arange(len(win))[:, None]
    weather = d["weather"][rows, safe].copy()
    weather[~within] = np.nan
    level = np.where(within & d["obs"][rows, safe], d["y"][rows, safe], np.nan)
    prev = np.clip(safe-1, 0, STOP-1)
    delta_valid = within & (pos > 0) & d["obs"][rows, safe] & d["obs"][rows, prev]
    change = np.where(delta_valid, d["y"][rows, safe]-d["y"][rows, prev], np.nan)
    wx = np.where(within, index[rows, safe], np.nan)
    slope = np.where(within & (pos > 0), index[rows, safe]-index[rows, prev], np.nan)
    basepos = first[:, None] + np.arange(-6, 0)
    baseobs = d["obs"][rows, basepos] & valid_peak[:, None]
    nb = baseobs.sum(1)
    baseline = np.divide(np.where(baseobs, d["y"][rows, basepos], 0).sum(1), nb,
                         out=np.full(len(win), np.nan), where=nb >= 4)
    # No artificial temporal location is assigned to constant/absent weather.
    first_cache, last_cache = first.copy(), last.copy()
    first_cache[~valid_peak] = -1
    last_cache[~valid_peak] = -1
    cut = weighted_quantiles(peak[valid_peak], d["w"][valid_peak], [1/3, 2/3])
    intensity = np.full(len(win), -1, np.int8)
    if all(c is not None for c in cut):
        intensity[valid_peak] = np.searchsorted(cut, peak[valid_peak], side="right")
    return dict(status=status, valid=valid_peak, peak=peak, floor=floor,
                first=first_cache, last=last_cache, ties=tied.sum(1),
                intensity=intensity, intensity_cutoffs=cut, pos=pos, level=level,
                change=change, departure=level-baseline[:, None], baseline=baseline,
                weather=weather, index=wx, slope=slope,
                earlier_context_exceeds_peak=np.nanmax(index[:, START-24:START], 1) > peak)


def lag_summary(d, a, ids):
    eligible = a["valid"][ids] & (d["outage_status"][ids] == 3)
    ii = ids[eligible]
    lo = d["outage_peak_first"][ii] - a["last"][ii]
    hi = d["outage_peak_last"][ii] - a["first"][ii]
    midpoint = (lo+hi)/2
    complete = ((d["forecast_coverage"][ii] == 1) & (d["outage_peak_first"][ii] > START)
                & (d["outage_peak_last"][ii] < STOP-1) & (a["first"][ii] > START)
                & (a["last"][ii] < STOP-1))
    def one(jj, ll, hh, mid):
        w = d["w"][jj]
        return dict(support=support(d, jj), midpoint_hours=signed_distribution(mid, w),
                    interval_width_hours=weighted_quantiles(hh-ll, w),
                    definitely_after_weather_fraction=weighted_mean((ll > 0).astype(float), w),
                    definitely_before_weather_fraction=weighted_mean((hh < 0).astype(float), w),
                    interval_includes_zero_fraction=weighted_mean(((ll <= 0)&(hh >= 0)).astype(float), w),
                    observed_outage_peak_tie_fraction=weighted_mean((d["outage_peak_ties"][jj]>1).astype(float), w))
    return dict(all_defined_observed_peaks=one(ii, lo, hi, midpoint),
                complete_forecast_interior_peaks=one(ii[complete], lo[complete], hi[complete], midpoint[complete]))


def phase_loops(d, a, ids):
    """Peak-phase differences at matched weather level, with net-stock-change companions."""
    value = a["index"][ids]
    span = (a["peak"]-a["floor"])[ids]
    z = np.divide(value-a["floor"][ids, None], span[:, None],
                  out=np.full_like(value, np.nan), where=span[:, None] > 0)
    valid = ((a["pos"][ids] >= START) & (a["pos"][ids] < STOP)
             & np.isfinite(value) & (z >= -1e-6) & (z <= 1+1e-6))
    bins = np.minimum((4*np.clip(np.nan_to_num(z), 0, 1)).astype(int), 3)
    output = []
    for b in range(4):
        shared = valid & (bins == b)
        before, after = shared & (OFFSETS < 0), shared & (OFFSETS > 0)
        item = dict(relative_weather_bin=[b/4, (b+1)/4], metrics={})
        for name, values in (("stock", a["level"][ids]), ("net_stock_change_per_hour", a["change"][ids])):
            pre, post = before & np.isfinite(values), after & np.isfinite(values)
            np0, np1 = pre.sum(1), post.sum(1)
            ok = (np0 >= 2) & (np1 >= 2)
            av0 = np.divide(np.where(pre, values, 0).sum(1), np0, out=np.full(len(ids), np.nan), where=np0 > 0)
            av1 = np.divide(np.where(post, values, 0).sum(1), np1, out=np.full(len(ids), np.nan), where=np1 > 0)
            t0 = np.divide((pre*OFFSETS).sum(1), np0, out=np.full(len(ids), np.nan), where=np0 > 0)
            t1 = np.divide((post*OFFSETS).sum(1), np1, out=np.full(len(ids), np.nan), where=np1 > 0)
            jj, w = ids[ok], d["w"][ids[ok]]
            diff = av1[ok]-av0[ok]
            pre_rise = np.divide((pre & (a["slope"][ids] > 0)).sum(1), np0,
                                  out=np.full(len(ids), np.nan), where=np0 > 0)
            post_fall = np.divide((post & (a["slope"][ids] < 0)).sum(1), np1,
                                   out=np.full(len(ids), np.nan), where=np1 > 0)
            item["metrics"][name] = dict(support=support(d, jj),
                post_minus_pre=signed_distribution(diff, w),
                event_group_difference_signs=grouped_signs(diff, jj, d),
                pre_mean=weighted_mean(av0[ok], w), post_mean=weighted_mean(av1[ok], w),
                mean_pre_lag_hours=weighted_mean(t0[ok], w), mean_post_lag_hours=weighted_mean(t1[ok], w),
                mean_elapsed_phase_gap_hours=weighted_mean(t1[ok]-t0[ok], w),
                mean_pre_peak_six_hour_stock=weighted_mean(a["baseline"][jj], w),
                before_peak_hours_actually_rising_fraction=weighted_mean(pre_rise[ok], w),
                after_peak_hours_actually_falling_fraction=weighted_mean(post_fall[ok], w))
        output.append(item)
    return output


def profile(d, a, ids, full=True, draws=200):
    validids = ids[a["valid"][ids]]
    w = d["w"][ids]
    status = d["outage_status"][ids]
    counts = dict(no_observed_forecast=int((status == 0).sum()), observed_zero=int((status == 1).sum()),
                  observed_positive_flat=int((status == 2).sum()), observed_positive_varying=int((status == 3).sum()),
                  zero_with_complete_forecast=int(((status == 1)&(d["forecast_coverage"][ids] == 1)).sum()),
                  weather_absent=int((a["status"][ids] == 1).sum()), weather_flat=int((a["status"][ids] == 2).sum()),
                  weather_peak_defined=len(validids),
                  weather_peak_at_search_boundary=int(((a["first"][ids] == START)|(a["last"][ids] == STOP-1)).sum()))
    use = np.arange(len(OFFSETS)) if full else COARSE
    out = dict(support=support(d, ids), statuses=counts,
               fixed_forecast_mean_outage=signed_distribution(d["forecast_mean"][ids], w),
               forecast_observation_coverage_quantiles=weighted_quantiles(d["forecast_coverage"][ids], w),
               weather_peak_value_quantiles=weighted_quantiles(a["peak"][ids], w),
               positive_peak_index_fraction=weighted_mean((a["peak"][ids] > 0).astype(float), w),
               earlier_context_exceeds_forecast_weather_peak=int(a["earlier_context_exceeds_peak"][ids].sum()),
               aligned_support=support(d, validids), offsets_hours=OFFSETS[use],
               weather_peak_lag=lag_summary(d, a, ids))
    out["curves"] = {name: curve(a[key][:, use], validids, d, draws) for name, key in [
        ("outage_stock", "level"), ("outage_minus_pre_peak_mean", "departure"),
        ("net_stock_change_per_hour", "change")]}
    out["curves"]["anchor_weather_index"] = curve(a["index"][:, use], validids, d, 0)
    if full:
        wx = a["weather"][validids]
        mask = np.isfinite(wx)
        ww = d["w"][validids, None, None]
        den = np.sum(ww*mask, axis=0)
        out["all_raw_weather_aligned_means"] = np.divide(np.sum(ww*np.where(mask, wx, 0), 0), den,
            out=np.full(wx.shape[1:], np.nan), where=den > 0)
        out["phase_loop"] = phase_loops(d, a, ids)
    return out


def no_peak_profile(d, ids, draws):
    """Keep absent/flat weather outcomes visible without inventing a peak time."""
    hours = np.array([72, 96, 120, 144, 168, 192, 215])
    level = np.where(d["obs"][:, hours], d["y"][:, hours], np.nan)
    change = np.where(d["obs"][:, hours] & d["obs"][:, hours-1],
                      d["y"][:, hours]-d["y"][:, hours-1], np.nan)
    return dict(support=support(d, ids), fixed_window_hours=hours,
                forecast_mean_outage=signed_distribution(d["forecast_mean"][ids], d["w"][ids]),
                observed_zero_fraction=weighted_mean((d["outage_status"][ids] == 1).astype(float), d["w"][ids]),
                outage_stock=curve(level, ids, d, draws), net_stock_change_per_hour=curve(change, ids, d, draws))


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(f"{value:.7g}") if np.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=200)
    args = ap.parse_args()
    if os.getpriority(os.PRIO_PROCESS, 0) < 15:
        os.nice(15-os.getpriority(os.PRIO_PROCESS, 0))
    if OUTPUT.exists() or CACHE.exists():
        raise RuntimeError("D02 dynamics outputs already exist; preserve them before a new run")
    if not LOOKUP.exists():
        raise RuntimeError(f"County structure lookup is required: {LOOKUP}")
    start = time.time()
    d = load_data()
    allids = np.arange(len(d["y"]))
    result = dict(meta=dict(scope="D-only descriptive county-event dynamics; no predictive or causal evaluation",
        type_definition="Outcome-blind county geo/context structure, including historical reliability; continuous structure partitioned into six coarse types, not natural categories or purely physical geography",
        raw_weather_channels=d["names"], offsets_hours=OFFSETS,
        weather_search_hours=[START, STOP-1], weather_tie_rule="earliest maximum for alignment; first/last maxima retained for lag uncertainty",
        peak_status_codes={"0":"varying weather with a defined peak", "1":"zero/nonpositive rain/snow/CAPE", "2":"flat weather"},
        outage_status_codes={"0":"no observations", "1":"all observed values zero", "2":"flat observed positive stock", "3":"varying observed positive stock"},
        cold_definition="negative temperature; a maximum is the coolest hour, not necessarily freezing or hazardous",
        phase_loop_definition="Before/after peak within +/-24h and forecast hours, matched in four bins of each county-event weather range; at least two observed hours per branch",
        phase_loop_caution="Stock accumulation, restoration lag, elapsed phase, prior stock and other weather can generate the loop; net stock changes are not failure rates",
        lag_caution="Observed outage and weather peak intervals in the fixed window; not propagation speed, damage time, or a causal delay",
        weights="Trimmed existing D county-event design weights, normalized only within each descriptive cohort and valid lag; raw weights retained in cache",
        missingness="Out-of-window and unobserved hours remain missing; each curve reports its changing observed denominator",
        uncertainty=dict(method="whole merged-event-group bootstrap within each descriptive cohort", draws=args.bootstrap, seed=SEED,
            minimum_groups_per_lag=4, interpretation="Conditional on these weather anchors and county types; descriptive pointwise intervals, no multiplicity correction or significance gate"),
        counts=support(d, allids), source_sha256=digest(Path(__file__)), data_sha256=digest(FEATURES),
        county_lookup_sha256=d["lookup_sha256"]), weather_anchors={})
    cache = dict(county=d["county"], type=d["type"], system=d["system"], group=d["group"], regime=d["regime"],
                 w=d["w"], w_raw=d["w_raw"], offsets=OFFSETS, anchor_names=np.array([x[0] for x in ANCHORS]),
                 outage_status=d["outage_status"], outage_peak_first=d["outage_peak_first"],
                 outage_peak_last=d["outage_peak_last"], forecast_coverage=d["forecast_coverage"])
    for name, source, sign in ANCHORS:
        a = align_weather(d, source, sign, name)
        entry = dict(source=source, multiplier=sign, intensity_cutoffs=a["intensity_cutoffs"],
            intensity_definition="D design-weighted weather-only thirds among varying/defined peaks; no outcome-based threshold",
            overall=profile(d, a, allids, draws=args.bootstrap), by_type={}, by_regime={},
            by_type_regime={}, by_type_intensity={}, without_weather_peak={})
        for status, label in [(1, "absent_rain_snow_cape"), (2, "flat_weather")]:
            ids = np.flatnonzero(a["status"] == status)
            entry["without_weather_peak"][label] = dict(overall=no_peak_profile(d, ids, args.bootstrap), by_type={})
            for typ in range(6):
                ii = ids[d["type"][ids] == typ]
                entry["without_weather_peak"][label]["by_type"][f"T{typ}"] = no_peak_profile(d, ii, args.bootstrap)
        for typ in range(6):
            ids = np.flatnonzero(d["type"] == typ)
            entry["by_type"][f"T{typ}"] = profile(d, a, ids, draws=args.bootstrap)
            for regime in REGIMES:
                ix = np.flatnonzero((d["type"] == typ)&(d["regime"] == regime))
                entry["by_type_regime"][f"T{typ}/{regime}"] = profile(d, a, ix, full=False, draws=args.bootstrap)
            for q, label in enumerate(["low", "middle", "high"]):
                ix = np.flatnonzero((d["type"] == typ)&(a["intensity"] == q))
                entry["by_type_intensity"][f"T{typ}/{label}"] = profile(d, a, ix, full=False, draws=args.bootstrap)
        for regime in REGIMES:
            ids = np.flatnonzero(d["regime"] == regime)
            entry["by_regime"][regime] = profile(d, a, ids, draws=args.bootstrap)
        result["weather_anchors"][name] = entry
        for key in ["status", "first", "last", "peak", "floor", "baseline", "level", "change", "index", "intensity"]:
            value = a[key]
            cache[f"{name}_{key}"] = value.astype(np.float32) if value.dtype.kind == "f" else value
        print(json.dumps(dict(anchor=name, defined_peaks=int(a["valid"].sum()),
                              absent=int((a["status"] == 1).sum()), flat=int((a["status"] == 2).sum()),
                              elapsed_seconds=round(time.time()-start, 1))), flush=True)
    result["meta"]["elapsed_seconds"] = time.time()-start
    result["meta"]["cache_file"] = str(CACHE.relative_to(ROOT))
    cache["metadata_json"] = np.array(json.dumps(clean(result["meta"]), ensure_ascii=False))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **cache)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(clean(result), ensure_ascii=False, separators=(",", ":"), allow_nan=False)+"\n")
    print(json.dumps(dict(output=str(OUTPUT.relative_to(ROOT)), bytes=OUTPUT.stat().st_size,
                          cache_bytes=CACHE.stat().st_size, elapsed_seconds=round(time.time()-start, 1))), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
