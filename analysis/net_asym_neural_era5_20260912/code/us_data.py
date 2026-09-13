"""US EAGLE-I + ERA5 data contract for the NET/ASYM round.

Information contract: RETROSPECTIVE_KNOWN_REANALYSIS_CONDITIONAL_RESPONSE. Both
models receive the ERA5 reanalysis path for the next 24 hours. This is a
retrospective conditional-response experiment, not a deployment forecast.

States are EXACT UTC top-of-hour snapshots of the 15-minute stock: y_h = y[4h].
No hourly mean replaces the stock.

Input legality (identical for both models; defined on inputs and masks only):
  * a state is legal iff its snapshot is observed AND finite;
  * a county's weather is legal iff its driver tensor is NOT identically zero in
    all 12 channels at all hours. The published drivers encode counties with no
    county-to-grid mapping as exact zeros (verified: temperature and pressure are
    both identically 0 for exactly those counties). Missing weather is never fed
    to a model as zero;
  * a window at origin t is legal iff states t-24..t+24 (49) are legal, weather
    hours t-23..t+24 are finite, and the county's weather is legal.
"""
from __future__ import annotations
import hashlib, json
from dataclasses import dataclass
from pathlib import Path
import numpy as np, pandas as pd

RAW_CHANNELS = ['cape', 'cloud', 'gust', 'precip', 'pressure', 'rh', 'snowfall',
                'soil_moisture', 't2m_c', 'u10', 'v10', 'wind_speed']
L_HIST, H_FUT, ORIGIN_STRIDE = 24, 24, 6
SPLIT_YEARS = {'train': (2018, 2019, 2020, 2021), 'validation': (2022,), 'evaluation': (2024,)}
N_HOURS = 169


@dataclass
class Corpus:
    events: list
    event_year: np.ndarray
    event_start: list
    event_end: list
    row_event: np.ndarray       # (R,)
    row_fips: np.ndarray        # (R,)
    Y: np.ndarray               # (R, 169) float64 snapshot, NaN where state illegal
    state_legal: np.ndarray     # (R, 169) bool
    X: np.ndarray               # (R, 169, 12) float32 raw ERA5-derived
    weather_legal: np.ndarray   # (R,) bool
    hour_utc: np.ndarray        # (E, 169) int
    weekend: np.ndarray         # (E, 169) bool
    ts_hour: list               # per event DatetimeIndex (169)


def load_corpus(root: Path) -> Corpus:
    root = Path(root); I = root / 'data/interim'
    man = json.loads((root / 'configs/panel_manifest_g3-all-26.json').read_text())
    if man['digest'] != 'db286b4960a4':
        raise ValueError('manifest digest mismatch')
    Ys, Ls, Xs, WL, RE, RF, HU, WK, TS, ST, EN = [], [], [], [], [], [], [], [], [], [], []
    for ei, e in enumerate(man['panels']):
        p = np.load(I / f'panel_{e}.npz', allow_pickle=True)
        d = np.load(I / f'drivers_{e}.npz', allow_pickle=True)
        pf, df_ = p['fips'].astype(str), d['fips'].astype(str)
        if [str(c) for c in d['channels']] != RAW_CHANNELS:
            raise ValueError(f'{e}: channel order differs')
        X = d['X']
        if not np.array_equal(pf, df_):                       # align by key, never by row count
            pos = {f: i for i, f in enumerate(df_)}
            if set(pf) != set(df_):
                raise ValueError(f'{e}: county sets differ')
            X = X[[pos[f] for f in pf]]
        pts = pd.to_datetime([str(t) for t in p['ts']])
        dts = pd.to_datetime([str(t) for t in d['ts']])
        snap = pts[::4]
        if len(snap) != N_HOURS or not (snap == dts).all() or pts[0].minute != 0:
            raise ValueError(f'{e}: time grids do not align on exact hourly snapshots')
        y = p['y'][:, ::4].astype(np.float64); o = p['observed'][:, ::4].astype(bool)
        legal = o & np.isfinite(y)
        Ys.append(np.where(legal, y, np.nan)); Ls.append(legal); Xs.append(X.astype(np.float32))
        WL.append(~(X == 0).all(axis=(1, 2)) & np.isfinite(X).all(axis=(1, 2)))
        RE.append(np.full(len(pf), ei, np.int16)); RF.append(pf)
        HU.append(np.asarray(dts.hour, dtype=np.int64)); WK.append(np.asarray(dts.dayofweek >= 5, dtype=bool))
        TS.append(dts); ST.append(dts[0]); EN.append(dts[-1])
    return Corpus(list(man['panels']), np.array([int(e[:4]) for e in man['panels']]), ST, EN,
                  np.concatenate(RE), np.concatenate(RF), np.concatenate(Ys), np.concatenate(Ls),
                  np.concatenate(Xs), np.concatenate(WL), np.stack(HU), np.stack(WK), TS)


def split_of_year(y: int) -> str:
    for s, ys in SPLIT_YEARS.items():
        if y in ys:
            return s
    return 'unassigned'


def enumerate_windows(c: Corpus) -> pd.DataFrame:
    R = len(c.row_event)
    bad = np.concatenate([np.zeros((R, 1), np.int32), np.cumsum(~c.state_legal, axis=1, dtype=np.int32)], axis=1)
    xbad = np.concatenate([np.zeros((R, 1), np.int32),
                           np.cumsum(~np.isfinite(c.X).all(axis=2), axis=1, dtype=np.int32)], axis=1)
    recs = []
    cand_t = np.arange(L_HIST, N_HOURS - H_FUT)                    # 24..144
    for t in cand_t:
        states_ok = (bad[:, t + H_FUT + 1] - bad[:, t - L_HIST]) == 0
        wx_ok = (xbad[:, t + H_FUT + 1] - xbad[:, t - L_HIST + 1]) == 0
        recs.append(pd.DataFrame({'row': np.arange(R, dtype=np.int32), 't': np.int16(t),
                                  'states_legal': states_ok, 'weather_finite': wx_ok}))
    w = pd.concat(recs, ignore_index=True)
    w['event_idx'] = c.row_event[w.row.to_numpy()]
    w['hour_utc'] = c.hour_utc[w.event_idx.to_numpy(), w.t.to_numpy()]
    w = w[w.hour_utc % ORIGIN_STRIDE == 0].copy()
    w['weather_legal'] = c.weather_legal[w.row.to_numpy()]
    w['fips'] = c.row_fips[w.row.to_numpy()]
    w['event'] = [c.events[i] for i in w.event_idx.to_numpy()]
    w['origin_utc'] = [str(c.ts_hour[e][t]) for e, t in zip(w.event_idx.to_numpy(), w.t.to_numpy())]
    w['split'] = [split_of_year(int(c.events[e][:4])) for e in w.event_idx.to_numpy()]
    w['legal'] = w.states_legal & w.weather_finite & w.weather_legal
    return w.reset_index(drop=True)


def deduplicate(w: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep one (fips, origin_utc); the earlier event_day owns it."""
    lw = w[w.legal].sort_values(['event', 'row', 't'], kind='mergesort')
    first = ~lw.duplicated(['fips', 'origin_utc'], keep='first')
    kept, dropped = lw[first], lw[~first]
    owner = kept.set_index(['fips', 'origin_utc']).event
    mapping = dropped[['event', 'fips', 'origin_utc', 'row', 't']].copy()
    mapping['kept_in_event'] = [owner[(f, o)] for f, o in zip(mapping.fips, mapping.origin_utc)]
    return kept.reset_index(drop=True), mapping.reset_index(drop=True)


def overlap_components(c: Corpus, events: list) -> list:
    """Connected components of events whose panel time windows intersect."""
    idx = {e: i for i, e in enumerate(events)}
    parent = list(range(len(events)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for a in events:
        for b in events:
            if a < b:
                ia, ib = c.events.index(a), c.events.index(b)
                if c.event_start[ia] <= c.event_end[ib] and c.event_start[ib] <= c.event_end[ia]:
                    parent[find(idx[a])] = find(idx[b])
    comps = {}
    for e in events:
        comps.setdefault(find(idx[e]), []).append(e)
    return sorted([sorted(v) for v in comps.values()])


def window_hash(df: pd.DataFrame) -> str:
    keys = sorted(f'{e}|{f}|{o}' for e, f, o in zip(df.event, df.fips, df.origin_utc))
    return hashlib.sha256('\n'.join(keys).encode()).hexdigest()


def build_splits(root: Path, out_locks: Path, out_intake: Path) -> dict:
    c = load_corpus(root)
    w = enumerate_windows(c)
    kept, mapping = deduplicate(w)
    # cross-split leakage: no (fips, hour) state or weather hour may be shared across splits
    used = {}
    for s, g in kept.groupby('split'):
        cells = set()
        for r, t in zip(g.row.to_numpy(), g.t.to_numpy()):
            e = int(c.row_event[r]); f = c.row_fips[r]
            for h in range(t - L_HIST, t + H_FUT + 1):
                cells.add((f, str(c.ts_hour[e][h])))
        used[s] = cells
    leak = {f'{a}&{b}': len(used[a] & used[b]) for a in used for b in used if a < b}
    per_event = []
    for ei, e in enumerate(c.events):
        rows = np.flatnonzero(c.row_event == ei)
        we = w[w.event_idx == ei]
        per_event.append({'event': e, 'split': split_of_year(int(e[:4])), 'counties': int(len(rows)),
                          'counties_weather_legal': int(c.weather_legal[rows].sum()),
                          'candidate_windows': int(len(we)),
                          'legal_states_only': int(we.states_legal.sum()),
                          'legal_states_and_weather': int(we.legal.sum()),
                          'excluded_by_missing_weather_only': int((we.states_legal & we.weather_finite & ~we.weather_legal).sum()),
                          'dedup_dropped': int((mapping.event == e).sum()),
                          'final_windows': int((kept.event == e).sum()),
                          'final_counties': int(kept[kept.event == e].fips.nunique())})
    pe = pd.DataFrame(per_event)
    splits = {}
    for s in SPLIT_YEARS:
        ev = [e for e in c.events if split_of_year(int(e[:4])) == s]
        g = kept[kept.split == s]
        splits[s] = {'years': list(SPLIT_YEARS[s]), 'events': ev, 'n_events': len(ev),
                     'windows': int(len(g)), 'county_event_pairs': int(g.groupby(['event', 'fips']).ngroups),
                     'window_id_sha256': window_hash(g),
                     'overlap_components': overlap_components(c, ev)}
    lock = {'information_contract': 'RETROSPECTIVE_KNOWN_REANALYSIS_CONDITIONAL_RESPONSE',
            'state_rule': 'exact UTC top-of-hour snapshot y[4h]; legal iff observed and finite',
            'weather_rule': 'county weather illegal iff all 12 channels are exactly 0 at all 169 hours',
            'window': {'L': L_HIST, 'H': H_FUT, 'origin_stride_hours': ORIGIN_STRIDE,
                       'origin_hour_range': [L_HIST, N_HOURS - 1 - H_FUT],
                       'states': 't-24..t+24 (49)', 'past_weather_hours': 't-23..t',
                       'future_weather_hours': 't+1..t+24', 'step_k_weather_hour': 't+k+1'},
            'dedup_rule': '(fips, origin_utc) kept once, earliest event_day owns it',
            'splits': splits, 'cross_split_shared_cells': leak,
            'cross_split_leakage': 'NONE' if all(v == 0 for v in leak.values()) else 'LEAK',
            'dedup_dropped_windows': int(len(mapping)),
            'expected_events': {'train': 14, 'validation': 6, 'evaluation': 6},
            'events_match_expected': all(splits[s]['n_events'] == n for s, n in
                                         {'train': 14, 'validation': 6, 'evaluation': 6}.items())}
    out_locks.mkdir(parents=True, exist_ok=True); out_intake.mkdir(parents=True, exist_ok=True)
    (out_locks / 'SPLITS.json').write_text(json.dumps(lock, indent=1))
    kept[['split', 'event', 'event_idx', 'row', 't', 'fips', 'origin_utc']].to_csv(
        out_locks / 'US_WINDOW_INDEX.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    mapping.to_csv(out_locks / 'US_DEDUP_MAPPING.csv', index=False)
    pe.to_csv(out_intake / 'US_WINDOW_COUNTS_BY_EVENT.csv', index=False)
    return lock, pe


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    for x in ('root', 'locks', 'intake'):
        ap.add_argument(f'--{x}', required=True)
    a = ap.parse_args()
    lock, pe = build_splits(Path(a.root), Path(a.locks), Path(a.intake))
    pd.set_option('display.width', 220)
    print(pe.to_string(index=False))
    print(json.dumps({s: {k: v for k, v in d.items() if k != 'events'} for s, d in lock['splits'].items()}, indent=1))
    print('cross-split leakage:', lock['cross_split_leakage'], lock['cross_split_shared_cells'])
    print('dedup dropped:', lock['dedup_dropped_windows'], ' events match 14/6/6:', lock['events_match_expected'])
