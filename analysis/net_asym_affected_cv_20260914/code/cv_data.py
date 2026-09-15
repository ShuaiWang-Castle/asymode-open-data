"""Grouped cross-validation data for the affected-county NET/ASYM round (PI decisions of 2026-09-14).

The US data contract of ../net_asym_neural_era5_20260912 is reused unchanged: exact UTC top-of-hour
snapshots, the legal-state and legal-weather rules, L=24 and H=24, 6-hour origins and its deduplicated
window index. This module adds three things.

* Event types: 'dominant' of data/interim/event_days_stratified.parquet, grouped for stratification into
  winter (winter), wind_tropical (wind, tropical) and convective_flood (convective, flood).
* Folds: the overlap components of the 26 events. Each component takes the type group of its event with
  the most windows, components are sorted by first event date within group and assigned round-robin to
  K=5 folds, and the rotation continues across groups in the order winter, wind_tropical,
  convective_flood. For outer fold k the inner validation fold is (k+1) mod K; the other three folds train.
* Affected county-event: peak legal snapshot over the 169-hour event panel >= 0.01. Training and inner
  validation use affected county-events only. Test scoring covers every legal window and reports all,
  affected and unaffected county-events separately. The affected label uses realized outages, so test
  results on the affected population are conditional on realized impact.
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch

HERE = Path(__file__).resolve().parent
ROUND_DIR = HERE.parents[1] / 'net_asym_neural_era5_20260912'
sys.path.insert(0, str(ROUND_DIR / 'code'))
import us_data as UD   # noqa: E402

K_FOLDS = 5
AFFECTED_PEAK = 0.01
TYPE_GROUP = {'winter': 'winter', 'wind': 'wind_tropical', 'tropical': 'wind_tropical',
              'convective': 'convective_flood', 'flood': 'convective_flood'}
GROUP_ORDER = ('winter', 'wind_tropical', 'convective_flood')


def sha_file(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def event_types(root: Path) -> dict:
    d = pd.read_parquet(Path(root) / 'data/interim/event_days_stratified.parquet')
    return dict(zip(d['day'].astype(str).str[:10], d['dominant'].astype(str)))


def build_fold_lock(root: Path, out_locks: Path) -> dict:
    root = Path(root); c = UD.load_corpus(root)
    idx = pd.read_csv(ROUND_DIR / 'locks/US_WINDOW_INDEX.csv.gz', dtype={'fips': str, 'event': str, 'origin_utc': str})
    spl = json.loads((ROUND_DIR / 'locks/SPLITS.json').read_text())['splits']
    for s, g in idx.groupby('split'):
        if UD.window_hash(g) != spl[s]['window_id_sha256']:
            raise ValueError(f'{s}: window index differs from the round lock')
    types = event_types(root)
    missing = [e for e in c.events if e not in types]
    if missing:
        raise ValueError(f'no event type for {missing}')
    nw = idx.groupby('event').size()
    comp_rows = []
    for cc in UD.overlap_components(c, list(c.events)):
        lead = max(cc, key=lambda e: (int(nw.get(e, 0)), -c.events.index(e)))
        comp_rows.append({'events': cc, 'first_event': min(cc), 'group': TYPE_GROUP[types[lead]], 'group_from_event': lead})
    offset = 0
    for grp in GROUP_ORDER:
        members = sorted([r for r in comp_rows if r['group'] == grp], key=lambda r: r['first_event'])
        for i, r in enumerate(members):
            r['fold'] = (offset + i) % K_FOLDS
        offset += len(members)
    fold_of_event = {e: r['fold'] for r in comp_rows for e in r['events']}

    peak = np.full(len(c.row_event), np.nan)
    for r in range(len(c.row_event)):
        if np.isfinite(c.Y[r]).any():
            peak[r] = np.nanmax(c.Y[r])
    w = idx.drop(columns=['split']).copy()
    w['fold'] = w.event.map(fold_of_event).astype(int)
    w['type'] = w.event.map(types)
    w['type_group'] = w.type.map(TYPE_GROUP)
    w['peak'] = peak[w.row.to_numpy()]
    w['affected'] = w.peak >= AFFECTED_PEAK
    w = w.sort_values(['event', 'fips', 't'], kind='mergesort').reset_index(drop=True)

    # no (county, hour) state or weather cell may be used by windows of two different folds
    hours = np.stack([np.asarray(ts.asi8 // 3_600_000_000_000, dtype=np.int64) for ts in c.ts_hour])
    fips_int = np.array([int(f) for f in c.row_fips], dtype=np.int64)
    off = np.arange(-UD.L_HIST, UD.H_FUT + 1)
    cells = {}
    for k, g in w.groupby('fold'):
        r, t = g.row.to_numpy(), g.t.to_numpy()
        hh = hours[c.row_event[r].astype(np.int64)[:, None], t[:, None] + off[None, :]]
        cells[int(k)] = np.unique((fips_int[r][:, None] * 1_000_000 + hh).ravel())
    shared = {f'{a}&{b}': int(len(np.intersect1d(cells[a], cells[b], assume_unique=True)))
              for a in range(K_FOLDS) for b in range(K_FOLDS) if a < b}

    roles = {k: {'test_fold': k, 'validation_fold': (k + 1) % K_FOLDS,
                 'train_folds': [j for j in range(K_FOLDS) if j not in (k, (k + 1) % K_FOLDS)]} for k in range(K_FOLDS)}
    per_fold = []
    for k in range(K_FOLDS):
        g = w[w.fold == k]
        per_fold.append({'fold': k, 'events': sorted(g.event.unique().tolist()),
                         'components': [r['events'] for r in comp_rows if r['fold'] == k],
                         'event_types': {e: types[e] for e in sorted(g.event.unique())},
                         'windows_all': int(len(g)), 'windows_affected': int(g.affected.sum()),
                         'county_events_all': int(g.groupby(['event', 'fips']).ngroups),
                         'county_events_affected': int(g[g.affected].groupby(['event', 'fips']).ngroups)})
    out_locks = Path(out_locks); out_locks.mkdir(parents=True, exist_ok=True)
    cols = ['event', 'event_idx', 'row', 't', 'fips', 'origin_utc', 'fold', 'type', 'type_group', 'peak', 'affected']
    w[cols].to_csv(out_locks / 'CV_WINDOW_INDEX.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    lock = {'information_contract': 'RETROSPECTIVE_KNOWN_REANALYSIS_CONDITIONAL_RESPONSE',
            'data_contract': 'unchanged from ../net_asym_neural_era5_20260912 (states, weather legality, windows, dedup)',
            'round_window_index_sha256': sha_file(ROUND_DIR / 'locks/US_WINDOW_INDEX.csv.gz'),
            'K': K_FOLDS, 'affected_rule': 'peak legal snapshot over the 169-hour event panel >= 0.01',
            'affected_use': 'training and inner validation: affected only; test: all, reported as all / affected / unaffected',
            'type_source': 'data/interim/event_days_stratified.parquet column dominant',
            'type_group_map': TYPE_GROUP, 'group_order': list(GROUP_ORDER),
            'fold_rule': 'components labelled by the type group of their event with most windows; sorted by first event date '
                         'within group; round-robin over K folds continuing across groups in group_order',
            'role_rule': 'outer fold k tests; fold (k+1) mod K is inner validation; the remaining folds train',
            'event_types': {e: types[e] for e in c.events}, 'components': comp_rows, 'roles': roles, 'folds': per_fold,
            'cross_fold_shared_cells': shared,
            'cross_fold_leakage': 'NONE' if all(v == 0 for v in shared.values()) else 'LEAK',
            'cv_window_index_sha256': sha_file(out_locks / 'CV_WINDOW_INDEX.csv.gz')}
    (out_locks / 'FOLDS.json').write_text(json.dumps(lock, indent=1))
    return lock


class CVData:
    """Fold views over one loaded corpus. Fit statistics come from the fold's training windows only."""

    def __init__(self, root: Path, locks: Path):
        locks = Path(locks)
        self.lock = json.loads((locks / 'FOLDS.json').read_text())
        if sha_file(locks / 'CV_WINDOW_INDEX.csv.gz') != self.lock['cv_window_index_sha256']:
            raise ValueError('CV window index differs from FOLDS.json')
        self.lock_sha = sha_file(locks / 'FOLDS.json')
        self.idx = pd.read_csv(locks / 'CV_WINDOW_INDEX.csv.gz',
                               dtype={'fips': str, 'event': str, 'origin_utc': str, 'type': str, 'type_group': str})
        self.c = UD.load_corpus(Path(root))
        if not (self.c.row_fips[self.idx.row.to_numpy()] == self.idx.fips.to_numpy()).all():
            raise ValueError('row/FIPS mismatch between corpus and CV window index')
        self.Y32 = torch.from_numpy(np.nan_to_num(self.c.Y, nan=0.0).astype(np.float32))
        ev = self.c.row_event.astype(np.int64); hrs = self.c.hour_utc[ev].astype(np.float64)
        self.sin = torch.from_numpy(np.sin(2 * np.pi * hrs / 24).astype(np.float32))
        self.cos = torch.from_numpy(np.cos(2 * np.pi * hrs / 24).astype(np.float32))
        self.wkd = torch.from_numpy(self.c.weekend[ev].astype(np.float32))
        self.off_hist, self.off_pw, self.off_fw = torch.arange(-24, 0), torch.arange(-23, 1), torch.arange(1, 25)
        self.fold = None

    def set_fold(self, k: int):
        r = self.lock['roles'][str(k)]; w = self.idx
        self.df = {'train': w[w.fold.isin(r['train_folds']) & w.affected].reset_index(drop=True),
                   'validation': w[(w.fold == r['validation_fold']) & w.affected].reset_index(drop=True),
                   'test': w[w.fold == r['test_fold']].reset_index(drop=True)}
        for s, d in self.df.items():
            if len(d) == 0:
                raise ValueError(f'fold {k} {s}: no windows')
            if not np.isfinite(self._gather_Y(d, np.arange(-24, 25))).all():
                raise ValueError(f'fold {k} {s}: illegal state inside a window')
            if not self.c.weather_legal[d.row.to_numpy()].all():
                raise ValueError(f'fold {k} {s}: missing-weather county inside a window')
        self.weights = {s: self._weights(d) for s, d in self.df.items()}
        self._fit_statistics()
        self.Xs = torch.from_numpy(((self.c.X - self.wx_mean) / self.wx_std).astype(np.float32))
        self.rows = {s: torch.from_numpy(d.row.to_numpy().astype(np.int64)) for s, d in self.df.items()}
        self.ts = {s: torch.from_numpy(d.t.to_numpy().astype(np.int64)) for s, d in self.df.items()}
        self.fold = k

    @staticmethod
    def _weights(d):
        n_origin = d.groupby(['event', 'fips']).t.transform('size').to_numpy(float)
        n_county = d.groupby('event').fips.transform('nunique').to_numpy(float)
        w = 1.0 / (float(d.event.nunique()) * n_county * n_origin)
        assert abs(w.sum() - 1.0) < 1e-9
        return w

    def _gather_Y(self, d, offsets):
        r, t = d.row.to_numpy(), d.t.to_numpy()
        return self.c.Y[r[:, None], t[:, None] + offsets[None, :]]

    def _fit_statistics(self):
        d, w = self.df['train'], self.weights['train']
        Ys = self._gather_Y(d, np.arange(-24, 1))
        self.scale = float(np.sqrt(np.sum(w * np.mean(Ys ** 2, axis=1))))
        self.source_mean = float(np.sum(w * np.mean(Ys, axis=1)))
        r, t = d.row.to_numpy(), d.t.to_numpy()
        m1, m2, off = np.zeros(12), np.zeros(12), np.arange(-23, 25)[None, :]
        for i in range(0, len(d), 8192):
            Xw = self.c.X[r[i:i + 8192, None], t[i:i + 8192, None] + off].astype(np.float64)
            m1 += np.einsum('i,ijk->k', w[i:i + 8192], Xw) / 48.0
            m2 += np.einsum('i,ijk->k', w[i:i + 8192], Xw ** 2) / 48.0
        self.wx_mean, self.wx_std = m1, np.sqrt(np.maximum(m2 - m1 ** 2, 0.0))
        if not (self.scale > 0 and (self.wx_std > 0).all()):
            raise ValueError('degenerate fit statistics')

    def batch(self, split, ids):
        r, t = self.rows[split][ids], self.ts[split][ids]
        R_, T_ = r[:, None], t[:, None]
        B, tf = len(ids), T_ + self.off_fw
        hist = self.Y32[R_, T_ + self.off_hist] / self.scale
        pw, fw = self.Xs[R_, T_ + self.off_pw], self.Xs[R_, tf]
        step = torch.cat([fw, self.sin[R_, tf][..., None], self.cos[R_, tf][..., None], self.wkd[R_, tf][..., None]], dim=-1)
        c = torch.cat([hist, pw.reshape(B, -1), fw.reshape(B, -1)], dim=1)
        return c, self.Y32[r, t], step, self.Y32[R_, tf]

    def truth64(self, split): return self._gather_Y(self.df[split], np.arange(1, 25))

    def y0_64(self, split):
        d = self.df[split]; return self.c.Y[d.row.to_numpy(), d.t.to_numpy()]


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--root', required=True); ap.add_argument('--locks', required=True)
    a = ap.parse_args()
    lock = build_fold_lock(Path(a.root), Path(a.locks))
    for f in lock['folds']:
        print(f"fold {f['fold']}: {f['event_types']} windows all={f['windows_all']} affected={f['windows_affected']} "
              f"county-events all={f['county_events_all']} affected={f['county_events_affected']}")
    print('cross-fold leakage:', lock['cross_fold_leakage'], lock['cross_fold_shared_cells'])
