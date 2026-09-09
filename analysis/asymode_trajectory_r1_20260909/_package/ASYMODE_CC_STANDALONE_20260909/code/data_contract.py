"""Read-only, leakage-limited data primitives. Main history length is 24, not zero."""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from functools import lru_cache

LEDGER_SHA256 = '03392a7f94bf9fbef6cf59c127afbd0ca829db27bcde0843fcaad666d5184e06'
PARTITIONS = [
    ('fold_A_fit', 2018, '2018-01-01', '2018-07-01', '2018-06-29'),
    ('fold_A_val', 2018, '2018-07-01', '2018-10-01', None),
    ('fold_B_fit', 2018, '2018-01-01', '2018-10-01', '2018-09-29'),
    ('fold_B_val', 2018, '2018-10-01', '2019-01-01', None),
    ('full_2018', 2018, '2018-01-01', '2019-01-01', None),
    ('reused_2019', 2019, '2019-01-01', '2020-01-01', None),
]


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_source(data_manifest: Path):
    """Read a verified local path manifest; no former handoff directory or aliases."""
    from prepare_local_data import validate_ledger, cohort, PACKAGE, file_sha
    info = json.loads(Path(data_manifest).read_text())
    if info.get('status') != 'PASS':
        raise ValueError('Data manifest has not passed local identity checks')
    ledger = Path(info['ledger_path'])
    if file_sha(ledger) != info['ledger_sha256']:
        raise ValueError('Local ledger changed after discovery')
    selection = PACKAGE/'metadata/unit_selection.csv'
    if file_sha(selection) != info['selection_sha256']:
        raise ValueError('Fixed cohort changed after discovery')
    validate_ledger(ledger)
    return np.load(ledger, allow_pickle=False), cohort()


@lru_cache(maxsize=4)
def calendar_grid(year: int):
    ts = pd.date_range(f'{year}-01-01', f'{year+1}-01-01', freq='h')
    cell = ts.hour.to_numpy()//6
    weekend = (ts.dayofweek.to_numpy() >= 5).astype(np.float64)
    known = np.column_stack([np.eye(4)[cell], weekend])
    inside = (ts.day.to_numpy() >= 2) & (ts.day.to_numpy() < ts.days_in_month.to_numpy())
    return ts, known, inside


def legal_origins(year: int, start: str, end: str, fit_target_cap: str|None,
                  lookback: int = 24, horizon: int = 24) -> np.ndarray:
    """Every observed history state and every forecast target lies inside partition/mask.
    Fit's last target is <= validation start minus 48h. Validation retains nominal dates.
    """
    ts, _, inside = calendar_grid(year)
    ok = inside & (ts >= pd.Timestamp(start)) & (ts < pd.Timestamp(end))
    if fit_target_cap is not None:
        ok &= ts <= pd.Timestamp(fit_target_cap)
    origins = np.arange(lookback, len(ts)-horizon, dtype=np.int64)
    bad_prefix = np.r_[0, np.cumsum(~ok)]
    n_bad = bad_prefix[origins+horizon+1] - bad_prefix[origins-lookback]
    return origins[n_bad == 0]


def company_group_probabilities(selected: pd.DataFrame) -> np.ndarray:
    sizes = selected.groupby('company').size()
    p = np.array([1/(len(sizes)*sizes[c]) for c in selected.company], dtype=np.float64)
    return p/p.sum()


def fit_state_stats(ledger, selected: pd.DataFrame, origins: np.ndarray, lookback: int = 24):
    """Fit-only RMS and mean, company-equal, based on AVAILABLE state histories/y0.
    Does not touch target-year values, direction arrays, or old frozen rates.
    """
    p = company_group_probabilities(selected)
    ms, sq = [], []
    offsets = np.arange(-lookback, 1)
    for g in selected.group:
        values = ledger[f'g{int(g)}_y2018_y'][origins[:,None]+offsets]
        ms.append(float(values.mean())); sq.append(float(np.square(values).mean()))
    return {'state_scale': max(float(np.sqrt(p@np.array(sq))), 1e-6),
            'source_mean': float(p@np.array(ms))}


def assemble_batch(ledger, selected: pd.DataFrame, year: int, group_indices: np.ndarray,
                   hours: np.ndarray, state_scale: float, lookback: int = 24, horizon: int = 24):
    if len(group_indices) != len(hours):
        raise ValueError('Mismatched batch IDs')
    _, clock, _ = calendar_grid(year)
    groups = selected.group.to_numpy(dtype=int)
    B = len(hours)
    history = np.empty((B, lookback), np.float64)
    states = np.empty((B, horizon+1), np.float64)
    # Group only, never infer contiguous group IDs or reshape compressed valid hours.
    for gi in np.unique(group_indices):
        ix = np.flatnonzero(group_indices == gi)
        y = ledger[f'g{groups[gi]}_y{year}_y']
        history[ix] = y[hours[ix, None]+np.arange(-lookback, 0)]
        states[ix] = y[hours[ix, None]+np.arange(horizon+1)]
    known_clock = clock[hours[:, None]+np.arange(horizon)]
    c = np.concatenate([np.eye(len(groups))[group_indices], history/state_scale,
                        known_clock.reshape(B, -1)], axis=1)
    return {'c': c, 'y0': states[:,0], 'known_clock': known_clock,
            'target': states[:,1:], 'true_states_for_step_ablation_only': states}


def preload_stocks(ledger, selected: pd.DataFrame, year: int) -> dict[str, np.ndarray]:
    """Materialize only stock arrays once; do not decompress NPZ in the training hot path.
    Training calls this only with year=2018. Evaluation loads 2019 after selection_lock.
    """
    data = {}
    for g in selected.group:
        key = f'g{int(g)}_y{int(year)}_y'
        x = np.ascontiguousarray(ledger[key], dtype=np.float64)
        x.setflags(write=False)
        data[key] = x
    return data
