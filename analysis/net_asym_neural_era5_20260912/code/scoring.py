"""Paired scoring utilities. Squared errors are always formed per seed first.

Delta = MSE_NET - MSE_ASYM. Intervals are descriptive dependence sensitivity on
retrospective data; they are not fresh-data significance tests.
"""
from __future__ import annotations
import itertools
import numpy as np, pandas as pd


def event_county_origin_weights(event: np.ndarray, county: np.ndarray) -> np.ndarray:
    df = pd.DataFrame({'e': event, 'c': county})
    n_origin = df.groupby(['e', 'c']).e.transform('size').to_numpy(float)
    n_county = df.groupby('e').c.transform('nunique').to_numpy(float)
    return 1.0 / (df.e.nunique() * n_county * n_origin)


def event_level_mse(se_window: np.ndarray, event: np.ndarray, county: np.ndarray) -> pd.Series:
    """Per-window loss -> county mean over origins -> event mean over counties."""
    df = pd.DataFrame({'e': event, 'c': county, 'l': se_window})
    return df.groupby(['e', 'c']).l.mean().groupby('e').mean()


def seed_mean_and_ensemble_risk(preds: list, truth: np.ndarray, w: np.ndarray) -> dict:
    per_seed = [float(np.sum(w * ((p - truth) ** 2).mean(axis=1))) for p in preds]
    ens = np.mean(preds, axis=0)
    return {'mean_of_seed_risks': float(np.mean(per_seed)),
            'risk_of_seed_ensemble': float(np.sum(w * ((ens - truth) ** 2).mean(axis=1)))}


def exact_signflip_p(x: np.ndarray) -> float:
    x = np.asarray(x, float); obs = abs(x.mean())
    vals = [abs(np.mean(x * np.array(s))) for s in itertools.product((-1.0, 1.0), repeat=len(x))]
    return float(np.mean(np.array(vals) >= obs - 1e-15))


def cluster_bootstrap(delta_seed_event: pd.DataFrame, components: list, n_boot: int = 4000, seed: int = 20260912) -> dict:
    """delta_seed_event: rows = seeds, columns = events, values = event-level Delta.

    Resample overlap components with replacement; each draw averages Delta over the
    events of the drawn components (with multiplicity), per seed, then over seeds."""
    comps = [list(c) for c in components]
    point = float(delta_seed_event.mean(axis=1).mean())
    rng = np.random.default_rng(seed); draws = []
    for _ in range(n_boot):
        ev = [e for k in rng.integers(0, len(comps), len(comps)) for e in comps[k]]
        draws.append(float(delta_seed_event[ev].mean(axis=1).mean()))
    comp_delta = np.array([float(delta_seed_event[c].mean(axis=1).mean()) for c in comps])
    return {'point': point, 'p2.5': float(np.percentile(draws, 2.5)), 'p50': float(np.percentile(draws, 50)),
            'p97.5': float(np.percentile(draws, 97.5)), 'n_components': len(comps),
            'component_delta': comp_delta.tolist(), 'components_positive': int((comp_delta > 0).sum()),
            'exact_signflip_p_over_components': exact_signflip_p(comp_delta),
            'min_attainable_p': 2.0 / 2 ** len(comps)}
