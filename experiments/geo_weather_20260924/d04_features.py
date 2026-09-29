"""Training-only nonlinear lag bases for D04. No targets enter weather/geography bases."""
from __future__ import annotations

from dataclasses import dataclass
import itertools
import numpy as np
from county_structure_d02 import BLOCKS

LAGS = ((0, 6), (6, 24), (24, 48))
CROSS = list(itertools.combinations(range(12), 2))
WITH_SELF = list(itertools.combinations_with_replacement(range(12), 2))
LAG_PAIRS = list(itertools.combinations(range(3), 2))
REGIMES = ['tropical', 'winter', 'synoptic_wind', 'convective', 'heavy_rain']
MAIN_DIM = 144
PAIR_DIM = 630


def weighted_quantile(x, w, q):
    keep = (w > 0) & np.isfinite(x)
    v, z = x[keep], w[keep]
    order = np.argsort(v, kind='stable')
    v, z = v[order], z[order]
    return np.interp(np.asarray(q) * z.sum(), np.cumsum(z), v)


def weighted_scale(x, w):
    w = np.asarray(w, dtype=np.float64); w = w / w.sum()
    mu = np.einsum('n,nj->j', w, x, dtype=np.float64)
    second = np.einsum('n,nj,nj->j', w, x, x, dtype=np.float64)
    sd = np.sqrt(np.maximum(second - mu * mu, 0))
    active = sd > 1e-7
    sd[~active] = 1
    return mu.astype('f4'), sd.astype('f4'), active


def transform_weather(panel):
    x = panel['weather'].copy()
    for name in ('cape', 'precip', 'snowfall'):
        j = panel['feature_names']['weather'].index(name)
        x[:, :, j] = np.log1p(np.maximum(x[:, :, j], 0))
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite weather')
    return x


def label_features(names):
    labels = []
    for lo, hi in LAGS:
        for name in names:
            labels.extend(f'main:{name}:past_{lo+1}_{hi}h:{basis}'
                          for basis in ('linear', 'square', 'hinge_q50', 'hinge_q90'))
    for lo, hi in LAGS:
        labels.extend(f'synchronous:{names[a]}*{names[b]}:past_{lo+1}_{hi}h' for a, b in CROSS)
    for near, far in LAG_PAIRS:
        labels.extend(f'symmetric_history:{names[a]}*{names[b]}:bands_{near}_{far}'
                      for a, b in WITH_SELF)
    for near, far in LAG_PAIRS:
        labels.extend(f'ordered_history:{names[a]}->{names[b]}:bands_{far}_to_{near}'
                      for a, b in CROSS)
    assert len(labels) == MAIN_DIM + PAIR_DIM
    return labels


def raw_weather_features(wx, rows, mu, sd, knots, chunk=4096):
    """Strict past: column lag l is weather at target t-1-l; no current hour."""
    n = len(rows['unit'])
    out = np.empty((n, MAIN_DIM + PAIR_DIM), dtype='f4')
    aa = np.array([p[0] for p in CROSS]); bb = np.array([p[1] for p in CROSS])
    sa = np.array([p[0] for p in WITH_SELF]); sb = np.array([p[1] for p in WITH_SELF])
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        u, t = rows['unit'][start:stop], rows['time'][start:stop]
        idx = t[:, None] - 1 - np.arange(48)[None]
        z = (wx[u[:, None], idx] - mu) / sd
        means, main, sync = [], [], []
        for lo, hi in LAGS:
            zz = z[:, lo:hi]
            means.append(zz.mean(1))
            basis = np.stack([zz.mean(1), (zz * zz).mean(1),
                              np.maximum(zz-knots[0], 0).mean(1),
                              np.maximum(zz-knots[1], 0).mean(1)], axis=2)
            main.append(basis.reshape(len(u), -1))
            # Average pointwise products, NOT product of averaged exposures.
            sync.append(np.einsum('nta,ntb->nab', zz, zz, optimize=True)[:, aa, bb] / (hi-lo))
        means = np.stack(means, axis=1)
        sym, order = [], []
        for near, far in LAG_PAIRS:
            a, b = means[:, far], means[:, near]
            s = (a[:, sa]*b[:, sb] + a[:, sb]*b[:, sa]) / 2
            sym.append(s)
            order.append((a[:, aa]*b[:, bb] - a[:, bb]*b[:, aa]) / 2)
        out[start:stop] = np.concatenate([*main, *sync, *sym, *order], axis=1)
    if not np.isfinite(out).all():
        raise ValueError('Nonfinite lag basis')
    return out


def fit_static(values, selected):
    x = values[selected].astype('f8')
    med = np.nanmedian(np.where(np.isfinite(x), x, np.nan), axis=0)
    if not np.isfinite(med).all():
        raise ValueError('Entirely missing training static descriptor')
    fixed = np.where(np.isfinite(x), x, med)
    mu = fixed.mean(0); sd = fixed.std(0); sd[sd < 1e-7] = 1
    return med.astype('f4'), mu.astype('f4'), sd.astype('f4')


def static_basis(panel, params):
    med, mu, sd = params['geo']
    g = (np.where(np.isfinite(panel['geo']), panel['geo'], med)-mu)/sd
    # All coordinates retained. Equal semantic-block total energy before nonlinear mixing.
    g = g / params['geo_divisor']
    nonlinear = np.tanh(g @ params['geo_projection'])
    G = np.concatenate([g, np.maximum(g, 0), nonlinear], axis=1).astype('f4')
    med, mu, sd = params['context']
    c = (np.where(np.isfinite(panel['context']), panel['context'], med)-mu)/sd
    C = np.concatenate([c, np.maximum(c, 0)], axis=1).astype('f4')
    return G, C


def nuisance_raw(panel, rows, G, C, history=True):
    u, t = rows['unit'], rows['time']
    reg = rows['regime']
    phase = (t-72)/143
    cols = [np.ones(len(u)), phase, phase*phase,
            np.sin(2*np.pi*t/24), np.cos(2*np.pi*t/24)]
    # Fixed origin season and year are public, outcome-independent controls.
    origin = np.asarray(panel['meta']['origin'][u], dtype='datetime64[h]')
    day = origin.astype('datetime64[D]')
    year = day.astype('datetime64[Y]')
    doy = (day-year).astype(int)
    cols.extend([np.sin(2*np.pi*doy/365.25), np.cos(2*np.pi*doy/365.25),
                 (year.astype(int)+1970-2020)/10])
    for r in REGIMES:
        mask = (reg == r).astype('f4')
        cols.extend([mask, mask*phase, mask*phase*phase])
    if history:
        p = rows['p_prev']
        cols.extend([p, p*p, np.sqrt(np.maximum(p, 0)), panel['y_full'][u, 71]])
    return np.column_stack([*cols, G[u], C[u]]).astype('f4')


@dataclass
class FeatureMap:
    weather_mu: np.ndarray
    weather_sd: np.ndarray
    knots: np.ndarray
    static_params: dict
    x_mu: np.ndarray
    x_sd: np.ndarray
    x_active: np.ndarray
    n_mu: np.ndarray
    n_sd: np.ndarray
    n_active: np.ndarray
    g_mu: np.ndarray
    g_sd: np.ndarray
    g_active: np.ndarray
    c_mu: np.ndarray
    c_sd: np.ndarray
    c_active: np.ndarray
    history: bool

    def transform(self, panel, wx, rows):
        X = raw_weather_features(wx, rows, self.weather_mu, self.weather_sd, self.knots)
        X -= self.x_mu; X /= self.x_sd; X[:, ~self.x_active] = 0
        G0, C0 = static_basis(panel, self.static_params)
        N = nuisance_raw(panel, rows, G0, C0, self.history)
        N -= self.n_mu; N /= self.n_sd; N[:, ~self.n_active] = 0; N[:, 0] = 1
        G = (G0[rows['unit']]-self.g_mu)/self.g_sd; G[:, ~self.g_active] = 0
        C = (C0[rows['unit']]-self.c_mu)/self.c_sd; C[:, ~self.c_active] = 0
        return X, N, G.astype('f4'), C.astype('f4')


def fit_map(panel, wx, rows, history=True):
    """Fit on supplied training rows only; outcome used nowhere in weather/static mapping."""
    u, t = rows['unit'], rows['time']
    w = rows['w'].astype('f8'); w /= w.sum()
    hour_weight = np.zeros(wx.shape[:2], dtype='f8')
    for lag in range(48):
        np.add.at(hour_weight, (u, t-1-lag), w/48)
    flat_w = hour_weight.ravel()
    mu, sd, _ = weighted_scale(wx.reshape(-1, 12), flat_w)
    knots = np.column_stack([weighted_quantile(wx[:, :, j].ravel(), flat_w, [.5, .9])
                             for j in range(12)])
    knots = ((knots-mu)/sd).astype('f4')
    # Deduplicate counties using training units only, so repeated storms do not set geo scaling.
    train_units = np.unique(u)
    _, first = np.unique(panel['meta']['county'][train_units], return_index=True)
    selected = train_units[first]
    names = panel['feature_names']['geo']
    counts = {name: len(features) for name, features in BLOCKS.items() if name != 'county_service_context'}
    membership = {f: b for b, fs in BLOCKS.items() for f in fs}
    div = np.array([np.sqrt(4*counts[membership[name]]) for name in names], dtype='f4')
    params = {'geo': fit_static(panel['geo'], selected),
              'context': fit_static(panel['context'], selected), 'geo_divisor': div,
              'geo_projection': np.random.default_rng(20260929).normal(size=(40,64)).astype('f4')}
    G0, C0 = static_basis(panel, params)
    X = raw_weather_features(wx, rows, mu, sd, knots)
    xm, xs, xa = weighted_scale(X, w)
    X -= xm; X /= xs; X[:, ~xa] = 0
    N = nuisance_raw(panel, rows, G0, C0, history)
    nm, ns, na = weighted_scale(N, w)
    N -= nm; N /= ns; N[:, ~na] = 0; N[:, 0] = 1
    gm, gs, ga = weighted_scale(G0[u], w)
    cm, cs, ca = weighted_scale(C0[u], w)
    G=(G0[u]-gm)/gs; G[:, ~ga]=0
    C=(C0[u]-cm)/cs; C[:, ~ca]=0
    fm = FeatureMap(mu,sd,knots,params,xm,xs,xa,nm,ns,na,gm,gs,ga,cm,cs,ca,history)
    return fm, (X,N,G.astype('f4'),C.astype('f4'))
