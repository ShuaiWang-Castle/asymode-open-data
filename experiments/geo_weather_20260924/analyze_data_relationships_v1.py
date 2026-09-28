"""D01 finite public-D exploratory probes; never invokes a neural training entry point.

See notes/D01_DATA_FIRST_PROTOCOL_20260928.md. Large OOF arrays stay in ignored runs/.
Run with BLAS thread counts set to 2 and nice >=15 on the shared computer.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import itertools
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FEAT = ROOT / 'data/interim/panel_v1/features_v1D.npz'
RUN = ROOT / 'runs/geo_weather_20260924/data_first_d01'
REG = ['tropical', 'winter', 'synoptic_wind', 'convective', 'heavy_rain']
WEATHER = ['gust', 'precip', 't2m_c', 'soil_moisture', 'snowfall', 'cape']
GEO = ['elev_mean', 'relief_p95_p5', 'canopy_mean', 'developed_frac', 'poorly_drained_share', 'forest_wet_coloc']
PAIRS = list(itertools.combinations(range(6), 2))
ANCHORS = [72, 96, 120, 144, 168, 192]
LAMBDAS = [0.001, 0.01, 0.1, 1.0]
MODELS = ['A', 'B', 'C', 'D', 'E', 'F', 'F_null1', 'F_null2', 'F_null3']
SEED = 20260928


def moments(x):
    """Centered covariance and equally weighted lag-pair symmetric/antisymmetric moments."""
    v = x - x.mean(1, keepdims=True)
    simultaneous, symmetric, ordered = [], [], []
    count = sum(x.shape[1] - lag for lag in range(1, 13))
    for i, j in PAIRS:
        simultaneous.append((v[:, :, i] * v[:, :, j]).mean(1))
        ab = sum((v[:, :-lag, i] * v[:, lag:, j]).sum(1) for lag in range(1, 13)) / count
        ba = sum((v[:, :-lag, j] * v[:, lag:, i]).sum(1) for lag in range(1, 13)) / count
        symmetric.append((ab + ba) / 2)
        ordered.append((ab - ba) / 2)
    return np.stack(simultaneous, 1), np.stack(symmetric, 1), np.stack(ordered, 1)


def summarize(x, future=False):
    if future:
        return np.concatenate([x.mean(1), x.std(1), x.min(1), x.max(1)], 1)
    return np.concatenate([x.mean(1), x.std(1), x.min(1), x.max(1), x[:, -1],
                           x[:, -6:].mean(1), x[:, -12:-6].mean(1), x[:, :-12].mean(1),
                           (23 - x.argmax(1)) / 24], 1)


def merged_groups(system, family, origin, county):
    systems = np.unique(system)
    parent = {s: s for s in systems}
    def find(s):
        while parent[s] != s:
            parent[s] = parent[parent[s]]
            s = parent[s]
        return s
    def union(a, b):
        aa, bb = find(a), find(b)
        parent[max(aa, bb)] = min(aa, bb)
    info = {}
    for s in systems:
        ii = np.flatnonzero(system == s)
        info[s] = (family[ii[0]], np.datetime64(origin[ii[0]]), set(county[ii]))
    for a, b in itertools.combinations(systems, 2):
        fa, oa, ca = info[a]; fb, ob, cb = info[b]
        if fa == fb or (abs(oa - ob) <= np.timedelta64(16, 'D') and ca & cb):
            union(a, b)
    return np.array([find(s) for s in system])


def load_rows():
    with np.load(FEAT, allow_pickle=False) as f:
        names = f['damage_features'].tolist(); gn = f['geo_features'].tolist()
        wx = f['xu'][:, :, :12].astype(np.float64)
        for c in ['cape', 'precip', 'snowfall']:
            wx[:, :, names.index(c)] = np.log1p(np.maximum(wx[:, :, names.index(c)], 0))
        yy, obs = f['y_full'].astype(float), f['obs_full'].astype(bool)
        n = len(yy)
        meta = {k: f[k] for k in ['system', 'family', 'fips', 'origin', 'regime', 'w', 'w_raw', 'geo']}
        static = f['xr'][:, 0, 14:20].astype(float)
        clock = f['xu'][:, 71, 12:14].astype(float)
    sp = json.loads((HERE / 'splits_v1D.json').read_text())
    unitfold = np.zeros(n, int)
    for k in range(1, 6):
        ids = sp['event'][str(k)]['outer']
        assert not unitfold[ids].any()
        unitfold[ids] = k
    assert (unitfold > 0).all()
    groups = merged_groups(meta['system'], meta['family'], meta['origin'], meta['fips'])
    for g in np.unique(groups):
        assert len(np.unique(unitfold[groups == g])) == 1
    units, anchors, past, future, history, targets = [], [], [], [], [], []
    for a in ANCHORS:
        ph, fh = obs[:, a-24:a], obs[:, a:a+24]
        ok = obs[:, a-1] & (ph.sum(1) >= 22) & (fh.sum(1) >= 22)
        idx = np.flatnonzero(ok)
        py = np.where(ph[idx], yy[idx, a-24:a], np.nan)
        fy = np.where(fh[idx], yy[idx, a:a+24], np.nan)
        p0 = yy[idx, a-1]
        hist = np.stack([p0, np.nanmean(py, 1), np.nanmax(py, 1),
                         p0-np.nanmean(py[:, :6], 1), ph[idx].mean(1)], 1)
        assert np.isfinite(hist).all()
        units.append(idx); anchors.append(np.full(len(idx), a))
        past.append(wx[idx, a-24:a]); future.append(wx[idx, a:a+24]); history.append(hist)
        targets.append(np.stack([np.nanmean(fy, 1), (np.nanmax(fy, 1)-p0 >= .01).astype(float)], 1))
    u, anchor = np.concatenate(units), np.concatenate(anchors)
    count = np.bincount(u, minlength=n)
    date = pd.to_datetime(meta['origin'][u])
    states = np.array([c[:2] for c in meta['fips'][u]])
    categorical = np.column_stack([*(meta['regime'][u] == r for r in REG),
                                  *(states == s for s in sorted(set(states))),
                                  *(anchor == a for a in ANCHORS)])
    control = np.column_stack([np.concatenate(history), static[u], clock[u],
                               np.sin(2*np.pi*date.dayofyear.to_numpy()/365.25),
                               np.cos(2*np.pi*date.dayofyear.to_numpy()/365.25), date.year.to_numpy()])
    d = dict(past=np.concatenate(past), future=np.concatenate(future), unit=u, anchor=anchor,
             y=np.concatenate(targets), control=control, cat=categorical.astype(float),
             fold=unitfold[u], group=groups[u], geo=meta['geo'][u].astype(float),
             gidx=[gn.index(c) for c in GEO], widx=[names.index(c) for c in WEATHER],
             w=meta['w'][u]/count[u], w_raw=meta['w_raw'][u]/count[u],
             reg=meta['regime'][u], county=meta['fips'][u], system=meta['system'][u])
    d['audit'] = dict(units=n, retained_units=int((count>0).sum()), rows=len(u),
                      proposed_rows=n*len(ANCHORS), systems=len(np.unique(d['system'])),
                      families=len(np.unique(meta['family'])), merged_groups=len(np.unique(groups)),
                      counties=len(np.unique(d['county'])),
                      per_regime={r: dict(units=int((meta['regime']==r).sum()),
                                         retained_rows=int((d['reg']==r).sum())) for r in REG},
                      missing_geo_entries=int((~np.isfinite(meta['geo'])).sum()))
    d['audit']['fold_regime_support'] = {str(k): {r: dict(
        rows=int(((d['fold']==k)&(d['reg']==r)).sum()),
        groups=len(np.unique(d['group'][(d['fold']==k)&(d['reg']==r)])),
        positive_burden=int((d['y'][(d['fold']==k)&(d['reg']==r),0]>0).sum()),
        positive_rise=int(d['y'][(d['fold']==k)&(d['reg']==r),1].sum())) for r in REG} for k in range(1,6)}
    strata = []
    for g in np.unique(d['group']):
        strata.append(int(np.argmax([d['w'][(d['group']==g)&(d['reg']==r)].sum() for r in REG])))
    d['audit']['merged_group_bootstrap_strata'] = dict(zip(REG, np.bincount(strata, minlength=5).tolist()))
    return d


def fit_transform(x, train):
    med = np.nanmedian(np.where(np.isfinite(x[train]), x[train], np.nan), axis=0)
    med = np.nan_to_num(med)
    filled = np.where(np.isfinite(x), x, med)
    mean = filled[train].mean(0)
    sd = filled[train].std(0)
    sd = np.where(sd > 1e-8, sd, 1.)
    return np.clip((filled-mean)/sd, -8, 8)


def donor_geography(d, train, seed):
    counties, first = np.unique(d['county'], return_index=True)
    allowed = np.unique(d['county'][train])
    idx = {c: i for c, i in zip(counties, first)}
    state_pool = {s: allowed[np.array([c[:2] == s for c in allowed])].tolist()
                  for s in {c[:2] for c in counties}}
    rng = np.random.default_rng(seed)
    values, fallback = {}, 0
    for c in counties:
        pool = [x for x in state_pool[c[:2]] if x != c]
        if not pool:
            pool = [x for x in allowed if x != c]
            fallback += 1
        donor = pool[int(rng.integers(len(pool)))]
        values[c] = d['geo'][idx[donor], d['gidx']]
    return np.stack([values[c] for c in d['county']]), fallback


def feature_blocks(d, train, spec):
    wx = d['future'] if spec == 'aligned_exposure' else d['past']
    rawmain = [summarize(wx), d['geo'], d['control']]
    if spec == 'future_adjusted':
        rawmain.append(summarize(d['future'], future=True))
    main = fit_transform(np.concatenate(rawmain, 1), train)
    # Fixed piecewise-linear basis; no knots selected on held-out responses.
    main = np.concatenate([main, *(np.maximum(main-c, 0) for c in [-1., 0., 1.]), d['cat']], 1)
    geo = fit_transform(d['geo'][:, d['gidx']], train)
    x = wx[:, :, d['widx']]
    lag = np.concatenate([x[:, -6:].mean(1), x[:, -12:-6].mean(1), x[:, :-12].mean(1)], 1)
    lag = fit_transform(lag, train)
    wg = fit_transform((lag[:, :, None]*geo[:, None, :]).reshape(len(x), -1), train)
    c, s, o = moments(x)
    means = x.mean(1)
    magnitude = np.stack([means[:, i]*means[:, j] for i, j in PAIRS], 1)
    comp = fit_transform(np.concatenate([magnitude, c, s], 1), train)
    order = fit_transform(o, train)
    cs = np.concatenate([comp, order], 1)
    cross = fit_transform((cs[:, :, None]*geo[:, None, :]).reshape(len(x), -1), train)
    base = np.concatenate([main, wg, comp, order], 1).astype(np.float32)
    sizes = dict(A=main.shape[1], B=main.shape[1]+wg.shape[1],
                 C=main.shape[1]+wg.shape[1]+comp.shape[1], D=base.shape[1])
    # Keep shared main effects once: four copies of the full E design waste >1 GB.
    blocks = {k: (sizes[k], None) for k in ['A', 'B', 'C', 'D']}
    cross = cross.astype(np.float32)
    blocks['E'] = (base.shape[1], cross[:, :comp.shape[1]*len(GEO)])
    blocks['F'] = (base.shape[1], cross)
    fallback = []
    for j in range(1, 4):
        donor, fb = donor_geography(d, train, SEED+j)
        # Standardization also uses fit rows; fixed lower-order features remain true.
        dg = fit_transform(donor, train)
        null = fit_transform((cs[:, :, None]*dg[:, None, :]).reshape(len(x), -1), train)
        blocks[f'F_null{j}'] = (base.shape[1], null.astype(np.float32))
        true_raw = d['geo'][:, d['gidx']]
        scale = np.nanstd(true_raw[train], axis=0)
        scale = np.where(scale>1e-8, scale, 1.)
        fallback.append(dict(fallback_counties=fb, recipient_counties=len(np.unique(d['county'])),
            fit_mean_shift_in_true_sd=((np.nanmean(donor[train],0)-np.nanmean(true_raw[train],0))/scale).tolist(),
            fit_sd_ratio=(np.nanstd(donor[train],0)/scale).tolist()))
    return base, blocks, fallback


def balanced_weights(d, rows):
    w = d['w'][rows].astype(float).copy()
    for r in REG:
        m = d['reg'][rows] == r
        w[m] /= w[m].sum()
    return w / w.sum()


class RidgeCache:
    """Share the weighted Gram matrix of all lower-order features across nested probes."""
    def __init__(self, base, y, train, test, weights):
        self.train, self.test, self.w = train, test, weights
        self.ym = (weights[:, None]*y[train]).sum(0)
        xm = weights @ base[train]
        self.z = base[train]-xm
        self.v = base[test]-xm
        self.wy = weights[:, None]*(y[train]-self.ym)
        self.gram = self.z.T @ (weights[:, None]*self.z)
        self.rhs = self.z.T @ self.wy

    def predict(self, block, lambdas):
        p, extra = block
        gram, rhs, xv = self.gram[:p, :p], self.rhs[:p], self.v[:, :p]
        if extra is not None:
            em = self.w @ extra[self.train]
            z = extra[self.train]-em
            wex = self.w[:, None]*z
            cross = self.z[:, :p].T @ wex
            gram = np.block([[gram, cross], [cross.T, z.T @ wex]])
            rhs = np.concatenate([rhs, z.T @ self.wy])
            xv = np.concatenate([xv, extra[self.test]-em], 1)
        preds = []
        for lam in lambdas:
            mat = gram.copy()
            mat.flat[::len(mat)+1] += lam
            coef = cho_solve(cho_factor(mat, lower=True, check_finite=False), rhs, check_finite=False)
            preds.append(np.clip(xv @ coef+self.ym, 0, 1))
        return np.stack(preds)


def run_cv(d, spec):
    result = np.full((len(d['y']), len(MODELS), 2), np.nan)
    records = []
    for outer in range(1, 6):
        outpath = RUN / f'{spec}_fold{outer}.npz'
        train = np.flatnonzero(d['fold'] != outer); test = np.flatnonzero(d['fold'] == outer)
        scores = np.zeros((len(MODELS), len(LAMBDAS), 2))
        for inner in range(1, 6):
            if inner == outer:
                continue
            it = np.flatnonzero((d['fold'] != outer) & (d['fold'] != inner))
            iv = np.flatnonzero(d['fold'] == inner)
            base, blocks, _ = feature_blocks(d, it, spec)
            wt, wv = balanced_weights(d, it), balanced_weights(d, iv)
            ridge = RidgeCache(base, d['y'], it, iv, wt)
            for m, name in enumerate(MODELS):
                p = ridge.predict(blocks[name], LAMBDAS)
                scores[m] += ((p-d['y'][iv][None])**2 * wv[None, :, None]).sum(1)
            del ridge, blocks, base
            print(json.dumps(dict(spec=spec, outer=outer, inner=inner, complete=True)), flush=True)
        best = scores.argmin(1)
        base, blocks, fallback = feature_blocks(d, train, spec)
        wt = balanced_weights(d, train)
        ridge = RidgeCache(base, d['y'], train, test, wt)
        for m, name in enumerate(MODELS):
            p = ridge.predict(blocks[name], LAMBDAS)
            for response in range(2):
                result[test, m, response] = p[best[m, response], :, response]
        rec = dict(fold=outer, selected_lambda={name: [LAMBDAS[x] for x in best[m]]
                     for m, name in enumerate(MODELS)}, dimensions={k: v[0]+(0 if v[1] is None else v[1].shape[1]) for k, v in blocks.items()},
                   null_county_fallback=fallback, train_rows=len(train), test_rows=len(test))
        records.append(rec)
        np.savez_compressed(outpath, idx=test, pred=result[test], meta=json.dumps(rec))
        print(json.dumps(dict(spec=spec, fold=outer, complete=True, dimensions=rec['dimensions'])), flush=True)
        del ridge, blocks, base
    assert np.isfinite(result).all()
    return result, records


def ratio_metrics(error, d, weights):
    # Shapes rows x model x outcome. Ratios are calculated within each regime first.
    v = np.stack([(weights[d['reg']==r, None, None]*error[d['reg']==r]).sum(0)
                 /weights[d['reg']==r].sum() for r in REG])
    return v


def relative(new, old):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(old>0, new/old-1, np.nan)


def clustered_draws(error, d, group, weights, b=2000):
    groups, gi = np.unique(group, return_inverse=True)
    sums = np.zeros((len(groups), len(REG), len(MODELS), 2))
    totals = np.zeros((len(groups), len(REG)))
    dominant = []
    for r, regime in enumerate(REG):
        m = d['reg'] == regime
        np.add.at(sums[:, r], gi[m], weights[m, None, None]*error[m])
        np.add.at(totals[:, r], gi[m], weights[m])
    for g in range(len(groups)):
        dominant.append(int(totals[g].argmax()))
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(b):
        idx = []
        for r in range(len(REG)):
            pool = np.flatnonzero(np.array(dominant) == r)
            if len(pool):
                idx.extend(rng.choice(pool, len(pool), replace=True))
        den = totals[idx].sum(0)
        if not (den>0).all():
            raise RuntimeError('Empty bootstrap regime; preserve outputs and inspect cluster support')
        draws.append(sums[idx].sum(0)/den[:, None, None])
    return np.stack(draws)


def summarize_results(d, p, records):
    err = (p-d['y'][:, None, :])**2
    point = ratio_metrics(err, d, d['w'])
    raw = ratio_metrics(err, d, d['w_raw'])
    event = clustered_draws(err, d, d['group'], d['w'])
    county = clustered_draws(err, d, d['county'], d['w'], b=1000)
    out = dict(fit_records=records, mse={name: {r: point[k, m].tolist() for k, r in enumerate(REG)}
                                       for m, name in enumerate(MODELS)}, contrasts={})
    contrasts = [('B','A'), ('C','B'), ('D','C'), ('E','D'), ('F','E'), ('F','D')]+[('F',f'F_null{i}') for i in range(1,4)]
    for new, old in contrasts:
        ni, oi = MODELS.index(new), MODELS.index(old)
        contrast = {}
        for t, target in enumerate(['burden', 'rise_risk']):
            gain = relative(point[:, ni, t],point[:, oi, t])
            boot = relative(event[:, :, ni, t],event[:, :, oi, t])
            cb = relative(county[:, :, ni, t],county[:, :, oi, t])
            foldgain = []
            for k in range(1, 6):
                mask = d['fold'] == k
                # Keep all rows but zero weights outside this held-out fold.
                fp = ratio_metrics(err, d, d['w']*mask)
                foldgain.append(relative(fp[:, ni, t],fp[:, oi, t]).tolist())
            # System contribution to the balanced paired delta uses fixed full-sample old denominators.
            contributions = []
            for s in np.unique(d['system']):
                sm = d['system'] == s
                delta = []
                for r in REG:
                    rm = d['reg'] == r
                    den = (d['w'][rm]*err[rm, oi, t]).sum()
                    delta.append(float((d['w'][sm & rm]*(err[sm & rm, ni, t]-err[sm & rm, oi, t])).sum()/den) if den>0 else np.nan)
                contributions.append((s, float(np.mean(delta[:2])), float(np.mean(delta))))
            biggest = min(contributions, key=lambda x:x[1])
            no = ratio_metrics(err, d, d['w']*(d['system'] != biggest[0]))
            gno = relative(no[:, ni, t],no[:, oi, t])
            contrast[target] = dict(per_regime=dict(zip(REG, gain.tolist())),
                headline=float(gain[:2].mean()), all5=float(gain.mean()),
                headline_ci95=np.quantile(boot[:, :2].mean(1), [.025,.975]).tolist(),
                all5_ci95=np.quantile(boot.mean(1), [.025,.975]).tolist(),
                per_regime_ci95={r:np.quantile(boot[:, k],[.025,.975]).tolist() for k,r in enumerate(REG)},
                county_headline_ci95=np.quantile(cb[:, :2].mean(1), [.025,.975]).tolist(),
                raw_headline=float(relative(raw[:2, ni, t],raw[:2, oi, t]).mean()),
                valid_event_draws=int(np.isfinite(boot).all(1).sum()),
                fold_headline=[float(np.mean(x[:2])) for x in foldgain],
                biggest_benefit_system=dict(system=biggest[0],headline_contribution=biggest[1],all5_contribution=biggest[2]),
                headline_without_biggest=float(gno[:2].mean()), all5_without_biggest=float(gno.mean()))
        out['contrasts'][f'{new}_vs_{old}'] = contrast
    return out


def finite_json(value):
    """Undefined relative errors are explicit nulls, never fictitious zeros."""
    if isinstance(value, dict):
        return {k: finite_json(v) for k,v in value.items()}
    if isinstance(value, list):
        return [finite_json(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', choices=['past_only','future_adjusted','aligned_exposure','all'], default='all')
    a = ap.parse_args()
    if os.getpriority(os.PRIO_PROCESS, 0) < 15:
        os.nice(15-os.getpriority(os.PRIO_PROCESS, 0))
    RUN.mkdir(parents=True, exist_ok=True)
    lock = (RUN/'ANALYSIS.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    (RUN/'RUN_STATUS.json').write_text(json.dumps(dict(pid=os.getpid(), state='running', specification=a.spec,
                                                       started=time.time()))+'\n')
    start = time.time()
    d = load_rows()
    print(json.dumps(d['audit']), flush=True)
    specs = ['past_only','future_adjusted','aligned_exposure'] if a.spec == 'all' else [a.spec]
    for spec in specs:
        final = HERE / f'results/v1/data_first_d01_{spec}.json'
        if final.exists() or list(RUN.glob(f'{spec}_fold*.npz')):
            raise RuntimeError(f'Existing {spec} outputs: preserve and inspect before any rerun')
        p, records = run_cv(d, spec)
        res = summarize_results(d, p, records)
        res['meta'] = dict(specification=spec, audit=d['audit'], models=MODELS, responses=['burden','rise_risk'],
                           exploratory=True, primary='F_vs_D/headline/burden', elapsed_seconds=time.time()-start,
                           source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           protocol_sha256=hashlib.sha256((HERE/'notes/D01_DATA_FIRST_PROTOCOL_20260928.md').read_bytes()).hexdigest(),
                           data_sha256=hashlib.sha256(FEAT.read_bytes()).hexdigest(),
                           bootstrap=dict(event_draws=2000,county_draws=1000,seed=SEED))
        final.write_text(json.dumps(finite_json(res), indent=2, allow_nan=False)+'\n')
        np.savez_compressed(RUN/f'{spec}_oof.npz', pred=p, target=d['y'], unit=d['unit'],anchor=d['anchor'])
        print(json.dumps(dict(spec=spec, result=str(final.relative_to(ROOT)), seconds=time.time()-start)), flush=True)
    (RUN/'RUN_STATUS.json').write_text(json.dumps(dict(pid=os.getpid(), state='complete', specifications=specs,
                                                       seconds=time.time()-start))+'\n')


if __name__ == '__main__':
    main()
