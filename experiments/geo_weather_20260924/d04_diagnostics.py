"""D04 local model sensitivities and a descriptive exposure-distance proxy.

Nothing here fits outcomes or loads a panel. Derivatives hold nuisance, history,
geography and county context fixed. The distance proxy is not true common support,
a probability, or a causal overlap test. Run --self-test for synthetic checks only.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '2'

from collections.abc import Mapping
import numpy as np

from d04_features import (CROSS, WITH_SELF, LAG_PAIRS, LAGS, MAIN_DIM, PAIR_DIM,
                          static_basis)

MAX_LANDMARKS = 512
MIN_REFERENCE_DISTANCES = 20
MIN_REFERENCE_GROUPS = 3
DISTANCE_BLOCKS = ((0, 144), (144, 342), (342, 576), (576, 774))


def _row_static(fmap, panel, unit):
    G0, C0 = static_basis(panel, fmap.static_params)
    G = (G0[unit] - fmap.g_mu) / fmap.g_sd
    C = (C0[unit] - fmap.c_mu) / fmap.c_sd
    G[:, ~fmap.g_active] = 0
    C[:, ~fmap.c_active] = 0
    return G.astype(np.float64), C.astype(np.float64)


def lag_sensitivity(model, fmap, panel, wx, rows, y_scale, geo_only=False):
    """Return [N,3,12] prediction-fraction changes per one training weather SD.

    Every hour of one channel in one lag band is shifted by the same epsilon in
    weather-z units. Lag bands are past 1..6, 7..24 and 25..48 hours; log1p
    channels use their transformed training SD. This is an analytic local model
    sensitivity with N/history/G/C fixed, not a physically feasible intervention.
    A hinge's symmetric slope at exact equality to its knot is defined as 0.5;
    away from the knot this is the usual zero/one derivative. The convention
    agrees with centered finite differences at the nondifferentiable knot.

    Models may use either the 144 main columns or all 774 weather columns.
    geo_only keeps only the geographic interaction's effective coefficient.
    Returned values are float64; no clipping or derivative-based selection occurs.
    """
    p = len(model.beta_x)
    if p not in (MAIN_DIM, MAIN_DIM + PAIR_DIM):
        raise ValueError('Expected a main-144 or all-774 weather model')
    if not np.isfinite(y_scale) or y_scale <= 0:
        raise ValueError('y_scale must be finite and positive')
    unit, time = np.asarray(rows['unit']), np.asarray(rows['time'])
    n = len(unit)
    if len(time) != n or np.any(time < 48) or np.any(time >= wx.shape[1]):
        raise ValueError('Rows do not have a full strictly-past 48-hour history')
    if len(fmap.x_sd) < p or np.any(np.asarray(fmap.x_sd[:p]) <= 0):
        raise ValueError('Invalid fitted feature scaling')
    answer = np.zeros((n, 3, 12), dtype=np.float64)
    if geo_only and model.u_geo.shape[1] == 0:
        return answer
    G, C = _row_static(fmap, panel, unit)
    ug, vg = np.asarray(model.u_geo, dtype=float), np.asarray(model.v_geo, dtype=float)
    uc, vc = np.asarray(model.u_context, dtype=float), np.asarray(model.v_context, dtype=float)
    if ug.shape[0] != p or uc.shape[0] != p:
        raise ValueError('Weather factor dimension differs from beta_x')
    aa, bb = np.array(CROSS).T
    sa, sb = np.array(WITH_SELF).T
    for start in range(0, n, 2048):
        stop = min(n, start + 2048)
        count = stop - start
        idx = time[start:stop, None] - 1 - np.arange(48)[None]
        # Match the feature map's normalization precision before evaluating its
        # hinge boundaries; then accumulate derivatives in float64.
        z = ((wx[unit[start:stop, None], idx] - fmap.weather_mu)
             / fmap.weather_sd).astype(np.float64)
        effective = (G[start:stop] @ vg) @ ug.T
        if not geo_only:
            effective += np.asarray(model.beta_x, dtype=float)[None]
            effective += (C[start:stop] @ vc) @ uc.T
        effective /= np.asarray(fmap.x_sd[:p], dtype=float)[None]
        effective[:, ~np.asarray(fmap.x_active[:p], dtype=bool)] = 0
        coeff_main = effective[:, :144].reshape(count, 3, 12, 4)
        means = np.empty((count, 3, 12), dtype=float)
        derivative = answer[start:stop]
        for band, (lo, hi) in enumerate(LAGS):
            zz = z[:, lo:hi]
            means[:, band] = zz.mean(1)
            mid0 = ((zz > fmap.knots[0]) + .5 * (zz == fmap.knots[0])).mean(1)
            mid1 = ((zz > fmap.knots[1]) + .5 * (zz == fmap.knots[1])).mean(1)
            derivative[:, band] += (coeff_main[:, band, :, 0]
                + 2 * means[:, band] * coeff_main[:, band, :, 1]
                + mid0 * coeff_main[:, band, :, 2]
                + mid1 * coeff_main[:, band, :, 3])
        if p == 144:
            continue
        synchronous = effective[:, 144:342].reshape(count, 3, 66)
        for band in range(3):
            matrix = np.zeros((count, 12, 12), dtype=float)
            matrix[:, aa, bb] = synchronous[:, band]
            matrix[:, bb, aa] = synchronous[:, band]
            derivative[:, band] += np.einsum('nij,nj->ni', matrix, means[:, band])
        symmetric = effective[:, 342:576].reshape(count, 3, 78)
        ordered = effective[:, 576:774].reshape(count, 3, 66)
        for pair, (near, far) in enumerate(LAG_PAIRS):
            near_mean, far_mean = means[:, near], means[:, far]
            for j, (a, b) in enumerate(zip(sa, sb)):
                coef = .5 * symmetric[:, pair, j]
                derivative[:, far, a] += coef * near_mean[:, b]
                derivative[:, far, b] += coef * near_mean[:, a]
                derivative[:, near, a] += coef * far_mean[:, b]
                derivative[:, near, b] += coef * far_mean[:, a]
            for j, (a, b) in enumerate(zip(aa, bb)):
                coef = .5 * ordered[:, pair, j]
                derivative[:, far, a] += coef * near_mean[:, b]
                derivative[:, far, b] -= coef * near_mean[:, a]
                derivative[:, near, b] += coef * far_mean[:, a]
                derivative[:, near, a] -= coef * far_mean[:, b]
    answer *= float(y_scale)
    if not np.isfinite(answer).all():
        raise FloatingPointError('Nonfinite local sensitivity')
    return answer


def _array_parts(arrays, rows):
    if isinstance(arrays, Mapping):
        X, C = arrays['X'], arrays['C']
    else:
        if len(arrays) != 4:
            raise ValueError('Arrays must be (X,N,G,C) or a mapping with X,C')
        X, _, _, C = arrays
    X, C = np.asarray(X), np.asarray(C)
    n = len(rows['unit'])
    if X.shape != (n, 774) or C.shape != (n, 12):
        raise ValueError('Support proxy requires all 774 X columns and 12 C columns')
    for key in ('regime', 'group', 'county', 'p_prev'):
        if len(rows[key]) != n:
            raise ValueError(f'Mismatched row field {key}')
    if 'time' in rows and np.any((np.asarray(rows['time']) - 72) % 6):
        raise ValueError('Support proxy is restricted to phase-zero diagnostics')
    return X, C


def _landmark_indices(indices, group, rng):
    """Equal-group round-robin random rows, without duplicate landmarks."""
    levels = np.unique(group[indices])
    pools = [rng.permutation(indices[group[indices] == g]) for g in levels]
    at = np.zeros(len(pools), dtype=int)
    chosen = []
    limit = min(MAX_LANDMARKS, len(indices))
    while len(chosen) < limit:
        active = np.flatnonzero(at < np.array([len(p) for p in pools]))
        if not len(active):
            break
        for g in rng.permutation(active):
            chosen.append(int(pools[g][at[g]]))
            at[g] += 1
            if len(chosen) == limit:
                break
    return np.asarray(chosen, dtype=np.int64)


def _distance_vectors(X, C, rows, indices, state_mean, state_sd):
    # Four weather blocks, context, and prior stock have equal aggregate scale.
    vectors = [np.asarray(X[indices, lo:hi], dtype=np.float32) / np.sqrt(hi - lo)
               for lo, hi in DISTANCE_BLOCKS]
    vectors.append(np.asarray(C[indices], dtype=np.float32) / np.sqrt(12))
    vectors.append(((np.asarray(rows['p_prev'])[indices] - state_mean) / state_sd)[:, None].astype('f4'))
    out = np.concatenate(vectors, axis=1).astype('f4')
    out *= np.float32(1 / np.sqrt(6))
    if not np.isfinite(out).all():
        raise ValueError('Nonfinite distance features')
    return out


def _distances(a, b):
    a2 = np.einsum('ij,ij->i', a, a, dtype=np.float32)
    b2 = np.einsum('ij,ij->i', b, b, dtype=np.float32)
    squared = a2[:, None] + b2[None] - 2 * (a @ b.T)
    np.maximum(squared, 0, out=squared)
    return np.sqrt(squared, out=squared)


def support_proxy(fmap, train_arrays, train_rows, query_arrays, query_rows, seed=20260929):
    """Six-block nearest-landmark distance, with explicit limited-reference flags.

    Arrays are the full fitted-map (X,N,G,C), including X columns unused by a
    main-only arm. Landmarks sample up to 512 rows per regime with equal merged
    group allocation where possible. Reference nearest distances exclude the
    same group and county. Outer queries exclude the same county; query groups
    must be disjoint from all training groups. A regime's reference q95 requires
    >=3 represented landmark groups and >=20 eligible reference distances.

    Returns distance/threshold (NaN when undefined), within_proxy (False when
    undefined), support_available, regime, and support_counts. Always use the
    availability mask to distinguish unknown from outside. p_prev uses the
    full training rows' design-weighted SD, never a query-fitted scale.
    No target-hour change or geography coordinates enter this proxy; observed
    previous stock is explicitly included as a historical control. It is not a support
    probability or proof that joint causal/physical exposure support is shared.
    """
    tx, tc = _array_parts(train_arrays, train_rows)
    qx, qc = _array_parts(query_arrays, query_rows)
    if len(fmap.x_mu) != 774 or len(fmap.c_mu) != 12:
        raise ValueError('Unexpected fitted-map feature dimensions')
    train_group = np.asarray(train_rows['group']).astype(str)
    query_group = np.asarray(query_rows['group']).astype(str)
    if np.intersect1d(np.unique(train_group), np.unique(query_group)).size:
        raise ValueError('Proxy queries must be held-out merged-event groups')
    train_county = np.asarray(train_rows['county']).astype(str)
    query_county = np.asarray(query_rows['county']).astype(str)
    train_reg = np.asarray(train_rows['regime']).astype(str)
    query_reg = np.asarray(query_rows['regime']).astype(str)
    n = len(query_reg)
    result = {'distance': np.full(n, np.nan), 'threshold': np.full(n, np.nan),
              'within_proxy': np.zeros(n, dtype=bool), 'support_available': np.zeros(n, dtype=bool),
              'regime': query_reg.copy(), 'support_counts': {},
              'definition': {'blocks': ['main144', 'sync198', 'symmetric234', 'ordered198', 'context12', 'p_prev1'],
                  'max_landmarks_per_regime': MAX_LANDMARKS, 'seed': int(seed),
                  'reference_quantile': .95, 'min_reference_distances': MIN_REFERENCE_DISTANCES,
                  'min_reference_groups': MIN_REFERENCE_GROUPS,
                  'unknown_within_proxy_is_false': True, 'probability_or_causal_overlap': False}}
    weights = np.asarray(train_rows['w'], dtype=float)
    stock = np.asarray(train_rows['p_prev'], dtype=float)
    if len(weights) != len(tx) or np.any(weights < 0) or not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError('Invalid training weights')
    if not np.isfinite(stock).all():
        raise ValueError('Nonfinite training prior stock')
    weights = weights / weights.sum()
    mean = float(weights @ stock)
    sd = float(np.sqrt(weights @ ((stock - mean) ** 2)))
    result['definition'].update(training_state_mean=mean, training_state_sd=sd)
    rng = np.random.default_rng(seed)
    for regime in np.union1d(np.unique(train_reg), np.unique(query_reg)):
        train = np.flatnonzero(train_reg == regime)
        query = np.flatnonzero(query_reg == regime)
        counts = {'train_rows': len(train), 'query_rows': len(query),
                  'training_groups': len(np.unique(train_group[train])),
                  'training_counties': len(np.unique(train_county[train])),
                  'landmarks': 0, 'landmark_groups': 0, 'landmark_counties': 0,
                  'eligible_reference_distances': 0, 'queries_with_eligible_county_neighbor': 0,
                  'threshold': None, 'reason': None}
        result['support_counts'][str(regime)] = counts
        if not len(train):
            counts['reason'] = 'no_training_rows_in_regime'
            continue
        if not np.isfinite(sd) or sd <= 1e-12:
            counts['reason'] = 'training_prior_stock_has_no_positive_standard_deviation'
            continue
        lm = _landmark_indices(train, train_group, rng)
        lg, lc = train_group[lm], train_county[lm]
        unique_group, group_count = np.unique(lg, return_counts=True)
        counts.update(landmarks=len(lm), landmark_groups=len(unique_group),
                      landmark_counties=len(np.unique(lc)),
                      landmarks_per_group={str(g): int(c) for g, c in zip(unique_group, group_count)})
        vec = _distance_vectors(tx, tc, train_rows, lm, mean, sd)
        reference = _distances(vec, vec)
        reference[(lg[:, None] == lg[None]) | (lc[:, None] == lc[None])] = np.inf
        nearest = reference.min(1)
        eligible = np.isfinite(nearest)
        counts['eligible_reference_distances'] = int(eligible.sum())
        threshold = np.nan
        if len(unique_group) >= MIN_REFERENCE_GROUPS and eligible.sum() >= MIN_REFERENCE_DISTANCES:
            threshold = float(np.quantile(nearest[eligible], .95))
            counts['threshold'] = threshold
        else:
            counts['reason'] = 'insufficient_cross_group_and_county_reference'
        for start in range(0, len(query), 1024):
            ii = query[start:start + 1024]
            qv = _distance_vectors(qx, qc, query_rows, ii, mean, sd)
            distance = _distances(qv, vec)
            distance[query_county[ii, None] == lc[None]] = np.inf
            nn = distance.min(1)
            valid = np.isfinite(nn)
            result['distance'][ii[valid]] = nn[valid]
            result['threshold'][ii] = threshold
            counts['queries_with_eligible_county_neighbor'] += int(valid.sum())
        if counts['queries_with_eligible_county_neighbor'] < len(query):
            counts['queries_without_eligible_county_neighbor'] = len(query) - counts['queries_with_eligible_county_neighbor']
    result['support_available'] = np.isfinite(result['distance']) & np.isfinite(result['threshold'])
    ok = result['support_available']
    result['within_proxy'][ok] = result['distance'][ok] <= result['threshold'][ok]
    return result


def self_test():
    """Synthetic finite differences and distance exclusions; no real panel or fit."""
    from types import SimpleNamespace
    from d04_features import raw_weather_features
    rng = np.random.default_rng(4407)
    n = 7
    wx = rng.normal(size=(n, 216, 12)).astype('f4')
    geo = rng.normal(size=(n, 40)).astype('f4')
    context = rng.normal(size=(n, 6)).astype('f4')
    panel = {'geo': geo, 'context': context}
    rows = {'unit': np.arange(n), 'time': np.full(n, 120)}
    params = {'geo': (np.zeros(40), np.zeros(40), np.ones(40)),
              'context': (np.zeros(6), np.zeros(6), np.ones(6)),
              'geo_divisor': np.ones(40), 'geo_projection': rng.normal(size=(40,64))*.1}
    fmap = SimpleNamespace(static_params=params, g_mu=np.zeros(144), g_sd=np.ones(144),
        c_mu=np.zeros(12), c_sd=np.ones(12), g_active=np.ones(144,dtype=bool), c_active=np.ones(12,dtype=bool),
        weather_mu=np.zeros(12), weather_sd=np.ones(12), knots=np.array([np.full(12,-.314159), np.full(12,.8271818)]),
        x_mu=rng.normal(size=774)*.1, x_sd=rng.uniform(.5,2,size=774), x_active=np.ones(774,dtype=bool))
    fmap.x_active[[5, 100, 401]] = False
    G,C = _row_static(fmap,panel,rows['unit'])
    y_scale=.017
    results={}
    for p,name in [(144,'main'),(774,'all')]:
        model=SimpleNamespace(beta_x=rng.normal(size=p)*.04, u_geo=rng.normal(size=(p,2))*.02,
            v_geo=rng.normal(size=(144,2))*.1,u_context=rng.normal(size=(p,2))*.02,
            v_context=rng.normal(size=(12,2))*.1)
        for geo_only in [False,True]:
            actual=lag_sensitivity(model,fmap,panel,wx,rows,y_scale,geo_only)
            coef=(G@model.v_geo)@model.u_geo.T
            if not geo_only:coef=coef+model.beta_x+(C@model.v_context)@model.u_context.T
            expected=np.zeros_like(actual)
            epsilon=1e-3
            for b,(lo,hi) in enumerate(LAGS):
                hours=120-1-np.arange(lo,hi)
                for channel in range(12):
                    plus,minus=wx.copy(),wx.copy()
                    plus[:,hours,channel]+=epsilon
                    minus[:,hours,channel]-=epsilon
                    xp=raw_weather_features(plus,rows,fmap.weather_mu,fmap.weather_sd,fmap.knots)[:,:p]
                    xm=raw_weather_features(minus,rows,fmap.weather_mu,fmap.weather_sd,fmap.knots)[:,:p]
                    diff=(xp.astype(float)-xm)/fmap.x_sd[:p]
                    diff[:,~fmap.x_active[:p]]=0
                    expected[:,b,channel]=np.sum(diff*coef,1)*y_scale/(2*epsilon)
            error=float(np.max(abs(actual-expected)))
            # Finite float32 bases and occasional epsilon-crossed knots bound precision.
            np.testing.assert_allclose(actual,expected,rtol=.02,atol=3e-5)
            results[name+('_geo_only' if geo_only else '_total')]=error
    # One exact hinge knot has the documented midpoint slope under central differences.
    simple=SimpleNamespace(beta_x=np.zeros(144),u_geo=np.zeros((144,0)),v_geo=np.zeros((144,0)),
        u_context=np.zeros((144,0)),v_context=np.zeros((12,0)))
    simple.beta_x[2]=1
    xzero=np.zeros_like(wx);fmap.knots[0]=0;fmap.x_sd[:]=1;fmap.x_active[:]=True
    at_knot=lag_sensitivity(simple,fmap,panel,xzero,rows,1.)
    np.testing.assert_allclose(at_knot[:,0,0],.5)
    assert np.all(lag_sensitivity(simple,fmap,panel,xzero,rows,1.,True)==0)
    results['hinge_exact_knot_midpoint']=float(at_knot[0,0,0])
    # Each group contributes equally when it has enough rows; nearest exclusions
    # are checked independently against a direct Euclidean implementation.
    nt,nq=90,8
    trainX=rng.normal(size=(nt,774)).astype('f4');trainC=rng.normal(size=(nt,12)).astype('f4')
    queryX=rng.normal(size=(nq,774)).astype('f4');queryC=rng.normal(size=(nq,12)).astype('f4')
    tr={'unit':np.arange(nt),'time':np.full(nt,72),'w':rng.uniform(.5,2,size=nt),
        'group':np.repeat(['a','b','c'],30),'county':np.array([str(i%30)for i in range(nt)]),
        'regime':np.repeat('winter',nt),'p_prev':rng.uniform(0,.1,size=nt)}
    qr={'unit':np.arange(nq),'time':np.full(nq,72),'group':np.repeat('outer',nq),
        'county':np.array([str(i)for i in range(nq)]),'regime':np.repeat('winter',nq),
        'p_prev':rng.uniform(0,.1,size=nq)}
    dummy=lambda x,c:(x,np.empty((len(x),0)),np.empty((len(x),0)),c)
    out=support_proxy(fmap,dummy(trainX,trainC),tr,dummy(queryX,queryC),qr)
    weight=tr['w']/tr['w'].sum();mean=weight@tr['p_prev'];sd=np.sqrt(weight@((tr['p_prev']-mean)**2))
    tv=_distance_vectors(trainX,trainC,tr,np.arange(nt),mean,sd)
    qv=_distance_vectors(queryX,queryC,qr,np.arange(nq),mean,sd)
    distance=np.sqrt(np.square(qv.astype(float)[:,None]-tv.astype(float)[None]).sum(2))
    distance[qr['county'][:,None]==tr['county'][None]]=np.inf
    np.testing.assert_allclose(out['distance'],distance.min(1),atol=2e-6,rtol=2e-6)
    reference=np.sqrt(np.square(tv.astype(float)[:,None]-tv.astype(float)[None]).sum(2))
    reference[(tr['county'][:,None]==tr['county'][None])|(tr['group'][:,None]==tr['group'][None])]=np.inf
    threshold=float(np.quantile(reference.min(1),.95))
    np.testing.assert_allclose(out['threshold'],threshold,atol=2e-6,rtol=2e-6)
    assert out['support_available'].all()
    few={k:v[:10]for k,v in tr.items()}
    small=support_proxy(fmap,dummy(trainX[:10],trainC[:10]),few,dummy(queryX,queryC),qr)
    assert np.isnan(small['threshold']).all()and not small['support_available'].any()
    qr_missing={**qr,'regime':np.repeat('missing_regime',nq)}
    absent=support_proxy(fmap,dummy(trainX,trainC),tr,dummy(queryX,queryC),qr_missing)
    assert np.isnan(absent['distance']).all()and not absent['support_available'].any()
    idx=_landmark_indices(np.arange(900),np.repeat(['a','b','c'],300),np.random.default_rng(8))
    _,cc=np.unique(np.repeat(['a','b','c'],300)[idx],return_counts=True)
    assert len(idx)==512 and len(np.unique(idx))==512 and cc.max()-cc.min()<=1
    results['support_distance_max_error']=float(np.max(abs(out['distance']-distance.min(1))))
    results['support_threshold_error']=float(abs(out['threshold'][0]-threshold))
    results['support_landmark_cap_and_equal_groups']='passed'
    results['support_sparse_and_missing_regime']='passed'
    return results


if __name__=='__main__':
    import argparse,json
    from threadpoolctl import threadpool_limits
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        os.nice(max(0,15-os.getpriority(os.PRIO_PROCESS,0)))
        with threadpool_limits(limits=2):
            print(json.dumps(self_test(),indent=2,allow_nan=False))
    else:parser.print_help()
