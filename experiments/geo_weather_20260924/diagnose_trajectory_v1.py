"""Frozen D-only trajectory diagnostics; no fitting, model selection or C access.

Phase SSE is an exact accounting partition, NOT causal attribution. Oracle scale
and shift repairs read held-out outcomes and are descriptive candidate repairs,
not eligible forecasts or globally optimal corrections. Recovery half-times are
censored at the window boundary.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '1')
import numpy as np
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REGIMES = ['tropical', 'winter', 'synoptic_wind', 'convective', 'heavy_rain']
LABELS = {'AsymODE': 'v1_host_s0', 'GCRK': 'v1_gcrk_s0', 'ST-GCRK': 'v1_stgcrk_s0'}


def stock(u, r, y0):
    p = y0.copy(); out = np.empty_like(u)
    for t in range(u.shape[1]):
        p = np.clip(p + u[:, t] * (1-p) - r[:, t] * p, 0, 1)
        out[:, t] = p
    return out


def quantile(x):
    x = np.asarray(x); x = x[np.isfinite(x)]
    return {'n': int(x.size), 'q10': float(np.quantile(x, .1)), 'median': float(np.median(x)),
            'q90': float(np.quantile(x, .9))} if len(x) else {'n': 0}


def first_cross(x, mask, threshold=.01):
    hit = (x >= threshold) & mask
    return np.where(hit.any(1), hit.argmax(1), -1)


def half_time(x, mask, min_peak=.01, sustain=6):
    """Elapsed time from own peak to first sustained <= half-peak crossing.

    NaN means no certified crossing (right censoring or missing follow-up), not
    infinite restoration time. Half-time measures stock shape, not repair rate.
    """
    peak_idx = np.argmax(np.where(mask, x, -np.inf), axis=1)
    peak = x[np.arange(len(x)), peak_idx]
    ans = np.full(len(x), np.nan)
    for i, t0 in enumerate(peak_idx):
        if peak[i] <= 0 or peak[i] < min_peak:
            continue
        for t in range(t0 + 1, x.shape[1] - sustain + 1):
            if mask[i, t:t+sustain].all() and (x[i, t:t+sustain] <= .5*peak[i]).all():
                ans[i] = t - t0
                break
    return ans


def shifted(p, lag):
    q = np.zeros_like(p)
    if lag > 0:
        q[:, lag:] = p[:, :-lag]
    elif lag < 0:
        q[:, :lag] = p[:, -lag:]
    else:
        q[:] = p
    return q


def oracle_errors(p, y, mask):
    """Per-unit best lag in +/-24 h and least-square positive amplitude repair.

    Outside the shifted window is zero. The unbounded L2 projection coefficient
    is followed by clipping to [0,1]; clipping cannot raise error for y in [0,1].
    Both lag searches include 0. The joint repair includes every scale-only path.
    """
    base = np.sum(mask*(p-y)**2, axis=1)
    timing = base.copy(); joint = base.copy(); best_lag = np.zeros(len(p), int)
    amplitude = None; scale_zero = None
    for lag in range(-24, 25):
        q = shifted(p, lag)
        se = np.sum(mask*(q-y)**2, axis=1)
        timing = np.minimum(timing, se)
        alpha = np.divide(np.sum(mask*q*y, 1), np.sum(mask*q*q, 1),
                          out=np.zeros(len(p)), where=np.sum(mask*q*q, 1)>0)
        scaled = np.clip(alpha[:, None]*q, 0, 1)
        ss = np.sum(mask*(scaled-y)**2, axis=1)
        improve = ss < joint
        joint[improve] = ss[improve]; best_lag[improve] = lag
        if lag == 0:
            amplitude = np.minimum(base, ss); scale_zero = alpha
    assert np.all(joint <= amplitude + 1e-12)
    return {'base': base, 'timing': timing, 'amplitude': amplitude, 'joint': joint}, scale_zero, best_lag


def bootstrap_ratio(a, b, w, reg, family):
    """Family-within-regime bootstrap; ratio of weighted sums, 2,000 draws."""
    rng = np.random.default_rng(20260924)
    da = np.zeros(2000); db = np.zeros(2000)
    for rr in REGIMES:
        families = np.unique(family[reg == rr])
        sa = np.array([np.sum(w[(reg == rr)&(family == f)]*a[(reg == rr)&(family == f)]) for f in families])
        sb = np.array([np.sum(w[(reg == rr)&(family == f)]*b[(reg == rr)&(family == f)]) for f in families])
        c = rng.multinomial(len(families), np.full(len(families), 1/len(families)), size=2000)
        da += c@sa; db += c@sb
    return {'point': float(np.sum(w*a)/np.sum(w*b)), 'ci95': np.quantile(da/db, [.025,.975]).tolist()}


def main():
    with np.load(ROOT/'data/interim/panel_v1/features_v1D.npz') as f:
        y, mask, y0 = f['y'].astype(float), f['m'].astype(bool), f['y0'].astype(float)
        w, reg, fam, system = f['w'].astype(float), f['regime'].astype(str), f['family'].astype(str), f['system'].astype(str)
        weather_names = f['weather_channels'].tolist()
        gust_peak = f['xu'][:, 72:, weather_names.index('gust')].max(1)
    n, tmax = y.shape
    onset = first_cross(y, mask)
    pk = np.max(np.where(mask, y, 0), 1); pkt = np.argmax(np.where(mask, y, -1), 1)
    tt = np.arange(tmax)[None, :]
    phases = {'below_1pct_entire_window': np.broadcast_to((onset<0)[:,None], mask.shape),
              'before_1pct_onset': (onset>=0)[:,None] & (tt<onset[:,None]),
              'onset_through_peak': (onset>=0)[:,None] & (tt>=onset[:,None]) & (tt<=pkt[:,None]),
              'after_peak': (onset>=0)[:,None] & (tt>pkt[:,None])}
    assert np.all(sum(phases.values()) == 1)
    complete = mask.all(1); big = pk>=.1
    ht_y = half_time(y, mask)
    prev_y = np.concatenate([y0[:,None], y[:,:-1]], 1)
    valid_pair = mask & np.concatenate([np.ones((n,1),bool), mask[:,:-1]], 1)
    dy = y - prev_y
    result = {'meta': {'tranche':'D', 'units':n, 'systems':len(np.unique(system)),
        'seed':0, 'onset_threshold':.01, 'sustained_recovery_hours':6,
        'bootstrap':{'draws':2000,'seed':20260924,'unit':'family within regime'},
        'phase_partition':'all observed hours, observed first >=1%, observed first global peak',
        'caveat':'Outcome-defined breakdowns and oracle repairs are descriptive only; rates are not identified from stocks.',
        'complete_paths':int(complete.sum()),'large_outages':int(big.sum())}, 'weather':{}, 'models':{}}
    for rr in REGIMES:
        ii = reg==rr
        result['weather'][rr] = {'county_event_max_gust_m_s':quantile(gust_peak[ii])}
    for name, label in LABELS.items():
        p, u, r = [np.zeros_like(y) for _ in range(3)]; seen=np.zeros(n,int)
        fold=np.zeros(n,int)
        for k in range(1,6):
            with np.load(ROOT/'runs/geo_weather_20260924'/label/f'fold{k:02d}'/'outer.npz') as z:
                idx=z['idx']; p[idx]=z['P']; u[idx]=z['u']; r[idx]=z['r']; seen[idx]+=1; fold[idx]=k
        assert (seen==1).all()
        reconstructed = stock(u,r,y0)
        err=float(np.max(np.abs(reconstructed-p)))
        assert err < 1e-6, err
        oracle, alpha, lag = oracle_errors(p,y,mask)
        ceiling=stock(u,np.zeros_like(r),y0)
        assert np.all(ceiling >= p-1e-6)
        ht_p = half_time(p, np.ones_like(mask), min_peak=0)
        on_p=first_cross(p,np.ones_like(mask))
        model={'stock_replay_max_abs':err, 'max_u_plus_r':float((u+r).max()),'by_regime':{},
               'oracle_remaining_sse_ratio': {key:bootstrap_ratio(oracle[key],oracle['base'],w,reg,fam)
                                             for key in ['timing','amplitude','joint']}}
        for rr in ['all']+REGIMES:
            ii=np.ones(n,bool) if rr=='all' else reg==rr
            wm=w[ii,None]*mask[ii]; se=wm*(p[ii]-y[ii])**2
            total=se.sum(); den=wm.sum()
            low=(y[ii]<.001); nearzero=ii & complete & (pk<.01)
            start=ii & complete & (y0<.01) & (onset>=0)
            hit=start & (on_p>=0)
            large=ii & big
            tail=ii & complete & big
            both=tail & np.isfinite(ht_y) & np.isfinite(ht_p)
            inc=valid_pair[ii] & (dy[ii]>0)
            req=np.divide(np.maximum(dy[ii],0),1-prev_y[ii],out=np.zeros_like(dy[ii]),where=prev_y[ii]<1)
            insuff=inc & (u[ii] < req)
            row={'n':int(ii.sum()),'rmse_pp':float(100*np.sqrt(total/den)),
                'mae_pp':float(100*np.sum(wm*np.abs(p[ii]-y[ii]))/den),
                'phase_sse_share':{key:float(np.sum(se*ph[ii])/total) for key,ph in phases.items()},
                'quiet_hour_weighted_share':float(wm[low].sum()/den),
                'quiet_hour_predicted_mean_pp':float(100*(wm*low*p[ii]).sum()/(wm*low).sum()),
                'quiet_hour_mae_share':float((wm*low*np.abs(p[ii]-y[ii])).sum()/(wm*np.abs(p[ii]-y[ii])).sum()),
                'all_below_1pct_counties':int(nearzero.sum()),
                'false_1pct_alarm_share_complete_quiet':float(np.mean(on_p[nearzero]>=0)) if nearzero.any() else None,
                'onset_eligible_complete_quiet_origin':int(start.sum()),
                'onset_miss_share':float(np.mean(on_p[start]<0)) if start.any() else None,
                'onset_error_hours_conditional_detected':quantile(on_p[hit]-onset[hit]),
                'large_peak_ratio':quantile(p.max(1)[large]/pk[large]),
                'large_peak_ratio_with_recovery_zero':quantile(ceiling.max(1)[large]/pk[large]),
                'large_damage_ceiling_below_half_observed_share':float(np.mean(ceiling.max(1)[large]<.5*pk[large])) if large.any() else None,
                'large_damage_ceiling_below_observed_share':float(np.mean(ceiling.max(1)[large]<pk[large])) if large.any() else None,
                'observed_half_time_hours_complete_large':quantile(ht_y[tail]),
                'predicted_half_time_hours_complete_large':quantile(ht_p[tail]),
                'half_time_error_hours_complete_large_both_uncensored':quantile(ht_p[both]-ht_y[both]),
                'observed_half_time_censored_count':int((tail & ~np.isfinite(ht_y)).sum()),
                'predicted_half_time_censored_count':int((tail & ~np.isfinite(ht_p)).sum()),
                'complete_large_n':int(tail.sum()),
                'oracle_scale_factor':quantile(alpha[ii]),
                'oracle_joint_lag_hours':quantile(lag[ii]),
                'oracle_remaining_sse_ratio':{key:float(np.sum(w[ii]*oracle[key][ii])/np.sum(w[ii]*oracle['base'][ii])) for key in ['timing','amplitude','joint']},
                'positive_increments_n':int(inc.sum()),
                'positive_increments_with_u_below_minimum_share':float(insuff.sum()/inc.sum()),
                'positive_increment_mass_with_u_below_minimum_share':float((np.maximum(dy[ii],0)*insuff).sum()/(np.maximum(dy[ii],0)*inc).sum())}
            model['by_regime'][rr]=row
        ss={s:float(np.sum(w[system==s]*oracle['base'][system==s])) for s in np.unique(system)}
        sorted_s=sorted(ss,key=ss.get,reverse=True)
        vals=np.array(list(ss.values()))
        model['error_concentration']={'top3_system_sse_share':float(sum(ss[s] for s in sorted_s[:3])/vals.sum()),
            'sse_effective_systems':float(vals.sum()**2/(vals@vals)),
            'top3_systems':[{'system':s,'regime':reg[np.where(system==s)[0][0]],'share':float(ss[s]/vals.sum())} for s in sorted_s[:3]]}
        result['models'][name]=model
        print(name, json.dumps(model['by_regime']['all']),flush=True)
    out=HERE/'results/v1/trajectory_diagnostics_s0.json'
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('Wrote',out.relative_to(ROOT))


if __name__=='__main__':
    main()
