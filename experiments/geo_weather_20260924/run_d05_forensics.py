"""Finite D-only large-outage audit. No fitting or source mutation."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='2'
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from d04_data import HERE, ROOT, load_panel
from d05_weather_features import fit_reference, extract_curves, summarize_anchors, flatten_summaries
from d05_frozen_inputs import load_frozen, load_host_extra

RUN=ROOT/'runs/geo_weather_20260924/large_outage_d05'
OUT=HERE/'results/v1'
OFF=np.arange(-48,25)
CLOCK=np.arange(72,216,12)

def clean(x):
    if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple,np.ndarray)): return [clean(v) for v in x]
    if isinstance(x,(np.integer,)): return int(x)
    if isinstance(x,(np.bool_,)): return bool(x)
    if isinstance(x,(np.floating,float)): return float(x) if np.isfinite(x) else None
    return x

def write(path,x):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(clean(x),ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
    tmp.replace(path)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def progress(stage):
    print(time.strftime('%Y-%m-%d %H:%M:%S'),stage,flush=True)
    write(RUN/'RUN_STATUS.json',dict(pid=os.getpid(),stage=stage,time=time.time()))

def quantile(x,w,q=(.1,.5,.9)):
    ok=np.isfinite(x)&np.isfinite(w)&(w>0)
    if not ok.any(): return np.full(len(q),np.nan)
    x,w=np.asarray(x)[ok],np.asarray(w)[ok]; ix=np.argsort(x)
    return np.interp(q,(np.cumsum(w[ix])-.5*w[ix])/w.sum(),x[ix])

def aggregate(matrix,unit,panel,boot=True):
    """Feature-specific complete support, conditional merged-event bootstrap."""
    x=np.asarray(matrix,dtype=float).reshape(len(unit),-1)
    w=panel['meta']['w'][unit].astype(float)
    levels,group=np.unique(panel['merged_group'][unit],return_inverse=True)
    valid=np.isfinite(x); numerator=np.zeros((len(levels),x.shape[1])); denom=np.zeros_like(numerator)
    np.add.at(numerator,group,np.where(valid,x,0)*w[:,None])
    np.add.at(denom,group,valid*w[:,None])
    den=denom.sum(0)
    mean=np.divide(numerator.sum(0),den,out=np.full(x.shape[1],np.nan),where=den>0)
    out=dict(mean=mean,n=valid.sum(0),weight=den,events=(denom>0).sum(0))
    if len(levels):
        out['event_kish']=np.divide(den**2,(denom**2).sum(0),out=np.full_like(den,np.nan),where=(denom**2).sum(0)>0)
        out['max_event_share']=np.divide(denom.max(0),den,out=np.full_like(den,np.nan),where=den>0)
    if boot:
        out.update(ci95=np.full((2,x.shape[1]),np.nan),valid_bootstrap_draws=np.zeros(x.shape[1],int),zero_support_draws=np.full(x.shape[1],999,int))
    if boot and len(levels)>1:
        rng=np.random.default_rng(20260928)
        counts=rng.multinomial(len(levels),np.full(len(levels),1/len(levels)),size=999)
        ci=np.full((2,x.shape[1]),np.nan); valid_draws=np.zeros(x.shape[1],int)
        for j in range(0,x.shape[1],256):
            d=counts@denom[:,j:j+256]; n=counts@numerator[:,j:j+256]
            b=np.divide(n,d,out=np.full_like(n,np.nan),where=d>0)
            valid_draws[j:j+256]=np.isfinite(b).sum(0)
            for k in range(b.shape[1]):
                if out['events'][j+k]>=2 and valid_draws[j+k]>0:
                    ci[:,j+k]=np.quantile(b[np.isfinite(b[:,k]),k],[.025,.975])
        out['ci95']=ci
        out['valid_bootstrap_draws']=valid_draws
        out['zero_support_draws']=999-valid_draws
    return out

def masks(panel):
    y,obs=panel['y_full'],panel['obs_full']
    p=np.where(obs&np.isfinite(y),y.astype(np.float64),np.nan)
    delta=np.full_like(p,np.nan); delta[:,1:]=p[:,1:]-p[:,:-1]
    return p,delta

def phenotypes(panel,p,d):
    n=len(p); us=np.arange(n); tJ=np.nanargmax(d[:,72:],axis=1)+72; tS=np.nanargmax(p[:,72:],axis=1)+72
    peakJ=d[us,tJ]; peakS=p[us,tS]; J=peakJ>=.01; S=peakS>=.10
    firstJ=np.argmax(np.nan_to_num(d[:,72:],nan=-np.inf)>=.01,axis=1)+72
    crossing=(p[:,72:]>=.10)&(p[:,71:215]<.10)
    crossok=crossing.any(1); firstS=crossing.argmax(1)+72
    anchors={'jump_max':(us[J],tJ[J]),'stock_max':(us[S],tS[S]),
             'jump_first':(us[J],firstJ[J]),'stock_cross':(us[S&crossok],firstS[S&crossok]),
             'fixed_clock':(us,np.full(n,120,dtype=int))}
    out={'counts':{'all':n,'J':int(J.sum()),'S':int(S.sum()),'J_and_S':int((J&S).sum()),
        'J_only':int((J&~S).sum()),'S_only':int((S&~J).sum()),'neither':int((~S&~J).sum()),
        'stock_ge20pct':int((peakS>=.2).sum()),'S_full_observed':int((S&np.isfinite(p[:,72:]).all(1)).sum()),
        'origin_already_10pct':int((p[:,71]>=.1).sum()),'S_no_upcross':int((S&~crossok).sum()),
        'valid_changes':int(np.isfinite(d[:,72:]).sum()),'missing_stock_hours':int((~np.isfinite(p[:,72:])).sum())},
        'definitions':{'J':'max one-hour net increase >=0.01','S':'max observed stock >=0.10','ties':'earliest','weights':'one original county-event design weight'},
        'strata':{},'persistence':{}}
    for name,sel in [('all',np.ones(n,bool)),('J',J),('S',S)]:
        for key in ['w','w_raw']:
            w=panel['meta'][key][sel]
            out['strata'][name+'_'+key]={'n':int(sel.sum()),'peak_stock_q10_50_90':quantile(peakS[sel],w),
              'peak_increment_q10_50_90':quantile(peakJ[sel],w),'origin_stock_q10_50_90':quantile(p[sel,71],w),
              'hours_stock_ge10pct_q10_50_90':quantile((p[sel,72:]>=.1).sum(1),w),
              'jump_hours_ge1pp_q10_50_90':quantile((d[sel,72:]>=.01).sum(1),w)}
    u,t=anchors['jump_max']; a=d[u,t]; prev=p[u,t-1]
    nxt=t+np.arange(1,7)[:,None]; good=nxt<216; vals=np.full((6,len(u)),np.nan)
    for j in range(6): vals[j,good[j]]=p[u[good[j]],nxt[j,good[j]]]
    coverage=np.isfinite(vals).all(0); nextgood=np.isfinite(vals[0]); nextdelta=vals[0]-p[u,t]
    sustained=np.where(coverage,(vals>=prev[None]+.5*a[None]).all(0).astype(float),np.nan)
    reversal=np.where(nextgood,(nextdelta<=-.8*a).astype(float),np.nan)
    out['persistence']={'fields':['immediate_80pct_reversal','six_hour_sustained_half_jump'],
       'statistics':aggregate(np.column_stack([reversal,sustained]),u,panel),
       'raw_counts':{'immediate_reversal':int(np.nansum(reversal)),'immediate_supported':int(nextgood.sum()),
                     'six_hour_sustained':int(np.nansum(sustained)),'six_hour_supported':int(coverage.sum())},
       'max_jump_boundary':int(np.isin(t,[72,215]).sum()),
       'max_jump_ties':int(((d[u,72:]==a[:,None]).sum(1)>1).sum()),
       'interpretation':'Outcome-selected shape diagnostic; reversal is not proof of a reporting artifact.'}
    return out,anchors

def model_audit(panel,p,d,anchors,frozen):
    out={'metadata':frozen['meta'],'models':{},'units':'fractions, hours; optional rates have no physical-flow interpretation'}
    for label,model in frozen['models'].items():
        pred=model['P'].astype(np.float64); common=np.where(np.isfinite(p[:,72:]),pred,np.nan)
        dp=np.diff(np.column_stack([p[:,71],pred]),axis=1)
        entry={'available_fields':model['available_fields'],'missing_folds':model['missing_folds']}
        for cohort in ['stock_max','jump_max']:
            u,t=anchors[cohort]; j=t-72; isstock=cohort=='stock_max'
            true=p[u,t] if isstock else d[u,t]; forecast=common[u] if isstock else np.where(np.isfinite(d[u,72:]),dp[u],np.nan)
            at=forecast[np.arange(len(u)),j]; signedmax=np.nanmax(forecast,axis=1)
            maximum=signedmax if isstock else np.maximum(signedmax,0)
            peak_t=(np.nanargmax(forecast,axis=1)+72).astype(float)
            peak_t[signedmax<=0]=np.nan
            columns=[at/true,maximum/true,peak_t-t]
            names=['at_true_anchor_ratio','max_on_observed_support_ratio','peak_time_error_hours']
            for radius in [3,6,12]:
                keep=abs(np.arange(72,216)[None]-t[:,None])<=radius
                mx=np.nanmax(np.where(keep,forecast,np.nan),axis=1)
                columns.append((mx if isstock else np.maximum(mx,0))/true);names.append(f'max_within_{radius}h_ratio')
            x=np.stack(columns,1)
            stats={'fields':names,'mean_ci':aggregate(x,u,panel),'weighted_q10_50_90':{},'raw_weight_q10_50_90':{},'unweighted_q10_50_90':np.nanquantile(x,[.1,.5,.9],axis=0)}
            for j,name in enumerate(names):
                stats['weighted_q10_50_90'][name]=quantile(x[:,j],panel['meta']['w'][u])
                stats['raw_weight_q10_50_90'][name]=quantile(x[:,j],panel['meta']['w_raw'][u])
            stats['all144_max_ratio_unweighted_q10_50_90']=np.quantile((np.max(pred[u],1) if isstock else np.maximum(np.max(dp[u],1),0))/true,[.1,.5,.9])
            if not isstock:stats['no_positive_predicted_increment']=int((signedmax<=0).sum())
            if isstock:
                stats['no_positive_predicted_stock']=int((signedmax<=0).sum())
                stats['fraction_predicted_reaches10pct']=float((np.max(pred[u],1)>=.10).mean())
                stats['fraction_predicted_reaches_half_true_max']=float((np.nanmax(common[u],1)>=.5*true).mean())
            entry[cohort]=stats
        out['models'][label]=entry
    return out

def curve_summary(panel,p,d,anchors,reference):
    result={'offsets':OFF,'weather_names':panel['feature_names']['weather'],'cohorts':{},
            'intervals':'Pointwise 95% merged-event bootstrap conditional on observed anchors/reference; no multiplicity correction.'}
    for name,(u,t) in anchors.items():
        curve=extract_curves(panel,u,t,reference=reference)
        y=np.full((len(u),len(OFF)),np.nan);dy=y.copy(); idx=t[:,None]+OFF
        valid=(idx>=0)&(idx<216); uu=np.broadcast_to(u[:,None],idx.shape)
        y[valid]=p[uu[valid],idx[valid]];dy[valid]=d[uu[valid],idx[valid]]
        x=np.concatenate([curve['standardized'],y[:,:,None],dy[:,:,None]],2)
        groups={'all':np.ones(len(u),bool)}
        if name in ['jump_max','stock_max']:
            groups.update({str(r):panel['meta']['regime'][u]==r for r in np.unique(panel['meta']['regime'])})
            groups.update({f'county_type_{k+1}':panel['county_type'][u]==k for k in range(6)})
        groups['complete_49h_outage_window']=((t>=24)&(t+24<216)&np.isfinite(y[:,24:]).all(1))
        result['cohorts'][name]={}
        for group,sel in groups.items():
            if sel.any(): result['cohorts'][name][group]={'n_units':int(sel.sum()),'shape':[73,14],
              'columns':panel['feature_names']['weather']+['stock','net_change'], 'summary':aggregate(x[sel],u[sel],panel),
              'raw_weather':aggregate(curve['raw'][sel],u[sel],panel,boot=False)}
    return result

def within_county(panel,reference,anchors,extra):
    out={'cohorts':{},'definition':'Case feature minus mean eligible fixed-clock feature in same county-event. Controls are not outcome filtered; exclude +/-12h.',
         'extra_label_warning':'Host extras retain stored labels; consult the separate source/cache semantics audit before physical interpretation.'}
    for name in ['jump_max','stock_max']:
        u,t=anchors[name]; s=summarize_anchors(panel,u,t,reference)
        f=flatten_summaries(s,blocks=('standardized','accumulation','synchronous','cross_band','delayed'))
        x=f['feature_matrix']; nf=x.shape[1]
        # Retain the interval-aligned near-hour values and every stored extra at -1/0/+1.
        near=s['near_hours']['standardized'].reshape(len(u),-1)
        en=np.full((len(u),3,28),np.nan)
        for j,off in enumerate([-1,0,1]):
            good=(t+off<216)&(t+off>=0); en[good,j]=extra['values'][u[good],t[good]+off]
        x=np.concatenate([x,near,en.reshape(len(u),-1)],1)
        names=f['feature_names']+[f'near_z:{off}:{wx}' for off in [-1,0,1] for wx in panel['feature_names']['weather']]+[f'host_stored:{off}:{wx}' for off in [-1,0,1] for wx in extra['feature_names']]
        controls=np.full_like(x,np.nan); eligible_n=np.zeros(len(u),int); control_jump=np.full(len(u),np.nan)
        _,delta=masks(panel)
        for start in range(0,len(u),48):
            ub=u[start:start+48];tb=t[start:start+48]
            cu=np.repeat(ub,len(CLOCK));ct=np.tile(CLOCK,len(ub)); owner=np.repeat(np.arange(len(ub)),len(CLOCK))
            keep=abs(ct-np.repeat(tb,len(CLOCK)))>12;cu,ct,owner=cu[keep],ct[keep],owner[keep]
            cs=summarize_anchors(panel,cu,ct,reference)
            cf=flatten_summaries(cs,blocks=('standardized','accumulation','synchronous','cross_band','delayed'))['feature_matrix']
            ce=np.stack([extra['values'][cu,ct+off] for off in [-1,0,1]],1).reshape(len(cu),-1)
            cx=np.concatenate([cf,cs['near_hours']['standardized'].reshape(len(cu),-1),ce],1)
            for j in range(len(ub)):
                select=owner==j; eligible_n[start+j]=select.sum(); vals=cx[select];finite=np.isfinite(vals);count=finite.sum(0)
                controls[start+j]=np.divide(np.where(finite,vals,0).sum(0),count,out=np.full(x.shape[1],np.nan),where=count>0)
                outcomes=delta[cu[select],ct[select]];good=np.isfinite(outcomes)
                if good.any(): control_jump[start+j]=np.mean(outcomes[good]>=.01)
        diff=x-controls
        groups={'all':np.ones(len(u),bool)}
        groups.update({str(r):panel['meta']['regime'][u]==r for r in np.unique(panel['meta']['regime'])})
        out['cohorts'][name]={'feature_names':names,'feature_strict_past':np.r_[f['feature_strict_past'],np.repeat(np.array([-1,0,1])<0,12),np.repeat(np.array([-1,0,1])<0,28)],
          'eligible_controls_min_max':[int(eligible_n.min()),int(eligible_n.max())],
          'control_jump_ge1pp_fraction':aggregate(control_jump[:,None],u,panel),
          'groups':{g:aggregate(diff[sel],u[sel],panel) for g,sel in groups.items() if sel.any()}}
        progress('within_county_'+name+'_done')
    return out

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    if not args.execute: raise SystemExit('Explicit --execute required')
    if RUN.exists(): raise SystemExit('Run directory exists: inspect and preserve; no overwrite/restart')
    RUN.mkdir(parents=True); OUT.mkdir(exist_ok=True)
    try:
        os.nice(max(0,15-os.getpriority(os.PRIO_PROCESS,0)))
        paths=[HERE/n for n in ['run_d05_forensics.py','d05_weather_features.py','d05_frozen_inputs.py','d05_observation_audit.py','d05_geography_matches.py','d04_data.py','notes/D05_LARGE_OUTAGE_FORENSICS_SCOPE_20260928.md']]
        manifest={str(p.relative_to(ROOT)):sha(p) for p in paths}
        write(RUN/'SCREEN_RUN.json',dict(source_hashes=manifest,threads=2,pid=os.getpid(),bootstrap_draws=999,data='D',neural_fits=0))
        with threadpool_limits(2):
            inputs=[ROOT/'data/interim/panel_v1/features_v1D.npz',HERE/'splits_v1D.json',ROOT/'runs/geo_weather_20260924/county_structure_d02/county_structure_d02_lookup.npz']
            for label in ['v1_host_s0','v1_gcrk_s0','v1_gcrk_georms_s0']:
                for k in range(1,6):
                    inputs.extend(ROOT/f'runs/geo_weather_20260924/{label}/fold{k:02d}'/f for f in ['outer.npz','DONE.json'])
            input_hashes={str(p.relative_to(ROOT)):sha(p) for p in inputs}
            write(RUN/'INPUT_MANIFEST.json',input_hashes)
            progress('load_D');panel=load_panel();p,d=masks(panel)
            with np.load(ROOT/'runs/geo_weather_20260924/county_structure_d02/county_structure_d02_lookup.npz',allow_pickle=False) as lookup:
                counties=lookup['county'].astype(str); labels=lookup['type']; mapping=dict(zip(counties,labels))
                if len(mapping)!=len(counties): raise ValueError('Duplicate D02 county')
                panel['county_type']=np.array([mapping[str(c)] for c in panel['meta']['county']],int)
                if not np.isin(panel['county_type'],np.arange(6)).all():raise ValueError('Invalid D02 type')
            progress('observation_audit')
            from d05_observation_audit import audit_d_panels
            observations=audit_d_panels(panel);write(OUT/'d05_observation_audit.json',observations)
            phenotype,anchors=phenotypes(panel,p,d);write(OUT/'d05_phenotypes.json',phenotype)
            progress('frozen_model_audit');frozen=load_frozen(panel)
            write(OUT/'d05_frozen_models.json',model_audit(panel,p,d,anchors,frozen));del frozen
            progress('weather_reference');reference=fit_reference(panel)
            write(RUN/'REFERENCE.json',reference)
            progress('aligned_curves');write(OUT/'d05_aligned_curves.json',curve_summary(panel,p,d,anchors,reference))
            progress('host_extra');extra=load_host_extra(panel)
            from d05_frozen_inputs import audit_extra_semantics
            write(OUT/'d05_host_feature_semantics.json',audit_extra_semantics(panel,extra))
            progress('within_county');write(OUT/'d05_within_county.json',within_county(panel,reference,anchors,extra));del extra
            progress('geographic_matching')
            from d05_geography_matches import analyze_matches
            write(OUT/'d05_geography_matches.json',analyze_matches(panel,reference,{k:anchors[k] for k in ['jump_max','stock_max']},bootstrap_draws=999))
        if manifest!={str(p.relative_to(ROOT)):sha(p) for p in paths}: raise RuntimeError('D05 source changed during run')
        if input_hashes!={str(p.relative_to(ROOT)):sha(p) for p in inputs}:raise RuntimeError('D05 input changed during run')
        write(RUN/'DONE.json',dict(source_hashes=manifest,results={p.name:sha(p) for p in OUT.glob('d05_*.json')},complete=True))
        progress('completed')
    except Exception as exc:
        write(RUN/'FAILED.json',dict(error=type(exc).__name__,message=str(exc),pid=os.getpid()))
        progress('failed');raise

if __name__=='__main__':main()
