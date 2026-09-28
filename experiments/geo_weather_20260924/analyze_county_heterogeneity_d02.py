"""Complete, descriptive county/weather association atlas; see D02_HETEROGENEITY_SCOPE.

One process, no neural fitting. Large/full-precision arrays remain under ignored runs/.
"""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '2'
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu
from scipy.stats import t as student_t
import analyze_data_relationships_v1 as D01

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RUN=ROOT/'runs/geo_weather_20260924/county_heterogeneity_d02'
OUT=HERE/'results/v1/county_heterogeneity_d02.json'
LOOKUP=ROOT/'runs/geo_weather_20260924/county_structure_d02/county_structure_d02_lookup.npz'
METHODS=['intercept','system_anchor','county','two_way']
RESPONSES=['mean_stock_change','positive_peak_rise','rise_risk']
ADJUST=['unadjusted','stock_history']
REG=D01.REG
FE_TOL=1e-9
FE_MAX=2000
DIAGNOSTICS={'projection_calls':0,'max_iterations':0,'direct_fallbacks':0,'maximum_direct_error':0.,'failed':0,
             'control_residual_rank_histogram':{}}


def weighted_quantile(x,w,ps=(.1,.9)):
    ok=np.isfinite(x)&np.isfinite(w)&(w>0)
    if not ok.any(): return np.full(len(ps),np.nan)
    x,w=x[ok],w[ok];o=np.argsort(x);x,w=x[o],w[o]
    return np.interp(ps,(np.cumsum(w)-.5*w)/w.sum(),x)


def group_operator(labels,w):
    names,code=np.unique(labels,return_inverse=True)
    den=np.bincount(code,weights=w,minlength=len(names))
    op=sparse.csr_matrix((w,(code,np.arange(len(w)))),shape=(len(names),len(w)))
    return names,code,den,op


def project_fe(mat,w,labels):
    """Weighted projection onto the complement of all supplied categorical effects."""
    if not labels: return mat-w@mat,0,True
    # Scale only for convergence/numerics; restore the original units afterwards.
    center=mat-w@mat
    scale=np.sqrt(w@(center*center));scale=np.where(scale>1e-12,scale,1.)
    z=center/scale
    ops=[group_operator(v,w)[1:] for v in labels]
    for step in range(1,FE_MAX+1):
        for code,den,op in ops:
            z-=(op@z/den[:,None])[code]
        if len(ops)==1 or step%5==0:
            err=max(float(np.max(np.abs(op@z/den[:,None]))) for _,den,op in ops)
            if err<FE_TOL:
                DIAGNOSTICS['projection_calls']+=1
                DIAGNOSTICS['max_iterations']=max(DIAGNOSTICS['max_iterations'],step)
                return z*scale,step,True
    # Sparse/disconnected incidence graphs can converge slowly under alternating
    # projection. Solve their exact weighted dummy projection after removing one
    # redundant dummy per connected component, then check the same group means.
    if len(ops)==2:
        sizes=[len(op[1]) for op in ops];codes=[ops[0][0],ops[1][0]+sizes[0]]
        rows=np.tile(np.arange(len(w)),2)
        design=sparse.csr_matrix((np.ones(2*len(w)),(rows,np.concatenate(codes))),shape=(len(w),sum(sizes)))
        gram=(design.T@design.multiply(w[:,None])).tocsc()
        _,components=connected_components(gram,directed=False)
        _,drop=np.unique(components,return_index=True)
        keep=np.ones(gram.shape[0],bool);keep[drop]=False
        degree=np.sqrt(gram.diagonal()[keep])
        reduced=gram[keep][:,keep].multiply(1/degree[:,None]).multiply(1/degree[None,:]).tocsc()
        rhs=np.asarray(design.T@(w[:,None]*(center/scale)))[keep]/degree[:,None]
        coef=splu(reduced).solve(rhs)/degree[:,None]
        z=center/scale-design[:,keep]@coef
        err=max(float(np.max(np.abs(op@z/den[:,None]))) for _,den,op in ops)
        DIAGNOSTICS['direct_fallbacks']+=1
        DIAGNOSTICS['maximum_direct_error']=max(DIAGNOSTICS['maximum_direct_error'],err)
        if err<FE_TOL:
            DIAGNOSTICS['projection_calls']+=1
            DIAGNOSTICS['max_iterations']=max(DIAGNOSTICS['max_iterations'],FE_MAX)
            return z*scale,FE_MAX,True
    DIAGNOSTICS['failed']+=1
    return z*scale,FE_MAX,False


def residual_controls(x,y,c,w):
    gram=c.T@(w[:,None]*c)
    eigen,vec=np.linalg.eigh(gram)
    # Controls were scaled to unit design-weighted variance before FE projection.
    # An absolute floor prevents inverting pure roundoff when FE absorbs all of C.
    threshold=max(1e-12,1e-10*max(float(eigen[-1]),0.))
    keep=eigen>threshold;basis=vec[:,keep]
    key=f'{int(keep.sum())}/{len(eigen)}'
    hist=DIAGNOSTICS['control_residual_rank_histogram'];hist[key]=hist.get(key,0)+1
    rhs=c.T@(w[:,None]*np.column_stack([x,y]))
    coef=basis@((basis.T@rhs)/eigen[keep,None])
    v=np.column_stack([x,y])-c@coef
    return v[:,:x.shape[1]],v[:,x.shape[1]:]


def slope_stats(x,y,w,cluster,ci=True):
    p,q=x.shape[1],y.shape[1]
    xx=w@(x*x)
    beta=np.divide(x.T@(w[:,None]*y),xx[:,None],out=np.full((p,q),np.nan),where=xx[:,None]>1e-12)
    out=np.full((p,q,3),np.nan);out[:,:,0]=beta
    diagnostics=np.full((p,3),np.nan)
    _,_,_,op=group_operator(cluster,w)
    gx=op@(x*x)
    leverage=np.divide(gx,xx[None,:],out=np.zeros_like(gx),where=xx[None,:]>1e-12)
    groups=(leverage>1e-8).sum(0)
    diagnostics[:,0]=groups;diagnostics[:,1]=leverage.max(0);diagnostics[:,2]=xx
    if ci:
        correction=np.divide(groups,groups-1,out=np.full(p,np.nan),where=groups>1)
        critical=student_t.ppf(.975,np.maximum(groups-1,1))
        for target in range(q):
            scores=op@(x*y[:,target,None])-gx*beta[:,target]
            se=np.sqrt((scores*scores).sum(0)*correction)/np.maximum(xx,1e-300)
            out[:,target,1]=beta[:,target]-critical*se
            out[:,target,2]=beta[:,target]+critical*se
    return out,diagnostics


def fit_batch(d,idx,cols,methods=METHODS,ci=True,speed=False,weight='w'):
    """Rows must have finite selected drivers. Two adjustment levels share each FE projection."""
    p=len(cols);res=np.full((p,len(methods),2,3,3),np.nan)
    support=np.full((p,len(methods),2,3),np.nan)
    if len(idx)<3:return res,support
    x=d['X'][np.ix_(idx,cols)];y=d['Y'][idx];c=d['C'][idx]
    w=d[weight][idx].astype(float);w/=w.sum()
    if speed:c=np.column_stack([c,d['speed'][idx]])
    cs=np.sqrt(w@((c-w@c)**2));cs=np.where(cs>1e-12,cs,1.)
    c=(c-w@c)/cs
    mat=np.column_stack([x,y,c])
    for k,method in enumerate(methods):
        labels=[]
        if method in ('system_anchor','two_way'):labels.append(d['sa'][idx])
        if method in ('county','two_way'):labels.append(d['county'][idx])
        z,_,ok=project_fe(mat,w,labels)
        if not ok:continue
        xx,yy,cc=z[:,:p],z[:,p:p+3],z[:,p+3:]
        for a in range(2):
            ax,ay=residual_controls(xx,yy,cc,w) if a else (xx,yy)
            res[:,k,a],support[:,k,a]=slope_stats(ax,ay,w,d['group'][idx],ci)
    return res,support


def support_info(d,idx):
    if len(idx)==0:return dict(rows=0,counties=0,merged_groups=0,repeated_family_counties=0,repeated_merged_group_counties=0,grey=True,cross_event_grey=True)
    w=d['w'][idx];counties=np.unique(d['county'][idx]);groups=np.unique(d['group'][idx])
    wc=np.array([w[d['county'][idx]==c].sum() for c in counties])
    wg=np.array([w[d['group'][idx]==g].sum() for g in groups])
    repeated=sum(len(np.unique(d['family'][idx][d['county'][idx]==c]))>=2 for c in counties)
    repeated_groups=sum(len(np.unique(d['group'][idx][d['county'][idx]==c]))>=2 for c in counties)
    return dict(rows=len(idx),county_events=len(np.unique(d['unit'][idx])),counties=len(counties),
        systems=len(np.unique(d['system'][idx])),families=len(np.unique(d['family'][idx])),merged_groups=len(groups),
        repeated_family_counties=int(repeated),repeated_merged_group_counties=int(repeated_groups),county_weight_kish=float(wc.sum()**2/(wc@wc)),
        group_weight_kish=float(wg.sum()**2/(wg@wg)),maximum_group_weight_share=float(wg.max()/wg.sum()),
        grey=bool(len(counties)<20 or len(groups)<5),cross_event_grey=bool(repeated_groups<10))


def prepare():
    d=D01.load_rows()
    with np.load(LOOKUP,allow_pickle=False) as z:
        counties=z['county'].astype(str);types=z['type6'] if 'type6' in z else z['type']
        fn=z['feature_names'].astype(str).tolist();geo=z['features']
        mapping={c:int(t) for c,t in zip(counties,types)}
        row={c:i for i,c in enumerate(counties)}
        g=geo[[row[c] for c in d['county']]]
        aspect=g[:,[fn.index('aspect_'+a) for a in ['N','NE','E','SE','S','SW','W','NW']]]
    typ=np.array([mapping[c] for c in d['county']])
    assert set(np.unique(typ))==set(range(6))
    with np.load(D01.FEAT,allow_pickle=False) as f:
        names=f['damage_features'][:12].astype(str).tolist()
        raw_y=f['y_full'].astype(float);mask=f['obs_full'].astype(bool)
        family=f['family'].astype(str)[d['unit']]
    peak=np.zeros(len(d['unit']))
    for a in D01.ANCHORS:
        rows=np.flatnonzero(d['anchor']==a);units=d['unit'][rows]
        v=np.where(mask[units,a:a+24],raw_y[units,a:a+24],np.nan)
        peak[rows]=np.maximum(np.nanmax(v,1)-d['control'][rows,0],0.)
    del raw_y,mask
    wx=d.pop('future');del d['past']
    means=wx.mean(1);contrast=wx[:,12:].mean(1)-wx[:,:12].mean(1)
    core=wx[:,:,d['widx']]
    compound=np.stack([(core[:,:,i]*core[:,:,j]).mean(1) for i,j in D01.PAIRS],1)
    _,_,order=D01.moments(core)
    theta=np.arange(8)*np.pi/4
    east=aspect@np.sin(theta);north=aspect@np.cos(theta)
    resultant=np.hypot(east,north)
    u=wx[:,:,names.index('u10')];v=wx[:,:,names.index('v10')]
    reliable=(np.hypot(u,v)>=2)&(resultant[:,None]>=.1)&np.isfinite(resultant[:,None])
    den=np.maximum(resultant,1e-300)
    along=(u*east[:,None]+v*north[:,None])/den[:,None]
    cross=(u*north[:,None]-v*east[:,None])/den[:,None]
    nh=reliable.sum(1);valid=nh>=12
    proj=np.column_stack([(np.where(reliable,z,0).sum(1)/np.maximum(nh,1)) for z in [along,cross]])
    proj[~valid]=np.nan
    X=np.column_stack([means,contrast,compound,order,proj])
    drivers=['mean:'+x for x in names]+['late_minus_early:'+x for x in names]
    drivers+=['compound:'+D01.WEATHER[i]+'*'+D01.WEATHER[j] for i,j in D01.PAIRS]
    drivers+=['order:'+D01.WEATHER[i]+'->'+D01.WEATHER[j] for i,j in D01.PAIRS]
    drivers+=['terrain_along_wind','terrain_cross_wind']
    assert X.shape[1]==len(drivers)==56
    scales=np.zeros((5,56));centers=np.zeros((5,56));quantiles=np.full((5,6,56,2),np.nan)
    for ri,r in enumerate(REG):
        ii=np.flatnonzero(d['reg']==r)
        for j in range(56):
            ok=ii[np.isfinite(X[ii,j])];ww=d['w'][ok];ww=ww/ww.sum()
            centers[ri,j]=ww@X[ok,j]
            scales[ri,j]=np.sqrt(ww@((X[ok,j]-centers[ri,j])**2))
            for k in range(6):
                kk=ii[typ[ii]==k];quantiles[ri,k,j]=weighted_quantile(X[kk,j],d['w'][kk])
            if scales[ri,j]>1e-12:X[ii,j]=(X[ii,j]-centers[ri,j])/scales[ri,j]
            else:X[ii,j]=np.nan
    Y=100*np.column_stack([d['y'][:,0]-d['control'][:,0],peak,d['y'][:,1]])
    result={k:d[k] for k in ['unit','anchor','county','system','fold','group','reg','w','w_raw']}
    result.update(X=X,Y=Y,C=d['control'][:,:5].copy(),family=family,type=typ,
        sa=np.array([f'{s}:{a}' for s,a in zip(d['system'],d['anchor'])]),
        speed=np.column_stack([means[:,names.index('wind_speed')],contrast[:,names.index('wind_speed')]]))
    audit=d['audit'];del d,wx,core,u,v,means,contrast,compound,order,g,aspect,along,cross
    return result,drivers,scales,centers,quantiles,audit,dict(reliable_projection_rows=int(valid.sum()),
        excluded_projection_rows=int((~valid).sum()),minimum_resultant=.1,minimum_wind_speed_ms=2,
        minimum_reliable_hours=12,mean_resultant=float(np.nanmean(resultant)))


def aggregate_events(d):
    units,first,code=np.unique(d['unit'],return_index=True,return_inverse=True)
    den=np.bincount(code,weights=d['w']);n=len(units)
    out={k:d[k][first] for k in ['unit','anchor','county','system','fold','group','reg','family','type','sa']}
    out['w']=den;out['w_raw']=np.bincount(code,weights=d['w_raw'])
    for key in ['X','Y','C','speed']:
        values=d[key];z=np.full((n,values.shape[1]),np.nan)
        for j in range(values.shape[1]):
            ok=np.isfinite(values[:,j]);dv=np.bincount(code[ok],weights=d['w'][ok],minlength=n)
            sm=np.bincount(code[ok],weights=d['w'][ok]*values[ok,j],minlength=n)
            z[:,j]=np.divide(sm,dv,out=np.full(n,np.nan),where=dv>0)
        out[key]=z
    return out


def clean(v):
    if isinstance(v,np.ndarray):return clean(v.tolist())
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if isinstance(v,(np.integer,)):return int(v)
    if isinstance(v,(np.bool_,)):return bool(v)
    if isinstance(v,(float,np.floating)):
        return float(f'{v:.7g}') if np.isfinite(v) else None
    return v


def cancellation(slopes,q):
    # Absent types have q=0; refuse missing represented types rather than reweighting by driver.
    active=q>0;slopes=slopes[active];q=q[active]
    valid=np.isfinite(slopes).all(0)
    signed=np.tensordot(q,slopes,axes=(0,0));absolute=np.tensordot(q,np.abs(slopes),axes=(0,0))
    index=np.divide(np.abs(signed),absolute,out=np.full_like(signed,np.nan),where=absolute>1e-10)
    return np.stack([np.where(valid,signed,np.nan),np.where(valid,absolute,np.nan),
                     np.where(valid,1-index,np.nan)],-1)


def self_test():
    rng=np.random.default_rng(31);n=240
    county=np.tile(np.arange(12),20);event=rng.integers(0,8,n);w=rng.uniform(.3,3,n);w/=w.sum()
    m=rng.normal(size=(n,4));z,_,ok=project_fe(m,w,[county,event]);assert ok
    design=np.column_stack([np.eye(12)[county],np.eye(8)[event]])
    expected=m-design@np.linalg.lstsq(design*np.sqrt(w[:,None]),m*np.sqrt(w[:,None]),rcond=None)[0]
    assert np.max(np.abs(z-expected))<2e-8
    # Force the sparse fallback on a disconnected, weakly connected graph.
    global FE_MAX
    previous=FE_MAX;FE_MAX=1
    county=np.repeat(np.arange(40),2);event=np.column_stack([np.arange(40),np.minimum(np.arange(40)+1,39)]).ravel()
    event[40:]+=1;w=rng.uniform(.1,2,len(county));w/=w.sum();m=rng.normal(size=(len(w),3))
    z,_,ok=project_fe(m,w,[county,event]);assert ok
    design=np.column_stack([np.eye(40)[county],np.eye(event.max()+1)[event]])
    expected=m-design@np.linalg.lstsq(design*np.sqrt(w[:,None]),m*np.sqrt(w[:,None]),rcond=None)[0]
    assert np.max(np.abs(z-expected))<2e-8
    FE_MAX=previous
    n=240;event=rng.integers(0,8,n);w=rng.uniform(.3,3,n);w/=w.sum()
    x=rng.normal(size=(n,2));y=3*x[:,:1]+rng.normal(0,.01,(n,1))
    b,_=slope_stats(x-x.mean(0),y-y.mean(0),np.full(n,1/n),event)
    assert abs(b[0,0,0]-3)<.01
    control=rng.normal(size=(n,3));xx=x-w@x;yy=y-w@y;cc=control-w@control
    rx,ry=residual_controls(xx,yy,cc,w)
    partial,_=slope_stats(rx,ry,w,event)
    design=np.column_stack([np.ones(n),x[:,0],control])
    direct=np.linalg.lstsq(design*np.sqrt(w[:,None]),y*np.sqrt(w[:,None]),rcond=None)[0][1,0]
    assert abs(partial[0,0,0]-direct)<1e-10
    # Fully absorbed controls have rank zero after FE; do not invert their roundoff.
    category=np.tile(np.arange(12),20);dummy=np.eye(12)[category]
    controls=dummy@rng.normal(size=(12,3))
    controls=(controls-w@controls)/np.sqrt(w@((controls-w@controls)**2))
    z,_,ok=project_fe(np.column_stack([x[:,:1],y,controls]),w,[category]);assert ok
    rx,ry=residual_controls(z[:,:1],z[:,1:2],z[:,2:],w)
    b,_=slope_stats(rx,ry,w,event)
    design=np.column_stack([x[:,0],dummy,controls])
    direct=np.linalg.lstsq(design*np.sqrt(w[:,None]),y*np.sqrt(w[:,None]),rcond=None)[0][0,0]
    assert abs(b[0,0,0]-direct)<1e-10
    assert DIAGNOSTICS['control_residual_rank_histogram'].get('0/3',0)>0
    assert np.allclose(cancellation(np.array([[2.],[-2.]]),np.array([.5,.5]))[0],[0,2,1])
    assert np.isnan(cancellation(np.array([[np.nan],[2.]]),np.array([.5,.5]))).all()
    print('self tests passed',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--self-test',action='store_true');a=ap.parse_args()
    if os.getpriority(os.PRIO_PROCESS,0)<15:os.nice(15-os.getpriority(os.PRIO_PROCESS,0))
    if a.self_test:self_test();return
    assert not OUT.exists(),'preserve existing result before any rerun'
    assert LOOKUP.exists(),'county lookup must exist before starting'
    RUN.mkdir(parents=True,exist_ok=True)
    frozen={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [Path(__file__),Path(D01.__file__),HERE/'notes/D02_HETEROGENEITY_SCOPE_20260928.md',LOOKUP]}
    started=time.time();d,drivers,scales,centers,quantiles,audit,orientation=prepare()
    print(json.dumps(dict(stage='prepared',rows=len(d['unit']),drivers=len(drivers))),flush=True)
    shape=(5,6,56,4,2,3)
    estimate=np.full(shape+(3,),np.nan);diag=np.full((5,6,56,4,2,3),np.nan)
    fold=np.full(shape+(5,),np.nan)
    fold_diag=np.full((5,6,56,4,2,3,5),np.nan)
    matched=np.full((5,6,56,2,3,3),np.nan)
    matched_support=np.zeros((5,6,56,3))
    matched_diag=np.full((5,6,56,2,3),np.nan)
    raw_weight=np.full((5,6,56,3),np.nan)
    wind=np.full((5,6,4,4,3,3),np.nan)
    support={};q=np.zeros((5,6));overlap=np.full((5,56,2),np.nan)
    windcols=[drivers.index(n) for n in ['mean:u10','mean:v10','late_minus_early:u10','late_minus_early:v10']]
    for ri,r in enumerate(REG):
        regidx=np.flatnonzero(d['reg']==r)
        q[ri]=[len(np.unique(d['county'][regidx[d['type'][regidx]==k]])) for k in range(6)];q[ri]/=q[ri].sum()
        overlap[ri,:,0]=np.max(quantiles[ri,q[ri]>0,:,0],axis=0)
        overlap[ri,:,1]=np.min(quantiles[ri,q[ri]>0,:,1],axis=0)
        for k in range(6):
            ii=regidx[d['type'][regidx]==k];support[f'{r}/type{k}']=support_info(d,ii)
            # Columns with identical finite-row patterns share projections.
            batches={}
            for j in range(56):
                key=np.isfinite(d['X'][ii,j]).tobytes();batches.setdefault(key,[]).append(j)
            for cols in batches.values():
                jj=ii[np.isfinite(d['X'][ii,cols[0]])]
                e,s=fit_batch(d,jj,cols);estimate[ri,k,cols]=e;diag[ri,k,cols]=s
                for f in range(1,6):
                    ff=jj[d['fold'][jj]==f];ef,sf=fit_batch(d,ff,cols,ci=False)
                    fold[ri,k,cols,...,f-1]=ef[...,0]
                    fold_diag[ri,k,cols,...,f-1]=sf
                rw,_=fit_batch(d,jj,cols,methods=['two_way'],ci=False,weight='w_raw')
                raw_weight[ri,k,cols]=rw[:,0,1,:,0]
            ww,_=fit_batch(d,ii,windcols,speed=True);wind[ri,k]=ww[:,:,1]
            for j in range(56):
                lo,hi=overlap[ri,j]
                if not np.isfinite(lo+hi) or hi-lo<1e-10 or scales[ri,j]<=1e-12:continue
                lower,upper=(np.array([lo,hi])-centers[ri,j])/scales[ri,j]
                jj=ii[(d['X'][ii,j]>=lower)&(d['X'][ii,j]<=upper)]
                if not len(jj):continue
                e,s=fit_batch(d,jj,[j],methods=['intercept','two_way'])
                matched[ri,k,j,0]=e[0,0,0];matched[ri,k,j,1]=e[0,1,1]
                matched_diag[ri,k,j,0]=s[0,0,0];matched_diag[ri,k,j,1]=s[0,1,1]
                matched_support[ri,k,j]=[len(jj),len(np.unique(d['county'][jj])),len(np.unique(d['group'][jj]))]
            print(json.dumps(dict(stage='cell_complete',regime=r,type=k,seconds=round(time.time()-started,1))),flush=True)
    event=aggregate_events(d)
    event_estimate=np.full((5,6,56,2,3,3),np.nan)
    event_support={}
    for ri,r in enumerate(REG):
        for k in range(6):
            ii=np.flatnonzero((event['reg']==r)&(event['type']==k));event_support[f'{r}/type{k}']=support_info(event,ii)
            batches={}
            # Terrain projections use a subset of windows; leave these event rows
            # unpublished because event-average Y/C use all retained windows.
            for j in range(54):batches.setdefault(np.isfinite(event['X'][ii,j]).tobytes(),[]).append(j)
            for cols in batches.values():
                jj=ii[np.isfinite(event['X'][ii,cols[0]])];e,_=fit_batch(event,jj,cols,methods=['county'])
                event_estimate[ri,k,cols]=e[:,0]
    assert DIAGNOSTICS['failed']==0,'FE projection did not converge; preserve run for inspection'
    full_cancel=np.stack([cancellation(estimate[ri,...,0],q[ri]) for ri in range(5)])
    matched_cancel=np.stack([cancellation(matched[ri,...,0],q[ri]) for ri in range(5)])
    foldcounts=np.stack([(fold<0).sum(-1),(fold>0).sum(-1),np.isfinite(fold).sum(-1)],-1)
    np.savez_compressed(RUN/'atlas.npz',estimate=estimate,fold_slopes=fold,fold_diagnostics=fold_diag,diagnostics=diag,matched=matched,matched_diagnostics=matched_diag,
        matched_support=matched_support,event_county_fe=event_estimate,raw_weight_two_way_adjusted=raw_weight,
        wind_speed_adjusted=wind,scales=scales,centers=centers,quantiles=quantiles,overlap=overlap,type_q=q,
        full_cancellation=full_cancel,matched_cancellation=matched_cancel,drivers=np.array(drivers))
    # All arrays are complete: no effect-size or significance selection.
    result=dict(meta=dict(analysis='D02 county-structure association atlas',exploratory=True,
        responses=RESPONSES,response_units='percentage points per fixed regime-driver SD',drivers=drivers,
        regimes=REG,types=list(range(6)),methods=METHODS,adjustments=ADJUST,estimate_last_axis=['slope','ci95_low','ci95_high'],
        frozen_sha256=frozen,elapsed_seconds=time.time()-started,resources=dict(nice=os.getpriority(os.PRIO_PROCESS,0),numerical_threads=2),
        inference='merged-group cluster sandwich, G/(G-1), t(G-1); descriptive, not simultaneous or causal',
        control_rank_threshold='max(1e-12,1e-10*largest residual Gram eigenvalue); controls standardized to weighted variance one before FE',
        full_precision_archive=str((RUN/'atlas.npz').relative_to(ROOT)),global_MSE_gate=False,
        q_definition='Within each regime, type shares among unique sampled counties; each county counts once; no outcome or design-weight weighting.',
        county_event_excluded_drivers=['terrain_along_wind','terrain_cross_wind']),
        audit=audit,orientation_support=orientation,projection_diagnostics=DIAGNOSTICS,cell_support=support,
        driver_regime_sd=scales,driver_regime_mean=centers,type_driver_p10_p90=quantiles,common_p10_p90_interval=overlap,type_q=q,
        estimates=estimate,fold_sign_counts=foldcounts,fold_sign_count_order=['negative','positive','finite'],
        fold_residual_driver_diagnostics=fold_diag,
        residual_driver_diagnostics=diag,residual_driver_diagnostic_order=['informative_merged_groups','maximum_group_leverage','weighted_residual_variance'],
        common_support_estimates=matched,common_support_methods=['intercept_unadjusted','two_way_stock_history'],
        common_support_counts=matched_support,common_support_count_order=['windows','counties','merged_groups'],
        common_support_residual_driver_diagnostics=matched_diag,
        county_event_aggregate_estimates=event_estimate,county_event_support=event_support,
        raw_weight_two_way_stock_history_slopes=raw_weight,wind_speed_adjusted_estimates=wind,wind_driver_indices=windcols,
        full_domain_cancellation=full_cancel,common_support_cancellation=matched_cancel,
        cancellation_last_axis=['signed_q_mean','absolute_q_mean','cancellation_index'],
        interpretation_limits=['Every driver is a separate bivariate/partial association; correlated weather is not independently intervened on.',
        'Window county FE includes within-event evolution; county-event aggregate FE is the distinct cross-event sensitivity.',
        'The fixed types use all D county descriptors; outcome blind, transductive, not spatial validation.',
        'Common P10-P90 support matches one weather marginal, not joint histories or all event composition.',
        'Cancellation indices have no standalone uncertainty guarantee and can be high when slopes are noise; use cell intervals, support and fold replication.',
        'Event-cluster intervals do not jointly account for repeated-county dependence or multiple comparisons.',
        'Five fold-subset slopes are descriptive replication estimates, not OOF forecast scores.',
        'Stock change and peak rise do not separate physical damage from repair; p0/history adjustment may absorb prior exposure effects.',
        'Grey support guidance: <20 counties or <5 merged groups; cross-event <10 counties with >=2 merged groups; inspect driver-specific residual groups and common-support counts.',
        'Terrain projection county-event aggregate slopes are omitted because missing-window support would differ from aggregated outcomes/controls; window-level projection slopes remain complete.',
        'No model selection, new neural fitting, sealed evaluation or manuscript change.'])
    for path,digest in frozen.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
    OUT.write_text(json.dumps(clean(result),separators=(',',':'),allow_nan=False)+'\n')
    print(json.dumps(dict(result=str(OUT.relative_to(ROOT)),seconds=round(time.time()-started,1),projection=DIAGNOSTICS)),flush=True)


if __name__=='__main__':main()
