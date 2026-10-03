"""Assemble fixed T03 OOF predictions, validation-selected blending, and all results."""
from t03_poster_trajectory import HERE,ROOT,PRIVATE,OUT,ARMS,indices,damping,mse,digest,dump
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import json
import torch,lightgbm

MODELS=('ZERO','PERSISTENCE','DAMPED','HOST','DOSE','GCRK','GEO_MLP','TREE_W','TREE_G','BLEND')
BLEND_ARMS=('HOST','DOSE','GCRK','GEO_MLP','TREE_W','TREE_G')

def summarize(p,y,mask,w):
    weight=np.asarray(w,dtype=np.float64)[:,None]*mask
    return dict(rmse=float(np.sqrt(np.sum(weight*(p-y)**2)/weight.sum())),
        mae=float(np.sum(weight*abs(p-y))/weight.sum()),cells=int(mask.sum()),weight_sum=float(weight.sum()),
        sse=float(np.sum(weight*(p-y)**2)),sae=float(np.sum(weight*abs(p-y))))

def main():
    d=dict(np.load(PRIVATE/'data.npz',allow_pickle=False));n=len(d['y'])
    build=json.loads((OUT/'build_audit.json').read_text());assert digest(PRIVATE/'data.npz')==build['cache_sha256']
    pred={a:np.full((n,144),np.nan) for a in MODELS};seedpred={(a,s):np.full((n,144),np.nan) for a in ARMS for s in range(3)}
    coverage=np.zeros(n,int);blends=[];training=[];trees=[];foldscores=[];checks=[]
    for fold in range(5):
        tr,va,te=indices(d,fold);coverage[te]+=1
        rate=damping(d,tr);pred['ZERO'][te]=0;pred['PERSISTENCE'][te]=d['y0'][te,None]
        pred['DAMPED'][te]=d['y0'][te,None]*np.exp(-rate*np.arange(1,145)[None,:])
        validation={}
        for arm in ARMS:
            vp=[];tp=[]
            for seed in range(3):
                z=np.load(PRIVATE/f'{arm}_f{fold}_s{seed}.npz');assert np.array_equal(z['va'],va) and np.array_equal(z['te'],te)
                vp.append(z['vp']);tp.append(z['tp']);seedpred[arm,seed][te]=z['tp']
                r=json.loads((PRIVATE/f'{arm}_f{fold}_s{seed}.json').read_text())
                assert r['paired_initial_max_error']<1e-6
                training.append(r)
            validation[arm]=np.mean(vp,axis=0);pred[arm][te]=np.mean(tp,axis=0)
        z=np.load(PRIVATE/f'tree_f{fold}.npz');assert np.array_equal(z['va'],va) and np.array_equal(z['te'],te)
        for arm in ('TREE_W','TREE_G'):validation[arm]=z[arm+'_vp'];pred[arm][te]=z[arm+'_tp']
        trees.append(json.loads((PRIVATE/f'tree_f{fold}.json').read_text()))
        x=np.stack([validation[a].ravel() for a in BLEND_ARMS],axis=1).astype(float)
        target=d['y'][va].ravel();weight=(d['m'][va]*d['w'][va,None]).ravel().astype(float);weight/=weight.sum()
        q=x.T@(weight[:,None]*x);b=x.T@(weight*target)
        def obj(w):return float(w@q@w-2*b@w)
        def jac(w):return 2*(q@w-b)
        result=minimize(obj,np.ones(6)/6,jac=jac,method='SLSQP',bounds=[(0,1)]*6,
            constraints=[dict(type='eq',fun=lambda w:w.sum()-1,jac=lambda w:np.ones(6))],options=dict(ftol=1e-12,maxiter=2000))
        assert result.success and abs(result.x.sum()-1)<1e-8
        bw=result.x.clip(0,1);bw/=bw.sum()
        pred['BLEND'][te]=sum(bw[j]*pred[a][te] for j,a in enumerate(BLEND_ARMS))
        valp=(x@bw).reshape(len(va),144)
        best_single=min(mse(validation[a],d,va) for a in BLEND_ARMS);blend_mse=mse(valp,d,va)
        assert blend_mse<=best_single+1e-8
        blends.append(dict(fold=fold,damping_rate=rate,validation_mse=blend_mse,best_single_validation_mse=best_single,weights=dict(zip(BLEND_ARMS,bw.tolist()))))
        for arm in MODELS:
            foldscores.append(dict(fold=fold,model=arm,**summarize(pred[arm][te],d['y'][te],d['m'][te],d['w'][te])))
    assert np.all(coverage==1)
    for a,p in pred.items():assert np.isfinite(p).all() and ((p>=-1e-8)&(p<=1+1e-8)).all(),a
    np.savez_compressed(PRIVATE/'oof_ensembles.npz',**pred)
    records=[];events=[];seed_scores=[]
    severe=np.max(np.where(d['m'],d['y'],-np.inf),axis=1)>=.1
    for arm,p in pred.items():
        for weighting,w in [('design',d['w']),('unweighted',np.ones(n))]:
            for scope,cols in [('full144',np.arange(144))]+[(f'exact+{h}',np.array([h-1])) for h in (1,6,24,48)]:
                records.append(dict(model=arm,weighting=weighting,scope=scope,group='all',**summarize(p[:,cols],d['y'][:,cols],d['m'][:,cols],w)))
            for label,sel in [('severe',severe),('nonsevere',~severe)]:
                records.append(dict(model=arm,weighting=weighting,scope='full144',group=label,**summarize(p[sel],d['y'][sel],d['m'][sel],w[sel])))
        for event in np.unique(d['system']):
            sel=d['system']==event;events.append(dict(model=arm,system=event,n=int(sel.sum()),**summarize(p[sel],d['y'][sel],d['m'][sel],d['w'][sel])))
    for (arm,seed),p in seedpred.items():
        assert np.isfinite(p).all()
        seed_scores.append(dict(model=arm,seed=seed,**summarize(p,d['y'],d['m'],d['w'])))
    metric=pd.DataFrame(records);folds=pd.DataFrame(foldscores)
    for arm in MODELS:
        row=metric[(metric.model==arm)&(metric.weighting=='design')&(metric.scope=='full144')&(metric.group=='all')].iloc[0]
        sub=folds[folds.model==arm];error=max(abs(row.rmse-np.sqrt(sub.sse.sum()/sub.weight_sum.sum())),abs(row.mae-sub.sae.sum()/sub.weight_sum.sum()))
        assert error<1e-12;checks.append(error)
    # Resample counties, keeping their repeated appearances in different storms together.
    counties,ix=np.unique(d['fips'],return_inverse=True);g=len(counties)
    cluster={a:np.bincount(ix,weights=np.sum((p-d['y'])**2*d['m']*d['w'][:,None],axis=1),minlength=g) for a,p in pred.items()}
    draws=np.random.default_rng(20261003).multinomial(g,np.ones(g)/g,size=2000)
    comparisons=[]
    pairs=[(a,'HOST') for a in MODELS if a!='HOST']+[('GCRK','DOSE'),('GCRK','GEO_MLP'),('BLEND','TREE_W'),('BLEND','TREE_G')]
    for candidate,base in pairs:
        delta=100*(1-np.sqrt((draws@cluster[candidate])/(draws@cluster[base])))
        gain=100*(1-np.sqrt(cluster[candidate].sum()/cluster[base].sum()))
        lo,hi=np.quantile(delta,[.025,.975]);comparisons.append(dict(model=candidate,reference=base,rmse_gain_pct=float(gain),ci025_pct=float(lo),ci975_pct=float(hi)))
    for name,rows in [('metrics',records),('per_event',events),('per_seed',seed_scores),('fold_scores',foldscores),('comparisons',comparisons)]:pd.DataFrame(rows).to_csv(OUT/f'{name}.csv',index=False)
    dump(OUT/'blend_choices.json',blends);dump(OUT/'tree_choices.json',trees)
    # Keep parameter update summaries and validation curves, no weights or county predictions.
    dump(OUT/'training_audit.json',training)
    assert len(training)==60 and len(trees)==5
    dump(OUT/'verification.json',dict(county_events=n,counties=g,neural_jobs=len(training),tree_folds=len(trees),oof_exact_once=True,
        pooled_reconstruction_max_error=max(checks),paired_initial_max_error=max(r['paired_initial_max_error'] for r in training),
        steps=600,seeds=[0,1,2],source_sha256=digest(HERE/'t03_poster_trajectory.py'),score_source_sha256=digest(__file__),
        bootstrap_count=2000,bootstrap_unit='county',core_source_sha256={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'src/asymode/asym_host.py',ROOT/'src/asymode/gcrk.py')},
        versions=dict(numpy=np.__version__,pandas=pd.__version__,torch=torch.__version__,lightgbm=lightgbm.__version__),
        scope='Examined development panel, conditional on fitted models; not independent confirmation'))
    full=metric[(metric.weighting=='design')&(metric.scope=='full144')&(metric.group=='all')]
    print(full[['model','rmse','mae','cells']].to_string(index=False));print(pd.DataFrame(comparisons).to_string(index=False))

if __name__=='__main__':main()
