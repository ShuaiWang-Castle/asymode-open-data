"""Score every registered T06 arm; preserve negative results and geographic controls."""
from t06_featured_subset import *
from score_t03 import summarize

PAIRS=[('GCRK','DOSE'),('CRK','DOSE'),('CRK','GCRK'),('CRK','CRK_SHUFFLE'),('CRK','GEO_MLP'),('GEO_MLP','DOSE')]

def main():
    d,meta=load_data();n=len(d['y']);covered=np.zeros(n,int)
    seedpred={(a,s):np.zeros((n,144),float) for a in ARMS for s in range(3)}
    training=[];folds=[]
    for f in range(5):
        tr,va,te=indices(d,f);covered[te]+=1
        for a in ARMS:
            for s in range(3):
                path=PRIVATE/f'{a}_f{f}_s{s}.npz';z=np.load(path)
                assert np.array_equal(z['te'],te) and np.array_equal(z['va'],va)
                assert np.isfinite(z['tp']).all() and ((z['tp']>=0)&(z['tp']<=1)).all()
                r=json.loads(path.with_suffix('.json').read_text())
                assert r['source_sha256']==sources() and r['cache_sha256']==meta['cache_sha256']
                assert abs(mse(z['vp'],d,va)-r['validation_mse'])<1e-10
                seedpred[a,s][te]=z['tp'];training.append(r)
            p=np.mean([seedpred[a,s][te] for s in range(3)],axis=0)
            folds.append(dict(model=a,fold=f,**summarize(p,d['y'][te],d['m'][te],d['w'][te])))
    assert (covered==1).all()
    pred={a:np.mean([seedpred[a,s] for s in range(3)],axis=0) for a in ARMS}
    observed_y=np.where(d['m'],d['y'],np.nan);peak=np.nanmax(observed_y,axis=1)
    jump=np.nanmax(np.where(d['m'][:,1:]&d['m'][:,:-1],np.diff(d['y'],axis=1),np.nan),axis=1)
    groups=dict(all=np.ones(n,bool),severe=peak>=.1,jump=jump>=.01,zero=peak==0,nonsevere=peak<.1)
    metrics=[];perseed=[];events=[];comparisons=[];tails=[];leave=[]
    for a in ARMS:
        for g,k in groups.items():
            for weighting,w in [('design',d['w']),('unweighted',np.ones(n))]:
                for scope,cols in [('full144',np.arange(144))]+[(f'exact+{h}',[h-1]) for h in (1,6,24,48)]:
                    if k.any():metrics.append(dict(model=a,group=g,weighting=weighting,scope=scope,
                        **summarize(pred[a][k][:,cols],d['y'][k][:,cols],d['m'][k][:,cols],w[k])))
            for s in range(3):
                if k.any():perseed.append(dict(model=a,group=g,seed=s,
                    **summarize(seedpred[a,s][k],d['y'][k],d['m'][k],d['w'][k])))
        for event in sorted(set(d['system'])):
            k=d['system']==event;events.append(dict(model=a,system=event,**summarize(pred[a][k],d['y'][k],d['m'][k],d['w'][k])))
        pp=np.nanmax(np.where(d['m'],pred[a],np.nan),axis=1)
        pkidx=np.nanargmax(observed_y,axis=1);atpeak=pred[a][np.arange(n),pkidx]
        for g,k in groups.items():
            if not k.any():continue
            wt=d['w'][k]
            r=dict(model=a,group=g,county_events=int(k.sum()),counties=len(set(d['fips'][k])),
                   peak_mae=float(np.average(abs(pp[k]-peak[k]),weights=wt)),
                   true_peak_time_mae=float(np.average(abs(atpeak[k]-peak[k]),weights=wt)),
                   peak_timing_mae_hours=float(np.average(abs(np.nanargmax(np.where(d['m'],pred[a],np.nan),axis=1)[k]-pkidx[k]),weights=wt)))
            if g=='severe':r['full_window_peak_ratio_median']=float(np.median(pp[k]/peak[k]))
            if g=='zero':r.update(false_peak_count=int((pp[k]>.001).sum()),false_peak_weighted_rate=float(np.average(pp[k]>.001,weights=wt)))
            if g=='nonsevere':r.update(false_peak_count=int((pp[k]>=.1).sum()),false_peak_weighted_rate=float(np.average(pp[k]>=.1,weights=wt)))
            tails.append(r)
    for g,k in groups.items():
        if not k.any():continue
        ids,ci=np.unique(d['fips'][k],return_inverse=True)
        draws=np.random.default_rng(20261004).multinomial(len(ids),np.ones(len(ids))/len(ids),size=2000)
        sums={a:np.bincount(ci,weights=np.sum((pred[a][k]-d['y'][k])**2*d['m'][k]*d['w'][k,None],axis=1),minlength=len(ids)) for a in ARMS}
        for a,b in PAIRS:
            delta=100*(1-np.sqrt((draws@sums[a])/(draws@sums[b])))
            assert np.isfinite(delta).all();lo,hi=np.quantile(delta,[.025,.975])
            comparisons.append(dict(group=g,model=a,reference=b,rmse_gain_pct=100*(1-np.sqrt(sums[a].sum()/sums[b].sum())),ci025_pct=lo,ci975_pct=hi))
    for event in sorted(set(d['system'])):
        k=d['system']!=event
        for a,b in PAIRS:
            ma=summarize(pred[a][k],d['y'][k],d['m'][k],d['w'][k]);mb=summarize(pred[b][k],d['y'][k],d['m'][k],d['w'][k])
            leave.append(dict(excluded_system=event,model=a,reference=b,rmse_gain_pct=100*(1-ma['rmse']/mb['rmse'])))
    _,ci=np.unique(d['fips'],return_inverse=True)
    keys=ci[:,None]*10000000+d['starts'][:,None]+np.arange(72,216)[None,:]
    valid=d['m'];unique,ii=np.unique(keys[valid],return_inverse=True);count=np.bincount(ii)
    factor=np.zeros(valid.shape);factor[valid]=1/count[ii]
    sensitivity=[]
    for a in ARMS:
        for weighting,w in [('unique_equal',factor),('design_divide_multiplicity',factor*d['w'][:,None])]:
            error=pred[a]-d['y'];sensitivity.append(dict(model=a,weighting=weighting,
                rmse=float(np.sqrt(np.sum(w*error**2)/w.sum())),mae=float(np.sum(w*abs(error))/w.sum()),unique_county_hours=len(unique)))
    replay={};diagnostics={}
    for a in ARMS:
        model,b,(tr,va,te),stats,paired=prepare(d,meta,0,0,a)
        checkpoint=torch.load(PRIVATE/f'{a}_f0_s0.pt',map_location='cpu',weights_only=False);model.load_state_dict(checkpoint['state'])
        expected=np.load(PRIVATE/f'{a}_f0_s0.npz')['tp'];err=float(np.max(abs(predict(model,b,te)-expected)));assert err==0;replay[a]=err
        ix=te[:8];model.eval()
        with torch.no_grad():ans=model(batches(b,ix),diagnostics=True)
        kd={}
        for name in ('kernel_state','kernel_tau','kernel_effect','kernel_response','kernel_order_masses','kernel_geography_features'):
            if name in ans:
                value=ans[name];assert torch.isfinite(value).all()
                kd[name]=dict(min=float(value.min()),max=float(value.max()),rms=float(value.square().mean().sqrt()))
        if model.kernel is not None:kd['alpha']=float(model.kernel.alpha.detach())
        diagnostics[a]=dict(outer_sample='first eight sorted source rows in fold0, seed0; descriptive only',**kd)
    mf=pd.DataFrame(metrics);ff=pd.DataFrame(folds);errors=[]
    for a in ARMS:
        r=mf.query("model==@a and group=='all' and weighting=='design' and scope=='full144'").iloc[0];f=ff[ff.model==a]
        errors.append(max(abs(r.rmse-np.sqrt(f.sse.sum()/f.weight_sum.sum())),abs(r.mae-f.sae.sum()/f.weight_sum.sum())))
    assert max(errors)<1e-12
    for name,rows in [('metrics',metrics),('comparisons',comparisons),('per_seed',perseed),('per_event',events),('fold_scores',folds),
                      ('tail_diagnostics',tails),('leave_one_window',leave),('unique_county_hour_sensitivity',sensitivity)]:
        pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False)
    dump(OUT/'training_audit.json',training);dump(OUT/'kernel_diagnostics.json',diagnostics)
    np.savez_compressed(PRIVATE/'oof_ensembles.npz',**pred)
    dump(OUT/'verification.json',dict(neural_fits=len(training),seeds=[0,1,2],steps=600,oof_exact_once=True,
        replay_max_error=replay,pooled_reconstruction_max_error=max(errors),paired_initial_max_error=max(r['paired_initial_max_error'] for r in training),
        valid_event_county_hours=int(valid.sum()),unique_county_hours=len(unique),duplicate_extra_county_hours=int(valid.sum()-len(unique)),
        groups={g:dict(county_events=int(k.sum()),counties=len(set(d['fips'][k]))) for g,k in groups.items()},
        source_sha256=sources(),score_source_sha256=digest(__file__),cache_sha256=meta['cache_sha256'],environment=meta['environment']))
    print(mf.query("group=='all' and weighting=='design' and scope=='full144'")[['model','mae','rmse']].to_string(index=False))
    print(pd.DataFrame(comparisons).query("group=='all'").to_string(index=False))

if __name__=='__main__':main()
