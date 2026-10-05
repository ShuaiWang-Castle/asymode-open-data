"""Score all registered repeated-weather arms with county-cluster intervals."""
from t05_repeated_weather import *
from score_t03 import summarize

def main():
    d,meta=load_data();n=len(d['y']);coverage=np.zeros(n,int)
    pred={a:np.zeros((n,144),float) for a in ARMS};seedpred={(a,s):np.zeros((n,144),float) for a in ARMS for s in range(3)}
    training=[];folds=[]
    for fold in range(5):
        tr,va,te=indices(d,fold);coverage[te]+=1
        for arm in ARMS:
            for seed in range(3):
                z=np.load(PRIVATE/f'{arm}_f{fold}_s{seed}.npz');assert np.array_equal(z['te'],te) and np.array_equal(z['va'],va)
                seedpred[arm,seed][te]=z['tp']
                r=json.loads((PRIVATE/f'{arm}_f{fold}_s{seed}.json').read_text())
                assert r['source_sha256']==digest(HERE/'t05_repeated_weather.py') and r['cache_sha256']==meta['cache_sha256']
                assert abs(mse(z['vp'],d,va)-r['validation_mse'])<1e-10
                training.append(r)
            pred[arm][te]=np.mean([seedpred[arm,s][te] for s in range(3)],axis=0)
            folds.append(dict(model=arm,fold=fold,**summarize(pred[arm][te],d['y'][te],d['m'][te],d['w'][te])))
    assert (coverage==1).all()
    groups=dict(all=np.ones(n,bool),repeated=d['repeated'],cross_cutoff=d['cross_cutoff'],nonrepeated=~d['repeated'])
    metrics=[];perseed=[];events=[];comparison=[]
    pairs=[('BASE_GCRK','HOST'),('DOSE_GCRK','DOSE'),('GEO_GAIN','SHARED'),('GEO_MIX','SHARED'),('DOSE','HOST'),('SHARED','DOSE')]
    for arm in ARMS:
        assert np.isfinite(pred[arm]).all() and ((pred[arm]>=0)&(pred[arm]<=1)).all()
        for group,keep in groups.items():
            for weighting,w in [('design',d['w']),('unweighted',np.ones(n))]:
                for horizon,cols in [('full144',np.arange(144))]+[(f'exact+{h}',np.array([h-1])) for h in (1,6,24,48)]:
                    metrics.append(dict(model=arm,group=group,weighting=weighting,scope=horizon,
                        **summarize(pred[arm][keep][:,cols],d['y'][keep][:,cols],d['m'][keep][:,cols],w[keep])))
            for seed in range(3):
                perseed.append(dict(model=arm,group=group,seed=seed,
                    **summarize(seedpred[arm,seed][keep],d['y'][keep],d['m'][keep],d['w'][keep])))
        for event in sorted(set(d['system'])):
            keep=d['system']==event;events.append(dict(model=arm,system=event,**summarize(pred[arm][keep],d['y'][keep],d['m'][keep],d['w'][keep])))
    for group,keep in groups.items():
        ids,ix=np.unique(d['fips'][keep],return_inverse=True)
        draws=np.random.default_rng(20261004).multinomial(len(ids),np.ones(len(ids))/len(ids),size=2000)
        sums={a:np.bincount(ix,weights=np.sum((pred[a][keep]-d['y'][keep])**2*d['m'][keep]*d['w'][keep,None],axis=1),minlength=len(ids)) for a in ARMS}
        for arm,ref in pairs:
            delta=100*(1-np.sqrt((draws@sums[arm])/(draws@sums[ref])));lo,hi=np.quantile(delta,[.025,.975])
            comparison.append(dict(group=group,model=arm,reference=ref,rmse_gain_pct=100*(1-np.sqrt(sums[arm].sum()/sums[ref].sum())),ci025_pct=lo,ci975_pct=hi))
    mf=pd.DataFrame(metrics);ff=pd.DataFrame(folds);errors=[]
    for arm in ARMS:
        row=mf.query("group=='all' and weighting=='design' and scope=='full144' and model==@arm").iloc[0];f=ff[ff.model==arm]
        errors.append(max(abs(row.rmse-np.sqrt(f.sse.sum()/f.weight_sum.sum())),abs(row.mae-f.sae.sum()/f.weight_sum.sum())))
    assert max(errors)<1e-12
    replay={}
    for arm in ARMS:
        model,host,b,(tr,va,te),stats,paired=prepare(d,meta,0,0,arm)
        state=torch.load(PRIVATE/f'{arm}_f0_s0.pt',map_location='cpu',weights_only=False)['state'];model.load_state_dict(state)
        err=float(np.max(abs(predict(model,b,te)-np.load(PRIVATE/f'{arm}_f0_s0.npz')['tp'])));assert err==0;replay[arm]=err
    # Overlapping event windows can repeat a county-hour. Report equal unique-cell sensitivity, not independent events.
    ids,ci=np.unique(d['fips'],return_inverse=True)
    hours=d['starts'][:,None]+np.arange(72,216)[None,:];keys=ci[:,None]*10000000+hours
    valid=d['m'];unique,group=np.unique(keys[valid],return_inverse=True);count=np.bincount(group)
    target=np.bincount(group,weights=d['y'][valid])/count
    target2=np.bincount(group,weights=d['y'][valid].astype(float)**2)/count
    assert np.max(abs(target2-target**2))<1e-7
    sensitivity=[]
    for arm in ARMS:
        error=pred[arm][valid]-d['y'][valid];factor=1/count[group]
        sensitivity.append(dict(model=arm,mae=float(np.sum(factor*abs(error))/len(unique)),rmse=float(np.sqrt(np.sum(factor*error**2)/len(unique))),unique_county_hours=len(unique)))
    for name,rows in [('metrics',metrics),('comparisons',comparison),('per_seed',perseed),('per_event',events),('fold_scores',folds),('unique_county_hour_sensitivity',sensitivity)]:pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False)
    dump(OUT/'training_audit.json',training)
    np.savez_compressed(PRIVATE/'oof_ensembles.npz',**pred)
    dump(OUT/'verification.json',dict(neural_fits=len(training),oof_exact_once=True,pooled_reconstruction_max_error=max(errors),replay_max_error=replay,
        paired_initial_max_error=max(r['paired_initial_max_error'] for r in training),
        valid_event_county_hours=int(valid.sum()),unique_county_hours=len(unique),duplicate_extra_county_hours=int(valid.sum()-len(unique)),
        groups={k:dict(county_events=int(v.sum()),counties=len(set(d['fips'][v]))) for k,v in groups.items()},
        seeds=[0,1,2],steps=600,cache_sha256=meta['cache_sha256'],
        source_sha256={p.name:digest(p) for p in [HERE/'t05_repeated_weather.py',HERE/'t05_screen.py',Path(__file__),HERE/'run_t05.py',HERE/'t04_simple_geography.py']},
        scope='Weather-selected examined development cohort; supplied weather; repeated peaks do not establish distinct storms; unique-cell sensitivity divides each error weight by its county-hour multiplicity; no prediction blending'))
    print(mf.query("group=='all' and weighting=='design' and scope=='full144'")[['model','mae','rmse']].to_string(index=False))
    print(pd.DataFrame(comparison).query("group=='all'").to_string(index=False))

if __name__=='__main__':main()
