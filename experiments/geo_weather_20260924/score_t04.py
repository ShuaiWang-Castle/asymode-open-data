"""T04 paired OOF scoring; all arms/seeds/scopes retained, no outer tuning."""
from t04_simple_geography import *
from score_t03 import summarize

MODELS=('HOST','DOSE','GCRK','GEO_MLP')+ARMS

def main():
    d,meta=load_data();n=len(d['y']);coverage=np.zeros(n,int)
    pred={a:np.zeros((n,144),float) for a in MODELS};seedpred={};training=[];folds=[]
    for arm in MODELS:
        for seed in range(3):seedpred[arm,seed]=np.zeros((n,144),float)
    for fold in range(5):
        tr,va,te=indices(d,fold);coverage[te]+=1
        for arm in MODELS:
            root=PRIVATE if arm in ARMS else T03_PRIVATE
            for seed in range(3):
                z=np.load(root/f'{arm}_f{fold}_s{seed}.npz')
                assert np.array_equal(z['va'],va) and np.array_equal(z['te'],te)
                seedpred[arm,seed][te]=z['tp']
                if arm in ARMS:
                    r=json.loads((root/f'{arm}_f{fold}_s{seed}.json').read_text())
                    assert r['source_sha256']==digest(HERE/'t04_simple_geography.py')
                    assert r['cache_sha256']==meta['cache_sha256']
                    assert abs(mse(z['vp'],d,va)-r['validation_mse'])<1e-10
                    training.append(r)
            pred[arm][te]=np.mean([seedpred[arm,s][te] for s in range(3)],axis=0)
            folds.append(dict(fold=fold,model=arm,**summarize(pred[arm][te],d['y'][te],d['m'][te],d['w'][te])))
    assert np.all(coverage==1)
    for p in pred.values():assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    scope=json.loads((OUT/'scope_registration.json').read_text())
    quality=np.isin(d['system'],scope['candidate_events'])
    original=np.load(DATA/'outcomes/panel_v1D_outcomes.npz',allow_pickle=False)
    complete=original['observed'][original['regime']=='tropical'].all(1)
    groups=dict(all=np.ones(n,bool),quality_candidate=quality,complete_observation=complete,
                quality_complete=quality&complete)
    metrics=[];per_seed=[];per_event=[];comparisons=[]
    for arm in MODELS:
        for group,keep in groups.items():
            for weighting,w in [('design',d['w']),('unweighted',np.ones(n))]:
                for horizon,cols in [('full144',np.arange(144))]+[(f'exact+{h}',np.array([h-1])) for h in (1,6,24,48)]:
                    metrics.append(dict(model=arm,group=group,weighting=weighting,scope=horizon,
                        **summarize(pred[arm][keep][:,cols],d['y'][keep][:,cols],d['m'][keep][:,cols],w[keep])))
            for seed in range(3):
                per_seed.append(dict(model=arm,seed=seed,group=group,
                    **summarize(seedpred[arm,seed][keep],d['y'][keep],d['m'][keep],d['w'][keep])))
        for event in sorted(set(d['system'])):
            keep=d['system']==event
            per_event.append(dict(model=arm,system=event,**summarize(pred[arm][keep],d['y'][keep],d['m'][keep],d['w'][keep])))
    pairs=[(a,'SHARED') for a in ('GEO_GAIN','GEO_MIX')]+[(a,'DOSE') for a in ARMS]+[(a,'HOST') for a in ARMS]
    for group,keep in groups.items():
        counties,ix=np.unique(d['fips'][keep],return_inverse=True)
        draws=np.random.default_rng(20261004).multinomial(len(counties),np.ones(len(counties))/len(counties),size=2000)
        sums={a:np.bincount(ix,weights=np.sum((pred[a][keep]-d['y'][keep])**2*d['m'][keep]*d['w'][keep,None],axis=1),minlength=len(counties)) for a in MODELS}
        for arm,ref in pairs:
            delta=100*(1-np.sqrt((draws@sums[arm])/(draws@sums[ref])))
            lo,hi=np.quantile(delta,[.025,.975])
            comparisons.append(dict(group=group,model=arm,reference=ref,rmse_gain_pct=100*(1-np.sqrt(sums[arm].sum()/sums[ref].sum())),ci025_pct=lo,ci975_pct=hi))
    metrics=pd.DataFrame(metrics);folds=pd.DataFrame(folds)
    errors=[]
    for arm in MODELS:
        row=metrics.query("model==@arm and group=='all' and weighting=='design' and scope=='full144'").iloc[0]
        f=folds[folds.model==arm]
        errors.append(max(abs(row.rmse-np.sqrt(f.sse.sum()/f.weight_sum.sum())),abs(row.mae-f.sae.sum()/f.weight_sum.sum())))
    assert max(errors)<1e-12
    replays={}
    for arm in ARMS:
        model,b,(tr,va,te),stats=prepare(d,meta,0,0,arm)
        saved=torch.load(PRIVATE/f'{arm}_f0_s0.pt',weights_only=False,map_location='cpu')
        model.load_state_dict(saved['state']);rep=predict(model,b,te)
        target=np.load(PRIVATE/f'{arm}_f0_s0.npz')['tp']
        err=float(np.max(abs(rep-target)));assert err==0;replays[arm]=err
    # Preserve full curves and parameter-change magnitudes; no county predictions/checkpoints in git.
    dump(OUT/'training_audit.json',training)
    for name,value in [('metrics',metrics),('per_seed',per_seed),('per_event',per_event),('comparisons',comparisons),('fold_scores',folds)]:
        pd.DataFrame(value).to_csv(OUT/(name+'.csv'),index=False)
    np.savez_compressed(PRIVATE/'oof_ensembles.npz',**pred)
    dump(OUT/'verification.json',dict(neural_fits=len(training),oof_exact_once=True,
        pooled_reconstruction_max_error=max(errors),checkpoint_replay_max_error=replays,
        paired_initial_max_error=max(t['paired_initial_max_error'] for t in training),
        groups={k:dict(county_events=int(v.sum()),counties=len(set(d['fips'][v])),events=len(set(d['system'][v]))) for k,v in groups.items()},
        seeds=[0,1,2],steps=600,cache_sha256=meta['cache_sha256'],
        source_sha256={p.name:digest(p) for p in [HERE/'t04_simple_geography.py',Path(__file__),HERE/'run_t04.py']},
        note='Development conditional hindcast; quality subset was fixed before fitting; no retraining on subset; descriptive county-cluster intervals conditional on fitted models'))
    print(metrics.query("group=='all' and weighting=='design' and scope=='full144'")[['model','mae','rmse']].to_string(index=False))
    print(pd.DataFrame(comparisons).query("group=='all'").to_string(index=False))

if __name__=='__main__':main()
