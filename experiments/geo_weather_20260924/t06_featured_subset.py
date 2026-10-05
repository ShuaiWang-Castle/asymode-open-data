"""Finite strong-wind/heterogeneous-terrain subset, with full CRK and paired controls."""
from t03_poster_trajectory import *
from asymode.controlled_relaxation import ControlledRelaxationLayer
import platform

PRIVATE = ROOT.parent/'t06_artifacts'
OUT = HERE/'results/t06'
ARMS = ('DOSE','GCRK','CRK','CRK_SHUFFLE','GEO_MLP')

def sources():
    files=[Path(__file__),HERE/'t03_poster_trajectory.py',ROOT/'src/asymode/asym_host.py',
           ROOT/'src/asymode/gcrk.py',ROOT/'src/asymode/controlled_relaxation.py',
           ROOT/'src/asymode/controlled_relaxation_scan.py']
    return {str(p.relative_to(ROOT)):digest(p) for p in files}

def build():
    PRIVATE.mkdir(exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    assert not (PRIVATE/'data.npz').exists(),'Existing cache must not be overwritten'
    scope=HERE/'results/t04/dataset_scope_audit.csv';f=pd.read_csv(scope)
    f['selected_t06']=f.quality_candidate & (f.era5_g_mean_peak_p90>=25) & (f.relief_p95_p5_iqr>=80)
    events=sorted(f.loc[f.selected_t06,'system']);assert events==['S00023','S00034','S00046','S00047']
    source=ROOT.parent/'t03_artifacts/data.npz';d=dict(np.load(source,allow_pickle=False))
    ix=np.flatnonzero(np.isin(d['system'],events));n=len(d['y'])
    subset={k:(v[ix] if v.ndim and len(v)==n else v) for k,v in d.items()}
    assert len(ix)==560 and subset['m'].any(1).all()
    # T03 metadata supplies the original source column ordering; cache is immutable.
    audit=json.loads((HERE/'results/t03/build_audit.json').read_text())
    orig=np.load(DATA/'outcomes/panel_v1D_outcomes.npz',allow_pickle=False)
    index={(str(f),str(s)):i for i,(f,s) in enumerate(zip(orig['fips'],orig['system']))}
    oi=np.array([index[str(f),str(s)] for f,s in zip(subset['fips'],subset['system'])])
    assert np.array_equal(subset['m'],orig['observed'][oi,72:])
    assert np.max(abs(subset['y']-np.nan_to_num(orig['y'][oi,72:])))<1e-7
    subset['starts']=np.array([np.datetime64(str(s),'h').astype('int64') for s in orig['window_start_utc'][oi]])
    subset['original_index']=oi
    np.savez_compressed(PRIVATE/'data.npz',**subset)
    f.to_csv(OUT/'dataset_selection.csv',index=False)
    splits={str(f):{k:sorted(set(subset['fips'][a])) for k,a in zip(('fit','validation','outer'),indices(subset,f))} for f in range(5)}
    dump(OUT/'county_splits.json',splits)
    meta=dict(events=events,n=len(ix),counties=len(set(subset['fips'])),cache_sha256=digest(PRIVATE/'data.npz'),
              source_cache_sha256=digest(source),selection_source_sha256=digest(scope),source_sha256=sources(),
              geography_columns=audit['geography_columns'],base_names=audit['base_names'],dose_names=audit['dose_names'],
              environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__),
              scope='Examined D; fixed weather/coverage/geography selection; supplied-weather county-held-out hindcast')
    dump(OUT/'build_audit.json',meta);print('BUILT',meta['n'],meta['counties'],events,flush=True)

def load_data():
    meta=json.loads((OUT/'build_audit.json').read_text())
    assert meta['cache_sha256']==digest(PRIVATE/'data.npz')
    assert meta['source_sha256']==sources(),'Training source changed after cache registration'
    return dict(np.load(PRIVATE/'data.npz',allow_pickle=False)),meta

def shuffle_geo(d,geo,parts):
    out=geo.copy();mapping={};singletons=[]
    rng=np.random.default_rng(20261004)
    for role,ix in zip(('fit','validation','outer'),parts):
        ids=sorted(set(d['fips'][ix]));look={str(f):int(np.flatnonzero(d['fips']==f)[0]) for f in ids}
        for state in sorted(set(f[:2] for f in ids)):
            group=np.array([f for f in ids if f[:2]==state]);order=rng.permutation(group)
            donors=np.roll(order,1)
            if len(group)==1:singletons.append(dict(role=role,fips=str(group[0])))
            for recipient,donor in zip(order,donors):
                out[d['fips']==recipient]=geo[look[str(donor)]];mapping[str(recipient)]=str(donor)
    assert len(mapping)==len(set(d['fips']))
    return out,dict(mapping=mapping,singleton_groups=singletons,changed_counties=sum(k!=v for k,v in mapping.items()))

def prepare(d,meta,fold,seed,arm):
    parts=indices(d,fold);tr,va,te=parts
    base,bs=standardize(d['base'],tr);full,fs=standardize(d['full'],tr);geo,gs=standardize(d['geo'],tr)
    shuffle={}
    if arm=='CRK_SHUFFLE':geo,shuffle=shuffle_geo(d,geo,parts)
    b=dict(xu=torch.from_numpy(full),xr=torch.from_numpy(base),xo=torch.from_numpy(base[:,:,d['occ']]),
           geo=torch.from_numpy(geo),ctx=torch.from_numpy(geo),y0=torch.from_numpy(d['y0']))
    torch.manual_seed(seed);model=AsymODE(base.shape[-1],base.shape[-1],len(d['occ']))
    model.expand_damage_inputs(full.shape[-1]-base.shape[-1]);reference=predict(model,b,tr[:8])
    if arm=='GCRK':model.attach_gcrk(torch.tanh(b['geo'][tr]/3).mean(0),private_seed=1729+seed)
    elif arm in ('CRK','CRK_SHUFFLE'):
        model.damage[2]=ControlledRelaxationLayer(model.damage[2],b['geo'][tr],d['fips'][tr],
                                                 private_seed=1729+seed,checkpoint_steps=0,use_adjoint=True,geo_checkpoint=True)
    elif arm=='GEO_MLP':model.attach_context_input(geo.shape[-1])
    if model.kernel is not None:
        with torch.no_grad():model.kernel.calibrate_(model.hidden(b['xu'][tr]),0)
    paired=float(np.max(abs(reference-predict(model,b,tr[:8]))));assert paired==0
    return model,b,parts,dict(base=bs,full=fs,geo=gs,shuffle=shuffle),paired

def train(fold,seed,arm):
    path=PRIVATE/f'{arm}_f{fold}_s{seed}.npz';receipt=path.with_suffix('.json')
    d,meta=load_data()
    if path.exists():
        r=json.loads(receipt.read_text());assert r['source_sha256']==sources() and r['cache_sha256']==meta['cache_sha256']
        assert path.with_suffix('.pt').exists();print('VERIFIED SKIP',path.name,flush=True);return
    if receipt.exists() or path.with_suffix('.pt').exists():raise RuntimeError('Partial artifact set; preserve and inspect')
    model,b,(tr,va,te),stats,paired=prepare(d,meta,fold,seed,arm)
    initial={n:p.detach().clone() for n,p in model.named_parameters()}
    hp,rp=model.parameter_groups();opt=torch.optim.Adam([dict(params=hp,lr=.003),dict(params=rp,lr=.0003)])
    y=torch.from_numpy(d['y']);wm=torch.from_numpy((d['m']*d['w'][:,None]/d['w'][tr].mean()).astype(np.float32))
    rng=np.random.default_rng(10000+seed);best=mse(predict(model,b,va),d,va)
    state=copy.deepcopy(model.state_dict());best_step=0;curve=[];start=time.time()
    for step in range(1,601):
        if model.kernel is not None:
            model.kernel.training_step.fill_(step)
            if step==1 or step%10==0:
                with torch.no_grad():model.kernel.calibrate_(model.hidden(b['xu'][tr]),step)
        ix=rng.choice(tr,size=min(64,len(tr)),replace=False);model.train();opt.zero_grad(set_to_none=True)
        p=model(batches(b,ix))['P'];loss=((p-y[ix])**2*wm[ix]).sum()/wm[ix].sum();assert torch.isfinite(loss)
        loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(grad);opt.step()
        if step%50==0:
            val=mse(predict(model,b,va),d,va)
            curve.append(dict(step=step,train_mse=float(loss.detach()),validation_mse=val,gradient_norm=float(grad)))
            if val<best:best=val;state=copy.deepcopy(model.state_dict());best_step=step
        if step%200==0:print(arm,fold,seed,step,round(time.time()-start),flush=True)
    model.load_state_dict(state);vp=predict(model,b,va);tp=predict(model,b,te)
    assert np.isfinite(tp).all() and ((tp>=0)&(tp<=1)).all()
    record=dict(arm=arm,fold=fold,seed=seed,seconds=time.time()-start,best_step=best_step,validation_mse=best,
                paired_initial_max_error=paired,curve=curve,parameter_count=sum(p.numel() for p in model.parameters()),
                parameter_changes={n:float((p.detach()-initial[n]).norm()) for n,p in model.named_parameters()},
                source_sha256=sources(),cache_sha256=meta['cache_sha256'],shuffle=stats['shuffle'])
    torch.save(dict(state=state,stats=stats,arm=arm,fold=fold,seed=seed),path.with_suffix('.pt'))
    dump(receipt,record);np.savez_compressed(path,va=va,te=te,vp=vp,tp=tp)
    print('DONE',arm,fold,seed,'best',best_step,flush=True)

def preflight():
    d,meta=load_data();checks={}
    for arm in ARMS:
        model,b,(tr,va,te),stats,paired=prepare(d,meta,0,0,arm)
        ix=tr[:8]
        if model.kernel is not None:
            model.kernel.training_step.fill_(1)
            model.kernel.alpha.data.fill_(.1)
            if hasattr(model.kernel,'set_drop_override'):model.kernel.set_drop_override(1)
        model.train();start=time.time();out=model(batches(b,ix));loss=(out['P']-torch.from_numpy(d['y'][ix])).square().mean();loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        checks[arm]=dict(paired_error=paired,gradient_finite=True,fit_only_batch=8,seconds=time.time()-start,
                        parameter_count=sum(p.numel() for p in model.parameters()))
    dump(OUT/'implementation_checks.json',checks);print(json.dumps(checks,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['build','train','preflight']);p.add_argument('--arm',choices=ARMS)
    p.add_argument('--fold',type=int);p.add_argument('--seed',type=int);a=p.parse_args()
    if a.command=='build':build()
    elif a.command=='preflight':preflight()
    else:train(a.fold,a.seed,a.arm)
