"""Rebuild a weather-selected multiregime cohort and fit seven paired AsymODE arms."""
from t03_poster_trajectory import *
from t04_simple_geography import SimpleMemory,GEO

PRIVATE=ROOT.parent/'t05_artifacts'
OUT=HERE/'results/t05'
ARMS=('HOST','BASE_GCRK','DOSE','DOSE_GCRK','SHARED','GEO_GAIN','GEO_MIX')

def build():
    select=json.loads((OUT/'selection.json').read_text());z=np.load(DATA/'outcomes/panel_v1D_outcomes.npz',allow_pickle=False)
    ix=np.flatnonzero(np.isin(z['system'],select['events']));ids=z['fips'][ix];systems=z['system'][ix]
    yy=z['y'][ix].astype(np.float32);mask=z['observed'][ix];assert mask[:,71].all()
    history=np.column_stack([yy[:,71],np.nanmean(np.where(mask[:,:72],yy[:,:72],np.nan),axis=1),
        np.nanmax(np.where(mask[:,:72],yy[:,:72],np.nan),axis=1),mask[:,:72].mean(1)])
    prior=json.loads((HERE/'results/t03/build_audit.json').read_text())
    cols=prior['geography_columns'];p1=DATA/'geography/county_geography_v1.parquet';p2=DATA/'geography/county_geography_ext_v1.parquet'
    geo=pd.read_parquet(p1).drop(columns=['n_land_pixels']);ext=pd.read_parquet(p2)
    geo=geo.join(ext[[c for c in cols if c not in geo]])[cols];geo.index=geo.index.astype(str).str.zfill(5)
    geography=geo.reindex(ids).to_numpy(np.float32);streams=[];names=[]
    for src in ('era5','hrrr'):
        data=None
        for event in np.unique(systems):
            e=np.load(DATA/f'weather/hazard_v2/{src}/{src}_{event}.npz',allow_pickle=False)
            chosen=[j for j,n in enumerate(e['names']) if '*' not in str(n)]
            ns=[f'{src}:{e["names"][j]}' for j in chosen]
            if data is None:data=np.full((len(ix),216,len(chosen)),np.nan,np.float32);source_names=ns
            assert ns==source_names
            ri=np.flatnonzero(systems==event);lookup={str(f):j for j,f in enumerate(e['fips'])}
            a=e['X'][[lookup[f] for f in ids[ri]]][:,:,chosen].astype(np.float32)
            start=pd.Timestamp(str(z['window_start_utc'][ix[ri[0]]]))
            for stamp in e['missing_hours']:
                h=int((pd.Timestamp(str(stamp))-start).total_seconds()/3600)
                if 0<=h<216:a[:,h]=np.nan
            data[ri]=a
        streams.extend([data,np.isfinite(data).all(2)[:,:,None].astype(np.float32)]);names.extend(source_names+[src+':available'])
    wx=np.concatenate(streams,axis=2);base=np.concatenate([wx,np.repeat(history[:,None,:],216,axis=1)],axis=2).astype(np.float32)
    doses=[];dose_names=[]
    for src in ('era5','hrrr'):
        for channel in ('g_mean','g_exc_mean','rain_mean','wet_wind_mean'):
            a=wx[:,:,names.index(src+':'+channel)];ok=np.isfinite(a)
            s=np.cumsum(np.where(ok,a,0),axis=1);n=np.cumsum(ok,axis=1)
            for span in (6,24,72):
                total=s.copy();count=n.copy();total[:,span:]-=s[:,:-span];count[:,span:]-=n[:,:-span]
                doses.append(np.divide(total,count,out=np.full_like(total,np.nan),where=count>0));dose_names.append(f'{src}:{channel}:mean{span}')
            maximum=np.maximum.accumulate(np.where(ok,a,-np.inf),axis=1);maximum[~np.isfinite(maximum)]=np.nan
            doses.append(maximum);dose_names.append(f'{src}:{channel}:max_since_start')
            for decay in (.9,.98):
                state=np.zeros(len(a),np.float32);mass=np.zeros(len(a),np.float32);v=np.empty_like(a)
                for t in range(216):
                    state=decay*state+np.where(ok[:,t],a[:,t],0);mass=decay*mass+ok[:,t]
                    v[:,t]=np.divide(state,mass,out=np.full_like(state,np.nan),where=mass>0)
                doses.append(v);dose_names.append(f'{src}:{channel}:ewma{decay}')
    full=np.concatenate([base,np.stack(doses,axis=2)],axis=2).astype(np.float32)
    details={r['index']:r for r in json.loads((PRIVATE/'county_weather_groups.json').read_text())}
    groups={k:np.array([details[int(i)][k] for i in ix]) for k in ('repeated','cross_cutoff','peak1','peak2')}
    starts=np.array([np.datetime64(str(s),'h').astype('int64') for s in z['window_start_utc'][ix]])
    occ=[names.index(f'{src}:{c}') for src in ('era5','hrrr') for c in ('g_mean','g_exc_mean','rain_mean','wet_wind_mean')]
    np.savez_compressed(PRIVATE/'data.npz',base=base,full=full,geo=geography,y=np.nan_to_num(yy[:,72:]),m=mask[:,72:],y0=yy[:,71],
        w=z['w'][ix],fips=ids,system=systems,outer=np.array([county_fold(f,20261003) for f in ids]),
        inner=np.array([county_fold(f,20261004) for f in ids]),occ=np.array(occ),starts=starts,original_index=ix,**groups)
    splits={}
    d=dict(np.load(PRIVATE/'data.npz',allow_pickle=False))
    for f in range(5):
        parts=indices(d,f);splits[str(f)]={key:sorted(set(ids[i])) for key,i in zip(('fit','validation','outer'),parts)}
    dump(OUT/'county_splits.json',splits)
    dump(OUT/'build_audit.json',dict(n=len(ix),counties=len(set(ids)),events=len(set(systems)),base_names=names+['hist_last','hist_mean','hist_max','hist_coverage'],dose_names=dose_names,
        geography_columns=cols,cache_sha256=digest(PRIVATE/'data.npz'),selection_sha256=digest(OUT/'selection.json'),source_sha256=digest(__file__),
        geography_sha256={str(p.relative_to(HERE)):digest(p) for p in (p1,p2)}))
    # Outcome statistics are descriptive AFTER weather-only selection, never a new selection criterion.
    rows=[]
    for event in sorted(set(systems)):
        take=np.flatnonzero(systems==event);rep=[j for j in take if groups['repeated'][j]];carry=0;renewed=0;valid=0
        for j in rep:
            b=int(groups['peak2'][j]);lo=max(0,b-6);hi=min(216,b+25)
            before=yy[j,lo:b][mask[j,lo:b]];after=yy[j,b:hi][mask[j,b:hi]]
            if len(before) and len(after):
                valid+=1;carry+=float(np.mean(before))>=.001
                renewed+=float(np.max(after)-np.mean(before))>=.01
        rows.append(dict(system=event,n=len(take),repeated_count=len(rep),cross_cutoff_count=int(groups['cross_cutoff'][take].sum()),
            descriptive_valid_pairs=valid,outage_before_second_ge_0p1pct=carry,increase_after_second_ge_1pp=renewed,
            mean_peak1=float(np.mean(groups['peak1'][rep])),mean_peak2=float(np.mean(groups['peak2'][rep]))))
    pd.DataFrame(rows).to_csv(OUT/'selected_outage_description.csv',index=False)
    print('BUILT',len(ix),'county-events',len(set(ids)),'counties',len(set(systems)),'windows',flush=True)

def load_data():
    meta=json.loads((OUT/'build_audit.json').read_text());assert digest(PRIVATE/'data.npz')==meta['cache_sha256']
    return dict(np.load(PRIVATE/'data.npz',allow_pickle=False)),meta

def prepare(d,meta,fold,seed,arm):
    tr,va,te=indices(d,fold);base,bs=standardize(d['base'],tr);full,fs=standardize(d['full'],tr);geo,gs=standardize(d['geo'],tr)
    use_base=arm in ('HOST','BASE_GCRK')
    b=dict(xu=torch.from_numpy(base if use_base else full),xr=torch.from_numpy(base),xo=torch.from_numpy(base[:,:,d['occ']]),
           geo=torch.from_numpy(geo),y0=torch.from_numpy(d['y0']))
    torch.manual_seed(seed);host=AsymODE(base.shape[-1],base.shape[-1],len(d['occ']))
    reference=predict(host,{**b,'xu':torch.from_numpy(base)},tr[:8])
    if not use_base:host.expand_damage_inputs(full.shape[-1]-base.shape[-1])
    if arm in ('BASE_GCRK','DOSE_GCRK'):
        host.attach_gcrk(torch.tanh(b['geo'][tr]/3).mean(0),private_seed=1729+seed)
        with torch.no_grad():host.kernel.calibrate_(host.hidden(b['xu'][tr]),0)
    model=host
    if arm in ('SHARED','GEO_GAIN','GEO_MIX'):
        g4,g4s=standardize(d['geo'][:,[meta['geography_columns'].index(c) for c in GEO]],tr)
        b['g4']=torch.from_numpy(g4);model=SimpleMemory(host,arm,meta,fs)
        raw=d['full'][:,:,model.pairs.numpy()]
        raw=np.where(np.isfinite(raw),raw,np.array(fs['mean'],np.float32)[model.pairs.numpy()])
        b['raw_memory']=torch.from_numpy(raw.astype(np.float32));gs['geo4']=g4s
    paired=float(np.max(abs(reference-predict(model,b,tr[:8]))));assert paired<1e-6
    return model,host,b,(tr,va,te),dict(base=bs,full=fs,geo=gs),paired

def train(fold,seed,arm):
    path=PRIVATE/f'{arm}_f{fold}_s{seed}.npz'
    if path.exists():print('skip',path.name,flush=True);return
    d,meta=load_data();model,host,b,(tr,va,te),stats,paired=prepare(d,meta,fold,seed,arm)
    initial={n:p.detach().clone() for n,p in model.named_parameters()}
    hp,rp=host.parameter_groups();extra=[]
    if model is not host:extra=[p for n,p in model.named_parameters() if not n.startswith('host.')]
    opt=torch.optim.Adam([dict(params=hp+extra,lr=.003),dict(params=rp,lr=.0003)])
    y=torch.from_numpy(d['y']);wm=torch.from_numpy((d['m']*d['w'][tr].mean()**-1*d['w'][:,None]).astype(np.float32))
    rng=np.random.default_rng(10000+seed);best=mse(predict(model,b,va),d,va);state=copy.deepcopy(model.state_dict());best_step=0;curve=[];start=time.time()
    for step in range(1,601):
        if host.kernel is not None:
            host.kernel.training_step.fill_(step)
            if step==1 or step%10==0:
                with torch.no_grad():host.kernel.calibrate_(host.hidden(b['xu'][tr]),step)
        sel=rng.choice(tr,size=min(64,len(tr)),replace=False);model.train();opt.zero_grad(set_to_none=True)
        p=model(batches(b,sel))['P'];data_loss=((p-y[sel])**2*wm[sel]).sum()/wm[sel].sum()
        penalty=model.penalty() if model is not host else data_loss.new_zeros(())
        loss=data_loss+penalty;assert torch.isfinite(loss)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if step%50==0:
            v=mse(predict(model,b,va),d,va);curve.append(dict(step=step,train_mse=float(data_loss.detach()),penalty=float(penalty.detach()),validation_mse=v))
            if v<best:best=v;state=copy.deepcopy(model.state_dict());best_step=step
        if step%200==0:print(arm,fold,seed,step,round(time.time()-start),flush=True)
    model.load_state_dict(state);vp=predict(model,b,va);tp=predict(model,b,te)
    assert np.isfinite(tp).all() and ((tp>=0)&(tp<=1)).all()
    torch.save(dict(state=state,stats=stats,arm=arm,fold=fold,seed=seed),PRIVATE/f'{arm}_f{fold}_s{seed}.pt')
    dump(PRIVATE/f'{arm}_f{fold}_s{seed}.json',dict(arm=arm,fold=fold,seed=seed,seconds=time.time()-start,best_step=best_step,validation_mse=best,
        paired_initial_max_error=paired,curve=curve,parameter_changes={n:float((p.detach()-initial[n]).norm()) for n,p in model.named_parameters()},
        source_sha256=digest(__file__),cache_sha256=meta['cache_sha256']))
    np.savez_compressed(path,va=va,te=te,vp=vp,tp=tp);print('DONE',arm,fold,seed,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['build','train']);p.add_argument('--arm',choices=ARMS);p.add_argument('--fold',type=int);p.add_argument('--seed',type=int);a=p.parse_args()
    if a.command=='build':build()
    else:train(a.fold,a.seed,a.arm)
