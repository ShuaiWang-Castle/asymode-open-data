"""Finite T03 county-held-out trajectory experiment. Private artifacts outside git.

Run build; neural --fold F --seed S --arm A; tree --fold F; score.
See notes/T03_POSTER_TRAJECTORY_PROTOCOL_20261003.md for the fixed scope.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import sys,json,hashlib,copy,time,argparse,warnings
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'src'))
from asymode.asym_host import AsymODE
torch.set_num_threads(1)
DATA=HERE/'data_v1'
PRIVATE=Path(os.environ.get('T03_ARTIFACTS',str(ROOT.parent/'t03_artifacts')))
OUT=HERE/'results/t03'
ARMS=('HOST','DOSE','GCRK','GEO_MLP')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def county_fold(f,salt,k=5):return int(hashlib.sha256(f'{salt}/{f}'.encode()).hexdigest()[:16],16)%k

def build():
    PRIVATE.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    if (PRIVATE/'data.npz').exists():raise RuntimeError('Existing T03 cache; use its registered inputs')
    zp=DATA/'outcomes/panel_v1D_outcomes.npz';z=np.load(zp,allow_pickle=False)
    ids=np.flatnonzero(z['regime']=='tropical');fips=z['fips'][ids];systems=z['system'][ids]
    yy=z['y'][ids].astype(np.float32);mask=z['observed'][ids]
    assert np.isfinite(yy[mask]).all() and mask[:,71].all()
    hist=np.where(mask[:,:72],yy[:,:72],np.nan)
    history=np.column_stack([yy[:,71],np.nanmean(hist,axis=1),np.nanmax(hist,axis=1),mask[:,:72].mean(1)])
    paths=[zp,DATA/'geography/county_geography_v1.parquet',DATA/'geography/county_geography_ext_v1.parquet']
    geo=pd.read_parquet(paths[1]).drop(columns=['n_land_pixels']);ext=pd.read_parquet(paths[2])
    cols=json.loads((HERE/'results/t01/audit.json').read_text())['geography_columns']
    geo=geo.join(ext[[c for c in cols if c not in geo]])[cols];geo.index=geo.index.astype(str).str.zfill(5)
    geography=geo.reindex(fips).to_numpy(np.float32)
    streams=[];names=[];audit_weather=[]
    for src in ('era5','hrrr'):
        data=None
        for system in np.unique(systems):
            p=DATA/f'weather/hazard_v2/{src}/{src}_{system}.npz';paths.append(p)
            e=np.load(p,allow_pickle=False);chosen=[j for j,n in enumerate(e['names']) if '*' not in str(n)]
            sn=[f'{src}:{e["names"][j]}' for j in chosen]
            if data is None:data=np.full((len(ids),216,len(chosen)),np.nan,np.float32);source_names=sn
            assert sn==source_names
            ri=np.flatnonzero(systems==system);lookup={str(f).zfill(5):j for j,f in enumerate(e['fips'])}
            a=e['X'][[lookup[f] for f in fips[ri]]][:,:,chosen].astype(np.float32)
            start=pd.Timestamp(str(z['window_start_utc'][ids[ri[0]]]));missing=[]
            for stamp in e['missing_hours']:
                offset=int((pd.Timestamp(str(stamp))-start).total_seconds()/3600)
                if 0<=offset<216:missing.append(offset)
            if missing:a[:,missing,:]=np.nan
            data[ri]=a
            audit_weather.append(dict(system=system,source=src,missing_hours=len(set(missing))))
        streams.extend([data,np.isfinite(data).all(2)[:,:,None].astype(np.float32)])
        names.extend(source_names+[src+':available'])
    wx=np.concatenate(streams,axis=2)
    base=np.concatenate([wx,np.repeat(history[:,None,:],216,axis=1)],axis=2).astype(np.float32)
    base_names=names+['hist_last','hist_mean','hist_max','hist_coverage']
    doses=[];dose_names=[]
    for src in ('era5','hrrr'):
        for channel in ('g_mean','g_exc_mean','rain_mean','wet_wind_mean'):
            a=wx[:,:,names.index(src+':'+channel)]
            ok=np.isfinite(a);s=np.cumsum(np.where(ok,a,0),axis=1);n=np.cumsum(ok,axis=1)
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
    outer=np.array([county_fold(f,20261003) for f in fips]);inner=np.array([county_fold(f,20261004) for f in fips])
    occ=[names.index(f'{src}:{c}') for src in ('era5','hrrr') for c in ('g_mean','g_exc_mean','rain_mean','wet_wind_mean')]
    np.savez_compressed(PRIVATE/'data.npz',base=base,full=full,geo=geography,y=np.nan_to_num(yy[:,72:]),m=mask[:,72:],y0=yy[:,71],w=z['w'][ids],fips=fips,system=systems,outer=outer,inner=inner,occ=np.array(occ))
    splits={}
    for fold in range(5):
        tr,va,te=indices(dict(np.load(PRIVATE/'data.npz')),fold)
        splits[str(fold)]={role:sorted(set(fips[ix])) for role,ix in [('fit',tr),('validation',va),('outer',te)]}
    dump(OUT/'county_splits.json',splits)
    dump(OUT/'build_audit.json',dict(n=len(ids),counties=len(set(fips)),events=len(set(systems)),base_names=base_names,dose_names=dose_names,geography_columns=cols,
        inputs={str(p.relative_to(HERE)):digest(p) for p in paths},weather_missing=audit_weather,cache_sha256=digest(PRIVATE/'data.npz'),source_sha256=digest(__file__),
        note='Development conditional hindcast; all 15 tropical events; global county folds; no future outage input'))
    print('built',len(ids),'county-events',base.shape,full.shape,flush=True)

def indices(d,fold):
    te=np.flatnonzero(d['outer']==fold);va=np.flatnonzero((d['outer']!=fold)&(d['inner']==fold));tr=np.flatnonzero((d['outer']!=fold)&(d['inner']!=fold))
    sets=[set(d['fips'][i]) for i in (tr,va,te)]
    assert all(len(i)>0 for i in (tr,va,te)) and not(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
    return tr,va,te

def standardize(x,tr):
    axes=tuple(range(x.ndim-1))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        mu=np.nanmean(x[tr],axis=axes);sd=np.nanstd(x[tr],axis=axes)
    mu=np.nan_to_num(mu);sd=np.where(np.isfinite(sd)&(sd>1e-6),sd,1)
    return np.nan_to_num((x-mu)/sd).clip(-10,10).astype(np.float32),dict(mean=mu.tolist(),std=sd.tolist())

def batches(b,idx):return {k:v[idx] for k,v in b.items()}
@torch.no_grad()
def predict(model,b,idx):
    model.eval();return np.concatenate([model(batches(b,idx[i:i+128]))['P'].numpy() for i in range(0,len(idx),128)])
def mse(p,d,idx):
    weight=d['m'][idx]*d['w'][idx,None]
    return float(np.sum(weight*(p-d['y'][idx])**2)/weight.sum())

def neural(fold,seed,arm):
    path=PRIVATE/f'{arm}_f{fold}_s{seed}.npz'
    if path.exists():print('skip',path.name,flush=True);return
    d=dict(np.load(PRIVATE/'data.npz',allow_pickle=False));tr,va,te=indices(d,fold)
    base,bs=standardize(d['base'],tr);full,fs=standardize(d['full'],tr);geo,gs=standardize(d['geo'],tr)
    b={'xu':torch.from_numpy(base if arm=='HOST' else full),'xr':torch.from_numpy(base),'xo':torch.from_numpy(base[:,:,d['occ']]),'geo':torch.from_numpy(geo),'ctx':torch.from_numpy(geo),'y0':torch.from_numpy(d['y0'])}
    torch.manual_seed(seed);model=AsymODE(base.shape[-1],base.shape[-1],len(d['occ']))
    with torch.no_grad():
        check={**b,'xu':torch.from_numpy(base)}
        reference=predict(model,check,tr[:8])
    if arm!='HOST':model.expand_damage_inputs(full.shape[-1]-base.shape[-1])
    if arm=='GCRK':
        model.attach_gcrk(torch.tanh(b['geo'][tr]/3).mean(0),private_seed=1729+seed)
        with torch.no_grad():model.kernel.calibrate_(model.hidden(b['xu'][tr]),0)
    if arm=='GEO_MLP':model.attach_context_input(geo.shape[-1])
    initial={n:p.detach().clone() for n,p in model.named_parameters()}
    paired_error=float(np.max(abs(reference-predict(model,b,tr[:8]))));assert paired_error<1e-6
    host,rec=model.parameter_groups();opt=torch.optim.Adam([{'params':host,'lr':.003},{'params':rec,'lr':.0003}])
    y=torch.from_numpy(d['y']);wm=torch.from_numpy((d['m']*d['w'][:,None]/d['w'][tr].mean()).astype(np.float32))
    rng=np.random.default_rng(10000+seed);best=mse(predict(model,b,va),d,va);best_state=copy.deepcopy(model.state_dict());best_step=0;curve=[];start=time.time()
    for step in range(1,601):
        if model.kernel is not None:
            model.kernel.training_step.fill_(step)
            if step==1 or step%10==0:
                with torch.no_grad():model.kernel.calibrate_(model.hidden(b['xu'][tr]),step)
        idx=rng.choice(tr,size=min(64,len(tr)),replace=False)
        model.train();opt.zero_grad(set_to_none=True);p=model(batches(b,idx))['P']
        loss=((p-y[idx])**2*wm[idx]).sum()/wm[idx].sum();assert torch.isfinite(loss)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if step%50==0:
            v=mse(predict(model,b,va),d,va);curve.append(dict(step=step,train_loss=float(loss.detach()),validation_mse=v))
            if v<best:best=v;best_state=copy.deepcopy(model.state_dict());best_step=step
        if step%200==0:print(arm,fold,seed,'step',step,'seconds',round(time.time()-start),flush=True)
    model.load_state_dict(best_state);vp=predict(model,b,va);tp=predict(model,b,te)
    assert np.isfinite(tp).all() and ((tp>=0)&(tp<=1)).all()
    changes={n:float(torch.linalg.vector_norm(p.detach()-initial[n])) for n,p in model.named_parameters()}
    torch.save(dict(state=best_state,stats=dict(base=bs,full=fs,geo=gs),arm=arm,seed=seed,fold=fold),PRIVATE/f'{arm}_f{fold}_s{seed}.pt')
    dump(PRIVATE/f'{arm}_f{fold}_s{seed}.json',dict(arm=arm,fold=fold,seed=seed,best_step=best_step,validation_mse=best,seconds=time.time()-start,paired_initial_max_error=paired_error,parameter_changes=changes,curve=curve))
    np.savez(path,va=va,te=te,vp=vp,tp=tp)
    print('DONE',arm,fold,seed,'best',best_step,flush=True)

def damping(d,tr):
    t=np.arange(1,145)[None,:];rates=np.linspace(0,.2,101)
    loss=[mse(d['y0'][tr,None]*np.exp(-rate*t),d,tr) for rate in rates]
    return float(rates[np.argmin(loss)])

def tree(fold):
    import lightgbm as lgb
    if (PRIVATE/f'tree_f{fold}.npz').exists():return
    d=dict(np.load(PRIVATE/'data.npz',allow_pickle=False));tr,va,te=indices(d,fold);rate=damping(d,tr)
    dp=d['y0'][:,None]*np.exp(-rate*np.arange(1,145)[None,:]);residual=d['y']-dp
    lead=np.broadcast_to(np.arange(1,145)[None,:,None]/144,(len(d['y']),144,1))
    x=np.concatenate([d['full'][:,72:],lead],axis=2).astype(np.float32)
    outputs=dict(va=va,te=te,rate=np.array(rate));choices=[]
    for arm in ('TREE_W','TREE_G'):
        xx=x if arm=='TREE_W' else np.concatenate([x,np.broadcast_to(d['geo'][:,None,:],(len(x),144,d['geo'].shape[-1]))],axis=2)
        xt=xx[tr].reshape(-1,xx.shape[-1]);xv=xx[va].reshape(-1,xx.shape[-1]);xtest=xx[te].reshape(-1,xx.shape[-1])
        mt=d['m'][tr].ravel();mv=d['m'][va].ravel();wt=np.broadcast_to(d['w'][tr,None],d['m'][tr].shape).ravel()[mt];wv=np.broadcast_to(d['w'][va,None],d['m'][va].shape).ravel()[mv]
        train=lgb.Dataset(xt[mt],label=residual[tr].ravel()[mt],weight=wt/wt.mean(),free_raw_data=False)
        valid=lgb.Dataset(xv[mv],label=residual[va].ravel()[mv],weight=wv/wv.mean(),reference=train,free_raw_data=False)
        best=np.inf
        for leaves,child,l2 in ((7,120,30),(15,80,20),(31,80,30)):
            params=dict(objective='regression',metric='None',learning_rate=.035,num_leaves=leaves,min_data_in_leaf=child,lambda_l2=l2,num_threads=1,verbosity=-1,seed=20261003,feature_pre_filter=False,deterministic=True,force_col_wise=True)
            baseline=dp[va].ravel()[mv];truth=d['y'][va].ravel()[mv]
            def evaluate(p,data):return 'clipped_mse',float(np.average(((p+baseline).clip(0,1)-truth)**2,weights=wv)),False
            model=lgb.train(params,train,num_boost_round=500,valid_sets=[valid],feval=evaluate,callbacks=[lgb.early_stopping(50,verbose=False)])
            vp=(model.predict(xv,num_threads=1).reshape(len(va),144)+dp[va]).clip(0,1);val=mse(vp,d,va)
            choices.append(dict(arm=arm,leaves=leaves,min_child_samples=child,l2=l2,best_iteration=model.best_iteration,validation_mse=val))
            if val<best:
                best=val;outputs[arm+'_vp']=vp;outputs[arm+'_tp']=(model.predict(xtest,num_threads=1).reshape(len(te),144)+dp[te]).clip(0,1)
                model.save_model(str(PRIVATE/f'{arm}_f{fold}.txt'))
        print('DONE',arm,fold,'validation',best,flush=True)
    np.savez(PRIVATE/f'tree_f{fold}.npz',**outputs);dump(PRIVATE/f'tree_f{fold}.json',dict(fold=fold,damping_rate=rate,choices=choices))

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['build','neural','tree']);p.add_argument('--fold',type=int);p.add_argument('--seed',type=int);p.add_argument('--arm',choices=ARMS);a=p.parse_args()
    if a.command=='build':build()
    elif a.command=='neural':neural(a.fold,a.seed,a.arm)
    else:tree(a.fold)
if __name__=='__main__':main()
