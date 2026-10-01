"""Nested event-fold tropical prediction campaign; see RUN_PROTOCOL.md."""
from __future__ import annotations
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ.setdefault(key,'1')
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
import lightgbm as lgb
from scipy.optimize import minimize

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'src'))
from asymode.asym_host import AsymODE
torch.set_num_threads(1)
OUT=ROOT/'runs/tropical_prediction_v1'
DATA=ROOT/'data/interim/tropical_prediction_v1/development.npz'
MODELS=['zero','persistence','damped','host','fusion','tree_era5_l2','tree_dual_l2','tree_dual_l1']
STEPS=900
CODE_HASH=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

def emit(**kwargs):print(json.dumps(kwargs),flush=True)

def history_features(y,m):
    h=np.zeros_like(y,dtype='float32')
    for t in range(72):h[:,t]=np.where(m[:,t],y[:,t],h[:,t-1] if t else 0)
    columns=[h[:,-1],m.mean(1)]
    for width in [3,6,12,24,72]:
        v=h[:,-width:]
        columns.extend([v.mean(1),v.max(1),v.std(1),(v[:,-1]-v[:,0])/max(width-1,1)])
    return h,np.stack(columns,1).astype('float32')

def load():
    a=dict(np.load(DATA,allow_pickle=False))
    a['target']=np.nan_to_num(a['target']);a['target_observed']=a['target_observed'].astype('float32')
    h,hf=history_features(a['history'],a['history_observed']);a['hf']=hf;a['y0']=h[:,-1]
    gn=a['geography_names'].astype(str)
    exclude=('n_land_pixels','n_samples','n_soil_samples','n_elev_points','soil_mapped_share','windthrow_rated_share','fia_state_evaluation')
    gi=[i for i,n in enumerate(gn) if not n.startswith('context__') and not any(x in n for x in exclude)]
    ci=[i for i,n in enumerate(gn) if n.startswith('context__')]
    a['geo']=a['geography'][:,gi];a['ctx']=a['geography'][:,ci]
    a['geo_names']=gn[gi];a['ctx_names']=gn[ci]
    # Clock only, never an event identifier or outcome-derived storm descriptor.
    start=pd.to_datetime(a['window_start_utc']);hr=(np.asarray(start.hour)[:,None]+np.arange(216))%24
    clock=np.stack([np.sin(2*np.pi*hr/24),np.cos(2*np.pi*hr/24)],-1).astype('float32')
    a['wx']={}
    for src in ['era5','dual']:
        pieces=[]
        for s in (['era5'] if src=='era5' else ['era5','hrrr']):
            cols=[i for i,n in enumerate(a[s+'_names'].astype(str)) if '*' not in n]
            w=a[s+'_weather'][:,:,cols]
            pieces.extend([w,a[s+'_weather_observed'][:,:,None].astype('float32')])
        pieces.append(clock)
        a['wx'][src]=np.concatenate(pieces,-1).astype('float32')
    return a

def normalize(x,idx):
    mean=np.nanmean(x[idx],axis=tuple(range(x.ndim-1)))
    mean=np.nan_to_num(mean)
    filled=np.where(np.isfinite(x),x,mean)
    sd=filled[idx].std(axis=tuple(range(x.ndim-1)))+1e-5
    return np.clip((filled-mean)/sd,-8,8).astype('float32'),dict(mean=mean,sd=sd)

class ResidualHead(nn.Module):
    def __init__(self,base):
        super().__init__();self.base=base
        self.adapter=nn.Sequential(nn.Linear(base.in_features,16),nn.Tanh(),nn.Linear(16,1))
        nn.init.zeros_(self.adapter[-1].weight);nn.init.zeros_(self.adapter[-1].bias)
    def forward(self,x):return self.base(x)+self.adapter(x)

def batches(a,fit):
    w,ws=normalize(a['wx']['dual'],fit)
    g,gs=normalize(a['geo'],fit);ctx,cs=normalize(a['ctx'],fit);hf,hs=normalize(a['hf'],fit)
    # Physical hazard memory (weather known as exogenous, not future outages).
    mem=[]
    for decay in [.5,.9,.98]:
        state=np.zeros((len(w),w.shape[-1]),'float32');p=[]
        for t in range(216):state=decay*state+(1-decay)*w[:,t];p.append(state.copy())
        mem.append(np.stack(p,1))
    # Three tropical weather channels per source: gust, exceedance, rain.
    dims=[0,1,2,23,24,25]
    xu=np.concatenate([w]+[v[:,:,dims] for v in mem],-1)
    static=np.concatenate([ctx,hf],-1)
    xr=np.concatenate([w,np.broadcast_to(static[:,None],(len(w),216,static.shape[-1]))],-1)
    result={k:torch.from_numpy(np.ascontiguousarray(v)) for k,v in
            dict(xu=xu,xr=xr,xo=w[:,:,dims],geo=g,ctx=ctx,y0=a['y0']).items()}
    return result,dict(weather=ws,geo=gs,ctx=cs,history=hs)

def model_for(b,kind,seed):
    torch.manual_seed(seed)
    model=AsymODE(b['xu'].shape[-1],b['xr'].shape[-1],b['xo'].shape[-1])
    model.attach_context_input(b['ctx'].shape[-1])
    if kind=='fusion':
        model.attach_gcrk(torch.tanh(b['geo']/3).mean(0),private_seed=1729)
        with torch.random.fork_rng():
            torch.manual_seed(1729)
            model.damage[4]=ResidualHead(model.damage[4]);model.recovery[-1]=ResidualHead(model.recovery[-1])
    return model

def cache_paths(tag):
    p=OUT/'cache'/tag;p.parent.mkdir(parents=True,exist_ok=True)
    return p.with_suffix('.npz'),p.with_suffix('.json'),p.with_suffix('.pt')

def fit_neural(a,fit,predict,kind,seed,tag,steps=STEPS):
    pp,jp,ck=cache_paths(tag)
    identity=dict(code=CODE_HASH,kind=kind,seed=seed,steps=steps,fit=fit.tolist(),predict=predict.tolist())
    if jp.exists():
        d=json.loads(jp.read_text());assert d['identity']==identity
        return np.load(pp)['P']
    t0=time.time();b,stats=batches(a,fit)
    model=model_for({k:v[fit] for k,v in b.items()},kind,seed)
    host,rec=model.parameter_groups();opt=torch.optim.Adam([dict(params=host,lr=.003),dict(params=rec,lr=.0003)])
    y=torch.from_numpy(a['target']);m=torch.from_numpy(a['target_observed'])
    w=torch.from_numpy((a['w']/a['w'][fit].mean()).astype('float32'))
    rng=np.random.default_rng(seed);cal=fit[np.linspace(0,len(fit)-1,min(256,len(fit)),dtype=int)]
    curve=[]
    def refresh(step):
        if model.kernel is not None:
            with torch.no_grad():
                model.kernel.training_step.fill_(step)
                model.kernel.calibrate_(model.hidden(b['xu'][cal],b['ctx'][cal]),step)
    refresh(0)
    for step in range(steps):
        if step%10==0:refresh(step)
        if model.kernel is not None:model.kernel.training_step.fill_(step)
        model.train();idx=rng.choice(fit,size=min(128,len(fit)),replace=False)
        opt.zero_grad(set_to_none=True);p=model({k:v[idx] for k,v in b.items()})['P']
        loss=((p-y[idx]).square()*m[idx]*w[idx,None]).sum()/m[idx].sum().clamp_min(1)
        if not torch.isfinite(loss):raise RuntimeError('nonfinite loss '+tag)
        loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);opt.step()
        if step%100==0:
            curve.append(dict(step=step+1,loss=float(loss.detach()),gradient_norm=float(norm)))
    refresh(steps);model.eval();pred=[]
    with torch.no_grad():
        for idx in np.array_split(predict,max(1,int(np.ceil(len(predict)/128)))):
            pred.append(model({k:v[idx] for k,v in b.items()})['P'].numpy())
    p=np.concatenate(pred)
    assert np.isfinite(p).all() and (p>=0).all() and (p<=1).all()
    torch.save(dict(state=model.state_dict(),stats=stats,identity=identity,geo_names=a['geo_names'],ctx_names=a['ctx_names']),ck)
    np.savez_compressed(pp,P=p)
    jp.write_text(json.dumps(dict(identity=identity,seconds=time.time()-t0,curve=curve),indent=2))
    emit(completed=tag,seconds=round(time.time()-t0,1))
    return p

def decay(a,fit):
    t=np.arange(1,145);y=a['target'][fit];wm=a['target_observed'][fit]*a['w'][fit,None]
    losses=[np.sum(wm*(a['y0'][fit,None]*np.exp(-r*t)-y)**2) for r in np.linspace(0,.2,101)]
    return float(np.linspace(0,.2,101)[np.argmin(losses)])

def baselines(a,fit,predict):
    rate=decay(a,fit);p=np.repeat(a['y0'][predict,None],144,1)
    return dict(zero=np.zeros_like(p),persistence=p,damped=p*np.exp(-rate*np.arange(1,145))),rate

def tree_features(a,src):
    w=a['wx'][src];n=len(w);future=w[:,72:]
    static=np.concatenate([a['geo'],a['ctx'],a['hf'],w[:,:72].mean(1),w[:,:72].max(1),future.max(1),future.mean(1)],-1)
    blocks=[np.broadcast_to(static[:,None],(n,144,static.shape[-1])),future]
    for tau in [6,24]:
        cs=np.concatenate([np.zeros_like(w[:,:1]),np.cumsum(w,1)],1)
        times=np.arange(72,216)
        blocks.append((cs[:,times+1]-cs[:,times+1-tau])/tau)
    for d in [.9,.98]:
        state=np.zeros_like(w[:,0]);mem=[]
        for t in range(216):state=d*state+(1-d)*w[:,t];mem.append(state.copy())
        blocks.append(np.stack(mem,1)[:,72:])
    lead=np.broadcast_to(np.arange(1,145,dtype='float32')[None,:,None],(n,144,1))
    blocks.append(lead)
    out=np.concatenate(blocks,-1).astype('float32')
    return np.where(np.isfinite(out),out,np.nan)

def fit_tree(a,features,fit,predict,kind,tag):
    pp,jp,_=cache_paths(tag)
    identity=dict(code=CODE_HASH,kind=kind,fit=fit.tolist(),predict=predict.tolist())
    if jp.exists():
        d=json.loads(jp.read_text());assert d['identity']==identity
        return np.load(pp)['P']
    t0=time.time();residual=kind.endswith('l2');rate=decay(a,fit)
    base=a['y0'][:,None]*np.exp(-rate*np.arange(1,145)) if residual else np.zeros_like(a['target'])
    flat=features[fit].reshape(-1,features.shape[-1]);ok=a['target_observed'][fit].reshape(-1).astype(bool)
    target=(a['target']-base)[fit].reshape(-1)
    weights=np.broadcast_to(a['w'][fit,None],(len(fit),144)).reshape(-1);weights=weights/weights[ok].mean()
    model=lgb.LGBMRegressor(objective='regression' if residual else 'regression_l1',n_estimators=300,
        num_leaves=15,min_child_samples=80,learning_rate=.035,reg_lambda=20,n_jobs=1,verbosity=-1,random_state=0)
    model.fit(flat[ok],target[ok],sample_weight=weights[ok])
    p=model.booster_.predict(features[predict].reshape(-1,features.shape[-1])).reshape(len(predict),144)
    p=np.clip(p+base[predict],0,1).astype('float32')
    model.booster_.save_model(str(jp.with_suffix('.txt')))
    np.savez_compressed(pp,P=p);jp.write_text(json.dumps(dict(identity=identity,seconds=time.time()-t0,decay_rate=rate),indent=2))
    emit(completed=tag,seconds=round(time.time()-t0,1));return p

def blend_weights(a,dev,preds):
    ok=a['target_observed'][dev].reshape(-1).astype(bool)
    x=np.stack([preds[n].reshape(-1)[ok] for n in MODELS],1).astype(float)
    y=a['target'][dev].reshape(-1)[ok].astype(float)
    w=np.broadcast_to(a['w'][dev,None],(len(dev),144)).reshape(-1)[ok].astype(float);w/=w.sum()
    scale=max(float(np.sum(w*y*y)),1e-8)
    Q=(x.T@(w[:,None]*x))/scale;c=x.T@(w*y)/scale
    def obj(v):return float(v@Q@v-2*c@v)
    r=minimize(obj,np.ones(len(MODELS))/len(MODELS),jac=lambda v:2*(Q@v-c),method='SLSQP',
        bounds=[(0,1)]*len(MODELS),constraints=[dict(type='eq',fun=lambda v:v.sum()-1,jac=lambda v:np.ones_like(v))],
        options=dict(ftol=1e-12,maxiter=1000))
    if not r.success:raise RuntimeError('blend optimization failed: '+r.message)
    v=np.clip(r.x,0,1);v/=v.sum()
    return v,dict(weights=dict(zip(MODELS,v.tolist())),optimizer=r.message,inner_mse=float(np.sum(w*(x@v-y)**2)))

def run_fold(a,k):
    OUT.mkdir(parents=True,exist_ok=True);result=OUT/f'fold{k}'
    if (result/'DONE.json').exists():
        d=json.loads((result/'DONE.json').read_text());assert d['code']==CODE_HASH;emit(cached_fold=k);return
    result.mkdir(exist_ok=True)
    sp=json.loads((HERE/'splits.json').read_text())[str(k)]
    dev=np.array(sp['dev']);outer=np.array(sp['outer'])
    features={s:tree_features(a,s) for s in ['era5','dual']}
    inner={n:np.full((len(a['y0']),144),np.nan,dtype='float32') for n in MODELS}
    emit(start_fold=k,dev=len(dev),outer=len(outer),steps=STEPS)
    for split in sp['inner']:
        j=split['validation_source_fold'];fit=np.array(split['train']);val=np.array(split['validation'])
        pred,rate=baselines(a,fit,val)
        for kind in ['host','fusion']:pred[kind]=fit_neural(a,fit,val,kind,0,f'f{k}_inner{j}_{kind}_s0')
        for kind in ['tree_era5_l2','tree_dual_l2','tree_dual_l1']:
            src='era5' if 'era5' in kind else 'dual'
            pred[kind]=fit_tree(a,features[src],fit,val,kind,f'f{k}_inner{j}_{kind}')
        for name in MODELS:inner[name][val]=pred[name]
    ip={name:p[dev] for name,p in inner.items()};assert all(np.isfinite(p).all() for p in ip.values())
    ww,selection=blend_weights(a,dev,ip)
    # This is written before fitting/exporting the outer predictions.
    (result/'selection.json').write_text(json.dumps(selection,indent=2))
    np.savez_compressed(result/'inner_predictions.npz',idx=dev,**ip)
    pred,rate=baselines(a,dev,outer)
    for kind in ['host','fusion']:
        pred[kind]=np.mean([fit_neural(a,dev,outer,kind,s,f'f{k}_outer_{kind}_s{s}') for s in [0,1,2]],0)
    for kind in ['tree_era5_l2','tree_dual_l2','tree_dual_l1']:
        src='era5' if 'era5' in kind else 'dual'
        pred[kind]=fit_tree(a,features[src],dev,outer,kind,f'f{k}_outer_{kind}')
    pred['blend']=sum(w*pred[name] for name,w in zip(MODELS,ww))
    np.savez_compressed(result/'outer_predictions.npz',idx=outer,**pred)
    (result/'DONE.json').write_text(json.dumps(dict(code=CODE_HASH,fold=k,steps=STEPS,outer_seeds=[0,1,2],
        data_sha256=hashlib.sha256(DATA.read_bytes()).hexdigest(),dev=dev.tolist(),outer=outer.tolist(),selection=selection),indent=2))
    emit(complete_fold=k)

def smoke(a):
    fit=np.flatnonzero(a['outer_fold']!=1)[:128];b,_=batches(a,fit);bb={k:v[fit] for k,v in b.items()}
    host=model_for(bb,'host',0);fusion=model_for(bb,'fusion',0)
    fusion.kernel.calibrate_(fusion.hidden(bb['xu'],bb['ctx']),0)
    host.eval();fusion.eval()
    with torch.no_grad():diff=float((host(bb)['P']-fusion(bb)['P']).abs().max())
    assert diff==0, diff
    for name,model in [('host',host),('fusion',fusion)]:
        model.train();opt=torch.optim.Adam(model.parameters(),lr=.001);t=time.time()
        for step in range(10):
            if model.kernel is not None:model.kernel.training_step.fill_(100)
            opt.zero_grad();p=model(bb)['P'];loss=p.square().mean();loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);opt.step()
        emit(smoke=name,paired_start_max_difference=diff,seconds_per_step=(time.time()-t)/10,shape=list(p.shape))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--fold',type=int,choices=range(1,6));parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();a=load()
    if args.smoke:smoke(a)
    elif args.fold:run_fold(a,args.fold)
    else:parser.error('choose --fold or --smoke')
