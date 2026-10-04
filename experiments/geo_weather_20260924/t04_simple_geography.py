"""T04: small geographic amplitude / two-timescale weather-memory kernels.

Reuses the registered T03 cache/splits; private predictions and weights stay outside git.
Commands: audit, check, train --arm SHARED|GEO_GAIN|GEO_MIX --fold F --seed S.
"""
from t03_poster_trajectory import *
import math
from torch import nn

T03_PRIVATE=PRIVATE
PRIVATE=ROOT.parent/'t04_artifacts'
OUT=HERE/'results/t04'
ARMS=('SHARED','GEO_GAIN','GEO_MIX')
GEO=('canopy_mean','relief_p95_p5','poorly_drained_share','developed_frac')
CHANNELS=('g_mean','g_exc_mean','rain_mean','wet_wind_mean')

def load_data():
    meta=json.loads((HERE/'results/t03/build_audit.json').read_text())
    assert digest(T03_PRIVATE/'data.npz')==meta['cache_sha256']
    return dict(np.load(T03_PRIVATE/'data.npz',allow_pickle=False)),meta

class SimpleMemory(nn.Module):
    def __init__(self,host,arm,meta,stats):
        super().__init__();self.host=host;self.arm=arm
        self.mix=nn.Parameter(torch.zeros(4));self.gain=nn.Parameter(torch.zeros(4))
        self.geo_coef=nn.Parameter(torch.zeros(4,4)) if arm!='SHARED' else None
        names=meta['base_names']+meta['dose_names']
        pairs=[[names.index(f'{src}:{c}:ewma{decay}') for decay in (.9,.98)]
               for src in ('era5','hrrr') for c in CHANNELS]
        self.register_buffer('pairs',torch.tensor(pairs))
        self.register_buffer('mu',torch.tensor(stats['mean'],dtype=torch.float32))
        self.register_buffer('sd',torch.tensor(stats['std'],dtype=torch.float32))

    def parameters_for_counties(self,geo):
        z=torch.tanh(geo/3)
        effect=z@self.geo_coef.T if self.geo_coef is not None else torch.zeros_like(z)
        a=torch.sigmoid(self.mix+(effect if self.arm=='GEO_MIX' else torch.zeros_like(z)))
        g=torch.exp(math.log(2)*torch.tanh(self.gain+(effect if self.arm=='GEO_GAIN' else torch.zeros_like(z))))
        return a,g

    def memory_inputs(self,b):
        a,g=self.parameters_for_counties(b['g4'])
        a=a.repeat(1,2)[:,None,:];g=g.repeat(1,2)[:,None,:]
        raw=b['raw_memory']
        mem=g*(a*raw[:,:,:,0]+(1-a)*raw[:,:,:,1])
        xu=b['xu'].clone()
        for k in range(2):
            ix=self.pairs[:,k]
            xu[:,:,ix]=((mem-self.mu[ix])/self.sd[ix]).clamp(-10,10)
        return xu

    def forward(self,b):
        return self.host({**b,'xu':self.memory_inputs(b)})

    def penalty(self):
        return 1e-4*self.geo_coef.square().sum() if self.geo_coef is not None else self.mix.new_zeros(())

def prepare(d,meta,fold,seed,arm):
    tr,va,te=indices(d,fold)
    base,bs=standardize(d['base'],tr);full,fs=standardize(d['full'],tr)
    g4,gs=standardize(d['geo'][:,[meta['geography_columns'].index(c) for c in GEO]],tr)
    torch.manual_seed(seed)
    host=AsymODE(base.shape[-1],base.shape[-1],len(d['occ']))
    host.expand_damage_inputs(full.shape[-1]-base.shape[-1])
    model=SimpleMemory(host,arm,meta,fs)
    raw=d['full'][:,:,model.pairs.numpy()]
    raw=np.where(np.isfinite(raw),raw,np.array(fs['mean'],np.float32)[model.pairs.numpy()])
    b=dict(xu=torch.from_numpy(full),xr=torch.from_numpy(base),xo=torch.from_numpy(base[:,:,d['occ']]),
           y0=torch.from_numpy(d['y0']),g4=torch.from_numpy(g4),raw_memory=torch.from_numpy(raw.astype(np.float32)))
    return model,b,(tr,va,te),dict(base=bs,full=fs,geo4=gs)

def audit():
    OUT.mkdir(parents=True,exist_ok=True);PRIVATE.mkdir(parents=True,exist_ok=True)
    d,meta=load_data();z=np.load(DATA/'outcomes/panel_v1D_outcomes.npz',allow_pickle=False)
    ix=np.flatnonzero(z['regime']=='tropical');assert np.array_equal(z['fips'][ix],d['fips'])
    catalog=pd.read_csv(HERE/'results/t01/candidate_audit.csv').set_index('system')
    rows=[]
    for event in sorted(set(d['system'])):
        keep=d['system']==event;obs=z['observed'][ix][keep];geo=d['geo'][keep][:,[meta['geography_columns'].index(c) for c in GEO]]
        row=dict(system=event,storm=catalog.loc[event,'storm'],county_events=int(keep.sum()),
                 observed_fraction=float(obs.mean()),complete_counties=int(obs.all(1).sum()),
                 geography_finite=float(np.isfinite(geo).mean()))
        for source in ('era5','hrrr'):
            row[source+'_available']=float(d['base'][keep,:,meta['base_names'].index(source+':available')].mean())
            for channel in ('g_mean','rain_mean'):
                v=d['base'][keep,:,meta['base_names'].index(source+':'+channel)]
                peaks=np.nanmax(v,axis=1)
                row[source+'_'+channel+'_peak_p50']=float(np.nanmedian(peaks))
                row[source+'_'+channel+'_peak_p90']=float(np.nanquantile(peaks,.9))
        for j,c in enumerate(GEO):
            row[c+'_iqr']=float(np.nanquantile(geo[:,j],.75)-np.nanquantile(geo[:,j],.25))
        reasons=[]
        if keep.sum()<50:reasons.append('counties<50')
        if obs.mean()<.99:reasons.append('outage_coverage<99%')
        for source in ('era5','hrrr'):
            if row[source+'_available']<.95:reasons.append(source+'_coverage<95%')
        if row['geography_finite']<.99:reasons.append('geo_coverage<99%')
        row['quality_candidate']=not reasons;row['exclusion_reason']='; '.join(reasons)
        rows.append(row)
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'dataset_scope_audit.csv',index=False)
    dump(OUT/'scope_registration.json',dict(source_cache_sha256=meta['cache_sha256'],
        candidate_events=frame.loc[frame.quality_candidate,'system'].tolist(),
        criterion='n>=50, outage coverage>=.99, each weather source availability>=.95, geo4 finite>=.99',
        model_scores_used=False,all_events_primary=True,source_sha256=digest(__file__)))
    print(frame[['system','storm','county_events','observed_fraction','hrrr_available','quality_candidate','exclusion_reason']].to_string(index=False))

def check():
    d,meta=load_data();outputs=[];records=[]
    for arm in ARMS:
        model,b,(tr,va,te),stats=prepare(d,meta,0,0,arm);idx=tr[:4]
        out=predict(model,b,idx);outputs.append(out)
        host_out=predict(model.host,b,idx)
        assert np.max(abs(out-host_out))==0
        batch=batches(b,idx)
        before=model.memory_inputs(batch).detach().clone()
        changed={**batch,'raw_memory':batch['raw_memory'].clone()}
        changed['raw_memory'][:,100:]+=10
        assert torch.equal(before[:,:100],model.memory_inputs(changed)[:,:100])
        # Random bounded conditioning must stay finite and receive gradients.
        if model.geo_coef is not None:
            with torch.no_grad():model.geo_coef.copy_(torch.arange(16).reshape(4,4)/20)
        a,g=model.parameters_for_counties(b['g4'])
        assert ((a>0)&(a<1)&(g>.5)&(g<2)).all()
        with torch.no_grad():model.host.damage[0].weight[:,58:]=.01
        loss=model(batch)['P'].square().mean();loss.backward()
        grad={n:float(p.grad.norm()) for n,p in model.named_parameters() if not n.startswith('host.')}
        assert all(np.isfinite(list(grad.values())))
        if model.geo_coef is not None:assert grad['geo_coef']>0
        records.append(dict(arm=arm,kernel_parameter_count=sum(p.numel() for n,p in model.named_parameters() if not n.startswith('host.')),gradient_norm=grad))
    error=max(float(np.max(abs(p-outputs[0]))) for p in outputs)
    assert error==0
    dump(OUT/'implementation_checks.json',dict(paired_initial_max_error=error,causal_input_check=True,bounded_parameters=True,records=records))
    print('CHECK PASSED',records)

def train(fold,seed,arm):
    PRIVATE.mkdir(parents=True,exist_ok=True)
    path=PRIVATE/f'{arm}_f{fold}_s{seed}.npz'
    if path.exists():print('skip',path.name,flush=True);return
    d,meta=load_data();model,b,(tr,va,te),stats=prepare(d,meta,fold,seed,arm)
    paired=float(np.max(abs(predict(model,b,tr[:8])-predict(model.host,b,tr[:8]))));assert paired==0
    initial={n:p.detach().clone() for n,p in model.named_parameters()}
    host,rec=model.host.parameter_groups()
    extra=[p for n,p in model.named_parameters() if not n.startswith('host.')]
    opt=torch.optim.Adam([dict(params=host+extra,lr=.003),dict(params=rec,lr=.0003)])
    y=torch.from_numpy(d['y']);wm=torch.from_numpy((d['m']*d['w'][:,None]/d['w'][tr].mean()).astype(np.float32))
    rng=np.random.default_rng(10000+seed);best=mse(predict(model,b,va),d,va);state=copy.deepcopy(model.state_dict());step_best=0;curve=[];start=time.time()
    for step in range(1,601):
        sel=rng.choice(tr,size=min(64,len(tr)),replace=False)
        model.train();opt.zero_grad(set_to_none=True);p=model(batches(b,sel))['P']
        data_loss=((p-y[sel])**2*wm[sel]).sum()/wm[sel].sum()
        loss=data_loss+model.penalty();assert torch.isfinite(loss)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if step%50==0:
            v=mse(predict(model,b,va),d,va)
            curve.append(dict(step=step,train_mse=float(data_loss.detach()),penalty=float(model.penalty().detach()),validation_mse=v))
            if v<best:best=v;state=copy.deepcopy(model.state_dict());step_best=step
        if step%200==0:print(arm,fold,seed,step,round(time.time()-start),flush=True)
    model.load_state_dict(state);vp=predict(model,b,va);tp=predict(model,b,te)
    with torch.no_grad():a,g=model.parameters_for_counties(b['g4'][te])
    changes={n:float((p.detach()-initial[n]).norm()) for n,p in model.named_parameters()}
    report=dict(arm=arm,fold=fold,seed=seed,best_step=step_best,validation_mse=best,seconds=time.time()-start,
                paired_initial_max_error=paired,parameter_changes=changes,curve=curve,
                alpha_outer_quantiles=np.quantile(a.numpy(),[0,.5,1],axis=0).tolist(),
                gain_outer_quantiles=np.quantile(g.numpy(),[0,.5,1],axis=0).tolist(),
                kernel_parameters={n:p.detach().tolist() for n,p in model.named_parameters() if not n.startswith('host.')},
                source_sha256=digest(__file__),cache_sha256=meta['cache_sha256'])
    assert np.isfinite(tp).all() and ((tp>=0)&(tp<=1)).all()
    torch.save(dict(state=state,stats=stats,arm=arm,fold=fold,seed=seed),PRIVATE/f'{arm}_f{fold}_s{seed}.pt')
    dump(PRIVATE/f'{arm}_f{fold}_s{seed}.json',report);np.savez_compressed(path,va=va,te=te,vp=vp,tp=tp)
    print('DONE',arm,fold,seed,'best',step_best,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['audit','check','train']);p.add_argument('--fold',type=int);p.add_argument('--seed',type=int);p.add_argument('--arm',choices=ARMS);a=p.parse_args()
    if a.command=='audit':audit()
    elif a.command=='check':check()
    else:train(a.fold,a.seed,a.arm)
