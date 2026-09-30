"""D10 frozen D09 response-chain experiments; no fitting or optimization of models."""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[name]='2'
import argparse, hashlib, json, math, subprocess, sys, time, types
from pathlib import Path
import numpy as np
import torch
import d09_score as S
import d09_train as T
from d10_downstream_oracle import project
sys.path.insert(0,str(S.ROOT/'src'))
from asymode.asym_host import AsymODE
from asymode.controlled_relaxation import ControlledRelaxationLayer
from asymode.gcrk_train import make_batch, _std
from d09_write import IncrementalWriteRelaxationLayer

OLD_COMMIT='eb91994fd124bc7178e7edc0b175c84113406b68'
VARIANTS=('baseline','closed','rho_anchor','eta_anchor','gain_anchor','joint_anchor')
OWN_SOURCES=('d10_replay.py','d10_downstream_oracle.py','d10_common_fit.py')

def register(commit,scope):
    full=subprocess.check_output(['git','rev-parse','--verify',commit+'^{commit}'],cwd=S.ROOT,text=True).strip()
    paths=[S.HERE/name for name in OWN_SOURCES]+[scope]
    hashes={}
    for path in paths:
        rel=S.relative(path)
        blob=subprocess.check_output(['git','show',full+':'+rel],cwd=S.ROOT)
        digest=hashlib.sha256(blob).hexdigest()
        if S.sha(path)!=digest:raise ValueError('New diagnostic differs from registration: '+rel)
        hashes[rel]=digest
    return full,hashes

def anchored(original,variant):
    """Freeze only specified non-deposit controls to each county's u=0 values."""
    if variant not in VARIANTS[2:]:raise ValueError('Unknown registered intervention')
    def call(self,u,fg,hg,anchor):
        actual=original(u,fg,hg,anchor)
        zero=original(torch.zeros_like(u[:,:1]),fg,hg,anchor)
        keys=[]
        if variant in ('rho_anchor','joint_anchor'):keys.extend(('rho','tau'))
        if variant in ('eta_anchor','joint_anchor'):keys.append('eta')
        if variant in ('gain_anchor','joint_anchor'):keys.append('gain')
        for key in keys:actual[key]=zero[key].expand_as(actual[key])
        return actual
    return call

def synthetic():
    class Fake:
        def controls(self,u,fg,hg,anchor):
            q=u[:,:,:1]
            return dict(deposit=q+7,tau=q+8,rho=q+9,eta=q+10,gain=q+11)
    fake=Fake();u=torch.arange(12,dtype=torch.float64).reshape(2,3,2)
    original=fake.controls;base=original(u,None,None,None);checks=0
    for v in VARIANTS[2:]:
        ans=anchored(original,v)(fake,u,None,None,None)
        assert torch.equal(ans['deposit'],base['deposit']);checks+=1
        for key in ('tau','rho','eta','gain'):
            freeze=(v=='joint_anchor' or v=='rho_anchor' and key in ('tau','rho') or v=='eta_anchor' and key=='eta' or v=='gain_anchor' and key=='gain')
            expected=torch.full_like(base[key],{'tau':8,'rho':9,'eta':10,'gain':11}[key]) if freeze else base[key]
            assert torch.equal(ans[key],expected);checks+=1
    return checks

def rebuild(data,folder,arm,fold):
    ck=torch.load(folder/'final.pt',map_location='cpu',weights_only=False)
    expected=T.job_indices(data,fold)['fit']
    if ck['arm']!=arm or ck['steps']!=900 or ck['seed']!=0 or ck['registration_commit']!=OLD_COMMIT:
        raise ValueError('Frozen checkpoint metadata mismatch')
    if not np.array_equal(ck['fit_idx'],expected) or not np.array_equal(ck['fit_unit_ids'],data['unit'][expected]):
        raise ValueError('Frozen FIT identity mismatch')
    state=torch.random.get_rng_state();torch.manual_seed(0)
    f=data['F'];model=AsymODE(f['xu'].shape[-1],f['xr'].shape[-1],f['xo'].shape[-1]);model.attach_context_input(6)
    if arm=='new':
        g=torch.from_numpy(_std(f['geo'][expected],ck['stats']['geo'],0))
        model.damage[2]=IncrementalWriteRelaxationLayer(model.damage[2],g,county_ids=f['fips'][expected],
                    checkpoint_steps=0,use_adjoint=True,geo_checkpoint=True)
    torch.random.set_rng_state(state)
    model.load_state_dict(ck['model_state'],strict=True);model.eval()
    return model,ck['stats']

def compute(model,stats,data,rows,variant,diagnostics=False):
    layer=model.kernel;original=None
    if layer is not None and variant in VARIANTS[2:]:
        original=layer._controls
        layer._controls=types.MethodType(anchored(original,variant),layer)
    pieces={}
    try:
        with torch.no_grad():
            for start in range(0,len(rows),512):
                batch=make_batch(data['F'],rows[start:start+512],stats)
                out=model(batch,exit_open=variant!='closed',diagnostics=diagnostics)
                scalars={k:out[k] for k in ('P','u','r','gate','background','conditional','logit','raw_logit','forget')}
                if diagnostics:
                    scalars.update(h1_active_fraction=(out['h1'][:,72:]>0).float().mean(-1),
                                   a2_active_fraction=(out['a2'][:,72:]>0).float().mean(-1))
                    if layer is not None:
                        scalars.update(effect_norm=out['kernel_effect'][:,72:].norm(dim=-1),
                            deposit_mode_norm=out['kernel_deposit'][:,72:].norm(dim=-1).mean(-1),
                            state_mode_norm=out['kernel_state'][:,72:].norm(dim=-1).mean(-1))
                for key,val in scalars.items():
                    arr=val.cpu().numpy()
                    if arr.shape!=(len(batch['y0']),144) or not np.isfinite(arr).all():raise ValueError('Nonfinite replay '+key)
                    pieces.setdefault(key,[]).append(arr)
    finally:
        if original is not None:layer._controls=original
    return {k:np.concatenate(v) for k,v in pieces.items()}

def summarize(meta,predictions):
    d={k:np.asarray(v,dtype=np.float64) if k in ('y','y0','w','pi') else v for k,v in meta.items()}
    ps={k:np.asarray(p,dtype=np.float64) for k,p in predictions.items()}
    ph=S.phenotype(d['y'],d['m'],d['y0']);cs=S.cohort_masks(ph)
    peaks={k:S.model_peak(p,d['m'],ph) for k,p in ps.items()};rows={}
    strata={'all':np.ones(len(d['y']),bool)}
    strata.update({'regime/'+r:d['regime']==r for r in S.REGIMES})
    names=('all','S','nonS','J','S_complete144','severe_hour_count/1','severe_hour_count/2..6','severe_hour_count/>=7',
           'max_contiguous_run/1','max_contiguous_run/2..6','max_contiguous_run/>=7')
    for st,ss in strata.items():
        rows[st]={}
        for name in names:
            keep=ss&cs[name];row=S.score_row(d,ps,peaks,ph,keep,keep,plans={})
            row['diagnostic_comparisons']={}
            for reference in ('baseline','closed'):
                if reference not in ps:continue
                for cand,p in ps.items():
                    if cand==reference:continue
                    row['diagnostic_comparisons'][cand+'_vs_'+reference]=dict(
                        relative_RMSE_reduction=S.improvement(row['models'][reference]['design_SSE'],row['models'][cand]['design_SSE']),
                        paired_gain=S.gain_summary(d['y'],d['m'],ps[reference],p,keep,d['w']))
            rows[st][name]=row
    alarms={key:S.alarm_summary(peaks['baseline']['predicted_peak'],pk['predicted_peak'],cs['nonS'],d['w'],.1)
            for key,pk in peaks.items()}
    return dict(rows=rows,nonS_severe_alarms=alarms,confidence_intervals=False)

def working(meta,base):
    y=meta['y'].astype(float);m=meta['m'];w=meta['w'].astype(float)
    peak=np.argmax(np.where(m,y,-np.inf),axis=1);pred=np.argmax(np.where(m,base['P'],-np.inf),axis=1)
    severe=np.max(np.where(m,y,-np.inf),axis=1)>=.1
    false=(~severe)&(np.max(np.where(m,base['P'],-np.inf),axis=1)>=.1)
    ans={}
    for cohort,chosen in [('S_true_peak',severe),('nonS_predicted_false_peak',false),('all_fixed_clock',np.ones(len(y),bool))]:
        vals={}
        for key,arr in base.items():
            if cohort=='all_fixed_clock':
                clock=np.zeros_like(m);clock[:,::12]=True;mask=clock&m
                weight=np.broadcast_to(w[:,None],m.shape)[mask];v=arr[mask]
            else:
                ids=np.flatnonzero(chosen);t=peak[ids] if cohort=='S_true_peak' else pred[ids]
                weight=w[ids];v=arr[ids,t]
            vals[key]=S.distribution(v.astype(float),weight)
        ans[cohort]=dict(units=int(chosen.sum()),values=vals)
    return ans

def run(args):
    registration,own=register(args.registration_commit,args.scope)
    oldscope=S.HERE/'notes/D09_INCREMENTAL_WRITE_DESIGN_20260929_ZH.md'
    commit=S.verify_registration(OLD_COMMIT,oldscope)
    held,predictions,files,provenance=S.read_verified_jobs(args.jobs,oldscope,commit)
    del held,predictions
    out=S.under_root(args.out)
    if not S.relative(out).startswith('runs/geo_weather_20260924/d10_'):raise ValueError('Use owned new D10 output')
    out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();report=dict(schema='d10_response_chain_v1',registration_commit=registration,
        fold=args.fold,original_training_commit=OLD_COMMIT,source_sha256=own,verified_files_sha256=files,
        new_training=False,new_model_forward=True,new_model_optimization=False,confidence_intervals=False)
    try:
        data=T.load_data();indices=T.job_indices(data,args.fold)
        other=T.job_indices(data,3 if args.fold==2 else 2)
        common=np.intersect1d(indices['fit'],other['fit'])
        if len(common)!=535:raise ValueError('Common FIT differs from fixed roster')
        jobs=json.loads(args.jobs.read_text())['jobs'];inventory={(j['arm'],j['heldout_fold']):S.under_root(j['path']) for j in jobs}
        result={}
        for arm in ('host','new'):
            folder=inventory[arm,args.fold];model,stats=rebuild(data,folder,arm,args.fold)
            before={k:v.clone() for k,v in model.state_dict().items() if isinstance(v,torch.Tensor)}
            result[arm]={}
            for surface,ids,cache_name in [('full_heldout',indices['held_full'],'step0900_held_full.npz'),('common_FIT',common,'step0900_fit_panel.npz')]:
                with np.load(folder/cache_name,allow_pickle=False) as z:
                    pos=np.searchsorted(z['idx'],data['unit'][ids]);assert np.array_equal(z['idx'][pos],data['unit'][ids])
                    meta={k:z[k][pos].copy() for k in S.META_KEYS}
                    cached={k:z[k][pos].copy() for k in ('P','u','r','raw_logit')}
                    cached_closed={k:z[k+'_closed'][pos].copy() for k in ('P','raw_logit')} if arm=='new' else {}
                baseline=compute(model,stats,data,ids,'baseline',diagnostics=True)
                parity={k:float(np.max(np.abs(baseline[k]-v))) for k,v in cached.items()}
                if max(parity.values())>2e-6:raise ValueError('Frozen baseline does not reproduce cache')
                ps={'baseline':baseline['P']}
                closed_parity={}
                if arm=='new':
                    for variant in VARIANTS[1:]:
                        calculated=compute(model,stats,data,ids,variant)
                        ps[variant]=calculated['P']
                        if variant=='closed':closed_parity={k:float(np.max(np.abs(calculated[k]-v))) for k,v in cached_closed.items()}
                    if max(closed_parity.values())>2e-6:raise ValueError('Closed replay mismatch')
                res=dict(replay_max_abs=parity,closed_replay_max_abs=closed_parity,scores=summarize(meta,ps),working_points=working(meta,baseline))
                payload={**meta,**{'P_'+k:v for k,v in ps.items()},**{'baseline_'+k:v for k,v in baseline.items()}}
                if surface=='full_heldout':
                    severe=np.max(np.where(meta['m'],meta['y'],-np.inf),axis=1)>=.1;selected=np.flatnonzero(severe)
                    paths=[];cert=[]
                    for j,i in enumerate(selected):
                        p,c=project(meta['y'][i].astype(float),meta['m'][i],float(meta['y0'][i]),
                            baseline['r'][i].astype(float),baseline['gate'][i].astype(float),baseline['background'][i].astype(float))
                        paths.append(p);cert.append(c)
                        if j%20==0:print(f'{arm} fold{args.fold} downstream oracle {j}/{len(selected)}',flush=True)
                    p=np.asarray(paths);w=meta['w'][selected].astype(float);m=meta['m'][selected]
                    denom=float(np.sum(w[:,None]*m));actual=float(np.sum(w[:,None]*m*(baseline['P'][selected].astype(float)-meta['y'][selected])**2))
                    lo=float(w @ np.array([c['lower'] for c in cert]));hi=float(w @ np.array([c['upper'] for c in cert]))
                    pp=np.max(np.where(m,p,-np.inf),axis=1);yp=np.max(np.where(m,meta['y'][selected],-np.inf),axis=1)
                    res['fixed_downstream_oracle']=dict(S_units=len(selected),design_lower_SSE=lo,design_upper_SSE=hi,
                        actual_SSE=actual,lower_fraction_of_actual_SSE=lo/actual,upper_fraction_of_actual_SSE=hi/actual,
                        RMSE_bracket=[math.sqrt(lo/denom),math.sqrt(hi/denom)],
                        peak_ratio=S.distribution(pp/yp,w),maximum_gap=max(c['gap'] for c in cert),
                        unresolved_gap_gt_1e8=sum(c['gap']>1e-8 for c in cert),certificates=cert)
                    payload.update(oracle_selected_ids=meta['idx'][selected],oracle_paths=p)
                path=out/(arm+'_'+surface+'.npz')
                with path.open('xb') as f:np.savez_compressed(f,**payload)
                res['local_cache']=dict(path=S.relative(path),sha256=S.sha(path))
                result[arm][surface]=res
                print(f'{arm} fold{args.fold} {surface} complete',flush=True)
            for key,v in before.items():
                if not torch.equal(v,model.state_dict()[key]):raise ValueError('Frozen model/buffer changed '+key)
            del model
        for name,digest in {**files,**own}.items():
            if S.sha(S.ROOT/name)!=digest:raise ValueError('Frozen file/source changed during D10 '+name)
        report.update(status='complete',passed=True,results=result,seconds=time.monotonic()-started,
                      state_dict_tensors_unchanged=True,synthetic_control_assertions=synthetic())
        with (out/'RESULT.json').open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    except Exception as exc:
        with (out/'FAILED.json').open('x') as f:json.dump(dict(status='failed',error_type=type(exc).__name__,error=str(exc)),f)
        raise

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--self-check',action='store_true')
    p.add_argument('--registration-commit');p.add_argument('--scope',type=Path);p.add_argument('--jobs',type=Path)
    p.add_argument('--fold',type=int,choices=(2,3));p.add_argument('--out',type=Path);a=p.parse_args()
    os.nice(max(0,15-os.getpriority(os.PRIO_PROCESS,0)));torch.set_num_threads(2);torch.set_num_interop_threads(2)
    if a.self_check:print(json.dumps({'synthetic_assertions':synthetic()}));return
    for key in ('registration_commit','scope','jobs','fold','out'):
        if getattr(a,key) is None:p.error('Missing '+key)
    run(a)
if __name__=='__main__':main()
