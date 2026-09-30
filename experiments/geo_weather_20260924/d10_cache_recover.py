"""Recover D10 scalar summaries from preserved caches; no model reconstruction."""
from __future__ import annotations
import argparse,hashlib,json,math,os,subprocess,time
from pathlib import Path
import d10_replay as R
import numpy as np

REGISTRATION='1ca63c44cc18c42397bdcd777ca44f013fa7db32'
S=R.S
SCOPE=S.HERE/'notes/D10_RESPONSE_CHAIN_SCOPE_20260929_ZH.md'
AMENDMENT=S.HERE/'notes/D10_SUMMARY_EXPORT_REPAIR_20260930_ZH.md'
BASE=S.ROOT/'runs/geo_weather_20260924/d10_response_chain_20260929'

def guarded(commit):
    full=subprocess.check_output(['git','rev-parse',commit+'^{commit}'],cwd=S.ROOT,text=True).strip()
    for path in (Path(__file__).resolve(),AMENDMENT):
        blob=subprocess.check_output(['git','show',full+':'+S.relative(path)],cwd=S.ROOT)
        if hashlib.sha256(blob).hexdigest()!=S.sha(path):raise ValueError('Recovery source/amendment changed')
    _,own=R.register(REGISTRATION,SCOPE)
    own.update({S.relative(Path(__file__).resolve()):S.sha(Path(__file__).resolve()),S.relative(AMENDMENT):S.sha(AMENDMENT)})
    snapshots=json.loads(AMENDMENT.read_text().split('```json\n')[1].split('\n```')[0])
    if len(snapshots)!=12 or sum(k.endswith('.npz') for k in snapshots)!=8:raise ValueError('Preserved inventory mismatch')
    for key,value in snapshots.items():
        if S.sha(S.ROOT/key)!=value:raise ValueError('Preserved artifact changed '+key)
    oldscope=S.HERE/'notes/D09_INCREMENTAL_WRITE_DESIGN_20260929_ZH.md'
    old=S.verify_registration(R.OLD_COMMIT,oldscope)
    held,preds,files,provenance=S.read_verified_jobs(S.ROOT/'runs/geo_weather_20260924/d09_write_screen_20260929/JOBS.json',oldscope,old)
    del held,preds
    return full,snapshots,{**files,**own}

def witness(path,y,m,y0,r,gate,bg):
    previous=np.r_[y0,path[:-1]];lo=bg+(1-r-bg)*previous;hi=bg+.5*gate+(1-r-bg-.5*gate)*previous
    violation=float(max(0.,np.max(lo-path),np.max(path-hi),-path.min(),path.max()-1))
    if violation>1e-10:raise ValueError('Preserved oracle path is not feasible')
    denominator=1-previous
    u=np.divide(path-(1-r)*previous,denominator,out=bg.copy(),where=denominator!=0)
    c=np.divide(u-bg,gate,out=np.zeros_like(u),where=gate!=0).clip(0,.5)
    replay=previous+(bg+gate*c)*(1-previous)-r*previous
    residual=float(np.max(np.abs(path-replay)))
    if residual>1e-10:raise ValueError('Preserved oracle rate reconstruction failed')
    return float(np.sum(np.where(m,(path-y)**2,0.))),violation,residual

def run(args):
    full,snapshots,files=guarded(args.recovery_commit)
    out=BASE/('fold'+str(args.fold)+'_v2.json')
    if out.exists():raise FileExistsError('Preserve existing recovery output')
    failure=json.loads((BASE/f'fold{args.fold}'/'FAILED.json').read_text())
    if failure.get('error_type')!='TypeError' or failure.get('error')!='Object of type int64 is not JSON serializable':raise ValueError('Unrecognized original failure')
    started=time.monotonic();results={}
    for arm in ('host','new'):
        results[arm]={}
        for surface in ('full_heldout','common_FIT'):
            path=BASE/f'fold{args.fold}'/(arm+'_'+surface+'.npz')
            with np.load(path,allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
            meta={k:d[k] for k in S.META_KEYS};ps={k[2:]:v for k,v in d.items() if k.startswith('P_')}
            baseline={k[9:]:v for k,v in d.items() if k.startswith('baseline_')}
            if len(meta['idx'])!=(535 if surface=='common_FIT' else (2100 if args.fold==2 else 1769)):raise ValueError('Full surface count mismatch')
            jobs=json.loads((S.ROOT/'runs/geo_weather_20260924/d09_write_screen_20260929/JOBS.json').read_text())['jobs']
            folder=next(S.under_root(j['path']) for j in jobs if j['arm']==arm and j['heldout_fold']==args.fold)
            cache=folder/('step0900_held_full.npz' if surface=='full_heldout' else 'step0900_fit_panel.npz')
            with np.load(cache,allow_pickle=False) as z:
                pos=np.searchsorted(z['idx'],meta['idx']);assert np.array_equal(z['idx'][pos],meta['idx'])
                if surface=='full_heldout':assert np.array_equal(z['idx'],meta['idx'])
                else:
                    other=next(S.under_root(j['path']) for j in jobs if j['arm']==arm and j['heldout_fold']!=(args.fold))
                    with np.load(other/'step0900_fit_panel.npz',allow_pickle=False) as oz:expected=np.intersect1d(z['idx'],oz['idx'])
                    assert np.array_equal(expected,meta['idx']) and len(expected)==535
                for k in S.META_KEYS:
                    assert np.array_equal(z[k][pos],meta[k],equal_nan=True) if meta[k].dtype.kind in 'fci' else np.array_equal(z[k][pos],meta[k])
                parity={k:float(np.max(np.abs(z[k][pos]-baseline[k]))) for k in ('P','u','r','raw_logit')}
                closed_parity={'P':float(np.max(np.abs(z['P_closed'][pos]-ps['closed'])))} if arm=='new' else {}
            if max(parity.values())>2e-6 or closed_parity and max(closed_parity.values())>2e-6:raise ValueError('Cached replay parity failed')
            res=dict(replay_max_abs=parity,closed_prediction_max_abs=closed_parity.get('P'),
                original_closed_raw_logit_guard_completed_before_serialization_failure=arm=='new',
                scores=R.summarize(meta,ps),working_points=R.working(meta,baseline),
                local_cache=dict(path=S.relative(path),sha256=S.sha(path)))
            if surface=='full_heldout':
                sel=np.flatnonzero(np.max(np.where(meta['m'],meta['y'],-np.inf),axis=1)>=.1)
                if len(sel)!=(117 if args.fold==2 else 183) or not np.array_equal(meta['idx'][sel],d['oracle_selected_ids']):raise ValueError('Complete S coverage mismatch')
                cert=[];peaks=[];deltas=[]
                for j,i in enumerate(sel):
                    y=meta['y'][i].astype(float);m=meta['m'][i];y0=float(meta['y0'][i])
                    r,g,b=(baseline[k][i].astype(float) for k in ('r','gate','background'))
                    oldp=d['oracle_paths'][j]
                    upper,violation,reconstruction=witness(oldp,y,m,y0,r,g,b)
                    p,c=R.project(y,m,y0,r,g,b)
                    if c['lower']>upper+1e-10:raise ValueError('Rebuilt lower bound exceeds preserved feasible upper')
                    c.update(preserved_path_upper=upper,preserved_path_band_violation=violation,preserved_path_reconstruction_residual=reconstruction,
                        path_replay_max_abs=float(np.max(np.abs(p-oldp))))
                    c['resolved_primal_upper']=c['upper'];c['preserved_path_is_tighter_upper']=bool(upper<c['upper'])
                    if upper<c['upper']:c['upper']=upper;p=oldp
                    c['gap']=max(0.,c['upper']-c['lower']);c['gap_certified']=bool(c['gap']<=1e-8)
                    cert.append(c);peaks.append(float(np.max(np.where(m,p,-np.inf))/np.max(np.where(m,y,-np.inf))))
                    if j%20==0:print(f'{arm} fold{args.fold} cache-only oracle certificate {j}/{len(sel)}',flush=True)
                w=meta['w'][sel].astype(float);m=meta['m'][sel];mass=float(np.sum(w[:,None]*m))
                actual=float(np.sum(w[:,None]*np.where(m,(baseline['P'][sel].astype(float)-meta['y'][sel])**2,0.)))
                lo=float(w @ np.array([c['lower'] for c in cert]));hi=float(w @ np.array([c['upper'] for c in cert]))
                res['fixed_downstream_oracle']=dict(S_units=int(len(sel)),design_lower_SSE=lo,design_upper_SSE=hi,actual_SSE=actual,
                    lower_fraction_of_actual_SSE=lo/actual,upper_fraction_of_actual_SSE=hi/actual,RMSE_bracket=[math.sqrt(lo/mass),math.sqrt(hi/mass)],
                    peak_ratio=S.distribution(np.array(peaks),w),maximum_gap=max(c['gap'] for c in cert),
                    unresolved_gap_gt_1e_minus8=int(sum(c['gap']>1e-8 for c in cert)),certificates=cert)
            results[arm][surface]=res
    for name,digest in {**snapshots,**files}.items():
        if S.sha(S.ROOT/name)!=digest:raise ValueError('Artifact changed during recovery '+name)
    report=dict(schema='d10_response_chain_recovered_v2',status='complete',passed=True,fold=args.fold,
        original_registration_commit=REGISTRATION,recovery_commit=full,original_failure=failure,preserved_files_sha256=snapshots,
        verified_original_files_sha256=files,new_forward=False,new_training=False,model_reconstruction=False,
        reused_cached_predictions=True,identical_QP_certificate_reconstruction=True,results=results,seconds=time.monotonic()-started)
    S.write_new(out,report);print(json.dumps(dict(output=S.relative(out),sha256=S.sha(out))),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--recovery-commit',required=True);p.add_argument('--fold',type=int,choices=(2,3),required=True)
    os.nice(max(0,15-os.getpriority(os.PRIO_PROCESS,0)));run(p.parse_args())

if __name__=='__main__':main()
