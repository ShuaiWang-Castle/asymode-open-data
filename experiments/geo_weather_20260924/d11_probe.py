"""Registered finite D11 residual information probes; no neural fitting/oracle labels."""
from __future__ import annotations
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[n]='2'
import argparse, gc, hashlib, json, math, resource, subprocess, time
from pathlib import Path
import numpy as np
import torch
import d09_score as S
import d09_train as T
import d10_replay as R
from asymode.gcrk_train import make_batch, _std
import d11_features as K
import d11_ridge as L

OLD_SCOPE=S.HERE/'notes/D09_INCREMENTAL_WRITE_DESIGN_20260929_ZH.md'
JOBS=S.ROOT/'runs/geo_weather_20260924/d09_write_screen_20260929/JOBS.json'
SCOPE=S.HERE/'notes/D11_PATH_GEOGRAPHY_PROBE_SCOPE_20260930_ZH.md'
OWN=('d11_probe.py','d11_features.py','d11_ridge.py')
CONDITIONS=('R_shared','R_real','R_permuted','H_shared','H_real','H_permuted')
PENALTIES=(1e-4,1e-2,1.)
COMPARISONS=tuple((c+'_vs_closed','closed',c) for c in CONDITIONS)+tuple(
    ('R_'+g+'_vs_H_'+g,'H_'+g,'R_'+g) for g in ('shared','real','permuted'))+tuple(
    (p+'_real_vs_'+g,p+'_'+g,p+'_real') for p in ('R','H') for g in ('shared','permuted'))

def register(commit):
    full=subprocess.check_output(['git','rev-parse',commit+'^{commit}'],cwd=S.ROOT,text=True).strip()
    hashes={}
    for path in [S.HERE/n for n in OWN]+[SCOPE]:
        digest=hashlib.sha256(S.git_bytes(full,path)).hexdigest()
        if digest!=S.sha(path): raise ValueError('D11 registered source changed: '+S.relative(path))
        hashes[S.relative(path)]=digest
    return full,hashes

def state_hash(model):
    h=hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):
        h.update(key.encode())
        if isinstance(value,torch.Tensor):
            a=value.detach().cpu().numpy();h.update(str((a.shape,a.dtype)).encode());h.update(a.tobytes())
        else:h.update(json.dumps(value,sort_keys=True,default=str).encode())
    return h.hexdigest()

def save_npz(path,values):
    with Path(path).open('xb') as stream:np.savez_compressed(stream,**values)

def log(phase,**extra):
    print(json.dumps(S.strict_json(dict(phase=phase,**extra)),ensure_ascii=False,allow_nan=False),flush=True)

def model_record(solution):
    moments=solution.moments
    return dict(penalty=solution.penalty,coefficient=solution.coefficient,intercept=solution.intercept,
        standardized_coefficient=solution.standardized_coefficient,condition_number=solution.condition_number,
        relative_normal_equation_residual=solution.relative_normal_equation_residual,
        effective_df=solution.effective_df_including_intercept,feature_mean=moments.feature_mean,
        feature_scale=moments.feature_scale,constant_features=np.flatnonzero(moments.constant_features),
        design_hour_mass=moments.design_hour_mass)

def preactivation_decomposition(models,stats,F,rows):
    z=[_std(F['xu'][rows],s['xu'],14).astype(float) for s in stats]
    c=[_std(F['xr'][rows],s['xr'],20)[:,0,14:20].astype(float) for s in stats]
    W=[m.damage[0].weight.detach().numpy().astype(float) for m in models]
    A=[m.ctx_in.weight.detach().numpy().astype(float) for m in models]
    b=[m.damage[0].bias.detach().numpy().astype(float) for m in models]
    return decompose(z,c,W,A,b)

def decompose(z,c,W,A,b):
    pre=(z[1]-z[0])@((W[0]+W[1])/2).T+((c[1]-c[0])@((A[0]+A[1])/2).T)[:,None,:]
    learned=((z[1]+z[0])/2)@(W[1]-W[0]).T+(((c[1]+c[0])/2)@(A[1]-A[0]).T)[:,None,:]+b[1]-b[0]
    a=[zz@ww.T+(cc@aa.T)[:,None,:]+bb for zz,cc,ww,aa,bb in zip(z,c,W,A,b)]
    difference=a[1]-a[0]
    error=float(np.max(np.abs(pre+learned-difference)))
    if error>1e-9*max(1,float(np.max(np.abs(difference)))):raise ValueError('Preactivation accounting identity failed')
    return dict(preprocess=pre,learned=learned,total=difference,a2=a[0],a3=a[1],identity_max_abs=error,
                xu_delta=z[1]-z[0],context_delta=c[1]-c[0])

def drift_summary(data,models,stats,rows):
    F=data['F'];d=preactivation_decomposition(models,stats,F,rows)
    w=F['w'][rows].astype(float);m=F['m'][rows].astype(bool);ph=S.cohort_masks(S.phenotype(F['y'][rows].astype(float),m,F['y0'][rows].astype(float)))
    result=dict(identity_max_abs=d['identity_max_abs'],rows=len(rows),unit_ids=data['unit'][rows].tolist(),cohorts={})
    for name,keep in [('all',np.ones(len(rows),bool)),('S',ph['S']),('nonS',~ph['S'])]:
        weight=w*keep;mass=weight.sum()*144*32
        pre=d['preprocess'][:,72:];learn=d['learned'][:,72:];total=d['total'][:,72:]
        sq=lambda a:float(np.sum(weight[:,None,None]*a*a)/mass) if mass else None
        cross=float(np.sum(2*weight[:,None,None]*pre*learn)/mass) if mass else None
        vals={k:sq(v) for k,v in [('preprocess_MSE',pre),('learned_MSE',learn),('total_MSE',total)]}
        if mass and not np.isclose(vals['preprocess_MSE']+vals['learned_MSE']+cross,vals['total_MSE'],rtol=1e-10,atol=1e-10):raise ValueError('Drift cross-term accounting')
        result['cohorts'][name]=dict(n=int(keep.sum()),design_kish=float(weight.sum()**2/(weight@weight)) if np.any(weight) else 0,
            **vals,cross_term=cross,preprocess_RMS=None if not mass else math.sqrt(vals['preprocess_MSE']),
            learned_RMS=None if not mass else math.sqrt(vals['learned_MSE']),total_RMS=None if not mass else math.sqrt(vals['total_MSE']),
            relu_hidden_RMS=None if not mass else math.sqrt(sq(np.maximum(d['a3'][:,72:],0)-np.maximum(d['a2'][:,72:],0))))
    result['standardized_input_drift']={}
    for key,arr in [('xu',d['xu_delta'][:,72:]),('context',d['context_delta']),('geo',_std(F['geo'][rows],stats[1]['geo'],0).astype(float)-_std(F['geo'][rows],stats[0]['geo'],0).astype(float))]:
        axes=tuple(range(1,arr.ndim));per=np.mean(arr*arr,axis=axes)
        result['standardized_input_drift'][key]=dict(weighted_RMS=float(np.sqrt(w@per/w.sum())),coordinate_RMS=np.sqrt(np.mean(arr*arr,axis=tuple(range(arr.ndim-1)))).tolist())
    weights=[]
    for fold in (2,3):
        _,info=T.design_weighted(F,T.job_indices(data,fold)['fit']);weights.append(info)
    result['D09_fit_weighting']=weights
    result['common_relative_optimization_weight_drift']={r:dict(
        inverse_Z_ratio_fold3_over_fold2=weights[0]['Z_regime'][r]/weights[1]['Z_regime'][r],
        actual_m_train_ratio_fold3_over_fold2=(weights[0]['Z_regime'][r]/weights[1]['Z_regime'][r])*(weights[1]['mean_observed_cell_weight_scale']/weights[0]['mean_observed_cell_weight_scale']))
        for r in weights[0]['Z_regime'] if weights[0]['Z_regime'][r]>0 and weights[1]['Z_regime'][r]>0}
    result['frozen_kernel_geometry']=[dict(level_scale=m.kernel.level_scale.detach().numpy().tolist(),departure_scale=m.kernel.departure_scale.detach().numpy().tolist(),
        geo_center=m.kernel.geo_center.detach().numpy().tolist(),geo_rms_scale=m.kernel.geo_rms_scale.detach().numpy().tolist(),
        landmarks_sha256=hashlib.sha256(m.kernel.landmarks.detach().numpy().tobytes()).hexdigest(),fit_metadata=m.kernel.fit_metadata) for m in models]
    result['interpretation']='Exact pre-ReLU two-component accounting with cross term; nonlinear output drift is descriptive, no causal attribution. Relative w/Zr appears in D09_fit_weighting.'
    return result

def run(args):
    commit,own_hash=register(args.registration_commit)
    if args.out.exists():raise FileExistsError('Preserve existing folder, including empty')
    args.out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    try:
        _,dependency_hashes=R.register('1ca63c4',S.HERE/'notes/D10_RESPONSE_CHAIN_SCOPE_20260929_ZH.md')
        old=S.verify_registration(R.OLD_COMMIT,OLD_SCOPE)
        meta,predictions,before,provenance=S.read_verified_jobs(JOBS,OLD_SCOPE,old)
        before.update(dependency_hashes)
        log('six_DONE_verified',fold=args.fold)
        data=T.load_data();F=data['F'];ix=T.job_indices(data,args.fold);fit=ix['fit'];held=ix['held_full']
        common=np.intersect1d(T.job_indices(data,2)['fit'],T.job_indices(data,3)['fit'])
        folder=S.ROOT/f'runs/geo_weather_20260924/d09_write_screen_20260929/new_f{args.fold}'
        model,stats=R.rebuild(data,folder,'new',args.fold);fingerprint=state_hash(model)
        selected=np.flatnonzero(meta['original_fold']==args.fold)
        if not np.array_equal(meta['idx'][selected],data['unit'][held]):raise ValueError('Held metadata identity')
        closed=predictions['new_closed'][selected].astype(float)
        with np.load(folder/'step0900_fit_panel.npz',allow_pickle=False) as a:
            if not np.array_equal(a['idx'],data['unit'][fit]):raise ValueError('FIT export identity')
            fitclosed=a['P_closed'].astype(float)
            fitmeta={k:a[k].copy() for k in S.META_KEYS}
            for key in ('y','m','w','y0'):
                if not np.array_equal(a[key],F[key][fit]):raise ValueError('FIT data changed '+key)
        Pclosed=np.zeros((len(F['y']),144));Pclosed[fit]=fitclosed;Pclosed[held]=closed
        hidden=np.zeros((len(F['y']),216,32),np.float32)
        with torch.no_grad():
            needed=np.union1d(fit,held)
            for s in range(0,len(needed),64):
                ii=needed[s:s+64];bb=make_batch(F,ii,stats)
                hidden[ii]=model.hidden(bb['xu'],bb['ctx']).numpy()
        if state_hash(model)!=fingerprint:raise ValueError('Frozen carrier changed during hidden extraction')
        inputs=dict(raw_path=F['xu'][...,:12],hidden_path=hidden,xu=F['xu'],context=F['xr'][:,0,14:20],y0=F['y0'],geo=F['geo'],
                    fips=F['fips'],unit_ids=data['unit'],groups=data['group'],w=F['w'])
        cv={c:[] for c in CONDITIONS};cvgeometry=[];solves=0
        for vf in sorted(np.unique(data['fold'][fit]).tolist()):
            train=fit[data['fold'][fit]!=vf];valid=fit[data['fold'][fit]==vf]
            if set(data['group'][train])&set(data['group'][valid]):raise ValueError('CV group leakage')
            family=K.fit_family(**inputs,fit_indices=train)
            xt=family.transform(train);xv=family.transform(valid)
            for c in CONDITIONS:
                moments=L.weighted_moments(xt[c],F['y'][train],Pclosed[train],F['m'][train],F['w'][train],scale_floor=1e-4)
                scores=[]
                for penalty in PENALTIES:
                    solution=L.fit_from_moments(moments,penalty);delta=L.predict_delta(solution,xv[c])
                    scores.append(dict(penalty=penalty,score=L.masked_residual_score(F['y'][valid],Pclosed[valid],delta,F['m'][valid],F['w'][valid]),fit=model_record(solution)))
                    solves+=1
                cv[c].append(dict(validation_original_fold=int(vf),train_unit_ids=data['unit'][train].tolist(),validation_unit_ids=data['unit'][valid].tolist(),scores=scores))
            cvgeometry.append(dict(validation_original_fold=int(vf),geometry=family.metadata(),query=family.diagnostics(valid)))
            del family,xt,xv;gc.collect();log('CV_complete',fold=args.fold,validation_fold=vf,solves=solves)
        chosen={}
        for c in CONDITIONS:
            scores=[sum(entry['scores'][j]['score']['candidate_SSE'] for entry in cv[c]) for j in range(3)]
            j=min(range(3),key=lambda k:(scores[k],-PENALTIES[k]));chosen[c]=dict(penalty=PENALTIES[j],validation_weighted_SSE=scores,rule='minimum pooled raw-w masked SSE; exact tie larger lambda')
        family=K.fit_family(**inputs,fit_indices=fit)
        output={k:meta[k][selected] for k in S.META_KEYS};output['closed']=closed;output['W']=predictions['host'][selected].astype(float)
        fitoutput=dict(fitmeta,closed=fitclosed)
        models={};fitscores={}
        final_fit_features=family.transform(fit)
        final_held_features=family.transform(held)
        for c in CONDITIONS:
            xt=final_fit_features.pop(c)
            moments=L.weighted_moments(xt,F['y'][fit],fitclosed,F['m'][fit],F['w'][fit],scale_floor=1e-4)
            solution=L.fit_from_moments(moments,chosen[c]['penalty']);solves+=1
            dfit=L.predict_delta(solution,xt);fitoutput[c]=fitclosed+dfit
            fitscores[c]=L.masked_residual_score(F['y'][fit],fitclosed,dfit,F['m'][fit],F['w'][fit])
            fitscores[c]['cohorts']=T.score(F,fitoutput[c],fit)
            fph=S.phenotype(F['y'][fit].astype(float),F['m'][fit].astype(bool),F['y0'][fit].astype(float));fm=S.cohort_masks(fph)['S']
            fp=S.model_peak(fitoutput[c],F['m'][fit].astype(bool),fph)
            fitscores[c]['cohorts']['S'].pop('peak_ratio_design_weight_median',None)
            fitscores[c]['cohorts']['S']['peak_ratio']=S.distribution(fp['peak_ratio'][fm],F['w'][fit][fm].astype(float))
            del xt;gc.collect()
            xh=final_held_features.pop(c);delta=L.predict_delta(solution,xh);output[c]=closed+delta
            models[c]=model_record(solution);del xh;gc.collect();log('endpoint_complete',fold=args.fold,condition=c,lambda_value=chosen[c]['penalty'])
        if solves!=60:raise ValueError('Finite solve budget mismatch')
        drift=None
        if args.fold==2:
            other,otherstats=R.rebuild(data,S.ROOT/'runs/geo_weather_20260924/d09_write_screen_20260929/new_f3','new',3)
            drift=drift_summary(data,[model,other],[stats,otherstats],common)
        if state_hash(model)!=fingerprint:raise ValueError('Frozen model changed')
        if any(S.sha(S.under_root(p))!=h for p,h in before.items()):raise ValueError('Frozen input/source/output changed')
        register(commit)
        for a in [*output.values(),*fitoutput.values()]:
            if np.asarray(a).dtype.kind in 'fc' and not np.isfinite(a).all():raise ValueError('Nonfinite export')
        save_npz(args.out/'held.npz',output);save_npz(args.out/'fit.npz',fitoutput)
        result=dict(schema='d11_fold_v1',registration_commit=commit,source_sha256=own_hash,dependency_sha256=dependency_hashes,fold=args.fold,provenance=provenance,
            carrier_state_sha256_before=fingerprint,carrier_state_sha256_after=state_hash(model),fit_n=len(fit),held_n=len(held),common_n=len(common),
            fit_unit_ids=data['unit'][fit].tolist(),held_unit_ids=data['unit'][held].tolist(),solve_count=solves,penalties=list(PENALTIES),CV=cv,chosen=chosen,
            models=models,FIT_residual_scores=fitscores,CV_input_geometry=cvgeometry,final_geometry=family.metadata(),
            held_support=family.diagnostics(held),FIT_support=family.diagnostics(fit),common_support=family.diagnostics(common),preprocessing_drift=drift,
            resources=dict(pid=os.getpid(),nice=os.getpriority(os.PRIO_PROCESS,0),threads=2,wall_seconds=time.monotonic()-start,
                peak_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**30))
        S.write_new(args.out/'RESULT.json',result)
        S.write_new(args.out/'DONE.json',dict(status='complete',registration_commit=commit,fold=args.fold,outputs={p.name:S.sha(p) for p in [args.out/'RESULT.json',args.out/'held.npz',args.out/'fit.npz']}))
        log('DONE',fold=args.fold,seconds=time.monotonic()-start)
    except BaseException as error:
        S.write_new(args.out/'FAILED.json',dict(error=repr(error),seconds=time.monotonic()-start,pid=os.getpid()))
        raise

def array_load(path):
    with np.load(path,allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}

def collect(out):
    folds=[];results=[];frozen_hashes={}
    for f in (2,3):
        folder=out/f'fold{f}';done=json.loads((folder/'DONE.json').read_text())
        if done['status']!='complete' or done['fold']!=f:raise ValueError('Incomplete fold')
        for n,digest in done['outputs'].items():
            if S.sha(folder/n)!=digest:raise ValueError('D11 output changed')
        result=json.loads((folder/'RESULT.json').read_text());register(result['registration_commit'])
        if done['registration_commit']!=result['registration_commit']:raise ValueError('Receipt registration mismatch')
        results.append(result);frozen_hashes.update(result['provenance']['files_sha256']);frozen_hashes.update(result['dependency_sha256'])
    if any(S.sha(S.under_root(p))!=h for p,h in frozen_hashes.items()):raise ValueError('Frozen dependency/source/data changed before score')
    R.register('1ca63c4',S.HERE/'notes/D10_RESPONSE_CHAIN_SCOPE_20260929_ZH.md')
    old=S.verify_registration(R.OLD_COMMIT,OLD_SCOPE)
    original,old_predictions,_,_=S.read_verified_jobs(JOBS,OLD_SCOPE,old)
    for f,result in zip((2,3),results):
        a=array_load(out/f'fold{f}'/'held.npz')
        if not np.array_equal(a['idx'],result['held_unit_ids']) or not np.all(a['original_fold']==f):raise ValueError('Actual export identity mismatch')
        choose=original['original_fold']==f
        S.assert_same_metadata(a,{k:v[choose] for k,v in original.items()})
        if not np.array_equal(a['closed'],old_predictions['new_closed'][choose].astype(float)) or not np.array_equal(a['W'],old_predictions['host'][choose].astype(float)):
            raise ValueError('Carrier references changed')
        folds.append(a)
    data={k:np.concatenate([a[k] for a in folds]) for k in S.META_KEYS}
    for key in ('y','w','y0','pi'):data[key]=data[key].astype(float)
    predictions={c:np.concatenate([a[c].astype(float) for a in folds]) for c in ('closed','W',*CONDITIONS)}
    if len(np.unique(data['idx']))!=3869 or len(data['idx'])!=3869:raise ValueError('Exactly once held coverage')
    return data,predictions,results

def score(args):
    commit,_=register(args.registration_commit);data,predictions,results=collect(args.out)
    if any(r['registration_commit']!=commit for r in results):raise ValueError('Fold registration mismatch')
    S.COMPARISONS=COMPARISONS
    surface=S.score_surface(data,predictions)
    ph=S.cohort_masks(S.phenotype(data['y'],data['m'],data['y0']));m=data['m'];w=data['w'];r=data['y']-predictions['closed']
    concentration={};range_checks={}
    for c in CONDITIONS:
        delta=predictions[c]-predictions['closed'];rowgain=np.sum(np.where(m,w[:,None]*(2*r*delta-delta**2),0),1)
        concentration[c]={}
        for label,keep in [('all',np.ones(len(w),bool)),('S',ph['S']),('nonS',~ph['S'])]:
            levels=np.unique(data['merged_group'][keep]);entries=[]
            for g in levels:
                ii=keep&(data['merged_group']==g)
                entries.append(dict(group=str(g),n=int(ii.sum()),weighted_gain=float(rowgain[ii].sum()),design_mass=float(w[ii].sum())))
            entries.sort(key=lambda x:abs(x['weighted_gain']),reverse=True)
            concentration[c][label]=entries
        invalid=(predictions[c]<0)|(predictions[c]>1)
        range_checks[c]=dict(cells=int(invalid.sum()),units=int(invalid.any(1).sum()),observed_cells=int((invalid&m).sum()),design_observed_invalid_mass=float(np.sum(w[:,None]*m*invalid)),minimum=float(predictions[c].min()),maximum=float(predictions[c].max()))
    stability={}
    fitarchives=[array_load(args.out/f'fold{f}'/'fit.npz') for f in (2,3)]
    common=np.intersect1d(fitarchives[0]['idx'],fitarchives[1]['idx'])
    sub=[]
    for a in fitarchives:
        positions=np.searchsorted(a['idx'],common)
        if not np.array_equal(a['idx'][positions],common):raise ValueError('Common fit identity')
        sub.append({k:v[positions] for k,v in a.items()})
    if len(common)!=535:raise ValueError('Common count')
    yy=sub[0]['y'].astype(float);mm=sub[0]['m'].astype(bool);ww=sub[0]['w'].astype(float);cp=S.phenotype(yy,mm,sub[0]['y0']);cm=S.cohort_masks(cp)
    for key in ('y','m','w','y0'):
        if not np.array_equal(sub[0][key],sub[1][key]):raise ValueError('Common metadata changed')
    for c in CONDITIONS:
        pp=[a[c].astype(float) for a in sub];dd=[pp[j]-sub[j]['closed'].astype(float) for j in range(2)]
        gains=[np.sum(np.where(mm,ww[:,None]*(2*(yy-sub[j]['closed'].astype(float))*dd[j]-dd[j]**2),0),1) for j in range(2)]
        stability[c]={}
        for label,keep in [('all',np.ones(len(ww),bool)),('S',cm['S']),('nonS',~cm['S'])]:
            mass=float(np.sum(ww[keep,None]*mm[keep]));pk=[S.model_peak(p,mm,cp) for p in pp]
            stability[c][label]=dict(n=int(keep.sum()),prediction_RMS=math.sqrt(float(np.sum(ww[keep,None]*mm[keep]*(pp[1][keep]-pp[0][keep])**2))/mass) if mass else None,
                correction_RMS=math.sqrt(float(np.sum(ww[keep,None]*mm[keep]*(dd[1][keep]-dd[0][keep])**2))/mass) if mass else None,
                benefit_counts=[int(np.sum(keep&(g>0))) for g in gains],benefit_design_mass=[float(ww[keep&(g>0)].sum()) for g in gains],
                peak_ratio=[S.distribution(p['peak_ratio'][keep],ww[keep]) for p in pk])
    def supported(comparison):
        pooled=surface['rows']['pooled']['S']['comparisons'][comparison]
        fold_values=[surface['rows'][f'fold{f}']['S']['comparisons'][comparison]['relative_RMSE_reduction'] for f in (2,3)]
        interval=pooled['bootstrap']['merged_group']
        point=pooled['relative_RMSE_reduction'];ci=interval['relative_RMSE_reduction_ci95']
        passed=point is not None and point>0 and all(v is not None and v>0 for v in fold_values) and interval['interval_supported'] and ci is not None and ci[0]>0
        return dict(pooled=point,folds=fold_values,bootstrap=interval,passed=bool(passed))
    result=dict(schema='d11_scores_v1',registration_commit=commit,primary=surface,range_checks=range_checks,
                event_gain_concentration=concentration,common_FIT_response_stability=stability,
                raw_path_evidence=supported('R_shared_vs_H_shared'),geography_evidence={k:supported(k) for k in ('R_real_vs_shared','R_real_vs_permuted')},
                automatic_neural_training=False,original_full_D_S_target=.1,
                interpretation='Residual probes are not legal stock trajectories or independent confirmation. FIT-only geography donor randomization is one diagnostic, not causal null.')
    S.write_new(args.out/'SCORES.json',result);log('scoring_DONE')

def audit(args):
    commit,own=register(args.registration_commit);data,preds,results=collect(args.out)
    scores=json.loads((args.out/'SCORES.json').read_text());checks=0;maximum=0.
    if scores['registration_commit']!=commit:raise ValueError('Scoring registration')
    for result in results:
        assert result['solve_count']==60 and result['carrier_state_sha256_before']==result['carrier_state_sha256_after'];checks+=2
        for c in CONDITIONS:
            curve=[sum(e['scores'][j]['score']['candidate_SSE'] for e in result['CV'][c]) for j in range(3)]
            j=min(range(3),key=lambda k:(curve[k],-PENALTIES[k]))
            assert result['chosen'][c]['penalty']==PENALTIES[j];checks+=1
    for fold,result in zip((2,3),results):
        actual=array_load(args.out/f'fold{fold}'/'fit.npz')
        original_fit=array_load(S.ROOT/f'runs/geo_weather_20260924/d09_write_screen_20260929/new_f{fold}/step0900_fit_panel.npz')
        S.assert_same_metadata(actual,original_fit);checks+=len(S.META_KEYS)
        assert np.array_equal(actual['closed'],original_fit['P_closed'].astype(float));checks+=1
    ph=S.phenotype(data['y'],data['m'],data['y0']);cohorts=S.cohort_masks(ph)
    risk_rows={k:np.sum(np.where(data['m'],data['w'][:,None]*(p-data['y'])**2,0),1) for k,p in preds.items()}
    plans={}
    for label,key in [('merged_group','merged_group'),('family_sensitivity','family')]:
        levels,code=np.unique(data[key],return_inverse=True);regime_index=np.array([S.REGIMES.index(r) for r in data['regime']])
        stratum_mass=np.zeros((len(levels),5));np.add.at(stratum_mass,(code,regime_index),data['w'])
        strata=stratum_mass.argmax(1);counts=np.zeros((1999,len(levels)),int);rng=np.random.default_rng(20260929)
        for j in range(5):
            group_indices=np.flatnonzero(strata==j)
            if len(group_indices):counts[:,group_indices]=rng.multinomial(len(group_indices),np.full(len(group_indices),1/len(group_indices)),size=1999)
        plans[label]=(code,counts,len(levels))
    for stratum,rows in scores['primary']['rows'].items():
        if stratum=='pooled':sel=np.ones(len(data['idx']),bool)
        elif stratum.startswith('fold'):
            parts=stratum.split('/');sel=data['original_fold']==int(parts[0][4:])
            if len(parts)>1:sel&=data['regime']==parts[2]
        else:sel=data['regime']==stratum.split('/')[1]
        for cohort,row in rows.items():
            keep=sel&cohorts[cohort];y=data['y'][keep];m=data['m'][keep];w=data['w'][keep,None]
            for name,P in preds.items():
                actual=float(np.sum(np.where(m,w*(P[keep]-y)**2,0)));saved=row['models'][name]['design_SSE']
                error=abs(actual-saved);maximum=max(maximum,error)
                assert np.isclose(actual,saved,rtol=1e-10,atol=2e-9);checks+=1
            for comp,base,candidate in COMPARISONS:
                delta=preds[candidate][keep]-preds[base][keep];residual=y-preds[base][keep]
                gain=float(np.sum(np.where(m,w*(2*residual*delta-delta*delta),0)))
                exact=row['models'][base]['design_SSE']-row['models'][candidate]['design_SSE']
                assert np.isclose(gain,exact,rtol=1e-9,atol=2e-9);checks+=1
                aligned=float(np.sum(np.where(m,w*2*residual*delta,0)));energy=float(np.sum(np.where(m,w*delta*delta,0)))
                saved_gain=row['comparisons'][comp]['paired_gain']
                assert np.isclose(aligned,saved_gain['alignment'],rtol=1e-10,atol=2e-9);checks+=1
                assert np.isclose(energy,saved_gain['modification_energy'],rtol=1e-10,atol=2e-9);checks+=1
                bsse=row['models'][base]['design_SSE'];csse=row['models'][candidate]['design_SSE']
                expected=None if bsse==0 else 1-math.sqrt(csse/bsse)
                saved=row['comparisons'][comp]['relative_RMSE_reduction']
                assert (expected is None and saved is None) or np.isclose(expected,saved,rtol=1e-10,atol=1e-12);checks+=1
                for label,(code,counts,n) in plans.items():
                    br=risk_rows[base];cr=risk_rows[candidate]
                    b=counts @ np.bincount(code,weights=np.where(keep,br,0),minlength=n)
                    c=counts @ np.bincount(code,weights=np.where(keep,cr,0),minlength=n)
                    valid=(b>0)&np.isfinite(b)&np.isfinite(c)&(c>=0)
                    calculated=np.quantile(1-np.sqrt(c[valid]/b[valid]),[.025,.975]) if valid.any() else None
                    saved=row['comparisons'][comp]['bootstrap'][label]['empirical_relative_RMSE_reduction_percentiles95']
                    assert (calculated is None and saved is None) or np.allclose(calculated,saved,rtol=1e-10,atol=1e-12);checks+=1
    for label,evidence in [('R_shared_vs_H_shared',scores['raw_path_evidence']),*scores['geography_evidence'].items()]:
        row=scores['primary']['rows']['pooled']['S']['comparisons'][label];ci=row['bootstrap']['merged_group']['relative_RMSE_reduction_ci95']
        fold_values=[scores['primary']['rows'][f'fold{f}']['S']['comparisons'][label]['relative_RMSE_reduction'] for f in (2,3)]
        expected=row['relative_RMSE_reduction'] is not None and row['relative_RMSE_reduction']>0 and all(v is not None and v>0 for v in fold_values) and row['bootstrap']['merged_group']['interval_supported'] and ci is not None and ci[0]>0
        assert evidence['passed']==expected;checks+=1
    record=dict(schema='d11_independent_arithmetic_audit_v1',registration_commit=commit,assertions=checks,
        source_sha256=own,held_coverage=3869,S_count=int(cohorts['S'].sum()),maximum_SSE_absolute_difference=maximum,
        outputs={S.relative(p):S.sha(p) for p in [args.out/'SCORES.json',*[args.out/f'fold{f}'/n for f in (2,3) for n in ('held.npz','fit.npz','RESULT.json','DONE.json')]]})
    S.write_new(args.out/'AUDIT.json',record);log('audit_DONE',assertions=checks)

def self_check():
    rng=np.random.default_rng(20260930);z=[rng.normal(size=(5,9,4)) for _ in range(2)];c=[rng.normal(size=(5,3)) for _ in range(2)]
    W=[rng.normal(size=(2,4)) for _ in range(2)];A=[rng.normal(size=(2,3)) for _ in range(2)];b=[rng.normal(size=2) for _ in range(2)]
    assert decompose(z,c,W,A,b)['identity_max_abs']<1e-13
    assert len(CONDITIONS)==6 and len(COMPARISONS)==13
    assert min(range(3),key=lambda k:([1,1,1][k],-PENALTIES[k]))==2
    return dict(root_assertions=3,ridge_assertions=L.self_check(),feature_assertions=K.self_check())

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('run','score','audit','self-check'))
    p.add_argument('--registration-commit');p.add_argument('--fold',type=int,choices=(2,3));p.add_argument('--out',type=Path)
    args=p.parse_args();os.nice(max(0,15-os.getpriority(os.PRIO_PROCESS,0)));torch.set_num_threads(2);torch.set_num_interop_threads(2)
    if args.mode=='self-check':print(json.dumps(dict(assertions=self_check())));return
    if not args.registration_commit or not args.out:p.error('registration commit and output required')
    args.out=S.under_root(args.out)
    if args.mode=='run' and not args.fold:p.error('run fold required')
    {'run':run,'score':score,'audit':audit}[args.mode](args)
if __name__=='__main__':main()
