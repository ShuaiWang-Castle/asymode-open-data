"""Validate and summarize frozen D04 OOF net-change predictions; no fitting."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key]='2'
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np
from threadpoolctl import threadpool_limits
from d04_data import ROOT,HERE,load_panel,make_rows,audit_counts
from d04_features import REGIMES
from run_d04_response import DEFAULT_RUN,ARMS,source_hashes


def clean(v):
    if isinstance(v,dict): return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,np.ndarray) and v.ndim==0:return clean(v.item())
    if isinstance(v,(list,tuple,np.ndarray)): return [clean(x) for x in v]
    if isinstance(v,(np.integer,)): return int(v)
    if isinstance(v,(np.floating,float)): return float(v) if np.isfinite(v) else None
    if isinstance(v,(np.bool_,)): return bool(v)
    return v


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_oof(run_dir,panel,rows):
    names=list(ARMS)
    pred=np.full((len(rows['y']),len(names)),np.nan,dtype='f4')
    seen=np.zeros(len(pred),dtype='i1'); bundles=[]; manifests=[]
    expected=source_hashes()
    for k in range(1,6):
        folder=run_dir/f'fold{k:02d}'
        done=json.loads((folder/'DONE.json').read_text())
        assert done['source_hashes']==expected and done['fold']==k
        for name,digest in done['artifact_sha256'].items(): assert sha(folder/name)==digest
        with np.load(folder/f'oof_fold{k}.npz',allow_pickle=False) as f:
            idx=f['row_index']; assert f['arm_names'].tolist()==names
            assert np.array_equal(idx,np.flatnonzero(rows['fold']==k))
            for key in ['unit','time','y','p_prev','w','w_raw','fold','group','county','regime']:
                assert np.array_equal(f[key],rows[key][idx]),(k,key)
            p=f['predictions'] if 'predictions' in f.files else f['pred']
            assert p.shape==(len(idx),len(names)) and np.isfinite(p).all()
            pred[idx]=p;seen[idx]+=1
        with (folder/'bundle.pkl').open('rb') as f: bundle=pickle.load(f)
        assert bundle['source_hashes']==expected
        assert bundle['fold']==k and list(bundle['models'])==names
        bundles.append(bundle);manifests.append(done)
    assert (seen==1).all() and np.isfinite(pred).all()
    return pred,bundles,manifests


def grouped_sums(error,w,reg,labels):
    levels,codes=np.unique(labels,return_inverse=True)
    ids=codes*5+reg
    mass=np.bincount(ids,weights=w,minlength=len(levels)*5).reshape(-1,5)
    sums=np.stack([np.bincount(ids,weights=w*error[:,j],minlength=len(levels)*5)
                   for j in range(error.shape[1])],axis=1).reshape(-1,5,error.shape[1])
    return levels,mass,sums


def statistics(mass,sums):
    # Broadcasting supports one summary or a batch of cluster resamples.
    mse=sums/mass[...,None]
    pooled=sums.sum(axis=-2)/mass.sum(axis=-1)[...,None]
    return {'per_regime_mse':mse,'headline_mse':mse[...,:2,:].mean(axis=-2),
            'all5_mse':mse.mean(axis=-2),'pooled_mse':pooled,'pooled_rmse':np.sqrt(pooled)}


def relative(a,b):
    return 100*(a/b-1) if np.all(np.asarray(b)>0) else np.full_like(a,np.nan)


def ratio_interval(point_a,point_b,draw_a,draw_b):
    pa,pb=np.asarray(point_a),np.asarray(point_b)
    da,db=np.asarray(draw_a),np.asarray(draw_b)
    good=np.isfinite(da)&np.isfinite(db)&(db>0)
    ratio=np.full(da.shape,np.nan,dtype='f8')
    np.divide(da,db,out=ratio,where=good);ratio=100*(ratio-1)
    flat=ratio.reshape(len(ratio),int(np.prod(ratio.shape[1:])) if ratio.ndim>1 else 1)
    ci=np.full((2,flat.shape[1]),np.nan)
    for j in range(flat.shape[1]):
        valid=np.isfinite(flat[:,j])
        if valid.any():ci[:,j]=np.quantile(flat[valid,j],[.025,.975])
    point=np.full(pa.shape,np.nan,dtype='f8')
    np.divide(pa,pb,out=point,where=np.isfinite(pb)&(pb>0));point=100*(point-1)
    return {'change_percent':point,
            'conditional_score_ci95_percent':ci.reshape((2,)+pa.shape),
            'valid_draws':good.sum(axis=0),
            'zero_denominator_draws':(db<=0).sum(axis=0),
            'nonfinite_draws':(~np.isfinite(da)|~np.isfinite(db)).sum(axis=0)}


def score_summary(rows,pred,weight='w',boot=2000):
    names=['zero_change',*ARMS]
    p=np.column_stack([np.zeros(len(pred)),pred]).astype('f8')
    y=rows['y'].astype('f8');err=np.square(p-y[:,None]);w=rows[weight].astype('f8')
    reg=np.array([REGIMES.index(r) for r in rows['regime']],dtype='i1')
    levels,mass,sums=grouped_sums(err,w,reg,rows['group'])
    point=statistics(mass.sum(0),sums.sum(0))
    pairs=[('A_shared_main','zero_change'),('B_shared_pairs','zero_change'),
           ('C_geo_main','zero_change'),('D_geo_pairs','zero_change'),
           ('C_geo_main','A_shared_main'),('D_geo_pairs','B_shared_pairs'),
           ('B_shared_pairs','A_shared_main'),('D_geo_pairs','C_geo_main')]
    comparisons={a+'_vs_'+b:{} for a,b in pairs}
    for cluster in ('merged_event','county'):
        if cluster=='county': _,mm,ss=grouped_sums(err,w,reg,rows['county'])
        else: mm,ss=mass,sums
        rng=np.random.default_rng(20260930 if cluster=='merged_event' else 20260931)
        draws=rng.multinomial(len(mm),np.full(len(mm),1/len(mm)),size=boot)
        bm=draws@mm;bs=(draws@ss.reshape(len(mm),-1)).reshape(boot,5,len(names))
        good=(bm>0).all(1)
        bscores=statistics(bm[good],bs[good])
        for a,b in pairs:
            ia,ib=names.index(a),names.index(b)
            result={}
            for key in ['per_regime_mse','headline_mse','all5_mse','pooled_rmse']:
                result[key]=ratio_interval(point[key][...,ia],point[key][...,ib],
                                           bscores[key][...,ia],bscores[key][...,ib])
                result[key]['missing_regime_draws']=int((~good).sum())
            comparisons[a+'_vs_'+b][cluster]=result
    by_arm={name:{key:(value[...,j].tolist() if np.asarray(value[...,j]).ndim else float(value[...,j]))
                   for key,value in point.items()} for j,name in enumerate(names)}
    clipped_diagnostic=None
    if 'p_prev' in rows:
        prior=rows['p_prev'].astype('f8')
        next_raw=prior[:,None]+p
        clipped_error=np.square(np.clip(next_raw,0,1)-(prior+y)[:,None])
        _,cm,cs=grouped_sums(clipped_error,w,reg,rows['group'])
        cv=statistics(cm.sum(0),cs.sum(0))
        clipped_diagnostic={'definition':'clip(observed p_prev + predicted net change, 0, 1) versus observed next stock; teacher-forced secondary diagnostic',
            'arms':{name:{key:value[...,j] for key,value in cv.items()} for j,name in enumerate(names)},
            'weighted_clipped_fraction':dict(zip(names,np.average((next_raw<0)|(next_raw>1),weights=w,axis=0)))}
    strata={}
    masks={'zero_change':y==0,'positive':y>0,'negative':y<0,
           'positive_at_least_1pp':y>=.01,'negative_at_most_minus1pp':y<=-.01}
    masks.update({f'clock_phase_{k}':(rows['time']-72)%6==k for k in range(6)})
    masks.update({f'fold_{k}':rows['fold']==k for k in range(1,6)})
    if 'unseen_county' in rows:
        masks.update({'unseen_county':rows['unseen_county'],
                      'seen_county':~rows['unseen_county']})
    for label,idx in masks.items():
        if not idx.any():strata[label]={'rows':0};continue
        sw=w[idx].copy()
        if label.startswith('clock_phase_'):
            full_count=np.bincount(rows['unit'])
            sub_count=np.bincount(rows['unit'][idx],minlength=len(full_count))
            sw*=full_count[rows['unit'][idx]]/sub_count[rows['unit'][idx]]
        strata[label]={'rows':int(idx.sum()),'county_events':len(np.unique(rows['unit'][idx])),
            'merged_groups':len(np.unique(rows['group'][idx])),
            'weighting':'county-event mass conserved within phase' if label.startswith('clock_phase_') else 'conditional slice of full-hour weights',
            'mse':dict(zip(names,np.average(err[idx],weights=sw,axis=0))),
            'mae':dict(zip(names,np.average(abs(p[idx]-y[idx,None]),weights=sw,axis=0))),
            'mean_prediction':dict(zip(names,np.average(p[idx],weights=sw,axis=0))),
            'mean_observed_change':float(np.average(y[idx],weights=sw))}
    # Center only for a descriptive within-event/time contrast score, never for prediction fitting.
    event_time=np.char.add(np.char.add(rows['system'].astype(str),':'),rows['time'].astype(str)) if 'system' in rows else None
    within=None
    if event_time is not None:
        _,ci=np.unique(event_time,return_inverse=True);den=np.bincount(ci,weights=w)
        centered=np.empty_like(p)
        cy=y-np.bincount(ci,weights=w*y)[ci]/den[ci]
        for j in range(len(names)):
            centered[:,j]=p[:,j]-np.bincount(ci,weights=w*p[:,j])[ci]/den[ci]
        within=dict(zip(names,np.average(np.square(centered-cy[:,None]),weights=w,axis=0)))
    return dict(weight=weight,arms=by_arm,comparisons=comparisons,strata=strata,
                clipped_next_stock_diagnostic=clipped_diagnostic,
                within_system_time_contrast_mse=within,
                interval_scope='fixed OOF predictions; event and county resampling are separate sensitivities, not joint model-fit uncertainty')


def optimization_summary(bundles):
    output=[]
    for bundle in bundles:
        fold=bundle['fold'];chosen=bundle['chosen'];outer=bundle['outer_records']
        for name,model in bundle['models'].items():
            record=outer[name]
            losses=[s['diagnostics']['loss'] for s in record['starts']]
            tensor=model.u_geo@model.v_geo.T
            singular=np.linalg.svd(tensor,compute_uv=False)[:model.u_geo.shape[1]]
            output.append(dict(fold=fold,arm=name,
                               inner_selected_config=chosen[name]['config'],
                               outer_fit_config=record['starts'][0]['config'],
                               y_scale=bundle['y_scale'],selected_seed=record['selected_seed'],
                               starts=[dict(seed=s['init_seed'],loss=s['diagnostics']['loss'],
                                   gradnorm=s['diagnostics'].get('gradnorm'),
                                   grad_max_abs=s['diagnostics'].get('grad_max_abs'),
                                   gradient_converged=s['diagnostics'].get('gradient_converged'),
                                   finitefit=s['diagnostics']['finitefit'],
                                   convergence=s['diagnostics']['convergence'],
                                   iterations=s['diagnostics']['iterations'],
                                   iteration_budget=s['config']['max_iter']) for s in record['starts']],
                               objective_start_gap=float(max(losses)-min(losses)),
                               objective_units='penalized standardized training objective; not evaluation MSE',
                               geo_rank=model.u_geo.shape[1],
                               geo_tensor_singular_values=singular,
                               singular_value_units='training-standardized response and feature bases; not physical effect strengths',
                               second_start_operator_saved=False))
    return output


def numerical_counts(bundles):
    result={}
    for phase in ('inner','outer'):
        records=[]
        for bundle in bundles:
            for arm in ARMS:
                rr=bundle[phase+'_records'][arm]
                records.extend(rr if phase=='inner' else rr['starts'])
        dd=[r['diagnostics'] for r in records]
        result[phase]={'fits':len(dd),
            'finite':sum(bool(d['finitefit']) for d in dd),
            'objective_improved':sum(bool(d['objective_improved']) for d in dd),
            'gradient_converged':sum(bool(d['gradient_converged']) for d in dd),
            'iteration_budget':sum(d['convergence']=='iteration_budget' for d in dd),
            'stopped_before_budget':sum(d['convergence']=='optimizer_stopped_before_budget' for d in dd),
            'gradient_max_abs_min_max':[min(d['grad_max_abs'] for d in dd),max(d['grad_max_abs'] for d in dd)]}
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    parser.add_argument('--out',type=Path,default=HERE/'results/v1/d04_response_scores.json')
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError('Preserve existing D04 score report')
    os.nice(max(0,15-os.getpriority(os.PRIO_PROCESS,0)))
    with threadpool_limits(limits=2):
        panel=load_panel();rows=make_rows(panel,phase=None);train=make_rows(panel,phase=0)
        pred,bundles,manifests=load_oof(args.run_dir,panel,rows)
        rows['system']=panel['meta']['system'][rows['unit']]
        rows['unseen_county']=np.zeros(len(rows['y']),dtype=bool)
        county_coverage=[]
        for bundle in bundles:
            mask=rows['fold']==bundle['fold']
            levels=bundle['models']['A_shared_main'].county_levels
            assert all(np.array_equal(model.county_levels,levels) for model in bundle['models'].values())
            rows['unseen_county'][mask]=~np.isin(rows['county'][mask],levels)
            unseen=mask&rows['unseen_county']
            county_coverage.append(dict(fold=bundle['fold'],
                heldout_counties=len(np.unique(rows['county'][mask])),
                unseen_counties=len(np.unique(rows['county'][unseen])),
                unseen_county_events=len(np.unique(rows['unit'][unseen])),
                unseen_rows=int(unseen.sum()),
                definition='county absent from this outer model training rows; county intercept is zero'))
        result={'meta':{'analysis':'D04 conditional one-hour net-change OOF statistics',
                   'interpretation':'weather-geography-impact-outage symptoms; no physical mediator identification',
                   'target_unit':'fraction difference; displayed percentage points require multiplication by 100',
                   'source_hashes':source_hashes(),'score_script_sha256':sha(__file__),
                   'fivefold_coverage_exactly_once':True,'all_predictions_finite':True,
                   'no_neural_screen_gate':True},
                'support':audit_counts(train,rows,panel),'county_coverage':county_coverage,
                'optimization':optimization_summary(bundles),
                'numerical_counts':numerical_counts(bundles),
                'design_weights':score_summary(rows,pred,'w'),
                'raw_weights_sensitivity':score_summary(rows,pred,'w_raw',boot=1000)}
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(clean(result),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'out':str(args.out.relative_to(ROOT)), 'outer_rows':len(rows['y'])},ensure_ascii=False))


if __name__=='__main__':main()
