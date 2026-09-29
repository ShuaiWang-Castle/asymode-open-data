"""Frozen-model D04 local response sensitivities and approximate path support."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key]='2'
import argparse
import gc
import json
from pathlib import Path
import pickle
import numpy as np
from threadpoolctl import threadpool_limits
from d04_data import ROOT,HERE,load_panel,make_rows
from d04_features import REGIMES,transform_weather
from d04_diagnostics import lag_sensitivity,support_proxy
from run_d04_response import DEFAULT_RUN,ARMS,subset,training_rows_and_scale
from report_d04_response import clean,sha,load_oof


def describe(values,rows,mask,weights=None):
    ii=np.flatnonzero(mask)
    if not len(ii):return {'rows':0,'supported':False}
    w=rows['w'][ii] if weights is None else weights[ii]
    w=w/w.sum();x=values[ii].astype('f8')
    mean=np.einsum('n,nbc->bc',w,x)
    rms=np.sqrt(np.einsum('n,nbc,nbc->bc',w,x,x))
    groups,codes=np.unique(rows['group'][ii],return_inverse=True)
    mass=np.bincount(codes,weights=w)
    flat=x.reshape(len(x),36)
    event_means=np.column_stack([np.bincount(codes,weights=w*flat[:,j],minlength=len(groups))
                                for j in range(36)])/mass[:,None]
    event_means=event_means.reshape(-1,3,12)
    county,cidx=np.unique(rows['county'][ii],return_inverse=True)
    cmass=np.bincount(cidx,weights=w)
    county_mean=np.column_stack([np.bincount(cidx,weights=w*flat[:,j],minlength=len(county))
                                for j in range(36)])/cmass[:,None]
    county_mean=county_mean.reshape(-1,3,12)
    return {'rows':len(ii),'counties':len(county),'merged_groups':len(groups),
            'event_weight_kish':1/np.square(mass).sum(),'max_event_weight_share':float(mass.max()),
            'mean':mean,'rms':rms,'county_mean_q10_q50_q90':np.quantile(county_mean,[.1,.5,.9],axis=0),
            'event_mean_min_max':np.stack([np.min(event_means,axis=0),np.max(event_means,axis=0)]),
            'variation_is_not_confidence_interval':True}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    parser.add_argument('--out',type=Path,default=HERE/'results/v1/d04_response_surfaces.json')
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError('Preserve existing surface report')
    os.nice(max(0,15-os.getpriority(os.PRIO_PROCESS,0)))
    with threadpool_limits(limits=2):
        panel=load_panel();wx=transform_weather(panel);rows=make_rows(panel,phase=0)
        all_rows=make_rows(panel,phase=None)
        pred,bundles,_=load_oof(args.run_dir,panel,all_rows)
        del pred,all_rows
        ref=json.loads((HERE/'results/v1/highdim_structure_d03.json').read_text())
        reference_sd=np.asarray(ref['weather_views']['all']['design']['path']['channel_scales'])
        with np.load(ROOT/'runs/geo_weather_20260924/county_structure_d02/county_structure_d02_lookup.npz',allow_pickle=False) as f:
            label_key='type' if 'type' in f.files else 'labels'
            type_lookup=dict(zip(f['county'].astype(str),f[label_key].astype(int)))
        types=np.array([type_lookup[str(c)] for c in rows['county']])
        kinds=[*ARMS,'C_geo_component','D_geo_component']
        values={k:np.full((len(rows['y']),3,12),np.nan,dtype='f4') for k in kinds}
        seen=np.zeros(len(rows['y']),dtype='i1')
        inside=np.zeros(len(rows['y']),dtype=bool);available=inside.copy()
        distance=np.full(len(inside),np.nan);threshold=distance.copy();support_records=[]
        for bundle in bundles:
            fold=bundle['fold'];fm=bundle['feature_map']
            test_idx=np.flatnonzero(rows['fold']==fold);train_idx=np.flatnonzero(rows['fold']!=fold)
            seen[test_idx]+=1
            tr,_=training_rows_and_scale(subset(rows,train_idx));qr=subset(rows,test_idx)
            train_arrays=fm.transform(panel,wx,tr);query_arrays=fm.transform(panel,wx,qr)
            sup=support_proxy(fm,train_arrays,tr,query_arrays,qr)
            inside[test_idx]=sup['within_proxy'];available[test_idx]=sup['support_available']
            distance[test_idx]=sup['distance'];threshold[test_idx]=sup['threshold']
            support_records.append({'fold':fold,'definition':sup['definition'],'counts':sup['support_counts']})
            del train_arrays,query_arrays,tr,qr;gc.collect()
            common_scale=(reference_sd/fm.weather_sd)[None,None,:]
            for start in range(0,len(test_idx),2048):
                idx=test_idx[start:start+2048];qr=subset(rows,idx)
                for arm,model in bundle['models'].items():
                    values[arm][idx]=lag_sensitivity(model,fm,panel,wx,qr,bundle['y_scale'])*common_scale
                    if arm in ('C_geo_main','D_geo_pairs'):
                        key=arm[0]+'_geo_component'
                        values[key][idx]=lag_sensitivity(model,fm,panel,wx,qr,bundle['y_scale'],geo_only=True)*common_scale
            print(json.dumps({'phase':'surface_fold_complete','fold':fold,'rows':len(test_idx),
                              'proxy_available':int(available[test_idx].sum()),
                              'within_proxy':int(inside[test_idx].sum())}),flush=True)
        assert np.all(seen==1) and all(np.isfinite(v).all() for v in values.values())
        summary={}
        for kind,v in values.items():
            summary[kind]={}
            for regime in ['all',*REGIMES]:
                ridx=np.ones(len(rows['y']),dtype=bool) if regime=='all' else rows['regime']==regime
                entry={}
                for domain,dmask in [('all_observed',np.ones(len(ridx),dtype=bool)),('within_proxy',inside&available)]:
                    use=ridx&dmask
                    entry[domain]={'pooled':describe(v,rows,use),
                                   'county_types':{str(t):describe(v,rows,use&(types==t)) for t in range(6)}}
                # Five fitted models are a stability display, not five independent replications.
                entry['fold_means']={str(k):describe(v,rows,ridx&(rows['fold']==k)) for k in range(1,6)}
                summary[kind][regime]=entry
        cache=args.run_dir/'local_response_cache.npz'
        if cache.exists():raise FileExistsError('Preserve derivative cache')
        np.savez_compressed(cache,**values,unit=rows['unit'],time=rows['time'],w=rows['w'],
                            county=rows['county'],group=rows['group'],fold=rows['fold'],regime=rows['regime'],
                            county_type=types,within_proxy=inside,support_available=available,
                            support_distance=distance,support_threshold=threshold)
        result={'meta':{'analysis':'D04 frozen-model local weather-lag sensitivity atlas',
                  'unit':'fraction change per common D03 transformed-weather SD, uniformly shifted over one lag band',
                  'interpretation':'local model sensitivity on observed paths; not a feasible weather intervention or causal impact',
                  'lag_bands_hours':[[1,6],[7,24],[25,48]],'weather_names':panel['feature_names']['weather'],
                  'reference_weather_sd':reference_sd,'report_script_sha256':sha(__file__),
                  'diagnostics_sha256':sha(HERE/'d04_diagnostics.py'),
                  'cache_sha256':sha(cache),'all_sensitivities_finite':True,
                  'geographic_component_is_not_unique_physical_contribution':True,
                  'summary_weights':'means/RMS use original design weights conditional on each slice; all is pooled, not regime-balanced',
                  'spread_weights':'county-mean quantiles weight counties equally; event extrema are unweighted ranges, not intervals',
                  'reference_scale_scope':'D03 full-D outcome-value-blind scale used only to express report units, never for fitting',
                  'no_fitted_surface_confidence_intervals':True},
                'support':{'rows':len(inside),'available':int(available.sum()),'inside':int(inside.sum()),
                           'outside':int((available&~inside).sum()),'unknown':int((~available).sum()),
                           'fold_details':support_records},'responses':summary}
        args.out.write_text(json.dumps(clean(result),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'out':str(args.out.relative_to(ROOT))}))


if __name__=='__main__':main()
