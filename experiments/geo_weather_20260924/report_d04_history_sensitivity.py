"""Descriptive full-D history-control sensitivity; no out-of-fold claims."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key]='2'
import argparse
import json
from pathlib import Path
import pickle
import numpy as np
from threadpoolctl import threadpool_limits
from d04_data import ROOT,HERE,load_panel,make_rows
from d04_features import REGIMES,transform_weather
from d04_models import predict,explain_components
from d04_diagnostics import lag_sensitivity
from run_d04_response import DEFAULT_RUN,ARMS,subset,source_hashes
from report_d04_response import clean,sha
from analyze_d04_response_surfaces import describe


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    parser.add_argument('--out',type=Path,default=HERE/'results/v1/d04_history_sensitivity.json')
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError('Preserve existing history report')
    os.nice(max(0,15-os.getpriority(os.PRIO_PROCESS,0)))
    with threadpool_limits(limits=2):
        panel=load_panel();wx=transform_weather(panel);rows=make_rows(panel,phase=0)
        ref=json.loads((HERE/'results/v1/highdim_structure_d03.json').read_text())
        refsd=np.asarray(ref['weather_views']['all']['design']['path']['channel_scales'])
        output={};pair_jac={}
        for label in ('adjusted','unadjusted'):
            folder=args.run_dir/'history_sensitivity'/label
            done=json.loads((folder/'DONE.json').read_text())
            assert sha(folder/'bundle.pkl')==done['bundle_sha256']
            with (folder/'bundle.pkl').open('rb') as f:bundle=pickle.load(f)
            assert bundle['is_oof'] is False and bundle['history']==(label=='adjusted')
            assert done['source_hashes']==bundle['source_hashes']
            assert done['training_source_hashes']==bundle['training_source_hashes']==source_hashes()
            assert all(sha(ROOT/path)==digest for path,digest in bundle['source_hashes'].items())
            assert bundle['feature_map'].history==bundle['history']
            assert bundle['response']=='adjacent_hour_fraction_change' and bundle['training_phase']==0
            assert bundle['train_rows']==len(rows['y']) and list(bundle['models'])==list(ARMS)
            assert done['selected']==bundle['selected']
            if label=='adjusted': selected_reference=bundle['selected']
            else: assert bundle['selected']==selected_reference
            fm=bundle['feature_map'];ys=bundle['y_scale']
            pred=np.empty((len(rows['y']),4),dtype='f4')
            geo=np.zeros_like(pred)
            jac=np.empty((len(rows['y']),3,12),dtype='f4')
            for start in range(0,len(pred),2048):
                stop=min(len(pred),start+2048);rr=subset(rows,slice(start,stop))
                X,N,G,C=fm.transform(panel,wx,rr)
                for col,arm in enumerate(ARMS):
                    model=bundle['models'][arm];p=len(model.beta_x)
                    pred[start:stop,col]=predict(model,X[:,:p],N,G,C,rr['county'])*ys
                    geo[start:stop,col]=explain_components(model,X[:,:p],N,G,C,rr['county'])['geo']*ys
                jac[start:stop]=lag_sensitivity(bundle['models']['D_geo_pairs'],fm,panel,wx,rr,ys,True)*(refsd/fm.weather_sd)[None,None,:]
            assert np.isfinite(pred).all() and np.isfinite(jac).all()
            regimes={}
            for regime in ['all',*REGIMES]:
                use=np.ones(len(pred),dtype=bool) if regime=='all' else rows['regime']==regime
                w=rows['w'][use];w=w/w.sum()
                regimes[regime]={'rows':int(use.sum()),
                    'in_sample_mse':dict(zip(ARMS,np.average((pred[use]-rows['y'][use,None])**2,axis=0,weights=w))),
                    'geo_component_mean_fraction':dict(zip(ARMS,np.average(geo[use],axis=0,weights=w))),
                    'geo_component_rms_fraction':dict(zip(ARMS,np.sqrt(np.average(geo[use]**2,axis=0,weights=w)))),
                    'D_geo_local_sensitivity':describe(jac,rows,use)}
            output[label]={'history':bundle['history'],'selected':bundle['selected'],'optimization':bundle['records'],
                           'y_scale':ys,'regimes':regimes,'bundle_sha256':sha(folder/'bundle.pkl')}
            pair_jac[label]=jac
            print(json.dumps({'phase':'history_report_complete','history':label}),flush=True)
        difference=pair_jac['unadjusted']-pair_jac['adjusted']
        result={'meta':{'analysis':'D04 full-D fixed-modal-config history sensitivity',
                  'is_oof':False,'descriptive_only':True,'rows':len(rows['y']),
                  'interpretation':'adjustment can absorb past pathways; changes are model associations, not causal mediation',
                  'removed_controls':'unadjusted removes p_prev, p_prev squared, sqrt(p_prev), and p71 only; geography, context, county pooling and all other nuisance terms remain',
                  'summary_weights':'original design weights conditional on slice; all is pooled, not the regime-balanced training objective',
                  'sensitivity_units':'fraction per D03 common transformed-weather SD',
                  'script_sha256':sha(__file__)},'fits':output,
                'unadjusted_minus_adjusted_geo_sensitivity':{r:describe(difference,rows,np.ones(len(difference),dtype=bool) if r=='all' else rows['regime']==r) for r in ['all',*REGIMES]}}
        args.out.write_text(json.dumps(clean(result),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'out':str(args.out.relative_to(ROOT))}))


if __name__=='__main__':main()
