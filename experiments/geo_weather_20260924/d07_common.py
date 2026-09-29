"""Frozen public-D access and scoring for the predeclared D07 diagnostics."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='2'
import json
from pathlib import Path
import sys
import numpy as np
import torch
from evaluate_cr_tail import (ROOT, HERE, RUNS, FEATURES, REGIMES, sha, clean, write_new,
                             load_outcomes, load_predictions, cohorts, bootstrap_plan)
from audit_d06_response import DATA_SHA, weighted_quantile
sys.path.insert(0,str(ROOT/'src'))
from asymode.asym_host import AsymODE
from asymode.gcrk_train import make_batch, _std
from asymode.controlled_relaxation import ControlledRelaxationLayer

OUT=RUNS/'d07_selectivity_20260929'
SCOPE_COMMIT='8f8ddef'
_PROVENANCE=None


def provenance():
    global _PROVENANCE
    if _PROVENANCE is None:
        assert sha(FEATURES)==DATA_SHA
        old=json.loads((HERE/'results/v1/i20_final_audit_s0.json').read_text())
        for name,h in old['source_sha256'].items(): assert sha(ROOT/name)==h,name
        for label,folds in old['exports'].items():
            for k,files in folds.items():
                for name,h in files.items():
                    assert sha(RUNS/label/f'fold{int(k):02d}'/name)==h,(label,k,name)
        _PROVENANCE=dict(scope_commit=SCOPE_COMMIT,data_sha256=DATA_SHA,
            frozen_source_sha256=old['source_sha256'],frozen_artifact_sha256=old['exports'],
            common_source_sha256=sha(__file__),previous_audit_sha256=sha(HERE/'results/v1/i20_final_audit_s0.json'))
    return _PROVENANCE


def load_data(load_inputs=False):
    provenance();data=load_outcomes()
    F=None
    if load_inputs:
        with np.load(FEATURES,allow_pickle=False) as z:
            F={k:z[k].copy() for k in ('xu','xr','xo','geo','fips','y','m','y0')}
        # Raw covariates may contain documented missing values; frozen fit
        # standardization handles them. Outcomes/masks remain authoritative.
        assert all(np.isfinite(F[k]).all() for k in ('y','y0'))
    return data,F


def load_model(F,data,label,fold=1):
    assert label in ('v1_host_s0','v1_crk_s0') and fold in range(1,6)
    provenance();folder=RUNS/label/f'fold{fold:02d}';path=folder/'final.pt'
    ck=torch.load(path,map_location='cpu',weights_only=False)
    assert ck['steps']==900 and ck['arm']==('CRK+Cin' if label=='v1_crk_s0' else 'W+Cin')
    model=AsymODE(F['xu'].shape[-1],F['xr'].shape[-1],F['xo'].shape[-1]);model.attach_context_input(6)
    if label=='v1_crk_s0':
        fit=np.sort(np.asarray(data['split']['event'][str(fold)]['dev']))
        g=torch.from_numpy(_std(F['geo'][fit],ck['stats']['geo'],0))
        model.damage[2]=ControlledRelaxationLayer(model.damage[2],g,county_ids=F['fips'][fit],
                        checkpoint_steps=0,use_adjoint=True,geo_checkpoint=True)
    model.load_state_dict(ck['model_state'],strict=True);model.eval()
    assert all(bool(torch.isfinite(v).all()) for v in model.state_dict().values() if isinstance(v,torch.Tensor) and v.is_floating_point())
    return model,ck['stats'],dict(label=label,fold=fold,checkpoint_sha256=sha(path),outer_sha256=sha(folder/'outer.npz'))


def tensor_state(model):
    return {k:v.detach().clone() for k,v in model.state_dict().items() if isinstance(v,torch.Tensor)}


def assert_state(model,state):
    now=model.state_dict()
    assert all(torch.equal(v,now[k]) for k,v in state.items()),'Frozen tensor state changed'


def fit_weights(data,fold=1):
    """Float32 row weights reproduce original m_train; all groups share full-fit denominator.

    OUTER rows receive the same fitting-defined class normalization for diagnostics,
    not for an actual optimization step. Original training uses only the FIT rows.
    """
    fit=np.asarray(data['split']['event'][str(fold)]['dev'])
    w=data['meta']['w'].astype(float);reg=data['meta']['regime'];m=data['m'];y=data['y']
    Z={r:float(np.sum(w[fit,None]*m[fit]*(reg[fit]==r)[:,None]*y[fit]**2)) for r in REGIMES}
    assert all(v>0 for v in Z.values())
    rw=w/np.array([Z[r] for r in reg]);scale=float(m[fit].sum()/np.sum(m[fit]*rw[fit,None]))
    rw=(rw*scale).astype(np.float32)
    denominator=float(torch.from_numpy(np.ascontiguousarray((m[fit]*rw[fit,None]).astype(np.float32))).sum())
    return rw,denominator,Z


def evaluate(model,F,stats,ids,chunk=64,exit_open=True):
    ids=np.asarray(ids);parts=[]
    with torch.no_grad():
        for start in range(0,len(ids),chunk):
            b=make_batch(F,ids[start:start+chunk],stats)
            parts.append(model(b,exit_open=exit_open)['P'].numpy())
    P=np.concatenate(parts);assert P.shape==(len(ids),144) and np.isfinite(P).all()
    assert np.min(P)>=0 and np.max(P)<=1
    return P


def distribution(values,weights):
    v=np.asarray(values,float).ravel();w=np.asarray(weights,float).ravel()
    valid=np.isfinite(v)&np.isfinite(w)&(w>0)
    if not valid.any(): return dict(n=0,missing=int(len(v)),mean=None,q10_median_q90=None)
    v,w=v[valid],w[valid]
    return dict(n=int(len(v)),missing=int((~valid).sum()),mean=float(np.sum(v*w)/w.sum()),
                q10_median_q90=weighted_quantile(v,w).tolist(),minimum=float(v.min()),maximum=float(v.max()))


def score(data,P_full,idx):
    P=np.asarray(P_full);idx=np.asarray(idx);n=len(data['y'])
    assert P.shape==(n,144) and np.isfinite(P[idx]).all()
    y,m,w=data['y'],data['m'],data['meta']['w'].astype(float);cs=cohorts(data)
    rw,_,Z=fit_weights(data);result={}
    peak=np.max(np.where(m,y,-np.inf),axis=1);pp=np.max(np.where(m,P,-np.inf),axis=1)
    for key in ('all','S','J','nonS'):
        ii=idx[cs[key][idx]]
        if not len(ii): result[key]=dict(n=0);continue
        err=m[ii]*(P[ii]-y[ii])**2;mass=np.sum(w[ii,None]*m[ii]);nmass=np.sum(rw[ii,None]*m[ii])
        row=dict(n=int(len(ii)),observed_hours=int(m[ii].sum()),design_RMSE=float(np.sqrt(np.sum(w[ii,None]*err)/mass)),
            design_SSE=float(np.sum(w[ii,None]*err)),class_normalized_MSE=float(np.sum(rw[ii,None]*err)/nmass))
        if key=='S':row['peak_ratio']=distribution(pp[ii]/peak[ii],w[ii])
        if key=='nonS':
            alarm=pp[ii]>=.1
            row['severe_alarms']=dict(n=int(alarm.sum()),weighted_rate=float(w[ii][alarm].sum()/w[ii].sum()))
        complete=ii[m[ii].all(1)]
        row['complete144']=dict(n=int(len(complete)),design_RMSE=float(np.sqrt(np.sum(w[complete,None]*(P[complete]-y[complete])**2)/(144*w[complete].sum()))) if len(complete) else None)
        result[key]=row
    return result
