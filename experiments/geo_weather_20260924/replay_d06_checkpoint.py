"""Read-only fold1 D06 working-point diagnostics with frozen buffers and weights."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='2'
import sys, json, math
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as tf
from evaluate_cr_tail import ROOT, HERE, RUNS, FEATURES, load_outcomes, cohorts, sha, write_new
sys.path.insert(0,str(ROOT/'src'))
from asymode.asym_host import AsymODE
from asymode.gcrk_train import make_batch, _std
from asymode.controlled_relaxation import ControlledRelaxationLayer
from audit_d06_response import DATA_SHA, OUT, weighted_quantile

RESULT=HERE/'results/v1/d06_checkpoint_audit.json'


def dist(v,w):
    v,w=np.asarray(v,float).ravel(),np.asarray(w,float).ravel()
    assert v.shape==w.shape and np.isfinite(v).all() and (w>0).all()
    if not len(v): return None
    return dict(n=len(v),mean=float(np.sum(w*v)/w.sum()),
                q10_median_q90=weighted_quantile(v,w).tolist(),min=float(v.min()),max=float(v.max()))


def geography(layer, g):
    e=layer.geography_features(g,deduplicate=False)
    raw=e[:,:40];basis=e[:,40:]; k=basis*math.sqrt(32)
    ans={}
    ones=np.ones(len(g))
    for name,a in [('raw_norm',raw.norm(dim=1)),('basis_norm',basis.norm(dim=1)),('kernel_nearest',k.max(1).values),('kernel_mean',k.mean(1))]:
        ans[name]=dist(a.numpy(),ones)
    for name,a in [('uncentered',basis),('centered',basis-basis.mean(0))]:
        s=torch.linalg.svdvals(a.double()).numpy();power=s*s;prob=power/power.sum()
        ans[name+'_spectrum']=dict(singular_values=s.tolist(),participation_rank=float(power.sum()**2/(power*power).sum()),
            entropy_rank=float(np.exp(-np.sum(prob[prob>0]*np.log(prob[prob>0])))),rank_relative_1e6=int((s>s[0]*1e-6).sum()))
    for name,param in [('fusion',layer.fusion_geo),('head',layer.head_geo)]:
        aa=tf.linear(raw,param[:,:40]);bb=tf.linear(basis,param[:,40:])
        ans[name+'_raw_term_norm']=dist(aa.norm(dim=1).numpy(),ones)
        ans[name+'_basis_term_norm']=dist(bb.norm(dim=1).numpy(),ones)
        ans[name+'_parameter_norms']=[float(param[:,:40].norm()),float(param[:,40:].norm())]
    # Mean weighted contribution of each order over every row-landmark pair.
    totals=np.zeros(40)
    for chunk in g.split(256):
        x=layer.normalized_geography(chunk)
        base=torch.exp(-.5*((x[:,None]-layer.landmarks[None])/layer.length_scales()).square())
        means=base.new_ones((*base.shape[:2],1))
        for j in range(1,41):
            orders=torch.arange(j+1,dtype=base.dtype)
            means=(j-orders)/j*tf.pad(means,(0,1))+orders/j*base[...,j-1,None]*tf.pad(means,(1,0))
        totals+=(means[...,1:]*layer.order_masses()).sum((0,1)).numpy()
    ans['kernel_value_share_by_order']=(totals/totals.sum()).tolist()
    return ans


def main():
    assert not RESULT.exists()
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    assert sha(FEATURES)==DATA_SHA
    data=load_outcomes();masks=cohorts(data);n=len(data['y'])
    with np.load(FEATURES,allow_pickle=False) as z:
        F={k:z[k].copy() for k in ('xu','xr','xo','geo','fips','y','m','y0')}
    fit=np.sort(np.array(data['split']['event']['1']['dev']));outer=data['expected'][1]
    w=data['meta']['w'].astype(float);m=data['m'];y=data['y']
    folds={'fit':fit,'outer':outer};output={};provenance={}
    for label in ('v1_host_s0','v1_crk_s0'):
        folder=RUNS/label/'fold01'; path=folder/'final.pt';digest=sha(path)
        ck=torch.load(path,map_location='cpu',weights_only=False)
        assert ck['steps']==900 and ck['arm']==('CRK+Cin' if label=='v1_crk_s0' else 'W+Cin')
        model=AsymODE(F['xu'].shape[-1],F['xr'].shape[-1],F['xo'].shape[-1]);model.attach_context_input(6)
        g=torch.from_numpy(_std(F['geo'],ck['stats']['geo'],0))
        if label=='v1_crk_s0':
            model.damage[2]=ControlledRelaxationLayer(model.damage[2],g[fit],county_ids=F['fips'][fit],
                        checkpoint_steps=0,use_adjoint=True,geo_checkpoint=True)
        model.load_state_dict(ck['model_state'],strict=True);model.eval()
        state_before={key:v.clone() for key,v in model.state_dict().items() if isinstance(v,torch.Tensor)}
        values={};tau=np.empty((n,144,4),dtype='f4') if model.kernel else None
        P=np.empty((n,144),dtype='f4');geo={}
        with torch.no_grad():
            if model.kernel:
                for split,ids in folds.items(): geo[split]=geography(model.kernel,g[ids])
                geo['length_scales']=model.kernel.length_scales().tolist();geo['order_masses']=model.kernel.order_masses().tolist()
                geo['beta']=float(torch.tanh(model.kernel.alpha));geo['scale']=float(model.kernel.scale)
                geo['calibration_step']=int(model.kernel.calibration_step);geo['training_step']=int(model.kernel.training_step)
            for start in range(0,n,64):
                ids=np.arange(start,min(start+64,n));batch=make_batch(F,ids,ck['stats'])
                out=model(batch,diagnostics=True);P[ids]=out['P'].numpy()
                d={key:out[key] for key in ('P','u','r','gate','background','conditional','raw_logit','logit','forget')}
                d['h1_norm']=out['h1'][:,72:].norm(dim=-1);d['a2_norm']=out['a2'][:,72:].norm(dim=-1)
                if model.kernel:
                    k=model.kernel
                    for name in ('response','effect','departure'):
                        d[name+'_norm']=out['kernel_'+name][:,72:].norm(dim=-1)
                    for name in ('deposit','state'):
                        d[name+'_mode_norm_mean']=out['kernel_'+name][:,72:].norm(dim=-1).mean(-1)
                    d['instant_write_mode_norm_mean']=((1-out['kernel_rho'][:,72:, :,None])*out['kernel_deposit'][:,72:]).norm(dim=-1).mean(-1)
                    tau[ids]=out['kernel_tau'][:,72:].numpy()
                    d['instant_coefficient_mean']=(1-out['kernel_rho'][:,72:]).mean(-1)
                    d['effect_to_h_norm']=d['effect_norm']/d['h1_norm'].clamp_min(1e-12)
                    e=out['kernel_geography_features'];fg,hg,anchor=k._geographic_terms(e)
                    u=torch.cat((torch.asinh(out['h1']/k.level_scale),torch.asinh(out['kernel_departure']/k.departure_scale)),-1)
                    psi=torch.tanh(fg[:,None]+tf.linear(u,k.fusion_weather))
                    raw=anchor[:,None]+tf.linear(u,k.head_weather)+tf.linear(psi-torch.tanh(fg)[:,None],k.head_fusion)
                    d['deposit_raw_saturated_fraction']=(torch.tanh(raw[:,72:,:32]).abs()>.99).float().mean(-1)
                    d['deposit_anchor_saturated_fraction']=(torch.tanh(anchor[:,:32]).abs()>.99).float().mean(-1)[:,None].expand(-1,144)
                for key,a in d.items():
                    v=a.numpy();assert v.shape==(len(ids),144) and np.isfinite(v).all(),key
                    if key not in values:values[key]=np.empty((n,144),dtype='f4')
                    values[key][ids]=v
                if start%1024==0:print(label,start,n,flush=True)
        with np.load(folder/'outer.npz') as z:
            replay=float(np.max(np.abs(P[z['idx']]-z['P'])))
        assert replay < 2e-6, replay
        for key,val in state_before.items():assert torch.equal(val,model.state_dict()[key]),key
        assert sha(path)==digest
        rows=[]; scores={}
        alarm=np.max(np.where(m,P,-np.inf),1)>=.1
        peak=np.argmax(np.where(m,y,-np.inf),1)
        peakmask=np.zeros_like(m);peakmask[np.arange(n),peak]=True;peakmask&=m
        clock=np.zeros_like(m);clock[:,np.arange(0,144,12)]=True;clock&=m
        for split,ids in folds.items():
            sel=np.zeros(n,bool);sel[ids]=True
            for cohort,cs in [('all',masks['all']),('S',masks['S']),('nonS',masks['nonS']),('nonS_alarm',masks['nonS']&alarm)]:
                ss=sel&cs
                if not ss.any():continue
                scores[split+'/'+cohort]=dict(n=int(ss.sum()),design_RMSE=float(np.sqrt(np.sum(w[ss,None]*m[ss]*(P[ss]-y[ss])**2)/np.sum(w[ss,None]*m[ss]))),
                    observed_peak_ratio=dist(np.max(np.where(m[ss],P[ss],-np.inf),1)/np.maximum(np.max(np.where(m[ss],y[ss],-np.inf),1),1e-12),w[ss]) if cohort=='S' else None)
                for when,tm in [('observed_forecast',m),('fixed_clock',clock),('true_peak',peakmask)]:
                    chosen=ss[:,None]&tm;ww=np.broadcast_to(w[:,None],m.shape)[chosen]
                    row=dict(split=split,cohort=cohort,time=when,units=int(ss.sum()),values={key:dist(val[chosen],ww) for key,val in values.items()})
                    if tau is not None:row['tau_by_mode']=[dist(tau[:,:,j][chosen],ww) for j in range(4)]
                    rows.append(row)
        # Honest FIT residual shares under the original class-normalized objective, from frozen predictions.
        Z={r:np.sum(w[fit]*(data['meta']['regime'][fit]==r)*(m[fit]*y[fit]**2).sum(1)) for r in np.unique(data['meta']['regime'])}
        rw=w/np.array([Z[r] for r in data['meta']['regime']]);se=(m*(P-y)**2).sum(1)
        fitloss={}
        for cname in ('all','S','J','nonS'):
            ids=fit[masks[cname][fit]]
            fitloss[cname]=dict(raw_design_SSE_share=float(np.sum(w[ids]*se[ids])/np.sum(w[fit]*se[fit])),
                class_normalized_SSE_share=float(np.sum(rw[ids]*se[ids])/np.sum(rw[fit]*se[fit])))
        output[label]=dict(replay_max_abs=replay,scores=scores,geography=geo,working_points=rows,fit_loss=fitloss)
        provenance[label]=dict(checkpoint_sha256=digest,outer_sha256=sha(folder/'outer.npz'))
        # Local scalar arrays support later audits without retaining full hidden sequences.
        with (OUT/(label+'_fold1_scalars.npz')).open('xb') as stream:
            np.savez_compressed(stream,**values,**({'tau':tau} if tau is not None else {}))
    write_new(RESULT,dict(scope_commit='5fa930e',fold=1,read_only=True,source_sha256=sha(__file__),data_sha256=DATA_SHA,
        checkpoint_provenance=provenance,results=output,local_artifact_hashes={name:sha(OUT/name) for name in ('v1_host_s0_fold1_scalars.npz','v1_crk_s0_fold1_scalars.npz')}))
    print('D06 frozen replay complete',flush=True)


if __name__=='__main__':main()
