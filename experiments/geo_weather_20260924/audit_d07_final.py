"""Independent final receipts and lossless public packaging for D07.

Reads only frozen public-D outcomes, registered models' exports, and D07 outputs.
No model forward, gradients, optimizer, or access to additional outcome sources.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '2'
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from evaluate_cr_tail import ROOT, HERE, RUNS, FEATURES, load_outcomes, load_predictions, cohorts

OUT = RUNS / 'd07_selectivity_20260929'
RESULTS = HERE / 'results/v1'
checks = 0


def check(condition, label):
    global checks
    if not condition:
        raise AssertionError(label)
    checks += 1


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_result(name):
    path = RESULTS / name
    raw = path.read_bytes() if path.exists() else gzip.decompress(Path(str(path)+'.gz').read_bytes())
    return json.loads(raw), raw


def finite_json(value):
    if isinstance(value, dict):
        for item in value.values(): finite_json(item)
    elif isinstance(value, list):
        for item in value: finite_json(item)
    elif isinstance(value, float):
        check(np.isfinite(value), 'finite JSON number')


def close(a, b, label, rtol=3e-7, atol=1e-10):
    check(np.allclose(a, b, rtol=rtol, atol=atol), label)


def direct_score(data, p, ids):
    y, m = data['y'][ids], data['m'][ids]
    w = data['meta']['w'][ids].astype(float)
    check(p.shape == data['y'].shape and np.isfinite(p).all(), 'finite full prediction shape')
    check(np.all((p >= 0) & (p <= 1)), 'prediction range')
    sse = np.sum(w[:, None] * np.where(m, (p[ids]-y)**2, 0))
    den = np.sum(w[:, None] * m)
    return float(sse), float(np.sqrt(sse/den))


def compact_attribution(s):
    bootstrap=s.get('merged_event_bootstrap',{})
    concentration=s.get('concentration',{}).get('positive_gain',{}).get('unit')
    return {'support':s['support'],'base_RMSE':s['base_rmse'],'candidate_RMSE':s['candidate_rmse'],
        'RMSE_improvement_fraction':s['rmse_improvement_fraction'],
        'improvement_CI95':bootstrap.get('rmse_improvement_fraction',{}).get('ci95'),
        'alignment':s['totals']['alignment'],'modification_energy':s['totals']['energy'],'net_gain':s['totals']['gain'],
        'unit_signs':s['unit_signs'],'positive_gain_concentration':concentration,'peak_ratio':s.get('peak_ratio')}


def main():
    if os.getpriority(os.PRIO_PROCESS, 0) < 15:
        os.nice(15-os.getpriority(os.PRIO_PROCESS, 0))
    a, raw_a = read_result('d07_attribution.json')
    b, raw_b = read_result('d07_controllers.json')
    c, raw_c = read_result('d07_gradients.json')
    for result in (a,b,c):
        finite_json(result)
        check(result['scope_commit']=='8f8ddef', 'registered D07 scope')
    original = json.loads((RESULTS/'i20_final_audit_s0.json').read_text())
    original_receipts = {}
    for name, digest in original['source_sha256'].items():
        check(sha(ROOT/name)==digest, 'frozen source '+name)
        original_receipts[name] = digest
    for label, folds in original['exports'].items():
        for fold, files in folds.items():
            for name, digest in files.items():
                path = RUNS/label/f'fold{int(fold):02d}'/name
                check(sha(path)==digest, 'frozen artifact '+str(path.relative_to(ROOT)))
    check(sha(FEATURES)==a['provenance']['data_sha256']==b['provenance']['data_sha256']
          ==c['frozen_provenance']['data_sha256'], 'original D data unchanged')
    check(sha(HERE/'d07_attribution.py')==a['provenance']['attribution_source_sha256'], 'A source')
    check(sha(HERE/'d07_controllers.py')==b['source_sha256'], 'B source')
    for name, digest in c['source_sha256'].items():
        check(sha(HERE/name)==digest, 'C source '+name)
    for result, key in ((b,'local_artifact_hashes'),(c,'local_artifacts')):
        for name, digest in result[key].items():
            check(sha(OUT/name)==digest, 'local output '+name)
    check(sha(OUT/'attribution_unit_rows.npz')==a['provenance']['local_unit_rows_sha256'], 'A local rows')
    data = load_outcomes(); n = len(data['y']); masks = cohorts(data)
    check(n==8457, 'public D unit count')
    fit = np.sort(np.asarray(data['split']['event']['1']['dev'], int))
    outer = np.sort(np.asarray(data['expected'][1], int))
    check(len(fit)==6350 and len(outer)==2107 and np.array_equal(np.sort(np.r_[fit,outer]),np.arange(n)), 'fold1 complete partition')
    host,_ = load_predictions('v1_host_s0','W+Cin',data['expected'],n)
    crk,_ = load_predictions('v1_crk_s0','CRK+Cin',data['expected'],n)
    closed = np.empty_like(crk); seen = np.zeros(n, int)
    for fold in range(1,6):
        with np.load(RUNS/'v1_crk_s0'/f'fold{fold:02d}'/'outer.npz',allow_pickle=False) as z:
            closed[z['idx']] = z['P_closed']; np.add.at(seen,z['idx'],1)
    check(np.all(seen==1), 'closed OOF exact-once coverage')
    with np.load(OUT/'controller_baseline_scalars.npz',allow_pickle=False) as z:
        p1=z['P'].astype(float)
        check(np.array_equal(z['idx'],np.arange(n)), 'B all-row identity')
    with np.load(OUT/'host_fold1_scalars.npz',allow_pickle=False) as z: h1=z['P'].astype(float)
    with np.load(OUT/'fold1_closed.npz',allow_pickle=False) as z: cl1=z['P_closed'].astype(float)
    pairs = {'oof_independent_host':(host,crk,np.arange(n)),
             'oof_same_crk_exit':(closed,crk,np.arange(n)),
             'fit1_independent_host':(h1,p1,fit),'outer1_independent_host':(h1,p1,outer),
             'fit1_same_crk_exit':(cl1,p1,fit),'outer1_same_crk_exit':(cl1,p1,outer)}
    maximum_identity_error=0.
    with np.load(OUT/'attribution_unit_rows.npz',allow_pickle=False) as z:
        for tag,(base,candidate,ids) in pairs.items():
            err0=np.where(data['m'],(data['y']-base)**2,0).sum(1)
            err1=np.where(data['m'],(data['y']-candidate)**2,0).sum(1)
            close(err0,z[tag+'__base_sse'],tag+' row base SSE',atol=2e-9)
            close(err1,z[tag+'__candidate_sse'],tag+' row candidate SSE',atol=2e-9)
            delta=candidate-base
            alignment=np.where(data['m'],2*(data['y']-base)*delta,0).sum(1)
            energy=np.where(data['m'],delta**2,0).sum(1)
            maximum_identity_error=max(maximum_identity_error,float(np.max(abs(err0-err1-alignment+energy))))
            for support in ('common_observed','complete144'):
                for cohort in ('all','S','J','nonS'):
                    ii=ids[masks[cohort][ids]]
                    if support=='complete144':ii=ii[data['m'][ii].all(1)]
                    s=a['summaries'][tag]['supports'][support][cohort]
                    sse0,rmse0=direct_score(data,base,ii);sse1,rmse1=direct_score(data,candidate,ii)
                    close([sse0,sse1,rmse0,rmse1],[s['totals']['base_sse'],s['totals']['candidate_sse'],s['base_rmse'],s['candidate_rmse']],tag+' aggregate')
                    check(int((err0[ii]>err1[ii]).sum())==s['unit_signs']['positive']['units'],tag+' positive-unit count')
    scale_summary=[]
    for scale in (.95,1.05):
        with np.load(OUT/f'controller_scale_{str(scale).replace(".","p")}.npz',allow_pickle=False) as z:p=z['P'].astype(float)
        row={'write_weather_scale':scale,'splits':{}}
        for split,ids in (('FIT',fit),('OUTER',outer)):
            row['splits'][split]={}
            for cohort in ('all','S','nonS','J'):
                ii=ids[masks[cohort][ids]]
                _,rmse=direct_score(data,p,ii);_,baseline=direct_score(data,p1,ii)
                scored=b['perturbations'][str(scale)]['scores'][split][cohort]['score'][cohort]
                close(rmse,scored['design_RMSE'],'B scale score')
                row['splits'][split][cohort]={'RMSE':rmse,'RMSE_change_fraction':rmse/baseline-1}
            ii=ids[masks['nonS'][ids]]
            peak=np.max(np.where(data['m'][ii],p[ii],-np.inf),1)
            transitions=b['perturbations'][str(scale)]['scores'][split]['false_peak_transitions']
            check(int((peak>=.1).sum())==transitions['intervention']['count'],'B false-peak count')
            row['splits'][split]['nonS_false_peak_count']=int((peak>=.1).sum())
        scale_summary.append(row)
    expected_interventions={(block,eps) for block in ('weather_control','raw_geography','kernel_basis') for eps in (.001,.01)}
    check(len(c['parameter_interventions'])==6 and
          {(r['block'],r['relative_epsilon']) for r in c['parameter_interventions']}==expected_interventions,
          'exactly the six registered parameter interventions')
    local_vectors=OUT/'gradients_vectors.npz'
    with np.load(local_vectors,allow_pickle=False) as z:
        close(z['partition_gradients'].sum(0),z['full_fit_gradient'],'independent gradient additivity',rtol=3e-4,atol=3e-9)
        check(np.array_equal(z['fit_ids'],fit) and np.array_equal(z['outer_ids'],outer), 'gradient IDs')
        block_ids=np.concatenate([z[k] for k in z.files if k.startswith('block_indices_')])
        check(np.array_equal(np.sort(block_ids),np.arange(len(z['full_fit_gradient']))),'exhaustive disjoint parameter blocks')
        rw=z['row_weight'].copy();gradient=z['full_fit_gradient'].copy();theta=z['baseline_parameters'].copy()
        for intervention in c['parameter_interventions']:
            ii=z['block_indices_'+intervention['block']]
            requested=-intervention['relative_epsilon']*np.linalg.norm(theta[ii])*gradient[ii]/np.linalg.norm(gradient[ii])
            # Source uses float32 in-place parameter addition, not float64 addition then casting.
            actual=(theta[ii].astype('f4')+requested.astype('f4')).astype(float)-theta[ii]
            close(np.linalg.norm(actual),intervention['achieved_step_l2'],'independent achieved step',rtol=1e-10,atol=1e-12)
            close(np.linalg.norm(actual)/np.linalg.norm(theta[ii]),intervention['relative_epsilon'],'registered relative step',rtol=1e-5,atol=1e-9)
    expected_partitions={(str(s),cohort) for s in np.unique(data['meta']['system'][fit]) for cohort in ('S','nonS')
                         if np.any((data['meta']['system'][fit]==s)&masks[cohort][fit])}
    check({(r['system'],r['cohort']) for r in c['partitions']}==expected_partitions,'original system partitions, not merged groups')
    check(len(c['partitions'])==len(expected_partitions),'no duplicate gradient partition receipts')
    for partition in c['partitions']:
        ids=fit[(data['meta']['system'][fit]==partition['system'])&masks[partition['cohort']][fit]]
        check(len(ids)==partition['n_units'],'partition unit count')
    parameter_summary=[]
    with np.load(OUT/'gradients_parameter_predictions.npz',allow_pickle=False) as z:
        close(z['baseline'],p1,'C/B baseline agreement',atol=2e-7)
        for intervention in c['parameter_interventions']:
            check(intervention['before_state_sha256']==intervention['restored_state_sha256']==c['initial_and_final_state_sha256'], 'runtime restoration hash receipt consistent')
            row={key:intervention[key] for key in ('name','block','relative_epsilon')};row['splits']={}
            p=z[intervention['name']]
            for split,ids in (('FIT',fit),('OUTER',outer)):
                row['splits'][split]={}
                for cohort in ('all','S','nonS','J'):
                    ii=ids[masks[cohort][ids]];_,rmse=direct_score(data,p,ii);_,baseline=direct_score(data,p1,ii)
                    scored=intervention['scores'][split][cohort]
                    close(rmse,scored['design_RMSE'],'C intervention score')
                    objective_delta=float(np.sum(rw[ii,None]*np.where(data['m'][ii],
                        (p[ii]-data['y'][ii])**2-(p1[ii]-data['y'][ii])**2,0))/c['objective']['common_full_FIT_denominator'])
                    close(objective_delta,scored['paired_objective_delta'],'C independent original objective delta',rtol=1e-6,atol=1e-15)
                    row['splits'][split][cohort]={'RMSE':rmse,'RMSE_change_fraction':rmse/baseline-1,
                        'original_FIT_denominator_objective_change':scored['paired_objective_delta']}
                ii=ids[masks['nonS'][ids]]
                peak=np.max(np.where(data['m'][ii],p[ii],-np.inf),1)
                count=int((peak>=.1).sum())
                check(count==intervention['scores'][split]['nonS']['observed_severe_false_alarms']['n'],'C independent severe false peaks')
                row['splits'][split]['nonS_false_peak_count']=count
            parameter_summary.append(row)
    check(maximum_identity_error<1e-8,'exact paired error identity')
    packs={}
    for name,raw in (('d07_attribution.json',raw_a),('d07_controllers.json',raw_b),('d07_gradients.json',raw_c)):
        packed=gzip.compress(raw,compresslevel=9,mtime=0);path=RESULTS/(name+'.gz')
        if path.exists():check(path.read_bytes()==packed,'existing deterministic package')
        else:path.write_bytes(packed)
        check(gzip.decompress(packed)==raw,'lossless summary package')
        packs[path.name]={'uncompressed_sha256':hashlib.sha256(raw).hexdigest(),'gzip_sha256':sha(path),'bytes':len(packed)}
    selected_points=[r for r in b['baseline_working_points'] if r['regime']=='all' and r['cohort'] in ('S','nonS','nonS_alarm')
                     and r['timing'] in ('true_peak','crk_predicted_peak','fixed_clock')]
    compact_points=[]
    for r in selected_points:
        compact_points.append({key:r[key] for key in ('split','cohort','timing','support')}|{
            'controller_q10_q50_q90':{k:([x['q10_q50_q90'] for x in r['values'][k]]
                 if isinstance(r['values'][k],list) else r['values'][k]['q10_q50_q90'])
                 for k in ('deposit_norm','temporal_deposit_cosine','jacobian_frobenius','jacobian_direction_norm','actual_delta_per_input_distance')},
            'joint_lock_fraction':[q['lock']['fraction'] for q in r['joint']]})
    evidence={'scope_commit':'8f8ddef','exploratory':True,'seed':0,'no_new_training':True,
        'sign_convention':'RMSE_change_fraction >0 means worse; A rmse_improvement_fraction >0 means better',
        'attribution':{tag:{support:{k:compact_attribution(a['summaries'][tag]['supports'][support][k]) for k in ('all','S','J','nonS')}
                            for support in ('common_observed','complete144')} for tag in pairs},
        'OOF_S_outcome_phenotypes':{support:{name:compact_attribution(s)|{'weighted_zero_SSE':s['weighted_zero_SSE'],
               'base_peak_ratio_q10_median_q90':s['base_peak_ratio_q10_median_q90'],
               'candidate_peak_ratio_q10_median_q90':s['candidate_peak_ratio_q10_median_q90']}
               for name,s in a['summaries']['oof_independent_host']['supports'][support]['S_outcome_phenotypes'].items()}
               for support in ('common_observed','complete144')},
        'controller_selected_working_points':compact_points,'write_interventions':scale_summary,
        'gradient_blocks':{name:{'gradient_l2':val['total']['gradient_l2'],
               'gradient_l2_per_parameter_l2':val['total']['gradient_l2_per_parameter_l2'],
               'S_gradient_l2':val['S']['gradient_l2'],'nonS_gradient_l2':val['nonS']['gradient_l2'],
               'S_nonS_cosine':val['S_nonS_cosine']} for name,val in c['blocks'].items()},
        'gradient_additivity':c['validation']['partition_additivity'],
        'gradient_cohort_loss':c['cohort_objective_contributions'],
        'parameter_interventions':parameter_summary,'full_summary_packages':packs,
        'limits':['One frozen seed; internal diagnostics fixed fold1.','Outcome phenotypes are descriptive, not deployable subgroup labels.',
                  'Internal hidden-control perturbations are not weather interventions or fair retraining gains.',
                  'No causal or net-geographic-information claim; no selected extra scales.']}
    receipt={'passed':True,'checks':checks,'maximum_unit_identity_abs_error':maximum_identity_error,
        'original_sources_and_30_artifacts_unchanged':True,'public_D_hash':sha(FEATURES),
        'exact_fivefold_coverage':True,'data_units':n,'FIT_units':len(fit),'OUTER_units':len(outer),
        'original_system_partition_verified':True,'eight_fixed_interventions_checked':True,
        'restoration_evidence':'Exact runtime state assertions and consistent receipt hashes; separately reconstructed perturbation norms. No restored model bytes were exported.',
        'source_sha256':sha(__file__),'packages':packs}
    for name,value in (('d07_evidence.json',evidence),('d07_final_audit.json',receipt)):
        path=RESULTS/name
        with path.open('x') as stream:json.dump(value,stream,ensure_ascii=False,indent=1,allow_nan=False);stream.write('\n')
    print(json.dumps(receipt,ensure_ascii=False))


if __name__=='__main__':main()
