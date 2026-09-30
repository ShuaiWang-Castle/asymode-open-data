"""D09 single-job worker: fixed small-panel inner-group training from scratch.

Only the frozen D08 roster supplies training units. Model preprocessing uses the
job's inner FIT only. At step900, score/export all original outer-FIT counties in
the heldout groups. No original outer1 arrays, checkpoints or D07 predictions are
used. A separate registered queue owns the two-process limit.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
import zipfile

import numpy as np
import torch
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from asymode import gcrk_train as G
from asymode.asym_host import AsymODE
import d08_panel as D

PANEL = ROOT / 'runs/geo_weather_20260924/d08_panel_20260929_iofix1'
REGIMES = ('tropical', 'winter', 'synoptic_wind', 'convective', 'heavy_rain')
ARMS = ('host', 'crk', 'new')
STEPS, SEED, PRIVATE_SEED, MICRO = 900, 0, 1729, 512
CHECKPOINTS = (0, 100, 300, 900)
PREFLIGHT_STEPS = 3
PROTOCOL = dict(steps=STEPS, seed=SEED, private_seed=PRIVATE_SEED,
    microbatch=MICRO, gradient_accumulation='one complete inner-FIT objective per Adam update',
    host_learning_rate=G.LR_HOST, recovery_learning_rate=G.LR_RECOVERY,
    training_weights='original panel w / inner-FIT regime zero-predictor SSE; not HT',
    calibration_every=10, checkpoints=list(CHECKPOINTS), early_stopping=False,
    heldout_full_export_step=900, numerical_threads=2, minimum_nice=15)
INPUT_MEMBERS = ('xu', 'xr', 'xo', 'geo', 'y', 'm', 'y0', 'obs_full')
METADATA_MEMBERS = ('fips', 'system', 'family', 'origin', 'regime', 'w')
SOURCE_FILES = [Path(__file__), HERE/'test_d09_train.py', HERE/'d09_write.py',
    HERE/'test_d09_write.py', HERE/'d09_score.py', HERE/'d09_queue.py', Path(D.__file__),
    ROOT/'src/asymode/gcrk_train.py', ROOT/'src/asymode/asym_host.py',
    ROOT/'src/asymode/gcrk.py', ROOT/'src/asymode/controlled_relaxation.py',
    ROOT/'src/asymode/controlled_relaxation_scan.py', ROOT/'src/asymode/geo_mech.py']


def utc():
    return datetime.now(timezone.utc).isoformat()


def status(path, **values):
    temp = path.with_name(path.name + '.tmp')
    temp.write_bytes(D.json_bytes(dict(updated_utc=utc(), pid=os.getpid(), **values)))
    os.replace(temp, path)


def verify_registration(commit, scope):
    if len(commit) < 7 or not set(commit).issubset(set('0123456789abcdef')):
        raise ValueError('Use an explicit hexadecimal registration commit')
    full = subprocess.check_output(['git', 'rev-parse', '--verify', commit+'^{commit}'],
                                  cwd=ROOT, stderr=subprocess.DEVNULL, text=True).strip()
    inventory = {}
    for path in [*SOURCE_FILES, scope]:
        name = D.relative(path)
        blob = subprocess.check_output(['git', 'show', full+':'+name], cwd=ROOT, stderr=subprocess.DEVNULL)
        if hashlib.sha256(blob).hexdigest() != D.sha(path):
            raise ValueError('Registered source differs: '+name)
        inventory[name] = D.sha(path)
    old = json.loads((HERE/'results/v1/i20_final_audit_s0.json').read_text())
    for name, value in old['source_sha256'].items():
        if D.sha(ROOT/name) != value:
            raise ValueError('An original frozen source changed: '+name)
    return full, inventory


def stream_subset(archive, member, units, n=8457):
    """Retain declared outer-FIT rows from C or Fortran NPY members only."""
    units = np.asarray(units, dtype=int)
    if not np.array_equal(units, np.sort(units)) or len(np.unique(units)) != len(units):
        raise ValueError('Source unit IDs must be sorted and unique')
    with archive.open(member+'.npy') as f:
        version = np.lib.format.read_magic(f)
        shape, fortran, dtype = np.lib.format._read_array_header(f, version)
        if dtype.hasobject or len(shape) not in (1, 2, 3) or shape[0] != n:
            raise ValueError('Unexpected registered input layout: '+member)
        result = np.empty((len(units), *shape[1:]), dtype=dtype)
        if fortran and len(shape) > 1:
            width = n * dtype.itemsize
            for column in range(int(np.prod(shape[1:]))):
                raw = f.read(width)
                if len(raw) != width:
                    raise ValueError('Truncated Fortran input: '+member)
                view = memoryview(raw)
                kept = b''.join(view[int(i)*dtype.itemsize:(int(i)+1)*dtype.itemsize] for i in units)
                coordinate = np.unravel_index(column, shape[1:], order='F')
                result[(slice(None), *coordinate)] = np.frombuffer(kept, dtype=dtype)
        else:
            width = int(np.prod(shape[1:])) * dtype.itemsize
            row = 0
            for original in range(n):
                raw = f.read(width)
                if len(raw) != width:
                    raise ValueError('Truncated C input: '+member)
                if row < len(units) and original == units[row]:
                    value = np.frombuffer(raw, dtype=dtype).reshape(shape[1:])
                    result[row] = value
                    row += 1
            if row != len(units):
                raise ValueError('Incomplete source unit selection')
        if f.read(1):
            raise ValueError('Trailing registered input data')
    return result


def load_data():
    frozen = json.loads((PANEL/'FROZEN.json').read_text())
    if frozen['status'] != 'input_roster_frozen':
        raise ValueError('D08 roster is not frozen')
    for name, key in [('manifest.json','manifest_sha256'), ('roster.json','roster_sha256'),
                      ('panel_inputs.npz','panel_inputs_sha256')]:
        if D.sha(PANEL/name) != frozen[key]:
            raise ValueError('D08 frozen artifact differs: '+name)
    manifest=json.loads((PANEL/'manifest.json').read_text())
    if D.sha(D.SPLITS)!=manifest['provenance']['split_sha256']:
        raise ValueError('Current original split differs from the D08 frozen split')
    if D.sha(D.__file__)!=manifest['provenance']['source_sha256']:
        raise ValueError('D08 input helper differs from its frozen source')
    if D.sha(D.FEATURES) != D.DATA_SHA:
        raise ValueError('Original public-D hash differs')
    split = json.loads(D.SPLITS.read_text())
    with np.load(PANEL/'panel_inputs.npz', allow_pickle=False) as z:
        ids = z['original_fit'].copy(); chosen = z['unit'].copy()
        group = z['population_group'].copy(); pi = z['population_pi'].copy()
    expected = np.sort(np.asarray(split['event']['1']['dev']))
    if len(ids) != 6350 or len(chosen) != 1219 or not np.array_equal(ids, expected):
        raise ValueError('D08 population/roster differs')
    with zipfile.ZipFile(D.FEATURES) as archive:
        F = {member: stream_subset(archive, member, ids) for member in INPUT_MEMBERS}
    with np.load(D.FEATURES, allow_pickle=False) as z:
        F.update({member: z[member][ids].copy() for member in METADATA_MEMBERS})
    if (F['xu'].shape[:2] != (6350,216) or F['xr'].shape[:2] != (6350,216)
            or F['xo'].shape[:2] != (6350,216) or F['geo'].shape != (6350,40)
            or F['y'].shape != (6350,144) or F['m'].shape != (6350,144)):
        raise ValueError('Original model input schema differs')
    if not np.isfinite(F['y']).all() or not np.isfinite(F['y0']).all():
        raise ValueError('Public observed-state arrays must be finite')
    if F['obs_full'].shape != (6350,216) or not np.all(F['obs_full'][:,71]):
        raise ValueError('The origin hour71 must be observed for every retained row')
    F['origin_observed']=F['obs_full'][:,71].astype(bool)
    if np.any((F['m'] != 0) & (F['m'] != 1)):
        raise ValueError('Evaluation mask must be the original binary observation mask')
    fold = np.zeros(len(ids), dtype=np.int8)
    for k in range(2,6):
        fold[np.isin(ids,split['event'][str(k)]['outer'])] = k
    if np.any(fold == 0) or any(len(np.unique(fold[group == g])) != 1 for g in np.unique(group)):
        raise ValueError('Original outer-FIT merged groups cross inner folds')
    return dict(F=F, unit=ids, group=group, fold=fold, pi=pi, panel=np.isin(ids,chosen),
                frozen=frozen, split_sha256=D.sha(D.SPLITS), frozen_sha256=D.sha(PANEL/'FROZEN.json'))


def job_indices(data, heldout_fold):
    if heldout_fold not in (2,3):
        raise ValueError('Only registered heldout folds2 and3 are allowed')
    held = data['fold'] == heldout_fold
    result = dict(fit=np.flatnonzero(data['panel'] & ~held),
                  held_panel=np.flatnonzero(data['panel'] & held), held_full=np.flatnonzero(held))
    if any(not len(v) for v in result.values()):
        raise ValueError('Empty registered training/heldout set')
    fit_groups = set(data['group'][result['fit']])
    if fit_groups & set(data['group'][result['held_full']]):
        raise ValueError('Merged-group leakage')
    return result


def design_weighted(F, fit):
    """Original screen.py objective, recomputed on this inner-panel FIT only."""
    fit=np.asarray(fit,dtype=int);w=F['w'].astype(float);reg=F['regime'].astype(str)
    m,y=F['m'].astype(float),F['y'].astype(float);row=np.zeros(len(w));Z={}
    for r in sorted(set(reg[fit])):
        ii=fit[reg[fit]==r];zero=float(np.sum(w[ii,None]*m[ii]*y[ii]**2));Z[r]=zero
        if zero>0:row[ii]=w[ii]/zero
    denominator=np.sum(m[fit]*row[fit,None])
    if denominator<=0:
        raise ValueError('No positive inner-FIT zero-predictor risk')
    scale=float(m[fit].sum()/denominator)
    output=dict(F);output['m_train']=(m*(row*scale)[:,None]).astype(np.float32)
    return output,dict(Z_regime=Z, mean_observed_cell_weight_scale=scale,
                      weighted_training_denominator=float(torch.from_numpy(output['m_train'][fit]).sum()),
                      observation_cells=int(m[fit].sum()))


def parameter_sha(parameters):
    h=hashlib.sha256()
    for name,value in sorted(parameters.items()):
        a=value.detach().cpu().contiguous().numpy()
        h.update(name.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


class Engine(G.Engine):
    def __init__(self,F,fit,arm):
        if arm not in ARMS:raise ValueError('Unregistered arm')
        super().__init__(F,fit,None,SEED,'W+Cin' if arm=='host' else 'CRK+Cin',private_seed=PRIVATE_SEED)
        self.microbatch_size=MICRO
        if arm=='new':
            from d09_write import IncrementalWriteRelaxationLayer
            old=self.model.kernel
            # Only reuse the shared W2/b2 host Parameters. All new controls are
            # constructor initialized with1729, not copied from trained weights.
            state=torch.random.get_rng_state()
            original=torch.nn.Linear(old.in_features,old.out_features)
            original.weight,original.bias=old.weight,old.bias
            self.model.damage[2]=IncrementalWriteRelaxationLayer(original,self.fit['geo'],
                county_ids=np.asarray(F['fips'])[self.fit_idx],private_seed=PRIVATE_SEED,
                checkpoint_steps=0,use_adjoint=True,geo_checkpoint=True)
            torch.random.set_rng_state(state)
            old_params=dict(old.named_parameters());new_params=dict(self.model.kernel.named_parameters())
            if set(old_params)!=set(new_params) or any(not torch.equal(old_params[n],new_params[n]) for n in old_params):
                raise ValueError('New constructor did not preserve paired fresh kernel initialization')
            host,rec=self.model.parameter_groups()
            self.opt=torch.optim.Adam([dict(params=host,lr=G.LR_HOST),dict(params=rec,lr=G.LR_RECOVERY)],lr=G.LR_HOST)
        self.d09_arm=arm
        trainable={id(p) for p in self.model.parameters() if p.requires_grad}
        assigned=[id(p) for group in self.opt.param_groups for p in group['params']]
        if set(assigned)!=trainable or len(assigned)!=len(set(assigned)):
            raise ValueError('Optimizer does not cover every trainable parameter exactly once')
        self.optimizer_covers_all_trainable=True
        # Comparing actual shared tensors catches accidental changes to paired
        # initialization, including contextual weights and W2/b2.
        state=torch.random.get_rng_state();torch.manual_seed(SEED)
        reference=AsymODE(F['xu'].shape[-1],F['xr'].shape[-1],F['xo'].shape[-1]);reference.attach_context_input(G.N_STATIC)
        torch.random.set_rng_state(state)
        shared=dict(reference.named_parameters());actual=dict(self.model.named_parameters())
        if any(n not in actual or not torch.equal(p,actual[n]) for n,p in shared.items()):
            raise ValueError('Host initialization is not paired with seed0W')
        self.initial_host_parameter_sha256=parameter_sha(shared)
        self.initial_kernel_parameter_sha256=None if self.model.kernel is None else parameter_sha(dict(self.model.kernel.named_parameters()))
        if not all(np.isfinite(a).all() for pair in self.stats.values() for a in pair):
            raise ValueError('Nonfinite inner-FIT model statistics')
        self.refresh()


def score(F,P,rows,weight=None):
    rows=np.asarray(rows,dtype=int);y=F['y'][rows].astype(float);m=F['m'][rows].astype(bool)
    w=F['w'][rows].astype(float) if weight is None else np.asarray(weight,dtype=float)
    observed=np.where(m,y,-np.inf);peak=observed.max(1);pp=np.where(m,P,-np.inf).max(1)
    severe=peak>=.1;positive=np.any(m & (y>0),axis=1);result={}
    cohorts={'all':np.ones(len(rows),bool),'S':severe,'nonS':~severe,'all_positive':positive}
    cohorts.update({'regime:'+str(r):F['regime'][rows]==r for r in REGIMES})
    for name,keep in cohorts.items():
        mass=float(np.sum(w[keep,None]*m[keep]));sse=float(np.sum(w[keep,None]*m[keep]*(P[keep]-y[keep])**2))
        row=dict(n=int(keep.sum()),observed_hours=int(m[keep].sum()),design_SSE=sse,
                 weighted_observation_mass=mass,design_RMSE=None if mass==0 else float(np.sqrt(sse/mass)))
        if name=='nonS':
            alarm=keep & (pp>=.1);row['severe_alarms']=dict(n=int(alarm.sum()),weighted_rate=None if not w[keep].sum() else float(w[alarm].sum()/w[keep].sum()))
        if name=='S' and keep.any():
            ratio=pp[keep]/peak[keep];order=np.argsort(ratio,kind='stable');ww=w[keep][order]
            positions=(np.cumsum(ww)-.5*ww)/ww.sum()
            row['peak_ratio_design_weight_median']=float(np.interp(.5,positions,ratio[order]))
        result[name]=row
    return result


def export(e,data,rows):
    result=G.export(e,data['F'],rows)
    result['idx']=data['unit'][rows]
    result.update({k:data['F'][k][rows] for k in ('y','m','y0',*METADATA_MEMBERS)})
    result['m']=result['m'].astype(bool)
    result['origin_observed']=data['F']['origin_observed'][rows]
    result.update(merged_group=data['group'][rows],original_fold=data['fold'][rows],
                  panel=data['panel'][rows],pi=data['pi'][rows])
    if not np.isfinite(result['P']).all() or np.any((result['P']<0)|(result['P']>1)):
        raise ValueError('Nonfinite/out-of-range prediction')
    return result


def save_export(path,result):
    with path.open('xb') as f:np.savez_compressed(f,**result)


def peak_rss_gib():
    raw=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(raw/(2**30 if sys.platform=='darwin' else 2**20))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=ARMS,required=True)
    parser.add_argument('--heldout-fold',type=int,choices=(2,3),required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--scope',type=Path,required=True)
    parser.add_argument('--scope-commit',required=True)
    parser.add_argument('--preflight',action='store_true',help='Exactly3FIT-only updates; discard model; no heldout export')
    args=parser.parse_args();args.out.resolve().relative_to(ROOT);args.scope.resolve().relative_to(ROOT)
    if args.out.exists():raise FileExistsError('Preserve existing job directory: '+D.relative(args.out))
    commit,sources=verify_registration(args.scope_commit,args.scope)
    args.out.mkdir(parents=True,exist_ok=False)
    os.nice(max(0,15-os.getpriority(os.PRIO_PROCESS,0)))
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    started=time.monotonic();state_path=args.out/'RUN_STATUS.json'
    status(state_path,phase='loading',arm=args.arm,heldout_fold=args.heldout_fold,step=0,registration_commit=commit)
    try:
        with threadpool_limits(limits=2):
            data=load_data();indices=job_indices(data,args.heldout_fold);fit=indices['fit']
            if len(np.unique(data['F']['fips'][fit]))<32:raise ValueError('At least32unique fitting counties required')
            Ff,weight_info=design_weighted(data['F'],fit)
            e=Engine(Ff,fit,args.arm)
            stats={k:[np.asarray(a).tolist() for a in pair] for k,pair in e.stats.items()}
            D.write_json_new(args.out/'stats.json',stats)
            info=dict(schema='d09_job_protocol_v1',arm=args.arm,heldout_fold=args.heldout_fold,
                registration_commit=commit,source_sha256=sources,scope_sha256=D.sha(args.scope),
                data_sha256=D.DATA_SHA,split_sha256=data['split_sha256'],d08_frozen_sha256=data['frozen_sha256'],
                d08_manifest_sha256=data['frozen']['manifest_sha256'],d08_roster_sha256=data['frozen']['roster_sha256'],
                protocol=PROTOCOL,fit_weighting=weight_info,original_outer_fit_population=6350,
                fit_unit_ids=data['unit'][fit].tolist(),held_panel_unit_ids=data['unit'][indices['held_panel']].tolist(),
                held_full_unit_ids=data['unit'][indices['held_full']].tolist(),
                initial_host_parameter_sha256=e.initial_host_parameter_sha256,
                initial_kernel_parameter_sha256=e.initial_kernel_parameter_sha256,
                optimizer_covers_all_trainable=e.optimizer_covers_all_trainable,
                model_statistics_fit_unit_ids=data['unit'][fit].tolist(),
                kernel_geography_fit_metadata=None if e.model.kernel is None else e.model.kernel.fit_metadata,
                checkpoint_diagnostics_are_not_selection=True,preflight=args.preflight,
                resources=dict(nice=os.getpriority(os.PRIO_PROCESS,0),threads=2,pid=os.getpid()))
            D.write_json_new(args.out/'PROTOCOL.json',info)
            print(json.dumps(dict(arm=args.arm,heldout_fold=args.heldout_fold,fit=len(fit),held_panel=len(indices['held_panel']),held_full=len(indices['held_full']),preflight=args.preflight)),flush=True)
            metrics=[]
            preflight_updates=[]
            if args.preflight:
                if e.model.kernel is None:
                    raise ValueError('Registered resource preflight is only for old/new kernels')
                # Fit-only stress test opens the already initialized kernel.
                # No altered state is saved as a resumable checkpoint.
                e.step=200
                with torch.no_grad():e.model.kernel.alpha.fill_(.1)
                e.model.kernel.training_step.fill_(200)
                e.model.kernel.set_drop_override(1.0)
                e.refresh()

            def checkpoint(step):
                e.refresh();e.model.eval();row=dict(step=step,fit_loss=None if step==0 else e.last_loss)
                full=None
                if step==STEPS:
                    full=export(e,data,indices['held_full'])
                    save_export(args.out/f'step{step:04d}_held_full.npz',full)
                    row['held_full']=score(data['F'],full['P'],indices['held_full'])
                for name in ('fit','held_panel'):
                    ii=indices[name]
                    if name=='held_panel' and full is not None:
                        positions=np.searchsorted(indices['held_full'],ii)
                        if not np.array_equal(indices['held_full'][positions],ii):
                            raise ValueError('Heldout panel must be an exact subset of full heldout groups')
                        result={key:value[positions] for key,value in full.items()}
                    else:
                        result=export(e,data,ii)
                    save_export(args.out/f'step{step:04d}_{"fit_panel" if name=="fit" else name}.npz',result)
                    row['fit_panel' if name=='fit' else name]=score(data['F'],result['P'],ii)
                if e.model.kernel is not None:row['calibration']=dict(e.model.kernel.last_calibration)
                metrics.append(row)
                D.write_json_new(args.out/f'step{step:04d}_metrics.json',row)

            if not args.preflight:checkpoint(0)
            budget=PREFLIGHT_STEPS if args.preflight else STEPS
            for step in range(1,budget+1):
                update_start=time.monotonic()
                e.train_step()
                if step%10==0:e.refresh()
                if args.preflight:
                    missing=[n for n,p in e.model.kernel.named_parameters() if p.requires_grad and p.grad is None]
                    preflight_updates.append(dict(update=step,engine_step=e.step,loss=e.last_loss,
                        elapsed_seconds=time.monotonic()-update_start,peak_rss_gib=peak_rss_gib(),
                        finite_gradients=all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in e.model.parameters()),
                        kernel_parameters_missing_gradients=missing,
                        optimizer_covers_all_trainable=e.optimizer_covers_all_trainable,
                        alpha=float(e.model.kernel.alpha.detach()),drop_mask=float(e.model.kernel.last_mask)))
                if step<=3 or step%20==0:
                    status(state_path,phase='training',arm=args.arm,heldout_fold=args.heldout_fold,
                           step=step,budget=budget,loss=e.last_loss,seconds=time.monotonic()-started,peak_rss_gib=peak_rss_gib())
                    print(f'arm={args.arm} heldout={args.heldout_fold} step={step}/{budget} loss={e.last_loss:.8g} peakRSSGiB={peak_rss_gib():.3f}',flush=True)
                if not args.preflight and step in CHECKPOINTS:checkpoint(step)
            e.refresh();e.model.eval()
            if args.preflight:
                preflight_passed=len(preflight_updates)==3 and all(np.isfinite(u['loss']) and u['finite_gradients'] and
                    u['drop_mask']==1.0 and not u['kernel_parameters_missing_gradients'] and
                    u['optimizer_covers_all_trainable'] for u in preflight_updates)
                D.write_json_new(args.out/'PREFLIGHT_RESOURCE.json',dict(schema='d09_fit_only_resource_v1',
                    arm=args.arm,heldout_fold=args.heldout_fold,registration_commit=commit,source_sha256=sources,
                    update_count=3,forced_initial_engine_step=200,forced_initial_alpha=.1,
                    forced_drop_path_open=True,updates=preflight_updates,peak_rss_gib=peak_rss_gib(),
                    passed=preflight_passed,
                    ru_maxrss_native_units='bytes' if sys.platform=='darwin' else 'kilobytes',
                    heldout_scores_computed=False,resumable_model_saved=False,seconds=time.monotonic()-started))
                if not preflight_passed:
                    raise RuntimeError('Registered fit-only preflight checks failed; resource receipt preserved')
            if not args.preflight:
                with (args.out/'final.pt').open('xb') as f:
                    torch.save(dict(model_state=e.model.state_dict(),stats=e.stats,arm=args.arm,steps=STEPS,seed=SEED,
                                    fit_idx=e.fit_idx,fit_unit_ids=data['unit'][fit],registration_commit=commit,
                                    kernel_rng=None if e.model.kernel is None else e.model.kernel._drop.get_state()),f)
            if D.sha(D.FEATURES)!=D.DATA_SHA or D.sha(D.SPLITS)!=data['split_sha256'] or any(D.sha(ROOT/name)!=value for name,value in sources.items()):
                raise ValueError('Source/data changed during job; no DONE receipt')
            final=dict(info,schema='d09_preflight_done_v1' if args.preflight else 'd09_job_done_v1',
                       steps=budget,seed=SEED,fit_loss=e.last_loss,seconds=time.monotonic()-started,
                       peak_rss_gib=peak_rss_gib(),kernel_last_calibration=None if e.model.kernel is None else e.model.kernel.last_calibration,
                       outputs={p.name:D.sha(p) for p in sorted(args.out.iterdir()) if p.is_file() and p.name not in ('RUN_STATUS.json','RUN_STATUS.json.tmp')})
            D.write_json_new(args.out/('PREFLIGHT_DONE.json' if args.preflight else 'DONE.json'),final)
            status(state_path,phase='completed',arm=args.arm,heldout_fold=args.heldout_fold,step=budget,preflight=args.preflight)
            print(json.dumps(dict(status='completed',arm=args.arm,heldout_fold=args.heldout_fold,steps=budget,seconds=final['seconds'],peak_rss_gib=final['peak_rss_gib'])),flush=True)
    except Exception as exc:
        status(state_path,phase='failed',arm=args.arm,heldout_fold=args.heldout_fold,error_type=type(exc).__name__,error=str(exc))
        raise


if __name__=='__main__':main()
