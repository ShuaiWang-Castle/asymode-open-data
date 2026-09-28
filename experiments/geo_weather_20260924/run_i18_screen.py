"""Finite, restart-safe I18 seed-0 screen and evaluation; no automatic expansion.

Run with the project Python. A local WORKERS file may contain 1 or 2; it only
controls future launches. No other session's process is inspected or signalled.
Training and large outputs stay in ignored runs/ and untracked logs/.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
LABEL='v1_gcrk_georms_s0'
ARM='GCRK+Cin-georms'
RUN=ROOT/'runs/geo_weather_20260924'/LABEL
LOG=HERE/'logs'
SOURCES=['src/asymode/gcrk.py','src/asymode/gcrk_train.py','src/asymode/asym_host.py',
         'experiments/geo_weather_20260924/screen.py',
         'experiments/geo_weather_20260924/compare_kernels_v1.py',
         'experiments/geo_weather_20260924/evaluate_v1.py',
         'experiments/geo_weather_20260924/run_i18_screen.py',
         'experiments/geo_weather_20260924/notes/I18_GEO_RMS_SCREEN_20260928.md']


def hashes():
    return {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES}


def complete(k):
    path=RUN/f'fold{k:02d}'
    if not (path/'DONE.json').exists():
        return False
    try:
        d=json.loads((path/'DONE.json').read_text())
    except json.JSONDecodeError:
        return False  # The training child may still be finishing its marker write.
    expected=dict(label=LABEL,arm=ARM,data='v1D',design='event',design_weights=True,fold=k,seed=0,steps=900)
    for key,value in expected.items():
        if d.get(key)!=value:
            raise RuntimeError(f'fold {k} incompatible metadata field {key}')
    if not all((path/f).exists() for f in ['final.pt','outer.npz']):
        raise RuntimeError(f'fold {k} has DONE but incomplete exports')
    return True


def validate_exports():
    import numpy as np
    import torch
    split=json.loads((HERE/'splits_v1D.json').read_text())
    n=split['n_units']; seen=np.zeros(n,int)
    for k in range(1,6):
        assert complete(k)
        path=RUN/f'fold{k:02d}'
        with np.load(path/'outer.npz') as z:
            idx=z['idx']
            assert np.array_equal(np.sort(idx),np.sort(split['event'][str(k)]['outer']))
            assert len(np.unique(idx))==len(idx)
            for key in ['P','u','r']:
                assert z[key].shape==(len(idx),144) and np.isfinite(z[key]).all()
            assert ((z['P']>=0)&(z['P']<=1)).all()
            seen[idx]+=1
        ck=torch.load(path/'final.pt',weights_only=False)
        assert ck['arm']==ARM and ck['steps']==900
        scale=ck['model_state']['damage.2.geo_rms_scale']
        assert scale.ndim==0 and torch.isfinite(scale) and scale>=.1
    assert (seen==1).all()


def evaluate():
    validate_exports()
    for ref,ref_label in [('host','v1_host_s0'),('gcrk','v1_gcrk_s0')]:
        for scope,extra in [('headline',['--headline','tropical','winter']),('all5',[])]:
            cmd=[sys.executable,str(HERE/'evaluate_v1.py'),'--arm',LABEL,'--host',ref_label,*extra,
                 '--out',f'results/v1/screen_i18_s0_{scope}_vs_{ref}.json']
            subprocess.run(cmd,cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(HERE/'compare_kernels_v1.py'),'--seed','0','--candidate',LABEL,
                    '--candidate-name','GCRK-RMS','--out','results/v1/kernels_georms_s0.json'],cwd=ROOT,check=True)
    get=lambda scope,ref:json.loads((HERE/f'results/v1/screen_i18_s0_{scope}_vs_{ref}.json').read_text())['w']
    host,twin=get('headline','host'),get('headline','gcrk')
    all5=get('all5','host')
    pooled=json.loads((HERE/'results/v1/kernels_georms_s0.json').read_text())
    metrics=pooled['metrics']
    gates=dict(headline_gt_1pct_vs_host=host['balanced_gain']<-.01,
        headline_better_than_gcrk=twin['balanced_gain']<0,
        pooled_h1_better_than_host=metrics['GCRK-RMS']['RMSE+1']<metrics['AsymODE']['RMSE+1'],
        pooled_h1_better_than_gcrk=metrics['GCRK-RMS']['RMSE+1']<metrics['GCRK']['RMSE+1'],
        all5_nonworse_than_host=all5['balanced_gain']<=0,
        nonheadline_noninferior=all(host['per_regime'][r]['arm']/host['per_regime'][r]['host']-1<=.02
                                     for r in ['synoptic_wind','convective','heavy_rain']))
    verdict=dict(label=LABEL,seed=0,folds=[1,2,3,4,5],gates=gates,
        verdict='candidate_for_discussion' if all(gates.values()) else 'screen_failed_no_seed_expansion',
        note='Single-seed development screen only; no trained NULL or extra seeds authorized by this runner.')
    (HERE/'results/v1/screen_i18_s0_verdict.json').write_text(json.dumps(verdict,indent=2)+'\n')
    print(json.dumps(verdict),flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--workers',type=int,choices=[1,2],default=1)
    ap.add_argument('--threads',type=int,default=2)
    a=ap.parse_args()
    if a.threads not in [1,2]:
        raise ValueError('use one or two threads on the shared machine')
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
        os.environ[key]=str(a.threads)
    if os.getpriority(os.PRIO_PROCESS,0)<15:
        os.nice(15-os.getpriority(os.PRIO_PROCESS,0))
    RUN.mkdir(parents=True,exist_ok=True); LOG.mkdir(exist_ok=True)
    lock=(RUN/'RUNNER.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    frozen=hashes()
    manifest=dict(label=LABEL,arm=ARM,seed=0,folds=[1,2,3,4,5],steps=900,threads=a.threads,
                  source_sha256=frozen,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    mp=RUN/'SCREEN_RUN.json'
    if mp.exists():
        old=json.loads(mp.read_text())
        for key in ['label','arm','seed','folds','steps','threads','source_sha256']:
            if old[key]!=manifest[key]: raise RuntimeError(f'run manifest mismatch: {key}')
    else:
        mp.write_text(json.dumps(manifest,indent=2)+'\n')
    control=RUN/'WORKERS'
    if not control.exists(): control.write_text(str(a.workers)+'\n')
    queue=[]
    for k in range(1,6):
        if complete(k): continue
        path=RUN/f'fold{k:02d}'
        # screen.py creates an empty directory before its long refit. Even an
        # empty directory can belong to a live orphaned child; never relaunch it.
        if path.exists():
            raise RuntimeError(f'fold {k} is partial; preserve and inspect it before explicit recovery')
        queue.append(k)
    active={}
    status=RUN/'RUN_STATUS.json'
    failed=False
    while queue or active:
        limit=int(control.read_text().strip())
        if limit not in (1,2): raise RuntimeError('WORKERS must be 1 or 2')
        while queue and len(active)<limit and not failed:
            if hashes()!=frozen: raise RuntimeError('registered sources changed while queue was running')
            k=queue.pop(0)
            cmd=[sys.executable,'-u',str(HERE/'screen.py'),'--label',LABEL,'--arm',ARM,'--data','v1D',
                 '--design','event','--design-weights','--seed','0','--steps','900','--folds',str(k),
                 '--threads',str(a.threads)]
            lp=LOG/f'screen_{LABEL}_f{k}.log'
            stream=lp.open('a')
            proc=subprocess.Popen(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            active[k]=(proc,stream)
            print(f'launched fold {k}, pid {proc.pid}, workers {limit}',flush=True)
        for k,(proc,stream) in list(active.items()):
            ret=proc.poll()
            if ret is None: continue
            stream.close(); del active[k]
            if ret!=0 or not complete(k):
                failed=True; print(f'fold {k} failed, return code {ret}; no new folds launched',flush=True)
            else: print(f'completed fold {k}',flush=True)
        snapshot=dict(runner_pid=os.getpid(),active={k:p.pid for k,(p,_) in active.items()},pending=queue,
                      done=[k for k in range(1,6) if complete(k)],failed=failed,updated_unix=time.time())
        status.write_text(json.dumps(snapshot,indent=2)+'\n')
        if failed and not active: raise RuntimeError('screen stopped after a fold failure')
        if queue or active: time.sleep(5)
    if hashes()!=frozen: raise RuntimeError('registered sources changed before evaluation')
    evaluate()
    status.write_text(json.dumps(dict(runner_pid=os.getpid(),active={},pending=[],done=[1,2,3,4,5],
        failed=False,evaluated=True,updated_unix=time.time()),indent=2)+'\n')
    print('I18 finite screen finished; no more jobs queued.',flush=True)


if __name__=='__main__': main()
