"""Finite D09 queue: six registered tasks, maximum two owned workers."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCOPE = HERE / 'notes/D09_INCREMENTAL_WRITE_DESIGN_20260929_ZH.md'
RUN = ROOT / 'runs/geo_weather_20260924/d09_write_screen_20260929'
JOBS = [(arm, fold) for fold in (2, 3) for arm in ('host', 'crk', 'new')]
SOURCES = [HERE / name for name in (
    'd09_queue.py', 'd09_train.py', 'd09_write.py', 'd09_score.py',
    'test_d09_write.py', 'test_d09_train.py', 'test_d09_score.py', 'test_d09_queue.py',
    'notes/D09_INCREMENTAL_WRITE_DESIGN_20260929_ZH.md',
    'notes/D09_WRITE_CONTROL_MATH_REVIEW_20260929_ZH.md')]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024 * 1024): h.update(block)
    return h.hexdigest()


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def registered(commit):
    full = subprocess.check_output(['git', 'rev-parse', '--verify', commit+'^{commit}'], cwd=ROOT, text=True).strip()
    for path in SOURCES:
        blob = subprocess.check_output(['git','show',full+':'+relative(path)],cwd=ROOT)
        if hashlib.sha256(blob).hexdigest() != sha(path):
            raise RuntimeError('Registered D09 source/scope mismatch: '+relative(path))
    return full, {relative(p):sha(p) for p in SOURCES}


def atomic_status(path, value):
    temporary = path.with_suffix('.next')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def worker_count(path):
    value = int(Path(path).read_text().strip())
    if value not in (1,2): raise ValueError('WORKERS must be 1 or 2')
    return value


def preflight_receipt(path, arm, commit):
    """Check actual open-kernel resource work, not a bare success flag."""
    path=Path(path)
    done=json.loads(path.read_text())
    resource_path=path.parent/'PREFLIGHT_RESOURCE.json'
    resource=json.loads(resource_path.read_text())
    if (done.get('schema')!='d09_preflight_done_v1' or not done.get('preflight') or
            done.get('arm')!=arm or done.get('heldout_fold')!=3 or done.get('steps')!=3 or
            done.get('registration_commit')!=commit or resource.get('registration_commit')!=commit):
        raise RuntimeError('Preflight protocol/commit mismatch: '+arm)
    for name,h in done['source_sha256'].items():
        if sha(ROOT/name)!=h:raise RuntimeError('Preflight source changed: '+name)
    for name,h in done['outputs'].items():
        if sha(path.parent/name)!=h:raise RuntimeError('Preflight output changed: '+name)
    updates=resource.get('updates',[])
    if (resource.get('schema')!='d09_fit_only_resource_v1' or resource.get('passed') is not True or
            resource.get('arm')!=arm or resource.get('heldout_fold')!=3 or len(updates)!=3 or
            resource.get('heldout_scores_computed') is not False or
            resource.get('resumable_model_saved') is not False or
            any(not row['finite_gradients'] or row['drop_mask']!=1.0 or
                not math.isfinite(row['loss']) or row['kernel_parameters_missing_gradients'] or
                not row['optimizer_covers_all_trainable'] or row['engine_step']!=201+i
                for i,row in enumerate(updates)) or
            resource.get('forced_initial_engine_step')!=200 or resource.get('forced_initial_alpha')!=.1):
        raise RuntimeError('Preflight did not cover registered open-kernel updates: '+arm)
    if (path.parent/'final.pt').exists() or list(path.parent.glob('*.npz')):
        raise RuntimeError('Preflight saved model/export; preserve for inspection')
    return resource


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scope-commit', required=True)
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--preflight-crk', type=Path, required=True)
    ap.add_argument('--preflight-new', type=Path, required=True)
    args = ap.parse_args()
    commit, hashes = registered(args.scope_commit)
    if not args.execute:
        print(json.dumps(dict(tasks=JOBS, workers=2, steps=900, seed=0, registered_commit=commit)))
        return
    # Real preflight receipts must be inspected by the caller before launch.
    preflight = {name:preflight_receipt(path,name,commit) for name,path in
                 [('crk',args.preflight_crk),('new',args.preflight_new)]}
    if os.getpriority(os.PRIO_PROCESS,0) < 15:
        os.nice(15-os.getpriority(os.PRIO_PROCESS,0))
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key]='2'
    RUN.mkdir(parents=True, exist_ok=False)  # Preserve every existing or empty run.
    (RUN/'WORKERS').write_text('2\n')
    jobs_manifest=dict(schema='d09_jobs_v1',registration_commit=commit,
        jobs=[dict(arm=arm,heldout_fold=fold,path=relative(RUN/f'{arm}_f{fold}')) for arm,fold in JOBS])
    (RUN/'JOBS.json').write_text(json.dumps(jobs_manifest,indent=2,allow_nan=False)+'\n')
    (RUN/'SCREEN_RUN.json').write_text(json.dumps(dict(commit=commit, source_sha256=hashes,
        jobs=JOBS, workers_maximum=2, threads_per_worker=2, steps=900, seed=0,
        jobs_manifest_sha256=sha(RUN/'JOBS.json'),
        preflight=preflight, authorization='PI requested new design and actual attempts after D08',
        started_unix=time.time()),indent=2,allow_nan=False)+'\n')
    pending=list(JOBS);active={};done=[];failed=[]
    def unchanged():
        if any(sha(ROOT/p)!=h for p,h in hashes.items()):
            raise RuntimeError('Frozen D09 source changed during queue')
    def status(state):
        atomic_status(RUN/'RUN_STATUS.json',dict(status=state,runner_pid=os.getpid(),
            active={key:row['process'].pid for key,row in active.items()}, pending=pending,
            done=done, failed=failed, workers=int((RUN/'WORKERS').read_text()),updated_unix=time.time()))
    while pending or active:
        unchanged()
        workers=worker_count(RUN/'WORKERS')
        while pending and len(active)<workers and not failed:
            arm,fold=pending.pop(0);key=f'{arm}_f{fold}'
            jobdir=RUN/key
            if jobdir.exists():raise RuntimeError('Preserve existing task directory: '+key)
            log=(RUN/(key+'.log')).open('x')
            command=[sys.executable,str(HERE/'d09_train.py'),'--arm',arm,'--heldout-fold',str(fold),
                '--out',str(jobdir),'--scope',str(SCOPE),'--scope-commit',commit]
            process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            active[key]=dict(process=process,log=log)
            print(f'D09 started {key} PID {process.pid}',flush=True)
        for key,row in list(active.items()):
            code=row['process'].poll()
            if code is None:continue
            row['log'].close();del active[key]
            receipt=RUN/key/'DONE.json'
            if code !=0 or not receipt.exists():
                failed.append(dict(job=key,exit_code=code,done_exists=receipt.exists()))
                print(f'D09 failed {key}: exit {code}',flush=True)
            else:
                done.append(key);print(f'D09 finished {key}',flush=True)
        status('failure_waiting_for_owned_workers' if failed else 'training')
        if failed and not active:break
        if pending or active:time.sleep(5)
    if failed:
        status('failed');raise SystemExit(1)
    unchanged();status('training_complete')
    print('D09 all six tasks complete; independent scoring follows',flush=True)


if __name__=='__main__':main()
