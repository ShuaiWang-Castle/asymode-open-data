"""I20 compute-only handoff: preserve the live fold, run at most three folds.

The original model/evaluation sources and SCREEN_RUN remain immutable. A separate
compute amendment records this authorized resource change. Only the identified
serial coordinator is terminated; training processes are never signalled.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

import run_cr_screen as R

EXTRA_SOURCES = [
    'experiments/geo_weather_20260924/run_cr_parallel.py',
    'experiments/geo_weather_20260924/notes/I20_COMPUTE_AMENDMENT_20260929.md',
]


def process_identity(pid):
    result = subprocess.run(['ps', '-p', str(pid), '-o', 'pid=,ppid=,stat=,lstart=,command='],
                            capture_output=True, text=True, env={**os.environ, 'LC_ALL': 'C'})
    if result.returncode == 1 and not result.stdout.strip():
        return None
    result.check_returncode()
    parts = result.stdout.strip().split(maxsplit=8)
    if len(parts) != 9:
        raise RuntimeError('Cannot verify process identity')
    if parts[2].startswith('Z'):
        return None
    return dict(pid=int(parts[0]), ppid=int(parts[1]), started=' '.join(parts[3:8]), command=parts[8])


def same_process(expected, current):
    return current is not None and all(expected[k] == current[k] for k in ('pid', 'started', 'command'))


def command_option(tokens, name, value):
    return any(tokens[i:i+2] == [name, str(value)] for i in range(len(tokens)-1))


def check_adoption(status, parent, child, runner_pid, fold, child_pid):
    if status.get('status') != 'training' or status.get('runner_pid') != runner_pid:
        raise RuntimeError('Serial runner status does not match requested handoff')
    if status.get('active') != {str(fold): child_pid}:
        raise RuntimeError('Requested adoption does not match the sole active fold')
    if parent is None or child is None or parent['pid'] != runner_pid or child['pid'] != child_pid:
        raise RuntimeError('Both identified processes must be alive before handoff')
    pt, ct = shlex.split(parent['command']), shlex.split(child['command'])
    if str(R.HERE / 'run_cr_screen.py') not in pt or '--execute' not in pt:
        raise RuntimeError('Parent is not the registered serial coordinator')
    if str(R.HERE / 'screen.py') not in ct or child['ppid'] != runner_pid:
        raise RuntimeError('Child is not owned by the serial coordinator')
    for name, value in [('--label', R.LABEL), ('--arm', R.ARM), ('--data', 'v1D'),
                        ('--folds', fold), ('--steps', 900), ('--seed', 0), ('--threads', 2)]:
        if not command_option(ct, name, value):
            raise RuntimeError(f'Adopted training command differs: {name}')


def pending_folds(done, active):
    if set(done) & set(active) or not (set(done) | set(active)) <= set(range(1, 6)):
        raise RuntimeError('Invalid or overlapping fold ownership')
    return [k for k in range(1, 6) if k not in done and k not in active]


def worker_count(path):
    workers = int(path.read_text().strip())
    if workers not in (1, 2, 3):
        raise ValueError('PARALLEL_WORKERS must be 1, 2 or 3')
    return workers


def terminate_coordinator(expected):
    # Recheck PID plus start time and command immediately before the sole signal.
    if not same_process(expected, process_identity(expected['pid'])):
        raise RuntimeError('Serial coordinator identity changed before handoff')
    os.kill(expected['pid'], signal.SIGTERM)


def registered_extra(E):
    tracked = subprocess.check_output(['git', 'ls-files', '-z', '--', *EXTRA_SOURCES], cwd=R.ROOT).decode().split('\0')
    if set(EXTRA_SOURCES) - set(tracked):
        raise RuntimeError('Compute amendment sources must be committed first')
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--', *EXTRA_SOURCES], cwd=R.ROOT, check=True)
    return {name: E.sha(R.ROOT / name) for name in EXTRA_SOURCES}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--runner-pid', type=int)
    ap.add_argument('--adopt-fold', type=int, choices=range(1, 6))
    ap.add_argument('--adopt-pid', type=int)
    ap.add_argument('--workers', type=int, choices=(2, 3), default=3)
    a = ap.parse_args()
    if not a.execute:
        print(json.dumps(dict(train=False, workers=a.workers, threads=2, preserve_live_training=True)))
        return
    if None in (a.runner_pid, a.adopt_fold, a.adopt_pid):
        ap.error('Explicit serial runner and live fold identities are required')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '2'
    if os.getpriority(os.PRIO_PROCESS, 0) < 15:
        os.nice(15 - os.getpriority(os.PRIO_PROCESS, 0))
    import evaluate_cr_tail as E
    extra_hashes = registered_extra(E)
    frozen = json.loads((R.RUN / 'SCREEN_RUN.json').read_text())
    R.unchanged(frozen, E)
    guard = (R.RUN / 'PARALLEL.lock').open('a+')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    amendment_path = R.RUN / 'COMPUTE_AMENDMENT.json'
    if amendment_path.exists():
        raise RuntimeError('Existing compute handoff must be inspected; no automatic second takeover')
    status_path = R.RUN / 'RUN_STATUS.json'
    old_status = json.loads(status_path.read_text())
    parent, child = process_identity(a.runner_pid), process_identity(a.adopt_pid)
    check_adoption(old_status, parent, child, a.runner_pid, a.adopt_fold, a.adopt_pid)
    data = E.load_outcomes()
    _, host_receipts = E.load_predictions(R.HOST, 'W+Cin', data['expected'], len(data['y']))
    if json.loads((R.RUN / 'HOST_REFERENCE.json').read_text()) != host_receipts:
        raise RuntimeError('Frozen host changed')
    done = [k for k in range(1, 6) if k != a.adopt_fold and R.completion(k, frozen, data, E)]
    if (R.RUN / f'fold{a.adopt_fold:02d}' / 'RUNNER_RECEIPT.json').exists():
        raise RuntimeError('Adopted fold already has a receipt; inspect transition first')
    amendment = dict(kind='compute_only', authorization='PI freed CPU and explicitly authorized multiple processes/threads',
        original_registration=frozen['git_commit'], original_manifest_sha256=E.sha(R.RUN / 'SCREEN_RUN.json'),
        original_status=old_status, old_runner=parent, adopted_fold=a.adopt_fold, adopted_process=child,
        workers=a.workers, maximum_workers=3, threads_per_fold=2, nice_minimum=15,
        model_data_loss_seed_steps_unchanged=True, controller_sha256=extra_hashes,
        controller_commit=R.committed_sources(), created_unix=time.time())
    E.write_new(amendment_path, amendment)
    workers_path = R.RUN / 'PARALLEL_WORKERS'
    with workers_path.open('x') as f:
        f.write(str(a.workers) + '\n')
    terminate_coordinator(parent)
    lock = (R.RUN / 'RUNNER.lock').open('a+')
    deadline = time.monotonic() + 10
    while True:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.monotonic() > deadline:
                raise RuntimeError('Serial coordinator did not release its lock; child left untouched')
            time.sleep(.1)
    if not same_process(child, process_identity(a.adopt_pid)):
        raise RuntimeError('Adopted process changed during handoff; artifacts preserved')
    active = {a.adopt_fold: dict(pid=a.adopt_pid, identity=child, process=None)}
    print(f'I20 adopted fold {a.adopt_fold}, pid {a.adopt_pid}; coordinator {os.getpid()}', flush=True)

    def unchanged():
        R.unchanged(frozen, E)
        if any(E.sha(R.ROOT / p) != h for p, h in extra_hashes.items()):
            raise RuntimeError('Compute controller source changed')

    def status(state, **extra):
        R.atomic_status(status_path, dict(status=state, runner_pid=os.getpid(), controller='parallel',
            active={str(k): v['pid'] for k, v in active.items()}, done=sorted(done),
            pending=pending_folds(done, active), workers=worker_count(workers_path),
            threads=2, updated_unix=time.time(), **extra))

    try:
        while active or len(done) < 5:
            for k, job in list(active.items()):
                if job['process'] is not None:
                    rc = job['process'].poll()
                    if rc is None:
                        continue
                    if rc != 0:
                        raise RuntimeError(f'Fold {k} exited {rc}; other training processes left untouched')
                else:
                    current = process_identity(job['pid'])
                    if current is not None:
                        if not same_process(job['identity'], current):
                            raise RuntimeError('Adopted PID was reused; no process signalled')
                        continue
                unchanged()
                E.write_new(R.RUN / f'fold{k:02d}' / 'RUNNER_RECEIPT.json', R.fold_receipt(k, frozen, data, E))
                done.append(k)
                del active[k]
                print(f'I20 validated fold {k}', flush=True)
            for k in pending_folds(done, active):
                if len(active) >= worker_count(workers_path):
                    break
                unchanged()
                if (R.RUN / f'fold{k:02d}').exists():
                    raise RuntimeError(f'Fold {k} directory already exists; preserved')
                cmd = [sys.executable, '-u', str(R.HERE / 'screen.py'), '--label', R.LABEL,
                    '--arm', R.ARM, '--data', 'v1D', '--design', 'event', '--design-weights',
                    '--seed', '0', '--steps', '900', '--folds', str(k), '--threads', '2']
                with (R.HERE / 'logs' / f'screen_{R.LABEL}_f{k}.log').open('x') as stream:
                    proc = subprocess.Popen(cmd, cwd=R.ROOT, stdin=subprocess.DEVNULL,
                                            stdout=stream, stderr=subprocess.STDOUT)
                active[k] = dict(pid=proc.pid, process=proc)
                print(f'I20 launched fold {k}, pid {proc.pid}', flush=True)
            status('training')
            if active:
                time.sleep(5)
        unchanged()
        status('evaluating')
        R.evaluate(frozen, data, E)
        status('completed', evaluated=True)
        print('I20 complete; no additional arms, seeds or updates queued.', flush=True)
    except BaseException as error:
        status('stopped', error=f'{type(error).__name__}: {error}',
               note='Other running folds and partial artifacts preserved; inspect actual PIDs before recovery')
        raise


if __name__ == '__main__':
    main()
