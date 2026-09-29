"""Finite I20: five public-D event folds, seed zero, one child, two threads.

Default is a plan only. --execute requires committed, unchanged registered
sources. Completed artifacts are checked and reused; partial folders are never
overwritten. No process outside this runner's child is inspected or signalled.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LABEL, ARM, HOST = 'v1_crk_s0', 'CRK+Cin', 'v1_host_s0'
RUN = ROOT / 'runs/geo_weather_20260924' / LABEL
SOURCES = [
    'src/asymode/controlled_relaxation.py', 'src/asymode/controlled_relaxation_scan.py', 'src/asymode/gcrk.py',
    'src/asymode/gcrk_train.py', 'src/asymode/asym_host.py',
    'experiments/geo_weather_20260924/screen.py',
    'experiments/geo_weather_20260924/run_cr_screen.py',
    'experiments/geo_weather_20260924/preflight_cr_training.py',
    'experiments/geo_weather_20260924/evaluate_cr_tail.py',
    'experiments/geo_weather_20260924/evaluate_v1.py',
    'experiments/geo_weather_20260924/d04_data.py',
    'experiments/geo_weather_20260924/splits_v1D.json',
    'experiments/geo_weather_20260924/notes/KERNEL_CONTROLLED_RELAXATION_DESIGN_20260928.md',
    'experiments/geo_weather_20260924/notes/I20_CONTROLLED_RESPONSE_SCREEN_20260928.md',
]
RESULT_NAMES = ['screen_i20_s0_headline_vs_host.json', 'screen_i20_s0_all5_vs_host.json',
                'i20_cr_tail_s0.json', 'screen_i20_s0_verdict.json']


def atomic_status(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=1, allow_nan=False) + '\n')
    temporary.replace(path)


def committed_sources():
    tracked = subprocess.check_output(['git', 'ls-files', '-z', '--', *SOURCES], cwd=ROOT).decode().split('\0')
    if set(SOURCES) - set(tracked): raise RuntimeError('Register and commit every I20 source before --execute')
    result = subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', *SOURCES], cwd=ROOT)
    if result.returncode != 0: raise RuntimeError('Registered I20 sources differ from committed HEAD')
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def manifest(primary, E):
    feature = E.FEATURES.stat()
    return dict(label=LABEL, arm=ARM, host=HOST, data='v1D', design='event', design_weights=True,
                seed=0, steps=900, folds=[1, 2, 3, 4, 5], workers=1, threads=2, nice_minimum=15,
                primary_cohort=primary, target_rmse_improvement=.10,
                bootstrap_draws=E.BOOTSTRAP, bootstrap_seed=E.SEED,
                source_sha256={p: E.sha(ROOT / p) for p in SOURCES},
                feature_file=dict(path=str(E.FEATURES.relative_to(ROOT)), size=feature.st_size,
                                  mtime_ns=feature.st_mtime_ns, sha256=E.sha(E.FEATURES)),
                git_commit=committed_sources())


def unchanged(frozen, E):
    current = manifest(frozen['primary_cohort'], E)
    # Unrelated commits may happen; exact registered source hashes remain binding.
    for key in frozen:
        if key != 'git_commit' and current[key] != frozen[key]:
            raise RuntimeError(f'I20 frozen manifest changed: {key}')


def fold_receipt(k, frozen, data, E):
    import numpy as np
    folder = RUN / f'fold{k:02d}'
    metadata = E.check_done(folder, LABEL, ARM, k)
    with np.load(folder / 'outer.npz', allow_pickle=False) as z:
        idx = E.indices(z['idx'], len(data['y']), f'I20 fold {k}')
        if not np.array_equal(np.sort(idx), np.sort(data['expected'][k])):
            raise ValueError('I20 held-out membership differs')
        for field in ('P', 'u', 'r', 'raw_logit', 'P_closed', 'raw_logit_closed'):
            if field not in z: raise ValueError(f'I20 export missing {field}')
            values = z[field]
            if values.shape != (len(idx), 144) or not np.isfinite(values).all():
                raise ValueError(f'I20 invalid {field}')
            if field in ('P', 'P_closed') and np.any((values < 0) | (values > 1)):
                raise ValueError(f'I20 {field} outside [0,1]')
    return dict(metadata=metadata, source_sha256=frozen['source_sha256'],
                artifact_sha256={f: E.sha(folder / f) for f in ('outer.npz', 'final.pt', 'DONE.json')})


def completion(k, frozen, data, E):
    folder = RUN / f'fold{k:02d}'
    if not folder.exists(): return False
    if not (folder / 'DONE.json').exists():
        raise RuntimeError(f'fold {k} partial directory preserved; explicit recovery is required')
    receipt = folder / 'RUNNER_RECEIPT.json'
    if not receipt.exists():
        raise RuntimeError(f'fold {k} completed without runner receipt; inspect provenance before recovery')
    if json.loads(receipt.read_text()) != fold_receipt(k, frozen, data, E):
        raise RuntimeError(f'fold {k} artifact/source receipt mismatch')
    return True


def evaluation_complete(frozen, E):
    paths = [HERE / 'results/v1' / name for name in RESULT_NAMES]
    receipt_path = RUN / 'EVALUATION_DONE.json'
    if not any(path.exists() for path in paths) and not receipt_path.exists(): return False
    if not receipt_path.exists() or not all(path.exists() for path in paths):
        raise RuntimeError('Partial/pre-existing evaluation preserved; inspect before explicit recovery')
    receipt = json.loads(receipt_path.read_text())
    if receipt['source_sha256'] != frozen['source_sha256'] or receipt['primary_cohort'] != frozen['primary_cohort']:
        raise RuntimeError('Evaluation receipt differs from frozen registration')
    if receipt['artifact_sha256'] != {path.name: E.sha(path) for path in paths}:
        raise RuntimeError('Evaluation artifacts changed')
    return True


def evaluate(frozen, data, E):
    # Mandatory strict validation BEFORE the legacy evaluator's weaker loader.
    if not all(completion(k, frozen, data, E) for k in range(1, 6)):
        raise RuntimeError('Evaluation requires all five validated fold receipts')
    P, candidate_receipts = E.load_predictions(LABEL, ARM, data['expected'], len(data['y']))
    H, host_receipts = E.load_predictions(HOST, 'W+Cin', data['expected'], len(data['y']))
    if json.loads((RUN / 'HOST_REFERENCE.json').read_text()) != host_receipts:
        raise RuntimeError('Frozen host predictions/metadata changed during training')
    if evaluation_complete(frozen, E): return
    staging = Path(tempfile.mkdtemp(prefix='scores_', dir=RUN))
    for name, headline in [(RESULT_NAMES[0], ['tropical', 'winter']), (RESULT_NAMES[1], E.REGIMES)]:
        subprocess.run([sys.executable, str(HERE / 'evaluate_v1.py'), '--arm', LABEL, '--host', HOST,
                        '--headline', *headline, '--out', str(staging / name)], cwd=ROOT, check=True)
    tail = E.report(data, P, H, frozen['primary_cohort'])
    tail['provenance'] = dict(candidate=LABEL, host=HOST, steps=900, folds=[1, 2, 3, 4, 5], seed=0,
                              source_sha256=frozen['source_sha256'],
                              exports={LABEL: candidate_receipts, HOST: host_receipts})
    E.write_new(staging / RESULT_NAMES[2], tail)
    headline = json.loads((staging / RESULT_NAMES[0]).read_text())['w']
    changes = {}
    for r in E.REGIMES:
        v = headline['per_regime'][r]
        changes[r] = v['arm'] / v['host'] - 1 if v['host'] > 0 else None
    legacy_guard = all(changes[r] is not None and changes[r] <= .02
                       for r in ('synoptic_wind', 'convective', 'heavy_rain'))
    verdict = dict(label=LABEL, arm=ARM, baseline=HOST, seed=0, steps=900, folds=[1, 2, 3, 4, 5],
                   primary_target=tail['primary_target'],
                   legacy=dict(headline_balanced_mse_gain=headline['balanced_gain'],
                               all_regime_mse_change=changes,
                               nonheadline_2pct_mse_guardrail_met=legacy_guard,
                               note='Legacy extra report; it does not replace the newly registered primary target'),
                   status='five_folds_complete_scored_no_automatic_expansion',
                   limitation='Equal 900-update budget to frozen host_s0; not a convergence certificate or multi-seed result')
    E.write_new(staging / RESULT_NAMES[3], verdict)
    unchanged(frozen, E)
    for name in RESULT_NAMES:
        target = HERE / 'results/v1' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with (staging / name).open('rb') as source, target.open('xb') as dest:
            shutil.copyfileobj(source, dest)
    E.write_new(RUN / 'EVALUATION_DONE.json', dict(source_sha256=frozen['source_sha256'],
        primary_cohort=frozen['primary_cohort'],
        artifact_sha256={name: E.sha(HERE / 'results/v1' / name) for name in RESULT_NAMES}))
    print(json.dumps(E.clean(verdict), allow_nan=False), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--execute', action='store_true', help='Run the finite, committed five-fold plan')
    ap.add_argument('--primary-cohort', choices=['S', 'any_positive', 'J'], default='S')
    a = ap.parse_args()
    if not a.execute:
        print(json.dumps(dict(label=LABEL, arm=ARM, host=HOST, folds=[1, 2, 3, 4, 5], seed=0,
            steps=900, workers=1, threads=2, primary_cohort=a.primary_cohort, train=False,
            note='Plan only; --execute requires committed registration and sources.'), indent=1))
        return
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '2'
    if os.getpriority(os.PRIO_PROCESS, 0) < 15: os.nice(15 - os.getpriority(os.PRIO_PROCESS, 0))
    import evaluate_cr_tail as E
    committed_sources()
    RUN.mkdir(parents=True, exist_ok=True)
    lock = (RUN / 'RUNNER.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    frozen = manifest(a.primary_cohort, E)
    mp = RUN / 'SCREEN_RUN.json'
    if mp.exists():
        previous = json.loads(mp.read_text())
        for key in frozen:
            if key != 'git_commit' and previous[key] != frozen[key]:
                raise RuntimeError(f'Existing I20 manifest differs: {key}')
        frozen = previous
    else:
        E.write_new(mp, frozen)
    data = E.load_outcomes()
    _, host_receipts = E.load_predictions(HOST, 'W+Cin', data['expected'], len(data['y']))
    host_path = RUN / 'HOST_REFERENCE.json'
    if host_path.exists():
        if json.loads(host_path.read_text()) != host_receipts: raise RuntimeError('Frozen host exports changed')
    else: E.write_new(host_path, host_receipts)
    done = [k for k in range(1, 6) if completion(k, frozen, data, E)]
    status = RUN / 'RUN_STATUS.json'
    log_dir = HERE / 'logs'; log_dir.mkdir(exist_ok=True)
    try:
        for k in range(1, 6):
            if k in done: continue
            unchanged(frozen, E)
            # Do not pre-create the fold folder: screen.py owns creation.
            if (RUN / f'fold{k:02d}').exists(): raise RuntimeError('New partial fold directory appeared')
            cmd = [sys.executable, '-u', str(HERE / 'screen.py'), '--label', LABEL, '--arm', ARM,
                   '--data', 'v1D', '--design', 'event', '--design-weights', '--seed', '0',
                   '--steps', '900', '--folds', str(k), '--threads', '2']
            with (log_dir / f'screen_{LABEL}_f{k}.log').open('a') as stream:
                child = subprocess.Popen(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                print(f'I20 launched fold {k}, pid {child.pid}', flush=True)
                while child.poll() is None:
                    atomic_status(status, dict(status='training', runner_pid=os.getpid(),
                        active={str(k): child.pid}, done=done, pending=list(range(k+1, 6)),
                        updated_unix=time.time()))
                    time.sleep(5)
            if child.returncode != 0: raise RuntimeError(f'Fold {k} exited {child.returncode}; no further launches')
            unchanged(frozen, E)
            E.write_new(RUN / f'fold{k:02d}' / 'RUNNER_RECEIPT.json', fold_receipt(k, frozen, data, E))
            done.append(k)
            print(f'I20 validated fold {k}', flush=True)
        unchanged(frozen, E)
        atomic_status(status, dict(status='evaluating', runner_pid=os.getpid(), active={}, done=done,
                                   pending=[], updated_unix=time.time()))
        evaluate(frozen, data, E)
        atomic_status(status, dict(status='completed', runner_pid=os.getpid(), active={}, done=done,
                                   pending=[], evaluated=True, updated_unix=time.time()))
    except BaseException as error:
        atomic_status(status, dict(status='stopped', runner_pid=os.getpid(), done=done,
            error=f'{type(error).__name__}: {error}',
            note='Partial artifacts preserved; no other process was signalled', updated_unix=time.time()))
        raise
    print('I20 complete; no seeds, NULL arms, or further training queued.', flush=True)


if __name__ == '__main__': main()
