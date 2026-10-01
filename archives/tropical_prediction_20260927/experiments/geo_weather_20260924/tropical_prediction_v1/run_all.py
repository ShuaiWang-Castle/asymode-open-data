"""Run the fixed campaign with memory-bounded concurrency and resumable caches."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, choices=[1, 2], default=2)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    root = here.parents[2]
    logs = root / 'runs/tropical_prediction_v1/logs'
    logs.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')

    def fold(k):
        with (logs / f'fold{k}.log').open('a') as handle:
            result = subprocess.run(
                [sys.executable, str(here / 'run_campaign.py'), '--fold', str(k)],
                cwd=root, env=env, stdout=handle, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'Fold {k} failed ({result.returncode}); inspect logs/fold{k}.log')
        return k

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(fold, k) for k in range(1, 6)]
        for job in as_completed(jobs):
            print(f'Fold {job.result()} completed or verified cached', flush=True)
    subprocess.run([sys.executable, str(here / 'summarize_campaign.py')],
                   cwd=root, env=env, check=True)


if __name__ == '__main__':
    main()
