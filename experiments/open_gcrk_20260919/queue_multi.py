"""Run several designs of one round through one worker pool, seed by seed (PREREG Amendment 3).

Tasks are ordered by seed, then GCRK before W (GCRK cells are slower), then design and fold, and
dispatched to `run.py worker` subprocesses with OPEN_GCRK_ROUND set. A cell with DONE.json is
skipped, so the queue can be restarted.
Usage: OPEN_GCRK_ROUND=e3r2 python queue_multi.py --designs main event --seeds 0 1 2 3 4 --workers 4
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

import run as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--designs", nargs="+", default=["main", "event"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(R.SEEDS))
    ap.add_argument("--arms", nargs="+", default=list(R.ARMS))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    sp = json.loads(R.SPLITS.read_text())
    tasks = [(s, arm, d, int(f)) for s in a.seeds for arm in a.arms for d in a.designs for f in sorted(sp[d], key=int)
             if not (R.cell_dir(d, s, int(f), arm) / "DONE.json").exists()]
    tasks.sort(key=lambda x: (x[0], x[1] != "GCRK", x[2], x[3]))
    logs = R.RUNS / "logs"; logs.mkdir(parents=True, exist_ok=True)
    live = []
    print(f"round {R.ROUND}: {len(tasks)} cells pending, {a.workers} workers", flush=True)
    while tasks or live:
        while tasks and len(live) < a.workers:
            s, arm, d, f = tasks.pop(0)
            fh = open(logs / f"{d}_{arm}_s{s}_f{f:02d}.log", "a")
            p = subprocess.Popen([sys.executable, "-u", str(R.HERE / "run.py"), "worker", "--design", d, "--seed", str(s),
                                  "--fold", str(f), "--arm", arm], stdout=fh, stderr=subprocess.STDOUT, cwd=R.HERE)
            live.append((p, fh, s, arm, d, f)); print("START", s, arm, d, f, p.pid, time.strftime("%H:%M"), flush=True)
        for item in list(live):
            p, fh, s, arm, d, f = item
            if p.poll() is not None:
                fh.close(); live.remove(item)
                print("FINISH" if p.returncode == 0 else f"FAILED rc={p.returncode}", s, arm, d, f, time.strftime("%H:%M"),
                      flush=True)
        time.sleep(5)
    print("QUEUE COMPLETE", R.ROUND, flush=True)


if __name__ == "__main__":
    main()
