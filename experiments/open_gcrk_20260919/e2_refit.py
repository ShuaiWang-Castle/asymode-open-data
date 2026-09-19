"""E2 (PREREG Amendment 2): W refit at the step count GCRK selected in the same cell.

For every (design, seed, fold) the W model is refit from scratch on the cell's development
set for GCRK's t* (from GCRK's DONE.json) with the cell's seed, and its OUTER rollouts are
exported. Outputs: runs/open_gcrk_20260919/e2/<design>/seed*/fold*/W/{final.pt, outer.npz,
DONE.json}.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import datetime as dt
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch

import common as C

sys.path.insert(0, str(C.ROOT / "src"))
from asymode import gcrk_train as G  # noqa: E402

OUT = C.RUNS / "e2"


def one(design: str, seed: int, fold: str) -> str:
    torch.set_num_threads(1)
    t0 = time.monotonic()
    d = OUT / design / f"seed{seed}" / f"fold{int(fold):02d}" / "W"
    if (d / "DONE.json").exists():
        return f"exists {d}"
    d.mkdir(parents=True, exist_ok=True)
    F = C.load_features()
    spec = C.load_splits()[design][fold]
    g = json.loads((C.cell(design, seed, fold, "GCRK") / "DONE.json").read_text())
    w = json.loads((C.cell(design, seed, fold, "W") / "DONE.json").read_text())
    steps = int(g["best_step"])
    e = G.refit(F, np.array(spec["dev"]), steps, seed, "W")
    torch.save(e.snapshot(), d / "final.pt")
    np.savez_compressed(d / "outer.npz", **G.export(e, F, np.array(spec["outer"])))
    done = dict(design=design, seed=seed, fold=int(fold), arm="W", steps=steps, gcrk_t_star=steps,
                w_t_star=int(w["best_step"]), seconds=time.monotonic() - t0,
                completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(), torch=torch.__version__)
    (d / "DONE.json").write_text(json.dumps(done, indent=1) + "\n")
    return f"DONE {design} seed {seed} fold {fold} steps {steps} (W t* {w['best_step']}) {time.monotonic() - t0:.0f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--designs", nargs="+", default=["main", "loeo"])
    a = ap.parse_args()
    tasks = [(dsn, s, f) for dsn in a.designs for s in C.SEEDS for f in sorted(C.load_splits()[dsn], key=int)]
    print(f"{len(tasks)} cells, {a.workers} workers", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(one, *t) for t in tasks]
        for fu in as_completed(futs):
            print(fu.result(), flush=True)
    print("E2 COMPLETE", flush=True)


if __name__ == "__main__":
    main()
