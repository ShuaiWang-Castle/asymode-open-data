"""One disposable full-fit resource/finite-gradient check on registered public D.

No held-out outcomes, scores, exports or reusable trained checkpoint are produced.
The opened kernel and disabled drop-path deliberately exercise the expensive path.
The finite five-fold runner always starts each model from its registered seed.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import resource
import sys
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "2"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print("Plan: one disposable registered CRK full-fit update, fold 1, two threads, nice>=15.")
        return
    if os.getpriority(os.PRIO_PROCESS, 0) < 15:
        os.nice(15 - os.getpriority(os.PRIO_PROCESS, 0))
    import numpy as np
    import torch
    import screen
    import run_cr_screen as runner
    import evaluate_cr_tail as evaluation
    torch.set_num_threads(2)
    commit = runner.committed_sources()
    source_hashes = {name: evaluation.sha(runner.ROOT / name) for name in runner.SOURCES}
    output = runner.ROOT / "runs/geo_weather_20260924/i20_preflight" / f"{commit[:12]}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = (output.parent / "PREFLIGHT.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if output.exists():
        raise FileExistsError("Preflight already recorded; inspect it before repeating")
    feature_hash = evaluation.sha(evaluation.FEATURES)
    started = time.monotonic()
    features = screen.load("v1D")
    split = json.loads(screen.SPLIT_FILES["v1D"].read_text())
    fit = np.asarray(split["event"]["1"]["dev"])
    features = screen.design_weighted(features, fit)
    engine = screen.G.Engine(features, fit, None, 0, "CRK+Cin")
    kernel = engine.model.kernel
    if kernel.fit_metadata["n_unique_counties"] < 32:
        raise RuntimeError("Real fit requires 32 distinct geography landmarks")
    initialization_seconds = time.monotonic() - started
    engine.step = 200  # Full opening, with a disposable model, not continuation of a run.
    kernel.training_step.fill_(200)
    with torch.no_grad():
        kernel.alpha.fill_(0.1)
    kernel.set_drop_override(1.0)
    update_started = time.monotonic()
    engine.train_step()  # Checks finite full-fit loss and every accumulated gradient.
    update_seconds = time.monotonic() - update_started
    gradient_norms = {name: float(p.grad.norm()) for name, p in kernel.named_parameters()
                      if p.grad is not None and name not in ("weight", "bias")}
    if not gradient_norms or not all(np.isfinite(list(gradient_norms.values()))):
        raise RuntimeError("Missing or nonfinite kernel gradients")
    if source_hashes != {name: evaluation.sha(runner.ROOT / name) for name in runner.SOURCES}:
        raise RuntimeError("Registered source changed during the preflight")
    if feature_hash != evaluation.sha(evaluation.FEATURES):
        raise RuntimeError("Public-D input file changed during the preflight")
    result = dict(
        status="passed", scope="disposable training-only resource check; no outer prediction or model selection",
        git_commit=commit, source_sha256=source_hashes,
        feature_sha256=feature_hash,
        data="v1D", fold=1, fit_county_events=len(fit),
        fit_unique_counties=kernel.fit_metadata["n_unique_counties"],
        new_parameters=kernel.n_new_parameters(), microbatch=engine.microbatch_size,
        threads=torch.get_num_threads(), nice=os.getpriority(os.PRIO_PROCESS, 0),
        initialization_seconds=initialization_seconds, full_fit_update_seconds=update_seconds,
        timing_scope="cold full-fit update including fit calibration and first JIT/optimizer initialization",
        max_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        * (1 if sys.platform == "darwin" else 1024),
        gradient_norms=gradient_norms, benchmark_opening_alpha=.1, benchmark_drop_mask=1,
        model_discarded=True, heldout_scores_computed=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    evaluation.write_new(output, result)
    print(json.dumps({key: result[key] for key in
                     ("status", "fit_county_events", "new_parameters", "microbatch",
                      "initialization_seconds", "full_fit_update_seconds", "max_rss_bytes")}), flush=True)


if __name__ == "__main__":
    main()
