"""Registered D04 full-D history-control sensitivity, AFTER five-fold completion.

This finite script performs eight fits per history specification (four arms,
two starts), using modal already-selected configurations and no new tuning.
Outputs are descriptive all-D fits, NEVER heldout predictions or independent
confirmation. 'unadjusted' removes observed outage-history controls only; other
nuisance terms, partial county pooling, and context terms remain.

It shares the original runner's OS LOCK, verifies all five immutable DONE
receipts and bundle checksums, and refuses any existing unfinished history
directory. No training-manifest source is modified. Importing does no fitting.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "2"

import argparse
from collections import Counter
import fcntl
import gc
import json
from pathlib import Path
import pickle
import time

import numpy as np

from d04_data import ROOT, load_panel, make_rows
from d04_features import fit_map, transform_weather
from d04_models import fit_model
from run_d04_response import (
    ARMS, DEFAULT_RUN, OUTER_SEEDS, OUTER_ITERATIONS, RIDGES, Queue,
    atomic_json, config_for, now, sha256, source_hashes, training_rows_and_scale,
)


def modal_configurations(bundles):
    """One vote per original fold; modal ties prefer stronger ridge/lower rank."""
    if len(bundles) != 5:
        raise ValueError("Exactly five completed original-fold bundles are required")
    folds = [int(bundle.get("outer_fold", bundle.get("fold", -1))) for bundle in bundles]
    if sorted(folds) != list(range(1, 6)):
        raise ValueError("Fold bundle identities must be exactly 1 through 5")
    records = {}
    for arm, specification in ARMS.items():
        votes = []
        for fold, bundle in sorted(zip(folds, bundles)):
            selected = bundle.get("selected", bundle.get("chosen"))
            if selected is None or arm not in selected:
                raise ValueError(f"Missing selected configuration for {arm}")
            config = selected[arm]["config"]
            rank, ridge = int(config["geo_rank"]), float(config["ridge"])
            if rank not in specification["ranks"] or ridge not in RIDGES:
                raise ValueError(f"Unregistered modal candidate for {arm}")
            votes.append(dict(fold=fold, geo_rank=rank, ridge=ridge))
        counts = Counter((vote["geo_rank"], vote["ridge"]) for vote in votes)
        rank, ridge = min(counts, key=lambda key: (-counts[key], -key[1], key[0]))
        records[arm] = dict(config=config_for(rank, ridge, OUTER_ITERATIONS),
                            selected_count=counts[(rank, ridge)], fold_votes=votes,
                            counts=[dict(geo_rank=key[0], ridge=key[1], count=count)
                                    for key, count in sorted(counts.items())])
    return records


def sensitivity_hashes(training_hashes):
    result = dict(training_hashes)
    this_file = Path(__file__).resolve()
    result[str(this_file.relative_to(ROOT))] = sha256(this_file)
    return result


def completed_training(run_dir):
    """Verify original source and completed artifacts before unpickling bundles."""
    manifest = json.loads((run_dir / "SCREEN_RUN.json").read_text())
    training_hashes = manifest["source_hashes"]
    if source_hashes() != training_hashes:
        raise RuntimeError("Current source/scope/splits differ from original D04 registration")
    queue = Queue(run_dir, training_hashes)
    bundles, bundle_sha = [], {}
    for fold in range(1, 6):
        if not queue.existing_done(fold):
            raise RuntimeError(f"Original fold {fold} is not complete; sensitivity cannot start")
        path = run_dir / f"fold{fold:02d}" / "bundle.pkl"
        bundle_sha[str(fold)] = sha256(path)
        with path.open("rb") as handle:
            bundle = pickle.load(handle)
        if (bundle["source_hashes"] != training_hashes
                or int(bundle.get("outer_fold", bundle.get("fold", -1))) != fold
                or bundle.get("response") != "adjacent_hour_fraction_change"):
            raise RuntimeError("Original bundle identity or response changed")
        bundles.append(bundle)
    modal = modal_configurations(bundles)
    del bundles
    gc.collect()
    return training_hashes, bundle_sha, modal


def check_existing(folder, hashes, bundle_sha, history):
    if not folder.exists():
        return False
    path = folder / "DONE.json"
    if not path.is_file():
        raise FileExistsError(f"Unfinished history directory preserved: {folder.name}")
    done = json.loads(path.read_text())
    if (done.get("source_hashes") != hashes or done.get("source_bundle_sha256") != bundle_sha
            or done.get("history") is not history or not done.get("descriptive_full_D")
            or done.get("is_oof") is not False):
        raise RuntimeError(f"Incompatible completed history fit: {folder.name}")
    if sha256(folder / "bundle.pkl") != done.get("bundle_sha256"):
        raise RuntimeError(f"History bundle checksum differs: {folder.name}")
    return True


def run(args):
    current_nice = os.getpriority(os.PRIO_PROCESS, 0)
    if current_nice < 15:
        os.nice(15 - current_nice)
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError("Original D04 run directory does not exist")
    # This is the same flock as the primary runner: a stale file alone is not a live lock.
    with (run_dir / "LOCK").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("D04 LOCK is held; no sensitivity fit was started") from error
        training_hashes, bundle_sha, modal = completed_training(run_dir)
        hashes = sensitivity_hashes(training_hashes)
        parent = run_dir / "history_sensitivity"
        histories = {"adjusted": True, "unadjusted": False}
        selected = list(histories) if args.history is None else ["adjusted" if args.history == "true" else "unadjusted"]
        # Inspect both specifications before starting either one.
        complete = []
        for name, history in histories.items():
            if check_existing(parent / name, hashes, bundle_sha, history):
                complete.append(name)
        todo = [name for name in selected if name not in complete]
        parent.mkdir(parents=True, exist_ok=True)
        manifest = dict(source_hashes=hashes, training_source_hashes=training_hashes,
                        source_bundle_sha256=bundle_sha, selected=modal,
                        starts=OUTER_SEEDS, max_iter=OUTER_ITERATIONS,
                        response="adjacent_hour_fraction_change", training_phase=0,
                        descriptive_full_D=True, is_oof=False,
                        histories=histories, expected_total_fits=16)
        manifest_path = parent / "SCREEN_RUN.json"
        if manifest_path.exists():
            old = json.loads(manifest_path.read_text())
            if old != manifest:
                raise RuntimeError("Existing sensitivity manifest does not match this finite design")
        else:
            atomic_json(manifest_path, manifest)
        lock.seek(0); lock.truncate()
        lock.write(json.dumps(dict(pid=os.getpid(), role="D04_history_sensitivity", started_at=now())) + "\n")
        lock.flush(); os.fsync(lock.fileno())
        started = time.monotonic()

        def status(state="running", **details):
            value = dict(state=state, pid=os.getpid(), updated_at=now(),
                         elapsed_seconds=time.monotonic() - started,
                         completed_histories=complete.copy(), selected_histories=selected,
                         source_hashes=hashes, descriptive_full_D=True, is_oof=False, **details)
            atomic_json(parent / "RUN_STATUS.json", value)
            print(json.dumps({key: val for key, val in value.items() if key != "source_hashes"},
                             ensure_ascii=False, allow_nan=False), flush=True)

        def unchanged():
            if (source_hashes() != training_hashes
                    or sensitivity_hashes(training_hashes) != hashes):
                raise RuntimeError("D04 source or sensitivity script changed during fitting")
            for fold, digest in bundle_sha.items():
                if sha256(run_dir / f"fold{int(fold):02d}" / "bundle.pkl") != digest:
                    raise RuntimeError("Original fold bundle changed during sensitivity fitting")

        try:
            if todo:
                status(phase="loading_public_D", queued_histories=todo)
                panel = load_panel()
                wx = transform_weather(panel)
                raw_rows = make_rows(panel, phase=0)
                rows, scale = training_rows_and_scale(raw_rows)
                del raw_rows
                for name in todo:
                    unchanged()
                    history = histories[name]
                    folder = parent / name
                    folder.mkdir()  # No overwrite/resume of an unfinished directory.
                    status(phase="fit_features", history_name=name, history=history)
                    feature_map, inputs = fit_map(panel, wx, rows, history=history)
                    models, records = {}, {}
                    for arm, specification in ARMS.items():
                        config = modal[arm]["config"]
                        candidates, starts = [], []
                        for seed in OUTER_SEEDS:
                            unchanged()
                            status(phase="full_D_fit", history_name=name, arm=arm,
                                   config=config, init_seed=seed)
                            x, n, g, c = inputs
                            tick = time.monotonic()
                            model = fit_model(x[:, :specification["columns"]], n, g, c,
                                              rows["y"] / scale, rows["w"], rows["county"],
                                              config, init_seed=seed)
                            if not model.diagnostics["finitefit"]:
                                raise RuntimeError("Nonfinite descriptive sensitivity fit")
                            starts.append(dict(init_seed=seed, config=config, diagnostics=model.diagnostics,
                                               seconds=time.monotonic() - tick))
                            candidates.append(model)
                            status(phase="full_D_fit_complete", history_name=name, arm=arm,
                                   init_seed=seed, loss=model.diagnostics["loss"],
                                   convergence=model.diagnostics["convergence"])
                        chosen = min(range(len(candidates)), key=lambda j: (
                            candidates[j].diagnostics["loss"], OUTER_SEEDS[j]))
                        models[arm] = candidates[chosen]
                        records[arm] = dict(starts=starts, selected_seed=OUTER_SEEDS[chosen],
                                            selected_train_objective=candidates[chosen].diagnostics["loss"])
                        del candidates, model
                        atomic_json(folder / "FIT_PROGRESS.json", dict(records=records, selected=modal,
                                    history=history, source_hashes=hashes, source_bundle_sha256=bundle_sha,
                                    descriptive_full_D=True, is_oof=False))
                    del inputs, x, n, g, c
                    gc.collect()
                    bundle = dict(feature_map=feature_map, models=models, y_scale=scale,
                                  selected=modal, modal_records=modal, records=records,
                                  history=history, history_name=name, outer_fold=None,
                                  source_hashes=hashes, training_source_hashes=training_hashes,
                                  source_bundle_sha256=bundle_sha, descriptive_full_D=True, is_oof=False,
                                  response="adjacent_hour_fraction_change", training_phase=0,
                                  train_rows=len(rows["unit"]),
                                  training_row_index=np.arange(len(rows["unit"]), dtype=np.int64),
                                  weather_information="strictly past t-1 through t-48")
                    output = folder / "bundle.pkl"
                    temporary = output.with_suffix(".pkl.part")
                    with temporary.open("wb") as handle:
                        pickle.dump(bundle, handle, protocol=5)
                        handle.flush(); os.fsync(handle.fileno())
                    os.replace(temporary, output)
                    unchanged()
                    atomic_json(folder / "DONE.json", dict(completed_at=now(), history=history,
                                source_hashes=hashes, training_source_hashes=training_hashes,
                                source_bundle_sha256=bundle_sha, bundle_sha256=sha256(output),
                                selected=modal, diagnostics=records, y_scale=scale,
                                train_rows=len(rows["unit"]), completed_fits=8,
                                descriptive_full_D=True, is_oof=False))
                    complete.append(name)
                    status(phase="history_complete", history_name=name)
                    del bundle, feature_map, models, records
                    gc.collect()
            status(state="completed" if len(complete) == 2 else "selected_history_completed",
                   phase="finite_queue_finished")
        except BaseException as error:
            status(state="failed", phase="finite_queue_stopped", error_type=type(error).__name__,
                   error=str(error), requires_review=True)
            raise


def self_test():
    bundles = []
    # Two configurations tie with two votes each; stronger ridge wins first.
    votes = [(2, .01), (2, .01), (4, .1), (4, .1), (2, .1)]
    for fold, (rank, ridge) in enumerate(votes, 1):
        chosen = {arm: {"config": dict(geo_rank=rank if spec["ranks"] != [0] else 0, ridge=ridge)}
                  for arm, spec in ARMS.items()}
        bundles.append(dict(outer_fold=fold, selected=chosen))
    modal = modal_configurations(bundles)
    assert modal["D_geo_pairs"]["config"]["geo_rank"] == 4
    assert modal["D_geo_pairs"]["config"]["ridge"] == .1
    # Equal ridge and vote count: lower rank wins.
    for fold, (rank, ridge) in enumerate([(2,.1),(2,.1),(4,.1),(4,.1),(4,.01)]):
        for arm, spec in ARMS.items():
            bundles[fold]["selected"][arm]["config"] = dict(
                geo_rank=rank if spec["ranks"] != [0] else 0, ridge=ridge)
    assert modal_configurations(bundles)["C_geo_main"]["config"]["geo_rank"] == 2
    try:
        modal_configurations(bundles[:4])
        raise AssertionError("Incomplete fold set accepted")
    except ValueError:
        pass
    return dict(modal_tie_checks="passed", five_fold_requirement="passed", real_data_loaded=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--history", choices=("true", "false"), default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
    else:
        run(args)


if __name__ == "__main__":
    main()
