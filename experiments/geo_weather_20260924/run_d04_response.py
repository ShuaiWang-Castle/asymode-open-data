"""Finite, locked D04 statistical queue. Run only after its scope is registered.

One process fits one model at a time. Training/selection use phase-zero rows;
outer prediction covers every retained adjacent-hour row. No neural jobs, new
seeds, or follow-up designs are started. Existing unfinished fold directories
(including empty ones) cause refusal, never overwrite or automatic retraining.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "2"

import argparse
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np

from d04_data import ROOT, HERE, SPLITS, load_panel, make_rows
from d04_features import MAIN_DIM, PAIR_DIM, REGIMES, fit_map, transform_weather
from d04_models import fit_model, predict

DEFAULT_RUN = ROOT / "runs/geo_weather_20260924/data_first_d04"
SCOPE = HERE / "notes/D04_CONDITIONAL_IMPACT_RESPONSE_SCOPE_20260928.md"
ARMS = {
    "A_shared_main": dict(columns=MAIN_DIM, ranks=[0]),
    "B_shared_pairs": dict(columns=MAIN_DIM + PAIR_DIM, ranks=[0]),
    "C_geo_main": dict(columns=MAIN_DIM, ranks=[2, 4]),
    "D_geo_pairs": dict(columns=MAIN_DIM + PAIR_DIM, ranks=[2, 4]),
}
RIDGES = [.01, .1]
INNER_SEED = 11
OUTER_SEEDS = [11, 29]
INNER_ITERATIONS = 60
OUTER_ITERATIONS = 120


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes():
    paths = [HERE / name for name in ("d04_data.py", "d04_features.py", "d04_models.py",
                                      "run_d04_response.py", "county_structure_d02.py")]
    paths += [SCOPE, SPLITS]
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("w") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def subset(rows, index):
    return {key: value[index] for key, value in rows.items()}


def balanced_weights(rows):
    out = np.zeros(len(rows["w"]), dtype=np.float64)
    for regime in REGIMES:
        selected = rows["regime"] == regime
        mass = float(rows["w"][selected].sum())
        if mass <= 0:
            raise ValueError(f"No positive fitting weight for regime {regime}")
        out[selected] = rows["w"][selected] / (len(REGIMES) * mass)
    if not np.all(out > 0) or not np.isclose(out.sum(), 1):
        raise ValueError("Invalid regime-balanced fitting weights")
    return out


def training_rows_and_scale(rows):
    result = dict(rows)
    result["w"] = balanced_weights(rows)
    mean = float(np.dot(result["w"], rows["y"]))
    scale = float(np.sqrt(np.dot(result["w"], (rows["y"] - mean) ** 2)))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Training response has no finite positive weighted standard deviation")
    return result, scale


def score_balanced(rows, prediction):
    prediction = np.asarray(prediction, dtype=np.float64)
    if prediction.shape != rows["y"].shape or not np.isfinite(prediction).all():
        raise ValueError("Invalid validation predictions")
    per_regime = {}
    for regime in REGIMES:
        use = rows["regime"] == regime
        if not np.any(use) or float(rows["w"][use].sum()) <= 0:
            raise ValueError(f"Validation lacks regime {regime}")
        per_regime[regime] = float(np.average((rows["y"][use] - prediction[use]) ** 2,
                                            weights=rows["w"][use]))
    return dict(balanced_mse=float(np.mean(list(per_regime.values()))), per_regime_mse=per_regime)


def choose_inner(records):
    best = min(record["score"]["balanced_mse"] for record in records)
    tolerance = max(1e-14, abs(best) * 1e-10)
    tied = [record for record in records if record["score"]["balanced_mse"] <= best + tolerance]
    return min(tied, key=lambda record: (-record["config"]["ridge"], record["config"]["geo_rank"]))


def config_for(rank, ridge, iterations):
    return dict(geo_rank=int(rank), context_rank=2, ridge=float(ridge), max_iter=iterations,
                county_ridge=.1, batch_size=8192, threads=2, intercept_index=0,
                tolerance_grad=1e-5, tolerance_change=1e-9, history_size=10)


class Queue:
    def __init__(self, run_dir, hashes):
        self.path = Path(run_dir)
        self.hashes = hashes
        self.started = now()
        self.elapsed_start = time.monotonic()
        self.completed = []

    def unchanged(self):
        if source_hashes() != self.hashes:
            raise RuntimeError("D04 source/scope/split changed during queue; refusing mixed implementations")

    def status(self, state="running", **details):
        value = dict(state=state, pid=os.getpid(), started_at=self.started, updated_at=now(),
                     elapsed_seconds=time.monotonic() - self.elapsed_start,
                     completed_folds=self.completed.copy(), source_hashes=self.hashes, **details)
        atomic_json(self.path / "RUN_STATUS.json", value)
        print(json.dumps({key: val for key, val in value.items() if key != "source_hashes"},
                         ensure_ascii=False, allow_nan=False), flush=True)

    def existing_done(self, fold):
        folder = self.path / f"fold{fold:02d}"
        if not folder.exists():
            return False
        done_path = folder / "DONE.json"
        if not done_path.is_file():
            raise FileExistsError(f"Unfinished fold directory retained; refusing overwrite: {folder.name}")
        done = json.loads(done_path.read_text())
        if done.get("source_hashes") != self.hashes or done.get("fold") != fold:
            raise RuntimeError(f"Completed fold {fold} has incompatible source hashes or identity")
        for filename, expected in done.get("artifact_sha256", {}).items():
            if sha256(folder / filename) != expected:
                raise RuntimeError(f"Completed fold {fold} artifact checksum differs: {filename}")
        expected_names = {"bundle.pkl", f"oof_fold{fold}.npz"}
        if set(done.get("artifact_sha256", {})) != expected_names:
            raise RuntimeError("DONE manifest lacks complete fold artifacts")
        self.completed.append(fold)
        return True

    def run_fold(self, fold, panel, wx, phase_rows, all_rows):
        self.unchanged()
        folder = self.path / f"fold{fold:02d}"
        folder.mkdir()  # Deliberately no exist_ok: even an empty partial folder is preserved.
        inner_fold = fold % 5 + 1
        tr_idx = np.flatnonzero((phase_rows["fold"] != fold) & (phase_rows["fold"] != inner_fold))
        va_idx = np.flatnonzero(phase_rows["fold"] == inner_fold)
        outer_train_idx = np.flatnonzero(phase_rows["fold"] != fold)
        test_idx = np.flatnonzero(all_rows["fold"] == fold)
        if any(len(index) == 0 for index in (tr_idx, va_idx, outer_train_idx, test_idx)):
            raise ValueError("A prescribed D04 split is empty")
        # Original folds must isolate merged groups; no validation outcomes enter fitting bases.
        for a, b in ((tr_idx, va_idx), (tr_idx, np.flatnonzero(phase_rows["fold"] == fold))):
            if np.intersect1d(phase_rows["group"][a], phase_rows["group"][b]).size:
                raise ValueError("Merged event group crosses training/validation boundary")
        inner_rows, inner_scale = training_rows_and_scale(subset(phase_rows, tr_idx))
        valid_rows = subset(phase_rows, va_idx)
        self.status(fold=fold, phase="inner_features", inner_fold=inner_fold)
        inner_map, inner_inputs = fit_map(panel, wx, inner_rows, history=True)
        validation_inputs = inner_map.transform(panel, wx, valid_rows)
        inner_records, chosen = {}, {}
        for arm, specification in ARMS.items():
            records = []
            for ridge in RIDGES:
                for rank in specification["ranks"]:
                    self.unchanged()
                    config = config_for(rank, ridge, INNER_ITERATIONS)
                    self.status(fold=fold, phase="inner_fit", arm=arm, config=config,
                                init_seed=INNER_SEED, inner_fold=inner_fold)
                    x, n, g, c = inner_inputs
                    started = time.monotonic()
                    model = fit_model(x[:, :specification["columns"]], n, g, c,
                                      inner_rows["y"] / inner_scale, inner_rows["w"],
                                      inner_rows["county"], config, init_seed=INNER_SEED)
                    if not model.diagnostics["finitefit"]:
                        raise RuntimeError("Nonfinite inner fitting result")
                    vx, vn, vg, vc = validation_inputs
                    val_prediction = predict(model, vx[:, :specification["columns"]], vn, vg, vc,
                                             valid_rows["county"]).astype(np.float64) * inner_scale
                    record = dict(config=config, init_seed=INNER_SEED, y_scale=inner_scale,
                                  score=score_balanced(valid_rows, val_prediction),
                                  diagnostics=model.diagnostics, seconds=time.monotonic() - started)
                    records.append(record)
                    atomic_json(folder / "INNER_PROGRESS.json", dict(fold=fold, completed=inner_records,
                                active_arm=arm, active_records=records, source_hashes=self.hashes))
                    self.status(fold=fold, phase="inner_fit_complete", arm=arm,
                                ridge=ridge, geo_rank=rank, balanced_mse=record["score"]["balanced_mse"],
                                convergence=model.diagnostics["convergence"])
                    del model, val_prediction
            inner_records[arm] = records
            chosen[arm] = choose_inner(records)
        atomic_json(folder / "INNER_PROGRESS.json", dict(fold=fold, completed=inner_records,
                    chosen=chosen, source_hashes=self.hashes))
        del inner_inputs, validation_inputs, inner_map, inner_rows, valid_rows, x, n, g, c, vx, vn, vg, vc
        gc.collect()

        train_rows, outer_scale = training_rows_and_scale(subset(phase_rows, outer_train_idx))
        self.unchanged()
        self.status(fold=fold, phase="outer_features")
        feature_map, train_inputs = fit_map(panel, wx, train_rows, history=True)
        models, outer_records = {}, {}
        for arm, specification in ARMS.items():
            selected_config = chosen[arm]["config"]
            config = config_for(selected_config["geo_rank"], selected_config["ridge"], OUTER_ITERATIONS)
            candidates, records = [], []
            for seed in OUTER_SEEDS:
                self.unchanged()
                self.status(fold=fold, phase="outer_fit", arm=arm, config=config, init_seed=seed)
                x, n, g, c = train_inputs
                started = time.monotonic()
                model = fit_model(x[:, :specification["columns"]], n, g, c,
                                  train_rows["y"] / outer_scale, train_rows["w"],
                                  train_rows["county"], config, init_seed=seed)
                if not model.diagnostics["finitefit"]:
                    raise RuntimeError("Nonfinite full fitting result")
                records.append(dict(config=config, init_seed=seed, diagnostics=model.diagnostics,
                                    seconds=time.monotonic() - started))
                candidates.append(model)
                self.status(fold=fold, phase="outer_fit_complete", arm=arm, init_seed=seed,
                            loss=model.diagnostics["loss"], convergence=model.diagnostics["convergence"])
            best = min(range(len(candidates)), key=lambda j: (candidates[j].diagnostics["loss"],
                                                             OUTER_SEEDS[j]))
            models[arm] = candidates[best]
            outer_records[arm] = dict(starts=records, selected_seed=OUTER_SEEDS[best],
                                      selected_train_objective=candidates[best].diagnostics["loss"])
            del candidates, model
            atomic_json(folder / "OUTER_PROGRESS.json", dict(fold=fold, completed=outer_records,
                        source_hashes=self.hashes))
        del train_inputs, train_rows, x, n, g, c
        gc.collect()

        self.unchanged()
        self.status(fold=fold, phase="outer_all_hours_prediction", outer_rows=len(test_idx))
        predictions = np.empty((len(test_idx), len(ARMS)), dtype=np.float32)
        for start in range(0, len(test_idx), 4096):
            stop = min(len(test_idx), start + 4096)
            rows = subset(all_rows, test_idx[start:stop])
            x, n, g, c = feature_map.transform(panel, wx, rows)
            for column, (arm, specification) in enumerate(ARMS.items()):
                predictions[start:stop, column] = predict(models[arm], x[:, :specification["columns"]],
                    n, g, c, rows["county"]) * outer_scale
            if start % (4096 * 16) == 0:
                self.status(fold=fold, phase="outer_all_hours_prediction", predicted_rows=stop,
                            outer_rows=len(test_idx))
        if not np.isfinite(predictions).all():
            raise ValueError("Outer predictions contain nonfinite entries")
        rows = subset(all_rows, test_idx)
        if (len(np.unique(test_idx)) != len(test_idx)
                or not np.all(rows["fold"] == fold)
                or not np.array_equal(test_idx, np.flatnonzero(all_rows["fold"] == fold))):
            raise ValueError("Outer heldout coverage is not exact")
        bundle = dict(feature_map=feature_map, models=models, y_scale=outer_scale,
                      selected=chosen, chosen=chosen,
                      records=dict(inner=inner_records, outer=outer_records),
                      inner_records=inner_records, outer_records=outer_records,
                      outer_fold=fold, fold=fold, inner_fold=inner_fold, source_hashes=self.hashes,
                      response="adjacent_hour_fraction_change", history=True,
                      weather_information="strictly past t-1 through t-48",
                      training_phase=0, outer_evaluation_phase="all",
                      train_phase_row_index=outer_train_idx, inner_train_row_index=tr_idx,
                      inner_validation_row_index=va_idx)
        bundle_path = folder / "bundle.pkl"
        with bundle_path.with_suffix(".pkl.part").open("wb") as handle:
            pickle.dump(bundle, handle, protocol=5)
            handle.flush(); os.fsync(handle.fileno())
        os.replace(bundle_path.with_suffix(".pkl.part"), bundle_path)
        oof_path = folder / f"oof_fold{fold}.npz"
        with oof_path.with_suffix(".npz.part").open("wb") as handle:
            np.savez_compressed(handle, row_index=test_idx, predictions=predictions,
                                arm_names=np.asarray(list(ARMS)), **rows,
                                source_hashes=np.asarray(json.dumps(self.hashes, sort_keys=True)))
            handle.flush(); os.fsync(handle.fileno())
        os.replace(oof_path.with_suffix(".npz.part"), oof_path)
        self.unchanged()
        with np.load(oof_path, allow_pickle=False) as saved:
            if (not np.array_equal(saved["row_index"], test_idx)
                    or not np.array_equal(saved["unit"], rows["unit"])
                    or not np.array_equal(saved["time"], rows["time"])
                    or saved["predictions"].shape != predictions.shape
                    or not np.isfinite(saved["predictions"]).all()):
                raise ValueError("Saved outer export verification failed")
        done = dict(fold=fold, completed_at=now(), source_hashes=self.hashes,
                    outer_rows=len(test_idx), outer_county_events=len(np.unique(rows["unit"])),
                    train_phase_rows=len(outer_train_idx), inner_fold=inner_fold,
                    all_hours_coverage_exact=True, predictions_finite=True,
                    y_scale=outer_scale, arms=list(ARMS),
                    artifact_sha256={bundle_path.name: sha256(bundle_path), oof_path.name: sha256(oof_path)})
        atomic_json(folder / "DONE.json", done)
        self.completed.append(fold)
        self.status(fold=fold, phase="fold_complete", outer_rows=len(test_idx))


def run(args):
    current_nice = os.getpriority(os.PRIO_PROCESS, 0)
    if current_nice < 15:
        os.nice(15 - current_nice)
    path = Path(args.run_dir).resolve()
    path.mkdir(parents=True, exist_ok=True)
    # flock is authoritative; an old lock file without an OS lock is not a live runner.
    with (path / "LOCK").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another D04 runner holds the exclusive LOCK") from error
        lock.seek(0); lock.truncate()
        lock.write(json.dumps(dict(pid=os.getpid(), started_at=now())) + "\n")
        lock.flush(); os.fsync(lock.fileno())
        hashes = source_hashes()
        queue = Queue(path, hashes)
        try:
            manifest_path = path / "SCREEN_RUN.json"
            manifest = dict(source_hashes=hashes, arms=ARMS, ridges=RIDGES,
                            inner_seed=INNER_SEED, outer_seeds=OUTER_SEEDS,
                            inner_iterations=INNER_ITERATIONS, outer_iterations=OUTER_ITERATIONS,
                            inner_fold_rule="outer % 5 + 1", training_phase=0,
                            outer_evaluation_phase="all", response="adjacent_hour_fraction_change",
                            history=True, weighting="equal five regimes; within-regime design weights",
                            target_scaling="training weighted standard deviation; no centering",
                            nice=os.getpriority(os.PRIO_PROCESS, 0), threads=2)
            if manifest_path.exists():
                old = json.loads(manifest_path.read_text())
                compare = {key: value for key, value in manifest.items() if key != "nice"}
                if any(old.get(key) != value for key, value in compare.items()):
                    raise RuntimeError("Existing SCREEN_RUN manifest differs from frozen queue")
            else:
                atomic_json(manifest_path, dict(manifest, created_at=now()))
            # Inspect every original fold, including folds not requested in this invocation.
            todo = []
            selected = [args.fold] if args.fold is not None else list(range(1, 6))
            for fold in range(1, 6):
                finished = queue.existing_done(fold)
                if fold in selected and not finished:
                    todo.append(fold)
            queue.status(phase="loading_public_D", selected_folds=selected, queued_folds=todo)
            if todo:
                panel = load_panel()
                wx = transform_weather(panel)
                phase_rows, all_rows = make_rows(panel, phase=0), make_rows(panel, phase=None)
                queue.status(phase="rows_ready", phase_rows=len(phase_rows["unit"]),
                             all_hour_rows=len(all_rows["unit"]), queued_folds=todo)
                for fold in todo:
                    queue.run_fold(fold, panel, wx, phase_rows, all_rows)
            queue.status(state="completed" if len(queue.completed) == 5 else "selected_folds_completed",
                         phase="queue_finished", selected_folds=selected)
        except BaseException as error:
            queue.status(state="failed", phase="queue_stopped", error_type=type(error).__name__,
                         error=str(error), requires_review=True)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=range(1, 6))
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
