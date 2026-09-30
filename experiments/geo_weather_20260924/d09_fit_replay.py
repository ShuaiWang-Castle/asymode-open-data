"""Registered D09 FIT allocation replay from completed checkpoint exports only.

Run only after all six formal jobs have completed and replay is authorized.
The frozen endpoint reader first verifies every DONE, source, output and identity
guard. Then replay existing step0/100/300/900 FIT-panel exports separately for
each inner fold. No model, forward, gradient, optimization, new arm, confidence
interval, cross-fold FIT pooling or advancement gate is introduced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import os

import d09_score as frozen  # Sets the two-thread environment before NumPy loads.
import numpy as np

STEPS = (0, 100, 300, 900)
COHORTS = ('all', 'S', 'nonS', 'any_positive', 'J')


def verify_replay_registration(commit):
    if len(commit) < 7 or any(c not in '0123456789abcdef' for c in commit):
        raise ValueError('An explicit hexadecimal replay registration commit is required')
    full = subprocess.check_output(['git', 'rev-parse', '--verify', commit + '^{commit}'],
                                   cwd=frozen.ROOT, text=True, stderr=subprocess.PIPE).strip()
    path = Path(__file__).resolve()
    blob = subprocess.check_output(['git', 'show', f'{full}:{frozen.relative(path)}'],
                                   cwd=frozen.ROOT, stderr=subprocess.PIPE)
    if hashlib.sha256(blob).hexdigest() != frozen.sha(path):
        raise ValueError('Registered FIT replay differs from current source')
    return full


def validate_fit_export(export, receipt, roster):
    """Reuse frozen per-original-fold checks without falsifying fold metadata."""
    ids = np.asarray(export.get('idx'))
    expected = np.asarray(receipt['fit_unit_ids'], int)
    if not np.array_equal(ids, expected):
        raise ValueError('FIT export differs from exact registered training IDs')
    folds = np.asarray(export['original_fold'])
    if not np.isin(folds, [2, 3, 4, 5]).all() or np.any(folds == receipt['heldout_fold']):
        raise ValueError('FIT contains original OUTER1 or this job heldout fold')
    for original_fold in np.unique(folds):
        select = folds == original_fold
        subset = {key: value[select] for key, value in export.items()}
        frozen.validate_export(subset, int(original_fold))
    if not np.asarray(export['panel']).all():
        raise ValueError('FIT export includes a county outside the fixed D08 panel')
    for row, unit in enumerate(ids):
        record = roster[int(unit)]
        for key in ('fips', 'system', 'family', 'origin', 'regime', 'merged_group',
                    'original_fold', 'w', 'pi'):
            if export[key][row] != record[key]:
                raise ValueError('FIT metadata differs from frozen roster: ' + key)


def score_fit(data, predictions):
    """Use the endpoint helpers and cumulative-mass quantiles, in float64."""
    promoted = dict(data)
    for key in ('y', 'y0', 'w', 'pi'):
        promoted[key] = np.asarray(data[key], dtype=np.float64)
    models = {key: np.asarray(P, dtype=np.float64) for key, P in predictions.items()}
    if set(models) != set(frozen.ARMS):
        raise ValueError('Exactly host/crk/new cached predictions required')
    ph = frozen.phenotype(promoted['y'], promoted['m'], promoted['y0'])
    cohorts = frozen.cohort_masks(ph)
    peaks = {key: frozen.model_peak(P, promoted['m'], ph) for key, P in models.items()}
    rows = {}
    for key in COHORTS:
        selected = cohorts[key]
        row = frozen.score_row(promoted, models, peaks, ph, selected, selected, plans={})
        # The shared helper's fixed denominator refers here to this FIT panel,
        # not a heldout or HT population. All other arithmetic stays verbatim.
        row['original_FIT_panel_cohort_hour_mass'] = row.pop('original_full_held_cohort_hour_mass')
        rows[key] = row
    alarms = {label: frozen.alarm_summary(peaks[base]['predicted_peak'], peaks[candidate]['predicted_peak'],
                                         cohorts['nonS'], promoted['w'], .1)
              for label, base, candidate in frozen.COMPARISONS
              if base in models and candidate in models}
    return dict(rows=rows, nonS_severe_alarms=alarms,
                confidence_intervals=False, bootstrap_performed=False,
                design_weighting='Original county-event w, not class-normalized training loss or w/pi.',
                peak_quantile_convention='Cumulative design-mass knots, identical to terminal heldout scorer; checkpoint diagnostic midpoint medians are not mixed in.',
                original_dtypes=dict(data={key: str(np.asarray(data[key]).dtype) for key in ('y', 'y0', 'w', 'pi')},
                                     predictions={key: str(np.asarray(P).dtype) for key, P in predictions.items()}),
                arithmetic_dtype='float64')


def replay(jobs_path, scope, training_commit, replay_commit):
    replay_registration = verify_replay_registration(replay_commit)
    training_registration = frozen.verify_registration(training_commit, scope)
    own_path = Path(__file__).resolve()
    own_sha = frozen.sha(own_path)
    # This is intentionally the original unmodified reader. All six terminal
    # DONE/source/output guards complete before it opens any endpoint outcome;
    # no FIT array is opened until the entire reader returns successfully.
    heldout, held_predictions, verified_files, provenance = frozen.read_verified_jobs(
        jobs_path, scope, training_registration)
    del heldout, held_predictions
    jobs = json.loads(Path(jobs_path).read_text())['jobs']
    inventory = {(job['arm'], int(job['heldout_fold'])): frozen.under_root(job['path']) for job in jobs}
    roster = {int(record['unit']): record for record in
              json.loads((frozen.D08 / 'roster.json').read_text())['records']}
    receipts = {(arm, fold): json.loads((folder / 'DONE.json').read_text())
                for (arm, fold), folder in inventory.items()}
    # All 24 requested FIT caches must exist in the already verified output
    # inventory before opening the first one, preserving partial jobs on error.
    for (arm, fold), folder in inventory.items():
        receipt = receipts[(arm, fold)]
        for step in STEPS:
            path = folder / f'step{step:04d}_fit_panel.npz'
            name = frozen.relative(path)
            if path.name not in receipt['outputs'] or name not in verified_files:
                raise ValueError('Requested checkpoint FIT cache not verified: ' + name)
            if frozen.sha(path) != receipt['outputs'][path.name] or verified_files[name] != receipt['outputs'][path.name]:
                raise ValueError('Checkpoint FIT cache changed: ' + name)
    result = {}
    for fold in frozen.FOLDS:
        reference, steps = None, {}
        for step in STEPS:
            metadata, predictions = None, {}
            for arm in frozen.ARMS:
                path = inventory[(arm, fold)] / f'step{step:04d}_fit_panel.npz'
                with np.load(path, allow_pickle=False) as archive:
                    export = {key: archive[key].copy() for key in archive.files}
                validate_fit_export(export, receipts[(arm, fold)], roster)
                current = {key: export[key] for key in frozen.META_KEYS}
                if metadata is None:
                    metadata = current
                else:
                    frozen.assert_same_metadata(metadata, current)
                if reference is None:
                    reference = current
                else:
                    frozen.assert_same_metadata(reference, current)
                predictions[arm] = export['P']
            steps[str(step)] = score_fit(metadata, predictions)
        result[str(fold)] = dict(inner_heldout_fold=fold, fit_units=len(reference['idx']),
                                fit_unit_ids_sha256=hashlib.sha256(np.asarray(reference['idx'], dtype='<i8').tobytes()).hexdigest(),
                                steps=steps)
    for name, digest in verified_files.items():
        if frozen.sha(frozen.ROOT / name) != digest:
            raise ValueError('Verified input/source/output changed during FIT replay: ' + name)
    if frozen.sha(own_path) != own_sha:
        raise ValueError('Replay source changed during execution')
    verified_files = dict(verified_files)
    verified_files[frozen.relative(own_path)] = own_sha
    provenance = dict(provenance, files_sha256=verified_files,
        FIT_replay=dict(path=frozen.relative(own_path), source_sha256=own_sha,
                        registration_commit=replay_registration, arithmetic_dtype='float64',
                        cached_steps=list(STEPS), original_endpoint_reader_unmodified=True))
    return dict(schema='d09_FIT_checkpoint_replay_v1', provenance=provenance, folds=result,
                interpretations=[
                    'Existing fixed FIT-panel caches only; each model was trained on the described counties and event groups. These are training-allocation diagnostics, never heldout evidence.',
                    'The two inner FIT sets overlap. They remain separate at every step and are never pooled as independent units.',
                    'Steps0/100/300/900 were pre-fixed checkpoint exports. Replaying them does not select a step, refit a model, extend the budget or modify the terminal advancement gate.',
                    'S/nonS/any-positive/J, observation masks, first tied observed peaks, original design weights, and cumulative-mass peak quantiles match the terminal heldout helpers.',
                    'The trainer checkpoint midpoint-mass peak median is a different diagnostic convention and is not substituted for these cumulative-mass q10/50/90 values.',
                    'Alignment and modification energy use the original strict paired-SSE identity in float64. Comparisons include all jointly trained model differences and are not isolated causal kernel effects.',
                    'Lower FIT trajectory RMSE need not mean widespread peak recovery; compare peak quantiles, positive design-weight coverage, positive-gain concentration and nonS severe alarms.',
                    'No confidence interval or new optimization/forward/gradient is performed; repeatedly explored D and training-set diagnostics cannot establish generalization or net geographic information.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay-commit', required=True)
    parser.add_argument('--jobs', type=Path, required=True)
    parser.add_argument('--scope', type=Path, required=True)
    parser.add_argument('--scope-commit', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    jobs, scope, out = (frozen.under_root(path) for path in (args.jobs, args.scope, args.out))
    if out.exists():
        raise FileExistsError('Preserve existing FIT replay: ' + frozen.relative(out))
    os.nice(max(0, 15 - os.getpriority(os.PRIO_PROCESS, 0)))
    result = replay(jobs, scope, args.scope_commit, args.replay_commit)
    out.parent.mkdir(parents=True, exist_ok=True)
    frozen.write_new(out, result)
    print(json.dumps(dict(result=frozen.relative(out), sha256=frozen.sha(out),
                          folds=len(result['folds']), steps=list(STEPS),
                          advancement_gate_changed=False), allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
