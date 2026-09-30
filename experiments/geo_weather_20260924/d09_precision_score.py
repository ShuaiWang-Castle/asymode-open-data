"""Registered float64 arithmetic adapter for the frozen D09 endpoint scorer.

No training source, model, label, stored prediction, weight, threshold, bootstrap
draw or advancement rule is changed. The original scorer performs every job,
registration, source, data, output and identity check before scoring. Only its
score_surface inputs are promoted to float64, avoiding float32 cancellation in
the strict paired-SSE algebra check. This file is registered separately from
the immutable training/scoring source inventory.

Literal threshold constants stay fixed. Boundary decisions need not be
bit-identical to float32 arithmetic: np.float32(.001) stores approximately
.0010000000474974513, so its exact promoted value exceeds the unchanged .001
tiny-false-peak threshold, whereas float32 scalar rounding treated it as equal.
Thus the adapter does not promise to preserve every prior alarm boolean.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess
import sys

import d09_score as frozen  # Sets the two-thread environment before NumPy loads.
import numpy as np

ORIGINAL_SCORE_SURFACE = frozen.score_surface
ORIGINAL_WRITE_NEW = frozen.write_new
DATA_ARRAYS = ('y', 'y0', 'w', 'pi')


def promote_inputs(data, predictions):
    """Value-preserving casts; observation masks, IDs and metadata are untouched."""
    promoted = dict(data)
    for key in DATA_ARRAYS:
        promoted[key] = np.asarray(data[key], dtype=np.float64)
    models = {key: np.asarray(value, dtype=np.float64)
              for key, value in predictions.items()}
    original_dtypes = dict(data={key: str(np.asarray(data[key]).dtype) for key in DATA_ARRAYS},
                           predictions={key: str(np.asarray(value).dtype)
                                        for key, value in predictions.items()})
    return promoted, models, original_dtypes


def score_surface_float64(data, predictions, *args, **kwargs):
    promoted, models, _ = promote_inputs(data, predictions)
    return ORIGINAL_SCORE_SURFACE(promoted, models, *args, **kwargs)


def verify_adapter_registration(commit):
    if len(commit) < 7 or any(c not in '0123456789abcdef' for c in commit):
        raise ValueError('An explicit hexadecimal precision registration commit is required')
    full = subprocess.check_output(['git', 'rev-parse', '--verify', commit + '^{commit}'],
                                   cwd=frozen.ROOT, text=True, stderr=subprocess.PIPE).strip()
    path = Path(__file__).resolve()
    blob = subprocess.check_output(['git', 'show', f'{full}:{frozen.relative(path)}'],
                                   cwd=frozen.ROOT, stderr=subprocess.PIPE)
    if hashlib.sha256(blob).hexdigest() != frozen.sha(path):
        raise ValueError('Registered precision adapter differs from current source')
    return full


def run_with_precision(original_argv, precision_commit):
    """Temporarily hook two in-memory entry points; original main remains intact."""
    commit = verify_adapter_registration(precision_commit)
    path = Path(__file__).resolve()
    adapter_sha = frozen.sha(path)
    calls = []

    def surface(data, predictions, *args, **kwargs):
        promoted, models, dtypes = promote_inputs(data, predictions)
        calls.append(dict(panel=bool(kwargs.get('panel', False)), original_dtypes=dtypes))
        return ORIGINAL_SCORE_SURFACE(promoted, models, *args, **kwargs)

    def write(path_out, result):
        if frozen.sha(path) != adapter_sha:
            raise ValueError('Precision adapter source changed during score')
        if not calls or any(call['original_dtypes'] != calls[0]['original_dtypes'] for call in calls[1:]):
            raise ValueError('Missing or inconsistent precision scoring inputs')
        # Preserve the frozen training provenance and file hashes verbatim,
        # adding a separately registered aggregation-only source entry.
        result['provenance']['precision_adapter'] = dict(
            path=frozen.relative(path), source_sha256=adapter_sha,
            registration_commit=commit, arithmetic_dtype='float64',
            converted_arrays=dict(data=list(DATA_ARRAYS),
                                  predictions=list(calls[0]['original_dtypes']['predictions'])),
            original_dtypes=calls[0]['original_dtypes'],
            scoring_calls=[dict(panel=call['panel']) for call in calls],
            frozen_score_source_sha256=frozen.sha(frozen.__file__),
            boundary_semantics='Compare exact promoted stored values with unchanged literal constants; float32 scalar-rounding decisions at threshold boundaries can differ. In particular float32(.001) stores .0010000000474974513 and therefore exceeds the unchanged .001 tiny-false-peak threshold in float64.',
            reason='Float32 subtraction/squaring/reduction can violate the unchanged 2e-10 paired-SSE identity check; promote exact stored values before arithmetic, without relaxing tolerance.')
        result['provenance']['files_sha256'][frozen.relative(path)] = adapter_sha
        result['interpretations'].append(
            'A separately registered aggregation-only precision adapter casts y/y0/w/pi and stored predictions to float64 before calling the frozen scorer. It preserves stored values, all source/receipt/identity guards, numeric threshold constants, weights, cohort definitions, bootstrap draws, point gate and the original strict paired-SSE tolerance; it neither reruns nor changes training or predictions.')
        result['interpretations'].append(
            'Float32 threshold-boundary booleans are not guaranteed bit-identical after value-preserving promotion: float32(.001) stores .0010000000474974513, which exceeds the unchanged .001 tiny-false-peak threshold in float64; the earlier float32 scalar comparison rounded the literal to the stored value and treated it as equal. This is explicitly reported arithmetic boundary semantics, not a change to the numerical threshold constant.')
        ORIGINAL_WRITE_NEW(path_out, result)

    prior_surface, prior_write, prior_argv = frozen.score_surface, frozen.write_new, sys.argv
    try:
        frozen.score_surface, frozen.write_new = surface, write
        sys.argv = [str(Path(frozen.__file__)), *original_argv]
        frozen.main()
    finally:
        frozen.score_surface, frozen.write_new = prior_surface, prior_write
        sys.argv = prior_argv


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--precision-commit', required=True)
    args, original_argv = parser.parse_known_args()
    run_with_precision(original_argv, args.precision_commit)


if __name__ == '__main__':
    main()
