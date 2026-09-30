"""Pure-array D11 conditional residual probe, with no model or file IO.

One scalar response rule is shared across all 144 forecast hours. Ridge fits
observed y minus a fixed carrier's closed prediction under original county-event
weights. The intercept is unpenalized. No S-specific penalty selection, clipping,
stock reconstruction, neural-network fitting or confidence interval is used.
"""
from __future__ import annotations

import os
for _thread_name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                     'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_thread_name] = '2'

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np

HOURS = 144
GROUP_FOLDS = 3
PENALTY_COUNT = 3
BATCH_UNITS = 16
PENALTIES = (1e-4, 1e-2, 1.)
SCALE_FLOOR = 1e-4
FEATURES = 113


def readonly(value):
    array = np.array(value, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class RidgeMoments:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    constant_features: np.ndarray
    residual_mean: float
    standardized_gram: np.ndarray
    standardized_rhs: np.ndarray
    design_hour_mass: float
    units: int
    observed_units: int
    observed_rows: int
    scale_floor: float


@dataclass(frozen=True)
class RidgeModel:
    moments: RidgeMoments
    standardized_coefficient: np.ndarray
    coefficient: np.ndarray
    intercept: float
    penalty: float
    condition_number: float
    relative_normal_equation_residual: float
    slope_effective_df: float
    effective_df_including_intercept: float


def validate_target(y, closed, mask, weights, allow_empty=False):
    y, closed, weights = (np.asarray(value, dtype=np.float64) for value in (y, closed, weights))
    raw_mask = np.asarray(mask)
    if y.ndim != 2 or y.shape[1] != HOURS or closed.shape != y.shape or raw_mask.shape != y.shape:
        raise ValueError('Aligned N by 144 target, carrier and original mask required')
    if weights.shape != (len(y),) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError('Strictly positive finite original county-event weights required')
    if not np.isin(raw_mask, [0, 1]).all():
        raise ValueError('Original observation mask must be binary')
    mask = raw_mask.astype(bool)
    if not np.isfinite(y[mask]).all() or not np.isfinite(closed).all():
        raise ValueError('Nonfinite observed target or carrier prediction')
    if not mask.any() and not allow_empty:
        raise ValueError('Empty observed residual population')
    # Promote stored values before subtraction, never after float32 arithmetic.
    residual = np.where(mask, y - closed, 0.)
    if not np.isfinite(residual).all():
        raise ValueError('Nonfinite observed residual')
    return y, closed, mask, weights, residual


def validate_features(features, units, feature_count=None):
    features = np.asarray(features)
    if features.ndim != 3 or features.shape[:2] != (units, HOURS) or features.shape[2] < 1:
        raise ValueError('Features must have shape N by 144 by positive D')
    if feature_count is not None and features.shape[2] != feature_count:
        raise ValueError('Feature count changed between conditional FIT and validation')
    if not np.issubdtype(features.dtype, np.number) or np.issubdtype(features.dtype, np.complexfloating):
        raise ValueError('Real numeric feature arrays required')
    for start in range(0, units, BATCH_UNITS):
        if not np.isfinite(features[start:start + BATCH_UNITS]).all():
            raise ValueError('Nonfinite features, including unobserved prediction clocks')
    return features


def weighted_moments(features, y, closed, mask, weights, *, scale_floor=SCALE_FLOOR):
    """Two centered passes, bounded working memory, no target-dependent features.

    Objective is sum(w*m*error^2)/sum(w*m) + lambda*||beta_standardized||^2.
    Mean, scale and residual intercept use this FIT's original observed weights.
    A constant column remains present and receives exact zero centered values.
    """
    if scale_floor != SCALE_FLOOR:
        raise ValueError('D11 feature scale floor is fixed at 1e-4')
    _, _, mask, weights, residual = validate_target(y, closed, mask, weights)
    features = validate_features(features, len(mask))
    d = features.shape[2]
    mass = float(weights @ mask.sum(1))
    if not math.isfinite(mass) or mass <= 0:
        raise ValueError('Invalid original-design observed-hour mass')
    first, residual_sum = np.zeros(d, np.float64), 0.
    minimum, maximum = np.full(d, np.inf), np.full(d, -np.inf)
    for start in range(0, len(mask), BATCH_UNITS):
        stop = min(start + BATCH_UNITS, len(mask))
        selected = mask[start:stop]
        if not selected.any():
            continue
        x = np.asarray(features[start:stop], np.float64)[selected]
        w = np.broadcast_to(weights[start:stop, None], selected.shape)[selected]
        r = residual[start:stop][selected]
        first += np.sum(x * w[:, None], axis=0)
        residual_sum += float(w @ r)
        minimum, maximum = np.minimum(minimum, x.min(0)), np.maximum(maximum, x.max(0))
    mean, residual_mean = first / mass, residual_sum / mass
    constant = minimum == maximum
    mean[constant] = minimum[constant]
    covariance, cross = np.zeros((d, d), np.float64), np.zeros(d, np.float64)
    for start in range(0, len(mask), BATCH_UNITS):
        stop = min(start + BATCH_UNITS, len(mask))
        selected = mask[start:stop]
        if not selected.any():
            continue
        x = np.asarray(features[start:stop], np.float64)[selected] - mean
        x[:, constant] = 0.
        w = np.broadcast_to(weights[start:stop, None], selected.shape)[selected] / mass
        r = residual[start:stop][selected] - residual_mean
        xw = x * np.sqrt(w)[:, None]
        covariance += xw.T @ xw
        cross += x.T @ (w * r)
    covariance = (covariance + covariance.T) / 2
    variance = np.diag(covariance)
    if np.any(variance < 0) or not np.isfinite(covariance).all() or not np.isfinite(cross).all():
        raise ValueError('Invalid weighted feature moments')
    scale = np.maximum(np.sqrt(variance), scale_floor)
    gram, rhs = covariance / np.outer(scale, scale), cross / scale
    gram[constant, :], gram[:, constant], rhs[constant] = 0., 0., 0.
    if not np.isfinite(mean).all() or not math.isfinite(residual_mean):
        raise ValueError('Nonfinite FIT mean')
    return RidgeMoments(readonly(mean), readonly(scale), readonly(constant), float(residual_mean),
                        readonly(gram), readonly(rhs), mass, len(mask), int(mask.any(1).sum()),
                        int(mask.sum()), float(scale_floor))


def fit_from_moments(moments, penalty):
    """Solve the same SPD Gram for one of the finite registered penalties."""
    penalty = float(penalty)
    if penalty not in PENALTIES:
        raise ValueError('D11 ridge solve requires a registered penalty: 1e-4,1e-2,1')
    gram = np.asarray(moments.standardized_gram, np.float64)
    rhs = np.asarray(moments.standardized_rhs, np.float64)
    matrix = gram + penalty * np.eye(len(rhs), dtype=np.float64)
    chol = np.linalg.cholesky(matrix)
    beta = np.linalg.solve(chol.T, np.linalg.solve(chol, rhs))
    beta[moments.constant_features] = 0.
    equation_residual = float(np.linalg.norm(matrix @ beta - rhs) / max(np.linalg.norm(rhs), 1.))
    if not np.isfinite(beta).all() or equation_residual > 1e-10:
        raise ArithmeticError('Ridge SPD solve did not satisfy its normal equation')
    coefficient = beta / moments.feature_scale
    intercept = float(moments.residual_mean - moments.feature_mean @ coefficient)
    condition = float(np.linalg.cond(matrix))
    slope_df = float(np.trace(np.linalg.solve(matrix, gram)))
    if not math.isfinite(condition) or not math.isfinite(intercept) or not math.isfinite(slope_df) or slope_df < -1e-8 or slope_df > len(rhs) + 1e-8:
        raise ArithmeticError('Nonfinite conditioned ridge solution')
    return RidgeModel(moments, readonly(beta), readonly(coefficient), intercept,
                      penalty, condition, equation_residual, slope_df, 1. + slope_df)


def fit_ridge(features, y, closed, mask, weights, penalty, *, scale_floor=SCALE_FLOOR):
    return fit_from_moments(weighted_moments(features, y, closed, mask, weights,
                                           scale_floor=scale_floor), penalty)


def predict_delta(model, features):
    features = validate_features(features, len(features), len(model.coefficient))
    predicted = np.empty(features.shape[:2], np.float64)
    for start in range(0, len(features), BATCH_UNITS):
        stop = min(start + BATCH_UNITS, len(features))
        z = (np.asarray(features[start:stop], np.float64) - model.moments.feature_mean) / model.moments.feature_scale
        z[:, :, model.moments.constant_features] = 0.
        predicted[start:stop] = z @ model.standardized_coefficient + model.moments.residual_mean
    if not np.isfinite(predicted).all():
        raise ArithmeticError('Nonfinite residual prediction')
    return predicted


def masked_residual_score(y, closed, delta, mask, weights):
    """Independent exact residual alignment/energy accounting in float64."""
    _, _, mask, weights, residual = validate_target(y, closed, mask, weights, allow_empty=True)
    delta = np.asarray(delta, np.float64)
    if delta.shape != mask.shape or not np.isfinite(delta).all():
        raise ValueError('Finite N by 144 residual correction required')
    observed_delta = np.where(mask, delta, 0.)
    baseline_rows = weights * np.sum(residual ** 2, axis=1)
    candidate_rows = weights * np.sum((residual - observed_delta) ** 2, axis=1)
    alignment_rows = weights * np.sum(2 * residual * observed_delta, axis=1)
    energy_rows = weights * np.sum(observed_delta ** 2, axis=1)
    gain_rows = alignment_rows - energy_rows
    if not np.allclose(gain_rows, baseline_rows - candidate_rows, rtol=2e-10, atol=2e-10):
        raise AssertionError('Residual probe paired SSE identity failed')
    hour_mass, valid = float(weights @ mask.sum(1)), mask.any(1)
    baseline, candidate = float(baseline_rows.sum()), float(candidate_rows.sum())
    case_mass = float(weights[valid].sum())
    return dict(units=len(mask), observed_units=int(valid.sum()), observed_rows=int(mask.sum()),
                original_design_hour_mass=hour_mass, baseline_SSE=baseline, candidate_SSE=candidate,
                baseline_RMSE=math.sqrt(baseline / hour_mass) if hour_mass else None,
                candidate_RMSE=math.sqrt(candidate / hour_mass) if hour_mass else None,
                relative_RMSE_reduction=1 - math.sqrt(candidate / baseline) if baseline else None,
                alignment=float(alignment_rows.sum()), modification_energy=float(energy_rows.sum()),
                net_gain=float(gain_rows.sum()), positive_units=int((valid & (gain_rows > 0)).sum()),
                negative_units=int((valid & (gain_rows < 0)).sum()), zero_units=int((valid & (gain_rows == 0)).sum()),
                positive_design_share=float(weights[valid & (gain_rows > 0)].sum() / case_mass) if case_mass else None,
                baseline_SSE_rows=baseline_rows, candidate_SSE_rows=candidate_rows,
                alignment_rows=alignment_rows, modification_energy_rows=energy_rows, net_gain_rows=gain_rows,
                correction_clipped=False, stock_constraints_imposed=False)


def deterministic_group_folds(groups, original_folds, heldout_fold):
    """The three existing event folds inside top-level FIT; no new assignment."""
    groups = np.asarray(groups)
    if groups.ndim != 1 or groups.dtype.kind not in ('U', 'S'):
        raise ValueError('One-dimensional nonempty merged-group strings required')
    groups = groups.astype(str)
    levels = np.unique(groups)
    if len(levels) < GROUP_FOLDS or np.any(levels == '') or heldout_fold not in (2, 3):
        raise ValueError('At least three nonempty merged groups required')
    folds = np.asarray(original_folds)
    if folds.shape != groups.shape or not np.issubdtype(folds.dtype, np.integer):
        raise ValueError('Integer original event folds must align with merged groups')
    expected = sorted(set((2, 3, 4, 5)) - {heldout_fold})
    if np.unique(folds).tolist() != expected:
        raise ValueError('D11 CV requires all three original FIT event folds and excludes OUTER1/top-level heldout')
    assignment = {}
    for group in levels:
        ids = np.unique(folds[groups == group])
        if len(ids) != 1:
            raise ValueError('Merged group crosses original event folds')
        assignment[str(group)] = int(ids[0])
    return folds, dict(group_assignment=assignment, groups=len(levels), folds=GROUP_FOLDS,
                       original_validation_folds=expected, top_level_heldout_fold=int(heldout_fold),
                       algorithm='Use the three original event folds inside this FIT; retain every whole merged group. No random reassignment, target values or S labels.')


def group_cv(features, y, closed, mask, weights, groups, original_folds, heldout_fold,
             penalties=PENALTIES, *, scale_floor=SCALE_FLOOR,
             feature_provider: Callable | None = None):
    """Finite three-penalty conditional CV, never S-targeted.

    feature_provider(train_positions, validation_positions) returns two aligned
    arrays whose input geometry was fit only on that CV FIT. It is called once
    per fold, not per penalty. If absent, features must already be supplied and
    their preprocessing scope is explicitly reported as caller-owned.
    The fixed carrier may have seen all top-level FIT labels; this is conditional
    regularizer selection for a residual probe, not independent NN validation.
    """
    y, closed, mask, weights, _ = validate_target(y, closed, mask, weights)
    groups = np.asarray(groups)
    if groups.shape != (len(y),):
        raise ValueError('Merged groups do not align with county-event targets')
    penalties = tuple(sorted(float(value) for value in penalties))
    if penalties != PENALTIES:
        raise ValueError('D11 penalties are fixed at 1e-4,1e-2,1')
    if feature_provider is None:
        features = validate_features(features, len(y))
    folds, assignment = deterministic_group_folds(groups, original_folds, heldout_fold)
    totals = {penalty: dict(candidate_SSE=0., baseline_SSE=0., alignment=0., modification_energy=0., net_gain=0.) for penalty in penalties}
    fold_reports, coverage, validation_mass = [], np.zeros(len(y), int), 0.
    d = FEATURES
    for fold in assignment['original_validation_folds']:
        train, held = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        if set(groups[train]) & set(groups[held]):
            raise AssertionError('Merged group split across conditional CV')
        if feature_provider is None:
            xtrain, xheld = features[train], features[held]
        else:
            xtrain, xheld = feature_provider(train.copy(), held.copy())
        xtrain = validate_features(xtrain, len(train), d)
        d = xtrain.shape[2]
        xheld = validate_features(xheld, len(held), d)
        moments = weighted_moments(xtrain, y[train], closed[train], mask[train], weights[train], scale_floor=scale_floor)
        if not mask[held].any():
            raise ValueError('Conditional CV validation fold has no observed target; do not discard it')
        masses = float(weights[held] @ mask[held].sum(1))
        validation_mass += masses
        coverage[held] += 1
        reports = []
        for penalty in penalties:
            model = fit_from_moments(moments, penalty)
            delta = predict_delta(model, xheld)
            score = masked_residual_score(y[held], closed[held], delta, mask[held], weights[held])
            for key in totals[penalty]:
                totals[penalty][key] += score[key]
            reports.append(dict(penalty=penalty, candidate_SSE=score['candidate_SSE'], baseline_SSE=score['baseline_SSE'],
                                alignment=score['alignment'], modification_energy=score['modification_energy'],
                                net_gain=score['net_gain'], condition_number=model.condition_number,
                                relative_normal_equation_residual=model.relative_normal_equation_residual,
                                slope_effective_df=model.slope_effective_df,
                                effective_df_including_intercept=model.effective_df_including_intercept))
        fold_reports.append(dict(fold=fold, FIT_units=len(train), validation_units=len(held),
                                 FIT_groups=len(np.unique(groups[train])), validation_groups=len(np.unique(groups[held])),
                                 validation_original_design_hour_mass=masses, features=d, penalties=reports))
    if not np.all(coverage == 1):
        raise AssertionError('Conditional CV validation units not covered exactly once')
    if not np.isclose(validation_mass, float(weights @ mask.sum(1)), rtol=2e-12, atol=2e-12):
        raise AssertionError('Conditional CV fixed denominator coverage failed')
    scores = []
    for penalty in penalties:
        score = dict(penalty=penalty, **totals[penalty])
        score['pooled_validation_MSE'] = score['candidate_SSE'] / validation_mass
        scores.append(score)
    # Only exact score ties are broken. No tolerance-based replacement of the
    # registered objective; maximum penalty is the fixed conservative tie rule.
    best = min(scores, key=lambda score: (score['candidate_SSE'], -score['penalty']))
    return dict(selected_penalty=best['penalty'], penalties=list(penalties), scores=scores,
                original_design_hour_mass=validation_mass, fold_reports=fold_reports,
                group_plan=assignment, county_validation_fold=folds.tolist(),
                selection_target='Pooled original-w observed complete144h residual SSE to the same fixed closed carrier, not S alone or regime-Zr/HT training weights.',
                tie_rule='Choose largest penalty among exact minimum-SSE ties.',
                geometry_scope='Each CV FIT rebuilt by feature_provider; caller must preserve its provenance.' if feature_provider is not None else 'Caller-supplied feature geometry; this module refits only ridge moments inside CV.',
                carrier_is_fixed=True, conditional_regularizer_selection=True,
                independent_neural_validation=False, final_FIT_model_fit=False,
                correction_clipped=False, stock_constraints_imposed=False,
                confidence_intervals=False)


def self_check():
    """Reproducible synthetic algebra/guard checks; no IO or real inputs."""
    checks = 0
    def check(value, name):
        nonlocal checks
        if not value:
            raise AssertionError(name)
        checks += 1
    def bad(call, name):
        try:
            call()
        except (ValueError, AssertionError):
            check(True, name)
        else:
            raise AssertionError(name)
    rng = np.random.default_rng(113294)
    x = rng.normal(size=(24, HOURS, FEATURES)).astype(np.float32)
    x[:, :, -1] = 7.
    closed = np.full((24, HOURS), .01, np.float32)
    y = (closed + .025 + .02 * x[:, :, 0] - .01 * x[:, :, 1] +
         rng.normal(0, .001, closed.shape)).astype(np.float32)
    mask = rng.random(y.shape) > .13
    weights = np.arange(1, 25, dtype=np.float32)
    mask[-1] = False
    moments = weighted_moments(x, y, closed, mask, weights)
    models = {penalty: fit_from_moments(moments, penalty) for penalty in PENALTIES}
    model = models[.01]
    delta = predict_delta(model, x)
    score = masked_residual_score(y, closed, delta, mask, weights)
    check(delta.shape == (24, HOURS), 'Shared full144 output')
    check(delta.dtype == np.float64, 'Float64 output')
    check(model.coefficient.shape == (FEATURES,), '113 coefficients')
    check(model.standardized_coefficient[-1] == 0, 'Constant feature exact zero')
    check(model.effective_df_including_intercept == 1 + model.slope_effective_df, 'Unpenalized bias df')
    check(0 <= model.slope_effective_df <= FEATURES - 1, 'Slope df support')
    check(model.relative_normal_equation_residual <= 1e-10, 'Normal equations')
    check(model.condition_number >= 1, 'Finite condition')
    check(moments.observed_units == 23, 'All-missing unit excluded from mass')
    check(moments.observed_rows == mask.sum(), 'Original mask count')
    check(abs(score['net_gain'] - score['alignment'] + score['modification_energy']) < 1e-10, 'Alignment minus energy')
    check(np.allclose(score['net_gain_rows'], score['baseline_SSE_rows'] - score['candidate_SSE_rows'],
                      rtol=2e-10, atol=2e-10), 'Per-row SSE identity')
    check(score['positive_units'] + score['negative_units'] + score['zero_units'] == 23, 'Case conservation')
    check(score['stock_constraints_imposed'] is False, 'Diagnostic is not stock')
    check(score['correction_clipped'] is False, 'Raw correction')
    # Independent augmented system: only the first (intercept) diagonal is zero.
    observed_x = np.asarray(x, np.float64)[mask]
    residual = (np.asarray(y, np.float64) - np.asarray(closed, np.float64))[mask]
    mass = np.broadcast_to(np.asarray(weights, np.float64)[:, None], mask.shape)[mask]
    mass /= mass.sum()
    z = (observed_x - moments.feature_mean) / moments.feature_scale
    z[:, moments.constant_features] = 0.
    augmented = np.column_stack([np.ones(len(z)), z])
    penalty_matrix = np.diag([0.] + [.01] * FEATURES)
    reference = np.linalg.solve((augmented * mass[:, None]).T @ augmented + penalty_matrix,
                                augmented.T @ (mass * residual))
    check(np.allclose(reference[1:], model.standardized_coefficient, rtol=1e-10, atol=1e-12), 'Independent augmented ridge')
    check(abs(reference[0] - moments.residual_mean) < 1e-12, 'Independent intercept')
    other = fit_ridge(x, y, closed, mask, weights * 7, .01)
    check(np.allclose(model.coefficient, other.coefficient, rtol=1e-10, atol=1e-12), 'Weight scale invariance')
    check(abs(model.intercept - other.intercept) < 1e-12, 'Intercept weight invariance')
    missing = y.copy()
    missing[~mask] = np.nan
    other = fit_ridge(x, missing, closed, mask, weights, .01)
    check(np.array_equal(model.coefficient, other.coefficient), 'Masked NaN is irrelevant')
    missing[~mask] = 9e30
    other = fit_ridge(x, missing, closed, mask, weights, .01)
    check(np.array_equal(model.coefficient, other.coefficient), 'Masked extreme is irrelevant')
    other = fit_ridge(x.astype(np.float64), y.astype(np.float64), closed.astype(np.float64),
                      mask, weights.astype(np.float64), .01)
    check(np.array_equal(model.coefficient, other.coefficient), 'Stored-f32 promotion before arithmetic')
    check(model.intercept == other.intercept, 'Stored-f32 intercept promotion')
    check(np.linalg.norm(models[1e-4].standardized_coefficient) >= np.linalg.norm(models[1.].standardized_coefficient), 'Ridge shrinkage')
    check(models[1e-4].slope_effective_df >= models[1.].slope_effective_df, 'Effective df shrinks')
    permutation = rng.permutation(len(y))
    other = fit_ridge(x[permutation], y[permutation], closed[permutation], mask[permutation], weights[permutation], .01)
    check(np.allclose(model.coefficient, other.coefficient, rtol=1e-9, atol=1e-11), 'Unit order invariance')
    constant = np.full_like(y, .123)
    intercept_only = fit_ridge(x, constant, np.zeros_like(closed), mask, weights, 1.)
    check(np.max(abs(predict_delta(intercept_only, x) - float(constant[0, 0]))) < 1e-13, 'Bias matches exact stored target')
    for name, call in [
            ('Empty FIT', lambda: weighted_moments(x, y, closed, np.zeros_like(mask), weights)),
            ('Nonbinary mask', lambda: weighted_moments(x, y, closed, mask.astype(int) * 2, weights)),
            ('Zero weight', lambda: weighted_moments(x, y, closed, mask, weights * 0)),
            ('Changed floor', lambda: weighted_moments(x, y, closed, mask, weights, scale_floor=1e-6)),
            ('Changed horizon', lambda: weighted_moments(x[:, :143], y[:, :143], closed[:, :143], mask[:, :143], weights)),
            ('Unregistered penalty', lambda: fit_from_moments(moments, .001)),
            ('Nonpositive penalty', lambda: fit_from_moments(moments, 0)),
            ('Changed feature count', lambda: predict_delta(model, x[:, :, :112]))]:
        bad(call, name)
    original_folds = np.repeat([3, 4, 5], 8)
    groups = np.array(['g' + str(j // 2) for j in range(len(y))])
    plan, info = deterministic_group_folds(groups, original_folds, 2)
    check(np.array_equal(plan, original_folds), 'Fixed original event-fold IDs')
    check(info['original_validation_folds'] == [3, 4, 5], 'Three original folds')
    other_plan, other_info = deterministic_group_folds(groups[permutation], original_folds[permutation], 2)
    check(other_info['group_assignment'] == info['group_assignment'], 'Group plan order invariance')
    check(np.array_equal(other_plan, original_folds[permutation]), 'Fold assignments preserved')
    wrong = original_folds.copy()
    wrong[1] = 4
    bad(lambda: deterministic_group_folds(groups, wrong, 2), 'Merged group cannot split')
    bad(lambda: deterministic_group_folds(groups, np.repeat([2, 4, 5], 8), 2), 'Top heldout forbidden')
    bad(lambda: deterministic_group_folds(groups, np.repeat([1, 4, 5], 8), 2), 'Original OUTER1 forbidden')
    cv = group_cv(x, y, closed, mask, weights, groups, original_folds, 2)
    check(len(cv['fold_reports']) == 3, 'Three CV reports')
    check(sum(len(row['penalties']) for row in cv['fold_reports']) == 9, 'Nine fixed CV solves')
    check(cv['selected_penalty'] in PENALTIES, 'Finite penalty selection')
    check(cv['final_FIT_model_fit'] is False, 'No automatic extra final solve')
    check(cv['independent_neural_validation'] is False, 'Conditional carrier limitation')
    check(abs(cv['original_design_hour_mass'] - moments.design_hour_mass) < 1e-10, 'CV fixed denominator')
    check(cv['county_validation_fold'] == original_folds.tolist(), 'Each unit validated in original fold')
    check(cv['stock_constraints_imposed'] is False, 'CV remains raw diagnostic')
    provider_calls = []
    def provider(train, held):
        provider_calls.append((train.copy(), held.copy()))
        check(not (set(groups[train]) & set(groups[held])), 'Feature-provider whole groups')
        return x[train], x[held]
    provided = group_cv(None, y, closed, mask, weights, groups, original_folds, 2, feature_provider=provider)
    check(len(provider_calls) == 3, 'Geometry rebuilt once per split')
    check(provided['selected_penalty'] == cv['selected_penalty'], 'Provider selection parity')
    check(provided['scores'] == cv['scores'], 'Provider Gram parity')
    zero = np.zeros_like(y)
    ties = group_cv(x, zero, zero, mask, weights, groups, original_folds, 2)
    check(ties['selected_penalty'] == 1., 'Exact tie chooses larger penalty')
    bad(lambda: group_cv(x, y, closed, mask, weights, groups, original_folds, 2,
                         penalties=(1e-4, .01, 1., 10.)), 'No expanded penalty search')
    bad(lambda: group_cv(x[:, :, :112], y, closed, mask, weights, groups, original_folds, 2), '113 columns fixed')
    empty_fold_mask = mask.copy()
    empty_fold_mask[original_folds == 3] = False
    bad(lambda: group_cv(x, y, closed, empty_fold_mask, weights, groups, original_folds, 2), 'Empty CV fold is failure, not dropped')
    return checks
