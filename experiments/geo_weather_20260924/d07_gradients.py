"""D07 C: frozen fold1 FIT gradients and six registered local perturbations.

No optimizer, calibration, checkpoint saving, or training-history inference.
All partition objectives use the same complete FIT denominator. Numerical
vectors and predictions remain in ignored local NPZ files for audit.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'

import copy
import hashlib
import json
from pathlib import Path
import sys
import traceback
from collections import OrderedDict

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'src'))
from asymode.gcrk_train import make_batch

CHUNK = 64
EPSILONS = (.001, .01)
PERTURB_BLOCKS = ('weather_control', 'raw_geography', 'kernel_basis')
NORM_FLOOR = 1e-3
COSINE_NORM_MIN = 1e-20
ADD_RTOL, ADD_ATOL = 3e-4, 3e-9


def finite_json(value):
    """JSON conversion; a nonfinite diagnostic is an error, never hidden."""
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [finite_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return finite_json(value.tolist())
    if isinstance(value, np.generic):
        return finite_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError('nonfinite JSON diagnostic')
    return value


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(finite_json(value), stream, ensure_ascii=False, indent=1,
                  allow_nan=False)
        stream.write('\n')


def weighted_objective(P, y, mask, row_weight, full_fit_denominator):
    """Masked original w/Z objective with one fixed, external FIT denominator."""
    if P.shape != y.shape or P.shape != mask.shape or row_weight.shape != P.shape[:1]:
        raise ValueError('objective shapes differ')
    denominator = torch.as_tensor(full_fit_denominator, dtype=P.dtype, device=P.device).detach()
    if not bool(torch.isfinite(denominator)) or float(denominator) <= 0:
        raise ValueError('full FIT denominator must be finite and positive')
    if not bool(torch.isfinite(row_weight).all()) or bool((row_weight < 0).any()):
        raise ValueError('invalid objective weights')
    return ((P-y).square() * mask * row_weight[:, None]).sum() / denominator


def layout(model):
    """A disjoint, exhaustive coordinate partition, including sliced maps."""
    params = OrderedDict(model.named_parameters())
    offsets, size = {}, 0
    for name, parameter in params.items():
        offsets[name] = np.arange(size, size + parameter.numel()).reshape(parameter.shape)
        size += parameter.numel()
    blocks = OrderedDict((name, []) for name in (
        'host', 'recovery', 'weather_control', 'raw_geography', 'kernel_basis',
        'shared_fusion', 'geometry', 'dynamics', 'exit'))
    kernel_names = {
        'weather_control': ('fusion_weather', 'head_weather'),
        'shared_fusion': ('head_fusion', 'fusion_bias', 'head_bias'),
        'geometry': ('length_raw', 'order_logits'),
        'dynamics': ('plane_a', 'plane_b'), 'exit': ('readout', 'alpha')}
    members = {name: [] for name in blocks}
    prefix = 'damage.2.'
    for name, parameter in params.items():
        short = name[len(prefix):] if name.startswith(prefix) else None
        if name.startswith('recovery.'):
            block = 'recovery'
        elif short in ('fusion_geo', 'head_geo'):
            if parameter.ndim != 2 or parameter.shape[1] != 72:
                raise ValueError('expected exactly 40 raw + 32 landmark mapping columns')
            for block, sl in [('raw_geography', slice(0, 40)),
                              ('kernel_basis', slice(40, None))]:
                blocks[block].append(offsets[name][:, sl].reshape(-1))
                members[block].append(dict(parameter=name, columns=[sl.start, sl.stop or 72]))
            continue
        elif short is None or short in ('weight', 'bias'):
            block = 'host'
        else:
            matches = [key for key, fields in kernel_names.items() if short in fields]
            if len(matches) != 1:
                raise ValueError('unclassified controlled-relaxation parameter: ' + name)
            block = matches[0]
        blocks[block].append(offsets[name].reshape(-1))
        members[block].append(dict(parameter=name, columns=None))
    blocks = OrderedDict((name, np.concatenate(parts).astype(np.int64)
                          if parts else np.empty(0, np.int64))
                         for name, parts in blocks.items())
    joined = np.concatenate(list(blocks.values()))
    if not np.array_equal(np.sort(joined), np.arange(size)):
        raise AssertionError('parameter blocks are not exhaustive and disjoint')
    if any(not len(ix) for ix in blocks.values()):
        raise AssertionError('a required gradient block is empty')
    baseline = np.concatenate([p.detach().cpu().double().numpy().reshape(-1)
                               for p in params.values()])
    return dict(params=params, offsets=offsets, blocks=blocks, members=members,
                size=size, baseline=baseline)


def collect_gradient(model, F, stats, ids, row_weight, denominator, title='', chunk=CHUNK):
    """Visit each supplied row once and accumulate one common-denominator gradient."""
    ids = np.asarray(ids, np.int64)
    if len(np.unique(ids)) != len(ids):
        raise ValueError('duplicate gradient rows')
    model.eval()
    model.zero_grad(set_to_none=True)
    loss, seen = 0.0, 0
    for start in range(0, len(ids), chunk):
        take = ids[start:start+chunk]
        batch = make_batch(F, take, stats)
        P = model(batch, exit_open=True)['P']
        weights = torch.as_tensor(row_weight[take], dtype=P.dtype, device=P.device)
        part = weighted_objective(P, batch['y'], batch['m'], weights, denominator)
        if not bool(torch.isfinite(part)):
            raise RuntimeError('nonfinite partition objective')
        part.backward()
        loss += float(part.detach())
        seen += len(take)
        if title and (start == 0 or start % 512 == 0):
            print(f'{title}: {seen}/{len(ids)} loss_contribution={loss:.9g}', flush=True)
        del batch, P, weights, part
    pieces, disconnected = [], []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            disconnected.append(name)
            pieces.append(np.zeros(parameter.numel(), np.float64))
        else:
            grad = parameter.grad.detach().cpu().double().numpy().reshape(-1).copy()
            if not np.isfinite(grad).all():
                raise RuntimeError('nonfinite gradient: ' + name)
            pieces.append(grad)
    assert seen == len(ids)
    return np.concatenate(pieces), loss, disconnected


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na <= COSINE_NORM_MIN or nb <= COSINE_NORM_MIN:
        return None
    return float(np.clip(np.dot(a, b)/(na*nb), -1, 1))


def vector_summary(vector, parameter, total=None):
    norm, pn = float(np.linalg.norm(vector)), float(np.linalg.norm(parameter))
    return dict(coordinates=len(vector), gradient_l2=norm, parameter_l2=pn,
                gradient_l2_per_parameter_l2=norm/max(pn, NORM_FLOOR),
                parameter_scale_floor=NORM_FLOOR,
                gradient_rms=norm/np.sqrt(len(vector)) if len(vector) else None,
                gradient_max_abs=float(np.max(np.abs(vector))) if len(vector) else None,
                cosine_to_total=cosine(vector, total) if total is not None else None)


def additivity_report(partitions, total):
    reconstructed = np.sum(partitions, axis=0, dtype=np.float64)
    error = reconstructed-total
    passed = bool(np.allclose(reconstructed, total, rtol=ADD_RTOL, atol=ADD_ATOL))
    return dict(passed=passed, rtol=ADD_RTOL, atol=ADD_ATOL,
                max_abs_error=float(np.max(np.abs(error))),
                error_l2=float(np.linalg.norm(error)),
                relative_l2_error=float(np.linalg.norm(error)/max(np.linalg.norm(total), 1e-30)),
                cosine=cosine(reconstructed, total)), reconstructed


def system_partitions(data, fit, masks):
    """Original system × cohort rows; merged groups are provenance, not IDs."""
    fit = np.asarray(fit, np.int64)
    systems = np.unique(data['meta']['system'][fit])
    rows, visited = [], np.zeros(len(data['y']), np.int64)
    for system in systems:
        system_ids = fit[data['meta']['system'][fit] == system]
        merged = np.unique(data['group'][system_ids])
        if len(merged) != 1:
            raise AssertionError('original system spans merged groups')
        for cohort in ('S', 'nonS'):
            ids = system_ids[masks[cohort][system_ids]]
            if len(ids):
                rows.append((str(system), cohort, str(merged[0]), ids))
                visited[ids] += 1
    if not np.array_equal(visited[fit], np.ones(len(fit), np.int64)) or visited.sum() != len(fit):
        raise AssertionError('system/cohort partitions do not cover FIT exactly once')
    return systems, rows


def direction_report(vectors, names, total):
    vectors = np.asarray(vectors, np.float64)
    norms = np.linalg.norm(vectors, axis=1)
    tn, mass = float(np.linalg.norm(total)), float(norms.sum())
    order = np.argsort(-norms, kind='stable')
    dots = vectors@total
    projection = dots/(tn*tn) if tn > COSINE_NORM_MIN else np.zeros(len(vectors))
    top = [dict(name=names[j], gradient_l2=float(norms[j]),
                norm_mass_share=float(norms[j]/mass) if mass else None,
                signed_projection_share=float(projection[j]) if tn > COSINE_NORM_MIN else None,
                cosine_to_total=cosine(vectors[j], total)) for j in order[:5]]
    pairwise = [[cosine(a, b) for b in vectors] for a in vectors]
    edges = [(pairwise[i][j], names[i], names[j])
             for i in range(len(names)) for j in range(i+1, len(names))
             if pairwise[i][j] is not None]
    edges.sort(key=lambda x: (x[0], x[1], x[2]))
    return dict(n_vectors=len(vectors), sum_individual_gradient_l2=mass,
                sum_gradient_l2=tn,
                cancellation_fraction=1-tn/mass if mass else None,
                norm_mass_top1=float(norms[order[:1]].sum()/mass) if mass else None,
                norm_mass_top3=float(norms[order[:3]].sum()/mass) if mass else None,
                norm_mass_top5=float(norms[order[:5]].sum()/mass) if mass else None,
                positive_projection_mass=float(projection[projection > 0].sum()) if tn > COSINE_NORM_MIN else None,
                negative_projection_mass=float(-projection[projection < 0].sum()) if tn > COSINE_NORM_MIN else None,
                top_norm_contributors=top,
                most_opposed_pairs=[dict(cosine=v, a=a, b=b) for v, a, b in edges[:10]],
                most_aligned_pairs=[dict(cosine=v, a=a, b=b) for v, a, b in edges[-10:][::-1]],
                pairwise_cosine_names=names, pairwise_cosines=pairwise)


def snapshot(model):
    return copy.deepcopy(model.state_dict())


def _same(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    return a == b


def assert_state(model, frozen):
    current = model.state_dict()
    if not _same(frozen, current):
        bad = [name for name in frozen if not _same(frozen[name], current[name])]
        raise AssertionError('frozen state changed: ' + ', '.join(bad))


def state_hash(state):
    h = hashlib.sha256()
    for name, value in state.items():
        h.update(name.encode())
        if isinstance(value, torch.Tensor):
            h.update(str(value.dtype).encode())
            h.update(str(tuple(value.shape)).encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
        else:
            h.update(json.dumps(finite_json(value), sort_keys=True, separators=(',', ':')).encode())
    return h.hexdigest()


def restore(model, frozen):
    model.load_state_dict(frozen, strict=True)
    model.eval()
    model.zero_grad(set_to_none=True)
    assert_state(model, frozen)


def apply_perturbation(model, spec, gradient, block, epsilon):
    """One normalized gradient step restricted to registered block coordinates."""
    if block not in PERTURB_BLOCKS or epsilon not in EPSILONS:
        raise ValueError('unregistered parameter intervention')
    selected = spec['blocks'][block]
    gb, pb = gradient[selected], spec['baseline'][selected]
    norm, pnorm = float(np.linalg.norm(gb)), float(np.linalg.norm(pb))
    scale = pnorm if pnorm > 0 else NORM_FLOOR
    if norm <= COSINE_NORM_MIN:
        raise ValueError('registered gradient direction is zero/undefined')
    step = np.zeros(spec['size'], np.float64)
    step[selected] = -epsilon*scale*gb/norm
    current = np.concatenate([p.detach().cpu().double().numpy().reshape(-1)
                              for p in spec['params'].values()])
    if not np.array_equal(current, spec['baseline']):
        raise AssertionError('every intervention must start from the frozen baseline')
    with torch.no_grad():
        for name, parameter in spec['params'].items():
            indices = spec['offsets'][name]
            parameter.add_(torch.as_tensor(step[indices], dtype=parameter.dtype,
                                            device=parameter.device))
    after = np.concatenate([p.detach().cpu().double().numpy().reshape(-1)
                            for p in spec['params'].values()])
    actual = after-spec['baseline']
    other = np.ones(spec['size'], bool); other[selected] = False
    if not np.array_equal(after[other], spec['baseline'][other]):
        raise AssertionError('intervention changed a coordinate outside its block')
    if not np.isfinite(after).all():
        raise RuntimeError('nonfinite perturbed parameters')
    return dict(block=block, relative_epsilon=epsilon, baseline_parameter_l2=pnorm,
                parameter_norm_for_step=scale, zero_parameter_norm_floor=NORM_FLOOR,
                gradient_l2=norm, requested_step_l2=epsilon*scale,
                achieved_step_l2=float(np.linalg.norm(actual[selected])),
                achieved_relative_step=float(np.linalg.norm(actual[selected])/scale),
                descent_cosine=cosine(actual[selected], -gb),
                first_order_full_fit_objective_delta=float(np.dot(gb, actual[selected])),
                changed_coordinates=int(np.count_nonzero(actual)),
                outside_block_exactly_unchanged=True)


def objective_numpy(data, P, ids, rw, denominator):
    ids = np.asarray(ids, np.int64)
    se = np.sum(data['m'][ids]*(P[ids].astype(float)-data['y'][ids])**2, axis=1)
    return float(np.sum(rw[ids]*se)/denominator)


def paired_metrics(data, P, baseline, ids, masks, rw, denominator):
    """Fixed observed support, fixed cohorts, design weights and FIT objective."""
    w, m, y = data['meta']['w'].astype(float), data['m'], data['y']
    ids = np.asarray(ids, np.int64)
    result = {}
    for name in ('all', 'S', 'nonS', 'J'):
        take = ids[masks[name][ids]]
        support = m[take]
        hours, design_hours = int(support.sum()), float(np.sum(w[take, None]*support))
        if not len(take) or not hours:
            result[name] = dict(n_units=len(take), observed_hours=hours,
                                design_observed_hour_mass=design_hours, supported=False)
            continue
        pp, bb, yy = P[take].astype(float), baseline[take].astype(float), y[take]
        se_p, se_b = np.sum(support*(pp-yy)**2, 1), np.sum(support*(bb-yy)**2, 1)
        change = np.sum(support*(pp-bb)**2, 1)
        delta = se_p-se_b
        sse_p, sse_b = float(w[take]@se_p), float(w[take]@se_b)
        peakp = np.max(np.where(support, pp, -np.inf), 1)
        peakb = np.max(np.where(support, bb, -np.inf), 1)
        alarm_p, alarm_b = peakp >= .1, peakb >= .1
        weight_mass = float(w[take].sum())
        row = dict(n_units=len(take), observed_hours=hours,
                   observed_units=int(support.any(1).sum()),
                   merged_event_groups=int(len(np.unique(data['group'][take]))),
                   original_systems=int(len(np.unique(data['meta']['system'][take]))),
                   design_unit_weight_mass=weight_mass,
                   design_observed_hour_mass=design_hours,
                   original_normalized_fit_objective=float(rw[take]@se_p/denominator),
                   baseline_original_normalized_fit_objective=float(rw[take]@se_b/denominator),
                   paired_objective_delta=float(rw[take]@delta/denominator),
                   design_RMSE=float(np.sqrt(sse_p/design_hours)),
                   baseline_design_RMSE=float(np.sqrt(sse_b/design_hours)),
                   paired_design_RMSE_delta=float(np.sqrt(sse_p/design_hours)-np.sqrt(sse_b/design_hours)),
                   paired_design_SSE_delta=float(w[take]@delta),
                   prediction_change_design_RMSE=float(np.sqrt(w[take]@change/design_hours)),
                   prediction_change_max_abs_observed=float(np.max(np.abs(pp-bb)[support])),
                   units_improved=int((delta < 0).sum()), units_worsened=int((delta > 0).sum()),
                   units_unchanged=int((delta == 0).sum()),
                   improved_design_weight_share=float(w[take][delta < 0].sum()/weight_mass),
                   worsened_design_weight_share=float(w[take][delta > 0].sum()/weight_mass),
                   observed_predicted_severe_peak=dict(n=int(alarm_p.sum()),
                      weighted_rate=float(w[take][alarm_p].sum()/weight_mass)),
                   baseline_observed_predicted_severe_peak=dict(n=int(alarm_b.sum()),
                      weighted_rate=float(w[take][alarm_b].sum()/weight_mass)),
                   paired_severe_peak=dict(new=int((alarm_p & ~alarm_b).sum()),
                      removed=int((alarm_b & ~alarm_p).sum()),
                      retained=int((alarm_p & alarm_b).sum()),
                      weighted_rate_delta=float(w[take]@(alarm_p.astype(float)-alarm_b.astype(float))/weight_mass)))
        if name == 'nonS':
            row['observed_severe_false_alarms'] = row['observed_predicted_severe_peak']
            row['baseline_observed_severe_false_alarms'] = row['baseline_observed_predicted_severe_peak']
            row['paired_severe_false_alarms'] = row['paired_severe_peak']
        result[name] = row
    return result


def main():
    import d07_common as C
    from evaluate_cr_tail import cohorts
    result_path = C.HERE / 'results/v1/d07_gradients.json'
    vector_path = C.OUT / 'gradients_vectors.npz'
    prediction_path = C.OUT / 'gradients_parameter_predictions.npz'
    for path in (result_path, vector_path, prediction_path):
        if path.exists():
            raise FileExistsError('Preserve earlier D07 artifact: ' + str(path))
    nice = os.getpriority(os.PRIO_PROCESS, 0)
    if nice < 15:
        raise RuntimeError('D07 numeric process must have nice >= 15')
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    print(f'D07 C pid={os.getpid()} nice={nice} threads={torch.get_num_threads()}', flush=True)
    frozen_provenance = C.provenance()
    own_sources = {Path(__file__).name: sha(__file__), 'd07_common.py': sha(C.HERE/'d07_common.py')}
    data, F = C.load_data(load_inputs=True)
    n = len(data['y']); assert n == 8457
    fit = np.sort(np.asarray(data['split']['event']['1']['dev'], np.int64))
    outer = np.sort(np.asarray(data['expected'][1], np.int64))
    assert len(fit) == 6350 and len(outer) == 2107
    assert np.array_equal(np.sort(np.r_[fit, outer]), np.arange(n))
    masks = cohorts(data)
    rw, denominator, Z = C.fit_weights(data, fold=1)
    rw = np.asarray(rw, np.float64); denominator = float(denominator)
    assert np.isfinite(rw).all() and (rw >= 0).all()
    assert np.isclose(denominator, np.sum(rw[fit, None]*data['m'][fit]))
    model, stats, receipt = C.load_model(F, data, 'v1_crk_s0', fold=1)
    model.eval()
    frozen = snapshot(model); initial_state_sha = state_hash(frozen)
    spec = layout(model)
    print(f'{spec["size"]} parameters; full FIT original denominator={denominator:.12g}', flush=True)
    baseline = C.evaluate(model, F, stats, np.arange(n), chunk=CHUNK, exit_open=True)
    assert baseline.shape == data['y'].shape and np.isfinite(baseline).all()
    assert_state(model, frozen)
    with np.load(C.RUNS/'v1_crk_s0/fold01/outer.npz', allow_pickle=False) as z:
        replay_error = float(np.max(np.abs(baseline[z['idx']]-z['P'])))
    assert replay_error < 2e-6, replay_error
    total, total_loss, total_disconnected = collect_gradient(
        model, F, stats, fit, rw, denominator, title='separate full FIT')
    assert_state(model, frozen)
    groups, partition_rows = system_partitions(data, fit, masks)
    partition_names, partition_info, vectors, visited = [], [], [], np.zeros(n, np.int64)
    for system, cohort, merged_group, ids in partition_rows:
        name = system + '/' + cohort
        vector, loss, disconnected = collect_gradient(
            model, F, stats, ids, rw, denominator, title=name)
        assert_state(model, frozen)
        visited[ids] += 1
        vectors.append(vector); partition_names.append(name)
        partition_info.append(dict(system=system, cohort=cohort, merged_group=merged_group,
            n_units=len(ids), observed_hours=int(data['m'][ids].sum()),
            design_weight_mass=float(data['meta']['w'][ids].sum()),
            normalized_hour_mass=float(np.sum(rw[ids, None]*data['m'][ids])),
            full_fit_objective_contribution=loss, disconnected_parameters=disconnected))
    assert np.array_equal(visited[fit], np.ones(len(fit), np.int64))
    assert visited[outer].sum() == 0
    vectors = np.asarray(vectors, np.float64)
    additivity, reconstructed = additivity_report(vectors, total)
    print('gradient additivity:', json.dumps(additivity), flush=True)
    if not additivity['passed']:
        failure = C.OUT/'gradients_additivity_failure.npz'
        with failure.open('xb') as stream:
            np.savez_compressed(stream, separate_total=total, reconstructed=reconstructed,
                                partition_vectors=vectors, partition_names=partition_names)
        raise AssertionError('partition gradients fail full FIT additivity')
    loss_sum = sum(row['full_fit_objective_contribution'] for row in partition_info)
    numpy_loss = objective_numpy(data, baseline, fit, rw, denominator)
    assert np.isclose(loss_sum, total_loss, rtol=1e-5, atol=1e-8)
    assert np.isclose(numpy_loss, total_loss, rtol=1e-5, atol=1e-8)
    cohort_vectors = {name: vectors[[i for i, row in enumerate(partition_info)
                                    if row['cohort'] == name]].sum(0)
                      for name in ('S', 'nonS')}
    event_vectors = np.stack([vectors[[i for i, row in enumerate(partition_info)
                                       if row['system'] == str(group)]].sum(0) for group in groups])
    event_names = [str(group) for group in groups]
    blocks = {}
    for name, selected in spec['blocks'].items():
        sb = total[selected]
        blocks[name] = dict(members=spec['members'][name],
            total=vector_summary(sb, spec['baseline'][selected]),
            partition_additivity=additivity_report(vectors[:, selected], sb)[0],
            partition_directions=direction_report(vectors[:, selected], partition_names, sb),
            system_directions=direction_report(event_vectors[:, selected], event_names, sb),
            S=vector_summary(cohort_vectors['S'][selected], spec['baseline'][selected], sb),
            nonS=vector_summary(cohort_vectors['nonS'][selected], spec['baseline'][selected], sb),
            S_nonS_cosine=cosine(cohort_vectors['S'][selected], cohort_vectors['nonS'][selected]))
    for i, row in enumerate(partition_info):
        row['gradient'] = vector_summary(vectors[i], spec['baseline'], total)
        row['block_gradients'] = {name: vector_summary(vectors[i, ix], spec['baseline'][ix], total[ix])
                                  for name, ix in spec['blocks'].items()}
    baseline_scores = {split: paired_metrics(data, baseline, baseline, ids, masks, rw, denominator)
                       for split, ids in [('FIT', fit), ('OUTER', outer)]}
    baseline_standard_scores = {split: C.score(data, baseline, ids)
                                for split, ids in [('FIT', fit), ('OUTER', outer)]}
    interventions, predictions = [], {'baseline': baseline}
    for block in PERTURB_BLOCKS:
        for epsilon in EPSILONS:
            restore(model, frozen)
            before_hash = state_hash(model.state_dict())
            assert before_hash == initial_state_sha
            tag = block + '_' + str(epsilon).replace('.', 'p')
            print('registered intervention ' + tag, flush=True)
            try:
                intervention = apply_perturbation(model, spec, total, block, epsilon)
                perturbed_hash = state_hash(model.state_dict())
                assert perturbed_hash != initial_state_sha
                P = C.evaluate(model, F, stats, np.arange(n), chunk=CHUNK, exit_open=True)
                assert np.isfinite(P).all() and P.shape == baseline.shape
                intervention.update(name=tag, before_state_sha256=before_hash,
                    perturbed_state_sha256=perturbed_hash,
                    scores={split: paired_metrics(data, P, baseline, ids, masks, rw, denominator)
                            for split, ids in [('FIT', fit), ('OUTER', outer)]},
                    standard_scores={split: C.score(data, P, ids)
                                     for split, ids in [('FIT', fit), ('OUTER', outer)]})
                predictions[tag] = P
            finally:
                restore(model, frozen)
            restored_hash = state_hash(model.state_dict())
            assert restored_hash == initial_state_sha
            intervention.update(restored_state_sha256=restored_hash,
                                exact_parameter_buffer_restoration=True)
            interventions.append(intervention)
            print(tag + ' FIT/OUTER objective delta ' + str([
                intervention['scores'][key]['all']['paired_objective_delta']
                for key in ('FIT', 'OUTER')]), flush=True)
    assert len(interventions) == 6
    # provenance() caches its validated receipt, so explicitly rehash every
    # sealed input here rather than merely comparing the cached dictionary.
    assert sha(C.FEATURES) == frozen_provenance['data_sha256']
    for filename, digest in frozen_provenance['frozen_source_sha256'].items():
        assert sha(C.ROOT/filename) == digest, filename
    for label, folds in frozen_provenance['frozen_artifact_sha256'].items():
        for fold, files in folds.items():
            for filename, digest in files.items():
                assert sha(C.RUNS/label/f'fold{int(fold):02d}'/filename) == digest, (label, fold, filename)
    assert own_sources == {Path(__file__).name: sha(__file__), 'd07_common.py': sha(C.HERE/'d07_common.py')}
    assert_state(model, frozen)
    vector_arrays = dict(full_fit_gradient=total, reconstructed_gradient=reconstructed,
        partition_gradients=vectors, partition_names=np.asarray(partition_names),
        system_gradients=event_vectors, system_names=np.asarray(event_names),
        S_gradient=cohort_vectors['S'], nonS_gradient=cohort_vectors['nonS'],
        baseline_parameters=spec['baseline'], parameter_names=np.asarray(list(spec['params'])),
        fit_ids=fit, outer_ids=outer, row_weight=rw,
        **{'block_indices_'+name: ix for name, ix in spec['blocks'].items()})
    with vector_path.open('xb') as stream:
        np.savez_compressed(stream, **vector_arrays)
    with prediction_path.open('xb') as stream:
        np.savez_compressed(stream, **predictions)
    output = dict(scope_commit='8f8ddef', phase='C', exploratory=True,
        checkpoint='v1_crk_s0/fold01 step900', gradients_are_current_working_point_not_training_history=True,
        no_optimizer_or_calibration=True, model_eval_mode=True,
        numeric_policy=dict(pid=os.getpid(), nice=nice, threads=2, chunk=CHUNK),
        source_sha256=own_sources, frozen_provenance=frozen_provenance,
        checkpoint_receipt=receipt, initial_and_final_state_sha256=initial_state_sha,
        objective=dict(formula='sum_i,t m_it*rw32_i*(P_it-y_it)^2 / torch_float32_sum_FIT_i,t(m_it*rw32_i)',
            row_weights='rw32_i=float32(scale*w_i/Z_regime(i)); original training cell-weight scaling',
            ideal_unrounded_equivalent='sum_i,t m_it*(w_i/Z_regime(i))*(P_it-y_it)^2 / sum_FIT_i,t m_it*(w_i/Z_regime(i))',
            Z_regime=Z, common_full_FIT_denominator=denominator,
            FIT_units=len(fit), OUTER_units=len(outer),
            FIT_original_systems=len(groups), FIT_merged_groups=len(np.unique(data['group'][fit])),
            same_full_FIT_denominator_for_all_partitions_and_OUTER=True,
            original_training_global_float32_rescaling_cancels=True,
            separate_full_FIT_loss=total_loss, partition_loss_sum=loss_sum,
            baseline_float64_objective=numpy_loss,
            separate_full_FIT_disconnected_parameters=total_disconnected),
        validation=dict(partition_membership_exactly_once=True, finite_gradients=True,
            partition_additivity=additivity, losses_additive=True,
            outer_checkpoint_replay_max_abs=replay_error,
            exactly_six_independent_registered_interventions=True,
            every_intervention_restored_exactly=True, frozen_hashes_unchanged=True),
        cosine_missing_below_norm=COSINE_NORM_MIN,
        partitions=partition_info, blocks=blocks,
        global_partition_directions=direction_report(vectors, partition_names, total),
        global_system_directions=direction_report(event_vectors, event_names, total),
        global_S_nonS_cosine=cosine(cohort_vectors['S'], cohort_vectors['nonS']),
        cohort_objective_contributions={name: sum(row['full_fit_objective_contribution']
             for row in partition_info if row['cohort'] == name) for name in ('S', 'nonS')},
        baseline_scores=baseline_scores, baseline_standard_scores=baseline_standard_scores,
        parameter_interventions=interventions,
        interpretation='Local frozen-state sensitivity only; no OUTER selection, no fair retraining claim.',
        earlier_attempts=[dict(log=name, sha256=sha(C.OUT/name), stage=stage, reason=reason,
                              completed_parameter_interventions=0)
            for name, stage, reason in (
                ('gradients_import_failure.log', 'launcher before numeric work',
                 'runpy did not put the script directory on sys.path; owned import path fixed'),
                ('gradients_partition_setup_interrupted.log', 'separate full FIT gradient pass',
                 'owned process stopped before partitions/interventions to correct original-system versus merged-group IDs'))
            if (C.OUT/name).exists()],
        local_artifacts={vector_path.name: sha(vector_path), prediction_path.name: sha(prediction_path)})
    write_new(result_path, output)
    print('D07 C complete ' + str(result_path), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
