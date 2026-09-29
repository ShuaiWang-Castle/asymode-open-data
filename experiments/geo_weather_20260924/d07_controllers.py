"""D07 B: frozen, public-D controller working points and write elasticity.

All timings use the existing observation mask. Controller derivatives are from
64 calibrated asinh hidden level/departure coordinates to 32 deposit coordinates,
not derivatives from weather observations to outage fractions. Only main reads D.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'
import copy
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import types

import numpy as np
import torch
from torch.nn import functional as tf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from asymode.gcrk_train import make_batch
from evaluate_cr_tail import cohorts, clean, sha, write_new, REGIMES

EPS = 1e-8
SCALES = (.95, 1.05)
SATURATION = .99
NORM_THRESHOLD, COS_THRESHOLD = .9, .95
NORM_BINS = np.array([0., .25, .5, .75, .9, 1.00001])
COS_BINS = np.array([-1.00001, -.5, 0., .5, .95, 1.00001])
SCALAR_KEYS = ('P', 'u', 'r', 'raw_logit', 'logit', 'gate', 'background',
               'conditional', 'forget')


def preactivation(u, fg, anchor, fusion_weather, head_weather, head_fusion):
    psi = torch.tanh(fg[:, None] + tf.linear(u, fusion_weather))
    raw = anchor[:, None] + tf.linear(u, head_weather) + tf.linear(
        psi - torch.tanh(fg)[:, None], head_fusion)
    return psi, raw


def scaled_deposit(raw, anchor, scale=1.):
    a = anchor[:, None, :32]
    z = a + scale * (raw[..., :32] - a)
    return ((torch.tanh(z) - torch.tanh(a)) / (2. * math.sqrt(8))).reshape(
        *raw.shape[:2], 4, 8)


def deposit_jacobian(raw, psi, anchor, fusion_weather, head_weather,
                     head_fusion, scale=1.):
    """Analytic [B,T,32,64] local derivative; geography stays fixed."""
    chain = head_weather[:32][None, None] + torch.einsum(
        'df,btf,fu->btdu', head_fusion[:32], 1. - psi.square(), fusion_weather)
    a = anchor[:, None, :32]
    z = a + scale * (raw[..., :32] - a)
    return (scale * (1. - torch.tanh(z).square()) /
            (2. * math.sqrt(8)))[..., None] * chain


@contextmanager
def write_scale(layer, scale):
    """Change only deposit. Original controls are called before replacement."""
    original = layer._controls
    had_instance_attribute = '_controls' in layer.__dict__

    def changed(self, u, fg, hg, anchor):
        controls = original(u, fg, hg, anchor)
        _, raw = preactivation(u, fg, anchor, self.fusion_weather,
                               self.head_weather, self.head_fusion)
        controls['deposit'] = scaled_deposit(raw, anchor, scale)
        return controls

    layer._controls = types.MethodType(changed, layer)
    try:
        yield
    finally:
        if had_instance_attribute:
            layer._controls = original
        else:
            del layer._controls


def cosine(a, b):
    an, bn = a.norm(dim=-1), b.norm(dim=-1)
    good = (an > EPS) & (bn > EPS)
    values = (a * b).sum(-1) / (an * bn).clamp_min(EPS * EPS)
    return torch.where(good, values.clamp(-1., 1.), torch.full_like(values, float('nan')))


def snapshot(model):
    return copy.deepcopy(model.state_dict())


def assert_frozen(model, before):
    after = model.state_dict()
    assert before.keys() == after.keys()
    for key, val in before.items():
        if isinstance(val, torch.Tensor):
            assert torch.equal(val, after[key]), key
        else:
            assert val == after[key], key


def state_digest(state):
    h = hashlib.sha256()
    for key in sorted(state):
        h.update(key.encode())
        val = state[key]
        if isinstance(val, torch.Tensor):
            h.update(str((tuple(val.shape), str(val.dtype))).encode())
            h.update(val.detach().cpu().numpy().tobytes())
        else:
            h.update(json.dumps(val, sort_keys=True).encode())
    return h.hexdigest()


def dist(values, weights):
    values, weights = np.asarray(values, 'f8').ravel(), np.asarray(weights, 'f8').ravel()
    assert values.shape == weights.shape
    good = np.isfinite(values)
    v, w = values[good], weights[good]
    ans = {'n': len(values), 'valid': len(v), 'missing': int((~good).sum()),
           'valid_weight': float(w.sum()), 'missing_weight': float(weights[~good].sum())}
    if not len(v):
        return {**ans, 'mean': None, 'q10_q50_q90': None}
    order = np.argsort(v, kind='stable')
    vs, ws = v[order], w[order]
    cdf = (np.cumsum(ws) - .5 * ws) / ws.sum()
    return {**ans, 'mean': float(np.sum(v * w) / w.sum()),
            'q10_q50_q90': np.interp([.1, .5, .9], cdf, vs).tolist()}


def fraction(values, weights, valid=None):
    values, weights = np.asarray(values), np.asarray(weights, 'f8')
    if valid is None:
        valid = np.ones(values.shape, dtype=bool)
    vv, ww = values[valid], weights[valid]
    return dict(valid=int(valid.sum()), missing=int((~valid).sum()),
                count=int(vv.sum()), design_weight=float(ww.sum()),
                fraction=float(np.sum(vv * ww) / ww.sum()) if ww.sum() else None)


def peak_times(data, P):
    m = data['m']
    t = np.argmax(np.where(m, P, -np.inf), 1)
    assert np.all(m[np.arange(len(t))[m.any(1)], t[m.any(1)]])
    return t


def timing_masks(data, crk, host, candidate=None):
    m = data['m']
    n = len(m)
    out = {}
    for name, values in [('true_peak', data['y']), ('crk_predicted_peak', crk),
                         ('host_predicted_peak', host)]:
        mask = np.zeros_like(m)
        mask[np.arange(n), peak_times(data, values)] = True
        out[name] = mask & m
    clock = np.zeros_like(m)
    clock[:, np.arange(0, 144, 12)] = True
    out['fixed_clock'] = clock & m
    out['observed_adjacent_hours'] = m & data['obs_full'][:, 71:215]
    if candidate is not None:
        mask = np.zeros_like(m)
        mask[np.arange(n), peak_times(data, candidate)] = True
        out['intervention_predicted_peak'] = mask & m
    return out


def allocate_write(values, key, ids, array, n):
    val = array.detach().cpu().numpy().astype('f4', copy=False)
    assert val.shape[0] == len(ids)
    assert not np.isinf(val).any(), key
    if key not in values:
        values[key] = np.full((n, *val.shape[1:]), np.nan, dtype='f4')
    values[key][ids] = val


def collect_diagnostics(out, model, ids, values, n, scale=1.):
    for key in SCALAR_KEYS:
        allocate_write(values, key, ids, out[key], n)
    for key in ('h1', 'a2'):
        allocate_write(values, key + '_norm', ids, out[key][:, 72:].norm(dim=-1), n)
    if not model.kernel:
        return
    k = model.kernel
    e = out['kernel_geography_features']
    fg, hg, anchor = k._geographic_terms(e)
    u = torch.cat((torch.asinh(out['h1'] / k.level_scale),
                   torch.asinh(out['kernel_departure'] / k.departure_scale)), -1)
    psi, raw = preactivation(u, fg, anchor, k.fusion_weather, k.head_weather, k.head_fusion)
    a = anchor[:, None, :32]
    write_raw = a + scale * (raw[..., :32] - a)
    deposit = out['kernel_deposit']
    expected = scaled_deposit(raw, anchor, scale)
    assert torch.allclose(deposit, expected, rtol=2e-6, atol=2e-7)
    controls = type(k)._controls(k, u, fg, hg, anchor)
    for name, diagnostic in [('tau', 'tau'), ('rho', 'rho'), ('eta', 'eta'), ('gain', 'mode_gain')]:
        assert torch.equal(controls[name], out['kernel_' + diagnostic]), name
    norm = deposit.norm(dim=-1)
    assert float(norm.max()) <= 1.000002
    anchor_tanh = torch.tanh(a).reshape(len(ids), 1, 4, 8)
    raw_tanh = torch.tanh(write_raw).reshape(len(ids), 216, 4, 8)
    anchor_sat = (anchor_tanh.abs() > SATURATION).float().mean(-1).expand(-1, 216, -1)
    raw_sat = (raw_tanh.abs() > SATURATION).float().mean(-1)
    anti = cosine(deposit, -anchor_tanh)
    joint = (norm >= NORM_THRESHOLD) & (anti >= COS_THRESHOLD)
    u_delta = u[:, 72:] - u[:, 71:215]
    d_delta = deposit[:, 72:] - deposit[:, 71:215]
    dcos = cosine(deposit[:, 72:], deposit[:, 71:215])
    entries = dict(deposit_norm=norm[:, 72:], anti_anchor_cosine=anti[:, 72:],
        anchor_saturated_fraction=anchor_sat[:, 72:], raw_saturated_fraction=raw_sat[:, 72:],
        raw_anchor_opposite_sign_fraction=((raw_tanh * anchor_tanh) < 0).float().mean(-1)[:, 72:],
        near_full_and_anti_anchor=joint[:, 72:].float(),
        near_full_anti_anchor_and_anchor_raw_saturated=(joint & (anchor_sat >= .9) & (raw_sat >= .9))[:, 72:].float(),
        temporal_deposit_cosine=dcos,
        temporal_direction_flip=torch.where(torch.isfinite(dcos), (dcos < 0).float(), torch.full_like(dcos, float('nan'))),
        deposit_delta_norm=d_delta.norm(dim=-1),
        coordinate_sign_flip_fraction=((deposit[:, 72:] * deposit[:, 71:215]) < 0).float().mean(-1),
        input_delta_norm=u_delta.norm(dim=-1), input_temporal_cosine=cosine(u[:, 72:], u[:, 71:215]),
        input_norm=u[:, 72:].norm(dim=-1),
        instantaneous_write_norm=((1. - out['kernel_rho'][..., None]) * deposit).norm(dim=-1)[:, 72:],
        state_norm=out['kernel_state'].norm(dim=-1)[:, 72:],
        response_norm=out['kernel_response'].norm(dim=-1)[:, 72:],
        effect_norm=out['kernel_effect'].norm(dim=-1)[:, 72:],
        tau=out['kernel_tau'][:, 72:], rho=out['kernel_rho'][:, 72:],
        mode_gain=out['kernel_mode_gain'][:, 72:])
    for key, value in entries.items():
        allocate_write(values, key, ids, value, n)
    # Keep selected-point Jacobians compact: union of the three peak times and
    # twelve clocks. The intervention own peak is included in its own run.
    return dict(u=u, psi=psi, raw=raw, anchor=anchor, u_delta=u_delta,
                d_delta=d_delta)


def collect_jacobians(model, context, ids, values, n, point_mask, scale=1.):
    k = model.kernel
    rows, times = np.nonzero(point_mask[ids])
    if not len(rows):
        return
    rr = torch.from_numpy(rows)
    tt = torch.from_numpy(times)
    raw = context['raw'][rr, tt + 72][:, None]
    psi = context['psi'][rr, tt + 72][:, None]
    anchor = context['anchor'][rr]
    jac = deposit_jacobian(raw, psi, anchor, k.fusion_weather,
                           k.head_weather, k.head_fusion, scale).squeeze(1)
    delta_u = context['u_delta'][rr, tt]
    unorm = delta_u.norm(dim=-1)
    delta_d = context['d_delta'][rr, tt].flatten(-2)
    jdu = torch.einsum('ndu,nu->nd', jac, delta_u)
    direction = jdu / unorm.clamp_min(EPS)[:, None]
    missing = unorm <= EPS
    def optional(v):
        return torch.where(missing, torch.full_like(v, float('nan')), v)
    entries = dict(jacobian_frobenius=jac.norm(dim=(-2, -1)),
        jacobian_direction_norm=optional(direction.norm(dim=-1)),
        jacobian_direction_vs_actual_delta_cosine=optional(cosine(jdu, delta_d)),
        jacobian_linearized_delta_norm=jdu.norm(dim=-1),
        actual_delta_norm=delta_d.norm(dim=-1),
        actual_delta_per_input_distance=optional(delta_d.norm(dim=-1) / unorm.clamp_min(EPS)),
        linearization_error_norm=(delta_d - jdu).norm(dim=-1))
    for key, val in entries.items():
        if key not in values:
            values[key] = np.full((n, 144), np.nan, dtype='f4')
        values[key][ids[rows], times] = val.cpu().numpy()


def run_diagnostics(F, data, model, stats, host, crk=None, scale=1., name='baseline'):
    n = len(data['y'])
    values = {}
    P = np.empty((n, 144), dtype='f4')
    for start in range(0, n, 64):
        ids = np.arange(start, min(start + 64, n))
        batch = make_batch(F, ids, stats)
        out = model(batch, diagnostics=True)
        P[ids] = out['P'].numpy()
        context = collect_diagnostics(out, model, ids, values, n, scale)
        if context is not None:
            # Only this batch's masks are needed and all model peak times use
            # the authoritative observed support, including the own peak.
            m = data['m'][ids]
            point = np.zeros_like(m)
            for pp in (data['y'][ids], host[ids], P[ids], *( () if crk is None else (crk[ids],))):
                t = np.argmax(np.where(m, pp, -np.inf), axis=1)
                point[np.arange(len(ids)), t] = True
            point[:, np.arange(0, 144, 12)] = True
            point &= m
            global_point = np.zeros_like(data['m'])
            global_point[ids] = point
            collect_jacobians(model, context, ids, values, n, global_point, scale)
        if start % 1024 == 0:
            print(json.dumps(dict(stage=name, rows_done=int(start + len(ids)), total=n)), flush=True)
    assert np.isfinite(P).all() and np.all((P >= 0) & (P <= 1))
    return P, values


def support(data, units, chosen):
    w = data['meta']['w'].astype('f8')
    counts = chosen.sum(1)
    active = units & (counts > 0)
    return dict(units=int(units.sum()), units_at_timing=int(active.sum()),
        observed_timing_points=int(chosen.sum()), weighted_timing_points=float(np.sum(w * counts)),
        design_unit_weight=float(w[active].sum()),
        systems=len(np.unique(data['meta']['system'][active])),
        counties=len(np.unique(data['meta']['county'][active])),
        merged_groups=len(np.unique(data['group'][active])))


def joint_distributions(values, chosen, weights):
    norm = values['deposit_norm'][chosen]
    cos = values['anti_anchor_cosine'][chosen]
    a = values['anchor_saturated_fraction'][chosen]
    r = values['raw_saturated_fraction'][chosen]
    ans = []
    for mode in range(4):
        good = np.isfinite(cos[:, mode])
        h = np.histogram2d(norm[good, mode], cos[good, mode],
                           bins=(NORM_BINS, COS_BINS), weights=weights[good])[0]
        den = weights.sum()
        lock = (norm[:, mode] >= .9) & (cos[:, mode] >= .95)
        # Missing cosines do not become zero; lock event is counted among valid
        # cosine support, while the histogram retains explicit missing mass.
        ans.append(dict(mode=mode,
            joint_norm_cosine_design_share=(h / den).tolist() if den else None,
            cosine_missing_count=int((~good).sum()),
            cosine_missing_design_share=float(weights[~good].sum() / den) if den else None,
            lock=fraction(lock, weights, good),
            lock_and_anchor_raw_saturated=fraction(lock & (a[:, mode] >= .9) & (r[:, mode] >= .9), weights, good),
            anchor_saturation_given_lock=dist(a[lock, mode], weights[lock]),
            raw_saturation_given_lock=dist(r[lock, mode], weights[lock])))
    return ans


def summaries(data, values, timings, splits, crk_alarm, *, regimes=True, host=False):
    masks = cohorts(data)
    w = data['meta']['w'].astype('f8')
    rows = []
    cohort_list = ('all', 'S', 'nonS', 'nonS_alarm')
    reg_list = ['all', *REGIMES] if regimes else ['all']
    for split, ids in splits.items():
        split_mask = np.zeros(len(w), bool)
        split_mask[ids] = True
        for cohort in cohort_list:
            cm = masks['nonS'] & crk_alarm if cohort == 'nonS_alarm' else masks[cohort]
            for regime in reg_list:
                units = split_mask & cm
                if regime != 'all':
                    units &= data['meta']['regime'] == regime
                for when, tm in timings.items():
                    chosen = units[:, None] & tm
                    if not chosen.any():
                        rows.append(dict(split=split, cohort=cohort, regime=regime, timing=when,
                                         support=support(data, units, chosen), values={}, joint=[]))
                        continue
                    weights = np.broadcast_to(w[:, None], data['m'].shape)[chosen]
                    summary = {}
                    for key, val in values.items():
                        # Adjacent-only quantities require adjacent mask even
                        # when the point itself is selected as a peak/clock.
                        is_temporal = (key.startswith(('temporal_', 'coordinate_sign_', 'input_delta_',
                                         'input_temporal_', 'deposit_delta_', 'jacobian_direction_',
                                         'jacobian_linearized_', 'actual_delta_', 'linearization_')))
                        sel = chosen & data['obs_full'][:, 71:215] if is_temporal else chosen
                        ww = np.broadcast_to(w[:, None], data['m'].shape)[sel]
                        v = val[sel]
                        summary[key] = ([dist(v[:, j], ww) for j in range(v.shape[1])]
                                        if v.ndim > 1 else dist(v, ww))
                    rows.append(dict(split=split, cohort=cohort, regime=regime, timing=when,
                        support=support(data, units, chosen), values=summary,
                        joint=[] if host else joint_distributions(values, chosen, weights)))
    return rows


def paired_scale_summary(data, baseline, intervention, ids):
    m, y = data['m'][ids], data['y'][ids]
    w = data['meta']['w'][ids].astype('f8')
    p, q = baseline[ids], intervention[ids]
    delta = q - p
    sse_delta = np.where(m, (q - y)**2 - (p - y)**2, 0).sum(1)
    peak_p = np.max(np.where(m, p, -np.inf), 1)
    peak_q = np.max(np.where(m, q, -np.inf), 1)
    peak_y = np.max(np.where(m, y, -np.inf), 1)
    valid = m.any(1)
    w = w[valid]
    sse_delta = sse_delta[valid]
    dpeak = (peak_q - peak_p)[valid]
    alignment = np.where(m, 2 * (y - p) * delta, 0).sum(1)[valid]
    energy = np.where(m, delta * delta, 0).sum(1)[valid]
    assert np.allclose(-sse_delta, alignment - energy, rtol=1e-5, atol=1e-6)
    d = delta[valid]
    mm = m[valid]
    means = np.where(mm, d, 0).sum(1) / mm.sum(1)
    return dict(n=len(w), sse_change=dist(sse_delta, w), mean_prediction_change=dist(means, w),
        observed_peak_change=dist(dpeak, w),
        observed_peak_abs_error_change=dist((abs(peak_q - peak_y) - abs(peak_p - peak_y))[valid], w),
        trajectory_improved=fraction(sse_delta < 0, w), trajectory_worsened=fraction(sse_delta > 0, w),
        prediction_mean_increased=fraction(means > 0, w), prediction_mean_decreased=fraction(means < 0, w),
        observed_peak_increased=fraction(dpeak > 0, w), observed_peak_decreased=fraction(dpeak < 0, w),
        max_abs_prediction_change=float(np.max(abs(d[mm]))) if mm.any() else None,
        alignment_total=float(np.sum(w * alignment)), modification_energy_total=float(np.sum(w * energy)))


def perturbation_scores(data, baseline, intervention, splits, score):
    masks = cohorts(data)
    output = {}
    for split, ids in splits.items():
        output[split] = {}
        for cohort in ('all', 'S', 'nonS', 'J'):
            selected = ids[masks[cohort][ids]]
            output[split][cohort] = dict(score=score(data, intervention, selected),
                paired_change=paired_scale_summary(data, baseline, intervention, selected))
        for regime in REGIMES:
            selected = ids[data['meta']['regime'][ids] == regime]
            output[split]['regime_' + regime] = dict(score=score(data, intervention, selected),
                paired_change=paired_scale_summary(data, baseline, intervention, selected))
        selected = ids[masks['nonS'][ids]]
        m = data['m'][selected]
        peak0 = np.max(np.where(m, baseline[selected], -np.inf), 1)
        peak1 = np.max(np.where(m, intervention[selected], -np.inf), 1)
        weights = data['meta']['w'][selected].astype('f8')
        output[split]['false_peak_transitions'] = dict(
            severe_threshold=.1, inclusive=True,
            baseline=fraction(peak0 >= .1, weights), intervention=fraction(peak1 >= .1, weights),
            resolved=fraction((peak0 >= .1) & (peak1 < .1), weights),
            introduced=fraction((peak0 < .1) & (peak1 >= .1), weights))
    return output


def save_npz_new(path, **arrays):
    with path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    return sha(path)


def verify_provenance(common, receipt):
    assert sha(common.FEATURES) == receipt['data_sha256']
    assert sha(common.__file__) == receipt['common_source_sha256']
    for name, digest in receipt['frozen_source_sha256'].items():
        assert sha(ROOT / name) == digest, name
    for label, folds in receipt['frozen_artifact_sha256'].items():
        for fold, files in folds.items():
            for name, digest in files.items():
                assert sha(common.RUNS / label / f'fold{int(fold):02d}' / name) == digest


def main():
    import d07_common as common
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    os.nice(max(0, 15 - os.getpriority(os.PRIO_PROCESS, 0)))
    result = HERE / 'results/v1/d07_controllers.json'
    assert not result.exists()
    started = time.time()
    data, F = common.load_data(load_inputs=True)
    n = len(data['y'])
    assert n == 8457
    fit = np.sort(np.array(data['split']['event']['1']['dev']))
    outer = data['expected'][1]
    assert len(fit) == 6350 and len(outer) == 2107
    splits = {'FIT': fit, 'OUTER': outer}
    provenance_before = common.provenance()
    local_hashes = {}
    host_model, host_stats, host_prov = common.load_model(F, data, 'v1_host_s0', fold=1)
    host_state = snapshot(host_model)
    # Host has no controller Jacobians; empty reference P is unused.
    with torch.no_grad():
        host, host_values = run_diagnostics(F, data, host_model, host_stats,
                                            np.empty((n, 144)), name='host')
    assert_frozen(host_model, host_state)
    host_digest = state_digest(host_state)
    del host_model, host_state
    model, stats, model_prov = common.load_model(F, data, 'v1_crk_s0', fold=1)
    frozen_state = snapshot(model)
    before_digest = state_digest(frozen_state)
    with torch.no_grad():
        closed = common.evaluate(model, F, stats, np.arange(n), exit_open=False)
    assert closed.shape == (n, 144)
    local_hashes['fold1_closed.npz'] = save_npz_new(common.OUT / 'fold1_closed.npz', idx=np.arange(n), P_closed=closed, P=closed)
    print(json.dumps(dict(stage='closed_export_ready', path=str(common.OUT / 'fold1_closed.npz'),
                          sha256=local_hashes['fold1_closed.npz'])), flush=True)
    with torch.no_grad():
        crk, values = run_diagnostics(F, data, model, stats, host, name='crk_baseline')
    with np.load(common.RUNS / 'v1_crk_s0/fold01/outer.npz') as z:
        replay = float(np.max(abs(crk[z['idx']] - z['P'])))
    assert replay < 2e-6
    with np.load(common.RUNS / 'v1_host_s0/fold01/outer.npz') as z:
        host_replay = float(np.max(abs(host[z['idx']] - z['P'])))
    assert host_replay < 2e-6
    assert_frozen(model, frozen_state)
    alarms = np.max(np.where(data['m'], crk, -np.inf), 1) >= .1
    timings = timing_masks(data, crk, host)
    baseline_rows = summaries(data, values, timings, splits, alarms)
    host_rows = summaries(data, host_values, timings, splits, alarms, regimes=False, host=True)
    local_hashes['controller_baseline_scalars.npz'] = save_npz_new(common.OUT / 'controller_baseline_scalars.npz',
        idx=np.arange(n), **values)
    local_hashes['host_fold1_scalars.npz'] = save_npz_new(common.OUT / 'host_fold1_scalars.npz', idx=np.arange(n), **host_values)
    del values, host_values
    baseline_scores = {label: {split: {cohort: common.score(data, p, ids[cohorts(data)[cohort][ids]])
        for cohort in ('all', 'S', 'nonS', 'J')} for split, ids in splits.items()}
        for label, p in [('host', host), ('crk_open', crk), ('crk_closed', closed)]}
    perturbations = {}
    for scale in SCALES:
        with write_scale(model.kernel, scale), torch.no_grad():
            p, sv = run_diagnostics(F, data, model, stats, host, crk=crk,
                                   scale=scale, name='write_scale_' + str(scale))
        assert_frozen(model, frozen_state)
        timing_scale = timing_masks(data, crk, host, p)
        per_unit_sse_delta = np.where(data['m'], (p - data['y'])**2 - (crk - data['y'])**2, 0).sum(1)
        peak0 = np.max(np.where(data['m'], crk, -np.inf), 1)
        peak1 = np.max(np.where(data['m'], p, -np.inf), 1)
        name = 'controller_scale_' + str(scale).replace('.', 'p') + '.npz'
        local_hashes[name] = save_npz_new(common.OUT / name, idx=np.arange(n), P=p,
            per_unit_sse_delta=per_unit_sse_delta, per_unit_peak_delta=peak1-peak0, **{k: v for k, v in sv.items() if k != 'P'})
        perturbations[str(scale)] = dict(scores=perturbation_scores(data, crk, p, splits, common.score),
            working_points=summaries(data, sv, timing_scale, splits, alarms, regimes=False),
            invariant_original_controls=True, frozen_state_restored=True)
        del p, sv
    assert_frozen(model, frozen_state)
    verify_provenance(common, provenance_before)
    unit_metadata = dict(input_dimension=64, output_dimension=32, modes=4, mode_dimension=8,
        geography_original_dimension=40, geography_basis_dimension=32, geography_features_dimension=72,
        controller_input='asinh(h1/level_scale), asinh((h1-prefix_reference(h1))/departure_scale)',
        input_units='Dimensionless calibrated hidden coordinates; not raw weather.',
        output_units='Dimensionless bounded deposit; each eight-dimensional mode lies in the unit ball.',
        jacobian='sech2(R_d)/(2sqrt8) * (head_weather_d + head_fusion_d diag(sech2(psi)) fusion_weather)',
        direction='J delta_u / norm(delta_u), actual consecutive observed-hour 64D hidden input direction',
        saturation_absolute_tanh_threshold=SATURATION, cosine_near_zero_threshold=EPS,
        lock=dict(mode_deposit_norm_at_least=.9, anti_anchor_cosine_at_least=.95),
        fully_saturated_joint='lock and anchor/raw each have >=.9 coordinates with abs(tanh)> .99 (all 8 coordinates)',
        geometric_caution='Near-full deposits in the anchored coordinate box are necessarily directed against saturated anchors; this joint operating regime is not independent evidence of weather insensitivity or a cause of errors.',
        histogram_norm_bins=NORM_BINS.tolist(), histogram_cosine_bins=COS_BINS.tolist(),
        timings='Peak times on observed support; fixed clocks 72,84,...,204; adjacent hours require t and t-1 observed.',
        jacobian_support='Three baseline peak times and fixed clocks; intervention own peak added. No Jacobians at remaining adjacent hours.',
        interventions='Only write weather preactivation difference scaled .95 or 1.05; parameters, host, retention, rotation, gain, recurrence equation and readout fixed.',
        interpretation='Frozen internal sensitivity, not new weather, training, or a deployable gain; no further scales selected.')
    write_new(result, dict(scope_commit='8f8ddef', phase='B', fold=1,
        source_sha256=sha(__file__), provenance=provenance_before,
        model_provenance={'host': host_prov, 'crk': model_prov},
        validation=dict(host_replay_max_abs=host_replay, crk_replay_max_abs=replay,
            all_frozen_tensors_and_buffers_unchanged=True, frozen_source_data_hashes_unchanged=True,
            host_state_sha256=host_digest, crk_state_sha256=before_digest,
            crk_state_after_sha256=state_digest(model.state_dict()),
            deposit_unit_ball_verified=True, original_tau_rho_eta_gain_exactly_unchanged=True),
        definitions=unit_metadata, baseline_scores=baseline_scores,
        baseline_working_points=baseline_rows, host_working_points=host_rows,
        perturbations=perturbations, local_artifact_hashes=local_hashes,
        wall_seconds=time.time() - started))
    print(json.dumps(dict(stage='complete', result=str(result), sha256=sha(result),
                          wall_seconds=time.time() - started)), flush=True)


if __name__ == '__main__':
    main()
