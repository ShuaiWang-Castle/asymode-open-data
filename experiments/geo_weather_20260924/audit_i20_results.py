"""Read-only final I20 audit; independent score arithmetic, no refit or inference.

The registered outcome loader supplies validated public-D rows and merged labels.
This audit independently assembles exports, resamples clusters, and computes scores.
It writes one new compact receipt only after every check passes.
"""
from __future__ import annotations

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '2'

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from evaluate_cr_tail import load_outcomes

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS = ROOT / 'runs/geo_weather_20260924'
RUN = RUNS / 'v1_crk_s0'
RESULTS = HERE / 'results/v1'
REGIMES = ['tropical', 'winter', 'synoptic_wind', 'convective', 'heavy_rain']
CHECKS = 0


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def check(condition, name):
    global CHECKS
    if not condition:
        raise AssertionError(name)
    CHECKS += 1


def near(a, b, name):
    check(np.allclose(a, b, rtol=1e-9, atol=1e-12), name)


def finite_tree(x, name):
    if isinstance(x, dict):
        for k, v in x.items():
            finite_tree(v, name + '/' + str(k))
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            finite_tree(v, name + '/' + str(i))
    elif isinstance(x, float):
        check(np.isfinite(x), name)
    elif isinstance(x, torch.Tensor):
        check(bool(torch.isfinite(x).all()), name)
    elif isinstance(x, np.ndarray) and x.dtype.kind in 'fci':
        check(np.isfinite(x).all(), name)


def plan(meta, labels):
    levels = sorted(set(labels))
    members = [np.flatnonzero(labels == level) for level in levels]
    strata = [int(np.argmax([meta['w'][idx][meta['regime'][idx] == r].sum()
                           for r in REGIMES])) for idx in members]
    rng = np.random.default_rng(20260928)
    count = np.zeros((1999, len(levels)), dtype=np.int32)
    for r in range(5):
        js = np.flatnonzero(np.asarray(strata) == r)
        if len(js):
            count[:, js] = rng.multinomial(len(js), np.full(len(js), 1 / len(js)), 1999)
    return members, count


def verify_interval(values, good, support, stored):
    valid = good & np.isfinite(values)
    check(int(valid.sum()) == stored['valid_draws'], 'valid bootstrap count')
    check(int((~valid).sum()) == stored['invalid_draws'], 'invalid bootstrap count')
    check(support == stored['supporting_clusters'], 'cluster support')
    if support < 2 or not valid.any():
        check(stored['ci95'] is None, 'unsupported interval')
    else:
        near(np.quantile(values[valid], [.025, .975]), stored['ci95'], 'CI')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=RESULTS / 'i20_final_audit_s0.json')
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if os.getpriority(os.PRIO_PROCESS, 0) < 15:
        os.nice(15 - os.getpriority(os.PRIO_PROCESS, 0))
    torch.set_num_threads(2)
    manifest, amendment = read(RUN / 'SCREEN_RUN.json'), read(RUN / 'COMPUTE_AMENDMENT.json')
    check(sha(RUN / 'SCREEN_RUN.json') == amendment['original_manifest_sha256'], 'manifest frozen')
    for path, expected in (manifest['source_sha256'] | amendment['controller_sha256']).items():
        check(sha(ROOT / path) == expected, 'source hash ' + path)
    feat = manifest['feature_file']; path = ROOT / feat['path']
    check(path.stat().st_size == feat['size'] and path.stat().st_mtime_ns == feat['mtime_ns'], 'data stat')
    check(sha(path) == feat['sha256'], 'data sha')
    status = read(RUN / 'RUN_STATUS.json')
    check(status['status'] == 'completed' and status['evaluated'] is True
          and status['done'] == [1, 2, 3, 4, 5] and not status['active'] and not status['pending'], 'complete status')
    ev = read(RUN / 'EVALUATION_DONE.json')
    check(ev['source_sha256'] == manifest['source_sha256'] and ev['primary_cohort'] == 'S', 'evaluation source')
    for path, expected in ev['artifact_sha256'].items():
        check(sha(RESULTS / path) == expected, 'evaluation hash ' + path)
        finite_tree(read(RESULTS / path), path)
    d = load_outcomes(); y, m, meta = d['y'], d['m'], d['meta']; n = len(y)
    host_ref = read(RUN / 'HOST_REFERENCE.json'); predictions = {}; folds = []; exports = {}
    for label, arm in [('v1_crk_s0', 'CRK+Cin'), ('v1_host_s0', 'W+Cin')]:
        p = np.empty_like(y); seen = np.zeros(n, dtype=int); exports[label] = {}
        for k in range(1, 6):
            folder = RUNS / label / f'fold{k:02d}'; done = read(folder / 'DONE.json')
            expected = dict(label=label, arm=arm, data='v1D', design='event', design_weights=True,
                            fold=k, seed=0, steps=900)
            check(all(type(done[key]) is type(v) and done[key] == v for key, v in expected.items()), 'DONE metadata')
            check(all(done.get(key) is None for key in ('phi', 'keep', 'ctx_geo', 'xu_phi')), 'no hazard')
            check(all(done.get(key, False) is False for key in ('train_mask', 'mask_placebo')), 'no mask change')
            finite_tree(done, label + '/DONE')
            hashes = {f: sha(folder / f) for f in ('outer.npz', 'DONE.json', 'final.pt')}
            exports[label][str(k)] = hashes
            if label == 'v1_crk_s0':
                receipt = read(folder / 'RUNNER_RECEIPT.json')
                check(receipt['metadata'] == expected and receipt['artifact_sha256'] == hashes
                      and receipt['source_sha256'] == manifest['source_sha256'], 'candidate receipt')
                check(done['microbatch'] == 512 and done['kernel_trace'][-1]['step'] == 900, 'update budget')
                folds.append(dict(fold=k, heldout=len(d['expected'][k]), seconds=done['seconds'],
                                  fit_loss=done['fit_loss'], final_trace=done['kernel_trace'][-1]))
            else:
                ref = host_ref[str(k)]
                check(ref['metadata'] == expected and ref['outer_sha256'] == hashes['outer.npz']
                      and ref['done_sha256'] == hashes['DONE.json'], 'host reference')
            finite_tree(torch.load(folder / 'final.pt', map_location='cpu', weights_only=False), label + '/checkpoint')
            with np.load(folder / 'outer.npz', allow_pickle=False) as z:
                idx = z['idx']
                check(idx.dtype.kind in 'iu' and np.array_equal(np.sort(idx), np.sort(d['expected'][k])), 'heldout rows')
                for key in ('P', 'u', 'r', 'raw_logit', 'P_closed', 'raw_logit_closed'):
                    # Older host exports need not carry closed-kernel diagnostics.
                    if label == 'v1_host_s0' and key not in z:
                        continue
                    a = z[key]
                    check(a.shape == (len(idx), 144) and np.isfinite(a).all(), 'export ' + key)
                    if key in ('P', 'P_closed'):
                        check(((a >= 0) & (a <= 1)).all(), 'probability range')
                p[idx] = z['P']; np.add.at(seen, idx, 1)
        check((seen == 1).all(), label + ' exactly once')
        predictions[label] = p
    P, H = predictions['v1_crk_s0'], predictions['v1_host_s0']
    pred = [P, H, np.zeros_like(y)]
    N = m.sum(1); peak = np.max(np.where(m, y, -np.inf), axis=1)
    pair = d['obs_full'][:, 72:] & d['obs_full'][:, 71:215]
    jump = np.max(np.where(pair, np.diff(d['y_full'][:, 71:], axis=1), -np.inf), axis=1)
    S = (N > 0) & (peak >= .1); J = pair.any(1) & (jump >= .01)
    masks = dict(all=N > 0, S=S, J=J, S_and_J=S & J, any_positive=(N > 0) & (peak > 0))
    plans = {name: plan(meta, labels) for name, labels in [('merged_event', d['group']), ('original_family', meta['family'])]}
    tail = read(RESULTS / 'i20_cr_tail_s0.json')

    def score(selected, w, stored, col=None):
        mask = m if col is None else m[:, [col]]
        target = y if col is None else y[:, [col]]
        nn = mask.sum(1); sw = w * (selected & (nn > 0)); den = sw @ nn
        errors = [(p if col is None else p[:, [col]]) - target for p in pred]
        se = np.array([np.where(mask, e * e, 0).sum(1) for e in errors]).T
        ae = np.array([np.where(mask, abs(e), 0).sum(1) for e in errors]).T
        mse = sw @ se / den; rmse = np.sqrt(mse)
        for j, name in enumerate(('candidate', 'host', 'zero')):
            near([mse[j], rmse[j], (sw @ ae)[j] / den],
                 [stored['models'][name][v] for v in ('mse', 'rmse', 'mae')], 'MSE/RMSE/MAE')
        near(1 - rmse[0] / rmse[1], stored['rmse_improvement_fraction'], 'RMSE improvement')
        check(int((sw > 0).sum()) == stored['support']['county_events'], 'cohort support')
        for name, iv in stored['intervals'].items():
            members, counts = plans[name]
            totals = np.array([(sw[idx, None] * se[idx, :2]).sum(0) for idx in members])
            mass = np.array([(sw[idx] * nn[idx]).sum() for idx in members])
            bs = counts @ totals; good = ((counts @ mass) > 0) & (bs[:, 1] > 0)
            v = np.full(len(counts), np.nan); v[good] = 1 - np.sqrt(bs[good, 0] / bs[good, 1])
            verify_interval(v, good, int((mass > 0).sum()), iv)

    for weight, content in tail['weights'].items():
        w = np.ones(n) if weight == 'unweighted' else meta[weight].astype(float)
        for name, selected in masks.items():
            block = content['cohorts'][name]
            score(selected, w, block['full_window'])
            for r in REGIMES:
                score(selected & (meta['regime'] == r), w, block['by_regime'][r])
            for k in range(1, 6):
                score(selected & (d['fold'] == k), w, block['by_fold'][str(k)])
            for h in (1, 6, 24, 48):
                score(selected, w, block['horizon_snapshots'][str(h)], h - 1)
        for name, threshold in [('all_observed_zero', .001), ('nonS', .1)]:
            selected = (N > 0) & (peak == 0 if name == 'all_observed_zero' else peak < .1)
            peaks = [np.max(np.where(m, p, -np.inf), axis=1) for p in (P, H)]
            alarms = [v > threshold if name == 'all_observed_zero' else v >= threshold for v in peaks]
            base = content['false_alarms'][name]
            blocks = [(selected, base), (selected & m.all(1), base['complete_144h'])]
            blocks += [(selected & (meta['regime'] == r), base['by_regime'][r]) for r in REGIMES]
            for selected_part, block in blocks:
                sw = w * selected_part
                rates = np.array([sw @ a / sw.sum() for a in alarms])
                near(rates, [block['candidate_rate'], block['host_rate']], 'alarm rates')
                near(rates[0] - rates[1], block['difference'], 'alarm difference')
                check([int((a & selected_part).sum()) for a in alarms]
                      == [block['candidate_count'], block['host_count']], 'alarm counts')
                for name2, (members, counts) in plans.items():
                    mass = np.array([sw[idx].sum() for idx in members])
                    delta = alarms[0].astype(float) - alarms[1].astype(float)
                    diff = np.array([(sw[idx] * delta[idx]).sum() for idx in members])
                    bm = counts @ mass; good = bm > 0; v = np.full(len(counts), np.nan)
                    v[good] = (counts @ diff)[good] / bm[good]
                    verify_interval(v, good, int((mass > 0).sum()), block['intervals'][name2])
    # Independent legacy point scores. Their different bootstrap protocol remains
    # explicitly identified; the new primary and every tail interval were replayed above.
    se = np.array([np.where(m, (p - y) ** 2, 0).sum(1) for p in pred]).T
    for which, headline in [('headline', REGIMES[:2]), ('all5', REGIMES)]:
        legacy = read(RESULTS / f'screen_i20_s0_{which}_vs_host.json')
        for weight in ('w', 'w_raw'):
            w = meta[weight].astype(float); gains = {}
            for r in REGIMES:
                sw = w * (meta['regime'] == r); mse = sw @ se / (sw @ N)
                near(mse, [legacy[weight]['per_regime'][r][v] for v in ('arm', 'host', 'zero')], 'legacy MSE')
                gains[r] = mse[0] / mse[1] - 1
            near(np.mean([gains[r] for r in headline]), legacy[weight]['balanced_gain'], 'legacy balanced')
            mse = w @ se / (w @ N)
            near(mse[0] / mse[1] - 1, legacy[weight]['pooled_gain'], 'legacy pooled')
    check(tail['primary_target'] == read(RESULTS / 'screen_i20_s0_verdict.json')['primary_target'], 'verdict agreement')
    check(not any(tail['primary_target'][k] for k in ('point_target_met', 'ci_supports_positive', 'ci_supports_at_least_10pct')), 'failed target')
    receipt = dict(status='passed', audit_utc=datetime.now(timezone.utc).isoformat(), checks=CHECKS,
                   audit_source_sha256=sha(Path(__file__)), registration=manifest['git_commit'],
                   manifest_sha256=sha(RUN / 'SCREEN_RUN.json'), data_sha256=feat['sha256'],
                   source_sha256=manifest['source_sha256'] | amendment['controller_sha256'],
                   exports=exports, result_sha256=ev['artifact_sha256'], folds=folds,
                   units=n, exactly_once_both_models=True, checkpoint_tensors_finite=True,
                   numeric_scope='Independent all tail points/intervals and legacy points; registered outcome/group loader reused. Legacy intervals hash-verified, not independently resampled.',
                   primary_target=tail['primary_target'])
    with args.out.open('x') as stream:
        json.dump(receipt, stream, indent=1, allow_nan=False); stream.write('\n')
    print(json.dumps(dict(status='passed', checks=CHECKS, units=n, primary_target=tail['primary_target'])))


if __name__ == '__main__':
    main()
