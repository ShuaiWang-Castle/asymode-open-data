"""Independent D09 terminal arithmetic audit; never import training/scoring code.

Nothing is opened without --execute. --self-check uses artificial arrays only.
Six authentic DONE receipts and all file hashes must pass before endpoint NPZs
are opened. This script never deserializes the original D NPZ or checkpoints,
performs a forward pass, or opens original OUTER1 outcomes. Register this file
separately, then pass both the frozen experiment and audit registration commits.
"""
from __future__ import annotations

import os
for _thread in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_thread] = '2'

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
E = ROOT / 'experiments/geo_weather_20260924'
EXPERIMENT_COMMIT = 'eb91994fd124bc7178e7edc0b175c84113406b68'
DATA_SHA = 'f043bb39e8abd48183670e2acc3c0cecebd7107ea9be0e28771912cfee358c48'
ARMS, FOLDS = ('host', 'crk', 'new'), (2, 3)
REGIMES = ('tropical', 'winter', 'synoptic_wind', 'convective', 'heavy_rain')
COMPS = {'new_vs_host': ('host', 'new'), 'crk_vs_host': ('host', 'crk'),
         'new_vs_crk': ('crk', 'new')}
META = ('idx', 'y', 'm', 'y0', 'w', 'regime', 'fips', 'system', 'family', 'origin',
        'merged_group', 'original_fold', 'panel', 'pi', 'origin_observed')
DRAW_COUNT, DRAW_SEED = 1999, 20260929
ATOL, RTOL = 2e-10, 2e-9


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def local(path):
    path = Path(path)
    path = (path if path.is_absolute() else ROOT / path).resolve()
    path.relative_to(ROOT)
    return path


def rel(path):
    return str(local(path).relative_to(ROOT))


def read_json(path):
    return json.loads(Path(path).read_text())


def git_hash(commit, path):
    content = subprocess.check_output(['git', 'show', commit + ':' + rel(path)],
                                      cwd=ROOT, stderr=subprocess.PIPE)
    return hashlib.sha256(content).hexdigest()


class Audit:
    def __init__(self):
        self.count = 0
        self.last = 'not_started'
        self.maximum_absolute_error = 0.
        self.maximum_relative_error = 0.

    def require(self, condition, label):
        self.last = label
        if not bool(condition):
            raise AssertionError(label)
        self.count += 1

    def equal(self, actual, expected, label):
        self.last = label
        if actual is None or expected is None:
            self.require(actual is None and expected is None, label)
        elif isinstance(expected, (bool, str, int, np.bool_, np.integer)):
            self.require(actual == expected, label)
        else:
            a, b = np.asarray(actual, float), np.asarray(expected, float)
            self.require(a.shape == b.shape and np.isfinite(a).all() and
                         np.isfinite(b).all(), label + '/shape_finite')
            error = np.abs(a - b)
            if error.size:
                self.maximum_absolute_error = max(self.maximum_absolute_error, float(error.max()))
                self.maximum_relative_error = max(self.maximum_relative_error,
                    float(np.max(error / np.maximum(np.abs(b), ATOL))))
            self.require(np.allclose(a, b, atol=ATOL, rtol=RTOL), label)

    def array_equal(self, a, b, label):
        numeric = np.issubdtype(np.asarray(a).dtype, np.number)
        self.require(np.array_equal(a, b, equal_nan=True) if numeric else
                     np.array_equal(a, b), label)


def med(values, weights):
    """Independent cumulative design-mass knot interpolation, D06/D07A rule."""
    good = [(float(x), float(w)) for x, w in zip(values, weights)
            if math.isfinite(float(x)) and math.isfinite(float(w)) and w > 0]
    if not good:
        return None
    ordered = sorted(good, key=lambda pair: pair[0])
    masses, values, total = [], [], 0.
    for x, w in ordered:
        total += w
        masses.append(total)
        values.append(x)
    return float(np.interp(.5 * total, masses, values))


def unit_arithmetic(data, predictions):
    """Loop over observed hours; no scoring helper or vectorized risk reuse."""
    n = len(data['idx'])
    result = {name: dict(sse=np.zeros(n), peak=np.full(n, np.nan),
                        peak_ratio=np.full(n, np.nan)) for name in ARMS}
    hours = np.zeros(n, int)
    true_peak = np.full(n, np.nan)
    max_rise = np.full(n, np.nan)
    for i in range(n):
        seen = np.flatnonzero(data['m'][i])
        hours[i] = len(seen)
        if not len(seen):
            continue
        truth = data['y'][i, seen].astype(float)
        true_peak[i] = float(max(truth))
        increments = []
        for t in seen:
            if t == 0:
                increments.append(float(data['y'][i, 0]) - float(data['y0'][i]))
            elif data['m'][i, t - 1]:
                increments.append(float(data['y'][i, t]) - float(data['y'][i, t - 1]))
        if increments:
            max_rise[i] = max(increments)
        for name in ARMS:
            p = predictions[name][i, seen].astype(float)
            result[name]['sse'][i] = float(data['w'][i]) * float(np.dot(p - truth, p - truth))
            result[name]['peak'][i] = float(max(p))
            if true_peak[i] > 0:
                result[name]['peak_ratio'][i] = result[name]['peak'][i] / true_peak[i]
    observed = hours > 0
    masks = dict(all=observed, S=observed & (true_peak >= .1),
                 nonS=observed & ~(true_peak >= .1),
                 any_positive=observed & (true_peak > 0),
                 J=observed & (max_rise >= .01))
    return result, hours, masks


def metrics(data, units, hours, selected):
    ids = np.flatnonzero(selected)
    mass = float(sum(float(data['w'][i]) * int(hours[i]) for i in ids))
    values = {}
    for name in ARMS:
        sse = float(sum(float(units[name]['sse'][i]) for i in ids))
        mse = sse / mass if mass > 0 else None
        values[name] = dict(design_SSE=sse, design_MSE=mse,
            design_RMSE=math.sqrt(mse) if mse is not None else None,
            peak_ratio_median=med(units[name]['peak_ratio'][ids], data['w'][ids]))
    return dict(units=len(ids), observed_hours=int(hours[ids].sum()),
                merged_groups=len(set(data['merged_group'][ids].astype(str))),
                design_hour_mass=mass, models=values)


def reduction(base, candidate):
    return 1. - math.sqrt(candidate / base) if base > 0 else None


def gains(data, predictions, units, selected, base, candidate):
    ids = np.flatnonzero(selected)
    if not len(ids):
        return dict(units=0, alignment=0., modification_energy=0., net_gain=0.,
                    positive_gain=0., negative_loss=0., positive_units=0,
                    negative_units=0, zero_units=0, positive_design_share=None,
                    top_10pct_all_cohort_units=0, top_10pct_share_of_positive_gain=None)
    gs, aligns, energies = [], [], []
    for i in ids:
        valid = data['m'][i]
        delta = predictions[candidate][i, valid].astype(float) - predictions[base][i, valid].astype(float)
        residual = data['y'][i, valid].astype(float) - predictions[base][i, valid].astype(float)
        aligns.append(float(data['w'][i]) * 2 * float(residual @ delta))
        energies.append(float(data['w'][i]) * float(delta @ delta))
        gs.append(float(units[base]['sse'][i] - units[candidate]['sse'][i]))
    gs = np.asarray(gs)
    positive = gs > 0
    positive_total = float(gs[positive].sum())
    top = math.ceil(len(ids) / 10)
    design_weights = np.asarray(data['w'][ids], dtype=np.float64)
    return dict(units=len(ids), alignment=float(sum(aligns)), modification_energy=float(sum(energies)),
        net_gain=float(gs.sum()), positive_gain=positive_total,
        negative_loss=float(-gs[gs < 0].sum()), positive_units=int(positive.sum()),
        negative_units=int((gs < 0).sum()), zero_units=int((gs == 0).sum()),
        positive_design_share=float(design_weights[positive].sum() / design_weights.sum()),
        top_10pct_all_cohort_units=top,
        top_10pct_share_of_positive_gain=float(sum(sorted(gs[positive], reverse=True)[:top]) / positive_total)
        if positive_total else None)


def alarms(data, units, selected, base, candidate):
    ids = np.flatnonzero(selected)
    if not len(ids):
        return dict(units=0, threshold=.1, base_count=0, candidate_count=0,
                    new_count=0, removed_count=0, retained_count=0,
                    base_design_rate=None, candidate_design_rate=None,
                    new_design_rate=None, removed_design_rate=None)
    base_set = {int(i) for i in ids if units[base]['peak'][i] >= .1}
    cand_set = {int(i) for i in ids if units[candidate]['peak'][i] >= .1}
    denominator = float(sum(float(data['w'][i]) for i in ids))
    rate = lambda group: float(sum(float(data['w'][i]) for i in sorted(group)) / denominator)
    return dict(units=len(ids), threshold=.1, base_count=len(base_set), candidate_count=len(cand_set),
        new_count=len(cand_set - base_set), removed_count=len(base_set - cand_set),
        retained_count=len(cand_set & base_set), base_design_rate=rate(base_set),
        candidate_design_rate=rate(cand_set), new_design_rate=rate(cand_set - base_set),
        removed_design_rate=rate(base_set - cand_set))


def cluster_plan(data, field, draws=DRAW_COUNT):
    labels = data[field].astype(str)
    levels = sorted(set(labels))
    index = {label: i for i, label in enumerate(levels)}
    code = np.asarray([index[label] for label in labels], int)
    masses = np.zeros((len(levels), 5), float)
    for i, label in enumerate(labels):
        masses[index[label], REGIMES.index(str(data['regime'][i]))] += float(data['w'][i])
    strata = np.argmax(masses, axis=1)
    rng = np.random.default_rng(DRAW_SEED)
    counts = np.zeros((draws, len(levels)), np.int32)
    for j in range(5):
        columns = [i for i in range(len(levels)) if strata[i] == j]
        if columns:
            counts[:, columns] = rng.multinomial(len(columns), np.ones(len(columns)) / len(columns), size=draws)
    summary = dict(clusters=len(levels),
        stratum_cluster_counts={r: int(sum(strata == j)) for j, r in enumerate(REGIMES)},
        singleton_strata=[r for j, r in enumerate(REGIMES) if sum(strata == j) == 1])
    digest = hashlib.sha256(counts.astype('<i4', copy=False).tobytes(order='C')).hexdigest()
    return dict(code=code, levels=levels, counts=counts, summary=summary, counts_sha256=digest)


def interval(units, selected, plan, base, candidate):
    k = len(plan['levels'])
    b, c = np.zeros(k), np.zeros(k)
    for i in np.flatnonzero(selected):
        b[plan['code'][i]] += units[base]['sse'][i]
        c[plan['code'][i]] += units[candidate]['sse'][i]
    bs, cs = plan['counts'] @ b, plan['counts'] @ c
    valid = (bs > 0) & np.isfinite(bs) & np.isfinite(cs) & (cs >= 0)
    samples = [1. - math.sqrt(float(y / x)) for x, y in zip(bs[valid], cs[valid])]
    empirical = np.percentile(samples, [2.5, 97.5]).tolist() if samples else None
    active = (b > 0) | (c > 0)
    vary = any(np.unique(plan['counts'][:, j]).size > 1 for j in np.flatnonzero(active))
    support = len(set(plan['code'][selected]))
    supported = support >= 2 and bool(samples) and vary
    ci = empirical if supported else None
    status = ('insufficient_support' if support < 2 else 'no_valid_draws' if not samples
              else 'no_effective_cluster_count_variation' if not vary else 'supported')
    return dict(supporting_clusters=support, draws=len(bs), valid_draws=int(valid.sum()),
        invalid_draws=int((~valid).sum()), interval_supported=supported,
        effective_risk_contribution_clusters=int(active.sum()), resample_counts_vary=vary,
        interval_status=status, empirical_relative_RMSE_reduction_percentiles95=empirical,
        relative_RMSE_reduction_ci95=ci, ci_excludes_zero_in_beneficial_direction=bool(ci and ci[0] > 0),
        ci_lower_bound_at_least_final_10pct=bool(ci and ci[0] >= .1))


def verify_files(args, audit):
    """No label/prediction NPZ opening in this phase."""
    audit.require(args.scope_commit == EXPERIMENT_COMMIT, 'fixed_experiment_registration')
    audit.require(local(args.features) == ROOT/'data/interim/panel_v1/features_v1D.npz', 'public_D_file_whitelist')
    audit.require(local(args.splits) == E/'splits_v1D.json', 'original_split_file_whitelist')
    audit.require(local(args.d08) == ROOT/'runs/geo_weather_20260924/d08_panel_20260929_iofix1', 'frozen_D08_directory_whitelist')
    audit.require(local(args.scope) == E/'notes/D09_INCREMENTAL_WRITE_DESIGN_20260929_ZH.md', 'fixed_scope_file_whitelist')
    paths = {key: local(getattr(args, key)) for key in ('jobs', 'scores', 'scope', 'splits', 'features', 'd08')}
    files = {}
    def remember(path, expected=None):
        path = local(path)
        digest = file_hash(path)
        if expected is not None:
            audit.equal(digest, expected, 'hash/'+rel(path))
        files[rel(path)] = digest
        return digest
    remember(Path(__file__), git_hash(args.audit_commit, Path(__file__)))
    adapter=local(args.precision_adapter)
    audit.require(adapter==E/'d09_precision_score.py','precision_adapter_file_whitelist')
    remember(adapter,git_hash(args.precision_commit,adapter))
    remember(paths['scope'], git_hash(EXPERIMENT_COMMIT, paths['scope']))
    remember(paths['jobs']); remember(paths['scores']); remember(paths['features'], DATA_SHA)
    d08 = paths['d08']
    remember(d08/'FROZEN.json')
    seal = read_json(d08/'FROZEN.json')
    audit.equal(seal['status'], 'input_roster_frozen', 'D08_frozen_status')
    for name, key in (('manifest.json', 'manifest_sha256'), ('roster.json', 'roster_sha256'),
                      ('panel_inputs.npz', 'panel_inputs_sha256')):
        remember(d08/name, seal[key])
    d08_manifest = read_json(d08/'manifest.json')
    remember(paths['splits'], d08_manifest['provenance']['split_sha256'])
    audit.equal(d08_manifest['provenance']['data_sha256'], DATA_SHA, 'D08_original_D_identity')
    roster = read_json(d08/'roster.json')
    split = read_json(paths['splits'])
    original_fit = sorted(split['event']['1']['dev'])
    audit.equal(len(original_fit), 6350, 'original_fit_population')
    audit.array_equal(np.asarray(original_fit), np.sort(roster['original_fit']), 'roster_original_fit')
    records = {int(r['unit']): r for r in roster['records']}
    audit.equal(len(records), 1219, 'fixed_input_roster_units')
    audit.equal(len(records), len(roster['records']), 'unique_input_roster_units')
    audit.require(set(records).issubset(original_fit), 'roster_within_original_fit')
    inventory = read_json(paths['jobs'])
    audit.equal(inventory['schema'], 'd09_jobs_v1', 'jobs_schema')
    audit.equal(inventory['registration_commit'], EXPERIMENT_COMMIT, 'jobs_registration')
    jobs = inventory['jobs']
    audit.equal(len(jobs), 6, 'six_jobs')
    audit.require({(j['arm'], int(j['heldout_fold'])) for j in jobs} ==
                  {(a, f) for a in ARMS for f in FOLDS}, 'six_distinct_registered_jobs')
    receipts, source_maps, protocols, host_hashes = {}, [], [], []
    protocol_fields = dict(steps=900, seed=0, private_seed=1729, microbatch=512,
        numerical_threads=2, early_stopping=False, checkpoints=[0,100,300,900],
        heldout_full_export_step=900, host_learning_rate=.003, recovery_learning_rate=.0003,
        calibration_every=10, minimum_nice=15,
        gradient_accumulation='one complete inner-FIT objective per Adam update',
        training_weights='original panel w / inner-FIT regime zero-predictor SSE; not HT')
    for job in jobs:
        arm, fold = job['arm'], int(job['heldout_fold'])
        folder = local(job['path'])
        audit.require(rel(folder).startswith('runs/geo_weather_20260924/d09_write_screen_20260929/') and
                      folder.name==arm+'_f'+str(fold),'owned_D09_terminal_directory')
        done = folder/'DONE.json'
        remember(done)
        receipt = read_json(done)
        prefix = arm+'_f'+str(fold)
        for name, expected in dict(schema='d09_job_done_v1', arm=arm, heldout_fold=fold,
                steps=900, seed=0, registration_commit=EXPERIMENT_COMMIT, preflight=False,
                optimizer_covers_all_trainable=True).items():
            audit.equal(receipt[name], expected, prefix+'/'+name)
        for name, expected in protocol_fields.items():
            audit.equal(receipt['protocol'][name], expected, prefix+'/protocol/'+name)
        audit.require(receipt['resources']['nice'] >= 15 and receipt['resources']['threads']==2,
                      prefix+'/resources')
        audit.require(math.isfinite(receipt['fit_loss']) and math.isfinite(receipt['seconds']) and
                      receipt['seconds']>0 and math.isfinite(receipt['peak_rss_gib']) and
                      receipt['peak_rss_gib']>0,prefix+'/finite_terminal_training_receipt')
        for field, path in (('scope_sha256',paths['scope']), ('data_sha256',paths['features']),
            ('split_sha256',paths['splits']), ('d08_frozen_sha256',d08/'FROZEN.json'),
            ('d08_manifest_sha256',d08/'manifest.json'), ('d08_roster_sha256',d08/'roster.json')):
            audit.equal(receipt[field], files[rel(path)], prefix+'/'+field)
        source = receipt['source_sha256']
        audit.require(bool(source), prefix+'/complete_declared_source_map_present')
        for name, digest in source.items():
            audit.require((name.startswith('src/asymode/') or
                name.startswith('experiments/geo_weather_20260924/')) and
                'paper_v1' not in Path(name).parts and Path(name).suffix in ('.py','.md') and
                rel(name)==name, prefix+'/source_whitelist')
            remember(ROOT/name, digest)
            audit.equal(git_hash(EXPERIMENT_COMMIT, ROOT/name), digest, prefix+'/registered/'+name)
        source_maps.append(source); protocols.append(receipt['protocol'])
        host_hashes.append(receipt['initial_host_parameter_sha256'])
        needed = ('step0900_held_full.npz','step0900_held_panel.npz','step0900_fit_panel.npz',
                  'final.pt','stats.json')
        audit.require(set(needed).issubset(receipt['outputs']), prefix+'/terminal_outputs_complete')
        for name, digest in receipt['outputs'].items():
            audit.require(Path(name).name==name, prefix+'/output_basename')
            remember(folder/name,digest)
        held = np.asarray(sorted(set(original_fit) & set(split['event'][str(fold)]['outer'])), int)
        train = np.asarray(sorted(i for i,r in records.items() if int(r['original_fold'])!=fold), int)
        panel = np.asarray(sorted(set(held) & set(records)), int)
        for name, expected in (('held_full_unit_ids',held), ('held_panel_unit_ids',panel),
                              ('fit_unit_ids',train), ('model_statistics_fit_unit_ids',train)):
            audit.array_equal(np.asarray(receipt[name]),expected,prefix+'/'+name)
        audit.require(not set(train)&set(held),prefix+'/no_training_unit_overlap')
        if arm!='host':
            km=receipt['kernel_geography_fit_metadata']
            audit.equal(km['n_fit_rows'],len(train),prefix+'/kernel_fit_rows')
            fit_counties={str(records[int(i)]['fips']) for i in train}
            audit.equal(km['n_unique_counties'],len(fit_counties),prefix+'/kernel_fit_counties')
            audit.require(set(km['landmark_county_ids']).issubset(fit_counties),prefix+'/fit_only_landmarks')
        receipts[(arm,fold)]=dict(folder=folder,receipt=receipt,held=held,panel=panel,train=train)
    audit.require(all(s==source_maps[0] for s in source_maps),'same_frozen_sources_across_jobs')
    audit.require(all(p==protocols[0] for p in protocols),'same_protocol_across_jobs')
    audit.require(bool(host_hashes[0]) and len(set(host_hashes))==1,'paired_fresh_host_initialization')
    for fold in FOLDS:
        old,new=(receipts[(a,fold)]['receipt'] for a in ('crk','new'))
        audit.require(bool(old['initial_kernel_parameter_sha256']) and
            old['initial_kernel_parameter_sha256']==new['initial_kernel_parameter_sha256'],
            'paired_fresh_kernel_f'+str(fold))
    audit.require(not set(receipts[('host',2)]['held'])&set(receipts[('host',3)]['held']),
                  'disjoint_two_full_heldout_populations')
    return paths, files, receipts, records


def open_endpoints(receipts, records, audit):
    """Called only after all six DONE/source/output hashes have passed."""
    by_fold, preds, prediction_dtypes = {}, {a:[] for a in ARMS}, {}
    for fold in FOLDS:
        for arm in ARMS:
            item=receipts[(arm,fold)]
            with np.load(item['folder']/'step0900_held_full.npz',allow_pickle=False) as archive:
                full={k:archive[k].copy() for k in archive.files}
            prefix=arm+'_f'+str(fold)
            audit.require(set(META+('P','u','r','raw_logit')).issubset(full),prefix+'/export_schema')
            n=len(item['held'])
            audit.require(np.issubdtype(full['idx'].dtype,np.integer),prefix+'/original_integer_ID_dtype')
            audit.array_equal(full['idx'],item['held'],prefix+'/exact_full_heldout_coverage')
            for name in META:
                audit.require(full[name].shape==((n,144) if name in ('y','m') else (n,)),prefix+'/shape/'+name)
            audit.require(np.isin(full['m'],[0,1]).all(),prefix+'/binary_mask')
            full['m']=full['m'].astype(bool)
            audit.require(np.isfinite(full['y'][full['m']]).all() and
                np.all((full['y'][full['m']]>=0)&(full['y'][full['m']]<=1)) and
                np.isfinite(full['y0']).all() and np.all((full['y0']>=0)&(full['y0']<=1)),prefix+'/observed_labels_finite_range')
            audit.require(np.isin(full['origin_observed'],[0,1]).all() and full['origin_observed'].all(),prefix+'/observed_origin71')
            audit.require(np.isfinite(full['w']).all() and np.all(full['w']>0),prefix+'/original_positive_weights')
            audit.require(np.isfinite(full['pi']).all() and np.all((full['pi']>0)&(full['pi']<=1)),prefix+'/fixed_positive_pi')
            audit.require(np.isin(full['regime'],REGIMES).all() and np.all(full['original_fold']==fold),prefix+'/class_fold_identity')
            if arm!='host':
                audit.require({'P_closed','raw_logit_closed'}.issubset(full),prefix+'/closed_export_present')
            for name in ('P','u','r','raw_logit','P_closed','raw_logit_closed'):
                if name in full:
                    audit.require(full[name].shape==(n,144) and np.isfinite(full[name]).all(),prefix+'/finite/'+name)
                    if name in ('P','P_closed'):
                        audit.require(np.all((full[name]>=0)&(full[name]<=1)),prefix+'/prediction_range/'+name)
            selected=np.isin(full['idx'],item['panel'])
            audit.require(np.isin(full['panel'],[0,1]).all(),prefix+'/binary_panel_flag')
            audit.array_equal(full['panel'].astype(bool),selected,prefix+'/fixed_panel_flag')
            for i in np.flatnonzero(selected):
                record=records[int(full['idx'][i])]
                for name in ('fips','system','family','origin','regime','merged_group','original_fold','w','pi'):
                    audit.equal(full[name][i].item(),record[name],prefix+'/roster/'+str(int(full['idx'][i]))+'/'+name)
            with np.load(item['folder']/'step0900_held_panel.npz',allow_pickle=False) as archive:
                panel={k:archive[k].copy() for k in archive.files}
            audit.require(set(full).issubset(panel),prefix+'/panel_export_schema')
            for name in full:
                audit.array_equal(full[name][selected],panel[name],prefix+'/full_panel_exact/'+name)
            if fold in by_fold:
                for name in META:
                    audit.array_equal(full[name],by_fold[fold][name],prefix+'/paired_metadata/'+name)
            else:
                by_fold[fold]={name:full[name] for name in META}
            preds[arm].append(full['P'])
            dtypes={arm:str(full['P'].dtype)}
            if arm!='host':
                dtypes[arm+'_closed']=str(full['P_closed'].dtype)
            for name,dtype in dtypes.items():
                if name in prediction_dtypes:
                    audit.equal(dtype,prediction_dtypes[name],'consistent_export_dtype/'+name)
                prediction_dtypes[name]=dtype
    data={name:np.concatenate([by_fold[f][name] for f in FOLDS]) for name in META}
    predictions={name:np.concatenate(parts) for name,parts in preds.items()}
    audit.equal(len(set(data['idx'])),len(data['idx']),'exactly_once_pooled_heldout_coverage')
    for group in sorted(set(data['merged_group'].astype(str))):
        audit.equal(len(set(data['original_fold'][data['merged_group']==group])),1,'merged_group_one_fold')
    for fold in FOLDS:
        train_groups={records[int(i)]['merged_group'] for i in receipts[('host',fold)]['train']}
        audit.require(not train_groups&set(by_fold[fold]['merged_group']),'no_merged_group_leakage_f'+str(fold))
    return data,predictions,dict(data={name:str(data[name].dtype) for name in ('y','y0','w','pi')},
                                 predictions=prediction_dtypes)


def recompute_and_compare(data,predictions,scores,audit):
    audit.equal(scores['schema'],'d09_endpoint_scores_v1','score_schema')
    primary=scores['primary']
    audit.equal(primary['surface'],'complete_heldout_primary','primary_surface')
    for name,value in dict(draws=DRAW_COUNT,seed=DRAW_SEED,paired=True,confidence=.95).items():
        audit.equal(primary['bootstrap'][name],value,'registered_bootstrap/'+name)
    units,hours,masks=unit_arithmetic(data,predictions)
    strata={'pooled':np.ones(len(hours),bool)}
    strata.update({'fold'+str(f):data['original_fold']==f for f in FOLDS})
    strata.update({'regime/'+r:data['regime']==r for r in REGIMES})
    rows, alarm_rows={},{}
    for stratum,keep in strata.items():
        rows[stratum],alarm_rows[stratum]={},{}
        for cohort,selection in masks.items():
            chosen=keep&selection
            expected=primary['rows'][stratum][cohort]
            actual=metrics(data,units,hours,chosen)
            for name in ('units','observed_hours','merged_groups','design_hour_mass'):
                audit.equal(expected[name],actual[name],stratum+'/'+cohort+'/'+name)
            for arm in ARMS:
                for name in ('design_SSE','design_RMSE'):
                    audit.equal(expected['models'][arm][name],actual['models'][arm][name],stratum+'/'+cohort+'/'+arm+'/'+name)
                quantiles=expected['models'][arm]['peak_ratio']['q10_median_q90']
                audit.equal(None if quantiles is None else quantiles[1],actual['models'][arm]['peak_ratio_median'],
                            stratum+'/'+cohort+'/'+arm+'/peak_ratio_median')
            comparisons={}
            for label,(base,candidate) in COMPS.items():
                value=reduction(actual['models'][base]['design_SSE'],actual['models'][candidate]['design_SSE'])
                audit.equal(expected['comparisons'][label]['relative_RMSE_reduction'],value,stratum+'/'+cohort+'/'+label)
                comparisons[label]=value
                if cohort=='S':
                    summary=gains(data,predictions,units,chosen,base,candidate)
                    for name,value_ in summary.items():
                        audit.equal(expected['comparisons'][label]['paired_gain'][name],value_,stratum+'/S/'+label+'/gain/'+name)
                    audit.equal(summary['net_gain'],summary['alignment']-summary['modification_energy'],stratum+'/S/'+label+'/SSE_identity')
                    actual.setdefault('gain',{})[label]=summary
            actual['comparisons']=comparisons
            rows[stratum][cohort]=actual
        for label,(base,candidate) in COMPS.items():
            actual=alarms(data,units,keep&masks['nonS'],base,candidate)
            for name,value in actual.items():
                audit.equal(primary['alarms'][stratum][label]['nonS_severe'][name],value,stratum+'/'+label+'/nonS_alarm/'+name)
            alarm_rows[stratum][label]=actual
    plans,intervals={},{}
    for label,field in (('merged_group','merged_group'),('family_sensitivity','family')):
        plan=cluster_plan(data,field)
        plans[label]=dict(plan['summary'],counts_sha256=plan['counts_sha256'])
        expected=primary['bootstrap']['plans'][label]
        audit.equal(expected['clusters'],plan['summary']['clusters'],'bootstrap/'+label+'/clusters')
        audit.require(expected['stratum_cluster_counts']==plan['summary']['stratum_cluster_counts'],'bootstrap/'+label+'/strata_counts')
        audit.require(expected['singleton_strata']==plan['summary']['singleton_strata'],'bootstrap/'+label+'/singleton_strata')
        for cohort in ('S','all'):
            for comparison,(base,candidate) in COMPS.items():
                actual=interval(units,masks[cohort],plan,base,candidate)
                expected=primary['rows']['pooled'][cohort]['comparisons'][comparison]['bootstrap'][label]
                for name,value in actual.items():
                    audit.equal(expected[name],value,'CI/'+label+'/'+cohort+'/'+comparison+'/'+name)
                intervals.setdefault(cohort,{}).setdefault(comparison,{})[label]=actual
    r=lambda stratum,cohort,label:rows[stratum][cohort]['comparisons'][label]
    pass_value=lambda value,predicate:value is not None and math.isfinite(value) and predicate(value)
    checks=dict(pooled_S_new_vs_host_at_least_5pct=pass_value(r('pooled','S','new_vs_host'),lambda x:x>=.05),
        pooled_S_new_vs_crk_positive=pass_value(r('pooled','S','new_vs_crk'),lambda x:x>0),
        fold2_S_new_vs_host_positive=pass_value(r('fold2','S','new_vs_host'),lambda x:x>0),
        fold3_S_new_vs_host_positive=pass_value(r('fold3','S','new_vs_host'),lambda x:x>0),
        pooled_all_new_vs_host_not_worse=pass_value(r('pooled','all','new_vs_host'),lambda x:x>=0))
    alarm=alarm_rows['pooled']['new_vs_host']
    checks['nonS_severe_alarm_count_not_higher']=alarm['candidate_count']<=alarm['base_count']
    checks['nonS_severe_alarm_design_rate_not_higher']=alarm['units']==0 or (
        alarm['candidate_design_rate'] is not None and alarm['base_design_rate'] is not None and
        alarm['candidate_design_rate']<=alarm['base_design_rate'])
    class_guards={}
    for regime in REGIMES[2:]:
        key='regime/'+regime
        groups=rows[key]['all']['merged_groups']
        value=r(key,'all','new_vs_host')
        applicable=groups>=2
        passed=pass_value(value,lambda x:x>=-.02) if applicable else None
        class_guards[regime]=dict(supporting_merged_groups=groups,applicable=applicable,
            relative_RMSE_reduction=value,maximum_allowed_RMSE_increase=.02,passed=passed)
        for name,value_ in class_guards[regime].items():
            audit.equal(scores['verdict']['nonheadline_class_guards'][regime][name],value_,'gate/class/'+regime+'/'+name)
        if applicable:
            checks['class_'+regime+'_all_RMSE_not_worse_than_2pct']=passed
    audit.require(scores['verdict']['checks']==checks,'exact_registered_point_gate_checks')
    passed=all(checks.values())
    audit.equal(scores['verdict']['point_gate_pass'],passed,'registered_point_gate_pass')
    audit.equal(scores['verdict']['S_new_vs_host_relative_RMSE_reduction'],r('pooled','S','new_vs_host'),'verdict_S_new_host_value')
    audit.equal(scores['verdict']['S_new_vs_crk_relative_RMSE_reduction'],r('pooled','S','new_vs_crk'),'verdict_S_new_crk_value')
    for plan,fields in intervals['S']['new_vs_host'].items():
        for name,value in fields.items():
            audit.equal(scores['verdict']['S_new_vs_host_bootstrap'][plan][name],value,'verdict_S_interval/'+plan+'/'+name)
    audit.equal(scores['verdict']['decision'],'may_propose_full_D_five_fold_comparison' if passed else
                'do_not_advance_this_candidate_by_registered_gate','verdict_decision')
    audit.equal(scores['verdict']['automatically_start_full_D'],False,'no_automatic_full_D')
    audit.equal(scores['verdict']['original_final_S_target_relative_RMSE_reduction'],.1,'unchanged_final_10pct_target')
    return dict(full_primary=rows,nonS_severe_alarms=alarm_rows,bootstrap_plans=plans,
                pooled_intervals=intervals,point_gate=dict(passed=passed,checks=checks,class_guards=class_guards))


def self_check():
    audit=Audit()
    y=np.zeros((4,144));y[0,:7]=.2;y[1,[0,2,4]]=.1;y[2,:]=.01;y[3,:]=0
    m=np.ones_like(y,bool);m[1,2]=False;y[1,2]=np.nan
    data=dict(idx=np.arange(4),y=y,m=m,y0=np.array([0,.099,0,0]),w=np.array([1,2,3,4.]),
              merged_group=np.array(['a','a','b','b']),family=np.array(['x','x','y','y']),
              regime=np.repeat('tropical',4),original_fold=np.array([2,2,3,3]))
    ps={a:np.where(m,y,0)*factor for a,factor in (('host',.5),('crk',.7),('new',.8))}
    unit,hours,masks=unit_arithmetic(data,ps)
    audit.array_equal(masks['S'],[True,True,False,False],'synthetic_S_mask')
    audit.array_equal(masks['J'],[True,True,True,False],'synthetic_J_origin_and_missing')
    audit.equal(hours[1],143,'synthetic_missing_hour')
    score=metrics(data,unit,hours,masks['S'])
    audit.equal(score['models']['host']['design_SSE'],.08,'synthetic_manual_S_SSE')
    audit.equal(score['models']['new']['design_SSE'],.0128,'synthetic_manual_candidate_SSE')
    audit.equal(reduction(.08,.0128),.6,'synthetic_manual_reduction')
    audit.equal(med([0,1],[1,1]),0.,'synthetic_cumulative_weighted_median')
    g=gains(data,ps,unit,masks['S'],'host','new')
    audit.equal(g['positive_design_share'],1.,'synthetic_gain_coverage')
    audit.equal(g['net_gain'],g['alignment']-g['modification_energy'],'synthetic_gain_identity')
    plan=cluster_plan(data,'merged_group',1999)
    ci=interval(unit,masks['all'],plan,'host','new')
    audit.equal(ci['interval_supported'],True,'synthetic_real_resampling_support')
    audit.equal(ci['relative_RMSE_reduction_ci95'],[.6,.6],'synthetic_proportional_zero_width_valid')
    singleton=dict(data,regime=np.array(['tropical','tropical','winter','winter']))
    ci=interval(unit,masks['all'],cluster_plan(singleton,'merged_group',1999),'host','new')
    audit.equal(ci['interval_supported'],False,'synthetic_singleton_no_confidence')
    audit.equal(ci['resample_counts_vary'],False,'synthetic_singleton_no_count_variation')
    audit.equal(ci['relative_RMSE_reduction_ci95'],None,'synthetic_singleton_null_CI')
    ps['new'][2,10]=.2
    unit,_,_=unit_arithmetic(data,ps)
    alarm=alarms(data,unit,masks['nonS'],'host','new')
    audit.equal(alarm['new_count'],1,'synthetic_new_false_peak')
    audit.equal(alarm['candidate_design_rate'],3/7,'synthetic_false_peak_original_design_rate')
    weighted=dict(data,w=np.asarray([.1,.7,1.3,2.1],dtype=np.float32))
    fp={name:p.copy() for name,p in ps.items()}
    fp['new'][1]=np.where(m[1],y[1],0)*.2
    unit,_,masks=unit_arithmetic(weighted,fp)
    g=gains(weighted,fp,unit,masks['S'],'host','new')
    expected=float(weighted['w'][0])/math.fsum(float(x) for x in weighted['w'][:2])
    audit.equal(g['positive_units'],1,'synthetic_nonuniform_float32_positive_count')
    audit.require(g['positive_design_share']==expected,'synthetic_nonuniform_float32_share_exact_double')
    return audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--self-check',action='store_true')
    for name in ('jobs','scores','scope','splits','features','d08','out','precision-adapter'):
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--scope-commit')
    parser.add_argument('--audit-commit')
    parser.add_argument('--precision-commit')
    args=parser.parse_args()
    if args.self_check:
        if args.execute:
            parser.error('Synthetic checks and real execution are separate modes')
        os.nice(max(0,15-os.getpriority(os.PRIO_PROCESS,0)))
        audit=self_check()
        print(json.dumps(dict(status='synthetic_checks_passed',assertions=audit.count,
                              maximum_absolute_error=audit.maximum_absolute_error)))
        return
    if not args.execute:
        print(json.dumps(dict(status='not_executed',requires_explicit_root_GO=True,
            experiment_commit=EXPERIMENT_COMMIT,bootstrap_draws=DRAW_COUNT,
            absolute_tolerance=ATOL,relative_tolerance=RTOL)))
        return
    for name in ('jobs','scores','scope','splits','features','d08','out','scope_commit','audit_commit',
                 'precision_adapter','precision_commit'):
        if getattr(args,name) is None:
            parser.error('--'+name.replace('_','-')+' is required for execution')
    out=local(args.out)
    if not rel(out).startswith('experiments/geo_weather_20260924/results/v1/') or out.suffix!='.json':
        raise ValueError('Use an owned compact result JSON output')
    if out.exists():
        raise FileExistsError('Preserve existing audit receipt')
    os.nice(max(0,15-os.getpriority(os.PRIO_PROCESS,0)))
    audit=Audit()
    started=datetime.now(timezone.utc).isoformat()
    report=dict(schema='d09_independent_terminal_audit_v1',started_utc=started,
                experiment_commit=EXPERIMENT_COMMIT,audit_registration_commit=args.audit_commit,
                numerical_tolerance=dict(absolute=ATOL,relative=RTOL),bootstrap_draws=DRAW_COUNT,
                bootstrap_seed=DRAW_SEED,reads_original_D_members=False,
                new_model_forward=False,new_training=False)
    try:
        paths,files,receipts,records=verify_files(args,audit)
        scores=read_json(paths['scores'])
        provenance=scores['provenance']
        audit.equal(provenance['registration_commit'],EXPERIMENT_COMMIT,'score_registration')
        audit.equal(provenance['data_sha256'],DATA_SHA,'score_original_D_hash')
        reference=receipts[('host',2)]['receipt']
        audit.require(provenance['source_sha256']==reference['source_sha256'],'score_frozen_training_source_map')
        audit.require(provenance['protocol']==reference['protocol'],'score_registered_protocol')
        for name,value in dict(scope_sha256=files[rel(paths['scope'])],original_fit_units=6350,
                original_heldout_folds=list(FOLDS),terminal_steps=900,seed=0,
                initial_host_parameter_sha256=reference['initial_host_parameter_sha256']).items():
            audit.equal(provenance[name],value,'score_provenance/'+name)
        precision=provenance['precision_adapter']
        for name,value in dict(path=rel(args.precision_adapter),source_sha256=files[rel(args.precision_adapter)],
                registration_commit=args.precision_commit,arithmetic_dtype='float64').items():
            audit.equal(precision[name],value,'registered_precision_adapter/'+name)
        audit.require(set(precision['converted_arrays']['data'])=={'y','y0','w','pi'},'precision_data_views')
        audit.require(set(precision['converted_arrays']['predictions'])==
                      {'host','crk','new','crk_closed','new_closed'},'precision_prediction_views')
        audit.require(precision['scoring_calls']==[{'panel':False},{'panel':True}],
                      'precision_primary_and_secondary_calls')
        audit.equal(precision['frozen_score_source_sha256'],files[rel(E/'d09_score.py')],
                    'precision_frozen_scorer_source_hash')
        for name,digest in provenance['files_sha256'].items():
            audit.require(name in files,'known_verified_score_inventory_path')
            audit.equal(files[name],digest,'score_current_inventory/'+name)
        data,predictions,original_dtypes=open_endpoints(receipts,records,audit)
        audit.require(precision['original_dtypes']==original_dtypes,'precision_original_export_dtypes')
        audit.equal(provenance['full_heldout_units'],len(data['idx']),'score_full_heldout_units')
        audit.equal(provenance['panel_heldout_units'],int(data['panel'].sum()),'score_panel_heldout_units')
        report['recomputed']=recompute_and_compare(data,predictions,scores,audit)
        for name,digest in files.items():
            audit.equal(file_hash(ROOT/name),digest,'unchanged_after_audit/'+name)
        report.update(status='passed',passed=True,assertions_passed=audit.count,
            full_heldout_units=len(data['idx']),heldout_folds=list(FOLDS),files_sha256=files,
            precision_adapter=precision,
            maximum_absolute_error=audit.maximum_absolute_error,
            maximum_relative_error=audit.maximum_relative_error,
            interpretation='Independent endpoint arithmetic; exploratory input-transductive D, single seed, no causal or net-geography claim.')
    except Exception as error:
        report.update(status='failed',passed=False,assertions_passed=audit.count,
                      failed_check=audit.last,error_type=type(error).__name__)
        out.parent.mkdir(parents=True,exist_ok=True)
        with out.open('x') as stream:
            json.dump(report,stream,ensure_ascii=False,allow_nan=False,indent=1);stream.write('\n')
        print(json.dumps(dict(status='failed',failed_check=audit.last,error_type=type(error).__name__,receipt=rel(out))))
        raise SystemExit(1)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as stream:
        json.dump(report,stream,ensure_ascii=False,allow_nan=False,indent=1);stream.write('\n')
    print(json.dumps(dict(status='passed',assertions=audit.count,receipt=rel(out),sha256=file_hash(out))))


if __name__=='__main__':
    main()
