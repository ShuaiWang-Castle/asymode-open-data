"""Small synthetic I20 evaluation/runner checks; no real panel or training."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'experiments/geo_weather_20260924'
sys.path.insert(0, str(HERE))
import evaluate_cr_tail as E
import run_cr_screen as R


def example():
    n = 5
    y = np.zeros((n, 144), dtype=float)
    y[0, :2] = [.2, .1]
    y[1, 0] = .01
    y[2, 0] = .001
    m = np.ones_like(y, dtype=bool)
    m[3, 1:] = False
    m[4] = False
    yf = np.zeros((n, 216))
    yf[:, 72:] = np.where(m, y, np.nan)
    obs = np.isfinite(yf)
    meta = dict(system=np.array(list('abcde')), family=np.array(list('ABCDE')),
        origin=np.array(['2020-01-01', '2020-03-01', '2020-05-01', '2020-07-01', '2020-09-01']),
        county=np.array(['00001', '00002', '00003', '00004', '00005']),
        regime=np.array(E.REGIMES), w=np.array([2., 3., 4., 5., 6.]),
        w_raw=np.array([3., 5., 7., 11., 13.]))
    split = dict(n_units=n, event={str(k): dict(outer=[k-1], dev=[i for i in range(n) if i != k-1])
                                      for k in range(1, 6)}, event_groups={s: j+1 for j, s in enumerate(meta['system'])})
    expected, fold = E.validate_split(split, n, meta, meta['system'])
    return dict(y=y, m=m, y_full=yf, obs_full=obs, y0=np.zeros(n), meta=meta,
                group=meta['system'], expected=expected, fold=fold, split=split)


def exports(tmp_path, data, label='candidate', arm='CRK+Cin'):
    for k, idx in data['expected'].items():
        folder = tmp_path / label / f'fold{k:02d}'
        folder.mkdir(parents=True)
        np.savez(folder / 'outer.npz', idx=idx, P=np.full((len(idx), 144), .05))
        (folder / 'final.pt').write_bytes(b'not loaded by the evaluator')
        (folder / 'DONE.json').write_text(json.dumps(dict(label=label, arm=arm, data='v1D',
            design='event', design_weights=True, fold=k, seed=0, steps=900)))


def test_split_rejects_duplicates_missing_and_wrong_systems():
    data = example()
    split = json.loads(json.dumps(data['split']))
    split['event']['1']['outer'] = [0, 0]
    with pytest.raises(ValueError, match='duplicate'): E.validate_split(split, 5)
    split = json.loads(json.dumps(data['split']))
    split['event']['1']['outer'] = []
    with pytest.raises(ValueError): E.validate_split(split, 5)
    split = json.loads(json.dumps(data['split']))
    split['event_groups']['a'] = 2
    with pytest.raises(ValueError, match='mapping'): E.validate_split(split, 5, data['meta'])


def test_load_predictions_exact_coverage_and_metadata(tmp_path):
    data = example(); exports(tmp_path, data)
    p, receipts = E.load_predictions('candidate', 'CRK+Cin', data['expected'], 5, tmp_path)
    assert p.shape == (5, 144) and np.all(p == .05) and len(receipts) == 5
    f = tmp_path / 'candidate/fold01/DONE.json'
    d = json.loads(f.read_text()); d['steps'] = 899; f.write_text(json.dumps(d))
    with pytest.raises(ValueError, match='steps'):
        E.load_predictions('candidate', 'CRK+Cin', data['expected'], 5, tmp_path)


@pytest.mark.parametrize('kind', ['duplicate', 'wrong_fold', 'nonfinite', 'range', 'columns', 'missing'])
def test_load_predictions_bad_exports(tmp_path, kind):
    data = example(); exports(tmp_path, data)
    f = tmp_path / 'candidate/fold01/outer.npz'
    idx, p = np.array([0]), np.full((1, 144), .1)
    if kind == 'duplicate': idx, p = np.array([0, 0]), np.zeros((2, 144))
    if kind == 'wrong_fold': idx = np.array([1])
    if kind == 'nonfinite': p[0, 0] = np.nan
    if kind == 'range': p[0, 0] = 1.01
    if kind == 'columns': p = p[:, :143]
    if kind == 'missing':
        f.unlink()
    else: np.savez(f, idx=idx, P=p)
    with pytest.raises((ValueError, FileNotFoundError)):
        E.load_predictions('candidate', 'CRK+Cin', data['expected'], 5, tmp_path)


def test_outcomes_loader_reads_no_weather_and_checks_mask(tmp_path):
    data = example(); path = tmp_path / 'tiny.npz'; sp = tmp_path / 'split.json'
    payload = dict(y=data['y'], m=data['m'], y_full=data['y_full'], obs_full=data['obs_full'],
                   y0=data['y0'], event=data['meta']['system'], fips=data['meta']['county'],
                   **{k: v for k, v in data['meta'].items() if k != 'county'})
    np.savez(path, **payload); sp.write_text(json.dumps(data['split']))
    actual = E.load_outcomes(path, sp, expected_units=5)
    assert np.array_equal(actual['m'], data['m'])
    payload['m'] = np.ones_like(data['m']); np.savez(path, **payload)
    with pytest.raises(ValueError, match='masks differ'): E.load_outcomes(path, sp, expected_units=5)


def test_cohorts_use_adjacent_mask_origin_and_exact_thresholds():
    data = example(); c = E.cohorts(data)
    assert np.array_equal(np.flatnonzero(c['S']), [0])
    assert np.array_equal(np.flatnonzero(c['J']), [0, 1])
    assert np.array_equal(np.flatnonzero(c['any_positive']), [0, 1, 2])
    assert np.array_equal(np.flatnonzero(c['all_observed_zero']), [3])
    assert not c['all'][4]
    data['obs_full'][1, 71] = False
    assert not E.cohorts(data)['J'][1]


def test_full_window_score_matches_direct_weighted_average():
    data = example(); p = np.full((5, 144), .04); h = np.full((5, 144), .08)
    sel = np.array([True, False, True, True, False]); w = data['meta']['w']
    result = E.score_block(data, p, h, sel, w)
    keep = data['m'] & sel[:, None]
    row_weight = np.broadcast_to(w[:, None], keep.shape)[keep]
    direct = np.sqrt(np.average((p[keep] - data['y'][keep]) ** 2, weights=row_weight))
    host = np.sqrt(np.average((h[keep] - data['y'][keep]) ** 2, weights=row_weight))
    assert result['models']['candidate']['rmse'] == pytest.approx(direct)
    assert result['rmse_improvement_fraction'] == pytest.approx(1 - direct / host)
    assert result['support']['observed_hours'] == 289
    assert result['support']['max_merged_group_weighted_hour_share'] == pytest.approx(4 * 144 / (2 * 144 + 4 * 144 + 5))
    # Snapshots are fixed-origin single columns, not cumulative first-six-hour means.
    snap = E.score_block(data, p, h, sel, w, columns=[5])
    assert snap['support']['observed_hours'] == 2


def test_paired_bootstrap_matches_explicit_cluster_replication():
    data = example(); data['meta']['regime'][:] = 'tropical'
    plan = E.bootstrap_plan(data['meta'], data['group'], draws=99, seed=17)
    p, h = np.full((5, 144), .04), np.full((5, 144), .08)
    selected = E.cohorts(data)['all']; w = data['meta']['w']
    result = E.score_block(data, p, h, selected, w, {'test': plan})
    direct = []
    for counts in plan['counts']:
        bw = w * counts[plan['code']]
        se_p = np.where(data['m'], (p - data['y']) ** 2, 0).sum(1)
        se_h = np.where(data['m'], (h - data['y']) ** 2, 0).sum(1)
        if (bw * data['m'].sum(1)).sum() > 0 and (bw * se_h).sum() > 0:
            direct.append(1 - np.sqrt((bw * se_p).sum() / (bw * se_h).sum()))
    assert np.allclose(result['intervals']['test']['ci95'], np.quantile(direct, [.025, .975]))
    assert result['intervals']['test']['valid_draws'] == len(direct)


def test_cross_regime_cluster_is_sampled_whole_and_single_cluster_has_no_ci():
    data = example(); labels = np.array(['ab', 'ab', 'c', 'd', 'e'])
    plan = E.bootstrap_plan(data['meta'], labels, draws=19)
    assert plan['code'][0] == plan['code'][1]
    assert E.REGIMES[plan['strata'][plan['code'][0]]] == 'winter'  # w=3 versus 2
    p, h = np.full((5, 144), .04), np.full((5, 144), .08)
    selected = np.array([True, True, False, False, False])
    result = E.score_block(data, p, h, selected, data['meta']['w'], {'merged': plan})
    assert result['intervals']['merged']['supporting_clusters'] == 1
    assert result['intervals']['merged']['ci95'] is None


def test_zero_host_denominator_empty_support_and_missing_false_peaks():
    data = example(); zero = np.zeros((5, 144)); selected = E.cohorts(data)['all_observed_zero']
    plan = E.bootstrap_plan(data['meta'], data['group'], draws=19)
    result = E.score_block(data, zero, zero, selected, data['meta']['w'], {'merged': plan})
    assert np.isnan(result['rmse_improvement_fraction'])
    assert result['intervals']['merged']['valid_draws'] == 0
    empty = E.score_block(data, zero, zero, np.zeros(5, bool), data['meta']['w'])
    assert E.clean(empty)['models']['candidate']['rmse'] is None
    p = zero.copy(); p[3, 1:] = .9  # unobserved: cannot establish a false alarm
    alarm = E.alarm_block(data, p, zero, selected, data['meta']['w'], .001, False, {})
    assert alarm['candidate_rate'] == 0 and alarm['candidate_count'] == 0
    p[3, 0] = .002
    alarm = E.alarm_block(data, p, zero, selected, data['meta']['w'], .001, False, {})
    assert alarm['candidate_rate'] == 1 and alarm['candidate_count'] == 1


def test_report_all_cohorts_and_no_overwrite(tmp_path):
    data = example(); p = np.full((5, 144), .04); h = np.full((5, 144), .08)
    report = E.report(data, p, h, primary='any_positive', draws=9)
    assert report['primary_target']['cohort'] == 'any_positive'
    assert set(report['weights']['w']['cohorts']) == {'all', 'S', 'J', 'S_and_J', 'any_positive'}
    assert set(report['weights']['w']['cohorts']['S']['by_fold']) == set('12345')
    assert report['weights']['w']['cohorts']['S']['by_fold']['1']['support']['county_events'] == 1
    assert report['weights']['w']['cohorts']['S']['by_fold']['2']['support']['county_events'] == 0
    path = tmp_path / 'out.json'; E.write_new(path, report)
    assert json.loads(path.read_text())['meta']['units'] == 5
    with pytest.raises(FileExistsError): E.write_new(path, report)


def test_runner_is_plan_only_and_preserves_partial_folders(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(sys, 'argv', ['run_cr_screen.py'])
    R.main()
    assert json.loads(capsys.readouterr().out)['train'] is False
    monkeypatch.setattr(R, 'RUN', tmp_path)
    assert R.completion(1, {}, {}, E) is False
    (tmp_path / 'fold01').mkdir()
    with pytest.raises(RuntimeError, match='partial'): R.completion(1, {}, {}, E)
    (tmp_path / 'fold01/DONE.json').write_text('{}')
    with pytest.raises(RuntimeError, match='receipt'): R.completion(1, {}, {}, E)


def test_runner_validated_receipt_reuses_and_detects_changed_artifacts(monkeypatch, tmp_path):
    data = example(); exports(tmp_path, data, label=R.LABEL, arm=R.ARM)
    monkeypatch.setattr(R, 'RUN', tmp_path / R.LABEL)
    frozen = {'source_sha256': {'synthetic_source': 'abc'}}
    folder = R.RUN / 'fold01'
    payload = {'idx': data['expected'][1]}
    payload.update({key: np.full((1, 144), .05) for key in
                    ('P', 'u', 'r', 'raw_logit', 'P_closed', 'raw_logit_closed')})
    np.savez(folder / 'outer.npz', **payload)
    receipt = R.fold_receipt(1, frozen, data, E)
    E.write_new(folder / 'RUNNER_RECEIPT.json', receipt)
    assert R.completion(1, frozen, data, E)
    payload['P'][0, 0] = .2
    np.savez(folder / 'outer.npz', **payload)
    with pytest.raises(RuntimeError, match='receipt mismatch'): R.completion(1, frozen, data, E)
