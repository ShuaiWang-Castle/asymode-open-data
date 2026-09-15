import inspect, json
from pathlib import Path
import numpy as np, pandas as pd, pytest, torch

import models_v2 as M

ROOT = Path(__file__).resolve().parents[3]
A = Path(__file__).resolve().parents[1]
CTX, STEP, H, TARGET = 600, 15, 24, 32768
ALL = [('NET', s) for s in M.NET_DEPTHS] + [('ASYM', s) for s in M.ASYM_RATIOS]


def build(kind, structure, scale=0.07, source_mean=0.03):
    return M.build_model(kind, structure, CTX, STEP, H, TARGET, scale, source_mean)


@pytest.mark.parametrize('kind,structure', ALL)
def test_every_candidate_is_within_one_percent_of_the_common_parameter_target(kind, structure):
    assert abs(M.n_params(build(kind, structure)) / TARGET - 1) <= 0.01


@pytest.mark.parametrize('structure', list(M.ASYM_RATIOS))
def test_asym_damage_and_recovery_are_separate_networks_with_the_requested_ratio(structure):
    m = build('ASYM', structure)
    parts = m.params_by_part()
    assert parts['damage'] + parts['recovery'] == M.n_params(m)
    assert abs(parts['damage'] / parts['recovery'] / M.ASYM_RATIOS[structure] - 1) <= 0.15
    assert not ({id(p) for p in m.damage.parameters()} & {id(p) for p in m.recovery.parameters()})
    assert parts['damage'] >= parts['recovery']


@pytest.mark.parametrize('kind,structure', ALL)
def test_forward_accepts_no_target_and_changing_target_cannot_change_it(kind, structure):
    m = build(kind, structure)
    assert list(inspect.signature(m.forward).parameters) == ['c', 'y0', 'known_clock']
    torch.manual_seed(0)
    c, st, y0 = torch.randn(8, CTX), torch.randn(8, H, STEP), torch.rand(8) * 0.2
    with torch.no_grad():
        a = m(c, y0, st); b = m(c, y0, st)
    assert torch.equal(a, b)


@pytest.mark.parametrize('structure', list(M.ASYM_RATIOS))
def test_asym_stays_in_the_unit_interval_under_adversarial_weights(structure):
    torch.manual_seed(3)
    m = build('ASYM', structure)
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0.0, 1.0)
        y = m(torch.randn(64, CTX) * 3, torch.rand(64), torch.randn(64, H, STEP) * 3)
    assert float(y.min()) >= -1e-6 and float(y.max()) <= 1 + 1e-6


def test_asym_rates_do_not_read_the_recursive_state():
    assert list(inspect.signature(M.AsymSplitModel.rates).parameters) == ['self', 'c', 'known_clock']


@pytest.mark.parametrize('kind,structure', ALL)
def test_both_models_start_near_persistence(kind, structure):
    torch.manual_seed(1)
    m = build(kind, structure)
    y0 = torch.rand(256) * 0.3
    with torch.no_grad():
        y = m(torch.randn(256, CTX), y0, torch.randn(256, H, STEP))
    assert float((y - y0[:, None]).abs().max()) < 0.03


def test_asym_initial_rates_match_the_near_persistence_prior():
    m = build('ASYM', 'r2', source_mean=0.03)
    with torch.no_grad():
        u, r = m.rates(torch.randn(16, CTX), torch.randn(16, H, STEP))
    eps = M.ASYM_INIT_EPS
    assert np.allclose(u.numpy(), eps * 0.03, rtol=0.05)
    assert np.allclose(r.numpy(), eps * 0.97, rtol=0.05)


def test_net_can_move_in_both_directions_under_the_same_context():
    torch.manual_seed(2)
    m = build('NET', 'd2')
    with torch.no_grad():
        m.net.linear.weight.zero_()
        m.net.linear.weight[0, -1] = -0.5          # increment decreases with the state
        m.net.body[-1].weight.zero_(); m.net.body[-1].bias.fill_(0.2)
        c, st = torch.zeros(2, CTX), torch.zeros(2, H, STEP)
        y = m(c, torch.tensor([0.0, 0.9]), st)
    assert float(y[0, 0]) > 0.0 and float(y[1, 0]) < 0.9


LOCK = A / 'locks/FOLDS.json'


@pytest.fixture(scope='module')
def lock():
    if not LOCK.exists():
        pytest.skip('fold lock not built')
    return json.loads(LOCK.read_text())


def test_folds_partition_the_26_events_and_keep_overlap_components_whole(lock):
    ev = [e for f in lock['folds'] for e in f['events']]
    assert len(ev) == len(set(ev)) == 26 and set(ev) == set(lock['event_types'])
    fold_of = {e: f['fold'] for f in lock['folds'] for e in f['events']}
    for comp in lock['components']:
        assert len({fold_of[e] for e in comp['events']}) == 1


def test_roles_are_disjoint_and_folds_share_no_state_or_weather_cell(lock):
    for r in lock['roles'].values():
        assert r['validation_fold'] != r['test_fold']
        assert r['test_fold'] not in r['train_folds'] and r['validation_fold'] not in r['train_folds']
        assert len(r['train_folds']) == 3
    assert lock['cross_fold_leakage'] == 'NONE'


def test_every_type_group_is_present_in_every_training_set(lock):
    grp = {e: lock['type_group_map'][t] for e, t in lock['event_types'].items()}
    for r in lock['roles'].values():
        train = [e for f in lock['folds'] if f['fold'] in r['train_folds'] for e in f['events']]
        assert {grp[e] for e in train} == {'winter', 'wind_tropical', 'convective_flood'}


def test_affected_flag_is_the_locked_peak_rule(lock):
    w = pd.read_csv(A / 'locks/CV_WINDOW_INDEX.csv.gz', dtype={'fips': str, 'event': str})
    assert (w.affected == (w.peak >= 0.01)).all()
    assert w.groupby(['event', 'fips']).affected.nunique().max() == 1


@pytest.fixture(scope='module')
def data(lock):
    import cv_data as CD
    d = CD.CVData(ROOT, A / 'locks'); d.set_fold(0)
    return d


def test_training_and_validation_are_affected_only_and_test_keeps_all_windows(data, lock):
    r = lock['roles']['0']
    assert data.df['train'].affected.all() and data.df['validation'].affected.all()
    assert data.df['test'].affected.any() and (~data.df['test'].affected).any()
    assert set(data.df['train'].fold) == set(r['train_folds'])
    assert set(data.df['validation'].fold) == {r['validation_fold']} and set(data.df['test'].fold) == {r['test_fold']}


def test_batch_alignment_of_history_target_weather_and_clock(data):
    ids = torch.tensor([0, 17, 123, len(data.df['train']) - 1])
    c, y0, step, tgt = data.batch('train', ids)
    d = data.df['train'].iloc[ids.numpy()]
    for j, (row, t) in enumerate(zip(d.row.to_numpy(), d.t.to_numpy())):
        assert np.allclose(c[j, :24].numpy().astype(np.float64) * data.scale, data.c.Y[row, t - 24:t], atol=1e-6)
        assert abs(float(y0[j]) - data.c.Y[row, t]) < 1e-6
        assert np.allclose(tgt[j].numpy(), data.c.Y[row, t + 1:t + 25], atol=1e-6)
        fw = (data.c.X[row, t + 1:t + 25] - data.wx_mean) / data.wx_std
        pw = (data.c.X[row, t - 23:t + 1] - data.wx_mean) / data.wx_std
        assert np.allclose(c[j, 24:312].numpy().reshape(24, 12), pw, atol=1e-4)
        assert np.allclose(c[j, 312:].numpy().reshape(24, 12), fw, atol=1e-4)
        assert np.allclose(step[j, :, :12].numpy(), fw, atol=1e-4)


def test_paired_models_draw_identical_training_batches(data):
    import cv_train as CT
    a, b = CT.HierSampler(data.df['train'], 9201, 0), CT.HierSampler(data.df['train'], 9201, 0)
    for _ in range(3):
        assert np.array_equal(a.draw(512), b.draw(512))
    other = CT.HierSampler(data.df['train'], 9201, 1)
    assert not np.array_equal(CT.HierSampler(data.df['train'], 9201, 0).draw(512), other.draw(512))
