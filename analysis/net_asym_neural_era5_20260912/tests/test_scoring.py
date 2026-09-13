import numpy as np, pandas as pd
from scoring import (cluster_bootstrap, event_county_origin_weights, event_level_mse, exact_signflip_p,
                     seed_mean_and_ensemble_risk)


def test_weights_are_event_equal_then_county_then_origin():
    e = np.array(['a'] * 3 + ['b'] * 5); c = np.array(['x', 'x', 'y', 'p', 'p', 'p', 'q', 'r'])
    w = event_county_origin_weights(e, c)
    assert np.isclose(w[e == 'a'].sum(), .5) and np.isclose(w[e == 'b'].sum(), .5)
    assert np.isclose(w[:2].sum(), w[2]) and np.isclose(w[3:6].sum(), w[6])


def test_event_level_mse_matches_weighted_sum():
    rng = np.random.default_rng(0)
    e = np.repeat(['a', 'b'], [7, 9]); c = np.array(list('xxxyyyz') + list('ppqqqrrrr')); l = rng.random(16)
    w = event_county_origin_weights(e, c)
    assert np.isclose(event_level_mse(l, e, c).mean(), np.sum(w * l))


def test_seed_mean_risk_is_not_ensemble_risk():
    truth = np.zeros((4, 3)); w = np.full(4, .25)
    r = seed_mean_and_ensemble_risk([truth + 1, truth - 1], truth, w)
    assert r['mean_of_seed_risks'] == 1.0 and r['risk_of_seed_ensemble'] == 0.0


def test_exact_signflip_enumeration():
    assert exact_signflip_p(np.ones(5)) == 2 / 32
    assert exact_signflip_p(np.array([1.0, -1.0])) == 1.0


def test_cluster_bootstrap_constant_delta_and_multiplicity():
    d = pd.DataFrame([[.2, .2, .2, .2]] * 3, columns=['e1', 'e2', 'e3', 'e4'])
    r = cluster_bootstrap(d, [['e1', 'e2'], ['e3'], ['e4']], n_boot=200)
    assert np.allclose([r['point'], r['p2.5'], r['p97.5']], .2, rtol=0, atol=1e-15) and r['components_positive'] == 3
    d2 = pd.DataFrame([[1.0, 1.0, 0.0]], columns=['e1', 'e2', 'e3'])
    r2 = cluster_bootstrap(d2, [['e1', 'e2'], ['e3']], n_boot=4000)
    assert 0 < r2['p50'] < 1 and abs(r2['component_delta'][0] - 1.0) < 1e-12
