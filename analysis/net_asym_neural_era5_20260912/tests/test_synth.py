"""Synthetic generator, evaluator, RNG-stream and computation-equivalence gates."""
import math
import numpy as np, pytest, torch
import synth_train as ST
from models import build_model
from synth_generator import (GAMMA, H, L_BLOCKS, M0, R, SCENE_HOURS, U, _binom_pmf, calibrate,
                             exact_law, rng_for, scene_x, transition_matrix)


def test_transition_rows_are_distributions():
    for g in (GAMMA, 0.0):
        for xk in (0.0, 1.0):
            P = transition_matrix(xk, g)
            assert P.shape == (M0 + 1, M0 + 1) and (P >= 0).all() and np.allclose(P.sum(1), 1, atol=1e-13)


def test_damage_and_restoration_use_the_current_state_disjoint_pools():
    P = transition_matrix(1.0)
    assert np.allclose(P[0], _binom_pmf(M0, U), atol=1e-15)                   # K=0: nothing to restore
    s = _binom_pmf(M0, R - GAMMA)                                              # K=M0: nothing to damage
    assert np.allclose(P[M0][::-1], s, atol=1e-15)


def test_zero_start_and_first_step_are_exact():
    law0 = exact_law(np.zeros(H))
    assert np.all(law0['mu'] == 0) and np.all(law0['Sigma'] == 0)
    law = exact_law(scene_x(8))
    assert abs(law['mu'][0] - U) < 1e-15 and abs(law['Sigma'][0, 0] - U * (1 - U) / M0) < 1e-15


def test_laws_have_valid_rho_and_c_identity():
    for l in calibrate()['laws']:
        assert 0.0 <= l['rho'] <= 1.0
        assert abs(l['rho'] + (1 - l['rho']) / L_BLOCKS - l['c']) < 1e-12


def test_streams_are_reproducible_independent_and_prefix_stable():
    a = rng_for(7201, 'FINAL', 8, 1, GAMMA, 'train').random(8)
    assert np.array_equal(a, rng_for(7201, 'FINAL', 8, 1, GAMMA, 'train').random(8))
    for other in (rng_for(7201, 'FINAL', 8, 1, GAMMA, 'test'), rng_for(7201, 'DEV', 8, 1, GAMMA, 'train'),
                  rng_for(7201, 'FINAL', 8, 1, 0.0, 'train'), rng_for(7201, 'FINAL', 4, 1, GAMMA, 'train'),
                  rng_for(7202, 'FINAL', 8, 1, GAMMA, 'train')):
        assert not np.array_equal(a, other.random(8))
    env = ST.Env('DEV', 6201, 1, GAMMA)
    t32, t128 = env.train_view(32)[0].reshape(3, 32, H), env.train_view(128)[0].reshape(3, 128, H)
    assert torch.equal(t32, t128[:, :32])


def test_generated_paths_bounded_and_float32_lossless():
    env = ST.Env('DEV', 6202, 1, GAMMA)
    for Y in env.Y.values():
        assert Y.min() >= 0 and Y.max() <= 1
        assert np.array_equal(Y.astype(np.float32).astype(np.float64), Y)


def test_networks_receive_only_x_time_and_zero_start():
    ctx, y0, step = ST.scene_inputs()
    X = np.stack([scene_x(h) for h in SCENE_HOURS]).astype(np.float32)
    assert ctx.shape == (3, H) and step.shape == (3, H, 2) and torch.all(y0 == 0)
    assert np.array_equal(ctx.numpy(), X) and np.array_equal(step[..., 0].numpy(), X)
    assert np.allclose(step[:, :, 1].numpy(), np.arange(H) / H)


@pytest.mark.parametrize('kind', ('NET', 'ASYM'))
def test_unique_scene_forward_equals_per_event_forward(kind):
    torch.manual_seed(0)
    m = build_model(kind, 32, 2, H, 8192, .0625, .1).double()
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * .1)
    ctx, y0, step = ST.scene_inputs(torch.float64)
    sc = torch.from_numpy(np.random.default_rng(1).integers(0, 3, 512))
    tgt = torch.rand(512, H, dtype=torch.float64) * .3
    m.zero_grad(); la = (((m(ctx[sc], y0[sc], step[sc]) - tgt) / .0625) ** 2).mean(); la.backward()
    ga = [p.grad.clone() for p in m.parameters()]
    m.zero_grad(); lb = (((m(ctx, y0, step)[sc] - tgt) / .0625) ** 2).mean(); lb.backward()
    assert abs(float(la) - float(lb)) <= 1e-13 * abs(float(la))
    assert all(torch.allclose(x, p.grad, rtol=1e-11, atol=1e-15) for x, p in zip(ga, m.parameters()))


def test_paired_models_see_identical_batch_ids():
    env = ST.Env('DEV', 6203, 1, GAMMA)
    cfg = ST.CANDIDATES[0]
    _, _, rn = ST.train_run(env, 'NET', cfg, 32, 20, keep_best=False)
    _, _, ra = ST.train_run(env, 'ASYM', cfg, 32, 20, keep_best=False)
    assert rn['first20_batch_ids_sha256'] == ra['first20_batch_ids_sha256']
