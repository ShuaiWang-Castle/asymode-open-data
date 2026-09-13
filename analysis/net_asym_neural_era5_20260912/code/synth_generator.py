"""Synthetic conditional-response generator and exact finite-state evaluator.

Local process, per service block with M0 response groups, K0 = 0:
    damage[k]  ~ Binomial(M0 - K[k], u * x[k])
    restore[k] ~ Binomial(K[k], r - gamma * K[k] / M0)
    K[k+1] = K[k] + damage[k] - restore[k],   Z = K / M0
Both draws condition on the SAME current K (two disjoint source pools).

Event mixture over L blocks, one B ~ Bernoulli(rho) per event:
    B = 1: Y = Z of a single shared block path
    B = 0: Y = mean over L independent block paths
    E[Y|x] = mu0,  Cov(Y|x) = c * Sigma0,  c = rho + (1 - rho) / L

Target convention: Y[j] = Z_{j+1}, the state after transition j, j = 0..H-1.
The evaluator is used only for calibration and independent scoring; no network
ever reads mu0, Sigma0, u, r, gamma, M0, L, rho, c or B.
"""
from __future__ import annotations
import math
import numpy as np

H = 32; M0 = 16; L_BLOCKS = 32; U = 0.06; R = 0.20; GAMMA = 0.04
SCENE_HOURS = (4, 8, 12); TARGET_SD = (0.011, 0.020, 0.060)
STAGE_CODE = {'DEV': 0, 'FINAL': 1, 'VERIFY': 9}
SPLIT_CODE = {'train': 0, 'validation': 1, 'calibration': 2, 'test': 3}
SPLIT_SIZE = {'train': 512, 'validation': 512, 'calibration': 512, 'test': 2048}


def scene_x(hours: int) -> np.ndarray:
    x = np.zeros(H); x[:hours] = 1.0
    return x


def _binom_pmf(n: int, p: float) -> np.ndarray:
    out = np.zeros(n + 1)
    if p <= 0: out[0] = 1.0; return out
    if p >= 1: out[n] = 1.0; return out
    for k in range(n + 1):
        out[k] = math.exp(math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
                          + k * math.log(p) + (n - k) * math.log1p(-p))
    return out


def transition_matrix(xk: float, gamma: float = GAMMA) -> np.ndarray:
    P = np.zeros((M0 + 1, M0 + 1))
    for K in range(M0 + 1):
        pr = R - gamma * K / M0
        if not (0.0 <= pr <= 1.0):
            raise ValueError('restore probability outside [0,1]')
        joint = np.outer(_binom_pmf(M0 - K, U * xk), _binom_pmf(K, pr))   # [damage, restore]
        for d in range(M0 - K + 1):
            for s in range(K + 1):
                P[K, K + d - s] += joint[d, s]
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-13)
    return P


def exact_law(x: np.ndarray, gamma: float = GAMMA) -> dict:
    z = np.arange(M0 + 1) / M0
    Ps = [transition_matrix(float(x[k]), gamma) for k in range(H)]
    pi = np.zeros((H + 1, M0 + 1)); pi[0, 0] = 1.0
    for k in range(H):
        pi[k + 1] = pi[k] @ Ps[k]
    mu = pi[1:] @ z
    M2 = np.zeros((H, H))
    for t in range(1, H + 1):
        v = z.copy()                                    # E[Z_t | K_t]
        for s in range(t, 0, -1):
            M2[s - 1, t - 1] = M2[t - 1, s - 1] = float(pi[s] @ (z * v))
            if s > 1:
                v = Ps[s - 1] @ v                       # E[Z_t | K_{s-1}]
    return {'mu': mu, 'Sigma': M2 - np.outer(mu, mu), 'pi': pi}


def calibrate(gamma_ref: float = GAMMA, ref_hours: int = 8) -> dict:
    ref = exact_law(scene_x(ref_hours), gamma_ref)
    sd0 = math.sqrt(np.trace(ref['Sigma']) / H)
    laws = []
    for i, tsd in enumerate(TARGET_SD):
        c = (tsd / sd0) ** 2
        rho = (c - 1.0 / L_BLOCKS) / (1.0 - 1.0 / L_BLOCKS)
        laws.append({'law_index': i, 'target_sd_8h_gamma004': tsd, 'c': c, 'rho': rho,
                     'rho_in_unit_interval': bool(0.0 <= rho <= 1.0)})
    return {'sd0_8h_gamma004': sd0, 'c_floor_1_over_L': 1.0 / L_BLOCKS,
            'min_achievable_sd_8h': sd0 * math.sqrt(1.0 / L_BLOCKS), 'laws': laws}


def rng_for(seed: int, stage: str, scene_hours: int, law: int, gamma: float, split: str) -> np.random.Generator:
    key = (STAGE_CODE[stage], SCENE_HOURS.index(scene_hours), law, 0 if gamma > 0 else 1, SPLIT_CODE[split])
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy=seed, spawn_key=key)))


def simulate_blocks(rng: np.random.Generator, x: np.ndarray, n: int, gamma: float = GAMMA) -> np.ndarray:
    K = np.zeros(n, dtype=np.int64); Z = np.empty((n, H))
    for k in range(H):
        dmg = rng.binomial(M0 - K, U * x[k])
        rst = rng.binomial(K, R - gamma * K / M0)       # same current K, drawn before update
        K = K + dmg - rst
        Z[:, k] = K / M0
    return Z


def simulate_events(rng: np.random.Generator, x: np.ndarray, n: int, rho: float,
                    gamma: float = GAMMA) -> tuple[np.ndarray, np.ndarray]:
    B = rng.random(n) < rho
    Y = np.empty((n, H))
    nb = int(B.sum())
    if nb:
        Y[B] = simulate_blocks(rng, x, nb, gamma)
    if n - nb:
        Y[~B] = simulate_blocks(rng, x, (n - nb) * L_BLOCKS, gamma).reshape(n - nb, L_BLOCKS, H).mean(axis=1)
    return Y, B


def corr_summary(S: np.ndarray) -> dict:
    d = np.sqrt(np.clip(np.diag(S), 0, None)); m = d > 1e-12
    C = np.full_like(S, np.nan); C[np.ix_(m, m)] = S[np.ix_(m, m)] / np.outer(d[m], d[m])
    return {f'mean_corr_lag{l}': float(np.nanmean(np.diagonal(C, offset=l))) for l in (1, 2, 4, 8, 16)}


if __name__ == '__main__':
    import argparse, json
    from pathlib import Path
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cal = calibrate()
    report = {'constants': {'H': H, 'M0': M0, 'L': L_BLOCKS, 'u': U, 'r': R, 'gamma_main': GAMMA,
                            'gamma_control': 0.0, 'scene_hours': SCENE_HOURS, 'population': 12288},
              'calibration': cal, 'scenes': {}, 'mc_checks': {}}
    arrays = {}
    for g in (GAMMA, 0.0):
        for hrs in SCENE_HOURS:
            law = exact_law(scene_x(hrs), g); tag = f'h{hrs}_g{g:g}'
            arrays[f'mu_{tag}'] = law['mu']; arrays[f'Sigma_{tag}'] = law['Sigma']
            base = math.sqrt(np.trace(law['Sigma']) / H)
            report['scenes'][tag] = {
                'sd0_single_block': base,
                'actual_path_sd_per_law': [math.sqrt(l['c']) * base for l in cal['laws']],
                'mu_max': float(law['mu'].max()), 'mu_argmax_index': int(law['mu'].argmax()),
                'mu_at_endpoints_1_6_24_32': [float(law['mu'][i]) for i in (0, 5, 23, 31)],
                'correlation': corr_summary(law['Sigma'])}
    # independent Monte Carlo checks on a verification-only stream (never used for training)
    vr = np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy=9901)))
    for g in (GAMMA, 0.0):
        for hrs in SCENE_HOURS:
            tag = f'h{hrs}_g{g:g}'; x = scene_x(hrs)
            Z = simulate_blocks(vr, x, 400_000, g)
            mu, S = arrays[f'mu_{tag}'], arrays[f'Sigma_{tag}']
            report['mc_checks'][f'single_block_{tag}'] = {
                'n_paths': 400_000, 'max_abs_mean_err': float(np.abs(Z.mean(0) - mu).max()),
                'mean_se_max': float((Z.std(0) / math.sqrt(400_000)).max()),
                'cov_rel_frobenius_err': float(np.linalg.norm(np.cov(Z.T) - S) / np.linalg.norm(S)),
                'paths_in_unit_interval': bool((Z >= 0).all() and (Z <= 1).all())}
    x8 = scene_x(8); mu8, S8 = arrays['mu_h8_g0.04'], arrays['Sigma_h8_g0.04']
    for l in cal['laws']:
        if not l['rho_in_unit_interval']:
            report['mc_checks'][f"mixture_law{l['law_index']}"] = 'SKIPPED_RHO_INVALID'; continue
        Y, B = simulate_events(vr, x8, 60_000, l['rho'])
        report['mc_checks'][f"mixture_law{l['law_index']}"] = {
            'n_events': 60_000, 'empirical_B_rate': float(B.mean()), 'rho': l['rho'],
            'max_abs_mean_err_vs_mu0': float(np.abs(Y.mean(0) - mu8).max()),
            'trace_ratio_emp_over_c_trace': float(np.trace(np.cov(Y.T)) / (l['c'] * np.trace(S8))),
            'cov_rel_frobenius_err_vs_cSigma0': float(np.linalg.norm(np.cov(Y.T) - l['c'] * S8) / np.linalg.norm(l['c'] * S8)),
            'paths_in_unit_interval': bool((Y >= 0).all() and (Y <= 1).all())}
    # zero initial state: the state before transition 0 is K0 = 0 for every path
    z0 = simulate_blocks(vr, np.zeros(H), 10_000)
    report['mc_checks']['zero_initial_state_no_forcing_stays_zero'] = bool((z0 == 0).all())
    first = simulate_blocks(vr, x8, 200_000)[:, 0]
    report['mc_checks']['first_step_matches_binomial_from_zero'] = {
        'emp_mean': float(first.mean()), 'exact': float(U), 'abs_err': float(abs(first.mean() - U))}
    np.savez_compressed(out / 'EXACT_LAWS.npz', **arrays)
    (out / 'EXACT_LAW_CALIBRATION.json').write_text(json.dumps(report, indent=1))
    print(json.dumps({'calibration': cal}, indent=1))
    for k, v in report['scenes'].items():
        print(f"  {k:10s} sd0={v['sd0_single_block']:.5f} actual SD per law="
              f"{[round(s, 5) for s in v['actual_path_sd_per_law']]} mu_max={v['mu_max']:.4f}")
    for k, v in report['mc_checks'].items():
        print(f"  MC {k}: {v}")
