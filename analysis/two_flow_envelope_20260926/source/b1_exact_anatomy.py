#!/usr/bin/env python3
"""B1: exact anatomy of the two-flow restriction on the neural-bridge design.

For gamma in {0, .04}: non-affine signal S = ||Qm||^2/d, feasibility gap F = dist(Pm, A)^2/d, the identity
min_{g in A} ||m-g||^2 = ||Qm||^2 + dist(Pm, A)^2, the active constraints of the constrained fit, the implied
rates of the per-horizon affine fit, and local variance terms tr(Sigma), tr(P Sigma), tr(P_face Sigma) for every
(rho, n) cell. Deterministic: uses only the exact moments in design.npz.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
W = HERE.parent
sys.path.insert(0, str(HERE))
import dualflow as DF  # noqa: E402

BRIDGE = W / 'inputs/neural_bridge'  # public layout; the as-run script read the same files from a private folder
H, M, L, U, R = 24, 16, 32, .06, .20
TRAIN_K = np.array([0, 2, 4, 8]); TEST_K = np.array([1, 3, 5, 6])
FORCING = np.stack([np.r_[np.ones(8), np.zeros(16)], np.r_[np.full(12, .8), np.zeros(12)]])
RHOS = [0., .25, 1.]; NS = [32, 128, 512]; GAMMAS = [0., .04]


def c_rho(rho):
    return rho + (1 - rho) / L


def jacobian_free(s, w, z, tol=1e-7):
    """Numerical Jacobian of the stacked path (len(z)*H) w.r.t. the parameters that are not at a bound."""
    H_ = len(s); th = np.r_[s, w]; eps = 1e-7
    base = DF.path(s, w, z).ravel()
    cols = []
    for j in range(2 * H_):
        if th[j] < tol or th[j] > 1 - tol:
            continue
        tp = th.copy(); tp[j] += eps
        cols.append((DF.path(tp[:H_], tp[H_:], z).ravel() - base) / eps)
    return np.array(cols).T if cols else np.zeros((len(base), 0))


def projector_onto(J, rtol=1e-8):
    if J.shape[1] == 0:
        return np.zeros((J.shape[0], J.shape[0]))
    Uu, sv, _ = np.linalg.svd(J, full_matrices=False)
    k = int(np.sum(sv > rtol * sv[0]))
    return Uu[:, :k] @ Uu[:, :k].T, k


def main():
    d_np = np.load(BRIDGE / 'design.npz')
    z = TRAIN_K / M; zt = TEST_K / M
    Pz = DF.affine_projector(z)
    out = {'status': 'EXACT_DETERMINISTIC', 'design': {'H': H, 'M': M, 'L': L, 'train_K': TRAIN_K.tolist(), 'test_K': TEST_K.tolist()},
           'normalisation': 'all squared quantities divided by d = 8 contexts x 24 hours = 192; risks in units of 1e-6 where labelled'}
    d = 8 * H
    for gi, g in enumerate(GAMMAS):
        m = d_np[f'mean_{gi}']; cov = d_np[f'cov_{gi}']; mt = d_np[f'mean_test_{gi}']
        rec = {'gamma': g}
        S = F = tot = 0.0; per_forcing = []
        face_rank = 0; P_face_blocks = []
        for f in range(2):
            T = m[4 * f:4 * f + 4]
            PT = Pz @ T; QT = T - PT
            true_s = 1 - U * FORCING[f] - R
            true_w = U * FORCING[f] / (U * FORCING[f] + R)
            start = [np.r_[true_s, true_w]]
            fit_P = DF.fit(PT, z, starts=start, n_random=8, seed=10 + f)
            fit_m = DF.fit(T, z, starts=start + [fit_P['theta']], n_random=8, seed=20 + f)
            Sf = float(np.sum(QT ** 2)); Ff = fit_P['loss']; Tf = fit_m['loss']
            S += Sf; F += Ff; tot += Tf
            aff = DF.rates_from_affine(PT, z)
            act = DF.active_set(fit_m)
            J = jacobian_free(fit_m['s'], fit_m['w'], z)
            Pf, k = projector_onto(J)
            face_rank += k; P_face_blocks.append(Pf)
            # interpolation-origin bias of the fitted rates versus the exact test means
            s_, w_ = fit_m['s'], fit_m['w']
            interp_bias = float(np.sum((DF.path(s_, w_, zt) - mt[4 * f:4 * f + 4]) ** 2))
            per_forcing.append({'forcing': f, 'S_sum': Sf, 'F_sum': Ff, 'direct_fit_sum': Tf, 'identity_gap': Tf - (Sf + Ff),
                                'fit_converged': [fit_P['converged'], fit_m['converged']],
                                'implied_affine_u_min': float(np.nanmin(aff['u'])), 'implied_affine_u_argmin': int(np.nanargmin(aff['u'])),
                                'implied_affine_s_min': float(np.nanmin(aff['s'])), 'implied_affine_s_max': float(np.nanmax(aff['s'])),
                                'active_constraints_fit_m': act, 'face_rank': k, 'interp_bias_sum': interp_bias,
                                'fitted_u': fit_m['u'].round(6).tolist(), 'fitted_r': fit_m['r'].round(6).tolist()})
        rec.update({'S': S / d, 'F': F / d, 'direct_min': tot / d, 'identity_relative_gap': (tot - S - F) / max(tot, 1e-300),
                    'F_over_S': (F / S) if S > 0 else None, 'face_rank_total': face_rank, 'per_forcing': per_forcing})
        # variance terms; contexts are independent, so Sigma is block diagonal over the 8 contexts
        diag = np.stack([np.diag(cov[c]) for c in range(8)])          # (8, H) single-block variances
        trS = float(diag.sum())
        trPS = float(sum(Pz[i, i] * diag[4 * f + i].sum() for f in range(2) for i in range(4)))
        trQS = trS - trPS
        # tr(P_face Sigma) with P_face acting on the stacked (z, h) index of each forcing; Sigma block over contexts
        trFS = 0.0
        for f in range(2):
            Sig_f = np.zeros((4 * H, 4 * H))
            for i in range(4):
                Sig_f[i * H:(i + 1) * H, i * H:(i + 1) * H] = cov[4 * f + i]
            trFS += float(np.trace(P_face_blocks[f] @ Sig_f))
        cells = []
        for rho in RHOS:
            for n in NS:
                c = c_rho(rho)
                v_sat = c * trS / n / d; v_aff = c * trPS / n / d; v_face = c * trFS / n / d
                cells.append({'rho': rho, 'n': n,
                              'risk_saturated': v_sat, 'risk_affine': S / d + v_aff,
                              'risk_dual_local_face': (S + F) / d + v_face,
                              'delta_sat_minus_affine': v_sat - (S / d + v_aff),
                              'delta_sat_minus_dual_local': v_sat - ((S + F) / d + v_face),
                              'predicted_common_variance_saving_face': v_aff - v_face})
        rec.update({'trace_Sigma0': trS, 'trace_P_Sigma0': trPS, 'trace_Q_Sigma0': trQS, 'trace_Pface_Sigma0': trFS, 'cells': cells})
        out[f'gamma_{g}'] = rec
    (W / 'results').mkdir(exist_ok=True)
    (W / 'results/b1_exact_anatomy.json').write_text(json.dumps(out, indent=1) + '\n')
    for g in GAMMAS:
        r = out[f'gamma_{g}']
        print(f"gamma={g}: S={r['S']:.4e}  F={r['F']:.4e}  F/S={r['F_over_S']}  direct={r['direct_min']:.4e}  identity_rel_gap={r['identity_relative_gap']:.2e}  face_rank={r['face_rank_total']}")
        for pf in r['per_forcing']:
            print(f"   forcing {pf['forcing']}: implied u_min={pf['implied_affine_u_min']:+.6f} at h={pf['implied_affine_u_argmin']}  s in [{pf['implied_affine_s_min']:.4f},{pf['implied_affine_s_max']:.4f}]  active={ {k: len(v) for k, v in pf['active_constraints_fit_m'].items()} }  face_rank={pf['face_rank']}")
        print('   rho    n   sat-aff(1e-6)  sat-dual_local(1e-6)  common_var_saving_face(1e-6)')
        for c in r['cells']:
            print(f"   {c['rho']:.2f} {c['n']:4d}   {c['delta_sat_minus_affine']*1e6:+9.4f}      {c['delta_sat_minus_dual_local']*1e6:+9.4f}            {c['predicted_common_variance_saving_face']*1e6:9.4f}")


if __name__ == '__main__':
    main()
